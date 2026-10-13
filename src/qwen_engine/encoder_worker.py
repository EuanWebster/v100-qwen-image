"""Local-only production CPU encoder child; no ML imports at module import.

Invoke a fresh child with -B and explicit request/response/manifest/models paths
and runtime headroom. Real package/memory gates currently refuse production.
No image-generation/adapter wiring, download, full-pipeline or cache fallback.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import stat
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from qwen_engine._jsonsafe import plain_json, reject_constant
from qwen_engine.encoder import (
    DEFAULT_MAX_RESPONSE_BYTES, RESULT_SCHEMA, EncodedPayload, EncoderError,
    EncoderProtocolError, EncoderRequest, EncoderRequestError, EncoderSetupError,
)
from qwen_engine.encoder_assets import (
    CHECKPOINT_DTYPE, EncoderAssetError, VerifiedEncoderAssets,
    preflight_encoder_tree, verify_encoder_assets,
)
from qwen_engine.encoder_identity import validate_worker_request_identity
from qwen_engine.encoder_production import encode_with_supplied_model, validate_production_request
from qwen_engine.local_bundle import ENCODER_ROLES, validate_local_reference_layout
from qwen_engine.manifest import ManifestError, ManifestFormatError, manifest_from_dict
from qwen_engine.mem_preflight import MemoryPreflightError, require_encoder_memory_fit

MAX_REQUEST_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_INDEX_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 40
MAX_JSON_NODES = 20000
_STAT_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
_OFFLINE_INSTALLED = False


class OfflineViolation(PermissionError):
    """Network operation denied before its OS side effect."""


def _offline_audit(event, args):
    if event.startswith("socket."):
        raise OfflineViolation(f"encoder worker offline policy refuses {event}")


def install_offline_guard() -> None:
    """Irreversible child-only socket guard, installed before lazy ML imports."""
    global _OFFLINE_INSTALLED
    if _OFFLINE_INSTALLED:
        return
    if any(m in sys.modules for m in ("torch", "diffusers", "transformers")):
        raise EncoderSetupError("offline guard must be installed in a fresh child before ML imports")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      NUMEXPR_NUM_THREADS="1")
    sys.dont_write_bytecode = True
    sys.addaudithook(_offline_audit)
    _OFFLINE_INSTALLED = True


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json(raw: bytes, label: str):
    try:
        text = raw.decode("utf-8")
        depth = 0
        quoted = escaped = False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
                if depth > MAX_JSON_DEPTH:
                    raise ValueError("JSON nesting cap exceeded")
            elif char in "]}":
                depth -= 1
        doc = json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=reject_constant)
        stack, nodes = [doc], 0
        while stack:
            node = stack.pop()
            nodes += 1
            if nodes > MAX_JSON_NODES:
                raise ValueError("JSON node cap exceeded")
            if isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        return doc
    except (ValueError, RecursionError) as exc:
        raise EncoderRequestError(f"{label}: invalid bounded UTF-8 JSON: {exc}") from exc


def _sig(info):
    return tuple(getattr(info, name) for name in _STAT_FIELDS)


def _path_parts(path):
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw.strip() or "\0" in raw:
        raise EncoderSetupError("worker paths must be nonblank strings without NUL")
    p = Path(raw)
    if p.name in ("", ".", ".."):
        raise EncoderSetupError("worker file path must name a file")
    return p.parent.resolve(strict=True), p.name


@contextmanager
def _parent(path):
    parent, name = _path_parts(path)
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        yield parent, name, fd
    finally:
        os.close(fd)


def read_bounded_json(path, cap: int, label: str):
    """Read only an opened regular no-follow FD, bounded and identity-checked."""
    if type(cap) is not int or cap < 1:
        raise EncoderRequestError(f"{label}: invalid input byte cap")
    try:
        with _parent(path) as (_, name, directory):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
                    raise EncoderRequestError(f"{label}: not a bounded regular file")
                data = bytearray()
                while True:
                    block = os.read(fd, min(65536, cap + 1 - len(data)))
                    if not block:
                        break
                    data.extend(block)
                    if len(data) > cap:
                        raise EncoderRequestError(f"{label}: byte cap exceeded")
                after = os.fstat(fd)
                published = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if len(data) != before.st_size or _sig(before) != _sig(after) or _sig(before) != _sig(published):
                    raise EncoderRequestError(f"{label}: changed during read")
                return _strict_json(bytes(data), label), bytes(data)
            finally:
                os.close(fd)
    except (OSError, ValueError, TypeError, RuntimeError, EncoderSetupError) as exc:
        raise EncoderRequestError(f"{label}: cannot read input file: {exc}") from exc


def validate_shard_index(assets: VerifiedEncoderAssets) -> None:
    path = assets.component_dirs["text_encoder"] / "model.safetensors.index.json"
    try:
        doc, raw = read_bounded_json(path, MAX_INDEX_BYTES, "encoder shard index")
    except EncoderRequestError as exc:
        raise EncoderAssetError(str(exc)) from exc
    index = next(a for a in assets.artifacts if a.identity == "text_encoder/model.safetensors.index.json")
    if len(raw) != index.bytes or hashlib.sha256(raw).hexdigest() != index.sha256:
        raise EncoderAssetError("encoder shard index changed since verification")
    shards = {Path(a.identity).name for a in assets.artifacts
              if a.role == "text_encoder" and a.identity.endswith(".safetensors")}
    if not isinstance(doc, dict) or set(doc) != {"metadata", "weight_map"}:
        raise EncoderAssetError("encoder shard index requires metadata/weight_map only")
    metadata, weights = doc["metadata"], doc["weight_map"]
    if (not isinstance(metadata, dict) or type(metadata.get("total_size")) is not int
            or not 0 < metadata["total_size"] <= assets.checkpoint_bytes
            or not isinstance(weights, dict) or not weights
            or any(not isinstance(k, str) or not k.strip() or len(k) > 512
                   or not isinstance(v, str) or v not in shards for k, v in weights.items())
            or set(weights.values()) != shards):
        raise EncoderAssetError("encoder shard index has invalid/unsafe shard references or sizes")


def _load_classes():
    try:
        torch = importlib.import_module("torch")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        transformers = importlib.import_module("transformers")
        return torch, transformers.Qwen3VLProcessor, transformers.Qwen3VLForConditionalGeneration
    except Exception as exc:
        raise EncoderSetupError(f"pinned encoder loader classes unavailable: {exc}") from exc


def _load_local_components(assets: VerifiedEncoderAssets, dtype: str):
    preflight_encoder_tree(assets.root)
    torch, processor_class, model_class = _load_classes()
    want = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    common = dict(local_files_only=True, trust_remote_code=False, token=False, force_download=False)
    try:
        with torch.inference_mode():
            processor = processor_class.from_pretrained(str(assets.component_dirs["processor"]), **common)
            model = model_class.from_pretrained(str(assets.component_dirs["text_encoder"]),
                **common, dtype=want, device_map={"": "cpu"}, use_safetensors=True,
                weights_only=True, low_cpu_mem_usage=True,
                attn_implementation="eager", use_kernels=False, allow_all_kernels=False)
            # Qwen has a nested text config; setting only the outer config
            # need not disable the language model's autoregressive KV cache.
            model.config.get_text_config().use_cache = False
            model.eval()
    except Exception as exc:
        raise EncoderSetupError(f"local CPU encoder component loading failed: {exc}") from exc
    return model, processor


def _encode_request(request, manifest, models_dir, headroom_bytes):
    validate_production_request(request)
    validate_local_reference_layout(manifest)
    checkpoint = sum(a.bytes for a in manifest.artifacts
                     if a.role == "text_encoder" and a.identity.endswith(".safetensors"))
    require_encoder_memory_fit(checkpoint, checkpoint_dtype=CHECKPOINT_DTYPE,
                              target_dtype=request.dtype, headroom_bytes=headroom_bytes)
    validate_worker_request_identity(request, manifest)
    # Rechecks actual memory immediately before fresh asset verification.
    assets = verify_encoder_assets(manifest, models_dir, target_dtype=request.dtype, headroom_bytes=headroom_bytes)
    if assets.encoder_artifact_set_sha256 != request.implementation_identity.encoder_artifact_set_sha256:
        raise EncoderRequestError("verified encoder artifacts disagree with request identity")
    validate_shard_index(assets)
    model, processor = _load_local_components(assets, request.dtype)
    return encode_with_supplied_model(model, processor, request)


def result_bytes(payload: EncodedPayload, request_id: str, cap=DEFAULT_MAX_RESPONSE_BYTES) -> bytes:
    """Budget complete JSON/base64 wire size BEFORE base64 expansion."""
    if not isinstance(payload, EncodedPayload) or type(cap) is not int or cap < 1:
        raise EncoderProtocolError("invalid payload/worker response byte cap")
    doc = dict(schema=RESULT_SCHEMA, request_id=request_id, status="ok",
               metadata=plain_json(payload.metadata),
               blobs=[dict(name=b.name, data_b64="") for b in payload.blobs])
    options = dict(sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    overhead = len(json.dumps(doc, **options).encode("utf-8"))
    planned = overhead + sum(4 * ((len(b.data) + 2) // 3) for b in payload.blobs)
    if planned > cap:
        raise EncoderProtocolError("complete worker response exceeds the wire byte cap")
    for spec, blob in zip(doc["blobs"], payload.blobs):
        spec["data_b64"] = base64.b64encode(blob.data).decode("ascii")
    raw = json.dumps(doc, **options).encode("utf-8")
    if len(raw) != planned:
        raise EncoderProtocolError("worker response size disagrees with exact preallocation budget")
    return raw


def _require_absent(name, fd):
    try:
        os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise EncoderSetupError("response destination already exists; refusing to overwrite")


def _publish_response(parent: Path, name: str, directory: int, raw: bytes):
    temporary = f".qwen-encoder-response-{uuid.uuid4().hex}.tmp"
    created = False
    try:
        _require_absent(name, directory)
        # /proc/self/fd keeps the open tied to the chosen directory while also
        # making the destination visible to Python audit write guards.
        target = Path("/proc/self/fd") / str(directory) / temporary
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        created = True
        try:
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError("zero-length response write")
                view = view[count:]
            os.fsync(fd)
            if _sig(os.fstat(fd)) != _sig(os.stat(temporary, dir_fd=directory, follow_symlinks=False)):
                raise EncoderSetupError("temporary worker response changed before publication")
        finally:
            os.close(fd)
        opened, published = os.fstat(directory), parent.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (published.st_dev, published.st_ino):
            raise EncoderSetupError("response parent changed during worker execution")
        os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.fsync(directory)
    except OSError as exc:
        raise EncoderSetupError(f"cannot publish no-clobber worker response: {exc}") from exc
    finally:
        if created:
            os.unlink(temporary, dir_fd=directory)


def execute_worker(request_path, response_path, manifest_path, models_dir, *, headroom_bytes: int):
    """Run only in a fresh child: offline guard lifetime is process-wide."""
    install_offline_guard()
    doc, _ = read_bounded_json(request_path, MAX_REQUEST_BYTES, "encoder request")
    request, rid = EncoderRequest.from_request_dict(doc)
    validate_production_request(request)
    if len(rid) > 128:
        raise EncoderRequestError("request_id exceeds 128 characters")
    try:
        manifest_doc, _ = read_bounded_json(manifest_path, MAX_MANIFEST_BYTES, "model manifest")
    except EncoderRequestError as exc:
        raise ManifestFormatError(str(exc)) from exc
    manifest = manifest_from_dict(manifest_doc)
    try:
        with _parent(response_path) as (parent, name, directory):
            _require_absent(name, directory)
            payload = _encode_request(request, manifest, models_dir, headroom_bytes)
            _publish_response(parent, name, directory, result_bytes(payload, rid))
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise EncoderSetupError(f"invalid worker response destination: {exc}") from exc


def _headroom_arg(value: str) -> int:
    try:
        amount = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("headroom must be a nonnegative byte count") from exc
    if amount < 0:
        raise argparse.ArgumentTypeError("headroom must be a nonnegative byte count")
    return amount


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local-only CPU encoder worker; memory/package gates enforced")
    for flag in ("request", "response", "manifest", "models-dir"):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--headroom-bytes", required=True, type=_headroom_arg)
    args = parser.parse_args(argv)
    try:
        execute_worker(args.request, args.response, args.manifest, args.models_dir,
                       headroom_bytes=args.headroom_bytes)
        return 0
    except MemoryPreflightError as exc:
        return _report_failure(3, exc)
    except ManifestError as exc:
        return _report_failure(4, exc)
    except EncoderRequestError as exc:
        return _report_failure(2, exc)
    except EncoderError as exc:
        return _report_failure(1, exc)
    except Exception as exc:
        print(f"error: unexpected encoder worker failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _report_failure(code, exc):
    print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
