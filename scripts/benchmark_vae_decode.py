#!/usr/bin/env python3
"""Standalone VAE tile-geometry decode benchmark (root runs this on the V100).

Purpose
-------
The production 2048² decode (FP32 tiled VAE, 1024/768) peaks at a measured
15.57 GiB with allocator retry — near OOM. This tool captures the REAL
generation latents once, then replays ONLY the VAE decode at FP32 for each
tile geometry (default 1024/768, 768/576, 512/384) over the identical saved
latent tensor, recording per-geometry memory (fresh allocator peak + NVML
device-wide sampled peak + ACTUAL ``num_alloc_retries`` delta), wall time, the
original PNG, and quantitative RGB/alpha differences against the 1024/768
reference.

Guarantees / boundaries
-----------------------
* No implicit GPU work: without ``--run`` the script only prints the resolved
  plan and exits 0.
* ``--run`` first applies the SAME gate the production loader applies
  (``_require_v100``: pinned CUDA 12.6 build, visible sm_70 V100, installed
  pins via ``_libraries()``) plus manifest/bundle/Q8-sidecar verification and
  the runtime-identity/device-UUID binding — all BEFORE any weight is read.
* The denoiser is fully released before any profile; each profile then gets a
  FRESH starting point (gc + ``empty_cache`` + ``reset_peak_memory_stats``
  inside the probe) so the previous profile's cached allocator blocks cannot
  contaminate the next NVML peak.
* Images come from the EXACT pinned ``VaeImageProcessor.postprocess`` used by
  production (via the loaded pipeline's ``image_processor``); no manual
  channel math, so RGBA/alpha and rounding match production byte-for-byte.
* The latent snapshot binds the SHA-256 of the RAW contiguous CPU float16
  tensor bytes (computed BEFORE the file exists — never a self-referential
  file hash). Replay re-hashes the loaded tensor bytes and refuses any
  mismatch, shape/dtype/density/finiteness disagreement, or oversized file.
* The snapshot is loaded with ``weights_only=True``; an arbitrary external
  pickle is never trusted.
* No output is ever overwritten: declared artifacts are collision-checked up
  front, the snapshot and PNGs are written ``O_EXCL``-exclusive, and the
  report is published atomically (temp + fsync + ``os.replace``).
* SIGINT/SIGTERM set ONE cooperative cancel event that is bound directly into
  the adapter (no forwarding shim) and checked during encode/denoise/profiles;
  owned threads/NVML sessions close even on failure.

Honest limitations: NVML peaks are interval samples (miss sub-interval
spikes); quality conclusions require the saved PNGs — nothing here claims a
smaller geometry preserves quality without them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_engine.config import (  # noqa: E402
    ConfigError, GenerationConfig, OutputFormat, Quant,
)
from qwen_engine.decode_bench import (  # noqa: E402
    DEFAULT_GEOMETRIES, alpha_stats, atomic_write_bytes, build_benchmark_report,
    build_snapshot_metadata, check_output_collisions, compare_rgb_bytes,
    exclusive_write_bytes, raw_tensor_sha256, read_benchmark_report, sha256_bytes,
    sha256_file, utc_now, validate_geometries, validate_snapshot_metadata,
    write_benchmark_report,
)
from qwen_engine.errors import EngineError, ExitCode  # noqa: E402

#: The reviewed baseline geometry; its PNG is the quality reference.
REFERENCE_GEOMETRY = (1024, 768)
#: Declared NVML sampling interval for the device-wide peak (a coverage
#: statement, NOT a safety budget: shorter spikes can be missed).
NVML_INTERVAL_MS = 50.0
#: Hard bound on an accepted snapshot file (2048² fp16 latent ≈ 2 MiB; the
#: bound rejects unbounded/hostile inputs long before torch.load runs).
MAX_SNAPSHOT_BYTES = 16 * 1024**2


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmark_vae_decode",
        description="Replay one saved real latent through FP32 tiled VAE decodes "
                    "at several geometries and report memory/time/image deltas.")
    parser.add_argument("--output-dir", required=True,
                        help="Explicit existing directory for snapshot/PNGs/report.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prompt", default="a test image")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--quant", choices=("fp16", "q8"), default="fp16")
    parser.add_argument("--q8-manifest-path", default=None,
                        help="Required Q8_0 sidecar identity when --quant q8.")
    parser.add_argument("--geometries", default=None,
                        help="Comma separated tile/stride pairs, e.g. 1024/768,768/576.")
    parser.add_argument("--latents", default=None,
                        help="Replay an existing snapshot .pt instead of generating.")
    parser.add_argument("--run", action="store_true",
                        help="Explicit opt-in to touch the GPU. Without it the "
                             "resolved plan is printed and nothing runs.")
    parser.add_argument("--attest-unbounded-host", action="store_true",
                        help="Operator attestation forwarded to the encoder "
                             "preflight (same semantics as the production CLI).")
    parser.add_argument("--report", default="decode-bench.json",
                        help="Report filename inside --output-dir.")
    args = parser.parse_args(argv)
    if args.quant == "q8" and not args.q8_manifest_path:
        parser.error("--quant q8 requires --q8-manifest-path (exact sidecar identity)")
    if args.quant == "fp16" and args.q8_manifest_path:
        parser.error("--q8-manifest-path is only valid with --quant q8")
    return args


def resolve_geometries(raw):
    if raw is None:
        return validate_geometries(list(DEFAULT_GEOMETRIES))
    pairs = []
    for entry in raw.split(","):
        parts = entry.strip().split("/")
        if len(parts) != 2:
            raise ConfigError(f"geometry {entry!r} must look like tile/stride")
        try:
            pairs.append((int(parts[0]), int(parts[1])))
        except ValueError:
            raise ConfigError(f"geometry {entry!r} contains a non-integer") from None
    return validate_geometries(pairs)


def build_config(args, geometries) -> GenerationConfig:
    return GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        quant=Quant(args.quant),
        vae_tiled_decode=True,
        output_format=OutputFormat.PNG,
        output_dir=str(Path(args.output_dir).resolve()),
        cache_dir=str(Path(args.cache_dir).resolve()),
        models_dir=str(Path(args.models_dir).resolve()),
        manifest_path=str(Path(args.manifest_path).resolve()),
        q8_manifest_path=(str(Path(args.q8_manifest_path).resolve())
                          if args.q8_manifest_path else None),
    )


def semantic_config_sha256(config: GenerationConfig) -> str:
    """Digest of the SEMANTIC generation parameters only — no runtime paths.

    Paths (output/cache/models/manifest) legitimately differ between capture
    and replay (a fresh output dir), so binding them would make every replay
    fail; artifact-set and runtime identities are bound as separate digests.
    """
    semantic = {
        "prompt": config.prompt, "negative_prompt": config.negative_prompt,
        "width": config.width, "height": config.height, "steps": config.steps,
        "seed": config.seed, "quant": config.quant.value,
        "vae_tiled_decode": True, "output_format": config.output_format.value,
    }
    payload = json.dumps(semantic, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_artifact_identity(config) -> str:
    """Verified bundle identity; for Q8 the EXACT sidecar hash is folded in."""
    from qwen_engine.local_bundle import resolve_local_bundle
    from qwen_engine.manifest import load_manifest

    manifest = load_manifest(config.manifest_path)
    bundle = resolve_local_bundle(manifest, config.models_dir)
    identity = bundle.fingerprints.bundle
    if config.quant is Quant.Q8:
        from qwen_engine.adapters.q8_loader import artifact_set_identity
        from qwen_engine.q8_artifact import resolve_q8_artifact

        q8 = resolve_q8_artifact(config.q8_manifest_path)
        identity = artifact_set_identity(identity, q8.identity_sha256)
    return identity


class CapturingAdapter:
    """Composition wrapper over the reviewed adapter: snapshot, then delegate.

    ``decode`` first copies the real latent to CPU float16, computes the
    SHA-256 of its RAW contiguous bytes BEFORE any file exists, saves the
    tensor + binding metadata with an exclusive ``O_EXCL`` write, and then
    performs the NORMAL reviewed decode so the run keeps the current-quality
    reference PNG. No private internals are edited; the inner adapter is
    untouched and already holds the SAME cancel event the script bound.
    """

    def __init__(self, inner, snapshot_path, metadata_builder, torch, cancel_event):
        super().__init__()
        self._inner = inner
        self._snapshot_path = Path(snapshot_path)
        self._metadata_builder = metadata_builder
        self._torch = torch
        self._cancel_event = cancel_event
        self.metadata = None
        self.latent_sha256 = None
        self.snapshot_sha256 = None

    def encode(self, config):
        result = self._inner.encode(config)
        if self._cancel_event.is_set():
            raise KeyboardInterrupt
        return result

    def denoise(self, config, conditioning):
        latents = self._inner.denoise(config, conditioning)
        if self._cancel_event.is_set():
            raise KeyboardInterrupt
        return latents

    def decode(self, latents):
        torch = self._torch
        latents_cpu = latents.detach().to("cpu").clone().half().contiguous()
        if not bool(torch.isfinite(latents_cpu).all()):
            raise EngineError("captured latents are not finite")
        # Bind the RAW tensor bytes BEFORE the file exists (a hash of the
        # file that contains its own hash would be self-referential).
        latent_sha, raw = raw_tensor_sha256(latents_cpu)
        metadata = self._metadata_builder(latent_shape=tuple(latents_cpu.shape),
                                          latent_sha256=latent_sha)
        payload = self._torch_serialize({"kind": "qwen.decode-bench.latents",
                                         "metadata": metadata,
                                         "latents": latents_cpu})
        exclusive_write_bytes(self._snapshot_path, payload)
        self.metadata = metadata
        self.latent_sha256 = latent_sha
        self.snapshot_sha256 = sha256_file(str(self._snapshot_path))
        del raw
        return self._inner.decode(latents)

    @staticmethod
    def _torch_serialize(payload: dict) -> bytes:
        import io

        buffer = io.BytesIO()
        import torch
        torch.save(payload, buffer)
        return buffer.getvalue()

    def save(self, image, config):
        return self._inner.save(image, config)

    def close(self):
        self._inner.close()


def capture_latents(args, config, cancel_event, out_dir: Path, artifact_sha: str,
                    torch, pipeline_cls, vae_cls):
    """One normal core-lifecycle generation; returns capture facts.

    The SAME ``cancel_event`` object is bound directly into the adapter —
    there is no second event and no callback forwarding.
    """
    from qwen_engine.adapters.qwen21_diffusers import Qwen21DiffusersAdapter
    from qwen_engine.local_bundle import resolve_local_bundle
    from qwen_engine.manifest import load_manifest
    from qwen_engine.runtime_identity import observe_generation_runtime

    manifest = load_manifest(config.manifest_path)
    bundle = resolve_local_bundle(manifest, config.models_dir)
    runtime = observe_generation_runtime()
    runtime_sha = runtime.sha256()
    inner = Qwen21DiffusersAdapter(trust_unbounded_host=args.attest_unbounded_host)
    inner.set_cancel_event(cancel_event)
    snapshot_path = out_dir / f"latents-seed{config.seed}.pt"
    semantic_sha = semantic_config_sha256(config)

    def build_metadata(*, latent_shape, latent_sha256):
        return build_snapshot_metadata(
            seed=config.seed, width=config.width, height=config.height,
            steps=config.steps, quant=config.quant.value,
            semantic_config_sha256=semantic_sha,
            artifact_set_sha256=artifact_sha,
            runtime_identity_sha256=runtime_sha,
            capture_geometry="tile{}-stride{}".format(*REFERENCE_GEOMETRY),
            latent_shape=latent_shape, latent_dtype="float16",
            latent_sha256=latent_sha256)

    capture = CapturingAdapter(inner, snapshot_path, build_metadata, torch, cancel_event)
    try:
        inner.load(config, bundle.to_assets())
        if cancel_event.is_set():
            raise KeyboardInterrupt
        conditioning = capture.encode(config)
        latents = capture.denoise(config, conditioning)
        if cancel_event.is_set():
            raise KeyboardInterrupt
        image = capture.decode(latents)
        capture.save(image, config)
        if capture.metadata is None or capture.latent_sha256 is None:
            raise EngineError("capture did not produce binding metadata")
        run_facts = {
            "prompt": config.prompt, "seed": config.seed,
            "width": config.width, "height": config.height,
            "steps": config.steps, "quant": config.quant.value,
            "semantic_config_sha256": semantic_sha,
            "artifact_set_sha256": artifact_sha,
            "runtime_identity_sha256": runtime_sha,
            "device_uuid": None, "nvml_error": None,
        }
        return {"snapshot": snapshot_path, "metadata": capture.metadata,
                "snapshot_sha256": capture.snapshot_sha256,
                "latent_sha256": capture.latent_sha256,
                "run_facts": run_facts}
    finally:
        # Denoiser/VAE/pipeline references are dropped even on failure; the
        # adapter's own close already released the transformer and VAE.
        capture.close()


def load_verified_snapshot(path: Path, expected: dict):
    """weights_only load + raw-bytes digest re-validation. Never trust pickles."""
    import torch

    size = path.stat().st_size
    if size <= 0 or size > MAX_SNAPSHOT_BYTES:
        raise EngineError(f"snapshot file size {size} is outside the accepted bound")
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if (type(payload) is not dict or payload.get("kind") != "qwen.decode-bench.latents"
            or set(payload) != {"kind", "metadata", "latents"}):
        raise EngineError("snapshot file is not a benchmark latent capture")
    metadata = validate_snapshot_metadata(payload["metadata"], expected=expected)
    tensor = payload["latents"]
    if not isinstance(tensor, torch.Tensor):
        raise EngineError("snapshot payload does not carry a tensor")
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise EngineError("snapshot tensor must be a dense contiguous CPU tensor")
    if tuple(tensor.shape) != tuple(metadata["latent_shape"]):
        raise EngineError("snapshot tensor shape disagrees with its metadata")
    if str(tensor.dtype).removeprefix('torch.') != metadata["latent_dtype"] or tensor.dtype != torch.float16:
        raise EngineError("snapshot tensor dtype disagrees with metadata")
    if not bool(torch.isfinite(tensor).all()):
        raise EngineError("snapshot tensor contains non-finite values")
    # The metadata digest binds the RAW tensor bytes; re-hash and compare so a
    # corrupted/tampered tensor is refused regardless of what the file says.
    digest, _raw = raw_tensor_sha256(tensor)
    if digest != metadata["latent_sha256"]:
        raise EngineError("snapshot latent bytes do not match the recorded digest")
    return tensor, metadata


def profile_geometry(geometry, *, latents, vae_dir, vae_cls, image_processor,
                     unpack, height, width, vae_scale_factor, torch, device,
                     out_dir: Path, cancel_event):
    """One FP32 tiled decode of the identical latent at one geometry."""
    from qwen_engine.decode_bench import PeakMemoryProbe

    label = geometry.label
    tile, stride = geometry.tile, geometry.stride
    png_path = out_dir / f"decode-{label}.png"
    if png_path.exists():
        raise EngineError(f"refusing to overwrite existing PNG: {png_path.name}")
    properties = torch.cuda.get_device_properties(0)
    probe = PeakMemoryProbe(device_uuid=str(properties.uuid),
                            device_total_bytes=int(properties.total_memory),
                            interval_ms=NVML_INTERVAL_MS, torch_module=torch)
    # probe.start() runs gc + synchronize + empty_cache + reset_peak_memory_stats
    # so this profile's NVML peak cannot inherit the previous profile's cache.
    probe.start()
    started = time.monotonic()
    vae = None
    try:
        vae = vae_cls.from_pretrained(str(vae_dir), torch_dtype=torch.float32,
                                      local_files_only=True)
        vae.to(device=device, dtype=torch.float32).eval()
        vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                          tile_sample_stride_height=stride,
                          tile_sample_stride_width=stride)
        with torch.inference_mode():
            # Exact production normalization (supplied adapter decode):
            unpacked = unpack(latents.to(device), height, width, vae_scale_factor)
            vae_dtype = vae.dtype
            mean = torch.tensor(vae.config.latents_mean, device=device,
                                dtype=vae_dtype).view(1, 64, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, device=device,
                               dtype=vae_dtype).view(1, 64, 1, 1, 1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()
                        and (std > 0).all()):
                raise EngineError("invalid VAE latent normalization")
            normalized = unpacked.to(vae_dtype) * std + mean
            if not bool(torch.isfinite(normalized).all()):
                raise EngineError("normalized latents are not finite")
            decoded = vae.decode(normalized, return_dict=False)[0][:, :, 0]
            if not bool(torch.isfinite(decoded).all()):
                raise EngineError("decoded pixels are not finite")
            if decoded.shape[1] != 4:
                raise EngineError(
                    f"VAE must emit 4 channels (3 RGB + 1 alpha); got {decoded.shape[1]}")
            # EXACT production postprocess: the pinned VaeImageProcessor on the
            # loaded pipeline, not manual channel math. This preserves alpha
            # and the production rounding/truncation byte-for-byte.
            images = image_processor.postprocess(decoded, output_type="pil")
        if (len(images) != 1 or images[0].size != (width, height)
                or images[0].mode not in ("RGB", "RGBA")):
            raise EngineError("VAE postprocessor did not return one correctly sized image")
        image = images[0]
        torch.cuda.synchronize()
        duration = time.monotonic() - started
        if cancel_event.is_set():
            raise KeyboardInterrupt
    finally:
        # Sampler thread + NVML session close even when the decode raised.
        observation = probe.finish(device_uuid=str(properties.uuid))
        if vae is not None:
            del vae
        import gc
        gc.collect()
    exclusive_write_bytes(png_path, _png_bytes(image))
    rgba = image.convert("RGBA")
    profile = {
        "label": label,
        "geometry": {"tile": tile, "stride": stride},
        "vae_dtype": "fp32",
        "observed_at": utc_now(),
        "duration_s": duration,
        "memory": observation,
        "image": {
            "png_path": png_path.name,
            "png_sha256": sha256_file(str(png_path)),
            "size": list(image.size),
            "mode": image.mode,
            "alpha": alpha_stats(rgba.tobytes()),
        },
    }
    return profile, image


def _png_bytes(image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        geometries = resolve_geometries(args.geometries)
        config = build_config(args, geometries)
    except ConfigError as exc:
        print(f"config refused: {exc}", file=sys.stderr)
        return int(ExitCode.CONFIG)
    out_dir = Path(args.output_dir)
    if not out_dir.is_dir():
        print("output dir must exist; refusing to create it", file=sys.stderr)
        return int(ExitCode.CONFIG)

    cancel_event = threading.Event()

    def handle_signal(signum, frame):
        cancel_event.set()
        print(f"signal {signum}: cancelling; owned resources will close",
              file=sys.stderr)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if not args.run:
        plan = {
            "would_run": True,
            "geometries": [geometry.label for geometry in geometries],
            "config": json.loads(json.dumps(config.to_dict(), default=str)),
            "nvml_interval_ms": NVML_INTERVAL_MS,
            "note": "pass --run to touch the GPU; nothing executed",
        }
        print(json.dumps(plan, sort_keys=True))
        return int(ExitCode.OK)

    try:
        # ---- collision check BEFORE any GPU/identity work (no overwrites) ----
        declared = [f"latents-seed{config.seed}.pt", args.report] + [
            f"decode-{geometry.label}.png" for geometry in geometries]
        if args.latents:
            declared.remove(f"latents-seed{config.seed}.pt")
        check_output_collisions(out_dir, declared)

        # ---- exact production gate: pins, V100, bundle, Q8, runtime ----------
        import torch
        from qwen_engine.adapters.qwen21_diffusers import (
            LOGICAL_DEVICE_INDEX, _libraries, _require_v100,
        )
        from qwen_engine.runtime_identity import observe_generation_runtime

        # _libraries() validates the installed pins before importing torch ML.
        torch_mod, pipeline_cls, transformer_cls, vae_cls, scheduler_cls, _ = _libraries()
        properties = _require_v100(torch_mod)  # exit-5 refusal on anything else
        device = torch.device("cuda", LOGICAL_DEVICE_INDEX)
        runtime = observe_generation_runtime()
        if not runtime.matches_cuda_device(uuid=str(properties.uuid),
                                           logical_device_index=LOGICAL_DEVICE_INDEX):
            raise EngineError(
                "generation runtime identity is not bound to the reviewed CUDA device")
        runtime_sha = runtime.sha256()
        artifact_sha = resolve_artifact_identity(config)  # Q8 sidecar folded in
        vae_dir = bundle_vae_dir(config)

        if args.latents:
            snapshot_path = Path(args.latents)
            if not snapshot_path.is_file():
                raise EngineError("--latents snapshot does not exist")
            expected = {
                "seed": config.seed, "width": config.width, "height": config.height,
                "steps": config.steps, "quant": config.quant.value,
                "semantic_config_sha256": semantic_config_sha256(config),
                "artifact_set_sha256": artifact_sha,
                "runtime_identity_sha256": runtime_sha,
            }
            latents, metadata = load_verified_snapshot(snapshot_path, expected)
            capture = {"snapshot": snapshot_path, "metadata": metadata,
                       "snapshot_sha256": sha256_file(str(snapshot_path)),
                       "latent_sha256": metadata["latent_sha256"],
                       "run_facts": {
                           "prompt": config.prompt, "seed": config.seed,
                           "width": config.width, "height": config.height,
                           "steps": config.steps, "quant": config.quant.value,
                           "semantic_config_sha256": expected["semantic_config_sha256"],
                           "artifact_set_sha256": artifact_sha,
                           "runtime_identity_sha256": runtime_sha,
                           "device_uuid": str(properties.uuid), "nvml_error": None}}
        else:
            capture = capture_latents(args, config, cancel_event, out_dir,
                                      artifact_sha, torch_mod, pipeline_cls, vae_cls)
            metadata = capture['metadata']
            latents, metadata = load_verified_snapshot(capture["snapshot"], {
                "seed": metadata["seed"], "width": metadata["width"],
                "height": metadata["height"], "steps": metadata["steps"],
                "quant": metadata["quant"],
                "semantic_config_sha256": metadata["semantic_config_sha256"],
                "artifact_set_sha256": metadata["artifact_set_sha256"],
                "runtime_identity_sha256": metadata["runtime_identity_sha256"],
            })
            capture["run_facts"]["device_uuid"] = str(properties.uuid)

        # The denoiser is gone (adapter closed); only a fresh FP32 VAE per
        # geometry is ever resident. _unpack_latents is the pinned pipeline's
        # static reshape helper; call it without instantiating a pipeline
        # (which would drag the released transformer back in). The image
        # processor is the pinned VaeImageProcessor class from the bundle's
        # processor directory — the exact production postprocess.
        from diffusers.image_processor import VaeImageProcessor

        unpack = getattr(pipeline_cls, "_unpack_latents")
        if hasattr(unpack, "__func__"):
            unpack = unpack.__func__
        image_processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
        vae_scale_factor = getattr(pipeline_cls, "vae_scale_factor", 16)

        profiles, images = [], {}
        for geometry in geometries:
            if cancel_event.is_set():
                raise KeyboardInterrupt
            profile, image = profile_geometry(
                geometry, latents=latents, vae_dir=vae_dir,
                vae_cls=vae_cls, image_processor=image_processor, unpack=unpack,
                height=config.height, width=config.width,
                vae_scale_factor=vae_scale_factor, torch=torch_mod, device=device,
                out_dir=out_dir, cancel_event=cancel_event)
            profiles.append(profile)
            images[geometry.label] = image
            capture["run_facts"]["device_uuid"] = profile["memory"].device_uuid
            capture["run_facts"]["nvml_error"] = None

        reference_label = "tile{}-stride{}".format(*REFERENCE_GEOMETRY)
        comparisons = []
        if reference_label in images:
            reference_mode = images[reference_label].mode
            reference_bytes = images[reference_label].tobytes()
            for label, image in images.items():
                if label == reference_label:
                    continue
                if image.mode != reference_mode:
                    raise EngineError(
                        "postprocess modes differ between geometries; not comparable")
                delta = compare_rgb_bytes(reference_bytes, image.tobytes(),
                                          channels=4 if reference_mode == "RGBA" else 3)
                delta.update({"label": label, "reference": reference_label})
                comparisons.append(delta)
        else:
            comparisons.append({"note": f"reference geometry {reference_label} "
                                        "was not profiled; no image comparison"})
        report = build_benchmark_report(
            run=capture["run_facts"],
            latents={"snapshot": capture["snapshot"].name,
                     "snapshot_sha256": capture["snapshot_sha256"],
                     "latent_sha256": capture["latent_sha256"],
                     "latent_shape": list(latents.shape),
                     "latent_dtype": str(latents.dtype)},
            profiles=profiles,
            comparisons=comparisons,
            limitations=[
                "NVML peaks are interval samples; sub-interval spikes can be missed",
                "allocator_retries is the torch.cuda.memory_stats num_alloc_retries "
                "delta measured across each profile",
                "quality acceptance requires human review of the saved PNGs; "
                "numeric deltas alone do not prove seam-freeness",
            ],
        )
        report_path = out_dir / args.report
        write_benchmark_report(report, str(report_path))
        read_benchmark_report(str(report_path))  # strict round-trip proof
        print(json.dumps({"report": str(report_path),
                          "profiles": [p["label"] for p in profiles]}, sort_keys=True))
        return int(ExitCode.OK)
    except KeyboardInterrupt:
        print("cancelled; partial artifacts kept, no report written", file=sys.stderr)
        return 130
    except ConfigError as exc:
        print(f"config refused: {exc}", file=sys.stderr)
        return int(ExitCode.CONFIG)
    except EngineError as exc:
        print(f"benchmark refused: {exc}", file=sys.stderr)
        return int(ExitCode.UNEXPECTED)


def bundle_vae_dir(config):
    """Re-resolve the verified bundle and return its VAE directory."""
    from qwen_engine.local_bundle import resolve_local_bundle
    from qwen_engine.manifest import load_manifest

    bundle = resolve_local_bundle(load_manifest(config.manifest_path), config.models_dir)
    return bundle.component_dirs["vae"]


if __name__ == "__main__":
    sys.exit(main())
