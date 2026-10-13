"""Guarded CPU worker tests: tiny fixtures/doubles, never real checkpoints."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_engine import encoder_worker as ew, encoder_assets as ea
from qwen_engine.encoder import (
    Blob, EncodedPayload, EncoderWorker, EncoderWorkerError, EncoderRequestError,
    EncoderSetupError, EncoderProtocolError,
)
from qwen_engine.local_bundle import LOCAL_REFERENCE_LAYOUT, reference_bundle_fingerprints
from qwen_engine.manifest import ModelArtifact, ModelManifest, ManifestError
from qwen_engine.mem_preflight import MemorySnapshot, MemoryPreflightError, require_encoder_memory_fit
from test_encoder_production import make_request, FakeTorch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(ew, "install_offline_guard", lambda: None)
    root = tmp_path / "models"
    root.mkdir()
    rows = []
    shards = LOCAL_REFERENCE_LAYOUT["text_encoder"][2:6]
    index = dict(metadata={"total_size": 4}, weight_map={f"w{i}": Path(name).name for i, name in enumerate(shards)})
    for role, names in LOCAL_REFERENCE_LAYOUT.items():
        for name in names:
            raw = json.dumps(index).encode() if name.endswith("model.safetensors.index.json") else name.encode()
            rows.append(ModelArtifact(role, "Qwen/Qwen-Image-2.1", "d26bb61231c349cf6b7896fa83353113880e1ba3",
                                      name, hashlib.sha256(raw).hexdigest(), len(raw)))
            if role in ea.ENCODER_ROLES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
    manifest = ModelManifest(1, tuple(rows))
    req = make_request()
    req = replace(req, implementation_identity=replace(req.implementation_identity,
                  encoder_artifact_set_sha256=reference_bundle_fingerprints(manifest).encoder))
    request, response, manifest_path = (tmp_path / name for name in ("request.json", "response.json", "manifest.json"))
    request.write_text(json.dumps(req.to_request_dict("fixed-id")))
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    def memory(size, **kwargs):
        return require_encoder_memory_fit(size, **kwargs,
            snapshot=MemorySnapshot(host_available_bytes=1024 ** 3, cgroup_state="unlimited"))
    monkeypatch.setattr(ew, "require_encoder_memory_fit", memory)
    monkeypatch.setattr(ea, "require_encoder_memory_fit", memory)
    return SimpleNamespace(root=root, manifest=manifest, req=req, request=request,
                           response=response, manifest_path=manifest_path)


def assets(inputs):
    return ea.verify_encoder_assets(inputs.manifest, inputs.root, target_dtype="fp16", headroom_bytes=0)


def execute(inputs):
    ew.execute_worker(inputs.request, inputs.response, inputs.manifest_path, inputs.root, headroom_bytes=0)


@pytest.mark.parametrize("raw", [b"\xff", b"{", b'{}{}', b'{"x":1,"x":2}',
                                 b'{"x":NaN}', b'{"x":Infinity}', b'[' * 41 + b'0' + b']' * 41])
def test_invalid_utf8_json_and_depth(tmp_path, raw):
    path = tmp_path / "input"
    path.write_bytes(raw)
    with pytest.raises(EncoderRequestError):
        ew.read_bounded_json(path, 4096, "request")


def test_reader_exact_cap_nodes_and_escaped_braces(tmp_path, monkeypatch):
    path = tmp_path / "input"
    raw = json.dumps({"quoted": '[{\\\"}]', "arr": [1, 2]}).encode()
    path.write_bytes(raw)
    assert ew.read_bounded_json(path, len(raw), "request")[1] == raw
    with pytest.raises(EncoderRequestError, match="bounded"):
        ew.read_bounded_json(path, len(raw) - 1, "request")
    monkeypatch.setattr(ew, "MAX_JSON_NODES", 2)
    with pytest.raises(EncoderRequestError, match="node cap"):
        ew.read_bounded_json(path, len(raw), "request")


@pytest.mark.parametrize("kind", ["fifo", "symlink", "directory", "missing"])
def test_reader_nonregular_paths_do_not_block_or_follow(tmp_path, kind):
    path = tmp_path / "input"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        real = tmp_path / "real"
        real.write_bytes(b"{}")
        path.symlink_to(real)
    elif kind == "directory":
        path.mkdir()
    with pytest.raises(EncoderRequestError):
        ew.read_bounded_json(path, 1024, "request")


@pytest.mark.parametrize("change", ["grow", "edit", "replace"])
def test_read_mutation_refused(tmp_path, monkeypatch, change):
    path = tmp_path / "input"
    path.write_bytes(b'{"x":1}')
    original = os.read
    changed = False
    def read(fd, size):
        nonlocal changed
        raw = original(fd, size)
        if raw and not changed:
            changed = True
            if change == "replace":
                other = tmp_path / "other"
                other.write_bytes(b'{"x":1}')
                other.replace(path)
            else:
                path.write_bytes(b'{"x":2}' + (b' ' if change == "grow" else b''))
        return raw
    monkeypatch.setattr(ew.os, "read", read)
    with pytest.raises(EncoderRequestError, match="changed"):
        ew.read_bounded_json(path, 1024, "request")


def test_valid_index(inputs):
    ew.validate_shard_index(assets(inputs))


@pytest.mark.parametrize("bad", [True, -1, 0, "4", None])
def test_index_invalid_total_size_refused(inputs, bad):
    path = inputs.root / "text_encoder/model.safetensors.index.json"
    doc = json.loads(path.read_bytes())
    doc["metadata"]["total_size"] = bad
    raw = json.dumps(doc).encode()
    path.write_bytes(raw)
    inputs.manifest = replace(inputs.manifest, artifacts=tuple(
        replace(a, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        if a.identity.endswith("model.safetensors.index.json") else a for a in inputs.manifest.artifacts))
    with pytest.raises(ea.EncoderAssetError):
        ew.validate_shard_index(assets(inputs))


@pytest.mark.parametrize("bad", ["../outside.safetensors", "/outside.safetensors", "https://x/y",
                                 "model.bin", "sub/model.safetensors", "other.safetensors", True])
def test_unsafe_index_paths_refused_with_matching_manifest_hash(inputs, bad):
    path = inputs.root / "text_encoder/model.safetensors.index.json"
    doc = json.loads(path.read_bytes())
    doc["weight_map"]["w0"] = bad
    raw = json.dumps(doc).encode()
    path.write_bytes(raw)
    inputs.manifest = replace(inputs.manifest, artifacts=tuple(
        replace(a, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        if a.identity.endswith("model.safetensors.index.json") else a for a in inputs.manifest.artifacts))
    with pytest.raises(ea.EncoderAssetError, match="unsafe"):
        ew.validate_shard_index(assets(inputs))


def test_index_changed_after_verification_refused(inputs):
    verified = assets(inputs)
    path = inputs.root / "text_encoder/model.safetensors.index.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ea.EncoderAssetError, match="changed"):
        ew.validate_shard_index(verified)


def test_local_loader_uses_fixed_paths_cpu_and_no_remote_code(inputs, monkeypatch):
    calls = []
    torch = FakeTorch()
    torch.float16, torch.bfloat16, torch.float32 = "F16", "BF16", "F32"
    text_config = SimpleNamespace(use_cache=True)
    model = SimpleNamespace(eval=lambda: calls.append("eval"),
                            config=SimpleNamespace(use_cache=True, get_text_config=lambda: text_config))
    def processor_load(path, **kwargs):
        calls.append(("processor", path, kwargs))
        return "processor"
    def model_load(path, **kwargs):
        calls.append(("model", path, kwargs))
        return model
    monkeypatch.setattr(ew, "_load_classes", lambda: (torch,
        SimpleNamespace(from_pretrained=processor_load), SimpleNamespace(from_pretrained=model_load)))
    assert ew._load_local_components(assets(inputs), "fp16") == (model, "processor")
    assert calls[0][1] == str(inputs.root / "processor")
    assert calls[1][1] == str(inputs.root / "text_encoder")
    for call in calls[:2]:
        assert call[2]["local_files_only"] and not call[2]["trust_remote_code"]
        assert call[2]["token"] is False and call[2]["force_download"] is False
    opts = calls[1][2]
    assert opts["device_map"] == {"": "cpu"} and opts["dtype"] == "F16"
    assert opts["use_safetensors"] and opts["weights_only"] and opts["low_cpu_mem_usage"]
    assert "use_cache" not in opts and opts["attn_implementation"] == "eager"
    assert not opts["use_kernels"] and not opts["allow_all_kernels"]
    assert calls[-1] == "eval" and torch.im_enter == torch.im_exit == 1
    assert text_config.use_cache is False


def test_worker_orders_memory_identity_assets_index_loader_encoder(inputs, monkeypatch):
    verified = assets(inputs)
    events = []
    monkeypatch.setattr(ew, "require_encoder_memory_fit", lambda *a, **k: events.append("memory"))
    monkeypatch.setattr(ew, "validate_worker_request_identity", lambda *a: events.append("identity"))
    monkeypatch.setattr(ew, "verify_encoder_assets", lambda *a, **k: events.append("assets") or verified)
    monkeypatch.setattr(ew, "validate_shard_index", lambda *a: events.append("index"))
    monkeypatch.setattr(ew, "_load_local_components", lambda *a: events.append("loader") or ("model", "proc"))
    payload = EncodedPayload(metadata={"synthetic": True}, blobs=(Blob("tensors", b"tiny"),))
    monkeypatch.setattr(ew, "encode_with_supplied_model", lambda *a: events.append("encoder") or payload)
    execute(inputs)
    assert events == ["memory", "identity", "assets", "index", "loader", "encoder"]
    result = json.loads(inputs.response.read_bytes())
    assert result["schema"] == "qwen_engine.encoder_result.v1" and result["request_id"] == "fixed-id"
    assert result["status"] == "ok" and base64.b64decode(result["blobs"][0]["data_b64"]) == b"tiny"
    assert not list(inputs.response.parent.glob(".qwen-encoder-response-*.tmp"))


@pytest.mark.parametrize("stage,error", [("require_encoder_memory_fit", MemoryPreflightError("memory")),
    ("validate_worker_request_identity", EncoderSetupError("identity")),
    ("verify_encoder_assets", ea.EncoderAssetError("assets")),
    ("validate_shard_index", ea.EncoderAssetError("index")),
    ("_load_local_components", EncoderSetupError("loader"))])
def test_refusals_never_encode_or_publish(inputs, monkeypatch, stage, error):
    monkeypatch.setattr(ew, "validate_worker_request_identity", lambda *a: None)
    def refused(*a, **k):
        raise error
    monkeypatch.setattr(ew, stage, refused)
    monkeypatch.setattr(ew, "encode_with_supplied_model", lambda *a: pytest.fail("encoded after refusal"))
    with pytest.raises(type(error)):
        execute(inputs)
    assert not inputs.response.exists()


def test_wire_cap_checked_before_base64_allocation(monkeypatch):
    payload = EncodedPayload(metadata={"x": "nonascii-\u00e9"}, blobs=(Blob("tensors", b"x" * 100),))
    raw = ew.result_bytes(payload, "id")
    assert ew.result_bytes(payload, "id", cap=len(raw)) == raw
    monkeypatch.setattr(ew.base64, "b64encode", lambda *a: pytest.fail("base64 allocation"))
    with pytest.raises(EncoderProtocolError, match="wire byte cap"):
        ew.result_bytes(payload, "id", cap=len(raw) - 1)


@pytest.mark.parametrize("kind", ["file", "symlink", "directory", "fifo"])
def test_existing_response_never_overwritten_or_loaded(inputs, monkeypatch, kind):
    if kind == "file":
        inputs.response.write_bytes(b"preserve")
    elif kind == "symlink":
        inputs.response.symlink_to(inputs.request)
    elif kind == "directory":
        inputs.response.mkdir()
    else:
        os.mkfifo(inputs.response)
    monkeypatch.setattr(ew, "_encode_request", lambda *a: pytest.fail("loaded for existing output"))
    with pytest.raises(EncoderSetupError, match="already exists"):
        execute(inputs)
    if kind == "file":
        assert inputs.response.read_bytes() == b"preserve"


def test_destination_race_is_no_clobber_and_temp_is_cleaned(tmp_path, monkeypatch):
    target = tmp_path / "response"
    original = os.link
    def raced(src, dst, *args, **kwargs):
        target.write_bytes(b"someone else")
        return original(src, dst, *args, **kwargs)
    monkeypatch.setattr(ew.os, "link", raced)
    with ew._parent(target) as (parent, name, fd):
        with pytest.raises(EncoderSetupError, match="no-clobber"):
            ew._publish_response(parent, name, fd, b"result")
    assert target.read_bytes() == b"someone else"
    assert not list(tmp_path.glob(".qwen-encoder-response-*.tmp"))


@pytest.mark.parametrize("failure", [OSError("disk full"), KeyboardInterrupt()])
def test_write_failure_closes_fd_and_removes_only_own_temp(tmp_path, monkeypatch, failure):
    target = tmp_path / "response"
    def failed(*args):
        raise failure
    monkeypatch.setattr(ew.os, "write", failed)
    before = len(list(Path("/proc/self/fd").iterdir()))
    with ew._parent(target) as (parent, name, fd):
        with pytest.raises(EncoderSetupError if isinstance(failure, OSError) else KeyboardInterrupt):
            ew._publish_response(parent, name, fd, b"result")
    assert len(list(Path("/proc/self/fd").iterdir())) == before
    assert not target.exists() and not list(tmp_path.glob(".qwen-encoder-response-*.tmp"))


def test_response_partial_writes_are_completed(tmp_path, monkeypatch):
    target = tmp_path / "response"
    original = os.write
    monkeypatch.setattr(ew.os, "write", lambda fd, data: original(fd, data[:3]))
    with ew._parent(target) as (parent, name, fd):
        ew._publish_response(parent, name, fd, b"full response despite short writes")
    assert target.read_bytes() == b"full response despite short writes"


def test_response_parent_swap_refused_without_redirected_write(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    target = parent / "response"
    moved = tmp_path / "moved"
    original = os.fsync
    changed = False
    def swapped(fd):
        nonlocal changed
        original(fd)
        if not changed:
            changed = True
            parent.rename(moved)
            parent.mkdir()
    monkeypatch.setattr(ew.os, "fsync", swapped)
    with ew._parent(target) as (chosen, name, fd):
        with pytest.raises(EncoderSetupError, match="parent changed"):
            ew._publish_response(chosen, name, fd, b"result")
    assert not list(parent.iterdir()) and not list(moved.iterdir())


def test_manifest_bad_json_has_asset_error_channel(inputs):
    inputs.manifest_path.write_bytes(b"{")
    with pytest.raises(ManifestError):
        execute(inputs)


def test_malformed_verified_index_has_asset_error_channel(inputs):
    verified = assets(inputs)
    (inputs.root / "text_encoder/model.safetensors.index.json").write_bytes(b"{")
    with pytest.raises(ea.EncoderAssetError):
        ew.validate_shard_index(verified)


def test_request_id_cap_precedes_loading(inputs, monkeypatch):
    inputs.request.write_text(json.dumps(inputs.req.to_request_dict("x" * 129)))
    monkeypatch.setattr(ew, "_encode_request", lambda *a: pytest.fail("loaded invalid request"))
    with pytest.raises(EncoderRequestError, match="request_id"):
        execute(inputs)


@pytest.mark.parametrize("value", ["-1", "not-a-number"])
def test_invalid_cli_headroom_precedes_guard(monkeypatch, value):
    monkeypatch.setattr(ew, "install_offline_guard", lambda: pytest.fail("guard installed"))
    with pytest.raises(SystemExit) as caught:
        ew.main(["--request", "r", "--response", "s", "--manifest", "m", "--models-dir", "d",
                 "--headroom-bytes", value])
    assert caught.value.code == 2


@pytest.mark.parametrize("error,code", [(MemoryPreflightError("memory"), 3), (ManifestError("asset"), 4),
    (EncoderRequestError("request"), 2), (EncoderSetupError("setup"), 1), (RuntimeError("unexpected"), 1)])
def test_main_typed_exit_diagnostics(monkeypatch, capsys, error, code):
    def failed(*a, **k):
        raise error
    monkeypatch.setattr(ew, "execute_worker", failed)
    assert ew.main(["--request", "r", "--response", "s", "--manifest", "m", "--models-dir", "d",
                    "--headroom-bytes", "0"]) == code
    assert "error:" in capsys.readouterr().err


def test_actual_worker_memory_refusal_through_generic_parent(tmp_path):
    worker = EncoderWorker(scratch_dir=tmp_path, timeout_s=20, python=sys.executable, command=[
        "{python}", "-B", "-m", "qwen_engine.encoder_worker", "--request", "{request}",
        "--response", "{response}", "--manifest", str(ROOT / "models/manifest.json"),
        "--models-dir", str(tmp_path / "never-accessed"), "--headroom-bytes", "0"])
    with pytest.raises(EncoderWorkerError) as caught:
        worker.run(make_request())
    assert caught.value.exit_code == 3
    assert "MemoryPreflightError" in caught.value.stderr
    assert not list(tmp_path.iterdir())  # parent reaps child and cleans scratch


def test_offline_guard_child_without_outer_guard(tmp_path):
    # Remove the test guard only in this child to prove the product guard itself.
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), QWEN_CPU_TEST_GUARD="0")
    code = """
import os, socket
from qwen_engine.encoder_worker import install_offline_guard, OfflineViolation
install_offline_guard()
assert os.environ['HF_HUB_OFFLINE'] == '1'
for attempt in (lambda: socket.socket(), lambda: socket.getaddrinfo('example.com', 443)):
    try: attempt()
    except OfflineViolation: pass
    else: raise AssertionError('network guard did not deny operation')
print('PRODUCT GUARD PASS')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and "PRODUCT GUARD PASS" in result.stdout, result.stderr
