"""ML-free production parent/child/cache boundary with guarded tiny fixtures."""
from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from qwen_engine import encoder_client as ec, encoder_identity as ei, encoder_production as ep
from qwen_engine.config import GenerationConfig
from qwen_engine.encoder import (
    Blob, EncodedPayload, EmbeddingCache, EncoderProtocolError, EncoderRequestError,
    EncoderSetupError, EncoderWorkerError, cache_key,
)
from qwen_engine.manifest import load_manifest
from test_encoder import make_identity
from test_encoder_production import triple, st_bytes

ROOT = Path(__file__).resolve().parents[1]


def payload_for(request):
    tensors = {}
    for side in ("positive", "negative") if request.kwargs["cfg_active"] else ("positive",):
        embeds, mask, images = triple(code=ep.DTYPE_TO_ST_CODE[request.dtype])
        tensors[side + ".embeds"], tensors[side + ".attention_mask"] = embeds, mask
        tensors[side + ".image_pad_mask"] = images
    descriptors = [{"name": n, "dtype": t.code, "shape": list(t.shape), "bytes": len(t.raw)}
                   for n, t in sorted(tensors.items())]
    canonical = lambda doc: json.dumps(doc, sort_keys=True, separators=(",", ":"))
    metadata = {"schema": ep.TENSOR_PAYLOAD_SCHEMA, "codec": ep.SAFETENSORS_CODEC,
        "cache_key": cache_key(request), "dtype": request.dtype, "cfg_active": request.kwargs["cfg_active"],
        "request": request.key_document(), "tensors": descriptors}
    st_metadata = {"codec": ep.SAFETENSORS_CODEC, "cache_key": cache_key(request), "dtype": request.dtype,
        "cfg_active": "true" if request.kwargs["cfg_active"] else "false",
        "request_identity": canonical(request.key_document()), "tensor_descriptors": canonical(descriptors)}
    return EncodedPayload(metadata=metadata, blobs=(Blob("tensors", st_bytes(tensors, st_metadata)),))


@pytest.fixture
def client(tmp_path):
    return ec.ProductionEncoder(manifest_path=ROOT / "models/manifest.json",
        models_dir=tmp_path / "never-accessed-models", scratch_dir=tmp_path,
        timeout_s=20, headroom_bytes=0, cache=EmbeddingCache(tmp_path / "cache"))


def stub_children(client, monkeypatch):
    events = []
    identity = make_identity(worker_implementation_revision="synthetic-selected-child-runtime",
        encoder_artifact_set_sha256=ei.reference_bundle_fingerprints(load_manifest(client.manifest_path)).encoder)
    def project(seed, *, request_id, cancel_event):
        events.append("projection")
        request = replace(seed, implementation_identity=identity)
        return EncodedPayload(metadata={"schema": ec.PROJECTION_SCHEMA,
            "request": request.to_request_dict(request_id)})
    def load(request, *, request_id, cancel_event):
        events.append("load")
        return payload_for(request)
    monkeypatch.setattr(client._projector, "run", project)
    monkeypatch.setattr(client._loader, "run", load)
    # A parent-runtime observation would fail this whole test.
    monkeypatch.setattr(ei, "observe_installed_packages", lambda: pytest.fail("parent package inspection"))
    return events, project


@pytest.mark.parametrize("dtype", ep.CANONICAL_DTYPES)
@pytest.mark.parametrize("active", [False, True])
def test_fresh_projection_always_precedes_miss_and_hit_without_ml(client, monkeypatch, dtype, active):
    events, _ = stub_children(client, monkeypatch)
    before = {n for n in sys.modules if n.split(".")[0] in ei.ML_MODULE_NAMES}
    config = GenerationConfig(negative_prompt=" bad ", true_cfg_scale=2 if active else 1)
    first = client.encode(config, encoder_dtype=dtype)
    second = client.encode(config, encoder_dtype=dtype)
    assert events == ["projection", "load", "projection"]
    assert first == second and first.request.implementation_identity is not None
    assert first.request.negative_prompt == " bad "
    assert first.request.kwargs["cfg_active"] is active
    assert before == {n for n in sys.modules if n.split(".")[0] in ei.ML_MODULE_NAMES}
    assert not client.models_dir.exists()


def test_child_runtime_drift_changes_key_and_never_serves_old_cache(client, monkeypatch):
    events, project = stub_children(client, monkeypatch)
    first = client.encode(GenerationConfig(), encoder_dtype="fp16")
    def changed(seed, **kwargs):
        result = project(seed, **kwargs)
        request, rid = ec.EncoderRequest.from_request_dict(ec.plain_json(result.metadata["request"]))
        request = replace(request, implementation_identity=replace(request.implementation_identity,
                          worker_implementation_revision="different-child-runtime"))
        return EncodedPayload(metadata={"schema": ec.PROJECTION_SCHEMA, "request": request.to_request_dict(rid)})
    monkeypatch.setattr(client._projector, "run", changed)
    second = client.encode(GenerationConfig(), encoder_dtype="fp16")
    assert cache_key(first.request) != cache_key(second.request)
    assert events == ["projection", "load", "projection", "load"]
    assert len(client.cache.entries()) == 2


@pytest.mark.parametrize("error", [EncoderSetupError("child packages refused"), KeyboardInterrupt()])
def test_projection_failure_never_locks_reads_writes_or_loads_cache(client, monkeypatch, error):
    def refused(*a, **k):
        raise error
    monkeypatch.setattr(client._projector, "run", refused)
    for name in ("lock_for", "get", "put"):
        monkeypatch.setattr(client.cache, name, lambda *a: pytest.fail("cache after projection failure"))
    monkeypatch.setattr(client._loader, "run", lambda *a, **k: pytest.fail("load after projection failure"))
    with pytest.raises(type(error)) as caught:
        client.encode(GenerationConfig(), encoder_dtype="fp16")
    assert caught.value is error


@pytest.mark.parametrize("change", ["schema", "blobs", "missing", "request-id", "prompt", "no-identity", "asset"])
def test_bad_projection_is_refused_before_cache(client, monkeypatch, change):
    _, project = stub_children(client, monkeypatch)
    def bad(seed, **kwargs):
        good = project(seed, **kwargs)
        md = ec.plain_json(good.metadata)
        if change == "schema":
            md["schema"] = "wrong"
        elif change == "missing":
            del md["request"]
        elif change == "request-id":
            md["request"]["request_id"] = "wrong"
        elif change == "prompt":
            md["request"]["prompt"] = "wrong"
        elif change == "no-identity":
            md["request"]["implementation_identity"] = None
        elif change == "asset":
            md["request"]["implementation_identity"]["encoder_artifact_set_sha256"] = "f" * 64
        return EncodedPayload(metadata=md, blobs=(Blob("not-metadata-only", b"x"),) if change == "blobs" else ())
    monkeypatch.setattr(client._projector, "run", bad)
    monkeypatch.setattr(client.cache, "lock_for", lambda *a: pytest.fail("cache for bad projection"))
    with pytest.raises(EncoderProtocolError):
        client.encode(GenerationConfig(), encoder_dtype="fp16")


def test_invalid_loaded_payload_is_never_cached(client, monkeypatch):
    stub_children(client, monkeypatch)
    monkeypatch.setattr(client._loader, "run", lambda *a, **k: EncodedPayload(blobs=(Blob("bad", b"x"),)))
    monkeypatch.setattr(client.cache, "put", lambda *a: pytest.fail("cached invalid structure"))
    with pytest.raises(EncoderProtocolError):
        client.encode(GenerationConfig(), encoder_dtype="fp16")


def test_invalid_cached_structure_is_refused_without_loading_or_overwrite(client, monkeypatch):
    events, _ = stub_children(client, monkeypatch)
    first = client.encode(GenerationConfig(), encoder_dtype="fp16")
    path = client.cache.put(cache_key(first.request), EncodedPayload(blobs=(Blob("bad", b"x"),)))
    raw = path.read_bytes()
    with pytest.raises(EncoderProtocolError):
        client.encode(GenerationConfig(), encoder_dtype="fp16")
    assert events == ["projection", "load", "projection"]
    assert path.read_bytes() == raw


def test_uncached_client_still_projects_every_call(client, monkeypatch):
    client.cache = None
    events, _ = stub_children(client, monkeypatch)
    client.encode(GenerationConfig(), encoder_dtype="fp16")
    client.encode(GenerationConfig(), encoder_dtype="fp16")
    assert events == ["projection", "load", "projection", "load"]


def test_generation_only_changes_share_projected_cache(client, monkeypatch):
    events, _ = stub_children(client, monkeypatch)
    first = client.encode(GenerationConfig(), encoder_dtype="fp16")
    second = client.encode(GenerationConfig(width=2048, steps=4, seed=999, quant="q8"), encoder_dtype="fp16")
    assert first == second and events == ["projection", "load", "projection"]


@pytest.mark.parametrize("field,value", [("headroom_bytes", True), ("headroom_bytes", -1),
    ("scratch_dir", None), ("models_dir", ""), ("manifest_path", "\0"),
    ("cache", {}), ("python", ""), ("python", 1)])
def test_invalid_client_setup_is_typed_before_child(tmp_path, field, value):
    args = dict(manifest_path=ROOT / "models/manifest.json", models_dir=tmp_path / "models",
                scratch_dir=tmp_path, headroom_bytes=0, timeout_s=20)
    args[field] = value
    with pytest.raises(EncoderSetupError):
        ec.ProductionEncoder(**args)


def test_client_paths_and_interpreter_cwd_stable_without_model_access(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = ec.ProductionEncoder(manifest_path="manifest.json", models_dir="models", scratch_dir=tmp_path,
                                 headroom_bytes=0, timeout_s=20, python="selected/python")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert client.models_dir == tmp_path / "models" and client.manifest_path == tmp_path / "manifest.json"
    assert client._projector.python == str(tmp_path / "selected/python") == client._loader.python
    assert "--project-identity" in client._projector.command and "--models-dir" not in client._projector.command
    assert "--headroom-bytes" in client._loader.command and not client.models_dir.exists()


def test_invalid_or_oversized_input_refuses_before_projection(client, monkeypatch):
    monkeypatch.setattr(client._projector, "run", lambda *a, **k: pytest.fail("projected bad input"))
    for config, dtype in (({}, "fp16"), (GenerationConfig(), "FP16"),
                          (GenerationConfig(prompt="x" * ec.MAX_REQUEST_BYTES), "fp16")):
        with pytest.raises(EncoderRequestError):
            client.encode(config, encoder_dtype=dtype)
    with pytest.raises(EncoderRequestError, match="cancel_event"):
        client.encode(GenerationConfig(), encoder_dtype="fp16", cancel_event=True)


def test_completed_identity_cannot_exceed_worker_request_cap(client, monkeypatch):
    _, project = stub_children(client, monkeypatch)
    def oversized(seed, **kwargs):
        result = project(seed, **kwargs)
        md = ec.plain_json(result.metadata)
        md["request"]["implementation_identity"]["worker_implementation_revision"] = "x" * ec.MAX_REQUEST_BYTES
        return EncodedPayload(metadata=md)
    monkeypatch.setattr(client._projector, "run", oversized)
    monkeypatch.setattr(client.cache, "lock_for", lambda *a: pytest.fail("cache for oversized request"))
    with pytest.raises(EncoderProtocolError, match="byte cap"):
        client.encode(GenerationConfig(), encoder_dtype="fp16")


# Real-child interpreter whose metadata view omits the reviewed vision wheel.
# The reviewed lock does install torchvision, so the absent-wheel scenario this
# test exists for must be simulated at the same metadata boundary the worker
# validates. Production code is untouched: the child still runs its own fresh
# package policy before any cache or model access.
NO_VISION_CHILD = """#!{python}
import runpy
import sys
from importlib import metadata

HIDDEN = "torchvision"
_original_distribution = metadata.distribution
_original_distributions = metadata.distributions


def _hidden(name):
    return isinstance(name, str) and name.lower().replace("_", "-") == HIDDEN


def distribution(name=None, *args, **kwargs):
    if _hidden(name):
        raise metadata.PackageNotFoundError(name)
    return _original_distribution(name, *args, **kwargs)


def distributions(*args, **kwargs):
    for dist in _original_distributions(*args, **kwargs):
        if not _hidden(dist.metadata.get("Name")):
            yield dist


metadata.distribution = distribution
metadata.distributions = distributions

arguments = sys.argv[1:]
if arguments[:3] != ["-B", "-m", "qwen_engine.encoder_worker"]:
    raise SystemExit("unexpected child invocation")
sys.argv = ["qwen_engine.encoder_worker", *arguments[3:]]
runpy.run_module("qwen_engine.encoder_worker", run_name="__main__")
"""


def test_actual_fresh_child_package_refusal_happens_before_cache(tmp_path):
    wrapper = tmp_path / "child-without-reviewed-vision"
    wrapper.write_text(NO_VISION_CHILD.format(python=sys.executable), encoding="ascii")
    wrapper.chmod(0o755)
    client = ec.ProductionEncoder(
        manifest_path=ROOT / "models/manifest.json",
        models_dir=tmp_path / "never-accessed-models",
        scratch_dir=tmp_path, timeout_s=20, headroom_bytes=0,
        python=str(wrapper), cache=EmbeddingCache(tmp_path / "cache"))
    for name in ("lock_for", "get", "put"):
        setattr(client.cache, name, lambda *a: pytest.fail("cache after actual refusal"))
    with pytest.raises(EncoderWorkerError) as caught:
        client.encode(GenerationConfig(), encoder_dtype="fp16")
    assert caught.value.exit_code == 1 and "torchvision: expected" in caught.value.stderr
    assert not client.models_dir.exists()
    assert not list(Path(client._projector.scratch_dir).glob("qwen-enc-*"))


def test_same_key_concurrent_clients_still_load_only_once(client, monkeypatch):
    stub_children(client, monkeypatch)
    loaded, release = threading.Event(), threading.Event()
    loads, results, errors = [], [], []
    def load(request, **kwargs):
        loads.append(request)
        loaded.set()
        assert release.wait(5)
        return payload_for(request)
    monkeypatch.setattr(client._loader, "run", load)
    def encode():
        try:
            results.append(client.encode(GenerationConfig(), encoder_dtype="fp16"))
        except BaseException as exc:
            errors.append(exc)
    first, second = threading.Thread(target=encode), threading.Thread(target=encode)
    first.start()
    try:
        assert loaded.wait(5)
        second.start()
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not errors and len(loads) == 1 and len(results) == 2 and results[0] == results[1]


# -- operator host-scope attestation ----------------------------------------


def test_attestation_flag_is_forwarded_only_to_the_loading_worker(tmp_path):
    args = dict(manifest_path="manifest.json", models_dir="models", scratch_dir=tmp_path,
                headroom_bytes=0, timeout_s=20, python="selected/python")
    plain = ec.ProductionEncoder(**args)
    attested = ec.ProductionEncoder(attested_unbounded_host=True, **args)
    assert "--attest-unbounded-host" not in plain._loader.command
    assert "--attest-unbounded-host" not in plain._projector.command
    assert attested._loader.command[-1] == "--attest-unbounded-host"
    assert "--attest-unbounded-host" not in attested._projector.command
    assert attested.models_dir == plain.models_dir


@pytest.mark.parametrize("value", ["yes", 1, 0, None])
def test_attestation_must_be_an_explicit_boolean(tmp_path, value):
    with pytest.raises(EncoderSetupError, match="explicit boolean"):
        ec.ProductionEncoder(manifest_path="manifest.json", models_dir="models",
                             scratch_dir=tmp_path, headroom_bytes=0,
                             timeout_s=20, python="selected/python",
                             attested_unbounded_host=value)
