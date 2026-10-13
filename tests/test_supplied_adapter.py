"""ML-free trusted-component boundary regressions; no real generation claims."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from qwen_engine.adapters import supplied_qwen21 as module
from qwen_engine.adapters.base import ManagedGenerationAdapter
from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.encoder import EncoderCancelledError
from qwen_engine.errors import EngineError, GenerationCancelled
from qwen_engine.local_bundle import (
    LOCAL_REFERENCE_LAYOUT, LOCAL_REFERENCE_PROFILE, ROLE_DIRECTORIES,
    LocalReferenceBundle, reference_bundle_fingerprints,
)
from qwen_engine.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    manifest = load_manifest(ROOT / "models/manifest.json")
    bundle = LocalReferenceBundle(profile=LOCAL_REFERENCE_PROFILE, root=tmp_path, root_resolved=tmp_path,
        repo=manifest.artifacts[0].repo, revision=manifest.artifacts[0].revision,
        assets={r: tuple(tmp_path / n for n in names) for r, names in LOCAL_REFERENCE_LAYOUT.items()},
        component_dirs={r: tmp_path / n for r, n in ROLE_DIRECTORIES.items()},
        fingerprints=reference_bundle_fingerprints(manifest))

    class Transformer:
        training = False
        def __init__(self):
            self.config = SimpleNamespace(in_channels=64, out_channels=64, context_in_dim=16)
            self.parameter = SimpleNamespace(is_meta=False, device="cpu", dtype="fp16")
            self.backend = None
        def parameters(self):
            return iter([self.parameter])
        def modules(self):
            return iter([self])
        def set_attention_backend(self, name):
            self.backend = name

    class VAE(Transformer):
        def __init__(self):
            super().__init__()
            self.config.z_dim = 64
            self.tiled = None
        def enable_tiling(self):
            self.tiled = True
        def disable_tiling(self):
            self.tiled = False

    class Scheduler:
        pass

    class Pipe:
        def __init__(self):
            self.transformer, self.vae, self.scheduler = Transformer(), VAE(), Scheduler()
            self.text_encoder, self.vae_scale_factor, self.latent_channels = None, 16, 64
            self._execution_device = "cpu"

    class Provider:
        def encode(self, *args, **kwargs):
            return object()

    pipe, provider = Pipe(), Provider()
    monkeypatch.setattr(module, "_libraries", lambda: (
        SimpleNamespace(float16="fp16", float32="fp32"), None, (Pipe, Transformer, VAE, Scheduler)))
    def make(**kwargs):
        options = dict(pipeline=pipe, encoder=provider, manifest=manifest, bundle=bundle, encoder_dtype="fp16")
        options.update(kwargs)
        return module.SuppliedQwen21Adapter(**options)
    return SimpleNamespace(make=make, pipe=pipe, provider=provider, bundle=bundle, manifest=manifest,
                           config=GenerationConfig(output_dir=tmp_path))


@pytest.mark.parametrize("options", [
    {"encoder_dtype": "float16"}, {"encoder_dtype": None}, {"execution_dtype": "bf16"},
    {"execution_dtype": " fp16"}, {"use_kv_cache": 1}, {"encoder": object()},
    {"manifest": None}, {"bundle": None},
])
def test_constructor_refuses_invalid_metadata(setup, options):
    with pytest.raises(ConfigError):
        setup.make(**options)


def test_bundle_path_and_digest_coherence(setup):
    for bundle in (replace(setup.bundle, assets={}), replace(setup.bundle, component_dirs={}),
                   replace(setup.bundle, fingerprints=replace(setup.bundle.fingerprints, encoder="0" * 64))):
        with pytest.raises(ConfigError, match="manifest/layout"):
            setup.make(bundle=bundle)


@pytest.mark.parametrize("changes", [{"quant": "q8"}, {"quant": "q4"}, {"offload": "offload"},
    {"offload": "streamed"}, {"attention_backend": "efficient"}, {"attention_backend": "eager"}])
def test_unsupported_policy_refused_before_libraries(setup, monkeypatch, changes):
    monkeypatch.setattr(module, "_libraries", lambda: pytest.fail("heavy imports reached"))
    with pytest.raises(ConfigError):
        setup.make().load(replace(setup.config, **changes), setup.bundle.to_assets())


def test_assets_refused_before_libraries(setup, monkeypatch):
    monkeypatch.setattr(module, "_libraries", lambda: pytest.fail("heavy imports reached"))
    with pytest.raises(ConfigError, match="assets"):
        setup.make().load(setup.config, {})


@pytest.mark.parametrize("tiled", [False, True])
def test_attach_exact_classes_native_backend_tiling_and_cleanup(setup, tiled):
    supplied = setup.make()
    assert isinstance(supplied, ManagedGenerationAdapter)
    supplied.load(replace(setup.config, vae_tiled_decode=tiled), setup.bundle.to_assets())
    assert setup.pipe.transformer.backend == "native" and setup.pipe.vae.tiled is tiled
    supplied.close()
    supplied.close()
    assert supplied._state == "closed" and supplied._pipeline is None and supplied._encoder is None


@pytest.mark.parametrize("mutation", ["pipeline", "transformer", "vae", "scheduler", "encoder", "hooks",
    "module_hook", "meta", "device", "dtype", "training", "scale", "channels", "zdim", "in", "out"])
def test_attach_rejects_incompatible_components(setup, mutation):
    pipeline = setup.pipe
    if mutation == "pipeline":
        pipeline = None
    elif mutation in ("transformer", "vae", "scheduler"):
        setattr(pipeline, mutation, object())
    elif mutation == "encoder":
        pipeline.text_encoder = object()
    elif mutation == "hooks":
        pipeline._all_hooks = [object()]
    elif mutation == "module_hook":
        pipeline.vae._hf_hook = object()
    elif mutation in ("meta", "device", "dtype"):
        setattr(pipeline.transformer.parameter, {"meta": "is_meta"}.get(mutation, mutation),
                {"meta": True, "device": "cuda", "dtype": "bf16"}[mutation])
    elif mutation == "training":
        pipeline.vae.training = True
    elif mutation == "scale":
        pipeline.vae_scale_factor = 8
    elif mutation == "channels":
        pipeline.latent_channels = 16
    else:
        target = pipeline.vae.config if mutation == "zdim" else pipeline.transformer.config
        setattr(target, {"zdim": "z_dim", "in": "in_channels", "out": "out_channels"}[mutation], 16)
    with pytest.raises(EngineError):
        setup.make(pipeline=pipeline).load(setup.config, setup.bundle.to_assets())


def test_phase_order_config_immutability_and_provider_output(setup):
    supplied = setup.make()
    with pytest.raises(EngineError):
        supplied.encode(setup.config)
    supplied.load(setup.config, setup.bundle.to_assets())
    with pytest.raises(ConfigError):
        supplied.encode(replace(setup.config, seed=1))
    with pytest.raises(EngineError, match="ProductionEncoding"):
        supplied.encode(setup.config)
    with pytest.raises(EngineError):
        supplied.load(setup.config, setup.bundle.to_assets())
    supplied.close()
    with pytest.raises(EngineError):
        supplied.load(setup.config, setup.bundle.to_assets())


@pytest.mark.parametrize("requested", [False, True])
def test_encoder_cancellation_only_acknowledges_own_event(setup, requested, monkeypatch):
    event = threading.Event()
    supplied = setup.make()
    supplied.set_cancel_event(event)
    supplied.load(setup.config, setup.bundle.to_assets())
    def encode(*args, **kwargs):
        assert kwargs["cancel_event"] is event
        if requested:
            event.set()
        raise EncoderCancelledError("fixture cancellation")
    monkeypatch.setattr(setup.provider, "encode", encode)
    with pytest.raises(GenerationCancelled if requested else EncoderCancelledError):
        supplied.encode(setup.config)
    supplied.close()


def test_precancel_prevents_library_import(setup, monkeypatch):
    event = threading.Event()
    event.set()
    supplied = setup.make()
    supplied.set_cancel_event(event)
    monkeypatch.setattr(module, "_libraries", lambda: pytest.fail("heavy import reached"))
    with pytest.raises(GenerationCancelled):
        supplied.load(setup.config, setup.bundle.to_assets())
    supplied.close()


def test_bad_binding_and_config_refused(setup):
    supplied = setup.make()
    with pytest.raises(ConfigError):
        supplied.set_cancel_event(object())
    with pytest.raises(ConfigError):
        supplied.load(None, setup.bundle.to_assets())
    supplied.load(setup.config, setup.bundle.to_assets())
    with pytest.raises(ConfigError):
        supplied.set_cancel_event(threading.Event())
    supplied.close()


def test_conditioning_requires_owned_object(setup):
    supplied = setup.make()
    supplied.load(setup.config, setup.bundle.to_assets())
    supplied._state = "encoded"  # private fixture state, no production payload
    with pytest.raises(EngineError, match="owned"):
        supplied.denoise(setup.config, object())
    supplied.close()


def test_fp32_execution_cannot_become_an_unreviewed_gpu_profile(setup):
    setup.pipe._execution_device = "cuda:0"
    for model in (setup.pipe.transformer, setup.pipe.vae):
        model.parameter.device, model.parameter.dtype = "cuda:0", "fp32"
    supplied = setup.make(execution_dtype="fp32")
    with pytest.raises(EngineError, match="CPU-witness-only"):
        supplied.load(setup.config, setup.bundle.to_assets())
    supplied.close()


def test_png_writer_caps_extent_and_refuses_fileno(monkeypatch):
    monkeypatch.setattr(module, "MAX_PNG_BYTES", 8)
    with module._BoundedPNG() as stream:
        stream.write(b"12345678")
        with pytest.raises(EngineError):
            stream.write(b"9")
        stream.seek(0)
        stream.write(b"ab")
        assert stream.getvalue() == b"ab345678"
        stream.seek(9)
        with pytest.raises(EngineError):
            stream.write(b"x")
        with pytest.raises(OSError):
            stream.fileno()
