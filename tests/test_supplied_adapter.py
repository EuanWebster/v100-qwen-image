"""ML-free trusted-component boundary regressions; no real generation claims."""
import hashlib
import json
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
from qwen_engine import encoder_production as ep, reference_store
from qwen_engine.encoder import Blob, EncodedPayload, cache_key
from qwen_engine.encoder_client import ProductionEncoding
from qwen_engine.encoder_identity import project_encoder_seed
from qwen_engine.manifest import load_manifest
from test_encoder import make_identity
from test_encoder_production import FakeTensor, FakeTorch, install_ml, st_bytes, triple

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


def test_calibration_facts_measure_the_processed_conditioning(setup):
    supplied = setup.make()
    supplied._state = "encoded"  # private fixture state, no production payload
    supplied._config = setup.config
    supplied._conditioning = SimpleNamespace(
        positive=SimpleNamespace(embeds=SimpleNamespace(shape=(1, 25, 16))),
        negative=None)
    setup.provider.last_cache_hit = True
    assert supplied.calibration_facts() == {
        "prompt_tokens": 25,
        "negative_prompt_tokens": None,
        "cfg_active": False,
        "encoder_dtype": "fp16",
        "execution_dtype": "fp16",
        "use_kv_cache": True,
        "embedding_cache_hit": True,
        "references": 0,
    }
    active = replace(setup.config, negative_prompt="blurry", true_cfg_scale=2.0)
    supplied._config = active
    supplied._conditioning = SimpleNamespace(
        positive=SimpleNamespace(embeds=SimpleNamespace(shape=(1, 25, 16))),
        negative=SimpleNamespace(embeds=SimpleNamespace(shape=(1, 12, 16))))
    facts = supplied.calibration_facts()
    assert facts["cfg_active"] is True
    assert facts["negative_prompt_tokens"] == 12
    # An encoder that never reported a cache decision stays unknown (None),
    # so the collector refuses rather than inventing a miss.
    del setup.provider.last_cache_hit
    assert supplied.calibration_facts()["embedding_cache_hit"] is None
    supplied.close()


# -- reference-image path (edit support, part B) -----------------------------


def _reference_capable() -> bool:
    from dataclasses import fields as _fields
    names = {f.name for f in _fields(GenerationConfig)}
    return {"references", "references_dir"} <= names


needs_references = pytest.mark.skipif(
    not _reference_capable(), reason="config references patch not applied yet")


class _MiniPipe:
    """Carries exactly the pinned ``encode_prompt`` signature the hook checks."""

    def __init__(self, extra_parameter=False):
        self.records = []
        self.extra_parameter = extra_parameter

    def encode_prompt(self, prompt, image, device, num_images_per_prompt,
                      prompt_embeds, prompt_embeds_mask, image_pad_mask):
        self.records.append(dict(prompt=prompt, image=image, device=device,
                                 num_images_per_prompt=num_images_per_prompt,
                                 prompt_embeds=prompt_embeds,
                                 prompt_embeds_mask=prompt_embeds_mask,
                                 image_pad_mask=image_pad_mask))
        return "original"


class _DriftPipe(_MiniPipe):
    def encode_prompt(self, prompt, image, device, num_images_per_prompt,  # noqa: D102
                      prompt_embeds, prompt_embeds_mask, image_pad_mask, extra):
        return super().encode_prompt(prompt, image, device, num_images_per_prompt,
                                     prompt_embeds, prompt_embeds_mask, image_pad_mask)


def _mask_bridge(pipe):
    from types import SimpleNamespace
    return SimpleNamespace(_pipeline=pipe, _device="cpu")


def _stub_mask():
    return SimpleNamespace(to=lambda device: ("mask", device))


def test_mask_hook_injects_by_identity_and_delegates_unmodified():
    pipe, embeds = _MiniPipe(), object()
    remove = module.SuppliedQwen21Adapter._install_image_pad_mask_hook(
        _mask_bridge(pipe), [(embeds, _stub_mask())])
    assert "encode_prompt" in vars(pipe)
    pipe.encode_prompt(prompt=None, image=["pil"], device="cpu",
                       num_images_per_prompt=1, prompt_embeds=embeds,
                       prompt_embeds_mask=None, image_pad_mask=None)
    assert pipe.records[-1]["image_pad_mask"] == ("mask", "cpu")
    # A foreign embeds tensor is passed through untouched (identity mismatch).
    pipe.encode_prompt(prompt=None, image=["pil"], device="cpu",
                       num_images_per_prompt=1, prompt_embeds=object(),
                       prompt_embeds_mask=None, image_pad_mask=None)
    assert pipe.records[-1]["image_pad_mask"] is None
    # An explicitly supplied mask is never overwritten.
    explicit = object()
    pipe.encode_prompt(prompt=None, image=["pil"], device="cpu",
                       num_images_per_prompt=1, prompt_embeds=embeds,
                       prompt_embeds_mask=None, image_pad_mask=explicit)
    assert pipe.records[-1]["image_pad_mask"] is explicit
    # Positional delegation keeps working (kwargs-only injection).
    pipe.encode_prompt(None, ["pil"], "cpu", 1, embeds, None, None)
    assert pipe.records[-1]["image_pad_mask"] is None
    assert all(record["image"] == ["pil"] for record in pipe.records)
    remove()
    assert "encode_prompt" not in vars(pipe)
    assert pipe.encode_prompt(prompt=None, image=None, device="cpu",
                              num_images_per_prompt=1, prompt_embeds=embeds,
                              prompt_embeds_mask=None,
                              image_pad_mask=None) == "original"


def test_mask_hook_matches_positive_and_negative_embeds_separately():
    pipe, positive, negative = _MiniPipe(), object(), object()
    positive_mask, negative_mask = _stub_mask(), SimpleNamespace(
        to=lambda device: ("negative", device))
    remove = module.SuppliedQwen21Adapter._install_image_pad_mask_hook(
        _mask_bridge(pipe), [(positive, positive_mask), (negative, negative_mask)])
    pipe.encode_prompt(prompt=None, image=["pil"], device="cpu",
                       num_images_per_prompt=1, prompt_embeds=negative,
                       prompt_embeds_mask=None, image_pad_mask=None)
    assert pipe.records[-1]["image_pad_mask"] == ("negative", "cpu")
    pipe.encode_prompt(prompt=None, image=["pil"], device="cpu",
                       num_images_per_prompt=1, prompt_embeds=positive,
                       prompt_embeds_mask=None, image_pad_mask=None)
    assert pipe.records[-1]["image_pad_mask"] == ("mask", "cpu")
    remove()


@pytest.mark.parametrize("case", ["drift", "stale", "missing-mask"])
def test_mask_hook_refuses_instead_of_guessing(case):
    pipe = _DriftPipe() if case == "drift" else _MiniPipe()
    bridge = _mask_bridge(pipe)
    stale = None
    if case == "drift":
        with pytest.raises(EngineError, match="signature"):
            module.SuppliedQwen21Adapter._install_image_pad_mask_hook(
                bridge, [(object(), _stub_mask())])
    elif case == "stale":
        stale = pipe.encode_prompt = lambda **kwargs: None  # stale instance override
        with pytest.raises(EngineError, match="override"):
            module.SuppliedQwen21Adapter._install_image_pad_mask_hook(
                bridge, [(object(), _stub_mask())])
    else:
        with pytest.raises(EngineError, match="image_pad_mask"):
            module.SuppliedQwen21Adapter._install_image_pad_mask_hook(
                bridge, [(object(), None)])
    if stale is None:  # refusals install nothing…
        assert "encode_prompt" not in vars(pipe)
    else:              # …and never clobber or clear an existing override
        assert vars(pipe)["encode_prompt"] is stale


class _FakePIL:
    def __init__(self, size=(64, 64), fail=False):
        self.size, self.fail, self.loaded = size, fail, False

    def load(self):
        if self.fail:
            raise ValueError("truncated pixel data")
        self.loaded = True


def _loader_bridge(monkeypatch, images_by_id, *, fail_open=()):
    from types import SimpleNamespace
    open_calls = []

    def open(stream):
        raw = stream.getvalue()
        open_calls.append(raw)
        if raw in fail_open:
            raise OSError("cannot identify image file")
        return images_by_id[raw]

    monkeypatch.setattr(module, "reference_store", SimpleNamespace(
        MAX_REFERENCE_DIM=4096,
        read_blob=lambda root, reference_id: next(
            raw for raw, image in images_by_id.items()
            if image is not None and reference_id == raw.decode())))
    # Module shape matches the real _libraries() return: the image MODULE at
    # _image (open() on it, its Image CLASS for isinstance checks).
    return SimpleNamespace(open=open, Image=_FakePIL), open_calls


def _reference_run(setup, monkeypatch, ids, images_by_id, *, fail_open=()):
    from types import SimpleNamespace
    image_module, open_calls = _loader_bridge(monkeypatch, images_by_id,
                                              fail_open=fail_open)
    supplied = setup.make()
    supplied.load(setup.config, setup.bundle.to_assets())
    supplied._image = image_module
    config = SimpleNamespace(references=ids, references_dir="/store")
    return supplied, config, open_calls


def test_reference_images_load_in_order_with_duplicates(setup, monkeypatch):
    first, second = _FakePIL((64, 64)), _FakePIL((32, 16))
    supplied, config, open_calls = _reference_run(
        setup, monkeypatch, ("first", "second", "first"),
        {b"first": first, b"second": second})
    images = supplied._load_reference_images(config)
    assert images == [first, second, first]
    assert all(image.loaded for image in images)
    assert open_calls == [b"first", b"second", b"first"]
    supplied.close()


def test_reference_images_require_id_and_store_pair(setup, monkeypatch):
    supplied, config, _ = _reference_run(setup, monkeypatch, (), {})
    with pytest.raises(EngineError, match="without ids or the store"):
        supplied._load_reference_images(config)
    config2 = SimpleNamespace(references=("first",), references_dir=None)
    with pytest.raises(EngineError, match="without ids or the store"):
        supplied._load_reference_images(config2)
    supplied.close()


def test_reference_decode_failures_are_typed(tmp_path, setup, monkeypatch):
    big = _FakePIL((5000, 40))
    supplied, config, _ = _reference_run(setup, monkeypatch, ("big",), {b"big": big})
    with pytest.raises(EngineError, match="must not exceed"):
        supplied._load_reference_images(config)
    supplied.close()
    supplied, config, _ = _reference_run(setup, monkeypatch, ("broken",),
                                         {b"broken": _FakePIL()}, fail_open=(b"broken",))
    with pytest.raises(EngineError, match="cannot be decoded"):
        supplied._load_reference_images(config)
    supplied.close()
    supplied, config, _ = _reference_run(setup, monkeypatch, ("truncated",),
                                         {b"truncated": _FakePIL(fail=True)})
    with pytest.raises(EngineError, match="cannot be decoded"):
        supplied._load_reference_images(config)
    supplied.close()


def test_reference_images_gate_on_encoded_state(setup, monkeypatch):
    supplied, config, _ = _reference_run(setup, monkeypatch, ("first",),
                                         {b"first": _FakePIL()})
    with pytest.raises(EngineError):  # still "loaded": encode has not run
        supplied.reference_images(config)
    supplied.close()


def test_reference_store_read_errors_propagate_verbatim(setup, monkeypatch):
    from types import SimpleNamespace
    from qwen_engine.reference_store import ReferenceStoreError

    def read_blob(root, reference_id):
        raise ReferenceStoreError("hash mismatch")

    monkeypatch.setattr(module, "reference_store", SimpleNamespace(
        MAX_REFERENCE_DIM=4096, read_blob=read_blob))
    supplied = setup.make()
    supplied.load(setup.config, setup.bundle.to_assets())
    supplied._image = SimpleNamespace(Image=None)
    with pytest.raises(ReferenceStoreError, match="hash mismatch"):
        supplied._load_reference_images(
            SimpleNamespace(references=("first",), references_dir="/store"))
    supplied.close()


@needs_references
def test_calibration_facts_count_the_ordered_references(setup):
    supplied = setup.make()
    supplied.load(setup.config, setup.bundle.to_assets())
    supplied._state = "encoded"  # private fixture state, no production payload
    supplied._config = replace(setup.config, references=("a" * 64, "b" * 64),
                               references_dir="/store")
    supplied._conditioning = SimpleNamespace(
        positive=SimpleNamespace(embeds=SimpleNamespace(shape=(1, 25, 16))),
        negative=None)
    setup.provider.last_cache_hit = True
    assert supplied.calibration_facts()["references"] == 2
    supplied.close()



def _image_payload_for(request):
    """Positive-only payload carrying a real image_pad_mask per branch (the
    same wire layout the real loading child produces)."""
    tensors = {}
    for side in ("positive", "negative") if request.kwargs["cfg_active"] else ("positive",):
        embeds, attention, image_mask = triple(shape=(1, 4, 16), code="F16", img_true=True)
        tensors[f"{side}.embeds"] = embeds
        tensors[f"{side}.attention_mask"] = attention
        tensors[f"{side}.image_pad_mask"] = image_mask
    descriptors = [{"name": name, "dtype": tensor.code, "shape": list(tensor.shape),
                    "bytes": len(tensor.raw)} for name, tensor in sorted(tensors.items())]
    canonical = lambda doc: json.dumps(doc, sort_keys=True, separators=(",", ":"))
    metadata = {"schema": ep.TENSOR_PAYLOAD_SCHEMA, "codec": ep.SAFETENSORS_CODEC,
                "cache_key": cache_key(request), "dtype": request.dtype,
                "cfg_active": request.kwargs["cfg_active"],
                "request": request.key_document(), "tensors": descriptors}
    st_metadata = {"codec": ep.SAFETENSORS_CODEC, "cache_key": cache_key(request),
                   "dtype": request.dtype,
                   "cfg_active": "true" if request.kwargs["cfg_active"] else "false",
                   "request_identity": canonical(request.key_document()),
                   "tensor_descriptors": canonical(descriptors)}
    return EncodedPayload(metadata=metadata,
                          blobs=(Blob("tensors", st_bytes(tensors, st_metadata)),))


@needs_references
def test_references_flow_end_to_end_from_config_through_the_pinned_call(tmp_path, monkeypatch):
    """Two references plus one duplicate travel the whole owned chain with
    only the ML edges faked: ``config.references`` binds ``request.images``
    through the REAL seed projection; the adapter resolves ids to store
    paths and hands them to the encoder as ``image_files``; the REAL store
    (bounded read + hash recheck) is hit once per position in order; real
    Pillow decodes the blobs; the pipeline call receives the ordered PIL
    list and the ``image_pad_mask`` instance hook injects the cached mask
    for exactly the conditioning that was encoded."""
    import io
    from types import SimpleNamespace
    from PIL import Image as PILImage

    class _Generator:
        def __init__(self, device=None):
            self.value = None
        def seed(self):
            return 7
        def manual_seed(self, value):
            self.value = value

    class _Torch(FakeTorch):
        float16 = "torch.float16"
        float32 = "torch.float32"
        Generator = _Generator
        @staticmethod
        def isfinite(tensor):
            return tensor.isfinite()

    torch = _Torch()
    install_ml(monkeypatch)  # ep-internal safetensors/torch use the shared fakes
    monkeypatch.setattr(FakeTensor, "to",
                        lambda self, device=None, dtype=None: self, raising=False)
    monkeypatch.setattr(module, "_libraries",
                        lambda: (torch, PILImage, (_E2EPipe, _E2ETransformer, _E2EVAE, _E2EScheduler)))

    manifest = load_manifest(ROOT / "models/manifest.json")
    bundle = LocalReferenceBundle(profile=LOCAL_REFERENCE_PROFILE, root=tmp_path,
        root_resolved=tmp_path, repo=manifest.artifacts[0].repo,
        revision=manifest.artifacts[0].revision,
        assets={r: tuple(tmp_path / n for n in names) for r, names in LOCAL_REFERENCE_LAYOUT.items()},
        component_dirs={r: tmp_path / n for r, n in ROLE_DIRECTORIES.items()},
        fingerprints=reference_bundle_fingerprints(manifest))
    store = tmp_path / "store"
    store.mkdir()
    ids = []
    for colour, mode in (((200, 0, 0), "RGB"), ((0, 0, 200), "RGB")):
        stream = io.BytesIO()
        PILImage.new(mode, (16, 8), colour).save(stream, format="PNG")
        ids.append(reference_store.put_bytes(store, stream.getvalue()))
    ordered = (ids[0], ids[1], ids[0])  # duplicate position is legal and ordered
    config = GenerationConfig(output_dir=tmp_path, height=256, width=256,
                              references=ordered, references_dir=str(store))

    reads = []
    real_read_blob = reference_store.read_blob

    def counted_read_blob(root, reference_id):
        reads.append(reference_id)
        return real_read_blob(root, reference_id)

    monkeypatch.setattr(reference_store, "read_blob", counted_read_blob)

    captured = {}
    encoder_artifact = reference_bundle_fingerprints(manifest).encoder

    def encode_provider(cfg, *, encoder_dtype, cancel_event, image_files):
        captured["image_files"] = tuple(image_files)
        seed = project_encoder_seed(cfg, manifest, encoder_dtype=encoder_dtype)
        assert seed.images == ordered, "the seed must carry the ordered ids"
        completed = replace(seed, implementation_identity=make_identity(
            worker_implementation_revision="synthetic-e2e-child",
            encoder_artifact_set_sha256=encoder_artifact))
        return ProductionEncoding(completed, _image_payload_for(completed))

    pipe = _E2EPipe()
    supplied = module.SuppliedQwen21Adapter(
        pipeline=pipe, encoder=SimpleNamespace(encode=encode_provider),
        manifest=manifest, bundle=bundle, encoder_dtype="fp16")
    supplied.load(config, bundle.to_assets())
    branches = supplied.encode(config)

    assert captured["image_files"] == tuple(
        str(reference_store.path_for(store, reference_id)) for reference_id in ordered)
    assert branches.negative is None

    latents = supplied.denoise(config, branches)

    # One store read per position, in order; the recheck happens inside.
    assert reads == list(ordered)
    # The pipeline call received the ordered PIL list, and the pinned
    # encode_prompt contract saw the cached mask only via the hook.
    images = pipe.call["image"]
    assert len(images) == 3 and images[0] is not images[1]
    assert (images[0].mode, images[0].size) == ("RGB", (16, 8))
    assert pipe.records[-1]["prompt_embeds"] is pipe.call["prompt_embeds"]
    assert pipe.records[-1]["image_pad_mask"] is branches.positive.image_pad_mask
    assert "encode_prompt" not in vars(pipe)  # the finally removed the override
    assert tuple(latents.shape) == (1, config.latent_tokens, 64)
    supplied.close()


class _E2ETransformer:
    training = False
    def __init__(self):
        self.config = SimpleNamespace(in_channels=64, out_channels=64, context_in_dim=16)
        self.parameter = SimpleNamespace(is_meta=False, device=SimpleNamespace(type="cpu"),
                                        dtype="torch.float16")
        self.backend = None
    def parameters(self):
        return iter([self.parameter])
    def modules(self):
        return iter([self])
    def set_attention_backend(self, name):
        self.backend = name


class _E2EVAE(_E2ETransformer):
    def __init__(self):
        super().__init__()
        self.config.z_dim = 64
        self.tiled = None
    def enable_tiling(self):
        self.tiled = True
    def disable_tiling(self):
        self.tiled = False


class _E2EScheduler:
    pass


class _E2EPipe:
    """Pinned-shape witness: encode_prompt has EXACTLY the signature the
    hook checks, and __call__ reproduces the pinned contract of fetching
    cached conditioning WITHOUT ever forwarding image_pad_mask."""
    def __init__(self):
        self.transformer, self.vae, self.scheduler = _E2ETransformer(), _E2EVAE(), _E2EScheduler()
        self.text_encoder, self.vae_scale_factor, self.latent_channels = None, 16, 64
        self._execution_device = SimpleNamespace(type="cpu")
        self.records, self.call = [], None

    def encode_prompt(self, prompt, image, device, num_images_per_prompt,
                      prompt_embeds, prompt_embeds_mask, image_pad_mask):
        self.records.append(dict(image=image, prompt_embeds=prompt_embeds,
                                 prompt_embeds_mask=prompt_embeds_mask,
                                 image_pad_mask=image_pad_mask))
        return "original"

    def __call__(self, prompt, image, height, width, **kwargs):
        self.call = dict(prompt=prompt, image=image, height=height, width=width, **kwargs)
        self.encode_prompt(None, image, "cpu", 1,
                           prompt_embeds=kwargs["prompt_embeds"],
                           prompt_embeds_mask=kwargs["prompt_embeds_mask"],
                           image_pad_mask=None)
        return SimpleNamespace(images=FakeTensor("F16", (1, 256, 64), values=[0.5] * 256))
