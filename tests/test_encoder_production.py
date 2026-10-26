"""Production encoder helper + bounded safetensors codec — synthetic tests.

Everything here is ML-free and offline: torch / safetensors / diffusers are
NEVER imported (the guarded runner denies them outright; the module itself
imports them only lazily inside production calls, which these tests redirect
to small fakes through the module's own ``_load_torch`` / ``_load_safetensors``
seams). The fake safetensors writes/reads the real byte layout (8-byte header
length + JSON header + data region) so the stdlib pre-deserialization bounds
are exercised against genuine framing. No real model, processor, tokenizer,
checkpoint or network is involved; the pinned-ML-interpreter synthetic pass
lives in ``tools/synthetic_ml_encoder_check.py`` (skipped explicitly when
those interpreters are absent — a skip is never a validation claim).
"""
from __future__ import annotations

import ast
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_engine.encoder import (
    PINNED_DIFFUSERS_COMMIT,
    PINNED_TRANSFORMERS_COMMIT,
    Blob,
    ComponentIdentity,
    EncodedPayload,
    EncoderImplementationIdentity,
    EncoderProtocolError,
    EncoderRequest,
    EncoderRequestError,
    EncoderSetupError,
    EncoderWorkerError,
    cache_key,
)
import qwen_engine.encoder_production as ep

HEAVY_MODULES = ("torch", "diffusers", "transformers", "safetensors")

REV_A = "a" * 40
SHA_A = "b" * 64
SHA_B = "c" * 64
SHA_C = "d" * 64

MODULE_SOURCE = (Path(__file__).resolve().parents[1] / "src" / "qwen_engine"
                 / "encoder_production.py")

ML_PYTHONS = [p for p in os.environ.get("QWEN_ML_PYTHONS", "").split(os.pathsep)
              if p.strip()]


# ---------------------------------------------------------------------------
# Synthetic tensor + safetensors fakes (real byte layout, no torch)
# ---------------------------------------------------------------------------

CODE_ITEMSIZE = {code: v[1]
                 for code, v in ep.ST_CODE_TO_TORCH_STR_ITEMSIZE.items()}
CODE_TORCH_STR = {code: v[0]
                  for code, v in ep.ST_CODE_TO_TORCH_STR_ITEMSIZE.items()}
FLOAT_CODES = frozenset({"F16", "BF16", "F32"})


def _pack(code: str, values) -> bytes:
    if code == "F32":
        return b"".join(struct.pack("<f", v) for v in values)
    if code == "F16":
        return b"".join(struct.pack("<e", v) for v in values)
    if code == "BF16":
        # truncated float32 — exact for the representable values used here
        return b"".join(struct.pack("<f", v)[2:] for v in values)
    if code == "I64":
        return b"".join(struct.pack("<q", v) for v in values)
    if code == "BOOL":
        return bytes(1 if v else 0 for v in values)
    raise AssertionError(code)


def _unpack(code: str, raw: bytes) -> list:
    n = CODE_ITEMSIZE[code]
    if code == "F32":
        return [struct.unpack("<f", raw[i:i + n])[0]
                for i in range(0, len(raw), n)]
    if code == "F16":
        return [struct.unpack("<e", raw[i:i + n])[0]
                for i in range(0, len(raw), n)]
    if code == "BF16":
        return [struct.unpack("<f", b"\x00\x00" + raw[i:i + n])[0]
                for i in range(0, len(raw), n)]
    if code == "I64":
        return [struct.unpack("<q", raw[i:i + n])[0]
                for i in range(0, len(raw), n)]
    if code == "BOOL":
        return [bool(b) for b in raw]
    raise AssertionError(code)


class _Flag:
    def __init__(self, ok: bool) -> None:
        self._ok = ok

    def all(self) -> "_Flag":
        return self

    def item(self) -> bool:
        return self._ok


class FakeTensor:
    """Test-only stand-in for torch.Tensor, including CPU/layout/reductions."""

    def __init__(self, code: str, shape, values=None, raw: bytes | None = None
                 ) -> None:
        self.code = code
        self.shape = tuple(shape)
        self.device = SimpleNamespace(type="cpu")
        self.layout = "strided"
        if raw is None:
            raw = _pack(code, values)
        self.raw = bytes(raw)

    @property
    def dtype(self) -> str:
        return CODE_TORCH_STR[self.code]

    def is_floating_point(self) -> bool:
        return self.code in FLOAT_CODES

    def is_contiguous(self):
        return True

    def min(self):
        return _Flag(min(self.tolist()))

    def max(self):
        return _Flag(max(self.tolist()))

    def any(self):
        return _Flag(any(self.tolist()))

    def isfinite(self) -> _Flag:
        return _Flag(all(math.isfinite(v) for v in self.tolist()))

    def flatten(self) -> "FakeTensor":
        return self

    def tolist(self) -> list:
        return _unpack(self.code, self.raw)

    def raw_bytes(self) -> bytes:
        return self.raw


def embeds_tensor(shape=(1, 4, 8), code="F16", nan=False) -> FakeTensor:
    n = shape[0] * shape[1] * shape[2]
    values = [0.5 * (i % 16) for i in range(n)]
    if nan:
        values[n // 2] = float("nan")
    return FakeTensor(code, shape, values)


def mask_tensor(values, code="I64", shape=None) -> FakeTensor:
    shape = shape or (1, len(values))
    return FakeTensor(code, shape, list(values))


def image_mask_tensor(shape=(1, 4), any_true=False) -> FakeTensor:
    n = shape[0] * shape[1]
    values = [False] * n
    if any_true:
        values[0] = True
    return FakeTensor("BOOL", shape, values)


def triple(shape=(1, 4, 8), code="F16", mask="ones", nan=False,
           img_true=False) -> tuple:
    s = shape[1]
    if mask == "ones":
        m = mask_tensor([1] * s)
    elif mask == "none":
        m = None
    elif mask == "zeros":
        m = mask_tensor([0] * s)
    elif mask == "twos":
        m = mask_tensor([1] * (s - 1) + [2])
    else:
        raise AssertionError(mask)
    return (embeds_tensor(shape, code, nan), m,
            image_mask_tensor((1, s), img_true))


def st_bytes(tensors: dict, metadata=None, *, header_text=None,
             header_len=None, data: bytes | None = None,
             header_over: dict | None = None) -> bytes:
    """Serialize the real safetensors byte layout from fake tensors (or a
    hand-made header for the malformed cases)."""
    entries: dict = {}
    body = b""
    for name in sorted(tensors):
        t = tensors[name]
        start = len(body)
        body += t.raw
        entries[name] = {"dtype": t.code, "shape": list(t.shape),
                         "data_offsets": [start, len(body)]}
    if header_over:
        for name, over in header_over.items():
            entries.setdefault(name, {}).update(over)
    if metadata is not None:
        entries["__metadata__"] = metadata
    text = (header_text if header_text is not None
            else json.dumps(entries, sort_keys=True, separators=(",", ":")))
    hb = text.encode("utf-8")
    length = len(hb) if header_len is None else header_len
    return struct.pack("<Q", length) + hb + (body if data is None else data)


def st_split(data: bytes) -> tuple[dict, bytes]:
    n = int.from_bytes(data[:8], "little")
    return json.loads(data[8:8 + n].decode("utf-8")), data[8 + n:]


class FakeST:
    """Fake ``safetensors.torch``: real byte layout, no library, no casts."""

    def __init__(self) -> None:
        self.save_calls: list = []
        self.load_error: Exception | None = None
        self.load_calls = 0

    def save(self, tensors, metadata=None) -> bytes:
        self.save_calls.append((sorted(tensors), metadata))
        return st_bytes(tensors, metadata)

    def load(self, data: bytes) -> dict:
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        header, body = st_split(data)
        out = {}
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            start, end = spec["data_offsets"]
            out[name] = FakeTensor(spec["dtype"], spec["shape"],
                                   raw=body[start:end])
        return out


class _IM:
    def __init__(self, torch: "FakeTorch") -> None:
        self.torch = torch

    def __enter__(self) -> "_IM":
        self.torch.im_enter += 1
        return self

    def __exit__(self, *exc) -> bool:
        self.torch.im_exit += 1
        return False


class FakeTorch:
    Tensor = FakeTensor
    strided = "strided"
    def __init__(self) -> None:
        self.im_enter = 0
        self.im_exit = 0
        self.devices: list = []

    def device(self, name: str):
        self.devices.append(name)
        return ("cpu-device", name)

    def inference_mode(self) -> _IM:
        return _IM(self)


class FakePipeline:
    """Stands in for the SUPPLIED pinned pipeline: only encode_prompt, and
    the helper must be the only caller."""

    def __init__(self, by_prompt: dict, raises=None) -> None:
        self.by_prompt = by_prompt
        self.raises = raises
        self.calls: list[dict] = []

    def encode_prompt(self, *, prompt, image, device, num_images_per_prompt):
        self.calls.append(dict(prompt=prompt, image=image, device=device,
                               num_images_per_prompt=num_images_per_prompt))
        if self.raises is not None:
            exc = self.raises(prompt) if callable(self.raises) else self.raises
            if exc is not None:
                raise exc
        return self.by_prompt[prompt]


def install_ml(monkeypatch):
    torch = FakeTorch()
    st = FakeST()
    monkeypatch.setattr(ep, "_load_torch", lambda: torch)
    monkeypatch.setattr(ep, "_load_safetensors", lambda: st)
    return torch, st


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def make_identity(**over) -> EncoderImplementationIdentity:
    base = dict(
        encoder_artifact_set_sha256=SHA_C,
        worker_protocol_revision="worker-protocol.v1",
        worker_implementation_revision="worker-impl.v1",
        prompt_extraction_revision="extraction.v1",
        diffusers_commit=PINNED_DIFFUSERS_COMMIT,
        transformers_commit=PINNED_TRANSFORMERS_COMMIT,
        torch_build="cu126",
        torch_version="2.14.1",
        safetensors_version="0.8.0",
        torchvision_version="absent",
    )
    base.update(over)
    return EncoderImplementationIdentity(**base)


def make_request(**over) -> EncoderRequest:
    base = dict(
        prompt="a red cube on a table",
        dtype="fp16",
        model_repo="Qwen/Qwen-Image-2.1",
        model_revision=REV_A,
        processor=ComponentIdentity(sha256=SHA_A),
        tokenizer=ComponentIdentity(revision=REV_A),
        chat_template=ComponentIdentity(sha256=SHA_B),
        kwargs={"cfg_active": False},
        implementation_identity=make_identity(),
    )
    base.update(over)
    return EncoderRequest(**base)


def prod_request(**over) -> EncoderRequest:
    base = dict(kwargs={"cfg_active": True}, negative_prompt="ugly, blurry")
    base.update(over)
    return make_request(**base)


# ---------------------------------------------------------------------------
# Import-boundary + static contract guards
# ---------------------------------------------------------------------------


def test_module_imports_no_heavy_modules_in_process():
    for name in HEAVY_MODULES:
        assert name not in sys.modules, f"{name} leaked into the suite"


def test_module_imports_no_heavy_modules_in_subprocess():
    proc = subprocess.run(
        [sys.executable, "-B", "-c",
         "import sys; sys.path.insert(0, %r); "
         "import qwen_engine.encoder_production as m; "
         "bad=[m2 for m2 in sys.modules if m2.split('.')[0] in %r]; "
         "print(bad); assert not bad" % (str(MODULE_SOURCE.parents[1] / "src"),
                                         list(HEAVY_MODULES))],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_source_top_level_imports_are_stdlib_or_engine_only():
    tree = ast.parse(MODULE_SOURCE.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & {"torch", "diffusers", "transformers", "safetensors"}, roots


def test_source_never_catches_baseexception():
    tree = ast.parse(MODULE_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            assert node.type is not None, "bare except: catches BaseException"
            name = ast.unparse(node.type)
            assert "BaseException" not in name, name


def test_source_does_not_reimplement_prompt_encoding():
    """Executable code (docstrings stripped) must not touch the tokenizer,
    the chat template, control tokens or hooks — the supplied pipeline owns
    prompt formatting."""
    tree = ast.parse(MODULE_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                    and len(body) > 1):
                body.pop(0)
    code = ast.unparse(tree)
    for forbidden in ("apply_chat_template", "im_start", "im_end",
                      "tokenizer", "chat_template", "register_forward_hook",
                      "_drop_idx"):
        assert forbidden not in code, forbidden


def test_generic_encoder_module_untouched_by_this_milestone():
    """The generic transport/cache module must stay stdlib-only: the whole
    encoder.py import graph stays heavy-free in this process."""
    assert "torch" not in sys.modules and "safetensors" not in sys.modules
    tree = ast.parse((MODULE_SOURCE.parents[0] / "encoder.py").read_text(
        encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & {"torch", "diffusers", "transformers", "safetensors"}


# ---------------------------------------------------------------------------
# Production request validation (before any heavy import)
# ---------------------------------------------------------------------------


def test_valid_inactive_and_active_requests_validate():
    v = ep.validate_production_request(make_request())
    assert v.cfg_active is False
    v = ep.validate_production_request(prod_request())
    assert v.cfg_active is True
    # the negative text stays in the identity even when inactive
    inactive = ep.validate_production_request(
        make_request(negative_prompt="ugly, blurry"))
    assert inactive.cfg_active is False
    assert inactive.request.key_document()["negative_prompt"] == "ugly, blurry"


@pytest.mark.parametrize("dtype", list(ep.CANONICAL_DTYPES))
def test_canonical_dtypes_accepted(dtype):
    v = ep.validate_production_request(make_request(dtype=dtype))
    assert v.request.dtype == dtype


@pytest.mark.parametrize("dtype", ["FP16", "f16", "fp64", "fp16 ", "", None, 16])
def test_noncanonical_dtype_refused(dtype):
    req = make_request(dtype="fp16")
    if not isinstance(dtype, str):
        object.__setattr__(req, "dtype", dtype)
    else:
        object.__setattr__(req, "dtype", dtype)
    with pytest.raises(EncoderRequestError, match="dtype"):
        ep.validate_production_request(req)


@pytest.mark.parametrize("kwargs", [
    {},
    {"cfg_active": 0},
    {"cfg_active": 1},
    {"cfg_active": "true"},
    {"cfg_active": None},
    {"cfg_active": False, "guidance_scale": 3.0},
    {"cfg_active": {"nested": True}},
])
def test_kwargs_must_be_exactly_cfg_active_bool(kwargs):
    req = make_request()
    object.__setattr__(req, "kwargs", kwargs)
    with pytest.raises(EncoderRequestError, match="kwargs"):
        ep.validate_production_request(req)


@pytest.mark.parametrize("negative", ["", "   ", "\n\t "])
def test_active_cfg_requires_nonblank_negative(negative):
    req = make_request(kwargs={"cfg_active": True}, negative_prompt="x")
    object.__setattr__(req, "negative_prompt", negative)
    with pytest.raises(EncoderRequestError, match="negative_prompt"):
        ep.validate_production_request(req)


@pytest.mark.parametrize("negative", ["", "anything"])
def test_inactive_cfg_never_validates_or_encodes_negative(negative, monkeypatch):
    req = make_request(negative_prompt=negative)
    v = ep.validate_production_request(req)
    assert v.cfg_active is False
    torch, st = install_ml(monkeypatch)
    pipeline = FakePipeline({"a red cube on a table": triple()})
    ep.encode_with_supplied_pipeline(pipeline, req)
    assert [c["prompt"] for c in pipeline.calls] == [req.prompt]
    assert negative in str(req.key_document())  # identity keeps the text


def test_prompt_scalar_nonblank_and_image_count_cap():
    req = make_request()
    object.__setattr__(req, "prompt", "   \n ")
    with pytest.raises(EncoderRequestError, match="prompt"):
        ep.validate_production_request(req)
    req = make_request()
    object.__setattr__(req, "prompt", ["not", "scalar"])
    with pytest.raises(EncoderRequestError, match="prompt"):
        ep.validate_production_request(req)
    # ordered references are accepted production conditioning up to the cap
    ok = make_request(images=tuple(f"{i:064x}"
                                   for i in range(ep.REFERENCE_MAX_FILES)))
    assert ep.validate_production_request(ok).cfg_active is False
    too_many = make_request(images=tuple(f"{i:064x}"
                                         for i in range(ep.REFERENCE_MAX_FILES + 1)))
    with pytest.raises(EncoderRequestError, match="images"):
        ep.validate_production_request(too_many)


def test_implementation_identity_mandatory():
    with pytest.raises(EncoderRequestError, match="implementation_identity"):
        ep.validate_production_request(make_request(implementation_identity=None))


def test_non_request_object_refused():
    with pytest.raises(EncoderRequestError, match="EncoderRequest"):
        ep.validate_production_request({"prompt": "dict is not a request"})


def test_validation_happens_before_heavy_imports(monkeypatch):
    calls = []

    def boom():
        calls.append("torch")
        raise AssertionError("heavy import must not happen")

    monkeypatch.setattr(ep, "_load_torch", boom)
    req = make_request(dtype="fp64")
    with pytest.raises(EncoderRequestError):
        ep.encode_with_supplied_pipeline(FakePipeline({}), req)
    assert calls == []


# ---------------------------------------------------------------------------
# encode_with_supplied_pipeline — supplied-model reuse contract
# ---------------------------------------------------------------------------


def test_upstream_called_cpu_one_image_inference_mode(monkeypatch):
    torch, _st = install_ml(monkeypatch)
    pipeline = FakePipeline({"a red cube on a table": triple(),
                             "ugly, blurry": triple(shape=(1, 3, 8))})
    ep.encode_with_supplied_pipeline(pipeline, prod_request())
    assert len(pipeline.calls) == 2
    for call in pipeline.calls:
        assert call["image"] is None
        assert call["device"] == ("cpu-device", "cpu")
        assert call["num_images_per_prompt"] == 1
    assert torch.devices == ["cpu"]
    assert torch.im_enter == 1 and torch.im_exit == 1


def test_pipeline_without_encode_prompt_is_setup_error(monkeypatch):
    install_ml(monkeypatch)

    class NoEncode:
        pass

    with pytest.raises(EncoderSetupError, match="encode_prompt"):
        ep.encode_with_supplied_pipeline(NoEncode(), prod_request())


def test_upstream_runtime_failure_is_typed_worker_error(monkeypatch):
    torch, _st = install_ml(monkeypatch)
    pipeline = FakePipeline({}, raises=RuntimeError("CUDA out of memory"))
    with pytest.raises(EncoderWorkerError, match="CUDA out of memory"):
        ep.encode_with_supplied_pipeline(pipeline, prod_request())
    assert torch.im_exit == 1  # inference_mode exited even on failure


def test_keyboardinterrupt_is_never_wrapped(monkeypatch):
    install_ml(monkeypatch)
    pipeline = FakePipeline({}, raises=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        ep.encode_with_supplied_pipeline(pipeline, prod_request())


@pytest.mark.parametrize("dtype,code", [("fp16", "F16"), ("bf16", "BF16"),
                                        ("fp32", "F32")])
def test_roundtrip_all_dtypes_and_cfg_branch_preserves_values(dtype, code,
                                                              monkeypatch):
    _torch, st = install_ml(monkeypatch)
    pos = triple(shape=(1, 5, 8), code=code)
    neg = triple(shape=(1, 3, 8), code=code)  # unequal S is allowed
    pipeline = FakePipeline({"a red cube on a table": pos,
                             "ugly, blurry": neg})
    req = prod_request(dtype=dtype)
    payload = ep.encode_with_supplied_pipeline(pipeline, req)
    assert [b.name for b in payload.blobs] == [ep.TENSOR_BLOB_NAME]
    bundle = ep.payload_to_branches(payload, expect_request=req)
    assert bundle.negative is not None
    for branch, want in ((bundle.positive, pos), (bundle.negative, neg)):
        assert branch.embeds.raw_bytes() == want[0].raw_bytes()
        assert branch.attention_mask.raw_bytes() == want[1].raw_bytes()
        assert branch.image_pad_mask.raw_bytes() == want[2].raw_bytes()
        assert len(st.save_calls) == 1


def test_payload_identity_metadata_exact(monkeypatch):
    install_ml(monkeypatch)
    req = prod_request(dtype="fp32")
    payload = ep.encode_with_supplied_pipeline(
        FakePipeline({"a red cube on a table": triple(code="F32"),
                      "ugly, blurry": triple(code="F32")}), req)
    md = payload.metadata
    assert set(md) == set(ep.PAYLOAD_METADATA_KEYS)
    assert md["schema"] == ep.TENSOR_PAYLOAD_SCHEMA
    assert md["codec"] == ep.SAFETENSORS_CODEC
    assert md["cache_key"] == cache_key(req)
    assert md["dtype"] == "fp32"
    assert md["cfg_active"] is True
    from qwen_engine._jsonsafe import plain_json
    assert plain_json(md["request"]) == req.key_document()
    names = {t["name"] for t in md["tensors"]}
    assert names == {"positive.embeds", "positive.attention_mask",
                     "positive.image_pad_mask", "negative.embeds",
                     "negative.attention_mask", "negative.image_pad_mask"}
    header, _body = st_split(payload.blobs[0].data)
    st_md = header["__metadata__"]
    assert set(st_md) == set(ep.ST_METADATA_KEYS)
    assert json.loads(st_md["request_identity"]) == req.key_document()


def test_nullable_attention_mask_is_absent_tensor_and_stays_null(monkeypatch):
    install_ml(monkeypatch)
    pipeline = FakePipeline({"a red cube on a table": triple(mask="none")})
    payload = ep.encode_with_supplied_pipeline(pipeline, make_request())
    names = {t["name"] for t in payload.metadata["tensors"]}
    assert "positive.attention_mask" not in names
    bundle = ep.payload_to_branches(payload, expect_request=make_request())
    assert bundle.positive.attention_mask is None  # null preserved
    assert bundle.negative is None


@pytest.mark.parametrize("mutation,match", [
    (lambda t: (t[0].__class__("F32", t[0].shape, [1.0] * 32), t[1], t[2]),
     "dtype"),                                    # silent cast attempt
    (lambda t: (FakeTensor("F16", (1, 4), [0.5] * 4), t[1], t[2]),
     r"\[1, S>=1, H>=1\]"),                       # 2-D embeds
    (lambda t: (FakeTensor("F16", (2, 4, 8), [0.5] * 64),
                mask_tensor([1] * 4), image_mask_tensor((2, 4))),
     r"\[1, S>=1, H>=1\]"),                       # batch 2
    (lambda t: (FakeTensor("F16", (1, 0, 8), []), mask_tensor([]),
                image_mask_tensor((1, 0))), r"\[1, S>=1, H>=1\]"),
    (lambda t: (FakeTensor("BOOL", (1, 4, 8), [True] * 32), t[1], t[2]),
     "floating-point"),
    (lambda t: (embeds_tensor(nan=True), t[1], t[2]), "non-finite"),
    (lambda t: (t[0], mask_tensor([1, 1, 1, 2]), t[2]), "binary"),
    (lambda t: (t[0], mask_tensor([0, 0, 0, 0]), t[2]), "no valid token"),
    (lambda t: (t[0], FakeTensor("F32", (1, 4), [1.0] * 4), t[2]), "int64"),
    (lambda t: (t[0], mask_tensor([1] * 5), t[2]), r"shape \[1, 4\]"),
    (lambda t: (t[0], t[1], None), "mandatory"),
    (lambda t: (t[0], t[1], mask_tensor([1] * 4)), "bool"),
    (lambda t: (t[0], t[1], image_mask_tensor(any_true=True)), "all-False"),
    (lambda t: (t[0], t[1], t[2], "extra"), "must return"),
])
def test_malformed_upstream_output_is_protocol_error(monkeypatch, mutation,
                                                     match):
    install_ml(monkeypatch)
    base = triple()
    pipeline = FakePipeline({"a red cube on a table": mutation(base)})
    with pytest.raises(EncoderProtocolError, match=match):
        ep.encode_with_supplied_pipeline(pipeline, make_request())


def test_branch_geometry_agreement(monkeypatch):
    install_ml(monkeypatch)
    # unequal S is fine
    pipeline = FakePipeline({"a red cube on a table": triple(shape=(1, 9, 8)),
                             "ugly, blurry": triple(shape=(1, 2, 8))})
    payload = ep.encode_with_supplied_pipeline(pipeline, prod_request())
    bundle = ep.payload_to_branches(payload, expect_request=prod_request())
    assert bundle.positive.embeds.shape == (1, 9, 8)
    assert bundle.negative.embeds.shape == (1, 2, 8)
    # H mismatch is refused
    pipeline = FakePipeline({"a red cube on a table": triple(shape=(1, 4, 8)),
                             "ugly, blurry": triple(shape=(1, 4, 16))})
    with pytest.raises(EncoderProtocolError, match="hidden"):
        ep.encode_with_supplied_pipeline(pipeline, prod_request())


# ---------------------------------------------------------------------------
# Codec — stdlib bounds BEFORE any library deserialization
# ---------------------------------------------------------------------------


def _valid_blob_dict():
    return {"positive.embeds": embeds_tensor(),
            "positive.attention_mask": mask_tensor([1, 1, 1, 1]),
            "positive.image_pad_mask": image_mask_tensor()}


def test_inspect_accepts_wellformed_bytes():
    tensors, metadata, data_len = ep._inspect_safetensors_bytes(
        st_bytes(_valid_blob_dict(), {"codec": "x"}))
    assert set(tensors) == set(_valid_blob_dict())
    assert metadata == {"codec": "x"}
    assert data_len == 32 * 2 + 4 * 8 + 4 * 1


def _reject(data, match=""):
    with pytest.raises(EncoderProtocolError, match=match):
        ep._inspect_safetensors_bytes(data)


def test_inspect_rejects_short_and_oversized_bytes():
    _reject(b"", "length field")
    _reject(b"\x01" * 7, "length field")
    _reject(b"\xff" * 8 + b"{}", "cap")
    big = struct.pack("<Q", ep.MAX_TENSOR_BLOB_BYTES + 1) + b"{}"
    _reject(big[:9], "cap")


def test_inspect_rejects_bad_header_length():
    _reject(struct.pack("<Q", 0) + b"{}" + b"\x00" * 8, "zero")
    _reject(struct.pack("<Q", 2 ** 62) + b"{}", "cap")
    _reject(struct.pack("<Q", 100) + b"{}" + b"\x00" * 4, "only")


def test_inspect_rejects_bad_header_json():
    _reject(st_bytes(_valid_blob_dict(),
                     header_text="{not json}"), "strict JSON")
    _reject(st_bytes(_valid_blob_dict(),
                     header_text='{"a": NaN}'), "non-finite")
    _reject(st_bytes(_valid_blob_dict(),
                     header_text='{"a": 1, "a": 2}'), "repeats the key")
    _reject(st_bytes(_valid_blob_dict(),
                     header_text='[1, 2]'), "JSON object")
    _reject(struct.pack("<Q", 4) + b"\xff\xff\xff\xff" + b"\x00" * 8,
            "UTF-8")


def test_inspect_bounds_nesting_and_count():
    deep = {"positive.embeds": {"dtype": "F16", "shape": [1, 4, 8],
                                "data_offsets": [0, 64]}}
    node: list = []
    for _ in range(ep.MAX_HEADER_JSON_DEPTH + 3):
        node = [node]
    deep["__metadata__"] = {"blob": node}
    _reject(st_bytes({}, header_text=json.dumps(deep),
                     data=b"\x00" * 64), "nests deeper")
    many = {f"t{i:03d}": {"dtype": "BOOL", "shape": [0],
                          "data_offsets": [0, 0]}
            for i in range(ep.MAX_SAFETENSORS_TENSORS + 1)}
    _reject(st_bytes({}, header_text=json.dumps(many), data=b""), "tensors")
    valid = st_bytes({}, header_text=json.dumps({"positive.embeds": {
        "dtype": "F16", "shape": [1, 4, 8], "data_offsets": [0, 64]}}),
        data=b"\x00" * 64)
    assert ep._inspect_safetensors_bytes(valid)[2] == 64


def test_inspect_rejects_malformed_entries():
    good = {"dtype": "F16", "shape": [1, 4, 8], "data_offsets": [0, 64]}

    def bad(entry):
        h = {"positive.embeds": entry,
             "positive.image_pad_mask": {"dtype": "BOOL", "shape": [1, 4],
                                         "data_offsets": [64, 68]}}
        return st_bytes({}, header_text=json.dumps(h), data=b"\x00" * 68)

    _reject(bad({**good, "dtype": "F64"}), "dtype code")
    _reject(bad({**good, "shape": [1, True, 8]}), "true non-negative")
    _reject(bad({**good, "shape": [1, 4.0, 8]}), "true non-negative")
    _reject(bad({**good, "shape": [1, -4, 8]}), "true non-negative")
    _reject(bad({**good, "shape": "[1, 4, 8]"}), "shape")
    _reject(bad({**good, "data_offsets": [0, True]}), "true integers")
    _reject(bad({**good, "data_offsets": [0, 64, 70]}), "pair")
    _reject(bad({**good, "data_offsets": [0, 70]}), "outside")
    _reject(bad({**good, "data_offsets": [70, 80]}), "outside")
    _reject(bad({**good, "extra": 1}), "exactly")
    _reject(bad({"dtype": "F16", "shape": [1, 5, 8],
                 "data_offsets": [0, 64]}), "does not account")
    _reject(bad({"dtype": "F32", "shape": [1, 4, 8],
                 "data_offsets": [0, 64]}), "bytes per element")


def test_inspect_rejects_bad_spans():
    h = {"a": {"dtype": "BOOL", "shape": [4], "data_offsets": [0, 4]},
         "b": {"dtype": "BOOL", "shape": [4], "data_offsets": [2, 6]}}
    _reject(st_bytes({}, header_text=json.dumps(h), data=b"\x00" * 6),
            "overlap")
    h = {"a": {"dtype": "BOOL", "shape": [4], "data_offsets": [0, 4]},
         "b": {"dtype": "BOOL", "shape": [4], "data_offsets": [5, 9]}}
    _reject(st_bytes({}, header_text=json.dumps(h), data=b"\x00" * 9),
            "cover")
    _reject(st_bytes({}, header_text=json.dumps({"a": {
        "dtype": "BOOL", "shape": [4], "data_offsets": [0, 4]}}),
        data=b"\x00" * 5), "cover")


# ---------------------------------------------------------------------------
# Codec — payload-level validation (fake library only AFTER the bounds)
# ---------------------------------------------------------------------------


def encode_payload(monkeypatch, req=None, pos=None, neg=None):
    req = req or make_request()
    pos = pos if pos is not None else triple()
    by = {req.prompt: pos}
    if req.kwargs.get("cfg_active") is True:
        by[req.negative_prompt] = neg if neg is not None else triple()
    install_ml(monkeypatch)
    return ep.encode_with_supplied_pipeline(FakePipeline(by), req), req


def rewrapped(payload, *, data=None, metadata=None):
    blob = Blob(name=payload.blobs[0].name,
                data=payload.blobs[0].data if data is None else data)
    return EncodedPayload(metadata=(dict(payload.metadata) if metadata is None
                                    else metadata),
                          blobs=(blob,))


def test_roundtrip_and_expect_request_identity(monkeypatch):
    payload, req = encode_payload(monkeypatch)
    bundle = ep.payload_to_branches(payload, expect_request=req)
    assert bundle.positive.embeds.shape == (1, 4, 8)
    other = make_request(prompt="a DIFFERENT cube")
    with pytest.raises(EncoderProtocolError, match="cache key"):
        ep.payload_to_branches(payload, expect_request=other)
    with pytest.raises(EncoderRequestError, match="expect_request"):
        ep.payload_to_branches(payload, expect_request={"not": "a request"})


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d[:-1], "outside|cover"),               # truncated data
    (lambda d: d + b"\x00", "cover"),                  # trailing byte
    (lambda d: d[:8] + b"{" * int.from_bytes(d[:8], "little") + d[8:],
     "strict JSON|nests deeper"),                       # header byte-shift
])
def test_tampered_blob_bytes_are_protocol_error(monkeypatch, mutate, match):
    payload, req = encode_payload(monkeypatch)
    with pytest.raises(EncoderProtocolError, match=match):
        ep.payload_to_branches(rewrapped(payload, data=mutate(
            payload.blobs[0].data)), expect_request=req)


def test_space_padded_header_is_accepted(monkeypatch):
    # safetensors pads the header with spaces for 8-byte alignment; the
    # bounds reader must still validate framing (the padding stays inside
    # the declared header length, so the data region is unchanged).
    payload, req = encode_payload(monkeypatch)
    data = payload.blobs[0].data
    n = int.from_bytes(data[:8], "little")
    pad = b" " * 8
    padded = struct.pack("<Q", n + 8) + data[8:8 + n] + pad + data[8 + n:]
    bundle = ep.payload_to_branches(rewrapped(payload, data=padded),
                                    expect_request=req)
    assert bundle.positive.embeds.shape == (1, 4, 8)


def test_library_errors_wrapped_but_baseexception_propagates(monkeypatch):
    payload, req = encode_payload(monkeypatch)
    _torch, st = install_ml(monkeypatch)
    st.load_error = ValueError("corrupt safetensors stream")
    with pytest.raises(EncoderProtocolError, match="deserialization failed"):
        ep.payload_to_branches(payload, expect_request=req)
    st.load_error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        ep.payload_to_branches(payload, expect_request=req)


def test_library_not_touched_before_bounds(monkeypatch):
    payload, _req = encode_payload(monkeypatch)
    _torch, st = install_ml(monkeypatch)
    st.load_calls = 0
    with pytest.raises(EncoderProtocolError):
        ep.payload_to_branches(rewrapped(payload, data=payload.blobs[0].data[:-1]))
    assert st.load_calls == 0  # refusal happened pre-deserialization


def test_payload_metadata_violations(monkeypatch):
    payload, req = encode_payload(monkeypatch)
    md = dict(payload.metadata)
    cases = []
    md2 = dict(md); md2.pop("codec"); cases.append((md2, "metadata keys"))
    md2 = dict(md); md2["schema"] = "other"; cases.append((md2, "schema"))
    md2 = dict(md); md2["codec"] = "other"; cases.append((md2, "codec"))
    md2 = dict(md); md2["cache_key"] = "z" * 64; cases.append((md2, "sha256"))
    md2 = dict(md); md2["dtype"] = "fp64"; cases.append((md2, "dtype"))
    md2 = dict(md); md2["cfg_active"] = "true"; cases.append((md2, "bool"))
    md2 = dict(md); md2["request"] = "nope"; cases.append((md2, "request"))
    for md_bad, match in cases:
        with pytest.raises(EncoderProtocolError, match=match):
            ep.payload_to_branches(rewrapped(payload, metadata=md_bad))


def test_metadata_silently_flipped_to_active_is_refused(monkeypatch):
    # payload metadata says cfg active AND the safetensors metadata agrees,
    # but no negative tensors exist -> the name plan refuses it
    payload, _req = encode_payload(monkeypatch)
    md = dict(payload.metadata)
    md["cfg_active"] = True
    header, body = st_split(payload.blobs[0].data)
    header["__metadata__"]["cfg_active"] = "true"
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    data = struct.pack("<Q", len(text)) + text + body
    with pytest.raises(EncoderProtocolError, match="cfg_active"):
        ep.payload_to_branches(rewrapped(payload, data=data, metadata=md))


def test_st_metadata_crosscheck(monkeypatch):
    payload, _req = encode_payload(monkeypatch)
    header, body = st_split(payload.blobs[0].data)
    for key, value, match in [
        ("codec", "other.tag", "codec"),
        ("dtype", "fp64", "dtype"),
        ("cache_key", "f" * 64, "cache key"),
    ]:
        h = dict(header)
        h["__metadata__"] = dict(header["__metadata__"])
        h["__metadata__"][key] = value
        text = json.dumps(h, sort_keys=True, separators=(",", ":")).encode()
        with pytest.raises(EncoderProtocolError, match=match):
            ep.payload_to_branches(rewrapped(
                payload, data=struct.pack("<Q", len(text)) + text + body))
    h = dict(header)
    h["__metadata__"] = {"codec": ep.SAFETENSORS_CODEC}
    text = json.dumps(h, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(EncoderProtocolError, match="metadata keys"):
        ep.payload_to_branches(rewrapped(
            payload, data=struct.pack("<Q", len(text)) + text + body))


def test_declared_vs_actual_tensor_mismatch(monkeypatch):
    # descriptor says F16, deserialized tensor is F32 (silent-cast attempt):
    # caught by the header/descriptor byte arithmetic before or at load
    payload, _req = encode_payload(monkeypatch)
    header, body = st_split(payload.blobs[0].data)
    h = dict(header)
    h["positive.embeds"] = dict(header["positive.embeds"], dtype="F32")
    text = json.dumps(h, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(EncoderProtocolError, match="does not account|dtype"):
        ep.payload_to_branches(rewrapped(
            payload, data=struct.pack("<Q", len(text)) + text + body))


def test_deserialized_tensor_name_surprise(monkeypatch):
    payload, _req = encode_payload(monkeypatch)
    _torch, st = install_ml(monkeypatch)
    real_load = st.load

    def load(data):
        out = real_load(data)
        out["positive.embeds"] = FakeTensor("F16", (1, 6, 8),
                                            [0.5] * 48)  # shape surprise
        return out

    st.load = load
    with pytest.raises(EncoderProtocolError, match="shape"):
        ep.payload_to_branches(payload)


def test_decode_semantic_mask_and_value_violations(monkeypatch):
    # NaN embeddings / non-binary mask / zero-valid mask / True image mask
    # that SURVIVED a (malformed) serialization must be caught on decode
    # by tampering valid bytes; production serialization also checks values.
    valid, _req = encode_payload(monkeypatch)
    header, _ = st_split(valid.blobs[0].data)

    def serialize(embeds, mask, image):
        data = st_bytes({"positive.embeds": embeds,
                         "positive.attention_mask": mask,
                         "positive.image_pad_mask": image},
                        metadata=header["__metadata__"])
        return rewrapped(valid, data=data)

    with pytest.raises(EncoderProtocolError, match="binary"):
        ep.payload_to_branches(serialize(embeds_tensor(),
                                         mask_tensor([1, 1, 1, 2]),
                                         image_mask_tensor()))
    with pytest.raises(EncoderProtocolError, match="no valid token"):
        ep.payload_to_branches(serialize(embeds_tensor(),
                                         mask_tensor([0, 0, 0, 0]),
                                         image_mask_tensor()))
    with pytest.raises(EncoderProtocolError, match="all-False"):
        ep.payload_to_branches(serialize(embeds_tensor(),
                                         mask_tensor([1, 1, 1, 1]),
                                         image_mask_tensor(any_true=True)))
    with pytest.raises(EncoderProtocolError, match="non-finite"):
        ep.payload_to_branches(serialize(embeds_tensor(nan=True),
                                         mask_tensor([1, 1, 1, 1]),
                                         image_mask_tensor()))


def test_negative_tensors_with_inactive_cfg_refused(monkeypatch):
    install_ml(monkeypatch)
    req = make_request()  # cfg inactive
    bundle = ep.BranchBundle(
        positive=ep.EncoderBranch(*triple()),
        negative=ep.EncoderBranch(*triple()))
    validated = ep.ValidatedProductionRequest(req, False)
    with pytest.raises(EncoderProtocolError, match="negative"):
        ep.serialize_branches(validated, bundle)


def test_active_cfg_payload_missing_negative_refused(monkeypatch):
    install_ml(monkeypatch)
    req = prod_request()
    bundle = ep.BranchBundle(positive=ep.EncoderBranch(*triple()))
    with pytest.raises(EncoderProtocolError, match="negative"):
        ep.serialize_branches(ep.ValidatedProductionRequest(req, True), bundle)


def test_blob_shape_contract(monkeypatch):
    payload, _req = encode_payload(monkeypatch)
    with pytest.raises(EncoderProtocolError, match="EncodedPayload"):
        ep.payload_to_branches({"not": "a payload"})
    with pytest.raises(EncoderProtocolError, match="one blob"):
        ep.payload_to_branches(EncodedPayload(
            metadata=payload.metadata,
            blobs=payload.blobs + (Blob("extra", b"x"),)))
    blob = payload.blobs[0]
    with pytest.raises(EncoderProtocolError, match="named"):
        ep.payload_to_branches(EncodedPayload(
            metadata=payload.metadata,
            blobs=(Blob("wrong_name", blob.data),)))


def test_unknown_tensor_name_refused(monkeypatch):
    encode_payload(monkeypatch)
    tensors = _valid_blob_dict()
    tensors["extra.weight"] = FakeTensor("F16", (1, 1, 1), [1.0])
    data = st_bytes(tensors, metadata={})
    # the byte framing itself is well-formed...
    header, _md, _n = ep._inspect_safetensors_bytes(data)
    assert "extra.weight" in header
    # ...but the name plan refuses it, and a payload whose descriptors carry
    # it is refused on decode.
    with pytest.raises(EncoderProtocolError, match="outside the codec"):
        ep._expected_names(False, set(header))


# ---------------------------------------------------------------------------
# Pinned dual-ML-interpreter synthetic pass (explicit skip when absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ML_PYTHONS or os.environ.get("QWEN_CPU_TEST_GUARD") == "1", reason=(
    "ML checks are separate from the guarded ML-free suite and require "
    "QWEN_ML_PYTHONS — this skip is NOT a validation claim"))
def test_synthetic_ml_encoder_check_on_pinned_interpreters():
    script = MODULE_SOURCE.parents[2] / "tools" / "synthetic_ml_encoder_check.py"
    for python in ML_PYTHONS:
        proc = subprocess.run([python, "-B", str(script)], capture_output=True,
                              text=True, timeout=600)
        assert proc.returncode == 0, f"{python}: {proc.stdout}\n{proc.stderr}"


def test_supplied_model_uses_upstream_pipeline_constructor(monkeypatch):
    torch, st = install_ml(monkeypatch)
    req = make_request()
    constructors = []

    class Model:
        def parameters(self):
            return iter(())

        def buffers(self):
            return iter(())

    torch.nn = SimpleNamespace(Module=Model)

    def constructor(**kwargs):
        constructors.append(kwargs)
        return FakePipeline({req.prompt: triple()})

    monkeypatch.setattr(ep, "_load_pipeline_class", lambda: constructor)
    model, processor = Model(), lambda **kwargs: None
    ep.encode_with_supplied_model(model, processor, req)
    assert constructors == [dict(scheduler=None, vae=None, transformer=None,
                                 text_encoder=model, processor=processor)]
    assert len(st.save_calls) == 1


def test_supplied_model_request_validation_precedes_imports(monkeypatch):
    monkeypatch.setattr(ep, "_load_torch", lambda: pytest.fail("heavy import"))
    with pytest.raises(EncoderRequestError):
        ep.encode_with_supplied_model(None, None, make_request(dtype="fp64"))


@pytest.mark.parametrize("invalid", ["model", "processor", "parameter", "buffer"])
def test_supplied_model_setup_refuses_invalid_components_before_pipeline(monkeypatch, invalid):
    torch, _ = install_ml(monkeypatch)

    class Model:
        def parameters(self):
            return iter([SimpleNamespace(device=SimpleNamespace(type="cuda"))]
                        if invalid == "parameter" else [])

        def buffers(self):
            return iter([SimpleNamespace(device=SimpleNamespace(type="meta"))]
                        if invalid == "buffer" else [])

    torch.nn = SimpleNamespace(Module=Model)
    monkeypatch.setattr(ep, "_load_pipeline_class", lambda: pytest.fail("pipeline import"))
    with pytest.raises(EncoderSetupError):
        ep.encode_with_supplied_model(
            object() if invalid == "model" else Model(),
            object() if invalid == "processor" else lambda **kwargs: None,
            make_request())


@pytest.mark.parametrize("change", ["device", "layout", "contiguous", "type"])
def test_outputs_must_be_dense_contiguous_cpu_tensors(monkeypatch, change):
    install_ml(monkeypatch)
    req = make_request()
    output = list(triple())
    if change == "device":
        output[0].device = SimpleNamespace(type="cuda")
    elif change == "layout":
        output[0].layout = "sparse"
    elif change == "contiguous":
        output[0].is_contiguous = lambda: False
    else:
        output[0] = SimpleNamespace(shape=(1, 4, 8), dtype="torch.float16")
    with pytest.raises(EncoderProtocolError):
        ep.encode_with_supplied_pipeline(FakePipeline({req.prompt: output}), req)


def test_serialization_bounds_and_values_precede_library(monkeypatch):
    _, st = install_ml(monkeypatch)
    req = make_request()
    validated = ep.validate_production_request(req)
    for output in (triple(nan=True), triple(mask="twos"), triple(mask="zeros"),
                   triple(img_true=True)):
        with pytest.raises(EncoderProtocolError):
            ep.serialize_branches(validated, ep.BranchBundle(ep.EncoderBranch(*output)))
    assert not st.save_calls
    giant = FakeTensor("F16", (1, 2 ** 200, 8), raw=b"")
    with pytest.raises(EncoderProtocolError, match="cap"):
        ep.serialize_branches(validated, ep.BranchBundle(
            ep.EncoderBranch(giant, None, image_mask_tensor())))
    monkeypatch.setattr(ep, "MAX_TENSOR_BLOB_BYTES", 64)
    with pytest.raises(EncoderProtocolError, match="cap"):
        ep.serialize_branches(validated, ep.BranchBundle(ep.EncoderBranch(*triple())))
    assert not st.save_calls


def test_forged_validated_policy_cannot_bypass_request(monkeypatch):
    _, st = install_ml(monkeypatch)
    with pytest.raises(EncoderRequestError, match="cfg_active"):
        ep.serialize_branches(ep.ValidatedProductionRequest(make_request(), True),
                              ep.BranchBundle(ep.EncoderBranch(*triple())))
    assert not st.save_calls


def test_payload_identity_validated_even_without_expected_request(monkeypatch):
    payload, _ = encode_payload(monkeypatch)
    _, st = install_ml(monkeypatch)
    md = dict(payload.metadata)
    md["request"] = {}
    with pytest.raises(EncoderProtocolError, match="request identity"):
        ep.payload_to_branches(rewrapped(payload, metadata=md))
    md = dict(payload.metadata)
    md["cache_key"] = "f" * 64
    with pytest.raises(EncoderProtocolError, match="cache key"):
        ep.payload_to_branches(rewrapped(payload, metadata=md))
    assert st.load_calls == 0


# ---------------------------------------------------------------------------
# Image conditioning (TI2I): resize wiring, mask policy, payload round-trip
# ---------------------------------------------------------------------------


class PilStub:
    """Stand-in for PIL's Image class at the lazy Pillow boundary."""

    def __init__(self, size=(200, 100), mode="RGB", load_error=None):
        self.size = size
        self.mode = mode
        self.load_error = load_error
        self.loaded = False

    def load(self):
        if self.load_error is not None:
            raise self.load_error
        self.loaded = True

    def convert(self, mode):
        return PilStub(self.size, mode, load_error=self.load_error)


def pil_module(image=None, open_error=None):
    def open(source):
        if open_error is not None:
            raise open_error
        return image
    return SimpleNamespace(Image=PilStub, open=open, LANCZOS=1)


def pinned_stub(log):
    def calculate_dimensions(area, ratio):
        log.append(("dimensions", area, ratio))
        return (1440, 704, None)

    class Processor:
        def resize(self, image, *, width, height):
            log.append(("resize", width, height, image.mode))
            return f"resized-{image.mode}"

    return calculate_dimensions, Processor()


@pytest.fixture
def stub_pil_and_pinned(monkeypatch):
    log = []
    monkeypatch.setattr(ep, "_load_pil_image", lambda: pil_module(PilStub()))
    monkeypatch.setattr(ep, "_pinned_reference_helpers", lambda: pinned_stub(log))
    return log


def test_reference_resize_follows_pinned_pipeline_rules(monkeypatch):
    log = []
    image = PilStub(size=(2000, 1000), mode="RGB")
    monkeypatch.setattr(ep, "_load_pil_image", lambda: pil_module(image))
    monkeypatch.setattr(ep, "_pinned_reference_helpers", lambda: pinned_stub(log))
    refs = ep.reference_images_from_blobs(make_request(images=(SHA_A,)), [b"fake"])
    assert refs == ["resized-RGBA"]  # RGBA-converted before the pinned resize
    assert image.loaded              # full decode happened for accepted sizes
    assert log == [("dimensions", ep.REFERENCE_TARGET_AREA, 2.0),
                   ("resize", 1440, 704, "RGBA")]


@pytest.mark.parametrize("size", [(5000, 10), (10, 5000), (4096, 4097)])
def test_reference_dimension_caps_refuse_before_full_load(monkeypatch, size):
    image = PilStub(size=size)
    monkeypatch.setattr(ep, "_load_pil_image", lambda: pil_module(image))
    monkeypatch.setattr(ep, "_pinned_reference_helpers", lambda: pinned_stub([]))
    with pytest.raises(EncoderRequestError, match="dimensions"):
        ep.reference_images_from_blobs(make_request(images=(SHA_A,)), [b"x"])
    assert not image.loaded


def test_reference_decode_and_count_failures_are_typed(monkeypatch):
    monkeypatch.setattr(ep, "_pinned_reference_helpers", lambda: pinned_stub([]))
    monkeypatch.setattr(ep, "_load_pil_image",
                        lambda: pil_module(open_error=OSError("truncated")))
    with pytest.raises(EncoderRequestError, match="cannot be opened"):
        ep.reference_images_from_blobs(make_request(images=(SHA_A,)), [b"x"])
    monkeypatch.setattr(ep, "_load_pil_image",
                        lambda: pil_module(PilStub(load_error=OSError("boom"))))
    with pytest.raises(EncoderRequestError, match="cannot be decoded"):
        ep.reference_images_from_blobs(make_request(images=(SHA_A,)), [b"x"])
    monkeypatch.setattr(ep, "_load_pil_image", lambda: pil_module(PilStub()))
    with pytest.raises(EncoderRequestError, match="ordered image hashes"):
        ep.reference_images_from_blobs(make_request(images=(SHA_A, SHA_B)), [b"x"])
    with pytest.raises(EncoderRequestError, match="text-only"):
        ep.reference_images_from_blobs(make_request(), [b"x"])


def test_image_branches_pass_resized_pil_to_upstream_both_sides(monkeypatch):
    install_ml(monkeypatch)
    monkeypatch.setattr(ep, "_load_pil_image", lambda: pil_module())
    refs = [PilStub(mode="RGBA"), PilStub(mode="RGBA")]
    pipeline = FakePipeline({"a red cube on a table": triple(img_true=True),
                             "ugly, blurry": triple(shape=(1, 3, 8), img_true=True)})
    req = prod_request(images=(SHA_A, SHA_B))
    payload = ep.encode_with_supplied_pipeline(pipeline, req, refs)
    assert [call["image"] for call in pipeline.calls] == [refs, refs]
    bundle = ep.payload_to_branches(payload, expect_request=req)
    assert any(bundle.positive.image_pad_mask.tolist())
    assert any(bundle.negative.image_pad_mask.tolist())


def test_image_branch_reference_image_argument_contract(monkeypatch):
    install_ml(monkeypatch)
    monkeypatch.setattr(ep, "_load_pil_image", lambda: pil_module())
    pipeline = FakePipeline({}, raises=AssertionError("must not encode"))
    req = make_request(images=(SHA_A,))
    with pytest.raises(EncoderRequestError, match="resized reference"):
        ep.encode_with_supplied_pipeline(pipeline, req)
    with pytest.raises(EncoderRequestError, match="ordered image hashes"):
        ep.encode_with_supplied_pipeline(
            pipeline, req, [PilStub(mode="RGBA"), PilStub(mode="RGBA")])
    with pytest.raises(EncoderRequestError, match="PIL Image"):
        ep.encode_with_supplied_pipeline(pipeline, req, ["not-an-image"])
    with pytest.raises(EncoderRequestError, match="text-only request"):
        ep.encode_with_supplied_pipeline(pipeline, make_request(),
                                         [PilStub(mode="RGBA")])


def test_mask_and_conditioning_must_agree_in_both_directions(monkeypatch):
    install_ml(monkeypatch)
    monkeypatch.setattr(ep, "_load_pil_image", lambda: pil_module())
    req = make_request(images=(SHA_A,))
    with pytest.raises(EncoderProtocolError, match="vision token"):
        ep.encode_with_supplied_pipeline(
            FakePipeline({req.prompt: triple(img_true=False)}), req,
            [PilStub(mode="RGBA")])
    t2i = make_request()
    with pytest.raises(EncoderProtocolError, match="all-False"):
        ep.encode_with_supplied_pipeline(
            FakePipeline({t2i.prompt: triple(img_true=True)}), t2i)


def test_image_payload_flattened_mask_is_refused_on_decode(monkeypatch):
    install_ml(monkeypatch)
    monkeypatch.setattr(ep, "_load_pil_image", lambda: pil_module())
    req = make_request(images=(SHA_A,))
    payload = ep.encode_with_supplied_pipeline(
        FakePipeline({req.prompt: triple(img_true=True)}), req,
        [PilStub(mode="RGBA")])
    header, _ = st_split(payload.blobs[0].data)
    tampered = st_bytes({"positive.embeds": embeds_tensor(),
                         "positive.attention_mask": mask_tensor([1, 1, 1, 1]),
                         "positive.image_pad_mask": image_mask_tensor()},
                        metadata=header["__metadata__"])
    with pytest.raises(EncoderProtocolError, match="vision token"):
        ep.payload_to_branches(rewrapped(payload, data=tampered),
                               expect_request=req)
