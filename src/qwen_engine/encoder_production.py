"""Production encoder helper + bounded safetensors tensor codec.

Scoped by ``docs/handoffs/2026-10-12-encoder-helper-next.md``: lazy reuse of
the pinned ``QwenImage21Pipeline.encode_prompt`` with supplied components (no loader, no
worker CLI, no request factory, no package validator, no adapter wiring —
those are later milestones) plus one exact, bounded safetensors payload
format. The generic transport/cache module :mod:`qwen_engine.encoder` stays
stdlib-only and untouched; this module is stdlib-only **at import time** —
``torch`` / ``safetensors`` are imported lazily inside production calls only.

Contract summary
----------------
* :func:`validate_production_request` runs **before any heavy import** and
  refuses (typed :class:`EncoderRequestError`): a non-scalar or blank
  positive prompt; any images; an absent implementation identity (production
  requires the complete :class:`qwen_engine.encoder.EncoderImplementationIdentity`
  — the record's own validation already proves completeness when present); a
  dtype that is not exactly one of ``fp16`` / ``bf16`` / ``fp32``; and
  ``kwargs`` that are not exactly ``{"cfg_active": bool}``. ``cfg_active=True``
  requires a non-blank negative prompt; ``cfg_active=False`` **never encodes**
  the negative but deliberately keeps its text in the request/cache identity.

* :func:`encode_with_supplied_model` constructs the pinned upstream pipeline
  with the supplied CPU model and processor; scheduler, VAE and transformer
  are None. It never loads weights or moves/casts a component.
  :func:`encode_with_supplied_pipeline` takes an already-constructed pipeline
  (the caller supplies the pinned ``QwenImage21Pipeline`` with only the text
  encoder and processor — this helper never loads, constructs, moves or
  casts anything) and calls its ``encode_prompt`` on CPU, with ``image=None``
  and ``num_images_per_prompt=1``, inside ``torch.inference_mode()``. It does
  not copy prompt control tokens, re-create them, call
  ``apply_chat_template`` / the tokenizer directly, or register any hook —
  the upstream method owns prompt formatting, the drop-index and the
  pre-RMSNorm hook (docs/interfaces.md §2). Upstream output dtypes are
  **checked, never coerced**.

* Branch outputs: floating **finite** embeddings ``[1, S>=1, H>=1]`` whose
  dtype equals the requested dtype exactly; a nullable ``int64`` attention
  mask ``[1, S]`` that is binary (0/1) with at least one valid token; a
  **mandatory** ``bool`` image-pad mask ``[1, S]`` that is all-False (T2I
  only — a True slot would mean image conditioning, which is out of scope).
  Positive and negative sequence lengths may differ; ``H`` and the embedding
  dtype must agree across branches. All tensors must be dense, strided,
  contiguous CPU tensors; tensor lookalikes are refused.

* Payload: exactly **one** blob (``"tensors"``) holding one safetensors
  serialization (``safetensors.torch.save``) with versioned metadata
  (codec tag, canonical full request identity, cache key, dtype, cfg_active,
  exact tensor descriptors) plus a strict :class:`qwen_engine.encoder.EncodedPayload`
  metadata document. Tensor names are exactly ``positive.embeds`` /
  ``positive.image_pad_mask`` / (optional) ``positive.attention_mask`` and
  the same three for ``negative.`` — the negative names only when
  ``cfg_active`` is true. A null attention mask is represented by **absence**
  of its tensor, and decoding preserves that null.

* Decode (:func:`payload_to_branches`) bounds bytes / header length / tensor
  count / JSON nesting **before** any library deserialization, validates
  the request's own identity/cache key/CFG policy, true-integer shapes (bool
  excluded), element products and offset spans
  against the actual bytes, and only then deserializes with safetensors —
  after which the *actual* tensor dtypes/shapes are validated against the
  descriptors and metadata with **no casts, reshapes or drops**. Values are
  preserved bit-exactly (the library round-trips the tensor bytes; nothing in
  this module converts them).

Typed failures: an invalid request raises
:class:`qwen_engine.encoder.EncoderRequestError`; a malformed payload (bytes,
header, descriptors, tensors, metadata) raises
:class:`qwen_engine.encoder.EncoderProtocolError`; missing heavy packages
raise :class:`qwen_engine.encoder.EncoderSetupError`; an upstream
``encode_prompt`` runtime failure is wrapped as
:class:`qwen_engine.encoder.EncoderWorkerError`. Library/runtime exceptions
are wrapped as ``Exception`` — ``BaseException`` (KeyboardInterrupt,
SystemExit) is never caught here.
"""
from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any

from qwen_engine._jsonsafe import plain_json, reject_constant
from qwen_engine.encoder import (
    Blob,
    EncodedPayload,
    EncoderProtocolError,
    EncoderRequest,
    EncoderRequestError,
    EncoderSetupError,
    EncoderWorkerError,
    REQUEST_SCHEMA,
    cache_key,
)

# ---------------------------------------------------------------------------
# Versioned tags + constants
# ---------------------------------------------------------------------------

#: Schema tag of the generic :class:`EncodedPayload` metadata document.
TENSOR_PAYLOAD_SCHEMA = "qwen_engine.encoder_tensor_payload.v1"

#: Versioned codec tag carried in the safetensors ``__metadata__`` block and
#: echoed in the payload metadata. Any future format change bumps this tag.
SAFETENSORS_CODEC = "qwen_engine.encoder_safetensors.v1"

#: The one blob name of a tensor payload (exactly one blob is allowed).
TENSOR_BLOB_NAME = "tensors"

#: Canonical encoder dtypes — exact lowercase strings, independent of any DiT
#: quantization choice. Whitespace-padded or upper-case spellings are NOT
#: canonical and never reach the ML layer.
CANONICAL_DTYPES = ("fp16", "bf16", "fp32")

#: canonical dtype -> safetensors dtype code
DTYPE_TO_ST_CODE = {"fp16": "F16", "bf16": "BF16", "fp32": "F32"}

#: canonical dtype -> str(torch dtype) of the tensors upstream must return
DTYPE_TO_TORCH_STR = {
    "fp16": "torch.float16",
    "bf16": "torch.bfloat16",
    "fp32": "torch.float32",
}

#: safetensors dtype code -> (str(torch dtype), element bytes). Only these
#: codes may appear in a payload header; anything else is a typed refusal.
ST_CODE_TO_TORCH_STR_ITEMSIZE = {
    "F16": ("torch.float16", 2),
    "BF16": ("torch.bfloat16", 2),
    "F32": ("torch.float32", 4),
    "I64": ("torch.int64", 8),
    "BOOL": ("torch.bool", 1),
}

_TORCH_STR_TO_ST_CODE = {
    torch_str: code
    for code, (torch_str, _size) in ST_CODE_TO_TORCH_STR_ITEMSIZE.items()
}

#: Branch names and the exact tensor-name suffixes per branch. The attention
#: mask suffix appears only when the upstream collapsed mask is not None.
BRANCH_NAMES = ("positive", "negative")
EMBEDS_SUFFIX = ".embeds"
ATTENTION_MASK_SUFFIX = ".attention_mask"
IMAGE_MASK_SUFFIX = ".image_pad_mask"

#: Payload metadata keys (exactly these, nothing else).
PAYLOAD_METADATA_KEYS = frozenset(
    {"schema", "codec", "cache_key", "dtype", "cfg_active", "request",
     "tensors"})

#: safetensors ``__metadata__`` keys (exactly these, all string values).
ST_METADATA_KEYS = frozenset(
    {"codec", "request_identity", "cache_key", "dtype", "cfg_active",
     "tensor_descriptors"})

# --- decode bounds (enforced BEFORE any library deserialization) ------------

#: Raw tensor-blob cap. Generic transport caps still apply independently:
#: base64 and response metadata make a wire response larger than this blob.
MAX_TENSOR_BLOB_BYTES = 256 * 1024 * 1024

#: Cap on the safetensors header (its uint64 length field and its bytes).
MAX_SAFETENSORS_HEADER_BYTES = 1024 * 1024

#: Cap on the number of tensor entries in one header (a payload has at most
#: six; anything near this cap is corruption or an attack, never a product
#: shape).
MAX_SAFETENSORS_TENSORS = 64

#: Bound on header JSON nesting depth and total node count. The header of a
#: valid payload is a flat dict of small dicts — deep nesting is corruption.
MAX_HEADER_JSON_DEPTH = 8
MAX_HEADER_JSON_NODES = 8192

#: Bound on header key / tensor-name length.
MAX_HEADER_KEY_CHARS = 256


# ---------------------------------------------------------------------------
# Typed records (tensors stay opaque here: no torch at import time)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedProductionRequest:
    """A production-validated request plus the resolved CFG decision. The
    request keeps carrying its full identity (:func:`cache_key` over
    ``key_document()``) even when ``cfg_active`` is False — the negative
    text stays in the identity; only the *encoding* is skipped."""

    request: EncoderRequest
    cfg_active: bool

    @property
    def cache_key(self) -> str:
        return cache_key(self.request)


@dataclass(frozen=True)
class EncoderBranch:
    """One conditioning branch as returned by upstream / recovered from a
    payload. Tensors are opaque (torch objects); the record itself performs
    no tensor work."""

    embeds: Any
    attention_mask: Any | None
    image_pad_mask: Any


@dataclass(frozen=True)
class BranchBundle:
    """Positive branch and, only when CFG was active, the negative branch."""

    positive: EncoderBranch
    negative: EncoderBranch | None = None


def _proto(msg: str) -> EncoderProtocolError:
    return EncoderProtocolError(f"tensor payload: {msg}")


def _true_int(value: Any) -> bool:
    """True integers only — ``bool`` is an ``int`` subclass in Python and is
    explicitly excluded wherever a shape/offset is validated."""
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"))


# ---------------------------------------------------------------------------
# Heavy imports — lazy, inside production calls only
# ---------------------------------------------------------------------------


def _load_torch() -> Any:
    """Import torch lazily (never at module import). A missing package is a
    setup failure, not a payload violation."""
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise EncoderSetupError(
            "production encoding needs the pinned torch build but it is not "
            f"importable here: {exc}") from exc


def _load_safetensors() -> Any:
    """Import ``safetensors.torch`` lazily for the actual (de)serialization
    step — the bounds checks above it are stdlib-only."""
    try:
        return importlib.import_module("safetensors.torch")
    except ImportError as exc:
        raise EncoderSetupError(
            "the tensor codec needs the pinned safetensors package but it is "
            f"not importable here: {exc}") from exc


def _load_pipeline_class() -> Any:
    try:
        return importlib.import_module("diffusers").QwenImage21Pipeline
    except (ImportError, AttributeError) as exc:
        raise EncoderSetupError("the pinned QwenImage21Pipeline is unavailable") from exc


# ---------------------------------------------------------------------------
# Production request validation (stdlib-only, BEFORE any heavy import)
# ---------------------------------------------------------------------------


def validate_production_request(request: Any) -> ValidatedProductionRequest:
    """Validate an encoder request against the PRODUCTION policy and resolve
    the CFG decision. Runs before any heavy import; every violation is
    collected and raised together as one :class:`EncoderRequestError`.

    Policy (handoff 2026-10-12): scalar non-blank positive prompt; no images;
    mandatory complete implementation identity; dtype exactly one of
    ``CANONICAL_DTYPES``; kwargs exactly ``{"cfg_active": bool}``; active CFG
    requires a non-blank negative; inactive CFG never encodes the negative
    but keeps its text in the request identity.
    """
    if not isinstance(request, EncoderRequest):
        raise EncoderRequestError(
            f"request: must be an EncoderRequest, got "
            f"{type(request).__name__}")
    errors: list[str] = []

    prompt = request.prompt
    if not isinstance(prompt, str):
        errors.append(f"prompt: must be a scalar string, got "
                      f"{type(prompt).__name__}")
    elif not prompt.strip():
        errors.append("prompt: must not be blank (whitespace-only prompts are "
                      "refused; production never sends a blank to upstream)")

    if request.images:
        errors.append(f"images: image conditioning is out of scope, got "
                      f"{len(request.images)} image hash(es)")

    if request.implementation_identity is None:
        errors.append("implementation_identity: production encoding requires "
                      "the complete implementation identity, got None "
                      "(generic fake-worker requests are not production "
                      "requests)")

    if request.dtype not in CANONICAL_DTYPES:
        errors.append(f"dtype: must be exactly one of {CANONICAL_DTYPES}, "
                      f"got {request.dtype!r}")

    cfg_active: Any = None
    kwargs = request.kwargs
    if not isinstance(kwargs, Mapping):
        errors.append(f"kwargs: must be exactly {{'cfg_active': bool}}, got "
                      f"{type(kwargs).__name__}")
    elif set(kwargs) != {"cfg_active"}:
        extra = sorted(str(k) for k in set(kwargs) - {"cfg_active"})
        missing = sorted(str(k) for k in {"cfg_active"} - set(kwargs))
        detail = "; ".join(filter(None, [
            f"unknown keys: {', '.join(extra)}" if extra else "",
            f"missing keys: {', '.join(missing)}" if missing else ""]))
        errors.append(f"kwargs: must contain exactly the single key "
                      f"'cfg_active' ({detail})")
    else:
        value = kwargs["cfg_active"]
        if not isinstance(value, bool):
            errors.append(f"kwargs.cfg_active: must be a bool, got "
                          f"{value!r} ({type(value).__name__})")
        else:
            cfg_active = value

    if cfg_active is True:
        negative = request.negative_prompt
        if not isinstance(negative, str):
            errors.append(f"negative_prompt: must be a string when CFG is "
                          f"active, got {type(negative).__name__}")
        elif not negative.strip():
            errors.append("negative_prompt: CFG is active, so the negative "
                          "prompt must not be blank (an active CFG around an "
                          "empty condition is exactly the upstream trap the "
                          "identity must not paper over)")
    # cfg_active False: the negative text is NOT validated for content and
    # never encoded — it stays in the cache/request identity by design.

    if errors:
        raise EncoderRequestError("production encoder request: "
                                  + "; ".join(errors))
    assert cfg_active is not None  # guaranteed: bool value validated above
    return ValidatedProductionRequest(request=request, cfg_active=cfg_active)


# ---------------------------------------------------------------------------
# Tensor validation (actual dense contiguous CPU torch tensors only)
# ---------------------------------------------------------------------------


def _dtype_code(tensor: Any, at: str) -> str:
    code = _TORCH_STR_TO_ST_CODE.get(str(getattr(tensor, "dtype", None)))
    if code is None:
        raise _proto(f"{at}: dtype {str(getattr(tensor, 'dtype', None))!r} "
                     "is not a supported tensor dtype (F16/BF16/F32/I64/BOOL "
                     "only)")
    return code


def _shape_of(tensor: Any, at: str) -> tuple[int, ...]:
    torch = _load_torch()
    if not isinstance(tensor, torch.Tensor):
        raise _proto(f"{at}: must be a torch Tensor")
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise _proto(f"{at}: must be a dense strided CPU tensor")
    if not tensor.is_contiguous():
        raise _proto(f"{at}: must be contiguous (values are never reshaped)")
    shape = getattr(tensor, "shape", None)
    try:
        dims = tuple(shape)
    except TypeError:
        raise _proto(f"{at}: shape {shape!r} is not a sequence") from None
    for d in dims:
        if not _true_int(d):
            raise _proto(f"{at}: shape {dims} contains a non-integer or bool "
                         f"dimension {d!r}")
    elements = _element_product(dims, at)
    code = _dtype_code(tensor, at)
    if elements * ST_CODE_TO_TORCH_STR_ITEMSIZE[code][1] > MAX_TENSOR_BLOB_BYTES:
        raise _proto(f"{at}: tensor exceeds the byte cap")
    return dims


def _all_finite(tensor: Any, at: str) -> None:
    try:
        finite = bool(tensor.isfinite().all().item())
    except EncoderProtocolError:
        raise
    except Exception as exc:
        raise _proto(f"{at}: finiteness check failed "
                     f"({type(exc).__name__}: {exc})") from exc
    if not finite:
        raise _proto(f"{at}: embeddings contain non-finite values")


def _validate_attention_mask(mask: Any, *, seq_len: int, at: str) -> None:
    shape = _shape_of(mask, at)
    if shape != (1, seq_len):
        raise _proto(f"{at}: attention mask must have shape [1, {seq_len}], "
                     f"got {list(shape)}")
    if _dtype_code(mask, at) != "I64":
        raise _proto(f"{at}: attention mask dtype must be int64, got "
                     f"{str(mask.dtype)!r}")
    if mask.min().item() < 0 or mask.max().item() > 1:
        raise _proto(f"{at}: attention mask must be binary (0/1) integers")
    if not mask.any().item():
        raise _proto(f"{at}: attention mask marks no valid token (all zero)")


def _validate_image_mask(mask: Any, *, seq_len: int, at: str) -> None:
    if mask is None:
        raise _proto(f"{at}: the image-pad mask is mandatory (upstream always "
                     "returns one; a missing mask is a contract violation)")
    shape = _shape_of(mask, at)
    if shape != (1, seq_len):
        raise _proto(f"{at}: image-pad mask must have shape [1, {seq_len}], "
                     f"got {list(shape)}")
    if _dtype_code(mask, at) != "BOOL":
        raise _proto(f"{at}: image-pad mask dtype must be bool, got "
                     f"{str(mask.dtype)!r}")
    if mask.any().item():
        raise _proto(f"{at}: image-pad mask must be all-False for T2I (a True "
                     "slot is image conditioning, out of scope)")


def _validate_branch_output(raw: Any, *, side: str,
                            want_torch_str: str,
                            ) -> tuple[EncoderBranch, int, int, str]:
    """Validate one upstream ``encode_prompt`` triple; returns the branch
    record plus ``(S, H, code)`` for the cross-branch agreement check."""
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        length = (len(raw) if isinstance(raw, (tuple, list)) else "?")
        raise _proto(f"{side} branch: upstream encode_prompt must return "
                     f"(embeds, attention_mask, image_pad_mask), got "
                     f"{type(raw).__name__} of length {length}")
    embeds, mask, image_mask = raw
    at = f"{side}.embeds"
    shape = _shape_of(embeds, at)
    if len(shape) != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] < 1:
        raise _proto(f"{at}: embeddings must have shape [1, S>=1, H>=1], got "
                     f"{list(shape)}")
    is_float = getattr(embeds, "is_floating_point", None)
    if not (callable(is_float) and bool(is_float())):
        raise _proto(f"{at}: embeddings must be a floating-point tensor, got "
                     f"dtype {str(getattr(embeds, 'dtype', None))!r}")
    code = _dtype_code(embeds, at)
    if str(embeds.dtype) != want_torch_str:
        raise _proto(f"{at}: embeddings dtype {str(embeds.dtype)!r} does not "
                     f"match the requested dtype {want_torch_str!r} — output "
                     "dtype is validated, never silently cast")
    _all_finite(embeds, at)
    seq_len, hidden = shape[1], shape[2]
    if mask is not None:
        _validate_attention_mask(mask, seq_len=seq_len,
                                 at=side + ATTENTION_MASK_SUFFIX)
    _validate_image_mask(image_mask, seq_len=seq_len,
                         at=side + IMAGE_MASK_SUFFIX)
    return (EncoderBranch(embeds=embeds, attention_mask=mask,
                          image_pad_mask=image_mask),
            seq_len, hidden, code)


# ---------------------------------------------------------------------------
# Lazy upstream encode_prompt reuse
# ---------------------------------------------------------------------------


def encode_with_supplied_model(model: Any, processor: Any,
                               request: EncoderRequest) -> EncodedPayload:
    """Reuse the pinned upstream pipeline with a supplied CPU model/processor.

    The pipeline supplies its own templates, extraction and norm-hook logic.
    No component is loaded or moved by this function.
    """
    validate_production_request(request)
    torch = _load_torch()
    if not isinstance(model, torch.nn.Module) or not callable(processor):
        raise EncoderSetupError("a supplied torch model and callable processor are required")
    if any(t.device.type != "cpu" for t in chain(model.parameters(), model.buffers())):
        raise EncoderSetupError("the supplied encoder model must already be on CPU")
    cls = _load_pipeline_class()
    try:
        with torch.inference_mode():
            pipeline = cls(scheduler=None, vae=None, transformer=None,
                           text_encoder=model, processor=processor)
    except Exception as exc:
        raise EncoderSetupError(f"upstream encoder pipeline setup failed: {exc}") from exc
    return encode_with_supplied_pipeline(pipeline, request)


def encode_with_supplied_pipeline(pipeline: Any,
                                  request: EncoderRequest) -> EncodedPayload:
    """Encode a production request by calling ``encode_prompt`` on the
    SUPPLIED pinned ``QwenImage21Pipeline`` — this helper never constructs,
    loads, moves or casts models, never touches the tokenizer / chat
    template / control tokens, and registers no hooks (upstream owns the
    prompt pipeline and its pre-RMSNorm hook; docs/interfaces.md §2).

    Returns an :class:`EncodedPayload` with exactly one safetensors blob and
    the full request identity in its metadata. The negative branch is encoded
    only when ``cfg_active`` is true."""
    validated = validate_production_request(request)  # BEFORE any heavy import
    req = validated.request
    torch = _load_torch()
    if not callable(getattr(pipeline, "encode_prompt", None)):
        raise EncoderSetupError(
            "the supplied pipeline must expose encode_prompt (a constructed "
            "pinned QwenImage21Pipeline with the text encoder and processor; "
            "this helper never loads a model itself)")

    device = torch.device("cpu")
    kwargs = dict(image=None, device=device, num_images_per_prompt=1)
    try:
        with torch.inference_mode():
            positive_raw = pipeline.encode_prompt(prompt=req.prompt, **kwargs)
            negative_raw = None
            if validated.cfg_active:
                negative_raw = pipeline.encode_prompt(
                    prompt=req.negative_prompt, **kwargs)
    except Exception as exc:  # library/runtime failures typed; NEVER
        # BaseException — KeyboardInterrupt/SystemExit propagate.
        raise EncoderWorkerError(
            f"upstream encode_prompt failed on the supplied pipeline: "
            f"{type(exc).__name__}: {exc}", exit_code=None) from exc

    want = DTYPE_TO_TORCH_STR[req.dtype]
    positive, _pos_s, pos_h, pos_code = _validate_branch_output(
        positive_raw, side="positive", want_torch_str=want)
    negative = None
    if validated.cfg_active:
        negative, _neg_s, neg_h, neg_code = _validate_branch_output(
            negative_raw, side="negative", want_torch_str=want)
        if neg_h != pos_h or neg_code != pos_code:
            raise _proto(f"branches: the negative branch (H={neg_h}, "
                         f"{neg_code}) must agree with the positive branch "
                         f"(H={pos_h}, {pos_code}) on hidden width and dtype; "
                         "sequence lengths may differ, H and dtype may not")
    return serialize_branches(validated, BranchBundle(positive, negative))


# ---------------------------------------------------------------------------
# Serialization: branch tensors -> exactly one safetensors blob
# ---------------------------------------------------------------------------


def _branch_tensors(bundle: BranchBundle) -> dict[str, Any]:
    """The exact tensor-name mapping of a payload (names are the contract)."""
    tensors: dict[str, Any] = {}
    for branch_name in BRANCH_NAMES:
        branch = (bundle.positive if branch_name == "positive"
                  else bundle.negative)
        if branch is None:
            continue
        tensors[branch_name + EMBEDS_SUFFIX] = branch.embeds
        if branch.attention_mask is not None:
            tensors[branch_name + ATTENTION_MASK_SUFFIX] = branch.attention_mask
        tensors[branch_name + IMAGE_MASK_SUFFIX] = branch.image_pad_mask
    return tensors


def _tensor_descriptors(tensors: Mapping[str, Any]) -> list[dict[str, Any]]:
    descriptors = []
    for name in sorted(tensors):
        tensor = tensors[name]
        descriptors.append({
            "name": name,
            "dtype": _dtype_code(tensor, f"tensor {name!r}"),
            "shape": list(_shape_of(tensor, f"tensor {name!r}")),
            "bytes": (_element_product(_shape_of(tensor, f"tensor {name!r}"),
                                       name)
                      * ST_CODE_TO_TORCH_STR_ITEMSIZE[
                          _dtype_code(tensor, f"tensor {name!r}")][1]),
        })
    return descriptors


def _element_product(shape: tuple[int, ...], at: str) -> int:
    if len(shape) > 3:
        raise _proto(f"tensor {at!r}: shape rank exceeds three")
    product = 1
    for d in shape:
        if not _true_int(d) or d < 0:
            raise _proto(f"tensor {at!r}: shape {list(shape)} contains a "
                         f"non-integer or negative dimension {d!r}")
        product *= d
        if d > MAX_TENSOR_BLOB_BYTES or product > MAX_TENSOR_BLOB_BYTES:
            raise _proto(f"tensor {at!r}: shape exceeds the tensor byte cap")
    return product


def serialize_branches(validated: ValidatedProductionRequest,
                       bundle: BranchBundle) -> EncodedPayload:
    """Serialize a validated branch bundle for a validated request into one
    :class:`EncodedPayload`: exactly one blob of safetensors bytes with
    versioned metadata, the full request identity and the exact tensor
    descriptors. No value is cast — tensors are handed to the library
    unchanged."""
    if not isinstance(validated, ValidatedProductionRequest):
        raise EncoderRequestError("a ValidatedProductionRequest is required")
    checked = validate_production_request(validated.request)
    if checked.cfg_active is not validated.cfg_active:
        raise EncoderRequestError("validated cfg_active disagrees with the request")
    if not isinstance(bundle, BranchBundle) or not isinstance(bundle.positive, EncoderBranch):
        raise _proto("a positive EncoderBranch is required")
    if checked.cfg_active != (bundle.negative is not None):
        raise _proto("negative branch presence must match cfg_active")
    req = checked.request
    hidden = None
    for side, branch in (("positive", bundle.positive), ("negative", bundle.negative)):
        if branch is None:
            continue
        if not isinstance(branch, EncoderBranch):
            raise _proto(f"{side}: must be an EncoderBranch")
        _, _, width, _ = _validate_branch_output(
            (branch.embeds, branch.attention_mask, branch.image_pad_mask),
            side=side, want_torch_str=DTYPE_TO_TORCH_STR[req.dtype])
        if hidden is not None and width != hidden:
            raise _proto("branches disagree on hidden width")
        hidden = width
    tensors = _branch_tensors(bundle)
    descriptors = _tensor_descriptors(tensors)
    if sum(d["bytes"] for d in descriptors) > MAX_TENSOR_BLOB_BYTES - MAX_SAFETENSORS_HEADER_BYTES - 8:
        raise _proto("tensor data exceeds the serialization byte cap")
    st_metadata = {
        "codec": SAFETENSORS_CODEC,
        "request_identity": _canonical_json(req.key_document()),
        "cache_key": cache_key(req),
        "dtype": req.dtype,
        "cfg_active": "true" if validated.cfg_active else "false",
        "tensor_descriptors": _canonical_json(descriptors),
    }
    if len(_canonical_json(st_metadata).encode("utf-8")) > MAX_SAFETENSORS_HEADER_BYTES // 2:
        raise _proto("serialization metadata exceeds the header byte cap")
    st = _load_safetensors()
    try:
        data = st.save(tensors, metadata=st_metadata)
    except Exception as exc:  # library failure = cannot form the payload
        raise _proto(f"safetensors serialization failed "
                     f"({type(exc).__name__}: {exc})") from exc
    if not isinstance(data, (bytes, bytearray)) or not len(data):
        raise _proto("safetensors serialization returned no bytes "
                     f"({type(data).__name__})")
    _inspect_safetensors_bytes(data)
    metadata = {
        "schema": TENSOR_PAYLOAD_SCHEMA,
        "codec": SAFETENSORS_CODEC,
        "cache_key": st_metadata["cache_key"],
        "dtype": req.dtype,
        "cfg_active": validated.cfg_active,
        "request": plain_json(req.key_document()),
        "tensors": [dict(d) for d in descriptors],
    }
    return EncodedPayload(metadata=metadata,
                          blobs=(Blob(name=TENSOR_BLOB_NAME,
                                      data=bytes(data)),))


# ---------------------------------------------------------------------------
# Bounded safetensors byte/header inspection (stdlib-only, pre-deserialization)
# ---------------------------------------------------------------------------


def _check_header_shape(node: Any) -> None:
    """Bound the parsed header JSON's nesting depth and total node count
    (iteratively — a parse already succeeded, so the shapes are JSON types).
    """
    stack: list[tuple[Any, int]] = [(node, 1)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_HEADER_JSON_NODES or depth > MAX_HEADER_JSON_DEPTH:
            raise _proto("safetensors header nests deeper or wider than the "
                         "codec bounds allow")
        if isinstance(current, dict):
            for key, value in current.items():
                if not isinstance(key, str) or not key \
                        or len(key) > MAX_HEADER_KEY_CHARS:
                    raise _proto("safetensors header has a non-string, empty "
                                 f"or over-long key ({str(key)[:80]!r})")
                stack.append((value, depth + 1))
        elif isinstance(current, list):
            for value in current:
                stack.append((value, depth + 1))


def _check_json_depth(text: str) -> None:
    """Bound nesting before the JSON parser allocates nested containers."""
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
            if depth > MAX_HEADER_JSON_DEPTH:
                raise _proto("safetensors header nests deeper than the codec bounds")
        elif char in "]}":
            depth -= 1


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _proto(f"safetensors header repeats the key {key!r}")
        out[key] = value
    return out


def _inspect_safetensors_bytes(data: Any) -> tuple[dict[str, Any],
                                                   dict[str, str] | None,
                                                   int]:
    """Fully validate the byte/header framing of a safetensors serialization
    with the **stdlib only**, before any library touches it:

    * total bytes <= :data:`MAX_TENSOR_BLOB_BYTES`;
    * the uint64 header length is nonzero, <= :data:`MAX_SAFETENSORS_HEADER_BYTES`
      and the header fits in the actual bytes;
    * the header parses as strict UTF-8 JSON (NaN/Infinity literals refused,
      duplicate keys refused) within the depth/node/key-length bounds;
    * every tensor entry carries exactly ``dtype``/``shape``/``data_offsets``
      with supported dtype codes, TRUE integer shapes (bool excluded) and
      offset spans that fit the data region, never overlap, cover it exactly,
      and whose ``end - start`` equals ``product(shape) * itemsize``.

    Returns ``(tensors_spec, metadata, data_region_len)``.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise _proto(f"blob bytes must be bytes, got {type(data).__name__}")
    if len(data) > MAX_TENSOR_BLOB_BYTES:
        raise _proto(f"blob is {len(data)} bytes, above the "
                     f"{MAX_TENSOR_BLOB_BYTES}-byte codec cap")
    if len(data) < 8:
        raise _proto("blob is shorter than the 8-byte safetensors length "
                     "field")
    header_len = int.from_bytes(bytes(data[:8]), "little")
    if header_len == 0:
        raise _proto("safetensors header length is zero")
    if header_len > MAX_SAFETENSORS_HEADER_BYTES:
        raise _proto(f"safetensors header length {header_len} exceeds the "
                     f"{MAX_SAFETENSORS_HEADER_BYTES}-byte cap")
    if 8 + header_len > len(data):
        raise _proto(f"safetensors header claims {header_len} bytes but only "
                     f"{len(data) - 8} bytes follow the length field")
    try:
        text = bytes(data[8:8 + header_len]).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _proto(f"safetensors header is not valid UTF-8: {exc}") from exc
    try:
        _check_json_depth(text)
        header = json.loads(text, parse_constant=reject_constant,
                            object_pairs_hook=_no_duplicate_keys)
    except (ValueError, RecursionError) as exc:
        raise _proto(f"safetensors header is not strict JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise _proto(f"safetensors header must be a JSON object, got "
                     f"{type(header).__name__}")
    _check_header_shape(header)

    metadata: dict[str, str] | None = None
    tensors: dict[str, Any] = {}
    for key, value in header.items():
        if key == "__metadata__":
            if not isinstance(value, dict) or any(
                    not isinstance(k, str) or not isinstance(v, str)
                    for k, v in value.items()):
                raise _proto("safetensors __metadata__ must be an object of "
                             "string values (or absent)")
            metadata = dict(value)
            continue
        tensors[key] = value
    if not tensors:
        raise _proto("safetensors header declares no tensors")
    if len(tensors) > MAX_SAFETENSORS_TENSORS:
        raise _proto(f"{len(tensors)} tensors exceed the "
                     f"{MAX_SAFETENSORS_TENSORS}-tensor codec cap")

    data_len = len(data) - 8 - header_len
    spans: list[tuple[int, int, str]] = []
    covered = 0
    for name, spec in sorted(tensors.items()):
        at = f"tensor {name!r}"
        if not isinstance(spec, dict):
            raise _proto(f"{at} entry must be an object, got "
                         f"{type(spec).__name__}")
        if set(spec) != {"dtype", "shape", "data_offsets"}:
            raise _proto(f"{at} entry keys must be exactly dtype/shape/"
                         f"data_offsets, got {sorted(map(str, spec))}")
        code = spec["dtype"]
        if not isinstance(code, str) or code not in ST_CODE_TO_TORCH_STR_ITEMSIZE:
            raise _proto(f"{at}: dtype code {code!r} is not supported "
                         f"(known: {sorted(ST_CODE_TO_TORCH_STR_ITEMSIZE)})")
        itemsize = ST_CODE_TO_TORCH_STR_ITEMSIZE[code][1]
        shape = spec["shape"]
        if (not isinstance(shape, list)
                or any(not _true_int(d) or d < 0 for d in shape)):
            raise _proto(f"{at}: shape {shape!r} must be a list of true "
                         "non-negative integers (bool is not an integer here)")
        offsets = spec["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(not _true_int(o) for o in offsets)):
            raise _proto(f"{at}: data_offsets {offsets!r} must be a pair of "
                         "true integers")
        start, end = offsets
        if start < 0 or end < start or end > data_len:
            raise _proto(f"{at}: data_offsets [{start}, {end}) lie outside "
                         f"the {data_len}-byte data region")
        nbytes = end - start
        if nbytes % itemsize:
            raise _proto(f"{at}: span of {nbytes} bytes is not a multiple of "
                         f"the {code} element size {itemsize}")
        product = _element_product(tuple(shape), name)
        if product * itemsize != nbytes:
            raise _proto(f"{at}: shape {shape} ({product} elements) does not "
                         f"account for the {nbytes}-byte span at {itemsize} "
                         "bytes per element")
        spans.append((start, end, name))
        covered += nbytes
    spans.sort()
    for (a_start, a_end, a_name), (b_start, _b_end, b_name) in zip(spans,
                                                                   spans[1:]):
        if a_end > b_start:
            raise _proto(f"tensor spans {a_name!r} [{a_start}, {a_end}) and "
                         f"{b_name!r} [{b_start}, ...) overlap")
    if covered != data_len:
        raise _proto(f"tensor spans cover {covered} of the {data_len} data "
                     "bytes (gaps or trailing bytes are not part of the "
                     "format)")
    return tensors, metadata, data_len


# ---------------------------------------------------------------------------
# Decode: payload -> branch records (bounds first, library second)
# ---------------------------------------------------------------------------


def _validate_st_metadata(metadata: dict[str, str] | None,
                          payload_md: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Cross-check the safetensors metadata block against the payload
    metadata; returns the declared tensor descriptors."""
    if metadata is None:
        raise _proto("safetensors __metadata__ block is missing (the codec "
                     "always writes one)")
    if set(metadata) != ST_METADATA_KEYS:
        raise _proto(f"safetensors metadata keys must be exactly "
                     f"{sorted(ST_METADATA_KEYS)}, got "
                     f"{sorted(map(str, metadata))}")
    if metadata["codec"] != SAFETENSORS_CODEC:
        raise _proto(f"safetensors codec tag {metadata['codec']!r} is not "
                     f"{SAFETENSORS_CODEC!r}")
    if metadata["dtype"] != payload_md["dtype"]:
        raise _proto("safetensors metadata dtype disagrees with the payload "
                     "metadata dtype")
    want_cfg = "true" if payload_md["cfg_active"] else "false"
    if metadata["cfg_active"] != want_cfg:
        raise _proto("safetensors metadata cfg_active disagrees with the "
                     "payload metadata")
    if metadata["cache_key"] != payload_md["cache_key"]:
        raise _proto("safetensors metadata cache key disagrees with the "
                     "payload metadata")
    if metadata["request_identity"] != _canonical_json(plain_json(payload_md["request"])):
        raise _proto("safetensors request identity disagrees with the payload "
                     "metadata request document")
    declared = plain_json(payload_md["tensors"])
    if metadata["tensor_descriptors"] != _canonical_json(declared):
        raise _proto("safetensors tensor descriptors disagree with the "
                     "payload metadata tensors")
    return declared


def _validate_descriptors(descriptors: Any) -> list[dict[str, Any]]:
    """Validate the payload metadata descriptor list (which arrives from the
    frozen payload snapshot as tuples/proxies) and return PLAIN dicts."""
    if not isinstance(descriptors, (list, tuple)) or not descriptors:
        raise _proto("metadata tensors must be a non-empty descriptor list")
    if len(descriptors) > 6:
        raise _proto("metadata declares more than six tensors")
    out = []
    for i, spec in enumerate(descriptors):
        at = f"tensors[{i}]"
        if not isinstance(spec, Mapping) or set(spec) != {"name", "dtype",
                                                          "shape", "bytes"}:
            raise _proto(f"{at} must carry exactly name/dtype/shape/bytes")
        if not isinstance(spec["name"], str) or not spec["name"] or len(spec["name"]) > MAX_HEADER_KEY_CHARS:
            raise _proto(f"{at}.name must be a non-empty string")
        if not isinstance(spec["dtype"], str) \
                or spec["dtype"] not in ST_CODE_TO_TORCH_STR_ITEMSIZE:
            raise _proto(f"{at}.dtype {spec['dtype']!r} is not a supported "
                         "dtype code")
        shape = spec["shape"]
        if (not isinstance(shape, (list, tuple))
                or any(not _true_int(d) or d < 0 for d in shape)):
            raise _proto(f"{at}.shape {shape!r} must be a list of true "
                         "non-negative integers")
        if not _true_int(spec["bytes"]) or spec["bytes"] < 0:
            raise _proto(f"{at}.bytes must be a true non-negative integer")
        if _element_product(tuple(shape), at) * ST_CODE_TO_TORCH_STR_ITEMSIZE[spec["dtype"]][1] != spec["bytes"]:
            raise _proto(f"{at}.bytes disagrees with the shape and dtype")
        out.append({"name": spec["name"], "dtype": spec["dtype"],
                    "shape": [int(d) for d in shape],
                    "bytes": int(spec["bytes"])})
    names = [s["name"] for s in out]
    if len(set(names)) != len(names):
        raise _proto("metadata tensors repeat a tensor name")
    return out


def _expected_names(cfg_active: bool,
                    declared_names: set[str]) -> dict[str, dict[str, bool]]:
    """Map branch -> {suffix: required} and reject any name outside the
    contract (unknown names, or negative names with inactive CFG)."""
    plan: dict[str, dict[str, bool]] = {}
    for branch in BRANCH_NAMES:
        prefix = branch + "."
        branch_names = {n for n in declared_names if n.startswith(prefix)}
        if branch == "negative" and not cfg_active:
            if branch_names:
                raise _proto("negative tensors are present although cfg_active "
                             "is false (an inactive CFG never encodes the "
                             "negative branch)")
            continue
        if branch == "negative" and not branch_names:
            raise _proto("cfg_active is true but no negative tensors are "
                         "declared")
        known = {EMBEDS_SUFFIX, ATTENTION_MASK_SUFFIX, IMAGE_MASK_SUFFIX}
        suffixes = set()
        for name in branch_names:
            suffix = name[len(branch):]
            if suffix not in known:
                raise _proto(f"tensor name {name!r} is not part of the codec "
                             "contract")
            suffixes.add(suffix)
        if EMBEDS_SUFFIX not in suffixes:
            raise _proto(f"branch {branch!r} has no {EMBEDS_SUFFIX} tensor")
        if IMAGE_MASK_SUFFIX not in suffixes:
            raise _proto(f"branch {branch!r} has no {IMAGE_MASK_SUFFIX} "
                         "tensor (the image-pad mask is mandatory)")
        plan[branch] = {suf: (suf in suffixes) for suf in known}
    if not plan.get("positive"):
        raise _proto("no positive tensors are declared")
    leftovers = set()
    for name in declared_names:
        if not any(name.startswith(b + ".") for b in BRANCH_NAMES):
            leftovers.add(name)
    if leftovers:
        raise _proto(f"tensor names outside the codec contract: "
                     f"{sorted(leftovers)}")
    return plan


def _inspect_payload(payload: EncodedPayload, expect_request: EncoderRequest | None):
    """Common stdlib-only identity/codec/header checks, before tensor loading."""
    if not isinstance(payload, EncodedPayload):
        raise _proto(f"payload must be an EncodedPayload, got "
                     f"{type(payload).__name__}")
    if len(payload.blobs) != 1:
        raise _proto(f"exactly one blob is required, got {len(payload.blobs)}")
    blob = payload.blobs[0]
    if blob.name != TENSOR_BLOB_NAME:
        raise _proto(f"the single blob must be named {TENSOR_BLOB_NAME!r}, "
                     f"got {blob.name!r}")
    md = payload.metadata
    if not isinstance(md, Mapping) or set(md) != PAYLOAD_METADATA_KEYS:
        raise _proto(f"payload metadata keys must be exactly "
                     f"{sorted(PAYLOAD_METADATA_KEYS)}")
    if md["schema"] != TENSOR_PAYLOAD_SCHEMA:
        raise _proto(f"payload schema {md['schema']!r} is not "
                     f"{TENSOR_PAYLOAD_SCHEMA!r}")
    if md["codec"] != SAFETENSORS_CODEC:
        raise _proto(f"payload codec tag {md['codec']!r} is not "
                     f"{SAFETENSORS_CODEC!r}")
    cache = md["cache_key"]
    if not isinstance(cache, str) or len(cache) != 64 \
            or any(c not in "0123456789abcdef" for c in cache):
        raise _proto(f"payload cache key {cache!r} is not a sha256 hex")
    dtype = md["dtype"]
    if dtype not in CANONICAL_DTYPES:
        raise _proto(f"payload dtype {dtype!r} is not canonical")
    cfg_active = md["cfg_active"]
    if not isinstance(cfg_active, bool):
        raise _proto(f"payload cfg_active must be a bool, got {cfg_active!r}")
    request_doc = md["request"]
    if not isinstance(request_doc, Mapping):
        raise _proto("payload request identity must be an object")
    wire_doc = plain_json(request_doc)
    wire_doc["schema"] = REQUEST_SCHEMA
    wire_doc["request_id"] = "tensor-codec-validation"
    try:
        recovered, _ = EncoderRequest.from_request_dict(wire_doc)
        recovered_policy = validate_production_request(recovered)
    except EncoderRequestError as exc:
        raise _proto(f"payload request identity is invalid: {exc}") from exc
    if plain_json(request_doc) != recovered.key_document():
        raise _proto("payload request identity is not the exact key document")
    if cache != cache_key(recovered):
        raise _proto("payload cache key disagrees with its request identity")
    if dtype != recovered.dtype or cfg_active != recovered_policy.cfg_active:
        raise _proto("payload dtype/cfg_active disagrees with its request identity")
    declared = _validate_descriptors(md["tensors"])

    if expect_request is not None:
        if not isinstance(expect_request, EncoderRequest):
            raise EncoderRequestError(
                f"expect_request: must be an EncoderRequest, got "
                f"{type(expect_request).__name__}")
        if cache_key(expect_request) != cache:
            raise _proto("payload cache key does not match the expected "
                         "request identity")
        if plain_json(request_doc) != expect_request.key_document():
            raise _proto("payload request document does not match the "
                         "expected request identity")

    # --- bounds before any library touches the bytes ------------------------
    header_tensors, st_md, _data_len = _inspect_safetensors_bytes(blob.data)
    _validate_st_metadata(st_md, md)  # returns the (equal) declared list

    # tensor names: header == descriptors == contract plan
    header_names = set(header_tensors)
    declared_names = {s["name"] for s in declared}
    if header_names != declared_names:
        missing = sorted(declared_names - header_names)
        extra = sorted(header_names - declared_names)
        raise _proto(f"header/descriptor tensor-name mismatch "
                     f"(missing={missing}, unexpected={extra})")
    plan = _expected_names(cfg_active, header_names)
    by_name = {s["name"]: s for s in declared}
    for name, spec in header_tensors.items():
        want = by_name[name]
        if spec["dtype"] != want["dtype"] or spec["shape"] != want["shape"]:
            raise _proto(f"tensor {name!r}: header entry "
                         f"{spec!r} disagrees with the declared descriptor "
                         f"{want!r}")
        if want["bytes"] != spec["data_offsets"][1] - spec["data_offsets"][0]:
            raise _proto(f"tensor {name!r}: descriptor byte count disagrees with header")
    hidden = None
    for side, suffixes in plan.items():
        spec = header_tensors[side + EMBEDS_SUFFIX]
        shape = spec["shape"]
        if len(shape) != 3 or shape[0] != 1 or min(shape[1:]) < 1:
            raise _proto(f"{side}: header embeddings must have shape [1, S>=1, H>=1]")
        if spec["dtype"] != DTYPE_TO_ST_CODE[dtype]:
            raise _proto(f"{side}: header embedding dtype disagrees with request")
        if hidden is not None and shape[2] != hidden:
            raise _proto("branches disagree on hidden width")
        hidden = shape[2]
        for suffix, code in ((ATTENTION_MASK_SUFFIX, "I64"), (IMAGE_MASK_SUFFIX, "BOOL")):
            if suffixes[suffix]:
                mask_spec = header_tensors[side + suffix]
                if mask_spec["dtype"] != code or mask_spec["shape"] != [1, shape[1]]:
                    raise _proto(f"{side + suffix}: header mask shape/dtype disagrees with embeddings")

    return plan, by_name, header_names, dtype, cfg_active, blob


def validate_serialized_payload(payload: EncodedPayload, *, expect_request: EncoderRequest) -> None:
    """Validate request/codec/header structure without ML imports or tensors.

    Does NOT validate actual embedding finiteness/mask values; consumers must
    still call payload_to_branches with this same expected request before use.
    """
    if not isinstance(expect_request, EncoderRequest):
        raise EncoderRequestError("expect_request must be an EncoderRequest")
    _inspect_payload(payload, expect_request)


def payload_to_branches(payload: EncodedPayload, *,
                        expect_request: EncoderRequest | None = None) -> BranchBundle:
    """Recover branches: stdlib bounds/identity first, then tensor values.

    No casts/reshapes/mask substitution. Actual dtypes/shapes/finiteness and
    mask values are validated after deserialization. Null masks remain null.
    With expect_request, identity must match its cache key/document exactly.
    """
    plan, by_name, header_names, dtype, cfg_active, blob = _inspect_payload(payload, expect_request)

    # --- library deserialization (only now) ---------------------------------
    st = _load_safetensors()
    try:
        tensors = st.load(blob.data)
    except Exception as exc:  # library failure is a malformed payload
        raise _proto(f"safetensors deserialization failed "
                     f"({type(exc).__name__}: {exc})") from exc
    if not isinstance(tensors, dict):
        raise _proto(f"safetensors returned {type(tensors).__name__}, not a "
                     "tensor mapping")
    if set(tensors) != header_names:
        raise _proto(f"deserialized tensors {sorted(tensors)} do not match "
                     f"the header {sorted(header_names)}")

    # --- actual tensors against the declarations (no casts/reshapes) --------
    want_embed_str = DTYPE_TO_TORCH_STR[dtype]
    for name, tensor in tensors.items():
        at = f"tensor {name!r}"
        actual_shape = _shape_of(tensor, at)
        if list(actual_shape) != by_name[name]["shape"]:
            raise _proto(f"{at}: actual shape {list(actual_shape)} differs "
                         f"from the declared shape {by_name[name]['shape']} "
                         "(shapes are validated, never reshaped)")
        actual_code = _dtype_code(tensor, at)
        if actual_code != by_name[name]["dtype"]:
            raise _proto(f"{at}: actual dtype {str(tensor.dtype)!r} "
                         f"({actual_code}) differs from the declared "
                         f"{by_name[name]['dtype']} — values are never cast")
        itemsize = ST_CODE_TO_TORCH_STR_ITEMSIZE[actual_code][1]
        if _element_product(actual_shape, name) * itemsize \
                != by_name[name]["bytes"]:
            raise _proto(f"{at}: element count * itemsize does not match the "
                         "declared byte count")

    branches: dict[str, EncoderBranch | None] = {"positive": None,
                                                 "negative": None}
    for branch_name, suffixes in plan.items():
        embeds = tensors[branch_name + EMBEDS_SUFFIX]
        at = f"tensor {branch_name + EMBEDS_SUFFIX!r}"
        shape = _shape_of(embeds, at)
        if len(shape) != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] < 1:
            raise _proto(f"{at}: embeddings must have shape [1, S>=1, H>=1], "
                         f"got {list(shape)}")
        if _dtype_code(embeds, at) != DTYPE_TO_ST_CODE[dtype]:
            raise _proto(f"{at}: embeddings dtype {str(embeds.dtype)!r} does "
                         f"not match the payload dtype {dtype!r}")
        if str(embeds.dtype) != want_embed_str:
            raise _proto(f"{at}: embeddings dtype {str(embeds.dtype)!r} is "
                         f"not {want_embed_str!r}")
        is_float = getattr(embeds, "is_floating_point", None)
        if not (callable(is_float) and bool(is_float())):
            raise _proto(f"{at}: embeddings must be a floating-point tensor")
        _all_finite(embeds, at)
        mask = (tensors[branch_name + ATTENTION_MASK_SUFFIX]
                if suffixes[ATTENTION_MASK_SUFFIX] else None)
        if mask is not None:
            _validate_attention_mask(mask, seq_len=shape[1],
                                     at=f"tensor {branch_name + ATTENTION_MASK_SUFFIX!r}")
        _validate_image_mask(tensors[branch_name + IMAGE_MASK_SUFFIX],
                             seq_len=shape[1],
                             at=f"tensor {branch_name + IMAGE_MASK_SUFFIX!r}")
        branches[branch_name] = EncoderBranch(
            embeds=embeds, attention_mask=mask,
            image_pad_mask=tensors[branch_name + IMAGE_MASK_SUFFIX])

    pos_shape = _shape_of(branches["positive"].embeds, "positive embeds")
    if cfg_active:
        neg_shape = _shape_of(branches["negative"].embeds, "negative embeds")
        if pos_shape[2] != neg_shape[2]:
            raise _proto(f"branches disagree on hidden width "
                         f"(positive H={pos_shape[2]}, negative H="
                         f"{neg_shape[2]}); sequence lengths may differ, H "
                         "may not")
        if str(branches["positive"].embeds.dtype) \
                != str(branches["negative"].embeds.dtype):
            raise _proto("branches disagree on the embedding dtype")
    return BranchBundle(positive=branches["positive"],
                        negative=branches["negative"])
