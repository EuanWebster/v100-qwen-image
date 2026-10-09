"""Encoder subprocess worker + embedding cache (plan R5, task 1.4).

Stdlib-only. The real encoder (Qwen3-VL, heavy imports) will run in a
*throwaway child process* so it costs no VRAM; this module is the lifecycle +
cache skeleton around it, exercised entirely by a **fake worker** in tests —
no model, no torch.

Pieces
------
* **Wire schemas** (versioned tags, strict validation):

  - ``qwen_engine.encoder_request.v2`` — what the engine hands the worker
    (v2 adds the optional :class:`EncoderImplementationIdentity` block).
  - ``qwen_engine.encoder_result.v1``  — what the worker writes back
    (status ok/error; payload blobs are opaque base64 bytes on the wire).

* **Cache key v3** (:func:`cache_key`, plan R5): model repo + revision/hash,
  processor / tokenizer / chat-template identities (revision *or* content
  hash), prompt, **negative prompt** (the LTX ``_embed_cache_key`` negative-key
  fix, tested both variants), **all** encoding kwargs, encoder dtype,
  **ordered** image content hashes for future image conditioning, and the
  optional immutable :class:`EncoderImplementationIdentity` (v3 — the worker
  protocol/implementation, prompt-extraction algorithm and installed-package
  revisions that determine *what code* produced the embeddings).

* **Entry format** ``qwen_engine.embed_entry.v1``: one canonical JSON header
  line + the raw concatenated blob bytes, each blob pinned by length and
  sha256. Blob payloads are *opaque bytes*, so a real tensor payload can be
  added later by the worker side (which may import torch) without the engine
  cache ever importing torch itself.

* **EmbeddingCache**: atomic writes (temp file + ``os.replace``), corrupt
  entry = **miss + quarantine** (never overwritten in place), LRU cap by
  mtime, configurable root (R7 — the caller decides where it lives;
  unconfigured = nothing written). All mutations run under a cache-wide
  in-process mutex associated with the **canonical root**, and per-key locks
  are shared the same way: instances on one root cannot race the LRU into
  emptying the cache, and cannot double-encode the same key. The configured
  root/scratch paths are resolved **once** to canonical absolute paths at
  construction (CWD
  changes can never redirect a later write or move the lock registry to a
  different directory), and the cache **only ever operates on files it owns
  under that root**: a leaf entry that is a symlink, carries more than one
  hard link, is not a regular file or is not owned by this user is refused
  as a miss — never read, never touched (no external file's bytes, link
  count or mtime can be moved through a cache path); quarantine is refused
  unless its directory is a real directory contained under the root, and a
  refused move leaves the evidence in place. Entry reads are bounded
  (``max_entry_bytes``: pre-size check + a ``cap + 1`` bounded read, no
  parse of oversized content), and a ``put`` that exceeds the cap fails
  **before** constructing the combined entry buffer. This is ordinary
  single-writer hygiene: no hostile-concurrent-filesystem safety is
  claimed.

* **EncoderWorker**: configurable ``command`` with ``{python}`` /
  ``{request}`` / ``{response}`` placeholders, timeout and cooperative
  cancellation (terminate → kill), **bounded response reads** (a configured
  ``max_response_bytes`` cap is enforced on a pre-read ``stat`` AND by a
  bounded read, so a huge or growing response file is refused with a typed
  protocol error before any unbounded read/JSON/base64), **stderr captured**
  into events and typed errors (never swallowed; excerpts are read with a
  bounded tail read), a terminal event on **every** failure branch, and
  **every exit path reaps a live child before the scratch temp dir is
  removed** — including interrupted waiting; setup failures are typed and
  clean up.

Conservative stdlib-only choices, documented per the task brief: the worker
command is required (there is no default heavy worker at M1); both
``{request}`` and ``{response}`` placeholders are required so the protocol is
explicit; cancel is cooperative via an injected :class:`threading.Event`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qwen_engine._jsonsafe import (
    freeze_json,
    json_violations,
    plain_json,
    reject_constant,
)
from qwen_engine.errors import EngineError
from qwen_engine.log import PhaseLogger

# ---------------------------------------------------------------------------
# Versioned schema tags
# ---------------------------------------------------------------------------

REQUEST_SCHEMA = "qwen_engine.encoder_request.v2"
RESULT_SCHEMA = "qwen_engine.encoder_result.v1"
CACHE_KEY_SCHEMA = "qwen_engine.embed_cache_key.v3"
ENTRY_SCHEMA = "qwen_engine.embed_entry.v1"

#: The known source pins (PLAN M0.1, docs/dependencies.md). These are the
#: commits the pinned checkouts at ``/tmp/v100_m0_ml/src/*`` prove out; an
#: identity record must carry the versions/commits of the packages ACTUALLY
#: installed in the worker environment — checking them against these pins is
#: a later loader step, never an assumption baked in here.
PINNED_DIFFUSERS_COMMIT = "c6df88a511a98740646ee55577b590c9852650ce"
PINNED_TRANSFORMERS_COMMIT = "856157a2f3e9594954310df18fdccc31ffddebe9"

#: LRU cap on cached embeddings (plan R5).
DEFAULT_MAX_ENTRIES = 128

#: How much captured worker stderr is kept in error messages / events.
STDERR_EXCERPT_CHARS = 4000

#: Default cap on the bytes of a worker **response file** the runner will
#: read (:attr:`EncoderWorker.max_response_bytes`). A real result is one
#: JSON document whose base64 blobs are embedding tensors — a few MiB at
#: realistic shapes (e.g. 1024×8192 fp16 ≈ 16.8 MB → ≈ 22.4 MB as base64) —
#: so 256 MiB leaves several orders of magnitude of headroom while still
#: bounding what a runaway or hostile child can push into the engine's RAM
#: through one response file. The cap is enforced twice: a pre-read
#: ``stat`` refuses an already-oversized file before opening it, and the
#: read itself is bounded to ``cap + 1`` bytes, so a file that was small at
#: ``stat`` time but grows while read still cannot evade the limit.
DEFAULT_MAX_RESPONSE_BYTES = 256 * 1024 * 1024

#: Default cap on the bytes of one cache **entry file** the cache will read
#: or accept (:attr:`EmbeddingCache.max_entry_bytes`). An entry is a JSON
#: header line plus the raw blob bytes — the header's JSON overhead makes
#: an entry a different size from the base64-encoded worker response that
#: produced it, so reusing the same 256 MiB default as
#: :data:`DEFAULT_MAX_RESPONSE_BYTES` is a shared policy bound, not a
#: mathematical guarantee; it still bounds what a corrupt or hostile entry
#: file can pull into RAM: the entry is pre-sized on the opened descriptor
#: and read at most ``cap + 1`` bytes, and oversized content is never
#: parsed. A cap must additionally keep ``cap + 1`` representable as an
#: index (``<= sys.maxsize``), so ``read(cap + 1)`` can never raise a raw
#: ``OverflowError`` — same representability rule as the worker response cap.
DEFAULT_MAX_ENTRY_BYTES = 256 * 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_ENTRY_SUFFIX = ".embed"
_ENTRY_NAME_RE = re.compile(r"^[0-9a-f]{64}\.embed$")


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class EncoderError(EngineError):
    """Base for encoder worker/cache failures (exit 1 via the CLI contract).
    Every typed failure carries the captured worker ``stderr`` excerpt (empty
    when nothing was captured) so diagnostics survive into the CLI/log."""

    def __init__(self, message: str = "", *, stderr: str = "") -> None:
        self.stderr = stderr
        super().__init__(message)


class EncoderRequestError(EncoderError):
    """The encoder request (or its parts) is invalid; names every problem."""


class EncoderSetupError(EncoderError):
    """The worker cannot be launched as configured (bad command, spawn failure)."""


class EncoderTimeoutError(EncoderError):
    """The worker exceeded its timeout; it was terminated (then killed)."""


class EncoderCancelledError(EncoderError):
    """The caller cancelled via the cancel event; the worker was terminated
    (then killed) — not an error condition in itself, but a typed outcome."""


class EncoderWorkerError(EncoderError):
    """The worker failed: nonzero exit or an ``error`` result. Captured stderr
    is kept on :attr:`stderr` and woven into the message."""

    def __init__(self, message: str, *, exit_code: int | None,
                 stderr: str = "", worker_error: Mapping[str, Any] | None = None,
                 ) -> None:
        self.exit_code = exit_code
        self.worker_error = dict(worker_error) if worker_error else None
        super().__init__(message, stderr=stderr)


class EncoderProtocolError(EncoderError):
    """The worker's response file violated the result schema (missing file,
    bad JSON, wrong request id, unknown status, undecodable blob)."""


# ---------------------------------------------------------------------------
# Identity + request
# ---------------------------------------------------------------------------


def _err_list(errors: list[str], field_name: str, msg: str) -> None:
    errors.append(f"{field_name}: {msg}")


@dataclass(frozen=True)
class ComponentIdentity:
    """Identity of a text-processing component (processor / tokenizer / chat
    template): a pinned 40-hex ``revision``, a 64-hex content ``sha256``, or
    both — at least one is **required**, so the cache key never silently
    tracks a floating "main"."""

    revision: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        validate_identity_fields(self.revision, self.sha256, "identity", errors)
        if errors:
            raise EncoderRequestError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {"revision": self.revision, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: Any) -> "ComponentIdentity":
        errors: list[str] = []
        if not isinstance(data, Mapping):
            raise EncoderRequestError(
                f"identity: must be an object, got {type(data).__name__}")
        unknown = sorted(set(data) - {"revision", "sha256"})
        for key in unknown:
            _err_list(errors, key, "unknown identity key")
        revision = data.get("revision")
        sha = data.get("sha256")
        validate_identity_fields(revision, sha, "identity", errors)
        if errors:
            raise EncoderRequestError("; ".join(errors))
        return cls(revision=revision, sha256=sha)


def validate_identity_fields(revision: Any, sha: Any, at: str,
                             errors: list[str]) -> None:
    if revision is not None and (not isinstance(revision, str)
                                 or not _REVISION_RE.fullmatch(revision)):
        _err_list(errors, f"{at}.revision",
                  f"must be a pinned 40-hex commit sha or null, got {revision!r}")
    if sha is not None and (not isinstance(sha, str)
                            or not _SHA256_RE.fullmatch(sha)):
        _err_list(errors, f"{at}.sha256",
                  f"must be 64 lowercase hex chars or null, got {sha!r}")
    if revision is None and sha is None:
        _err_list(errors, at,
                  "at least one of revision/sha256 is required (a floating "
                  "component must not enter a cache key)")


_IDENTITY_FIELDS = (
    "encoder_artifact_set_sha256", "worker_protocol_revision",
    "worker_implementation_revision", "prompt_extraction_revision",
    "diffusers_commit", "transformers_commit", "torch_build", "torch_version",
    "safetensors_version", "torchvision_version",
)


def _nonempty_str(errors: list[str], at: str, value: Any, what: str) -> None:
    if value is _MISSING:
        return  # absent from a mapping: reported as missing by the caller
    if not isinstance(value, str) or not value.strip():
        _err_list(errors, at, f"must be a non-empty string ({what}), got {value!r}")
    elif value != value.strip():
        _err_list(errors, at, f"must not have surrounding whitespace, got {value!r}")


@dataclass(frozen=True)
class EncoderImplementationIdentity:
    """Immutable *implementation* identity of a production encoder worker.

    The cache key's model/processor/tokenizer/chat-template fields record
    *what data* is encoded; this record pins *which code* encodes it, so one
    key can never silently name two different encoder implementations:

    * ``encoder_artifact_set_sha256`` — the version-tagged artifact-set digest
      of the encoder + processor files (:mod:`qwen_engine.local_bundle`).
      This is the **asset component** of the key, never the whole key; a
      worker cache key additionally carries the revisions below.
    * ``worker_protocol_revision`` / ``worker_implementation_revision`` —
      wire-protocol revision and revision of the worker implementation code.
    * ``prompt_extraction_revision`` — revision of the prompt-formatting /
      hidden-state-extraction algorithm (the fidelity-critical path).
    * ``diffusers_commit`` / ``transformers_commit`` — exact 40-hex commits.
      Known source pins: :data:`PINNED_DIFFUSERS_COMMIT` /
      :data:`PINNED_TRANSFORMERS_COMMIT`; the values must come from the
      *installed* packages (a later loader step), not from these constants.
    * ``torch_build`` / ``torch_version`` — torch build flavour (e.g. the
      CUDA/CPU build tag) and version actually importable by the worker.
    * ``safetensors_version`` / ``torchvision_version`` — installed versions
      actually importable by the worker (torchvision is a known real-load
      gap: record what the environment check observed, e.g. an explicit
      "absent" marker — never a fabricated version and never silently omit
      the field).

    Every field is required and validated (64-hex / 40-hex where applicable);
    generic fake requests may simply carry no identity at all (``None`` on
    :class:`EncoderRequest`), but a future production worker must construct
    one from observed environment facts and refuse to run without it.
    """

    encoder_artifact_set_sha256: str
    worker_protocol_revision: str
    worker_implementation_revision: str
    prompt_extraction_revision: str
    diffusers_commit: str
    transformers_commit: str
    torch_build: str
    torch_version: str
    safetensors_version: str
    torchvision_version: str

    def __post_init__(self) -> None:
        errors: list[str] = []
        validate_implementation_identity_fields(self, "implementation_identity",
                                                errors)
        if errors:
            raise EncoderRequestError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _IDENTITY_FIELDS}

    @classmethod
    def from_dict(cls, data: Any) -> "EncoderImplementationIdentity":
        """Strict parse: every field required, unknown keys refused, all with
        the shared typed errors."""
        errors: list[str] = []
        if not isinstance(data, Mapping):
            raise EncoderRequestError(
                "implementation_identity: must be an object, got "
                f"{type(data).__name__}")
        for key in sorted(set(data) - set(_IDENTITY_FIELDS)):
            _err_list(errors, f"implementation_identity.{key}",
                      "unknown implementation-identity key")
        for name in _IDENTITY_FIELDS:
            if name not in data:
                _err_list(errors, f"implementation_identity.{name}",
                          "missing required field")
        validate_implementation_identity_fields(data, "implementation_identity",
                                                errors, missing_ok=True)
        if errors:
            raise EncoderRequestError("; ".join(errors))
        return cls(**{name: data[name] for name in _IDENTITY_FIELDS})


class _Missing:
    """Sentinel: field absent (already reported elsewhere)."""


_MISSING = _Missing()


def validate_implementation_identity_fields(obj: Any, at: str,
                                            errors: list[str], *,
                                            missing_ok: bool = False) -> None:
    """Collect every violation of :class:`EncoderImplementationIdentity`
    (attribute access on an instance, or key access on a mapping) into
    ``errors``. A field that is merely *absent* from a mapping is skipped
    when ``missing_ok`` (the parser reports it as missing); an explicit
    ``null`` is always a violation."""
    def get(name: str) -> Any:
        if isinstance(obj, Mapping):
            if name not in obj:
                return _MISSING if missing_ok else None
            return obj[name]
        value = getattr(obj, name, _MISSING)
        return value

    sha = get("encoder_artifact_set_sha256")
    if sha is not _MISSING and (not isinstance(sha, str)
                                or not _SHA256_RE.fullmatch(sha)):
        _err_list(errors, f"{at}.encoder_artifact_set_sha256",
                  f"must be 64 lowercase hex chars, got {sha!r}")
    for field_name in ("diffusers_commit", "transformers_commit"):
        value = get(field_name)
        if value is not _MISSING and (not isinstance(value, str)
                                      or not _REVISION_RE.fullmatch(value)):
            _err_list(errors, f"{at}.{field_name}",
                      f"must be an exact 40-hex commit sha, got {value!r}")
    _nonempty_str(errors, f"{at}.worker_protocol_revision",
                  get("worker_protocol_revision"), "wire protocol revision")
    _nonempty_str(errors, f"{at}.worker_implementation_revision",
                  get("worker_implementation_revision"),
                  "worker implementation revision")
    _nonempty_str(errors, f"{at}.prompt_extraction_revision",
                  get("prompt_extraction_revision"),
                  "prompt-extraction algorithm revision")
    _nonempty_str(errors, f"{at}.torch_build", get("torch_build"),
                  "torch build flavour")
    _nonempty_str(errors, f"{at}.torch_version", get("torch_version"),
                  "torch version")
    _nonempty_str(errors, f"{at}.safetensors_version",
                  get("safetensors_version"), "safetensors version")
    _nonempty_str(errors, f"{at}.torchvision_version",
                  get("torchvision_version"),
                  "torchvision version (an explicit absence marker is the "
                  "honest value where torchvision is not installed)")


@dataclass(frozen=True)
class EncoderRequest:
    """Everything that determines an encoding — the cache-key inputs live
    here, so an identical *result* is impossible from requests that differ in
    any of them (:func:`cache_key` hashes the whole document). Validated
    JSON values (``kwargs``) are deep-frozen snapshots taken at
    construction: the caller mutating its own (nested) mapping afterwards
    changes neither the cache key nor the wire document — they are computed
    from the same immutable snapshot. ``implementation_identity`` is itself a
    frozen record of validated strings, so it enjoys the same guarantee: no
    caller-side mutation can move the key or the wire document."""

    prompt: str
    dtype: str
    model_repo: str
    processor: ComponentIdentity
    tokenizer: ComponentIdentity
    chat_template: ComponentIdentity
    negative_prompt: str = ""
    model_revision: str | None = None
    model_sha256: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: Ordered image content hashes (future image conditioning; order matters).
    images: tuple[str, ...] = ()
    #: Optional production implementation identity (v3 key documents). Generic
    #: fake requests may omit it; a future production worker must set it.
    implementation_identity: EncoderImplementationIdentity | None = None

    def __post_init__(self) -> None:
        errors: list[str] = validate_request_fields(self)
        # Deep-freeze the kwargs snapshot *after* validation: the record owns
        # its own immutable copy, so mutating the caller's (possibly nested)
        # mapping afterwards can never change the cache key or the wire
        # document — one request, one key, one request file.
        if not errors:
            object.__setattr__(self, "kwargs",
                               freeze_json(self.kwargs, "kwargs", errors))
        if errors:
            raise EncoderRequestError("invalid encoder request: " + "; ".join(errors))

    def key_document(self) -> dict[str, Any]:
        """The canonical, key-sorted document that :func:`cache_key` hashes."""
        return {
            "schema": CACHE_KEY_SCHEMA,
            "model": {"repo": self.model_repo, "revision": self.model_revision,
                      "sha256": self.model_sha256},
            "processor": self.processor.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "chat_template": self.chat_template.to_dict(),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "dtype": self.dtype,
            "kwargs": plain_json(self.kwargs),
            "images": list(self.images),
            "implementation_identity": (
                None if self.implementation_identity is None
                else self.implementation_identity.to_dict()),
        }

    def to_request_dict(self, request_id: str) -> dict[str, Any]:
        doc = self.key_document()
        doc["schema"] = REQUEST_SCHEMA
        doc["request_id"] = request_id
        return doc

    @classmethod
    def from_request_dict(cls, data: Any) -> tuple["EncoderRequest", str]:
        """Parse a request document; returns ``(request, request_id)``.
        Strict: unknown keys, wrong types and unpinned identities are all
        reported together."""
        errors: list[str] = []
        if not isinstance(data, Mapping):
            raise EncoderRequestError(
                f"request: must be an object, got {type(data).__name__}")
        if data.get("schema") != REQUEST_SCHEMA:
            _err_list(errors, "schema",
                      f"must be {REQUEST_SCHEMA!r}, got {data.get('schema')!r}")
        known = {"schema", "request_id", "prompt", "negative_prompt", "dtype",
                 "model", "processor", "tokenizer", "chat_template", "kwargs",
                 "images", "implementation_identity"}
        for key in sorted(set(data) - known):
            _err_list(errors, key, "unknown request key")
        rid = data.get("request_id")
        if not isinstance(rid, str) or not rid.strip():
            _err_list(errors, "request_id", f"must be a non-empty string, got {rid!r}")
        model = data.get("model")
        model_revision = model_sha = None
        model_repo: Any = None
        if isinstance(model, Mapping):
            unknown = sorted(set(model) - {"repo", "revision", "sha256"})
            for key in unknown:
                _err_list(errors, f"model.{key}", "unknown model key")
            model_repo = model.get("repo")
            model_revision = model.get("revision")
            model_sha = model.get("sha256")
        else:
            _err_list(errors, "model", f"must be an object, got {model!r}")
        request = None
        components: dict[str, Any] = {}
        for comp in ("processor", "tokenizer", "chat_template"):
            value = data.get(comp)
            if not isinstance(value, Mapping):
                _err_list(errors, comp, f"must be an identity object, got {value!r}")
                continue
            comp_errors: list[str] = []
            unknown = sorted(set(value) - {"revision", "sha256"})
            for key in unknown:
                _err_list(comp_errors, f"{comp}.{key}", "unknown identity key")
            validate_identity_fields(value.get("revision"), value.get("sha256"),
                                     comp, comp_errors)
            if comp_errors:
                errors.extend(comp_errors)
                continue
            components[comp] = ComponentIdentity(revision=value.get("revision"),
                                                 sha256=value.get("sha256"))
        images = data.get("images", [])
        if not isinstance(images, list) or any(
                not isinstance(i, str) or not _SHA256_RE.fullmatch(i)
                for i in images):
            _err_list(errors, "images",
                      f"must be a list of 64-hex sha256 content hashes, got {images!r}")
        impl: EncoderImplementationIdentity | None = None
        if "implementation_identity" in data and data["implementation_identity"] is not None:
            impl_value = data["implementation_identity"]
            if not isinstance(impl_value, Mapping):
                _err_list(errors, "implementation_identity",
                          f"must be an object or null, got {type(impl_value).__name__}")
            else:
                try:
                    impl = EncoderImplementationIdentity.from_dict(impl_value)
                except EncoderRequestError as exc:
                    errors.append(str(exc))
        kwargs = data.get("kwargs", {})
        if not isinstance(kwargs, dict):
            _err_list(errors, "kwargs", f"must be an object, got {kwargs!r}")
            kwargs = {}
        if errors:
            raise EncoderRequestError("invalid encoder request: " + "; ".join(errors))
        try:
            request = cls(
                prompt=data.get("prompt"),
                negative_prompt=data.get("negative_prompt", ""),
                dtype=data.get("dtype"),
                model_repo=model_repo,
                model_revision=model_revision,
                model_sha256=model_sha,
                processor=components["processor"],
                tokenizer=components["tokenizer"],
                chat_template=components["chat_template"],
                kwargs=kwargs,
                images=tuple(images),
                implementation_identity=impl,
            )
        except EncoderRequestError as exc:
            raise EncoderRequestError(f"invalid encoder request: {exc}") from exc
        return request, rid


def _nonempty(errors: list[str], name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        _err_list(errors, name, f"must be a non-empty string, got {value!r}")


def validate_request_fields(req: EncoderRequest) -> list[str]:
    errors: list[str] = []
    _nonempty(errors, "prompt", req.prompt)
    if not isinstance(req.negative_prompt, str):
        _err_list(errors, "negative_prompt",
                  f"must be a string, got {req.negative_prompt!r}")
    _nonempty(errors, "dtype", req.dtype)
    if isinstance(req.dtype, str) and req.dtype != req.dtype.strip():
        _err_list(errors, "dtype", f"must not have surrounding whitespace, got {req.dtype!r}")
    _nonempty(errors, "model_repo", req.model_repo)
    validate_identity_fields(req.model_revision, req.model_sha256, "model", errors)
    for comp in ("processor", "tokenizer", "chat_template"):
        value = getattr(req, comp)
        if not isinstance(value, ComponentIdentity):
            _err_list(errors, comp, f"must be a ComponentIdentity, got "
                                    f"{type(value).__name__}")
    if req.implementation_identity is not None and not isinstance(
            req.implementation_identity, EncoderImplementationIdentity):
        _err_list(errors, "implementation_identity",
                  f"must be an EncoderImplementationIdentity or None, got "
                  f"{type(req.implementation_identity).__name__}")
    if not isinstance(req.kwargs, Mapping):
        _err_list(errors, "kwargs",
                  f"must be a mapping, got {type(req.kwargs).__name__}")
    else:
        # Strict JSON: string keys, finite numbers, supported types only —
        # json.dumps alone would silently accept NaN and int keys.
        errors.extend(json_violations(req.kwargs, "kwargs"))
    if isinstance(req.images, (str, bytes)) or not isinstance(req.images, tuple):
        _err_list(errors, "images", f"must be a tuple of sha256 strings, got {req.images!r}")
    else:
        for i, img in enumerate(req.images):
            if not isinstance(img, str) or not _SHA256_RE.fullmatch(img):
                _err_list(errors, f"images[{i}]",
                          f"must be a 64-hex content sha256, got {img!r}")
    return errors


def cache_key(request: EncoderRequest) -> str:
    """Cache key v3 (R5): sha256 over the canonical key document. The
    negative prompt is part of the key (the LTX negative-key fix), as are all
    kwargs, dtype, component identities, ordered image hashes, and the
    optional production implementation identity (v3 — distinct encoder
    implementations can never share a key)."""
    doc = json.dumps(request.key_document(), sort_keys=True, ensure_ascii=True,
                     separators=(",", ":"))
    return hashlib.sha256(doc.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Payload + entry format (tensor-ready without importing torch)
# ---------------------------------------------------------------------------


class _EntryCorrupt(Exception):
    """Internal: an entry (or response payload) violates the blob format."""


def _entry_header_line(payload: "EncodedPayload", key: str) -> bytes:
    """The canonical JSON header line (no terminator) of an entry. Kept
    separate from the blob concatenation so a size check can measure the
    exact entry size BEFORE the combined buffer is ever constructed."""
    header = {
        "schema": ENTRY_SCHEMA,
        "key": key,
        "metadata": plain_json(payload.metadata),
        "blobs": [{"name": b.name, "bytes": len(b.data), "sha256": b.sha256}
                  for b in payload.blobs],
    }
    return json.dumps(header, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class Blob:
    """An opaque named byte payload. The engine never interprets blob bytes —
    tensor payloads arrive later from the worker side, so the cache needs no
    torch to store or validate them."""

    name: str
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise EncoderRequestError(f"blob name: must be non-empty, got {self.name!r}")
        if isinstance(self.data, bytearray):
            # Immutable snapshot: mutating the caller's bytearray afterwards
            # must not change the blob's bytes, sha256 or entry layout.
            object.__setattr__(self, "data", bytes(self.data))
        if not isinstance(self.data, bytes):
            raise EncoderRequestError(
                f"blob {self.name!r} data: must be bytes, got {type(self.data).__name__}")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(bytes(self.data)).hexdigest()


@dataclass(frozen=True)
class EncodedPayload:
    """What the encoder produces: JSON metadata + ordered opaque blobs."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    blobs: tuple[Blob, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping):
            raise EncoderRequestError(
                f"payload metadata must be a mapping, got "
                f"{type(self.metadata).__name__}")
        errors: list[str] = []
        snapshot = freeze_json(self.metadata, "metadata", errors)
        if isinstance(self.blobs, (str, bytes)) or not isinstance(self.blobs, tuple):
            _err_list(errors, "blobs", f"must be a tuple of Blob, got "
                                       f"{type(self.blobs).__name__}")
        else:
            bad = [f"blobs[{i}]: {type(b).__name__}"
                   for i, b in enumerate(self.blobs) if not isinstance(b, Blob)]
            errors.extend(bad)
            names = [b.name for b in self.blobs if isinstance(b, Blob)]
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                _err_list(errors, "blobs",
                          f"duplicate blob names: {', '.join(dupes)}")
        if errors:
            raise EncoderRequestError("invalid encoded payload: "
                                      + "; ".join(errors))
        object.__setattr__(self, "metadata", snapshot)

    def to_entry_bytes(self, key: str) -> bytes:
        """Canonical entry bytes: one JSON header line + raw blob payloads."""
        return (_entry_header_line(self, key) + b"\n"
                + b"".join(bytes(b.data) for b in self.blobs))

    @classmethod
    def from_entry_bytes(cls, data: bytes, key: str) -> "EncodedPayload":
        """Strict read of :meth:`to_entry_bytes` output; raises
        :class:`_EntryCorrupt` on anything that does not verify exactly."""
        if not isinstance(data, (bytes, bytearray)):
            raise _EntryCorrupt("entry is not bytes")
        head, sep, body = bytes(data).partition(b"\n")
        if not sep:
            raise _EntryCorrupt("no header line terminator")
        try:
            header = json.loads(head.decode("utf-8"),
                                parse_constant=reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise _EntryCorrupt(f"bad header: {exc}") from exc
        if not isinstance(header, Mapping):
            raise _EntryCorrupt(f"header must be an object, got {type(header).__name__}")
        unknown = sorted(k for k in header if not isinstance(k, str))
        unknown += sorted(str(k) for k in set(header) - {"schema", "key",
                                                         "metadata", "blobs"}
                          if isinstance(k, str))
        if unknown:
            raise _EntryCorrupt(
                f"unknown entry header keys: {', '.join(unknown)}")
        if header.get("schema") != ENTRY_SCHEMA:
            raise _EntryCorrupt(f"unknown entry schema {header.get('schema')!r}")
        if header.get("key") != key:
            raise _EntryCorrupt(
                f"entry key {header.get('key')!r} does not match its file name")
        declared = header.get("blobs")
        if not isinstance(declared, list):
            raise _EntryCorrupt("blobs must be a list")
        blobs: list[Blob] = []
        offset = 0
        for i, spec in enumerate(declared):
            if (not isinstance(spec, Mapping) or not isinstance(spec.get("name"), str)
                    or not spec["name"].strip()
                    or not isinstance(spec.get("bytes"), int)
                    or isinstance(spec.get("bytes"), bool) or spec["bytes"] < 0
                    or not isinstance(spec.get("sha256"), str)
                    or not _SHA256_RE.fullmatch(spec["sha256"])):
                raise _EntryCorrupt(f"blob[{i}] descriptor is malformed: {spec!r}")
            extra = sorted(str(k) for k in set(spec) - {"name", "bytes", "sha256"})
            if extra:
                raise _EntryCorrupt(
                    f"blob[{i}] descriptor has unknown keys: {', '.join(extra)}")
            end = offset + spec["bytes"]
            chunk = bytes(body[offset:end])
            if len(chunk) != spec["bytes"]:
                raise _EntryCorrupt(
                    f"blob {spec['name']!r} truncated: want {spec['bytes']} bytes, "
                    f"found {len(chunk)}")
            if hashlib.sha256(chunk).hexdigest() != spec["sha256"]:
                raise _EntryCorrupt(f"blob {spec['name']!r} sha256 mismatch")
            blobs.append(Blob(name=spec["name"], data=chunk))
            offset = end
        if offset != len(body):
            raise _EntryCorrupt(
                f"{len(body) - offset} trailing bytes after the last blob")
        metadata = header.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise _EntryCorrupt("metadata must be an object")
        try:
            return cls(metadata=dict(metadata), blobs=tuple(blobs))
        except EncoderRequestError as exc:
            raise _EntryCorrupt(str(exc)) from exc


# ---------------------------------------------------------------------------
# Embedding cache (atomic writes, quarantine, per-key locks, LRU cap)
# ---------------------------------------------------------------------------


def _canonicalize_dir(path: Any, field: str) -> Path:
    """Resolve a configured directory path **once** at construction into a
    canonical **absolute** path, stored verbatim for BOTH file I/O and the
    lock registry (so the I/O target and the mutation lock can never
    disagree): later CWD changes cannot redirect a cache/scratch write, and
    retargeting an explicitly chosen root symlink afterwards cannot move
    the writes — or the registry entry — to a different root. The original
    string is rejected when empty/whitespace *before* ``Path`` conversion
    turns ``""`` into ``.``; NUL bytes and non-path types keep their typed
    errors. Nothing is created here: no mkdir at construction; no
    hostile-concurrent-filesystem safety is claimed (resolution is a
    one-time snapshot, not a continuous guarantee)."""
    if not isinstance(path, (str, os.PathLike)):
        raise EncoderRequestError(f"{field}: must be a path string or "
                                  f"os.PathLike, got {type(path).__name__}")
    raw = os.fsdecode(path)
    if not raw.strip():
        raise EncoderRequestError(f"{field}: must not be blank")
    if "\0" in raw:
        raise EncoderRequestError(f"{field}: must not contain a NUL byte")
    return Path(os.path.realpath(raw))


def _contained_under(root: Path, path: Path) -> bool:
    """True when ``path`` is ``root`` itself or lives under it after full
    symlink resolution (both sides canonicalized, so a root reached through
    an explicitly chosen symlink stays consistent)."""
    try:
        real_root = os.path.realpath(root)
        real_path = os.path.realpath(path)
    except OSError:  # pragma: no cover - exotic path failures
        return False
    return real_path == real_root or real_path.startswith(real_root + os.sep)


_REGISTRY_GUARD = threading.Lock()
_ROOT_STATES: dict[str, "_RootState"] = {}


class _RootState:
    """In-process state shared by **every** cache instance pointing at one
    canonical root: a cache-wide mutex covering all mutations (publish,
    quarantine, LRU) and the per-key encode locks."""

    def __init__(self) -> None:
        self.mutex = threading.RLock()   # mutations / quarantine / LRU
        self.keys_guard = threading.Lock()  # guards the key_locks table only
        self.key_locks: dict[str, threading.Lock] = {}


def _entry_identity(st: os.stat_result) -> tuple:
    """Identity of one on-disk entry snapshot: enough stat fields to prove a
    later ``stat`` describes the *same* file that was read (an ``os.replace``
    publish produces a different inode)."""
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_mode)


def _owned_by_current_user(st: os.stat_result) -> bool:
    """Ownership rule on an ALREADY-STATED snapshot: a cache entry must
    belong to the effective user running us. Applied to the lstat of the
    path (miss decisions, quarantine, LRU) and to the fstat of an opened
    descriptor, so no generic OSError routing can undo the refusal."""
    return not (hasattr(os, "geteuid") and st.st_uid != os.geteuid())


def _lru_evictable(st: os.stat_result) -> bool:
    """LRU-safe name (from a NO-FOLLOW stat): regular, singly linked and
    owned by the current user — exactly the shape an atomic cache file of
    this cache has. Anything else neither counts toward the cap nor may
    be deleted as an eviction."""
    return (stat.S_IFMT(st.st_mode) == stat.S_IFREG and st.st_nlink == 1
            and _owned_by_current_user(st))


def _root_state(root: Path) -> _RootState:
    try:
        canonical = str(root.resolve())
    except OSError:  # pragma: no cover - exotic path failures
        canonical = str(Path(os.path.abspath(root)))
    with _REGISTRY_GUARD:
        state = _ROOT_STATES.get(canonical)
        if state is None:
            state = _ROOT_STATES[canonical] = _RootState()
        return state


class EmbeddingCache:
    """Disk cache of :class:`EncodedPayload` under ``root``.

    * **Ownership**: the configured root is resolved once to a canonical
      absolute path at construction, used verbatim for I/O and the lock
      registry (CWD-stable; a later root-symlink retarget cannot redirect
      the writes or move the lock), and the cache only ever reads, touches
      or
      moves files *it owns under that root*: an entry leaf that is a
      symlink, has more than one hard link or is not a regular file is
      refused — never read, never mtime-touched (atomic cache files
      have single ownership), only moved aside by name; a *foreign-owned
      regular* entry is likewise refused as a miss, but its name and
      bytes are **preserved** — never read, touched or renamed, in
      :meth:`get` and in the quarantine path alike (ownership is
      rechecked on the lstat and on the opened descriptor, so generic
      OSError routing cannot undo that refusal).
      Quarantine is refused unless ``root/quarantine`` is a real directory
      contained under the canonical root; a refused move leaves the
      evidence exactly where it is and still reports a miss. The LRU touch
      never follows symlinks. FIFOs and other nonregular squats cannot
      block a read (opened ``O_NONBLOCK``, validated on the opened
      descriptor). Ordinary single-writer hygiene: no hostile-concurrent
      filesystem safety is claimed.
    * **Bounded reads**: an entry is pre-sized on the opened descriptor and
      read at most ``max_entry_bytes + 1`` bytes — before any parse, and
      again in the quarantine re-check (never an unbounded second read);
      oversized content is a corrupt miss and is never parsed. A
      :meth:`put` whose entry would exceed the cap fails *before* the
      combined entry buffer is constructed.
    * Atomic publish (temp file + ``os.replace``) — readers never see partial
      entries, which is also the cross-process concurrency story; within one
      process :meth:`lock_for` serialises same-key work.
    * All mutations (publish, quarantine, LRU) run under a **cache-wide
      in-process mutex shared by every instance on the same canonical root**
      — two instances (or two concurrent puts) cannot interleave an LRU
      snapshot with another eviction and empty the cache. Per-key encode
      locks are shared the same way, so two instances on one root never
      double-encode the same key.
    * A corrupt / unreadable entry is a **miss**; the bad file is moved to
      ``root/quarantine/``, never silently overwritten in place. The move
      happens under the root mutex and only after an **identity check**
      proves the file is still the exact one that read corrupt — a valid
      entry published in the meantime is never quarantined.
    * LRU cap by mtime (hits refresh mtime); eviction is a plain delete.
      Only regular, singly linked, current-user-owned entries are counted
      toward the cap or evicted — counted, ordered and re-checked by a
      no-follow stat — so an unsafe entry-shaped name never uses up the
      valid-entry budget and is never deleted as an eviction (an external
      inode's nlink can never be decremented through the LRU).
    """

    def __init__(self, root: str | Path, *, max_entries: int = DEFAULT_MAX_ENTRIES,
                 max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES) -> None:
        self.root = _canonicalize_dir(root, "cache root")
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) \
                or max_entries < 1:
            raise EncoderRequestError(
                f"max_entries: must be a positive integer, got {max_entries!r}")
        if isinstance(max_entry_bytes, bool) or not isinstance(
                max_entry_bytes, int) or max_entry_bytes < 1:
            raise EncoderRequestError(
                f"max_entry_bytes: must be a positive integer, got "
                f"{max_entry_bytes!r}")
        if max_entry_bytes + 1 > sys.maxsize:
            # read(cap + 1) must stay representable as an index-sized
            # integer (the same rule the worker response cap follows):
            # 10**100 is a typed configuration refusal, never a raw
            # OverflowError at the bounded read.
            raise EncoderRequestError(
                f"max_entry_bytes: {max_entry_bytes!r} is not a representable "
                f"bound (cap + 1 must be <= sys.maxsize = {sys.maxsize})")
        self.max_entries = max_entries
        self.max_entry_bytes = max_entry_bytes
        self._state = _root_state(self.root)

    # -- paths ---------------------------------------------------------------

    @property
    def entries_dir(self) -> Path:
        return self.root

    @property
    def quarantine_dir(self) -> Path:
        return self.root / "quarantine"

    def entry_path(self, key: str) -> Path:
        if not isinstance(key, str) or not _ENTRY_NAME_RE.fullmatch(key + _ENTRY_SUFFIX):
            raise EncoderRequestError(f"cache key: must be 64 lowercase hex chars, got {key!r}")
        return self.root / (key + _ENTRY_SUFFIX)

    def lock_for(self, key: str) -> threading.Lock:
        if not isinstance(key, str) or not _SHA256_RE.fullmatch(key):
            raise EncoderRequestError(f"cache key: must be 64 lowercase hex chars, got {key!r}")
        # Shared across every instance on this canonical root: two caches on
        # one root serialise on the same lock, so the same key is never
        # encoded twice in one process.
        with self._state.keys_guard:
            return self._state.key_locks.setdefault(key, threading.Lock())

    # -- ops -----------------------------------------------------------------

    def get(self, key: str) -> EncodedPayload | None:
        """Return the cached payload, or ``None`` on miss.

        Ownership first: the entry name is examined with ``lstat`` and only
        a *regular, singly linked* file is opened at all — a symlink, hard
        link, or nonregular squat is refused as a miss and moved aside by
        name, never read and never mtime-touched, so an external file can
        neither answer through a cache path nor have its bytes / link /
        mtime changed through one. The opened descriptor must likewise be
        regular, singly linked and owned (O_NOFOLLOW + O_NONBLOCK + fstat:
        no FIFO block, no symlink follow, no read of a foreign file). The
        read is bounded (pre-size, then ``cap + 1``); oversized content is
        a corrupt miss and is never parsed. A corrupt entry is a miss
        **and** gets quarantined — but only *that same file*: the
        quarantine moves the entry aside only after an identity check
        under the cache-wide root mutex, so a valid entry published after a
        stale corrupt read is never quarantined (a confirmed race otherwise
        moved the NEW entry into quarantine). The LRU touch likewise runs
        under the mutex, next to the LRU eviction it races with, and never
        follows symlinks. Lock order stays key-lock → root-mutex (never
        the reverse)."""
        path = self.entry_path(key)
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return None
        except OSError:
            return self._quarantine(path, key, read_ident=None)
        if stat.S_IFMT(st.st_mode) != stat.S_IFREG or st.st_nlink != 1:
            # Symlinked / multiply linked / nonregular squat: refuse to
            # read or touch (an atomic cache file is a singly linked
            # regular file); move the NAME aside — the move follows
            # nothing and cannot modify any external target.
            return self._quarantine(path, key, read_ident=None)
        if not _owned_by_current_user(st):
            # Foreign-owned regular entry: a MISS with no open, no read,
            # no touch and no rename — the evidence and the name stay
            # exactly where they are. Checked on the lstat BEFORE the
            # descriptor, so the OSError path below can never route this
            # refusal into a quarantine move.
            return None
        read_ident: tuple | None = None
        try:
            data, read_ident = self._bounded_entry_read(path)
        except OSError:
            return self._quarantine(path, key, read_ident=None)
        if data is None:
            # Over the cap (at pre-size or by the bounded read): a corrupt
            # miss — the oversized bytes were never read whole, never
            # parsed.
            return self._quarantine(path, key, read_ident=read_ident)
        try:
            payload = EncodedPayload.from_entry_bytes(data, key)
        except _EntryCorrupt:
            return self._quarantine(path, key, read_ident=read_ident)
        with self._state.mutex:  # LRU touch serialised with eviction
            try:
                # follow_symlinks=False: a hit can never refresh the mtime
                # of anything reached through a link (no-follow by the same
                # rule as the read above).
                os.utime(path, None, follow_symlinks=False)
            except OSError:
                pass  # a failed touch must not lose a hit
        return payload

    def put(self, key: str, payload: EncodedPayload) -> Path:
        """Atomically publish ``payload`` under ``key``, then enforce the LRU
        cap (the just-written entry is protected). Publish and eviction are
        one transaction under the cache-wide root mutex: concurrent puts can
        never snapshot an entry list and then evict each other's entries.
        The size bound is enforced on the *declared* size (header line plus
        blob lengths) BEFORE the combined buffer is constructed, so a put
        that exceeds ``max_entry_bytes`` fails typed without ever building
        the huge payload buffer."""
        path = self.entry_path(key)
        data = self._entry_bytes_bounded(payload, key)
        with self._state.mutex:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            try:
                with tmp.open("xb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                raise EncoderError(f"cache write failed for {key}: {exc}") from exc
            self._enforce_lru(protect=key)
        return path

    def entries(self) -> list[str]:
        """The keys whose entry-shaped NAMES are present under the root.
        This is a name listing only — no content, kind, link or ownership
        validation is claimed or performed (quarantine-refused and LRU-
        unsafe squats list too, and refused-corrupt files are left in
        place, not repaired); only :meth:`get` validates an entry."""
        if not self.root.is_dir():
            return []
        keys = []
        for entry in sorted(self.root.iterdir()):
            if _ENTRY_NAME_RE.fullmatch(entry.name):
                keys.append(entry.name[: -len(_ENTRY_SUFFIX)])
        return keys

    # -- ownership + bounds internals -----------------------------------------

    def _bounded_entry_read(self, path: Path) -> tuple[bytes | None, tuple]:
        """Safely open one entry and read it BOUNDED by
        ``max_entry_bytes``. Returns ``(data, identity)``; ``data`` is
        ``None`` when the entry is over the cap — pre-sized on the opened
        descriptor, then read at most ``cap + 1`` bytes, so an entry that
        lies about its size is still caught and oversized content is
        never read whole or parsed. Raises ``OSError`` when the path cannot
        be opened safely (symlink swap → ELOOP, nonregular / multiply
        linked / foreign-owned descriptor → refused before any read)."""
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK
        fd = os.open(path, flags)
        try:
            st = os.fstat(fd)
            if stat.S_IFMT(st.st_mode) != stat.S_IFREG:
                raise OSError(f"entry is not a regular file: {path}")
            if st.st_nlink != 1:
                raise OSError(f"entry has {st.st_nlink} links (not singly "
                              f"owned): {path}")
            if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
                raise OSError(f"entry is not owned by this user: {path}")
            cap = self.max_entry_bytes
            if st.st_size > cap:
                return None, _entry_identity(st)  # pre-size refusal, no read
            chunks: list[bytes] = []
            got = 0
            while got < cap + 1:  # bounded: at most cap + 1 bytes, ever
                chunk = os.read(fd, min(cap + 1 - got, 1 << 20))
                if not chunk:
                    break
                chunks.append(chunk)
                got += len(chunk)
            data = b"".join(chunks) if got <= cap else None
            return data, _entry_identity(os.fstat(fd))
        finally:
            os.close(fd)

    def _quarantine_dir_ok(self) -> bool:
        """True only when ``root/quarantine`` IS a real directory contained
        under the canonical root. An external symlink (or any other wrong
        kind) squatting on the name is refused — never mkdir'd through,
        never moved into — so a quarantine redirection can never relocate
        cache evidence outside the configured root."""
        qd = self.quarantine_dir
        try:
            st = os.lstat(qd)
        except FileNotFoundError:
            try:
                qd.mkdir(parents=True, exist_ok=True)
                st = os.lstat(qd)
            except OSError:
                return False
        except OSError:
            return False
        if os.path.islink(qd) or not stat.S_ISDIR(st.st_mode):
            return False
        return _contained_under(self.root, qd)

    def _entry_bytes_bounded(self, payload: EncodedPayload, key: str) -> bytes:
        """Entry bytes with the cap enforced on the DECLARED size (header
        line + blob lengths, no concatenation), so an over-cap put fails
        typed before a huge combined buffer exists."""
        line = _entry_header_line(payload, key)
        total = len(line) + 1 + sum(len(b.data) for b in payload.blobs)
        if total > self.max_entry_bytes:
            raise EncoderError(
                f"cache entry for {key} would be {total} bytes, over "
                f"max_entry_bytes={self.max_entry_bytes}; refusing before "
                f"constructing the entry buffer")
        return line + b"\n" + b"".join(bytes(b.data) for b in payload.blobs)

    def _quarantine(self, path: Path, key: str, *,
                    read_ident: tuple | None) -> None:
        """Move a corrupt entry aside under the cache-wide root mutex.

        ``read_ident`` is the identity of the bytes that were actually read
        and found corrupt; when it is not ``None`` the file is quarantined
        ONLY if it still carries that identity — if a valid entry was
        published in the meantime (different inode), the NEW entry stands
        and this read simply reports a miss. With ``read_ident=None`` (an
        ownership refusal, an entry-name squat, or a failed read) whatever
        currently sits at the path is re-checked: a singly linked, owned
        regular file is only quarantined after a fresh BOUNDED parse proves
        it corrupt — never a freshly published valid entry; symlinked,
        multiply linked or nonregular names are moved aside unread (the
        move follows nothing and modifies no external target). Before ANY
        move the quarantine directory itself must pass
        :meth:`_quarantine_dir_ok`; a refused or failed move leaves the
        evidence in place and still reports a miss.
        """
        with self._state.mutex:  # quarantine is a mutation too
            try:
                st = os.lstat(path)
            except FileNotFoundError:
                return None  # vanished: miss, nothing to move
            except OSError:
                return None
            if read_ident is not None:
                if _entry_identity(st) != read_ident:
                    return None  # replaced after the corrupt read
            elif (stat.S_IFMT(st.st_mode) == stat.S_IFREG
                    and st.st_nlink == 1):
                if not _owned_by_current_user(st):
                    # Foreign-owned regular entry: the evidence and the
                    # name are PRESERVED, never moved — the ownership
                    # refusal is never undone on this path either.
                    return None
                # Re-check the CURRENT bytes before quarantining, bounded:
                # only content that is actually corrupt (or unreadable /
                # over the cap) may be moved — never a valid entry.
                try:
                    data, _ = self._bounded_entry_read(path)
                    if data is not None:
                        EncodedPayload.from_entry_bytes(data, key)
                        return None  # parses valid after all — leave it
                except (_EntryCorrupt, OSError):
                    pass  # corrupt / unreadable / over cap: quarantine below
            if not self._quarantine_dir_ok():
                # Evidence preserved where a move is unsafe (redirected or
                # unusable quarantine directory): miss, nothing moved.
                return None
            try:
                target = self.quarantine_dir / (
                    f"{key}.{os.getpid()}.{time.time_ns()}.corrupt")
                os.replace(path, target)
            except OSError:
                # Quarantine itself failed: still a miss; leave the file for
                # the next reader rather than deleting operator-visible
                # evidence.
                pass
        return None

    def _enforce_lru(self, protect: str) -> None:
        """Enforce the cap over the VALID entries only. Counting, mtime
        ordering and eviction all use a NO-FOLLOW stat: only regular,
        singly linked, current-user-owned names are valid entries — an
        unsafe entry-shaped name (symlink, extra hard link, nonregular or
        foreign-owned squat) neither uses up the valid-entry budget nor is
        ever deleted as an eviction, so an external inode's link count,
        bytes and mtime can never be decremented through the LRU. The
        shape is re-checked by lstat right before each unlink."""
        candidates: list[tuple[float, str, Path]] = []
        for key in self.entries():
            path = self.root / (key + _ENTRY_SUFFIX)
            try:
                st = os.lstat(path)
            except OSError:
                continue  # vanished: no budget, nothing to evict
            if not _lru_evictable(st):
                continue  # unsafe squat: no budget, never evicted
            candidates.append((st.st_mtime, key, path))
        if len(candidates) <= self.max_entries:
            return

        candidates.sort(key=lambda item: (item[0], item[1]))
        overflow = len(candidates) - self.max_entries
        for _, key, path in candidates:
            if overflow <= 0:
                break
            if key == protect:
                continue
            try:
                st = os.lstat(path)
            except OSError:
                continue  # vanished since the snapshot: no eviction spent
            if not _lru_evictable(st):
                continue  # went unsafe: no eviction spent
            try:
                path.unlink(missing_ok=True)
                overflow -= 1
            except OSError:
                pass


def embedding_cache_for(config: Any) -> EmbeddingCache | None:
    """Cache for a :class:`~qwen_engine.config.GenerationConfig`:
    ``<cache_dir>/embeds`` (plan: ``cache/embeds/``), or ``None`` when
    ``cache_dir`` is unconfigured — unconfigured means **nothing is written**
    (R7: no implicit default write targets)."""
    if config.cache_dir is None:
        return None
    return EmbeddingCache(Path(config.cache_dir) / "embeds")


# ---------------------------------------------------------------------------
# Worker runner
# ---------------------------------------------------------------------------


def _finite_float(value: Any) -> float | None:
    """Float coercion that never raises: a huge int (``10**400``) overflows
    at conversion — a configuration error, not an ``OverflowError`` escaping
    the constructor. Non-finite floats (NaN/Inf) are likewise rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        f = float(value)
    except (OverflowError, ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def _excerpt(text: str) -> str:
    if len(text) <= STDERR_EXCERPT_CHARS:
        return text
    return "...(truncated)...\n" + text[-STDERR_EXCERPT_CHARS:]


#: Upper bound on bytes *read* per log file: the tail of a huge log is read
#: by seeking, so an arbitrarily large log never lands in RAM.
_EXCERPT_READ_BYTES = 4 * STDERR_EXCERPT_CHARS


def _read_excerpt(path: Path) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size > _EXCERPT_READ_BYTES:
                fh.seek(size - _EXCERPT_READ_BYTES)
            else:
                fh.seek(0)
            data = fh.read()
    except OSError:
        return ""
    return _excerpt(data.decode("utf-8", errors="replace"))


class EncoderWorker:
    """Run the encoder as a throwaway child process with cache-first lookup.

    ``command`` tokens may contain ``{python}`` (the interpreter),
    ``{request}`` / ``{response}`` (scratch-file paths) placeholders; both
    path placeholders are required so the wire protocol is explicit. On
    timeout or cancellation the child gets ``terminate()`` then, after
    ``cancel_grace_s``, ``kill()`` — never an orphaned worker. Captured stderr
    goes into events and into every typed failure; scratch files are removed
    on every path. The response file is read only under the explicit
    :attr:`max_response_bytes` cap (stat pre-check + bounded read; oversized
    or growing files raise :class:`EncoderProtocolError` before any
    unbounded read/JSON/base64 — no new transport, no pickle).
    (Pinned-host resources are an M4 concern of the real worker; this runner
    owns only its temp files.)
    """

    def __init__(self, *, command: Sequence[str], timeout_s: float,
                 cache: EmbeddingCache | None = None,
                 scratch_dir: str | Path | None = None,
                 cancel_grace_s: float = 5.0,
                 python: str | None = None,
                 max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
                 logger: PhaseLogger | None = None) -> None:
        cmd = list(command) if (not isinstance(command, (str, bytes))
                                and isinstance(command, Sequence)) else None
        if cmd is None:
            raise EncoderSetupError(
                f"command: must be a non-empty list of non-empty strings, got {command!r}")
        if not cmd or any(not isinstance(tok, str) or not tok for tok in cmd):
            raise EncoderSetupError(
                f"command: must be a non-empty list of non-empty strings, got {command!r}")
        joined = " ".join(cmd)
        for needed in ("{request}", "{response}"):
            if needed not in joined:
                raise EncoderSetupError(
                    f"command: token {needed!r} is required (the worker protocol "
                    f"passes the {needed.strip('{}')} file by path)")
        timeout = _finite_float(timeout_s)
        if timeout is None or timeout <= 0:
            # NaN/Inf must not disable the timeout (a NaN deadline makes
            # every `<= 0` check false: the poll loop would wait forever),
            # and a huge int overflows at float() — both typed setup errors.
            raise EncoderSetupError(
                f"timeout_s: must be a finite number > 0, got {timeout_s!r}")
        grace = _finite_float(cancel_grace_s)
        if grace is None or grace < 0:
            raise EncoderSetupError(
                f"cancel_grace_s: must be a finite number >= 0, "
                f"got {cancel_grace_s!r}")
        self.command = cmd
        self.timeout_s = timeout
        self.cache = cache
        self.python = python if python is not None else (sys.executable or "python3")
        if not isinstance(self.python, str) or not self.python:
            raise EncoderSetupError(
                f"python: must be a non-empty string or null, got {python!r}")
        self.scratch_dir = str(_canonicalize_dir(
            scratch_dir if scratch_dir is not None else tempfile.gettempdir(),
            "scratch_dir"))
        self.cancel_grace_s = grace
        if isinstance(max_response_bytes, bool) or not isinstance(
                max_response_bytes, int) or max_response_bytes < 1:
            # A float/bool/zero/negative cap would silently disable or break
            # the response-size bound (and 1<<64 is a legitimate int); only a
            # positive int is a meaningful byte count.
            raise EncoderSetupError(
                f"max_response_bytes: must be a positive integer, got "
                f"{max_response_bytes!r}")
        self.max_response_bytes = max_response_bytes
        self.logger = logger or PhaseLogger()

    # -- public --------------------------------------------------------------

    def encode(self, request: EncoderRequest, *, request_id: str | None = None,
               cancel_event: threading.Event | None = None) -> EncodedPayload:
        """Cache-first encode. Same-key callers within one process serialise
        on the (root-shared) key lock, so a miss cannot double-encode;
        cross-process safety comes from atomic publishing. The request is
        immutable from construction, so this key and the request file
        :meth:`run` writes describe exactly the same request."""
        key = cache_key(request)
        if self.cache is None:
            return self.run(request, request_id=request_id, cancel_event=cancel_event)
        with self.cache.lock_for(key):
            hit = self.cache.get(key)
            if hit is not None:
                self._event("encoder_cache_hit", cache_key=key)
                return hit
            self._event("encoder_cache_miss", cache_key=key)
            payload = self.run(request, request_id=request_id,
                               cancel_event=cancel_event)
            self.cache.put(key, payload)
            return payload

    def run(self, request: EncoderRequest, *, request_id: str | None = None,
            cancel_event: threading.Event | None = None) -> EncodedPayload:
        """One worker process, no cache. Raises a typed :class:`EncoderError`
        subclass on every failure path; **every exit path** — success, typed
        failure, unexpected exception, ``KeyboardInterrupt`` — reaps a live
        child before the scratch dir is removed."""
        rid = request_id or uuid.uuid4().hex
        key = cache_key(request)
        try:
            scratch = Path(tempfile.mkdtemp(prefix="qwen-enc-",
                                            dir=self.scratch_dir))
        except OSError as exc:
            raise EncoderSetupError(
                f"failed to create scratch dir in {self.scratch_dir!r}: {exc}"
            ) from exc
        req_path = scratch / "request.json"
        resp_path = scratch / "response.json"
        try:
            try:
                req_path.write_bytes(
                    (json.dumps(request.to_request_dict(rid), sort_keys=True,
                                ensure_ascii=True) + "\n").encode("utf-8"))
            except OSError as exc:
                raise EncoderSetupError(
                    f"failed to write worker request file: {exc}") from exc
            cmd = [tok.replace("{python}", self.python)
                   .replace("{request}", str(req_path))
                   .replace("{response}", str(resp_path))
                   for tok in self.command]
            self._event("encoder_worker_start", cache_key=key, request_id=rid,
                        command=cmd)
            # stdio goes to scratch *files*: repeated polling can never lose
            # output and a chatty worker can never deadlock on a full pipe.
            out_path = scratch / "stdout.log"
            err_path = scratch / "stderr.log"
            try:
                out_fh = out_path.open("wb")
            except OSError as exc:
                raise EncoderSetupError(
                    f"failed to open worker stdout log: {exc}") from exc
            try:
                err_fh = err_path.open("wb")
            except OSError as exc:
                out_fh.close()
                raise EncoderSetupError(
                    f"failed to open worker stderr log: {exc}") from exc
            try:
                try:
                    proc = subprocess.Popen(  # noqa: S603 - operator-configured command
                        cmd, stdin=subprocess.DEVNULL,
                        stdout=out_fh, stderr=err_fh)
                except OSError as exc:
                    raise EncoderSetupError(
                        f"failed to launch encoder worker: {exc}") from exc
                try:
                    return self._await(proc, resp_path, out_path, err_path,
                                       key, rid, cancel_event)
                finally:
                    # Any exception escaping the wait (unexpected error,
                    # KeyboardInterrupt, ...) must never orphan the child.
                    if proc.poll() is None:
                        self._terminate(proc)
            finally:
                for fh in (out_fh, err_fh):
                    try:
                        fh.close()
                    except OSError:  # pragma: no cover - close on scratch fs
                        pass
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    # -- internals -----------------------------------------------------------

    def _event(self, name: str, **fields: Any) -> None:
        """Log an event, tolerating a dead sink: a broken log must never
        mask a typed encoder outcome or kill the run next to a live child."""
        try:
            self.logger.event(name, **fields)
        except Exception:
            pass

    def _await(self, proc: subprocess.Popen, resp_path: Path, out_path: Path,
               err_path: Path, key: str, rid: str,
               cancel_event: threading.Event | None) -> EncodedPayload:
        deadline = time.monotonic() + self.timeout_s
        timed_out = cancelled = False
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            time.sleep(min(0.02, remaining))
        if timed_out or cancelled:
            self._terminate(proc)
            err_x = _read_excerpt(err_path)
            kind = "timed out" if timed_out else "cancelled"
            self._event("encoder_worker_timeout" if timed_out
                        else "encoder_worker_cancelled",
                        cache_key=key, request_id=rid,
                        timeout_s=self.timeout_s, stderr=err_x)
            message = (f"encoder worker {kind} after {self.timeout_s:.3f}s "
                       f"(terminated, then killed if needed); captured stderr:\n{err_x}")
            if timed_out:
                raise EncoderTimeoutError(message, stderr=err_x)
            raise EncoderCancelledError(message, stderr=err_x)

        out_x = _read_excerpt(out_path)
        err_x = _read_excerpt(err_path)
        try:
            payload = self._result_from_files(proc, resp_path, err_x, key, rid)
        except EncoderError as exc:
            # Every failure branch is terminal and gets an event carrying
            # the captured stderr; a dead sink cannot hide it or hide this.
            self._event("encoder_worker_failed", cache_key=key, request_id=rid,
                        exit_code=proc.returncode, error=type(exc).__name__,
                        stderr=getattr(exc, "stderr", "") or err_x)
            raise
        self._event("encoder_worker_ok", cache_key=key, request_id=rid,
                    exit_code=proc.returncode, stderr=err_x or None,
                    stdout=out_x or None)
        return payload

    def _result_from_files(self, proc: subprocess.Popen, resp_path: Path,
                           err_x: str, key: str, rid: str) -> EncodedPayload:
        # Returncode is checked BEFORE any response read: a nonzero exit is
        # a worker failure whatever (or however valid) the leftover file is.
        if proc.returncode != 0:
            raise EncoderWorkerError(
                f"encoder worker exited {proc.returncode}; captured stderr:\n{err_x}",
                exit_code=proc.returncode, stderr=err_x)
        cap = self.max_response_bytes
        # BOUNDED response handling: a size cap (validated positive at
        # construction) is checked on stat BEFORE the file is opened, and
        # the read itself never reads more than cap + 1 bytes — so neither
        # an already-huge response nor one that grows after the stat can
        # land in memory unbounded. Refusal is a typed protocol error; no
        # JSON/base64 is ever attempted on oversized content.
        try:
            size = os.stat(resp_path).st_size
        except FileNotFoundError as exc:
            raise EncoderProtocolError(
                f"encoder worker exited 0 but wrote no response file; "
                f"captured stderr:\n{err_x}", stderr=err_x) from exc
        except OSError as exc:
            raise EncoderProtocolError(
                f"response file unreadable: {exc}; "
                f"captured stderr:\n{err_x}", stderr=err_x) from exc
        if size > cap:
            raise EncoderProtocolError(
                f"response file is {size} bytes, over max_response_bytes={cap}; "
                f"refusing to read it (configured cap, not a worker verdict); "
                f"captured stderr:\n{err_x}", stderr=err_x)
        try:
            with open(resp_path, "rb") as fh:
                raw = fh.read(cap + 1)
        except FileNotFoundError as exc:
            raise EncoderProtocolError(
                f"encoder worker exited 0 but wrote no response file; "
                f"captured stderr:\n{err_x}", stderr=err_x) from exc
        except OSError as exc:
            raise EncoderProtocolError(
                f"response file unreadable: {exc}; "
                f"captured stderr:\n{err_x}", stderr=err_x) from exc
        if len(raw) > cap:
            raise EncoderProtocolError(
                f"response file grew past max_response_bytes={cap} between "
                f"stat and read (file was {size} bytes at stat); refusing to "
                f"decode a bounded-prefix guess; captured stderr:\n{err_x}",
                stderr=err_x)
        try:
            doc = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise EncoderProtocolError(
                f"worker response is not valid UTF-8 JSON with only finite "
                f"numbers: {exc}; captured stderr:\n{err_x}",
                stderr=err_x) from exc
        return self._payload_from_result(doc, rid, err_x, key)

    def _payload_from_result(self, doc: Any, rid: str, err_x: str,
                             key: str) -> EncodedPayload:
        def proto(msg: str) -> EncoderProtocolError:
            return EncoderProtocolError(
                f"{msg}; captured stderr:\n{err_x}" if err_x else msg,
                stderr=err_x)

        if not isinstance(doc, Mapping):
            raise proto(f"result must be a JSON object, got {type(doc).__name__}")
        extra = sorted(str(k) for k in set(doc) - {"schema", "request_id",
                                                   "status", "metadata",
                                                   "blobs", "error"})
        if extra:
            raise proto(f"result has unknown fields: {', '.join(extra)}")
        if doc.get("schema") != RESULT_SCHEMA:
            raise proto(f"result schema must be {RESULT_SCHEMA!r}, "
                        f"got {doc.get('schema')!r}")
        if doc.get("request_id") != rid:
            raise proto(f"result request_id {doc.get('request_id')!r} does not "
                        f"match request {rid!r}")
        status = doc.get("status")
        if status == "error":
            info = doc.get("error")
            if not isinstance(info, Mapping):
                raise proto("error result requires an object {type?, message}")
            bad = sorted(str(k) for k in set(info) - {"type", "message"})
            if bad:
                raise proto(f"error descriptor has unknown fields: "
                            f"{', '.join(bad)}")
            message = info.get("message")
            if not isinstance(message, str) or not message.strip():
                raise proto(f"error descriptor requires a non-empty string "
                            f"'message', got {message!r}")
            etype = info.get("type", "WorkerError")
            if not isinstance(etype, str) or not etype.strip():
                raise proto(f"error descriptor 'type' must be a non-empty "
                            f"string when present, got {etype!r}")
            raise EncoderWorkerError(
                f"encoder worker reported error {etype}: {message}; "
                f"captured stderr:\n{err_x}" if err_x else
                f"encoder worker reported error {etype}: {message}",
                exit_code=0, stderr=err_x, worker_error=dict(info))
        if status != "ok":
            raise proto(f"unknown result status {status!r}")
        metadata = doc.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise proto("metadata must be an object")
        raw_blobs = doc.get("blobs", [])
        if not isinstance(raw_blobs, list):
            raise proto("blobs must be a list")
        blobs: list[Blob] = []
        for i, b in enumerate(raw_blobs):
            if (not isinstance(b, Mapping) or not isinstance(b.get("name"), str)
                    or not b["name"].strip() or not isinstance(b.get("data_b64"), str)):
                raise proto(f"blob[{i}] must be {{name, data_b64}}, got {b!r}")
            extra_b = sorted(str(k) for k in set(b) - {"name", "data_b64"})
            if extra_b:
                raise proto(f"blob[{i}] descriptor has unknown fields: "
                            f"{', '.join(extra_b)}")
            try:
                data = base64.b64decode(b["data_b64"], validate=True)
            except (ValueError, TypeError) as exc:
                raise proto(f"blob {b['name']!r} is not valid base64: {exc}") from exc
            try:
                blobs.append(Blob(name=b["name"], data=data))
            except EncoderRequestError as exc:
                raise proto(f"blob[{i}]: {exc}") from exc
        try:
            return EncodedPayload(metadata=dict(metadata), blobs=tuple(blobs))
        except EncoderRequestError as exc:
            raise proto(str(exc)) from exc

    def _terminate(self, proc: subprocess.Popen) -> None:
        """terminate → wait(grace) → kill; never leaves an orphan, even if
        the teardown itself is interrupted (best-effort kill on any raise)."""
        try:
            if proc.poll() is not None:
                return
            try:
                proc.terminate()
            except OSError:
                pass
            try:
                proc.wait(timeout=self.cancel_grace_s)
                return
            except subprocess.TimeoutExpired:
                pass
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
                pass
        except BaseException:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                # Blocking reap that cannot itself be interrupted by a
                # patched/blocked sleep: kill already ended the child.
                os.waitpid(proc.pid, 0)
            except (ChildProcessError, OSError):
                pass  # already reaped by Popen.wait
            raise

