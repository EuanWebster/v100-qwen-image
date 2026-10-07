"""Encoder subprocess worker + embedding cache (plan R5, task 1.4).

Stdlib-only. The real encoder (Qwen3-VL, heavy imports) will run in a
*throwaway child process* so it costs no VRAM; this module is the lifecycle +
cache skeleton around it, exercised entirely by a **fake worker** in tests —
no model, no torch.

Pieces
------
* **Wire schemas** (versioned tags, strict validation):

  - ``qwen_engine.encoder_request.v1`` — what the engine hands the worker.
  - ``qwen_engine.encoder_result.v1``  — what the worker writes back
    (status ok/error; payload blobs are opaque base64 bytes on the wire).

* **Cache key v2** (:func:`cache_key`, plan R5): model repo + revision/hash,
  processor / tokenizer / chat-template identities (revision *or* content
  hash), prompt, **negative prompt** (the LTX ``_embed_cache_key`` negative-key
  fix, tested both variants), **all** encoding kwargs, encoder dtype, and
  **ordered** image content hashes for future image conditioning.

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
  emptying the cache, and cannot double-encode the same key.

* **EncoderWorker**: configurable ``command`` with ``{python}`` /
  ``{request}`` / ``{response}`` placeholders, timeout and cooperative
  cancellation (terminate → kill), **stderr captured** into events and typed
  errors (never swallowed; excerpts are read with a bounded tail read), a
  terminal event on **every** failure branch, and **every exit path reaps a
  live child before the scratch temp dir is removed** — including
  interrupted waiting; setup failures are typed and clean up.

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

REQUEST_SCHEMA = "qwen_engine.encoder_request.v1"
RESULT_SCHEMA = "qwen_engine.encoder_result.v1"
CACHE_KEY_SCHEMA = "qwen_engine.embed_cache_key.v2"
ENTRY_SCHEMA = "qwen_engine.embed_entry.v1"

#: LRU cap on cached embeddings (plan R5).
DEFAULT_MAX_ENTRIES = 128

#: How much captured worker stderr is kept in error messages / events.
STDERR_EXCERPT_CHARS = 4000

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


@dataclass(frozen=True)
class EncoderRequest:
    """Everything that determines an encoding — the cache-key inputs live
    here, so an identical *result* is impossible from requests that differ in
    any of them (:func:`cache_key` hashes the whole document). Validated
    JSON values (``kwargs``) are deep-frozen snapshots taken at
    construction: the caller mutating its own (nested) mapping afterwards
    changes neither the cache key nor the wire document — they are computed
    from the same immutable snapshot."""

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
                 "images"}
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
    """Cache key v2 (R5): sha256 over the canonical key document. The
    negative prompt is part of the key (the LTX negative-key fix), as are all
    kwargs, dtype, component identities and ordered image hashes."""
    doc = json.dumps(request.key_document(), sort_keys=True, ensure_ascii=True,
                     separators=(",", ":"))
    return hashlib.sha256(doc.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Payload + entry format (tensor-ready without importing torch)
# ---------------------------------------------------------------------------


class _EntryCorrupt(Exception):
    """Internal: an entry (or response payload) violates the blob format."""


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
        header = {
            "schema": ENTRY_SCHEMA,
            "key": key,
            "metadata": plain_json(self.metadata),
            "blobs": [{"name": b.name, "bytes": len(b.data), "sha256": b.sha256}
                      for b in self.blobs],
        }
        line = json.dumps(header, sort_keys=True, ensure_ascii=True,
                          separators=(",", ":"))
        return line.encode("utf-8") + b"\n" + b"".join(bytes(b.data) for b in self.blobs)

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
    if not isinstance(path, (str, os.PathLike)):
        raise EncoderRequestError(f"{field}: must be a path string or "
                                  f"os.PathLike, got {type(path).__name__}")
    p = Path(path)
    if str(p) == "" or not str(p).strip():
        raise EncoderRequestError(f"{field}: must not be blank")
    if "\0" in str(p):
        raise EncoderRequestError(f"{field}: must not contain a NUL byte")
    return p


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
    """

    def __init__(self, root: str | Path, *, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.root = _canonicalize_dir(root, "cache root")
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) \
                or max_entries < 1:
            raise EncoderRequestError(
                f"max_entries: must be a positive integer, got {max_entries!r}")
        self.max_entries = max_entries
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
        """Return the cached payload, or ``None`` on miss. A corrupt entry is
        a miss **and** gets quarantined — but only *that same file*: the
        quarantine moves the entry aside only after an identity check under
        the cache-wide root mutex, so a valid entry published after a stale
        corrupt read is never quarantined (a confirmed race otherwise moved
        the NEW entry into quarantine). The LRU touch likewise runs under
        the mutex, next to the LRU eviction it races with. Lock order stays
        key-lock → root-mutex (never the reverse)."""
        path = self.entry_path(key)
        read_ident: tuple | None = None
        try:
            if not path.exists():
                return None
            if not path.is_file():
                # A directory / FIFO squatting on the entry name: miss, and
                # quarantine it (never overwrite or read in place).
                return self._quarantine(path, key, read_ident=None)
            with path.open("rb") as fh:
                data = fh.read()
                read_ident = _entry_identity(os.fstat(fh.fileno()))
        except OSError:
            return self._quarantine(path, key, read_ident=None)
        try:
            payload = EncodedPayload.from_entry_bytes(data, key)
        except _EntryCorrupt:
            return self._quarantine(path, key, read_ident=read_ident)
        with self._state.mutex:  # LRU touch serialised with eviction
            try:
                os.utime(path, None)  # LRU refresh; a failed touch must not lose a hit
            except OSError:
                pass
        return payload

    def put(self, key: str, payload: EncodedPayload) -> Path:
        """Atomically publish ``payload`` under ``key``, then enforce the LRU
        cap (the just-written entry is protected). Publish and eviction are
        one transaction under the cache-wide root mutex: concurrent puts can
        never snapshot an entry list and then evict each other's entries."""
        path = self.entry_path(key)
        data = payload.to_entry_bytes(key)
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
        """Current valid cache keys (corrupt files are skipped, not repaired)."""
        if not self.root.is_dir():
            return []
        keys = []
        for entry in sorted(self.root.iterdir()):
            if _ENTRY_NAME_RE.fullmatch(entry.name):
                keys.append(entry.name[: -len(_ENTRY_SUFFIX)])
        return keys

    def _quarantine(self, path: Path, key: str, *,
                    read_ident: tuple | None) -> None:
        """Move a corrupt entry aside under the cache-wide root mutex.

        ``read_ident`` is the identity of the bytes that were actually read
        and found corrupt; when it is not ``None`` the file is quarantined
        ONLY if it still carries that identity — if a valid entry was
        published in the meantime (different inode), the NEW entry stands
        and this read simply reports a miss. With ``read_ident=None`` (an
        entry-name squat, or a failed read) whatever currently sits at the
        path is re-checked: a regular file is only quarantined after a
        fresh parse proves it corrupt, never a freshly published valid
        entry."""
        with self._state.mutex:  # quarantine is a mutation too
            if read_ident is not None:
                try:
                    current = _entry_identity(path.stat())
                except OSError:
                    return None  # vanished between read and lock
                if current != read_ident:
                    return None  # replaced after the corrupt read
            elif path.is_file():
                try:
                    with path.open("rb") as fh:
                        data = fh.read()
                    EncodedPayload.from_entry_bytes(data, key)
                    return None  # parses valid after all — never quarantine it
                except (_EntryCorrupt, OSError):
                    pass  # corrupt / unreadable: quarantine below
            try:
                self.quarantine_dir.mkdir(parents=True, exist_ok=True)
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
        paths = [(self.root / (k + _ENTRY_SUFFIX), k) for k in self.entries()]
        if len(paths) <= self.max_entries:
            return

        def _mtime(item: tuple[Path, str]) -> float:
            try:
                return item[0].stat().st_mtime
            except OSError:
                return float("-inf")

        paths.sort(key=lambda item: (_mtime(item), item[1]))
        overflow = len(paths) - self.max_entries
        for path, key in paths:
            if overflow <= 0:
                break
            if key == protect:
                continue
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
    on every path. (Pinned-host resources are an M4 concern of the real
    worker; this runner owns only its temp files.)
    """

    def __init__(self, *, command: Sequence[str], timeout_s: float,
                 cache: EmbeddingCache | None = None,
                 scratch_dir: str | Path | None = None,
                 cancel_grace_s: float = 5.0,
                 python: str | None = None,
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
        try:
            raw = resp_path.read_bytes()
        except FileNotFoundError as exc:
            raise EncoderProtocolError(
                f"encoder worker exited 0 but wrote no response file; "
                f"captured stderr:\n{err_x}", stderr=err_x) from exc
        except OSError as exc:
            raise EncoderProtocolError(
                f"response file unreadable: {exc}; "
                f"captured stderr:\n{err_x}", stderr=err_x) from exc
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

