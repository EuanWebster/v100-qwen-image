"""Content-addressed store for user-supplied reference images (editing, §2.3).

The browser/CLI hands bytes to the server once; everything after that is
identified by the sha256 of the ORIGINAL bytes and stored under
``<root>/<id>.img`` plus a sidecar ``<id>.json`` and a preview
``<id>.thumb.png``. Blobs are immutable: "replacing" an image is a new upload
with a new id, so a conditioning cache key that binds the id can never be
served stale bytes.

Layout and caps (design docs/image-edit-design.md §2.3):

* identity = sha256(raw bytes), lowercase 64-hex — ids are content hashes;
* accepted formats by magic sniff: PNG, JPEG, WebP (the same set the web
  layer already content-types); decoded pixels capped
  (``MAX_REFERENCE_DIM`` per side, ``MAX_REFERENCE_PIXELS`` total) BEFORE any
  full decode completes, checked from the image header;
* per-file cap ``MAX_REFERENCE_BYTES``; store-wide caps
  ``MAX_STORE_BLOBS``/``MAX_STORE_BYTES`` — exceeding a cap is a typed
  refusal, never an eviction (no automatic GC);
* publish is no-clobber (``O_EXCL``); a concurrent identical upload dedupes
  onto the same id;
* ``resolve`` re-hashes every file it hands out, so a tampered or truncated
  blob fails closed long before a model sees it.

Stdlib at import; Pillow is imported lazily inside :func:`put_bytes` only
(``resolve``/reads need no imaging library), so web/jobs/worker code paths
that only look up bytes stay import-cheap.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import threading
from pathlib import Path
from typing import Any

from qwen_engine.errors import EngineError

#: Schema tag of the sidecar document.
REFERENCE_SIDECAR_SCHEMA = "qwen_engine.reference_sidecar.v1"

#: Hard caps (single source of truth: the worker, the web upload path and
#: the store itself all read these constants).
MAX_REFERENCE_BYTES = 32 * 1024 * 1024
MAX_REFERENCE_DIM = 4096
MAX_REFERENCE_PIXELS = 4096 * 4096
MAX_STORE_BLOBS = 512
MAX_STORE_BYTES = 2 * 1024 ** 3
THUMB_MAX_DIM = 256
#: The sidecar is a tiny fixed-shape document; anything larger is corruption.
MAX_SIDECAR_BYTES = 16 * 1024

BLOB_SUFFIX = ".img"
SIDECAR_SUFFIX = ".json"
THUMB_SUFFIX = ".thumb.png"

_ID_RE = re.compile(r"[0-9a-f]{64}\Z")

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"RIFF", "webp"),  # full check below: bytes 8:12 must be WEBP
)

_STORE_LOCK = threading.RLock()


class ReferenceStoreError(EngineError):
    """Typed refusal of the reference store (bad id/blob/cap/format)."""


def is_reference_id(value: Any) -> bool:
    """True exactly for the canonical content-id grammar."""
    return isinstance(value, str) and _ID_RE.fullmatch(value) is not None


def _check_id(value: Any) -> str:
    if not is_reference_id(value):
        raise ReferenceStoreError(
            f"reference id must be a lowercase 64-hex sha256, got {value!r}")
    return value


def sniff_format(raw: bytes) -> str:
    """``"png"`` / ``"jpeg"`` / ``"webp"`` by magic, else typed refusal.

    Never trust an extension or a content-type header: identity and format
    come from the bytes themselves.
    """
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "webp"
    raise ReferenceStoreError(
        "reference images must be PNG, JPEG or WebP (checked by magic bytes)")


def _load_pil():
    """The lazy Pillow module (``PIL.Image``); uploads are the only path
    that needs an imaging library."""
    try:
        import importlib
        return importlib.import_module("PIL.Image")
    except ImportError as exc:  # pragma: no cover - pinned env has Pillow
        raise ReferenceStoreError(f"Pillow is required to accept uploads: {exc}") from exc


def _canonical_root(root: str | os.PathLike) -> Path:
    if not isinstance(root, (str, os.PathLike)):
        raise ReferenceStoreError(
            f"reference root must be a path, got {type(root).__name__}")
    raw = os.fspath(root)
    if not isinstance(raw, str) or not raw.strip() or "\0" in raw:
        raise ReferenceStoreError("reference root must be a nonblank path without NUL")
    path = Path(os.path.realpath(raw))
    if not path.is_dir():
        raise ReferenceStoreError(f"reference root is not an existing directory: {path}")
    return path


def _read_limited_bytes(path: Path, cap: int, label: str) -> bytes:
    """One bounded, no-follow, regular-file read with identity checks."""
    try:
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY
                            | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=directory)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
                    raise ReferenceStoreError(
                        f"{label}: not a bounded regular file of at most {cap} bytes")
                data = bytearray()
                while True:
                    block = os.read(fd, min(65536, cap + 1 - len(data)))
                    if not block:
                        break
                    data.extend(block)
                    if len(data) > cap:
                        raise ReferenceStoreError(f"{label}: byte cap exceeded")
                if bytes(data) == b"":
                    raise ReferenceStoreError(f"{label}: empty file")
                after = os.fstat(fd)
                if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
                    raise ReferenceStoreError(f"{label}: changed during read")
                return bytes(data)
            finally:
                os.close(fd)
        finally:
            os.close(directory)
    except ReferenceStoreError:
        raise
    except OSError as exc:
        raise ReferenceStoreError(f"{label}: cannot read: {exc}") from exc


def _usage(root: Path) -> tuple[int, int]:
    blobs = 0
    total = 0
    for entry in root.iterdir():
        if not entry.name.endswith(BLOB_SUFFIX):
            continue
        stem = entry.name[:-len(BLOB_SUFFIX)]
        if not is_reference_id(stem):
            continue
        try:
            info = os.lstat(entry)
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            blobs += 1
            total += info.st_size
    return blobs, total


def _decode_checks(raw: bytes):
    """Header-only decode gate: format agreement and pixel caps BEFORE the
    full load. Returns the opened (lazy) PIL image, fully loaded. Every
    Pillow failure — unopenable header, truncated stream, decode error —
    is typed, never a raw library exception."""
    pil = _load_pil()
    try:
        image = pil.open(io.BytesIO(raw))
        width, height = image.size
    except Exception as exc:
        raise ReferenceStoreError(
            f"reference image cannot be opened as a stored image: {exc}") from exc
    if width < 1 or height < 1:
        raise ReferenceStoreError("reference image has empty dimensions")
    if width > MAX_REFERENCE_DIM or height > MAX_REFERENCE_DIM:
        raise ReferenceStoreError(
            f"reference image sides must be at most {MAX_REFERENCE_DIM} px, "
            f"got {width}x{height}")
    if width * height > MAX_REFERENCE_PIXELS:
        raise ReferenceStoreError(
            f"reference image pixel budget exceeded: {width}x{height} "
            f"> {MAX_REFERENCE_PIXELS}")
    if (image.format or "").lower() not in ("png", "jpeg", "webp"):
        raise ReferenceStoreError(
            f"reference image decoded as unsupported format {image.format!r}")
    try:
        image.load()
    except Exception as exc:
        raise ReferenceStoreError(
            f"reference image cannot be decoded ({type(exc).__name__}: {exc})") from exc
    return image


def _sidecar_document(reference_id: str, raw: bytes, image) -> dict:
    mode = image.mode
    return {
        "schema": REFERENCE_SIDECAR_SCHEMA,
        "id": reference_id,
        "bytes": len(raw),
        "format": (image.format or sniff_format(raw)).lower(),
        "width": image.size[0],
        "height": image.size[1],
        "mode": mode,
        "has_alpha": mode in ("RGBA", "LA", "PA") or "transparency" in image.info,
    }


def _publish(root: Path, name: str, data: bytes) -> None:
    """Atomic no-clobber publish; a lost race keeps the existing bytes.
    Opens the absolute path (O_NOFOLLOW): the guarded CPU test runner
    resolves write targets lexically and cannot see through dir_fd-relative
    opens; the root is realpath-canonicalized before this point anyway."""
    path = root / name
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    except FileExistsError:
        return
    except OSError as exc:
        raise ReferenceStoreError(f"cannot publish {name!r}: {exc}") from exc
    try:
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("zero-length write")
            view = view[count:]
        os.fsync(fd)
    except OSError as exc:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise ReferenceStoreError(f"cannot write {name!r}: {exc}") from exc
    os.close(fd)


def put_bytes(root: str | os.PathLike, raw: bytes) -> str:
    """Validate, store and return the content id of ``raw``.

    Idempotent: storing identical bytes twice returns the same id without
    touching the disk. Caps are refusals, never evictions.
    """
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if not isinstance(raw, bytes) or not raw:
        raise ReferenceStoreError("reference bytes must be a non-empty bytes value")
    if len(raw) > MAX_REFERENCE_BYTES:
        raise ReferenceStoreError(
            f"reference upload exceeds {MAX_REFERENCE_BYTES} bytes (got {len(raw)})")
    sniff_format(raw)  # magic first: cheap, before any decoder touches it
    reference_id = hashlib.sha256(raw).hexdigest()
    base_dir = _canonical_root(root)
    with _STORE_LOCK:
        blob = base_dir / (reference_id + BLOB_SUFFIX)
        if blob.exists():
            _verify_blob(base_dir, reference_id)  # dedupe path re-validates
            return reference_id
        blobs, total = _usage(base_dir)
        if blobs >= MAX_STORE_BLOBS or total + len(raw) > MAX_STORE_BYTES:
            raise ReferenceStoreError(
                f"reference store is full ({blobs} blobs, {total} bytes; caps "
                f"{MAX_STORE_BLOBS}/{MAX_STORE_BYTES}); an operator must prune it")
        image = _decode_checks(raw)
        sidecar = _sidecar_document(reference_id, raw, image)
        thumb_class = image.convert("RGBA") if sidecar["has_alpha"] else image
        thumb_class.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM), _load_pil().LANCZOS)
        buffer = io.BytesIO()
        thumb_class.save(buffer, format="PNG")
        _publish(base_dir, reference_id + SIDECAR_SUFFIX,
                 (json.dumps(sidecar, sort_keys=True) + "\n").encode("utf-8"))
        _publish(base_dir, reference_id + THUMB_SUFFIX, buffer.getvalue())
        _publish(base_dir, reference_id + BLOB_SUFFIX, raw)
    return reference_id


def path_for(root: str | os.PathLike, reference_id: str) -> Path:
    """The strict-id blob path, WITHOUT touching the filesystem (the worker
    IPC uses this after :func:`resolve` already bound the hash)."""
    _check_id(reference_id)
    return _canonical_root(root) / (reference_id + BLOB_SUFFIX)


def _verify_blob(base_dir: Path, reference_id: str) -> bytes:
    raw = _read_limited_bytes(base_dir / (reference_id + BLOB_SUFFIX),
                              MAX_REFERENCE_BYTES, f"reference {reference_id[:12]}")
    if hashlib.sha256(raw).hexdigest() != reference_id:
        raise ReferenceStoreError(
            f"reference {reference_id[:12]} content no longer matches its id "
            "(tampered or truncated blob); re-upload the image")
    return raw


def resolve(root: str | os.PathLike, ids: tuple[str, ...] | list[str]) -> tuple[Path, ...]:
    """Strict-resolve ordered ids, re-hashing every blob; typed refusal on
    any miss, traversal-shaped id, or content mismatch."""
    base_dir = _canonical_root(root)
    if isinstance(ids, (str, bytes)) or not isinstance(ids, (tuple, list)):
        raise ReferenceStoreError("reference ids must be a tuple/list of ids")
    with _STORE_LOCK:
        paths: list[Path] = []
        for reference_id in ids:
            _check_id(reference_id)
            _verify_blob(base_dir, reference_id)
            paths.append(base_dir / (reference_id + BLOB_SUFFIX))
    return tuple(paths)


def read_blob(root: str | os.PathLike, reference_id: str) -> bytes:
    """Hash-verified blob bytes (for serving originals)."""
    _check_id(reference_id)
    base_dir = _canonical_root(root)
    with _STORE_LOCK:
        return _verify_blob(base_dir, reference_id)


def read_sidecar(root: str | os.PathLike, reference_id: str) -> dict:
    """The validated sidecar document of one stored reference."""
    _check_id(reference_id)
    base_dir = _canonical_root(root)
    doc = _read_limited_bytes(base_dir / (reference_id + SIDECAR_SUFFIX),
                              MAX_SIDECAR_BYTES, f"reference sidecar {reference_id[:12]}")
    try:
        parsed = json.loads(doc.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceStoreError(f"reference sidecar is not valid JSON: {exc}") from exc
    if (not isinstance(parsed, dict) or parsed.get("schema") != REFERENCE_SIDECAR_SCHEMA
            or parsed.get("id") != reference_id):
        raise ReferenceStoreError("reference sidecar identity disagrees")
    return parsed


def read_thumb(root: str | os.PathLike, reference_id: str) -> bytes:
    """Stored PNG preview bytes (bounded; not re-hashed — it is cosmetic)."""
    _check_id(reference_id)
    data = _read_limited_bytes(_canonical_root(root) / (reference_id + THUMB_SUFFIX),
                               MAX_REFERENCE_BYTES, "reference thumbnail")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ReferenceStoreError("reference thumbnail is not a PNG")
    return data


def list_ids(root: str | os.PathLike) -> tuple[str, ...]:
    """Stored ids (name-shaped only; no reads)."""
    base_dir = _canonical_root(root)
    found = []
    for entry in base_dir.iterdir():
        if entry.name.endswith(BLOB_SUFFIX):
            stem = entry.name[:-len(BLOB_SUFFIX)]
            if is_reference_id(stem):
                found.append(stem)
    return tuple(sorted(found))


def usage(root: str | os.PathLike) -> tuple[int, int]:
    """(blob_count, total_bytes) of owned blobs — for status surfaces."""
    return _usage(_canonical_root(root))
