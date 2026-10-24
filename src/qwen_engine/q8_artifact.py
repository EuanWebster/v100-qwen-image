"""Verified single-GGUF Q8 *sidecar* artifact support (stdlib-only, no ML imports).

The closed 24-file full-reference bundle (:mod:`qwen_engine.local_bundle`) is
untouched by this module and stays exactly as closed as it was: a Q8 GGUF file
is **not** an allowed entry of that bundle. Instead it is a *sidecar* — one
pinned GGUF file whose metadata lives in a small JSON document
(``qwen_engine.q8_artifact.v1``) that sits **outside** the closed
``models``/``assets`` tree. This module parses that sidecar strictly, then
verifies the file it names on disk, and returns a frozen :class:`Q8Artifact`
with a canonical path and an identity fingerprint. It never downloads, never
imports ML libraries, and reads no more of the GGUF than it must.

Trust model — read this before relying on anything here:

* The sidecar's ``sha256`` is **operator-reviewed identity metadata**, not a
  numerical or provenance proof. Matching it demonstrates that the bytes on
  disk are the bytes the operator reviewed; it says nothing about whether the
  quantization is faithful, whether the file came from the claimed upstream
  repo, or whether the weights are good.
* :func:`resolve_q8_artifact` validates the *container*: bounded regular file,
  containment under the data root, declared size, streamed SHA256, GGUF magic
  (``GGUF``), version 2 or 3, and stability of the file while hashing. The
  reader, architecture, and Q8 tensor-level validation remain the **loader's**
  responsibility; nothing here parses tensor metadata beyond the 8-byte header.
* The sidecar's ``source_model_revision`` must equal the base-model revision
  this engine profile was reviewed against (default
  :data:`~qwen_engine.local_bundle.FULL_REFERENCE_REVISION`). A Q8 file
  quantized from a different base revision is a different, unreviewed model
  identity and is refused — never silently accepted.
* The candidate upstream producer is ``unsloth/Qwen-Image-2.1-GGUF``, but this
  module deliberately hardcodes **no** repo, revision, hash, or size for it:
  those are per-deployment reviewed values carried by the sidecar itself.

Sidecar JSON schema (exact keys; unknown/missing/duplicate keys are errors):

``schema``
    Exactly ``"qwen_engine.q8_artifact.v1"``.
``repo`` / ``revision``
    Upstream HF repo id (``org/name``) and the pinned 40-hex GGUF revision.
``filename``
    Safe relative POSIX path ending in ``.gguf`` (no absolute, no ``..``, no
    backslashes, no drive letters) resolved under the data root.
``sha256``
    64 lowercase hex chars, streamed over the whole file.
``bytes``
    Positive ``int`` (a ``bool`` is refused) — the exact file size.
``source_model_revision``
    Pinned 40-hex revision of the base model the GGUF was quantized from.
``quant_format``
    Exactly ``"gguf-q8_0"``.

Data root: an explicit ``models_dir`` argument, else the sidecar's parent
directory. Because the closed-tree rule places sidecar data outside the
``models``/``assets`` tree, the typical layout is e.g.
``data/q8/q8_artifact.json`` next to ``data/q8/model-q8_0.gguf``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat as stat_module
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from qwen_engine._jsonsafe import strict_json_loads
from qwen_engine._local_io import read_bounded_file
from qwen_engine.errors import EngineError
from qwen_engine.local_bundle import FULL_REFERENCE_REVISION
from qwen_engine.manifest import ManifestError, ManifestFormatError

#: Exact schema id of a Q8 sidecar document.
Q8_ARTIFACT_SCHEMA_VERSION = "qwen_engine.q8_artifact.v1"

#: The only quant format this sidecar contract accepts.
Q8_QUANT_FORMAT = "gguf-q8_0"

#: Version tag folded into :attr:`Q8Artifact.identity_sha256`.
Q8_IDENTITY_FINGERPRINT_VERSION = "qwen-image.q8-artifact.v1"

#: Hard bound on the sidecar JSON document (1 MiB).
MAX_Q8_SIDECAR_BYTES = 1 << 20

#: GGUF container constants (only the 8-byte header is inspected).
GGUF_MAGIC = b"GGUF"
GGUF_SUPPORTED_VERSIONS = (2, 3)
_GGUF_HEADER_BYTES = 8

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SIDECAR_KEYS = frozenset({
    "schema", "repo", "revision", "filename", "sha256", "bytes",
    "source_model_revision", "quant_format",
})
_HASH_CHUNK = 1 << 20
_STAT_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")


class Q8SidecarFormatError(ManifestFormatError):
    """The sidecar JSON is malformed/invalid; its message names every
    offending field. Exit code 4 (asset channel)."""


class Q8FileError(ManifestError, EngineError):
    """The GGUF file named by a sidecar failed verification. Carries the
    sidecar identity fields (repo/revision/filename) in the same structured,
    typed channel the manifest resolver uses. Exit code 4."""

    problem = "Q8 GGUF failed verification"

    def __init__(self, spec: "Q8ArtifactSpec", detail: str = "") -> None:
        self.spec = spec
        self.role = "q8_gguf"
        self.repo = spec.repo
        self.revision = spec.revision
        self.filename = spec.filename
        self.detail = detail
        base = (
            f"{self.problem}: '{spec.filename}' "
            f"(role=q8_gguf, repo={spec.repo}, revision={spec.revision})"
        )
        super().__init__(f"{base}: {detail}" if detail else base)


class Q8FileMissingError(Q8FileError):
    problem = "Q8 GGUF missing on disk"


class Q8FilePathError(Q8FileError):
    """The file canonicalizes outside the data root (symlink escape)."""

    problem = "Q8 GGUF path escapes data root"


class Q8FileSizeMismatchError(Q8FileError):
    problem = "Q8 GGUF byte size mismatch"

    def __init__(self, spec: "Q8ArtifactSpec", expected_bytes: int,
                 actual_bytes: int, detail: str = "") -> None:
        self.expected_bytes = expected_bytes
        self.actual_bytes = actual_bytes
        msg = f"expected {expected_bytes} bytes, found {actual_bytes}"
        super().__init__(spec, f"{msg} ({detail})" if detail else msg)


class Q8FileHashMismatchError(Q8FileError):
    problem = "Q8 GGUF sha256 mismatch"

    def __init__(self, spec: "Q8ArtifactSpec", expected: str, actual: str) -> None:
        self.expected_sha256 = expected
        self.actual_sha256 = actual
        super().__init__(spec, f"expected {expected}, computed {actual}")


class Q8FileUnreadableError(Q8FileError):
    """Exists but cannot be read/verified (permissions, non-regular file,
    bad GGUF magic/version, or changed while hashing)."""

    problem = "Q8 GGUF unreadable or not a valid GGUF container"


# ---------------------------------------------------------------------------
# Field validators (shared by JSON parsing and direct construction)
# ---------------------------------------------------------------------------


def _safe_gguf_filename(filename: Any) -> str | None:
    """Error string if ``filename`` is not a safe relative POSIX ``.gguf``
    path (no traversal, absolutes, backslashes, drive letters, or NULs)."""
    if not isinstance(filename, str) or not filename:
        return f"must be a non-empty string, got {filename!r}"
    if "\0" in filename:
        return f"contains a NUL byte: {filename!r}"
    if "\\" in filename:
        return (f"must be a POSIX path without backslashes (a Windows-style "
                f"separator is a traversal risk), got {filename!r}")
    if PurePosixPath(filename).is_absolute():
        return f"must be relative, got absolute path {filename!r}"
    if re.match(r"^[A-Za-z]:", filename):
        return f"looks like a Windows drive-qualified path, got {filename!r}"
    for part in filename.split("/"):
        if part == "":
            return f"empty path component in {filename!r}"
        if part == ".":
            return f"dot path component in {filename!r}"
        if part == "..":
            return f"parent-directory component in {filename!r}"
    if not filename.endswith(".gguf"):
        return f"must name a .gguf file, got {filename!r}"
    return None


def _safe_repo(repo: Any) -> str | None:
    """Error string unless ``repo`` is a plain ``org/name`` HF repo id."""
    if not isinstance(repo, str):
        return f"must be a string, got {repo!r}"
    parts = repo.split("/")
    if len(parts) != 2:
        return (f"must be an 'org/name' repo id with exactly one separator, "
                f"got {repo!r}")
    for part in parts:
        if not _REPO_RE.fullmatch(part):
            return (f"repo component {part!r} must be 1-64 chars of "
                    f"[A-Za-z0-9._-], starting alphanumeric, in {repo!r}")
    return None


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_spec_fields(spec: Mapping[str, Any], at: str,
                       errors: list[str]) -> None:
    """Append every field-level problem with a sidecar mapping to *errors*
    (all problems reported; every field checked independently)."""
    def bad(field: str, msg: str) -> None:
        errors.append(f"{at}.{field}: {msg}")

    schema = spec.get("schema")
    if schema != Q8_ARTIFACT_SCHEMA_VERSION:
        bad("schema", f"must be exactly {Q8_ARTIFACT_SCHEMA_VERSION!r}, got {schema!r}")
    quant = spec.get("quant_format")
    if quant != Q8_QUANT_FORMAT:
        bad("quant_format", f"must be exactly {Q8_QUANT_FORMAT!r}, got {quant!r}")
    repo = spec.get("repo")
    problem = _safe_repo(repo)
    if problem is not None:
        bad("repo", problem)
    filename = spec.get("filename")
    problem = _safe_gguf_filename(filename)
    if problem is not None:
        bad("filename", problem)
    for field in ("revision", "source_model_revision"):
        value = spec.get(field)
        if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
            bad(field, "must be a pinned 40-hex commit sha (not a branch or "
                       f"tag), got {value!r}")
    sha = spec.get("sha256")
    if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
        bad("sha256", f"must be 64 lowercase hex chars, got {sha!r}")
    size = spec.get("bytes")
    if not _positive_int(size):
        bad("bytes", f"must be a positive integer (not a bool), got {size!r}")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Q8ArtifactSpec:
    """The *manifest identity* of one pinned Q8 GGUF file, exactly as the
    sidecar declares it. Validated on every construction (direct or parsed)."""

    schema: str
    repo: str
    revision: str
    filename: str
    sha256: str
    bytes: int
    source_model_revision: str
    quant_format: str

    def __post_init__(self) -> None:
        errors: list[str] = []
        _check_spec_fields(self.to_dict(), "q8_artifact", errors)
        if errors:
            raise Q8SidecarFormatError("invalid Q8 artifact spec: " + "; ".join(errors))

    @property
    def identity(self) -> str:
        """Normalized relative filename (the single file's identity)."""
        return "/".join(self.filename.split("/"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "repo": self.repo,
            "revision": self.revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "source_model_revision": self.source_model_revision,
            "quant_format": self.quant_format,
        }


@dataclass(frozen=True)
class Q8Artifact:
    """A Q8 GGUF whose on-disk bytes matched its sidecar. ``path`` is the
    canonical (resolved) absolute path; ``identity_sha256`` is a version-tagged
    digest over the canonical manifest fields only — no ambient paths — so the
    same reviewed file yields the same fingerprint on any machine.

    Both fields are **identity metadata for operator review and cache keys**;
    they are not numerical or provenance proofs, and they do not certify that
    any loader can read the file (reader/architecture/Q8 tensor validation is
    the loader's responsibility)."""

    spec: Q8ArtifactSpec
    path: Path
    identity_sha256: str

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.spec, Q8ArtifactSpec):
            errors.append(f"spec: must be a Q8ArtifactSpec, got {type(self.spec).__name__}")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            errors.append(f"path: must be an absolute Path, got {self.path!r}")
        if not isinstance(self.identity_sha256, str) or not _SHA256_RE.fullmatch(self.identity_sha256):
            errors.append(f"identity_sha256: must be 64 lowercase hex chars, got {self.identity_sha256!r}")
        if errors:
            raise Q8SidecarFormatError("invalid resolved Q8 artifact: " + "; ".join(errors))

    @property
    def repo(self) -> str:
        return self.spec.repo

    @property
    def revision(self) -> str:
        return self.spec.revision

    @property
    def filename(self) -> str:
        return self.spec.filename

    @property
    def sha256(self) -> str:
        return self.spec.sha256

    @property
    def bytes(self) -> int:
        return self.spec.bytes

    @property
    def source_model_revision(self) -> str:
        return self.spec.source_model_revision

    @property
    def quant_format(self) -> str:
        return self.spec.quant_format

    def to_dict(self) -> dict[str, Any]:
        """Identity view (manifest fields + fingerprint). Deliberately
        excludes ``path``: the fingerprint covers exactly this field set, so
        the same file on two machines prints two paths but one identity."""
        out = self.spec.to_dict()
        out["identity_sha256"] = self.identity_sha256
        return out


def q8_identity_sha256(spec: Q8ArtifactSpec) -> str:
    """SHA256 over the version-tagged canonical JSON of the sidecar's
    manifest fields (sorted keys, no ambient paths). Changes exactly when any
    reviewed identity field changes; proves nothing beyond that."""
    payload = {"version": Q8_IDENTITY_FINGERPRINT_VERSION, **spec.to_dict()}
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sidecar parsing (strict; all problems reported)
# ---------------------------------------------------------------------------


def q8_artifact_spec_from_dict(data: Any) -> Q8ArtifactSpec:
    """Validate a parsed sidecar mapping; raise :class:`Q8SidecarFormatError`
    naming every offending field. Unknown keys, missing keys, wrong types,
    bool-as-int, non-``gguf-q8_0`` quant formats and foreign schemas are all
    refused here, before any filesystem access."""
    if not isinstance(data, Mapping):
        raise Q8SidecarFormatError(
            f"q8 sidecar: must be a JSON object, got {type(data).__name__}")
    errors: list[str] = []
    for key in sorted(set(data) - _SIDECAR_KEYS, key=repr):
        errors.append(f"q8_artifact.{key}: unknown sidecar key")
    missing = sorted(_SIDECAR_KEYS - set(data), key=repr)
    for key in missing:
        errors.append(f"q8_artifact.{key}: missing required sidecar key")
    spec = dict(data)
    if not missing:
        _check_spec_fields(spec, "q8_artifact", errors)
    if errors:
        raise Q8SidecarFormatError("invalid Q8 sidecar: " + "; ".join(errors))
    return Q8ArtifactSpec(**spec)


def load_q8_sidecar(path: str | Path) -> Q8ArtifactSpec:
    """Read and strictly parse a sidecar JSON file (bounded 1 MiB, no
    duplicate keys, no nonfinite constants, no unknown/missing fields)."""
    path = Path(path)
    try:
        raw = read_bounded_file(path, MAX_Q8_SIDECAR_BYTES)
    except (OSError, ValueError, RuntimeError) as exc:
        raise Q8SidecarFormatError(
            f"Q8 sidecar not readable: {path}: "
            f"{exc.strerror or exc if isinstance(exc, OSError) else exc}"
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Q8SidecarFormatError(f"Q8 sidecar is not valid UTF-8: {path}: {exc}") from exc
    try:
        data = strict_json_loads(text)
    except (ValueError, RecursionError) as exc:
        raise Q8SidecarFormatError(f"Q8 sidecar is not valid JSON: {path}: {exc}") from exc
    return q8_artifact_spec_from_dict(data)


# ---------------------------------------------------------------------------
# Filesystem verification (no-follow FDs, streamed hash, GGUF header)
# ---------------------------------------------------------------------------


def _stat_signature(info: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(info, field) for field in _STAT_FIELDS)


def _os_detail(exc: OSError) -> str:
    return (f"errno {exc.errno} ({exc.strerror or type(exc).__name__}): "
            f"{exc.filename or ''}").rstrip()


class _FileChangedDuringHash(OSError):
    """Internal: the open file mutated between fstat and end of hashing."""


@contextmanager
def _data_fd(root: Path, resolved: Path):
    """Open a canonical in-root target through anchored no-follow directory
    FDs, so no path component can be swapped to a symlink mid-verification."""
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
    parent_fd, fd = root_fd, None
    try:
        parts = resolved.relative_to(root).parts
        for part in parts[:-1]:
            child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
                               dir_fd=parent_fd)
            if parent_fd != root_fd:
                os.close(parent_fd)
            parent_fd = child_fd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        yield root_fd, parent_fd, fd
    finally:
        if fd is not None:
            os.close(fd)
        if parent_fd != root_fd:
            os.close(parent_fd)
        os.close(root_fd)


def _hash_and_read_gguf_header(fd: int, pre: os.stat_result) -> tuple[str, bytes]:
    """Stream-hash the whole open descriptor (never buffering it in RAM) and
    capture the first 8 bytes; raise :class:`_FileChangedDuringHash` if the
    file grows past its initial size or its identity changes while hashing."""
    digest = hashlib.sha256()
    head = b""
    os.lseek(fd, 0, os.SEEK_SET)
    count = 0
    while chunk := os.read(fd, min(_HASH_CHUNK, pre.st_size + 1 - count)):
        if len(head) < _GGUF_HEADER_BYTES:
            head += chunk[:_GGUF_HEADER_BYTES - len(head)]
        count += len(chunk)
        if count > pre.st_size:
            raise _FileChangedDuringHash("file grew during verification")
        digest.update(chunk)
    after = os.fstat(fd)
    if count != pre.st_size or _stat_signature(after) != _stat_signature(pre):
        raise _FileChangedDuringHash(
            "file changed during verification (size, mode, timestamps or identity)")
    return digest.hexdigest(), head


def _check_gguf_header(head: bytes, spec: "Q8ArtifactSpec") -> None:
    """Validate magic and version from the 8-byte GGUF header (little-endian
    uint32 magic + uint32 version). Content-level validation stays with the
    loader; this only refuses non-GGUF or unsupported-version containers."""
    if len(head) < _GGUF_HEADER_BYTES or head[:4] != GGUF_MAGIC:
        raise Q8FileUnreadableError(
            spec, "bad GGUF magic: expected 'GGUF' at offset 0")
    version = int.from_bytes(head[4:8], "little")
    if version not in GGUF_SUPPORTED_VERSIONS:
        raise Q8FileUnreadableError(
            spec,
            f"unsupported GGUF version {version} (supported: "
            f"{', '.join(str(v) for v in GGUF_SUPPORTED_VERSIONS)})")


def _resolve_or_fail(path: Path) -> Path | None:
    """``Path.resolve`` is not exception-free (symlink loops, permissions);
    no raw resolution failure may escape the typed boundary."""
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def verify_q8_file(spec: Q8ArtifactSpec, data_dir: str | Path) -> Path:
    """Verify the GGUF file ``spec`` claims, under ``data_dir``, and return its
    canonical path. Checks, in order, each raising a typed :class:`Q8FileError`
    (all under :class:`~qwen_engine.manifest.ManifestError`, CLI exit 4):

    * resolution/containment: the target must canonicalize inside the data
      root (a symlink out of the root is :class:`Q8FilePathError`);
    * regular file (directories/FIFOs/devices are refused before open);
    * exact declared size;
    * streamed SHA256 == declared ``sha256`` (the file is never read into
      RAM wholesale);
    * GGUF magic and version 2 or 3 from the header bytes captured while
      hashing;
    * stability: the file, its directory entry and the root are re-checked
      after hashing, so a swap mid-verification is a typed failure.

    Performs no network access and no downloads."""
    root = Path(data_dir)
    root_resolved = _resolve_or_fail(root)
    if root_resolved is None:
        raise Q8FileUnreadableError(spec, f"data root cannot be resolved: {root}")
    path = root_resolved / spec.filename
    resolved = _resolve_or_fail(path)
    if resolved is None:
        raise Q8FileUnreadableError(spec, f"cannot resolve {path}")
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise Q8FilePathError(spec, f"resolves to {resolved}, outside {root_resolved}")
    try:
        st = resolved.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise Q8FileMissingError(spec) from None
    except OSError as exc:
        raise Q8FileUnreadableError(spec, _os_detail(exc)) from exc
    if not stat_module.S_ISREG(st.st_mode):
        kind = ("directory" if stat_module.S_ISDIR(st.st_mode)
                else "FIFO" if stat_module.S_ISFIFO(st.st_mode)
                else f"non-regular file (mode {oct(st.st_mode)})")
        raise Q8FileUnreadableError(spec, f"not a regular file: {kind}")
    if st.st_size != spec.bytes:
        raise Q8FileSizeMismatchError(spec, spec.bytes, st.st_size)
    try:
        with _data_fd(root_resolved, resolved) as (root_fd, parent_fd, fd):
            pre = os.fstat(fd)
            if not stat_module.S_ISREG(pre.st_mode):
                raise OSError("not a regular file (swapped after stat)")
            if pre.st_size != spec.bytes:
                raise Q8FileSizeMismatchError(spec, spec.bytes, pre.st_size)
            if _stat_signature(pre) != _stat_signature(st):
                raise OSError("file changed between stat and open")
            try:
                actual_sha, head = _hash_and_read_gguf_header(fd, pre)
            except _FileChangedDuringHash as exc:
                raise Q8FileSizeMismatchError(
                    spec, spec.bytes, os.fstat(fd).st_size,
                    detail="file changed during hashing") from exc
            published = os.stat(resolved.name, dir_fd=parent_fd, follow_symlinks=False)
            root_open, root_now = os.fstat(root_fd), root_resolved.stat(follow_symlinks=False)
            if (_resolve_or_fail(path) != resolved
                    or _stat_signature(published) != _stat_signature(pre)
                    or _stat_signature(path.stat()) != _stat_signature(pre)
                    or (root_open.st_dev, root_open.st_ino)
                    != (root_now.st_dev, root_now.st_ino)):
                raise OSError("file or data root changed during verification")
    except FileNotFoundError:
        raise Q8FileMissingError(spec) from None
    except OSError as exc:
        raise Q8FileUnreadableError(
            spec, _os_detail(exc) if exc.errno is not None else str(exc)) from exc
    if actual_sha != spec.sha256:
        raise Q8FileHashMismatchError(spec, spec.sha256, actual_sha)
    _check_gguf_header(head, spec)
    return resolved


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_q8_artifact(
    manifest_path: str | Path,
    models_dir: str | Path | None = None,
    *,
    expected_source_revision: str = FULL_REFERENCE_REVISION,
) -> Q8Artifact:
    """Parse the Q8 sidecar at ``manifest_path``, verify the GGUF it names,
    and return a frozen :class:`Q8Artifact`.

    ``models_dir`` is the *data root* holding the GGUF; it defaults to the
    sidecar's parent directory. Note this root is for the **sidecar data
    only** — the closed 24-file ``models``/``assets`` bundle is a different,
    separately verified tree and a GGUF must never be placed inside it.

    ``expected_source_revision`` pins which base-model revision the GGUF must
    have been quantized from (default: the full-reference revision this engine
    profile was reviewed against). A sidecar declaring any other
    ``source_model_revision`` is refused with :class:`Q8SidecarFormatError`.

    Failures are typed :class:`~qwen_engine.manifest.ManifestError`
    subclasses (CLI exit 4). No downloads, no ML imports, no writes; the file
    is streamed, never slurped. The returned ``identity_sha256`` fingerprints
    the canonical manifest fields only (no ambient paths)."""
    if (not isinstance(expected_source_revision, str)
            or not _REVISION_RE.fullmatch(expected_source_revision)):
        raise Q8SidecarFormatError(
            "expected_source_revision must be a pinned 40-hex commit sha, got "
            f"{expected_source_revision!r}")
    manifest_path = Path(manifest_path)
    spec = load_q8_sidecar(manifest_path)
    if spec.source_model_revision != expected_source_revision:
        raise Q8SidecarFormatError(
            "Q8 sidecar source_model_revision mismatch: sidecar declares "
            f"{spec.source_model_revision!r} but this engine profile requires "
            f"{expected_source_revision!r} (the GGUF was quantized from a "
            "different base model revision — a new reviewed profile is "
            "required, never a relaxed check)")
    root = Path(models_dir) if models_dir is not None else manifest_path.resolve().parent
    path = verify_q8_file(spec, root)
    return Q8Artifact(spec=spec, path=path, identity_sha256=q8_identity_sha256(spec))
