"""Versioned model-manifest schema + **no-download** resolver (plan R3, task 1.2).

Stdlib-only. The manifest (``models/manifest.json``) pins, per artifact, the
``repo`` / ``revision`` / ``filename`` / ``sha256`` / ``bytes`` of every model
file a run requires. A *role* may cover several files (sharded weights, index
files, configs); identity is the **unique normalized filename**, not the role.
The resolver *verifies* those claims on disk and never downloads: a missing or
mismatched artifact is a typed, structured error that names the
repo/revision/file so the operator can fetch it deliberately (fetching is a
separate documented action, never implicit).

Validation runs on **direct dataclass construction as well as JSON parsing**,
so a hand-built :class:`ModelManifest` cannot smuggle in a floating schema
version, a branch-name "revision", or an unsafe filename.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat as stat_module
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

#: Schema version of ``models/manifest.json`` (finalize values in task 2.1).
MANIFEST_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
#: A pinned revision is an exact 40-hex commit sha — never a branch or tag.
_REVISION_RE = re.compile(r"[0-9a-f]{40}")

_ARTIFACT_KEYS = frozenset({"role", "repo", "revision", "filename", "sha256", "bytes"})
_MANIFEST_KEYS = frozenset({"schema_version", "artifacts"})

#: Artifact states reported in dry-run status dicts.
STATE_PRESENT = "present"
STATE_MISSING = "missing"
STATE_SIZE_MISMATCH = "size_mismatch"
STATE_HASH_MISMATCH = "hash_mismatch"
STATE_UNREADABLE = "unreadable"
STATE_PATH_ESCAPE = "path_escape"

_HASH_CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# Field validators (shared by JSON parsing and direct construction)
# ---------------------------------------------------------------------------


def _safe_filename(filename: Any) -> str | None:
    """Return an error string if ``filename`` is not a safe relative path."""
    if not isinstance(filename, str) or not filename:
        return f"must be a non-empty string, got {filename!r}"
    if "\0" in filename:
        return f"contains a NUL byte: {filename!r}"
    if PurePosixPath(filename).is_absolute():
        return f"must be relative, got absolute path {filename!r}"
    parts = filename.split("/")
    for part in parts:
        if part == "":
            return f"empty path component in {filename!r}"
        if part == ".":
            return f"dot path component in {filename!r}"
        if part == "..":
            return f"parent-directory component in {filename!r}"
    return None


def _normalized(filename: str) -> str:
    """Canonical identity of a validated safe relative filename."""
    return "/".join(filename.split("/"))


def _check_artifact_fields(art: Mapping[str, Any], at: str,
                           errors: list[str]) -> None:
    """Append every field-level problem with ``art`` to *errors* (all-problems
    report: every field is checked independently)."""
    def bad(field: str, msg: str) -> None:
        errors.append(f"{at}.{field}: {msg}")

    for field in ("role", "repo", "revision"):
        value = art.get(field)
        if not isinstance(value, str) or not value.strip():
            bad(field, f"must be a non-empty string, got {value!r}")
    revision = art.get("revision")
    if isinstance(revision, str) and revision.strip():
        if not _REVISION_RE.fullmatch(revision):
            bad("revision", "must be a pinned 40-hex commit sha (not a branch "
                            f"or tag), got {revision!r}")
    filename = art.get("filename")
    problem = _safe_filename(filename)
    if problem is not None:
        bad("filename", problem)
    sha = art.get("sha256")
    if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
        bad("sha256", f"must be 64 lowercase hex chars, got {sha!r}")
    size = art.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        bad("bytes", f"must be a positive integer, got {size!r}")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelArtifact:
    """One pinned model file: where it came from and what it must look like.

    Validated on *every* construction (direct or parsed) — a frozen dataclass
    is not a validation exemption.
    """

    role: str
    repo: str
    revision: str
    filename: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        errors: list[str] = []
        _check_artifact_fields(self.to_dict(), "artifact", errors)
        if errors:
            raise ManifestFormatError("invalid model artifact: " + "; ".join(errors))

    @property
    def identity(self) -> str:
        """Unique normalized filename identity (roles may repeat; files may not)."""
        return _normalized(self.filename)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "repo": self.repo,
            "revision": self.revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    artifacts: tuple[ModelArtifact, ...]

    def __post_init__(self) -> None:
        errors: list[str] = []
        version = self.schema_version
        if not isinstance(version, int) or isinstance(version, bool) or version != MANIFEST_SCHEMA_VERSION:
            errors.append(
                f"schema_version: expected exact int {MANIFEST_SCHEMA_VERSION}, "
                f"got {version!r}"
            )
        raw = self.artifacts
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
            errors.append(f"artifacts: must be a tuple of ModelArtifact, got {type(raw).__name__}")
            if errors:
                raise ManifestFormatError("invalid model manifest: " + "; ".join(errors))
        items = list(raw)
        if not items:
            errors.append("artifacts: must be a non-empty collection")
        bad_items = [a for a in items if not isinstance(a, ModelArtifact)]
        if bad_items:
            errors.append(
                "artifacts: every entry must be a ModelArtifact, got "
                + ", ".join(type(a).__name__ for a in bad_items)
            )
        if not errors:
            seen: dict[str, int] = {}
            for i, art in enumerate(items):
                ident = art.identity
                if ident in seen:
                    errors.append(
                        f"artifacts[{i}].filename: duplicate artifact identity "
                        f"{ident!r} (also at artifacts[{seen[ident]}]; roles may "
                        f"repeat, filenames may not)"
                    )
                else:
                    seen[ident] = i
        if errors:
            raise ManifestFormatError("invalid model manifest: " + "; ".join(errors))
        object.__setattr__(self, "artifacts", tuple(items))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class ManifestError(Exception):
    """Base class for manifest/asset problems. Exit code 4 in the CLI."""


class ManifestFormatError(ManifestError):
    """The manifest file (or object) is malformed; its message names every
    offending field."""


class AmbiguousRoleError(ManifestError):
    """The role covers several files; use :func:`resolve_model_paths`."""


class ArtifactError(ManifestError):
    """One artifact failed verification; names repo/revision/file."""

    problem = "failed verification"

    def __init__(self, artifact: ModelArtifact, detail: str = "") -> None:
        self.artifact = artifact
        self.role = artifact.role
        self.repo = artifact.repo
        self.revision = artifact.revision
        self.filename = artifact.filename
        self.detail = detail
        base = (
            f"{self.problem}: '{artifact.filename}' "
            f"(role={artifact.role}, repo={artifact.repo}, revision={artifact.revision})"
        )
        super().__init__(f"{base}: {detail}" if detail else base)


class ArtifactMissingError(ArtifactError):
    problem = "artifact missing on disk"


class ArtifactPathError(ArtifactError):
    """The artifact resolves outside the model storage root (symlink escape)."""

    problem = "artifact path escapes model root"


class ArtifactSizeMismatchError(ArtifactError):
    problem = "byte size mismatch"

    def __init__(self, artifact: ModelArtifact, expected_bytes: int, actual_bytes: int,
                 detail: str = "") -> None:
        self.expected_bytes = expected_bytes
        self.actual_bytes = actual_bytes
        msg = f"expected {expected_bytes} bytes, found {actual_bytes}"
        super().__init__(artifact, f"{msg} ({detail})" if detail else msg)


class ArtifactHashMismatchError(ArtifactError):
    problem = "sha256 mismatch"

    def __init__(self, artifact: ModelArtifact, expected: str, actual: str) -> None:
        self.expected_sha256 = expected
        self.actual_sha256 = actual
        super().__init__(artifact, f"expected {expected}, computed {actual}")


class ArtifactUnreadableError(ArtifactError):
    """Exists but cannot be read/hashed (permissions, I/O, not a regular file,
    changed mid-verification). Never blocks: FIFOs and directories are
    identified from ``stat`` *before* any open."""

    problem = "artifact unreadable"


def _os_detail(exc: OSError) -> str:
    return f"errno {exc.errno} ({exc.strerror or type(exc).__name__}): {exc.filename or ''}".rstrip()


def _resolve_or_fail(path: Path) -> tuple[Path | None, str | None]:
    """``Path.resolve`` is NOT exception-free on supported interpreters:
    CPython 3.11 may raise ``RuntimeError`` on a symlink loop and an
    ``OSError`` (permission on a parent component, …) is always possible.
    No resolution failure may escape the typed boundary as a raw error —
    every one becomes a structured detail string the caller reports as an
    :class:`ArtifactUnreadableError` / unreadable status. (Observed:
    3.13.5 tolerates symlink loops at default ``resolve()``; re-checked
    2026-10-07 on the CPython 3.11.15 build used for the R7 evidence —
    default ``resolve()`` tolerates loops there too, raising only
    ``FileNotFoundError`` under ``strict=True``. The ``RuntimeError``
    variant is documented for some 3.11 patch levels; the guard covers
    all of it either way.)"""
    try:
        return path.resolve(), None
    except (OSError, RuntimeError, ValueError) as exc:
        detail = (f"{type(exc).__name__}: {exc}"
                  + (f" [{exc.strerror}]" if isinstance(exc, OSError)
                     and exc.strerror else ""))
        return None, detail


# ---------------------------------------------------------------------------
# Parsing (strict known keys; all problems reported)
# ---------------------------------------------------------------------------


def _artifact_from_dict(index: int, data: Any, errors: list[str]) -> ModelArtifact | None:
    def bad(field: str, msg: str) -> None:
        errors.append(f"artifacts[{index}].{field}: {msg}")

    if not isinstance(data, Mapping):
        bad("(entry)", f"must be an object, got {type(data).__name__}")
        return None
    art: dict[str, Any] = dict(data)
    for key in sorted(set(art) - _ARTIFACT_KEYS):
        bad(key, "unknown artifact key")
    _check_artifact_fields(art, f"artifacts[{index}]", errors)
    if any(err.startswith(f"artifacts[{index}].") for err in errors):
        return None
    return ModelArtifact(**art)


def manifest_from_dict(data: Any) -> ModelManifest:
    """Validate a manifest mapping; raise :class:`ManifestFormatError` naming
    every offending field."""
    errors: list[str] = []
    if not isinstance(data, Mapping):
        raise ManifestFormatError(f"manifest: must be a JSON object, got {type(data).__name__}")
    for key in sorted(set(data) - _MANIFEST_KEYS):
        errors.append(f"{key}: unknown manifest key")
    version = data.get("schema_version")
    if (not isinstance(version, int) or isinstance(version, bool)
            or version != MANIFEST_SCHEMA_VERSION):
        errors.append(
            f"schema_version: expected exact int {MANIFEST_SCHEMA_VERSION}, "
            f"got {version!r}"
        )
    raw = data.get("artifacts")
    artifacts: list[ModelArtifact] = []
    if not isinstance(raw, list) or not raw:
        errors.append("artifacts: must be a non-empty list")
    else:
        seen: dict[str, int] = {}
        for i, entry in enumerate(raw):
            art = _artifact_from_dict(i, entry, errors)
            if art is None:
                continue
            if art.identity in seen:
                errors.append(
                    f"artifacts[{i}].filename: duplicate artifact identity "
                    f"{art.identity!r} (also at artifacts[{seen[art.identity]}]; "
                    f"roles may repeat, filenames may not)"
                )
                continue
            seen[art.identity] = i
            artifacts.append(art)
    if errors:
        raise ManifestFormatError("invalid model manifest: " + "; ".join(errors))
    return ModelManifest(schema_version=version, artifacts=tuple(artifacts))


def load_manifest(path: str | Path) -> ModelManifest:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except (OSError, ValueError) as exc:  # ValueError: embedded NUL in path
        raise ManifestFormatError(
            f"manifest file not readable: {path}: {exc.strerror or exc if isinstance(exc, OSError) else exc}"
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestFormatError(f"manifest file is not valid UTF-8: {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestFormatError(f"manifest file is not valid JSON: {path}: {exc}") from exc
    return manifest_from_dict(data)


# ---------------------------------------------------------------------------
# Resolution (verify only — never fetch)
# ---------------------------------------------------------------------------


class _FileChangedDuringHash(OSError):
    """Internal: the open file mutated between fstat and the end of hashing."""


def _hash_and_check(fd: int, pre: os.stat_result) -> str:
    """Stream-hash an already-open descriptor; raise :class:`_FileChangedDuringHash`
    if the file changed (size or mtime) while hashing."""
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while chunk := os.read(fd, _HASH_CHUNK):
        digest.update(chunk)
    after = os.fstat(fd)
    if after.st_size != pre.st_size:
        raise _FileChangedDuringHash(
            f"file changed during verification (size {pre.st_size} -> {after.st_size})"
        )
    if (getattr(after, "st_mtime_ns", None) is not None
            and after.st_mtime_ns != pre.st_mtime_ns):
        raise _FileChangedDuringHash("file modified during verification (mtime changed)")
    return digest.hexdigest()


def _verify_one(artifact: ModelArtifact, root: Path,
                root_resolved: Path) -> ArtifactStatus:
    path = root / artifact.filename
    resolved, fail = _resolve_or_fail(path)
    if resolved is None:  # resolution itself failed: typed, never raw
        return ArtifactStatus(artifact, STATE_UNREADABLE,
                              ArtifactUnreadableError(
                                  artifact, f"cannot resolve {path}: {fail}"))
    if resolved != root_resolved and root_resolved not in resolved.parents:
        # Symlink (or platform quirk) took us outside the model storage root.
        return ArtifactStatus(
            artifact, STATE_PATH_ESCAPE,
            ArtifactPathError(artifact, f"resolves to {resolved}, outside {root_resolved}"),
        )
    try:
        st = resolved.stat()
    except FileNotFoundError:
        return ArtifactStatus(artifact, STATE_MISSING, ArtifactMissingError(artifact))
    except OSError as exc:  # permission on a path component, ELOOP, ...
        return ArtifactStatus(artifact, STATE_UNREADABLE,
                              ArtifactUnreadableError(artifact, _os_detail(exc)))
    if not stat_module.S_ISREG(st.st_mode):
        # FIFOs block on open and directories read() — never hash either.
        kind = ("directory" if stat_module.S_ISDIR(st.st_mode)
                else "FIFO" if stat_module.S_ISFIFO(st.st_mode)
                else f"non-regular file (mode {oct(st.st_mode)})")
        return ArtifactStatus(
            artifact, STATE_UNREADABLE,
            ArtifactUnreadableError(artifact, f"not a regular file: {kind}"),
        )
    if st.st_size != artifact.bytes:
        return ArtifactStatus(
            artifact, STATE_SIZE_MISMATCH,
            ArtifactSizeMismatchError(artifact, artifact.bytes, st.st_size),
        )
    try:
        fd = os.open(resolved, os.O_RDONLY)
    except FileNotFoundError:
        return ArtifactStatus(artifact, STATE_MISSING, ArtifactMissingError(artifact))
    except OSError as exc:
        return ArtifactStatus(artifact, STATE_UNREADABLE,
                              ArtifactUnreadableError(artifact, _os_detail(exc)))
    try:
        try:
            pre = os.fstat(fd)
        except OSError as exc:
            return ArtifactStatus(artifact, STATE_UNREADABLE,
                                  ArtifactUnreadableError(artifact, _os_detail(exc)))
        if not stat_module.S_ISREG(pre.st_mode):
            return ArtifactStatus(
                artifact, STATE_UNREADABLE,
                ArtifactUnreadableError(artifact, "not a regular file (swapped after stat)"),
            )
        if pre.st_size != artifact.bytes:
            return ArtifactStatus(
                artifact, STATE_SIZE_MISMATCH,
                ArtifactSizeMismatchError(artifact, artifact.bytes, pre.st_size),
            )
        try:
            actual_sha = _hash_and_check(fd, pre)
        except _FileChangedDuringHash as exc:
            return ArtifactStatus(
                artifact, STATE_SIZE_MISMATCH,
                ArtifactSizeMismatchError(artifact, artifact.bytes,
                                          os.fstat(fd).st_size,
                                          detail="file changed during hashing"),
            )
        except OSError as exc:
            return ArtifactStatus(artifact, STATE_UNREADABLE,
                                  ArtifactUnreadableError(
                                      artifact, str(exc) or _os_detail(exc)))
    finally:
        os.close(fd)
    if actual_sha != artifact.sha256:
        return ArtifactStatus(
            artifact, STATE_HASH_MISMATCH,
            ArtifactHashMismatchError(artifact, artifact.sha256, actual_sha),
        )
    return ArtifactStatus(artifact, STATE_PRESENT, path=path)


@dataclass(frozen=True)
class ArtifactStatus:
    artifact: ModelArtifact
    state: str
    error: ArtifactError | None = None
    path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        a = self.artifact
        return {
            "filename": a.filename,
            "repo": a.repo,
            "revision": a.revision,
            "role": a.role,
            "status": self.state,
        }


@dataclass(frozen=True)
class ResolutionReport:
    ok: bool
    #: Verified paths grouped by role; a role may cover several files
    #: (shards, indexes, configs), so every value is a tuple.
    resolved: dict[str, tuple[Path, ...]]
    statuses: tuple[ArtifactStatus, ...]
    errors: tuple[ArtifactError, ...]


def resolve_manifest(manifest: ModelManifest, models_dir: str | Path) -> ResolutionReport:
    """Verify every artifact under ``models_dir``. Collects **all** problems;
    performs **no** network access and no downloads.

    Symlinks are allowed when their *resolved* targets stay inside the model
    storage root (HF caches legitimately use them); anything that canonicalizes
    outside the root is a typed :class:`ArtifactPathError`.
    """
    root = Path(models_dir)
    root_resolved, root_fail = _resolve_or_fail(root)
    if root_resolved is None:
        # The storage root itself cannot be canonicalized: every artifact
        # is reported unreadable through the same structured channel —
        # resolve_manifest never lets a raw resolve error escape.
        detail = f"model root cannot be resolved: {root} ({root_fail})"
        statuses = tuple(
            ArtifactStatus(a, STATE_UNREADABLE,
                           ArtifactUnreadableError(a, detail))
            for a in manifest.artifacts)
        return ResolutionReport(ok=False, resolved={}, statuses=statuses,
                                errors=tuple(s.error for s in statuses))
    statuses = tuple(_verify_one(a, root, root_resolved) for a in manifest.artifacts)
    resolved: dict[str, tuple[Path, ...]] = {}
    for s in statuses:
        if s.state == STATE_PRESENT and s.path is not None:
            resolved.setdefault(s.artifact.role, ())
            resolved[s.artifact.role] += (s.path,)
    errors = tuple(s.error for s in statuses if s.error is not None)
    return ResolutionReport(
        ok=not errors, resolved=resolved, statuses=statuses, errors=errors
    )


def _role_artifacts(manifest: ModelManifest, role: str) -> tuple[ModelArtifact, ...]:
    return tuple(a for a in manifest.artifacts if a.role == role)


def resolve_model_paths(manifest: ModelManifest, models_dir: str | Path,
                        role: str) -> tuple[Path, ...]:
    """Verified paths for **every** artifact with ``role`` (shards included),
    or a typed :class:`ArtifactError`. Never downloads."""
    arts = _role_artifacts(manifest, role)
    if not arts:
        raise ManifestError(
            f"manifest has no artifact with role {role!r} "
            f"(known roles: {', '.join(sorted({a.role for a in manifest.artifacts}))})"
        )
    root = Path(models_dir)
    root_resolved, root_fail = _resolve_or_fail(root)
    if root_resolved is None:
        raise ArtifactUnreadableError(
            arts[0], f"model root cannot be resolved: {root} ({root_fail})")
    statuses = [_verify_one(a, root, root_resolved) for a in arts]
    for status in statuses:
        if status.error is not None:
            raise status.error
    return tuple(s.path for s in statuses)


def resolve_model_path(manifest: ModelManifest, models_dir: str | Path, role: str) -> Path:
    """Path for a role that names exactly **one** verified artifact.

    Ambiguous multi-file roles (shards/indexes) are rejected — use
    :func:`resolve_model_paths`; silently handing back one shard of a
    sharded model would be worse than refusing. Never downloads.
    """
    arts = _role_artifacts(manifest, role)
    if not arts:
        raise ManifestError(
            f"manifest has no artifact with role {role!r} "
            f"(known roles: {', '.join(sorted({a.role for a in manifest.artifacts}))})"
        )
    if len(arts) > 1:
        raise AmbiguousRoleError(
            f"role {role!r} covers {len(arts)} artifacts "
            f"({', '.join(a.filename for a in arts)}); use resolve_model_paths()"
        )
    paths = resolve_model_paths(manifest, models_dir, role)
    assert len(paths) == 1
    return paths[0]


# ---------------------------------------------------------------------------
# Dry-run status (deterministic, timestamp-free — M1 criterion 4)
# ---------------------------------------------------------------------------


def manifest_status(
    manifest_path: str | Path | None, models_dir: str | Path | None
) -> dict[str, Any]:
    """Deterministic status dict for ``--dry-run``; never raises, never
    downloads, never fabricates: unresolved assets are reported as such."""
    if manifest_path is None:
        return {"configured": False, "status": "unconfigured", "artifacts": []}
    path = Path(manifest_path)
    out: dict[str, Any] = {"configured": True, "path": str(path), "artifacts": []}
    try:
        manifest = load_manifest(path)
    except ManifestFormatError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            return {**out, "status": "manifest_file_missing"}
        return {**out, "status": "manifest_invalid", "error": str(exc)}
    root = Path(models_dir) if models_dir is not None else path.parent
    report = resolve_manifest(manifest, root)
    out["schema_version"] = manifest.schema_version
    out["status"] = "ok" if report.ok else "incomplete"
    out["artifacts"] = [
        s.to_dict()
        for s in sorted(report.statuses, key=lambda s: (s.artifact.role, s.artifact.filename))
    ]
    return out
