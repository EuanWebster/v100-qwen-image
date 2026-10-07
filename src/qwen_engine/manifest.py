"""Versioned model-manifest schema + **no-download** resolver (plan R3, task 1.2).

Stdlib-only. The manifest (``models/manifest.json``) pins, per artifact, the
``repo`` / ``revision`` / ``filename`` / ``sha256`` / ``bytes`` of every model
file a run requires. The resolver *verifies* those claims on disk and never
downloads: a missing or mismatched artifact is a typed, structured error that
names the repo/revision/file so the operator can fetch it deliberately
(fetching is a separate documented action, never implicit).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Schema version of ``models/manifest.json`` (finalize values in task 2.1).
MANIFEST_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Artifact states reported in dry-run status dicts.
STATE_PRESENT = "present"
STATE_MISSING = "missing"
STATE_SIZE_MISMATCH = "size_mismatch"
STATE_HASH_MISMATCH = "hash_mismatch"
STATE_UNREADABLE = "unreadable"

_HASH_CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelArtifact:
    """One pinned model file: where it came from and what it must look like."""

    role: str
    repo: str
    revision: str
    filename: str
    sha256: str
    bytes: int

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
    """The manifest file itself is malformed; its message names the field."""


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


class ArtifactSizeMismatchError(ArtifactError):
    problem = "byte size mismatch"

    def __init__(self, artifact: ModelArtifact, expected_bytes: int, actual_bytes: int) -> None:
        self.expected_bytes = expected_bytes
        self.actual_bytes = actual_bytes
        super().__init__(
            artifact, f"expected {expected_bytes} bytes, found {actual_bytes}"
        )


class ArtifactHashMismatchError(ArtifactError):
    problem = "sha256 mismatch"

    def __init__(self, artifact: ModelArtifact, expected: str, actual: str) -> None:
        self.expected_sha256 = expected
        self.actual_sha256 = actual
        super().__init__(artifact, f"expected {expected}, computed {actual}")


class ArtifactUnreadableError(ArtifactError):
    problem = "artifact unreadable"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _artifact_from_dict(index: int, data: Any, errors: list[str]) -> ModelArtifact | None:
    def bad(field: str, msg: str) -> None:
        errors.append(f"artifacts[{index}].{field}: {msg}")

    if not isinstance(data, Mapping):
        bad("(entry)", f"must be an object, got {type(data).__name__}")
        return None
    art: dict[str, Any] = dict(data)
    for field in ("role", "repo", "revision", "filename"):
        value = art.get(field)
        if not isinstance(value, str) or not value.strip():
            bad(field, f"must be a non-empty string, got {value!r}")
            return None
    filename = art["filename"]
    fpath = Path(filename)
    if fpath.is_absolute() or ".." in fpath.parts or not filename:
        bad("filename", f"must be a safe relative path, got {filename!r}")
        return None
    sha = art.get("sha256")
    if not isinstance(sha, str) or not _SHA256_RE.match(sha):
        bad("sha256", f"must be 64 lowercase hex chars, got {sha!r}")
        return None
    size = art.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        bad("bytes", f"must be a positive integer, got {size!r}")
        return None
    return ModelArtifact(
        role=art["role"], repo=art["repo"], revision=art["revision"],
        filename=filename, sha256=sha, bytes=size,
    )


def manifest_from_dict(data: Any) -> ModelManifest:
    """Validate a manifest mapping; raise :class:`ManifestFormatError` naming
    every offending field."""
    errors: list[str] = []
    if not isinstance(data, Mapping):
        raise ManifestFormatError(f"manifest: must be a JSON object, got {type(data).__name__}")
    version = data.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"schema_version: expected {MANIFEST_SCHEMA_VERSION}, got {version!r}"
        )
    raw = data.get("artifacts")
    artifacts: list[ModelArtifact] = []
    if not isinstance(raw, list) or not raw:
        errors.append("artifacts: must be a non-empty list")
    else:
        seen: set[str] = set()
        for i, entry in enumerate(raw):
            art = _artifact_from_dict(i, entry, errors)
            if art is None:
                continue
            if art.role in seen:
                errors.append(f"artifacts[{i}].role: duplicate role {art.role!r}")
                continue
            seen.add(art.role)
            artifacts.append(art)
    if errors:
        raise ManifestFormatError("invalid model manifest: " + "; ".join(errors))
    return ModelManifest(schema_version=version, artifacts=tuple(artifacts))


def load_manifest(path: str | Path) -> ModelManifest:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestFormatError(f"manifest file not readable: {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestFormatError(f"manifest file is not valid JSON: {path}: {exc}") from exc
    return manifest_from_dict(data)


# ---------------------------------------------------------------------------
# Resolution (verify only — never fetch)
# ---------------------------------------------------------------------------


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactStatus:
    artifact: ModelArtifact
    state: str
    error: ArtifactError | None = None

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
    resolved: dict[str, Path]
    statuses: tuple[ArtifactStatus, ...]
    errors: tuple[ArtifactError, ...]


def _verify(artifact: ModelArtifact, root: Path) -> ArtifactStatus:
    path = root / artifact.filename
    try:
        actual_size = path.stat().st_size
    except OSError:
        return ArtifactStatus(artifact, STATE_MISSING, ArtifactMissingError(artifact))
    if actual_size != artifact.bytes:
        return ArtifactStatus(
            artifact, STATE_SIZE_MISMATCH,
            ArtifactSizeMismatchError(artifact, artifact.bytes, actual_size),
        )
    try:
        actual_sha = _sha256_of(path)
    except OSError as exc:
        return ArtifactStatus(artifact, STATE_UNREADABLE,
                              ArtifactUnreadableError(artifact, str(exc)))
    if actual_sha != artifact.sha256:
        return ArtifactStatus(
            artifact, STATE_HASH_MISMATCH,
            ArtifactHashMismatchError(artifact, artifact.sha256, actual_sha),
        )
    return ArtifactStatus(artifact, STATE_PRESENT)


def resolve_manifest(manifest: ModelManifest, models_dir: str | Path) -> ResolutionReport:
    """Verify every artifact under ``models_dir``. Collects **all** problems;
    performs **no** network access and no downloads."""
    root = Path(models_dir)
    statuses = tuple(_verify(a, root) for a in manifest.artifacts)
    resolved = {
        s.artifact.role: root / s.artifact.filename
        for s in statuses if s.state == STATE_PRESENT
    }
    errors = tuple(s.error for s in statuses if s.error is not None)
    return ResolutionReport(
        ok=not errors, resolved=resolved, statuses=statuses, errors=errors
    )


def resolve_model_path(manifest: ModelManifest, models_dir: str | Path, role: str) -> Path:
    """Path for one verified artifact, or a typed :class:`ArtifactError`.

    Never downloads: a missing file raises :class:`ArtifactMissingError`
    naming repo/revision/file — fetching is an explicit operator action.
    """
    artifact = next((a for a in manifest.artifacts if a.role == role), None)
    if artifact is None:
        raise ManifestError(
            f"manifest has no artifact with role {role!r} "
            f"(known roles: {', '.join(a.role for a in manifest.artifacts)})"
        )
    status = _verify(artifact, Path(models_dir))
    if status.error is not None:
        raise status.error
    return Path(models_dir) / artifact.filename


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
    if not path.is_file():
        return {**out, "status": "manifest_file_missing"}
    try:
        manifest = load_manifest(path)
    except ManifestFormatError as exc:
        return {**out, "status": "manifest_invalid", "error": str(exc)}
    root = Path(models_dir) if models_dir is not None else path.parent
    report = resolve_manifest(manifest, root)
    out["schema_version"] = manifest.schema_version
    out["status"] = "ok" if report.ok else "incomplete"
    out["artifacts"] = [
        s.to_dict()
        for s in sorted(report.statuses, key=lambda s: s.artifact.role)
    ]
    return out
