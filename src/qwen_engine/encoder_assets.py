"""Memory-gated, encoder-only verification of the pinned reference assets.

Only processor/ and text_encoder/ are enumerated/read. Unrelated DiT/VAE or
fetch debris outside those directories is deliberately outside this scope.
No ML imports, writes, downloads, or generic whole-bundle resolver calls.
Fresh verification is point-in-time evidence, not protection against later
mutations before a library reopens the files; production needs settled assets.
"""
from __future__ import annotations

import hashlib
import os
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from qwen_engine.encoder_production import CANONICAL_DTYPES
from qwen_engine.local_bundle import (
    ENCODER_ROLES, FULL_REFERENCE_REPO, FULL_REFERENCE_REVISION,
    LOCAL_REFERENCE_LAYOUT, artifact_set_fingerprint, validate_local_reference_layout,
)
from qwen_engine.manifest import ModelArtifact, ModelManifest, ManifestError
from qwen_engine.mem_preflight import MemoryPreflightReport, require_encoder_memory_fit

CHECKPOINT_DTYPE = "bf16"  # reviewed full-reference encoder storage profile
ENCODER_FILENAMES = frozenset(f for role in ENCODER_ROLES for f in LOCAL_REFERENCE_LAYOUT[role])
_HASH_CHUNK = 1024 * 1024
_STAT_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")


class EncoderAssetError(ManifestError):
    """Unsafe/missing/unexpected or byte-mismatched encoder component asset."""


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(info, field) for field in _STAT_FIELDS)


def _checkpoint_bytes(artifacts: tuple[ModelArtifact, ...]) -> int:
    return sum(a.bytes for a in artifacts if a.role == "text_encoder" and a.identity.endswith(".safetensors"))


@dataclass(frozen=True)
class VerifiedEncoderAssets:
    """Verification result, not an unforgeable certificate on direct construction.

    Obtain from verify_encoder_assets, never hand-build a production proof.
    Published paths are canonical absolute paths; no ambient HF cache/model ID.
    """
    root: Path
    artifacts: tuple[ModelArtifact, ...]
    memory_report: MemoryPreflightReport

    def __post_init__(self) -> None:
        if (not isinstance(self.root, Path) or not self.root.is_absolute()
                or not isinstance(self.artifacts, tuple)
                or any(not isinstance(a, ModelArtifact) for a in self.artifacts)
                or len(self.artifacts) != len(ENCODER_FILENAMES)
                or {a.identity for a in self.artifacts} != ENCODER_FILENAMES
                or any(a.repo != FULL_REFERENCE_REPO or a.revision != FULL_REFERENCE_REVISION
                       or a.role not in ENCODER_ROLES
                       or a.identity not in LOCAL_REFERENCE_LAYOUT[a.role] for a in self.artifacts)
                or not isinstance(self.memory_report, MemoryPreflightReport)
                or self.memory_report.decision != "proceed"
                or self.memory_report.checkpoint_bytes != _checkpoint_bytes(self.artifacts)):
            raise EncoderAssetError("invalid encoder verification result")

    @property
    def component_dirs(self):
        return MappingProxyType({role: self.root / role for role in ENCODER_ROLES})

    @property
    def assets(self):
        return MappingProxyType({role: tuple(self.root / a.identity for a in self.artifacts if a.role == role)
                                 for role in ENCODER_ROLES})

    @property
    def encoder_artifact_set_sha256(self) -> str:
        return artifact_set_fingerprint(self.artifacts)

    @property
    def checkpoint_bytes(self) -> int:
        return _checkpoint_bytes(self.artifacts)


def _canonical_root(models_dir: str | Path) -> Path:
    try:
        root = Path(models_dir).resolve(strict=True)
        if not root.is_dir():
            raise EncoderAssetError("model root must be a directory")
        return root
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise EncoderAssetError(f"cannot resolve model root: {exc}") from exc


def _open_dir(path, *, dir_fd=None) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
                   dir_fd=dir_fd)


def _scan_component(fd: int, role: str) -> dict[str, tuple[int, ...]]:
    """Closed, flat reviewed component tree; never follow a symlink or recurse."""
    expected = {Path(f).name for f in LOCAL_REFERENCE_LAYOUT[role]}
    entries = {}
    with os.scandir(fd) as scan:
        for entry in scan:
            if entry.name not in expected:
                raise EncoderAssetError(f"{role}/{entry.name}: unexpected component entry")
            info = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise EncoderAssetError(f"{role}/{entry.name}: expected regular file, no links/special entries")
            entries[entry.name] = _signature(info)
    missing = sorted(expected - set(entries))
    if missing:
        raise EncoderAssetError(f"{role}: missing component files: {', '.join(missing)}")
    return entries


@contextmanager
def _preflight_open_tree(root: Path):
    """Own opened directory fds; cleanup every partial setup on failure."""
    with ExitStack() as cleanup:
        root_fd = _open_dir(root)
        cleanup.callback(os.close, root_fd)
        components = {}
        for role in ENCODER_ROLES:
            fd = _open_dir(role, dir_fd=root_fd)
            cleanup.callback(os.close, fd)
            components[role] = fd
            _scan_component(fd, role)
        yield root_fd, components


def preflight_encoder_tree(models_dir: str | Path) -> tuple[str, ...]:
    """Structural check only, before model-library imports; reads no contents.

    This intentionally does not need/check DiT, VAE, scheduler or root-level
    pipeline files. It never means the 16 model files' bytes were verified.
    """
    try:
        root = _canonical_root(models_dir)
        with _preflight_open_tree(root):
            return tuple(sorted((*ENCODER_ROLES, *ENCODER_FILENAMES)))
    except OSError as exc:
        raise EncoderAssetError(f"encoder component tree refused: {exc}") from exc


def _verify_one(artifact: ModelArtifact, component_fd: int) -> tuple[int, ...]:
    name = Path(artifact.identity).name
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=component_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != artifact.bytes:
            raise EncoderAssetError(f"{artifact.identity}: not a regular file of expected size {artifact.bytes}")
        digest, count = hashlib.sha256(), 0
        while True:
            block = os.read(fd, min(_HASH_CHUNK, artifact.bytes + 1 - count))
            if not block:
                break
            count += len(block)
            if count > artifact.bytes:
                raise EncoderAssetError(f"{artifact.identity}: grew during verification")
            digest.update(block)
        after = os.fstat(fd)
        published = os.stat(name, dir_fd=component_fd, follow_symlinks=False)
        if (count != artifact.bytes or _signature(before) != _signature(after)
                or _signature(before) != _signature(published)):
            raise EncoderAssetError(f"{artifact.identity}: changed during verification")
        if digest.hexdigest() != artifact.sha256:
            raise EncoderAssetError(f"{artifact.identity}: sha256 mismatch")
        return _signature(after)
    finally:
        os.close(fd)


def verify_encoder_assets(manifest: ModelManifest, models_dir: str | Path, *,
                          target_dtype: str, headroom_bytes: int,
                          attested_unbounded_host: bool = False) -> VerifiedEncoderAssets:
    """Validate full profile, memory prerequisite, then freshly verify 16 files.

    Headroom is required, never part of encoding identity. There is no public
    memory-snapshot/provider/trusted-root bypass: actual discovery must proceed,
    and the only alternative is the explicit boolean operator attestation
    (``attested_unbounded_host``), which carries no numbers and still applies
    a freshly read host availability.
    The narrow file set is checked before hashing and checked again afterwards.
    Other component files may be missing/partial; they are never inspected.
    """
    if not isinstance(manifest, ModelManifest):
        raise EncoderAssetError("a full-reference manifest is required")
    validate_local_reference_layout(manifest)
    if target_dtype not in CANONICAL_DTYPES:
        raise EncoderAssetError("target encoder dtype must be fp16/bf16/fp32")
    artifacts = tuple(a for a in manifest.artifacts if a.role in ENCODER_ROLES)
    memory_report = require_encoder_memory_fit(_checkpoint_bytes(artifacts),
        checkpoint_dtype=CHECKPOINT_DTYPE, target_dtype=target_dtype,
        headroom_bytes=headroom_bytes,
        attested_unbounded_host=attested_unbounded_host)
    # NO root canonicalization, directory access, asset read or ML import above this line.
    try:
        root = _canonical_root(models_dir)
        with _preflight_open_tree(root) as (root_fd, components):
            directory_ids = {role: _signature(os.fstat(fd)) for role, fd in components.items()}
            verified = {role: {} for role in ENCODER_ROLES}
            for artifact in artifacts:
                verified[artifact.role][Path(artifact.identity).name] = _verify_one(artifact, components[artifact.role])
            for role, fd in components.items():
                if (_scan_component(fd, role) != verified[role]
                        or _signature(os.fstat(fd)) != directory_ids[role]
                        or _signature(os.stat(role, dir_fd=root_fd, follow_symlinks=False)) != directory_ids[role]):
                    raise EncoderAssetError(f"{role}: component tree changed during verification")
            opened = os.fstat(root_fd)
            published = root.stat(follow_symlinks=False)
            if (opened.st_dev, opened.st_ino) != (published.st_dev, published.st_ino):
                raise EncoderAssetError("model root changed during verification")
            return VerifiedEncoderAssets(root, artifacts, memory_report)
    except OSError as exc:
        raise EncoderAssetError(f"encoder asset verification refused: {exc}") from exc
