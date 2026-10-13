"""Production request/implementation identity, without ML imports or asset reads.

Observe packages in THIS interpreter, check reviewed pins, and fingerprint this
package's own Python sources. Metadata is source/version evidence, not proof
that installed package bytes are untampered or that a model can load.

The manifest-only projection does not verify assets. The bundle projection
trusts its caller to have obtained a fresh resolver-verified bundle; its checks
establish coherence, not an unforgeable verification token. The future worker
must re-verify its narrow encoder subtree after memory preflight, and compare
the request against identities observed in the worker's own interpreter.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from qwen_engine._jsonsafe import reject_constant
from qwen_engine.config import GenerationConfig
from qwen_engine.encoder import (
    REQUEST_SCHEMA, ComponentIdentity, EncoderImplementationIdentity,
    EncoderRequest, EncoderRequestError, EncoderSetupError,
    PINNED_DIFFUSERS_COMMIT, PINNED_TRANSFORMERS_COMMIT,
)
from qwen_engine.encoder_production import CANONICAL_DTYPES, validate_production_request
from qwen_engine.local_bundle import (
    LOCAL_REFERENCE_LAYOUT, LOCAL_REFERENCE_PROFILE, ROLE_DIRECTORIES,
    LocalReferenceBundle, artifact_set_fingerprint,
    reference_bundle_fingerprints, validate_local_reference_layout,
)
from qwen_engine.manifest import ModelManifest

PACKAGE_VERSIONS = MappingProxyType({
    "torch": "2.14.1+cu126", "diffusers": "0.41.0.dev0",
    "transformers": "5.17.0", "safetensors": "0.8.0",
})
VCS_PINS = MappingProxyType({
    "diffusers": ("https://github.com/huggingface/diffusers.git", PINNED_DIFFUSERS_COMMIT),
    "transformers": ("https://github.com/huggingface/transformers.git", PINNED_TRANSFORMERS_COMMIT),
})
# No reviewed torchvision artifact is in requirements.lock. Never infer a
# compatible version or permit one merely because somebody installed it.
PINNED_TORCHVISION_VERSION: str | None = None
PACKAGE_NAMES = (*PACKAGE_VERSIONS, "torchvision")
SOURCE_DIGEST_SCHEMA = "qwen_engine.worker-source.v1"
PROMPT_EXTRACTION_REVISION = "qwen-image21.upstream-pre-norm.t2i.v1"
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 16 * 1024 * 1024
MAX_SOURCE_FILES = 256
MAX_DIRECT_URL_CHARS = 16 * 1024
TOKENIZER_FILES = frozenset({
    "processor/added_tokens.json", "processor/merges.txt",
    "processor/special_tokens_map.json", "processor/tokenizer.json",
    "processor/tokenizer_config.json", "processor/vocab.json",
})


@dataclass(frozen=True)
class PackageObservation:
    version: str | None
    url: str | None = None
    vcs: str | None = None
    commit: str | None = None

    def __post_init__(self) -> None:
        if any(v is not None and (not isinstance(v, str) or not v or v != v.strip())
               for v in (self.version, self.url, self.vcs, self.commit)):
            raise EncoderSetupError("invalid installed package observation")
        if self.version is None and any(v is not None for v in (self.url, self.vcs, self.commit)):
            raise EncoderSetupError("absent package cannot carry source metadata")


@dataclass(frozen=True)
class InstalledEncoderPackages:
    executable: str
    python_version: tuple[int, int, int]
    packages: Mapping[str, PackageObservation]

    def __post_init__(self) -> None:
        if (not isinstance(self.executable, str) or not self.executable
                or not isinstance(self.python_version, tuple)
                or len(self.python_version) != 3
                or any(type(v) is not int or v < 0 for v in self.python_version)
                or not isinstance(self.packages, Mapping)
                or set(self.packages) != set(PACKAGE_NAMES)
                or any(not isinstance(v, PackageObservation) for v in self.packages.values())):
            raise EncoderSetupError("invalid installed encoder package snapshot")
        object.__setattr__(self, "packages", MappingProxyType(dict(self.packages)))


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate direct_url key {key!r}")
        result[key] = value
    return result


def observe_installed_packages() -> InstalledEncoderPackages:
    """Read current-interpreter distribution metadata, never import ML.

    Missing distributions are explicit observations; broken metadata raises a
    setup error rather than being relabelled as absent. No other interpreter
    is inspected and no dependency is installed by this function.
    """
    observed = {}
    for name in PACKAGE_NAMES:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            observed[name] = PackageObservation(None)
            continue
        except Exception as exc:
            raise EncoderSetupError(f"cannot inspect installed {name} metadata: {exc}") from exc
        try:
            version = dist.version
            if not isinstance(version, str) or not version.strip():
                raise ValueError("missing package version")
            direct = dist.read_text("direct_url.json")
            url = vcs = commit = None
            if direct is not None:
                if not isinstance(direct, str) or len(direct) > MAX_DIRECT_URL_CHARS:
                    raise ValueError("oversized/invalid direct_url metadata")
                doc = json.loads(direct, object_pairs_hook=_unique_pairs,
                                 parse_constant=reject_constant)
                if not isinstance(doc, dict) or not isinstance(doc.get("url"), str):
                    raise ValueError("direct_url needs an object and URL string")
                url = doc["url"]
                info = doc.get("vcs_info")
                if info is not None:
                    if set(doc) - {"url", "vcs_info"}:
                        raise ValueError("VCS subdirectories/mixed source metadata are unsupported")
                    if not isinstance(info, dict):
                        raise ValueError("vcs_info must be an object")
                    vcs, commit = info.get("vcs"), info.get("commit_id")
                    if not isinstance(vcs, str) or not isinstance(commit, str):
                        raise ValueError("vcs_info needs vcs/commit_id strings")
            observed[name] = PackageObservation(version, url, vcs, commit)
        except Exception as exc:
            raise EncoderSetupError(f"cannot inspect installed {name} metadata: {exc}") from exc
    return InstalledEncoderPackages(sys.executable, tuple(sys.version_info[:3]), observed)


def validate_installed_packages(packages: InstalledEncoderPackages) -> None:
    """Fail closed on wrong versions/provenance or the unreviewed vision gate."""
    if not isinstance(packages, InstalledEncoderPackages):
        raise EncoderSetupError("an InstalledEncoderPackages snapshot is required")
    errors = []
    if packages.python_version[:2] not in ((3, 11), (3, 13)):
        errors.append("only Python 3.11/3.13 have a reviewed dependency lock")
    for name, version in PACKAGE_VERSIONS.items():
        got = packages.packages[name]
        if got.version != version:
            errors.append(f"{name}: expected {version}, observed {got.version or 'absent'}")
        if name in VCS_PINS:
            url, commit = VCS_PINS[name]
            if (got.url, got.vcs, got.commit) != (url, "git", commit):
                errors.append(f"{name}: expected exact reviewed git URL/commit")
        elif any(v is not None for v in (got.url, got.vcs, got.commit)):
            errors.append(f"{name}: unexpected direct-source installation")
    vision = packages.packages["torchvision"]
    if PINNED_TORCHVISION_VERSION is None:
        errors.append("torchvision: no reviewed pin in requirements.lock "
                      f"(observed {vision.version or 'absent'}); production loading is blocked")
    elif vision.version != PINNED_TORCHVISION_VERSION or any(
            v is not None for v in (vision.url, vision.vcs, vision.commit)):
        errors.append("torchvision: installed identity does not match the reviewed pin")
    if errors:
        raise EncoderSetupError("installed encoder packages refused: " + "; ".join(errors))


def _source_digest(root: Path) -> str:
    """Bounded source-only digest; the test seam is not a production root option."""
    rows = []
    total = 0
    try:
        if root.is_symlink() or not root.is_dir():
            raise EncoderSetupError("worker source root must be a real directory")
        def walk_error(exc: OSError) -> None:
            raise exc
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            if any((Path(directory) / d).is_symlink() for d in dirs):
                raise EncoderSetupError("worker source directory symlinks are refused")
            for filename in sorted(f for f in files if f.endswith(".py")):
                path = Path(directory) / filename
                relative = path.relative_to(root)
                if len(rows) >= MAX_SOURCE_FILES:
                    raise EncoderSetupError("worker source file-count cap exceeded")
                # Open every parent with NOFOLLOW using a directory fd. A
                # parent swapped to a symlink between enumeration and open
                # must not redirect source reads outside this package.
                parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    for part in relative.parts[:-1]:
                        child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                           dir_fd=parent_fd)
                        os.close(parent_fd)
                        parent_fd = child_fd
                    fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=parent_fd)
                    try:
                        before = os.fstat(fd)
                        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_SOURCE_BYTES:
                            raise EncoderSetupError("worker sources must be bounded regular files")
                        digest, count = hashlib.sha256(), 0
                        while True:
                            block = os.read(fd, min(65536, MAX_SOURCE_BYTES + 1 - count))
                            if not block:
                                break
                            count += len(block)
                            total += len(block)
                            if count > MAX_SOURCE_BYTES or total > MAX_SOURCE_TOTAL_BYTES:
                                raise EncoderSetupError("worker source byte cap exceeded")
                            digest.update(block)
                        after = os.fstat(fd)
                        published = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
                        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                        if count != before.st_size or any(
                                getattr(before, f) != getattr(after, f)
                                or getattr(before, f) != getattr(published, f) for f in fields):
                            raise EncoderSetupError("worker source changed during fingerprinting")
                        rows.append([relative.as_posix(), count, digest.hexdigest()])
                    finally:
                        os.close(fd)
                finally:
                    os.close(parent_fd)
    except OSError as exc:
        raise EncoderSetupError(f"cannot fingerprint worker source: {exc}") from exc
    if not rows:
        raise EncoderSetupError("worker source fingerprint requires Python source files")
    doc = {"schema": SOURCE_DIGEST_SCHEMA, "files": sorted(rows)}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def worker_source_digest() -> str:
    """Hash own .py names/lengths/content, never absolute paths or mtimes.

    Conservatively includes all qwen_engine source, including web modules.
    Source-less/zip-only deployments are not supported by this worker profile.
    No caller-selected root or manual revision overrides are accepted.
    """
    return _source_digest(Path(__file__).parent)


def _implementation_identity(manifest: ModelManifest) -> EncoderImplementationIdentity:
    packages = observe_installed_packages()
    validate_installed_packages(packages)
    facts = packages.packages
    return EncoderImplementationIdentity(
        encoder_artifact_set_sha256=reference_bundle_fingerprints(manifest).encoder,
        worker_protocol_revision=REQUEST_SCHEMA,
        worker_implementation_revision=f"{SOURCE_DIGEST_SCHEMA}:{worker_source_digest()}",
        prompt_extraction_revision=PROMPT_EXTRACTION_REVISION,
        diffusers_commit=facts["diffusers"].commit,
        transformers_commit=facts["transformers"].commit,
        torch_build=facts["torch"].version.split("+", 1)[1],
        torch_version=facts["torch"].version,
        safetensors_version=facts["safetensors"].version,
        torchvision_version=facts["torchvision"].version,
    )


def _project_request(manifest: ModelManifest, *, prompt: str, negative: str,
                     dtype: str, cfg_active: bool) -> EncoderRequest:
    if not isinstance(manifest, ModelManifest):
        raise EncoderRequestError("a full-reference ModelManifest is required")
    repo, revision = validate_local_reference_layout(manifest)
    if (not isinstance(prompt, str) or not prompt.strip()
            or not isinstance(negative, str) or dtype not in CANONICAL_DTYPES
            or type(cfg_active) is not bool or cfg_active and not negative.strip()):
        raise EncoderRequestError("invalid production prompt/dtype/CFG policy")
    artifacts = manifest.artifacts
    processor_digest = artifact_set_fingerprint(a for a in artifacts if a.role == "processor")
    tokenizer_digest = artifact_set_fingerprint(a for a in artifacts if a.identity in TOKENIZER_FILES)
    template = next(a for a in artifacts if a.identity == "processor/chat_template.jinja")
    fields = dict(prompt=prompt, negative_prompt=negative, dtype=dtype,
                  kwargs={"cfg_active": cfg_active}, model_repo=repo, model_revision=revision,
                  processor=ComponentIdentity(revision=revision, sha256=processor_digest),
                  tokenizer=ComponentIdentity(revision=revision, sha256=tokenizer_digest),
                  chat_template=ComponentIdentity(revision=revision, sha256=template.sha256))
    request = EncoderRequest(**fields, implementation_identity=_implementation_identity(manifest))
    validate_production_request(request)
    return request


def project_encoder_request(config: GenerationConfig, manifest: ModelManifest, *,
                            encoder_dtype: str) -> EncoderRequest:
    """Manifest-only identity projection, NOT asset verification or loading.

    Encoder dtype is mandatory and independent of DiT quantization. Generation
    scale/count/seed/dimensions/offload/headroom never enter encoding kwargs.
    """
    if not isinstance(config, GenerationConfig):
        raise EncoderRequestError("a GenerationConfig is required")
    return _project_request(manifest, prompt=config.prompt, negative=config.negative_prompt,
                            dtype=encoder_dtype, cfg_active=config.cfg_active)


def request_from_verified_bundle(config: GenerationConfig, manifest: ModelManifest,
                                 bundle: LocalReferenceBundle, *, encoder_dtype: str) -> EncoderRequest:
    """Project a caller's fresh verified bundle; check coherence, rehash nothing.

    Direct construction of a LocalReferenceBundle does not prove verification.
    Use the existing resolver to obtain the supplied bundle. This is not the
    future encoder-only resolver: whole-bundle verification may require DiT/VAE.
    """
    if not isinstance(manifest, ModelManifest) or not isinstance(bundle, LocalReferenceBundle):
        raise EncoderRequestError("a manifest and LocalReferenceBundle are required")
    repo, revision = validate_local_reference_layout(manifest)
    if (bundle.profile != LOCAL_REFERENCE_PROFILE or (bundle.repo, bundle.revision) != (repo, revision)
            or bundle.fingerprints != reference_bundle_fingerprints(manifest)):
        raise EncoderRequestError("bundle identity/fingerprints disagree with the manifest")
    expected_assets = {role: tuple(bundle.root / a.identity for a in manifest.artifacts if a.role == role)
                       for role in LOCAL_REFERENCE_LAYOUT}
    expected_dirs = {role: bundle.root / directory for role, directory in ROLE_DIRECTORIES.items()}
    if dict(bundle.assets) != expected_assets or dict(bundle.component_dirs) != expected_dirs:
        raise EncoderRequestError("bundle paths/roles disagree with the manifest")
    return project_encoder_request(config, manifest, encoder_dtype=encoder_dtype)


def validate_worker_request_identity(request: EncoderRequest, manifest: ModelManifest) -> None:
    """Bind a production request to current worker source/packages/manifest.

    Call in the executing worker, not in a different parent environment; this
    checks identity only. It neither verifies asset bytes nor loads a model.
    """
    validated = validate_production_request(request)
    expected = _project_request(manifest, prompt=request.prompt, negative=request.negative_prompt,
                                dtype=request.dtype, cfg_active=validated.cfg_active)
    if request.key_document() != expected.key_document():
        raise EncoderRequestError("production request disagrees with worker/manifest identity")
