"""Local reference-bundle contract (M2.2a, part 1) — stdlib-only, no ML imports.

This module **projects** an already resolver-verified local file set into the
shape a future adapter's ``load(config, assets)`` expects. It is deliberately
NOT a second downloader, NOT a second verifier pass, and NOT a fake successful
load: :func:`resolve_manifest` (``manifest.py``) has already hashed and sized
every byte; re-opening or re-hashing 33 GB merely to project paths would be
worse than useless. What happens here, all of it before any ML import:

1. The supplied schema-v1 :class:`~qwen_engine.manifest.ModelManifest` must
   match the :data:`LOCAL_REFERENCE_LAYOUT` table exactly — the committed
   24 relative filenames, their role mapping, and one single pinned
   repo/revision for the whole set. The layout is an explicit immutable table
   so it stays reviewable; per-file SHA/byte sizes are *not* hardcoded here
   (the manifest owns those claims; offline tests fabricate tiny contents
   with the same layout).
2. The supplied :class:`~qwen_engine.manifest.ResolutionReport` must be a
   *coherent* success claim: complete, ``ok``, with an empty ``errors``
   tuple, every status ``present`` with ``error is None``, and matching the
   manifest **positionally and field-for-field**; ``report.resolved`` must
   additionally equal the role-grouped status paths (keys and tuples, in
   manifest order). A reordered, truncated, padded, field-forged or
   internally inconsistent report is refused with a typed error. These are
   *shape* checks on the success claim, not byte re-verification: a caller
   who hand-forges both the manifest and the report can still produce a
   bundle — only running the trusted resolver
   (:func:`~qwen_engine.manifest.resolve_manifest`) over the real bytes is
   verification. This projection is not cryptographic re-verification.
3. Every projected path gets a *cheap* structural probe only — regular file,
   resolves inside the explicit model root, expected relative filename — so
   missing files, escaped symlinks, and file-vs-directory swaps cannot leak
   into an adapter. No content hashing, no size comparison (the resolver has
   already enforced those against the manifest).
4. Canonical **artifact-set fingerprints** (version-tagged SHA256 over the
   sorted ``repo/revision/filename/sha256/bytes`` identities) provide a
   whole-bundle digest and an encoder+processor digest for future cache keys.
   A digest is *identity metadata*: it changes exactly when one of those
   identities changes, and proves nothing about tensor load or quality.

Paths come only from the explicit ``models_dir`` root plus manifest filenames
— never from a repo id, never from an ambient Hugging Face cache. Everything
the bundle *publishes* (root, ``assets``, ``component_dirs``) is an absolute
path under the canonical resolved root, so a future worker started from a
different working directory can still load it even when the caller passed a
relative ``models_dir``; the report is still matched against the caller's
lexical ``root / filename``. This profile is the full-precision reference
set only: no GGUF, no LoRA, no single-file checkpoint. Future variants are
separate profiles, not fallbacks tried here.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat as stat_module
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from qwen_engine.manifest import (
    STATE_PRESENT,
    ModelArtifact,
    ModelManifest,
    ManifestError,
    ResolutionReport,
)

#: Name of the one profile this module knows. Future variants (GGUF ladder,
#: LoRA, single-file) are *new profiles with their own tables*, never guessed
#: fallbacks inside this one.
LOCAL_REFERENCE_PROFILE = "full-reference"

#: The committed full-reference layout: role -> relative filenames, in
#: deterministic (sorted-filename) order. 24 files, 6 roles. Identity is the
#: filename; the table pins the role mapping and the directory structure.
LOCAL_REFERENCE_LAYOUT: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "pipeline": (
        "model_index.json",
    ),
    "processor": (
        "processor/added_tokens.json",
        "processor/chat_template.jinja",
        "processor/merges.txt",
        "processor/preprocessor_config.json",
        "processor/special_tokens_map.json",
        "processor/tokenizer.json",
        "processor/tokenizer_config.json",
        "processor/video_preprocessor_config.json",
        "processor/vocab.json",
    ),
    "scheduler": (
        "scheduler/scheduler_config.json",
    ),
    "text_encoder": (
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/model-00001-of-00004.safetensors",
        "text_encoder/model-00002-of-00004.safetensors",
        "text_encoder/model-00003-of-00004.safetensors",
        "text_encoder/model-00004-of-00004.safetensors",
        "text_encoder/model.safetensors.index.json",
    ),
    "dit": (
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    ),
    "vae": (
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ),
})

#: Component directories per role, relative to the model root. The ``dit``
#: role's files live under ``transformer/`` (upstream layout); the pipeline
#: index sits at the root, so the pipeline *component dir* is the root itself
#: — the target of a future ``from_pretrained(<dir>, local_files_only=True)``.
ROLE_DIRECTORIES: Mapping[str, str] = MappingProxyType({
    "pipeline": ".",
    "processor": "processor",
    "scheduler": "scheduler",
    "text_encoder": "text_encoder",
    "dit": "transformer",
    "vae": "vae",
})

#: Roles whose combined identities key the encoder cache (the Qwen3-VL text
#: encoder and its processor/tokenizer/chat-template side): a change to any
#: encoder shard, its index/config, the tokenizer or the chat template must
#: invalidate future encoder cache entries.
ENCODER_ROLES: tuple[str, ...] = ("processor", "text_encoder")

#: Version tag folded into every artifact-set fingerprint digest.
ARTIFACT_SET_FINGERPRINT_VERSION = "qwen-image.artifact-set.v1"

#: Total file count pinned by the layout (reviewability guard).
LAYOUT_FILE_COUNT = sum(len(v) for v in LOCAL_REFERENCE_LAYOUT.values())


# ---------------------------------------------------------------------------
# Typed errors (all under ManifestError -> CLI exit 4, the asset channel)
# ---------------------------------------------------------------------------


class LocalBundleError(ManifestError):
    """Base for local-reference-bundle projection failures (exit 4)."""


class BundleLayoutError(LocalBundleError):
    """The manifest does not match the full-reference layout table exactly
    (missing/extra/unexpected filename, wrong role/file mapping, or a set
    without one single pinned repo/revision)."""


class BundleReportError(LocalBundleError):
    """The ResolutionReport is incomplete, not ``ok``, internally
    inconsistent (nonempty ``errors`` despite success, a present status
    carrying ``error``, a ``resolved`` mapping that does not equal the
    role-grouped status paths in manifest order), or does not match the
    supplied manifest positionally and field-for-field (missing, reordered,
    extra, or forged status entries; paths that are not ``root/filename``)."""


class BundlePathError(LocalBundleError):
    """A projected path failed the cheap structural probe: missing, escaped
    the model root (symlink), or is not a regular file (directory/FIFO swap).
    Never re-hashes content — the resolver owns byte-level verification."""


# ---------------------------------------------------------------------------
# Layout validation (manifest vs the immutable table — pure data, no disk)
# ---------------------------------------------------------------------------


def _layout_expectations() -> dict[str, str]:
    """filename -> expected role, from the layout table."""
    expect: dict[str, str] = {}
    for role, filenames in LOCAL_REFERENCE_LAYOUT.items():
        for filename in filenames:
            expect[filename] = role
    return expect


def validate_local_reference_layout(manifest: ModelManifest) -> tuple[str, str]:
    """Check the manifest is exactly the full-reference set and return the
    single ``(repo, revision)`` pinned for the whole set.

    Raises :class:`BundleLayoutError` listing *every* deviation: filename-set
    differences (missing / extra), per-file role mismatches, and repo or
    revision fragmentation. Pure data — touches no disk, imports no ML.
    """
    errors: list[str] = []
    expected = _layout_expectations()
    seen: dict[str, str] = {}
    for art in manifest.artifacts:
        ident = art.identity
        if ident in seen:
            continue  # ModelManifest already forbids duplicates; belt and braces
        seen[ident] = art.role
    missing = sorted(set(expected) - set(seen))
    extra = sorted(set(seen) - set(expected))
    wrong_role = sorted((f, seen[f], expected[f]) for f in seen
                        if f in expected and seen[f] != expected[f])
    if missing:
        errors.append("missing from manifest: " + ", ".join(missing))
    if extra:
        errors.append("unexpected files for the full-reference profile: "
                      + ", ".join(extra))
    for filename, got, want in wrong_role:
        errors.append(f"role mismatch for {filename}: manifest says "
                      f"{got!r}, layout requires {want!r}")
    repos = sorted({a.repo for a in manifest.artifacts})
    revisions = sorted({a.revision for a in manifest.artifacts})
    if len(repos) != 1:
        errors.append(f"the bundle pins ONE repo; manifest mixes: "
                      + ", ".join(repos))
    if len(revisions) != 1:
        errors.append(f"the bundle pins ONE revision; manifest mixes: "
                      + ", ".join(revisions))
    if errors:
        raise BundleLayoutError(
            f"{LOCAL_REFERENCE_PROFILE} layout violation "
            f"({LAYOUT_FILE_COUNT} files expected across "
            f"{len(LOCAL_REFERENCE_LAYOUT)} roles): " + "; ".join(errors))
    return repos[0], revisions[0]


# ---------------------------------------------------------------------------
# Fingerprints (canonical identity digest; version-tagged, order-free)
# ---------------------------------------------------------------------------


def _canonical_digest(rows: list[list[Any]], repo: str, revision: str) -> str:
    payload = {
        "version": ARTIFACT_SET_FINGERPRINT_VERSION,
        "repo": repo,
        "revision": revision,
        "artifacts": rows,
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def artifact_set_fingerprint(artifacts: Iterable[ModelArtifact]) -> str:
    """SHA256 over the sorted ``[filename, sha256, bytes]`` identity rows of
    *artifacts*, plus the version tag and the single repo/revision they share.

    Independent of input ordering and of manifest ordering — re-listing the
    same 24 files yields the same digest; changing *any* repo, revision,
    filename, sha256 or byte size changes it. Identity metadata only: it is
    not evidence a model loaded, ran, or produced anything good.
    """
    items = list(artifacts)
    if not items:
        raise BundleLayoutError("artifact-set fingerprint needs at least "
                                "one artifact")
    repos = sorted({a.repo for a in items})
    revisions = sorted({a.revision for a in items})
    if len(repos) != 1 or len(revisions) != 1:
        raise BundleLayoutError(
            "artifact-set fingerprint needs one repo and one revision; got "
            f"repos={repos} revisions={revisions}")
    rows = sorted([a.identity, a.sha256, a.bytes] for a in items)
    return _canonical_digest(rows, repos[0], revisions[0])


@dataclass(frozen=True)
class ArtifactSetFingerprints:
    """The two digests the cache design needs. ``model_sha256``-style naming
    would lie: neither digest is *one shard* — each covers a whole set."""

    bundle: str  #: all roles — any file's identity change flips it
    encoder: str  #: ENCODER_ROLES only — invalidates encoder cache entries

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": ARTIFACT_SET_FINGERPRINT_VERSION,
            "bundle_artifact_set_sha256": self.bundle,
            "encoder_artifact_set_sha256": self.encoder,
        }


def reference_bundle_fingerprints(manifest: ModelManifest) -> ArtifactSetFingerprints:
    """Whole-bundle and encoder+processor fingerprints for a manifest that
    satisfies the layout table (validated here first, so a mislabelled role
    can never move a file between the two digests). Disk-free: pure manifest
    identities — usable for cache-key planning with nothing downloaded."""
    repo, revision = validate_local_reference_layout(manifest)
    del repo, revision  # validated; artifact_set_fingerprint re-derives them
    encoder_rows = [a for a in manifest.artifacts if a.role in ENCODER_ROLES]
    return ArtifactSetFingerprints(
        bundle=artifact_set_fingerprint(manifest.artifacts),
        encoder=artifact_set_fingerprint(encoder_rows),
    )


# ---------------------------------------------------------------------------
# The projected bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalReferenceBundle:
    """Resolver-verified local files of the full-reference set, projected for
    ``GenerationAdapter.load(config, assets)`` and for future
    ``local_files_only=True`` component probes.

    Every mapping and path tuple is snapshotted into an immutable copy in
    :meth:`__post_init__` (defensive freeze — mutable dicts or lists handed
    to the constructor cannot mutate the bundle afterwards), and
    :meth:`to_assets` hands out a defensive plain-dict copy for callers that
    want one. Published paths are absolute (``root / filename`` for the
    layout's filenames, rooted at the canonical resolved root) — never repo
    ids, never an ambient HF cache.

    **Direct construction performs no resolver verification.** It only
    shapes and freezes whatever it is given; use
    :func:`local_bundle_from_resolution` for the checked projection.
    """

    profile: str
    #: Canonical *absolute* model root all published paths sit under.
    root: Path
    #: Canonical resolved form of the caller's root (equals ``root``).
    root_resolved: Path
    repo: str
    revision: str
    #: role -> verified absolute paths in manifest order (adapter
    #: ``assets`` shape); immutable whatever was passed in.
    assets: Mapping[str, tuple[Path, ...]]
    #: role -> explicit absolute component directory (pipeline's is the
    #: root itself); immutable whatever was passed in.
    component_dirs: Mapping[str, Path]
    fingerprints: ArtifactSetFingerprints

    def __post_init__(self) -> None:
        # Defensive snapshot: copy the mappings, tuple-ise and Path-ise the
        # values, and re-freeze — a caller holding the original dicts/lists
        # can no longer mutate the "immutable" bundle afterwards. This does
        # NOT verify anything against disk or a resolver report.
        object.__setattr__(self, "assets", MappingProxyType({
            role: tuple(Path(p) for p in paths)
            for role, paths in self.assets.items()}))
        object.__setattr__(self, "component_dirs", MappingProxyType({
            role: Path(directory)
            for role, directory in self.component_dirs.items()}))

    def to_assets(self) -> dict[str, tuple[Path, ...]]:
        """Defensive copy for ``adapter.load(config, adapter_assets)``."""
        return {role: tuple(paths) for role, paths in self.assets.items()}

    @property
    def pipeline_dir(self) -> Path:
        """Target for a future ``from_pretrained(<dir>, local_files_only=True)``."""
        return self.root

    def to_dict(self) -> dict[str, Any]:
        """Deterministic, timestamp-free description for status/logging.
        Metadata about identities — no claim that anything was loaded."""
        return {
            "profile": self.profile,
            "root": str(self.root),
            "repo": self.repo,
            "revision": self.revision,
            "roles": {role: [str(p) for p in paths]
                      for role, paths in self.assets.items()},
            "component_dirs": {role: str(d)
                               for role, d in self.component_dirs.items()},
            "fingerprints": self.fingerprints.to_dict(),
        }


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _resolve_or_fail(path: Path) -> tuple[Path | None, str | None]:
    """Same defensive resolve contract as ``manifest._resolve_or_fail``."""
    try:
        return path.resolve(), None
    except (OSError, RuntimeError, ValueError) as exc:
        detail = (f"{type(exc).__name__}: {exc}"
                  + (f" [{exc.strerror}]" if isinstance(exc, OSError)
                     and exc.strerror else ""))
        return None, detail


def _probe_regular_file_under_root(expected: Path, role: str,
                                  root_resolved: Path) -> None:
    """Cheap structural probe of one projected path — stat + containment only.
    NO content hashing and NO size check: the resolver already did those (and
    offline fixtures legitimately have tiny contents)."""
    resolved, fail = _resolve_or_fail(expected)
    if resolved is None:
        raise BundlePathError(
            f"{role}: projected path {expected} cannot be resolved: {fail}")
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise BundlePathError(
            f"{role}: projected path {expected} resolves to {resolved}, "
            f"outside the model root {root_resolved} (symlink escape?)")
    try:
        st = resolved.stat()
    except FileNotFoundError:
        raise BundlePathError(
            f"{role}: projected path {expected} does not exist") from None
    except OSError as exc:
        raise BundlePathError(
            f"{role}: projected path {expected} cannot be stat'ed: "
            f"errno {exc.errno} ({exc.strerror or type(exc).__name__})") from exc
    if not stat_module.S_ISREG(st.st_mode):
        kind = ("directory" if stat_module.S_ISDIR(st.st_mode)
                else "FIFO" if stat_module.S_ISFIFO(st.st_mode)
                else f"non-regular file (mode {oct(st.st_mode)})")
        raise BundlePathError(
            f"{role}: projected path {expected} is a {kind}, not a regular "
            "file (unexpected layout)")


def local_bundle_from_resolution(manifest: ModelManifest,
                                 report: ResolutionReport,
                                 models_dir: str | Path) -> LocalReferenceBundle:
    """Project a successful :func:`~qwen_engine.manifest.resolve_manifest`
    report into a :class:`LocalReferenceBundle`.

    ``manifest`` must satisfy :data:`LOCAL_REFERENCE_LAYOUT` (exact filenames,
    role mapping, single repo+revision); ``report`` must be the complete,
    ``ok`` report *for that manifest* — statuses align positionally, every
    field of every status artifact equals the manifest's, every state is
    ``present`` with ``error is None`` and path exactly ``root/filename``
    (the caller's lexical root), ``report.errors`` is empty, and
    ``report.resolved`` equals the role-grouped status paths (keys and
    tuples, manifest order). The bundle **publishes absolute paths under the
    canonical resolved root** so a future worker started from a different
    working directory can load them even when ``models_dir`` was relative.
    Published paths are probed (regular file, under the root) without
    re-reading content. These checks police the *shape* of the success
    claim — they are not cryptographic re-verification: forging bytes still
    requires defeating the trusted resolver itself. All refusal happens with
    typed errors before any ML import. Performs no network access, no mkdir,
    no writes.
    """
    repo, revision = validate_local_reference_layout(manifest)

    root = Path(models_dir)
    root_resolved, root_fail = _resolve_or_fail(root)
    if root_resolved is None:
        raise BundlePathError(
            f"model root cannot be resolved: {root} ({root_fail})")
    if not root_resolved.is_dir():
        raise BundlePathError(f"model root is not a directory: {root_resolved}")

    if not report.ok:
        problems = ", ".join(
            f"{s.artifact.filename}: {s.state}"
            for s in report.statuses if s.state != STATE_PRESENT)
        raise BundleReportError(
            "resolution report is not successful (not every artifact is "
            f"verified present/size/hash-correct): {problems or 'no statuses'}")

    if report.errors:
        raise BundleReportError(
            f"a successful report must carry no errors, got "
            f"{len(report.errors)} (ok=True is contradicted by "
            f"report.errors)")

    statuses = tuple(report.statuses)
    artifacts = tuple(manifest.artifacts)
    if len(statuses) != len(artifacts):
        raise BundleReportError(
            f"report has {len(statuses)} statuses but the manifest has "
            f"{len(artifacts)} artifacts (reports align positionally, one per "
            "manifest artifact, in manifest order)")
    for i, (status, art) in enumerate(zip(statuses, artifacts)):
        if status.artifact != art:
            raise BundleReportError(
                f"status[{i}] does not match manifest artifact "
                f"{art.to_dict()}: got {status.artifact.to_dict()} "
                "(missing, reordered, extra or forged report entries are "
                "refused)")
        if status.state != STATE_PRESENT:
            raise BundleReportError(
                f"status[{i}] for {art.filename} has state {status.state!r}; "
                "only a fully present report projects")
        if status.error is not None:
            raise BundleReportError(
                f"status[{i}] for {art.filename} is marked present but "
                f"carries error {status.error!r}; a successful report must "
                "have no per-artifact errors")
        expected = root / art.identity
        if status.path is None or status.path != expected:
            raise BundleReportError(
                f"status[{i}] path {status.path!r} is not the expected "
                f"{expected} (report does not match this root/manifest)")

    # Role grouping of the statuses in manifest order, compared against the
    # caller's *lexical* root (how resolve_manifest builds report.resolved).
    resolved_grouping: dict[str, tuple[Path, ...]] = {
        role: () for role in LOCAL_REFERENCE_LAYOUT}
    # Published paths, rooted at the canonical ABSOLUTE root so they stay
    # loadable from any future working directory (relative models_dir safe).
    assets: dict[str, tuple[Path, ...]] = {
        role: () for role in LOCAL_REFERENCE_LAYOUT}
    for art in artifacts:
        resolved_grouping[art.role] += (root / art.identity,)
        assets[art.role] += (root_resolved / art.identity,)
    if dict(report.resolved) != resolved_grouping:
        raise BundleReportError(
            f"report.resolved does not match the role-grouped status paths "
            f"in manifest order (keys and tuples) for root {root} (missing, "
            "extra or reordered resolved entries are refused)")

    for role, paths in assets.items():
        for path in paths:
            _probe_regular_file_under_root(path, role, root_resolved)

    # Explicit component directories, all under the canonical ABSOLUTE root:
    # every path here is literally `root/<component dir>` — the only load
    # paths in existence, stable across working-directory changes.
    component_dirs = {
        role: (root_resolved if directory == "."
               else root_resolved.joinpath(*directory.split("/")))
        for role, directory in ROLE_DIRECTORIES.items()
    }
    fingerprints = reference_bundle_fingerprints(manifest)

    return LocalReferenceBundle(
        profile=LOCAL_REFERENCE_PROFILE,
        root=root_resolved,
        root_resolved=root_resolved,
        repo=repo,
        revision=revision,
        assets=assets,
        component_dirs=component_dirs,
        fingerprints=fingerprints,
    )
