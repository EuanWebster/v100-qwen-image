"""Local reference-bundle contract (M2.2a) — stdlib-only, no ML imports.

This module **projects** a local file set into the shape a future adapter's
``load(config, assets)`` expects, in two honestly separated APIs:
:func:`resolve_local_bundle` (the *trusted factory*: runs the trusted
resolver itself — full byte re-verification, expensive on the real 33 GB
set) and :func:`project_resolution_report` (the lower-level projection,
which **trusts its caller's** :class:`~qwen_engine.manifest.ResolutionReport`
and rehashes nothing). It is deliberately NOT a second downloader, NOT a
fake successful load, and — in the metadata-only projection path — NOT a
second verification pass. What happens in the projection, all of it before
any ML import:

1. The supplied schema-v1 :class:`~qwen_engine.manifest.ModelManifest` must
   match the :data:`LOCAL_REFERENCE_LAYOUT` table exactly — the committed
   24 relative filenames, their role mapping — and carry exactly the
   profile's pinned identity :data:`FULL_REFERENCE_REPO` @
   :data:`FULL_REFERENCE_REVISION`. Internal consistency is NOT enough: any
   other repo or revision (whole-set or mixed) is a different model identity
   and requires a NEW reviewed profile, never a relaxed check here. The
   layout is an explicit immutable table so it stays reviewable; per-file
   SHA/byte sizes are *not* hardcoded here (the manifest owns those claims;
   offline tests fabricate tiny contents with the same layout).
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
   The encoder digest is only the **asset component** of a future encoder
   worker cache key — the production worker key must additionally carry the
   worker protocol/implementation revision and the exact
   diffusers/transformers pins (the worker milestone owns that; see
   :class:`ArtifactSetFingerprints`).
5. A future ``from_pretrained(component_dir)`` may open files the manifest
   never listed. :func:`preflight_closed_tree` (also a method on the bundle)
   enumerates the *whole* tree under an explicit root and refuses any entry
   outside the 24-file allowlist and its required directories — extra
   config/weight/code files, leftover ``*.partial`` fetch debris and
   symlinks included — reading no contents. It is opt-in for the future
   probe and must not be run on a root with a fetch in progress.

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
    resolve_manifest,
)

#: Name of the one profile this module knows. Future variants (GGUF ladder,
#: LoRA, single-file) are *new profiles with their own tables*, never guessed
#: fallbacks inside this one.
LOCAL_REFERENCE_PROFILE = "full-reference"

#: The EXACT upstream identity this reviewed profile is pinned to. The named
#: profile is not "whatever internally consistent repo/revision the manifest
#: happens to declare" — it is *this* repo at *this* revision, so the pinned
#: interface assumptions behind the profile hold. A manifest at any other
#: repo or revision — however internally consistent — is a different model
#: identity and requires a NEW reviewed profile (new name, new table, new
#: review), never a relaxed check here.
FULL_REFERENCE_REPO = "Qwen/Qwen-Image-2.1"
FULL_REFERENCE_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"

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


class BundleTreeError(LocalBundleError):
    """The *closed tree* under an explicit model root deviates from the
    profile: an extra or unexpected entry (file, symlink or directory —
    including leftover ``*.partial`` fetch debris and extra config/weight/
    code files), an allowlisted path occupied by a symlink or the wrong
    entry kind, or a missing allowlisted entry. Raised by
    :func:`preflight_closed_tree` before any ML import; contents are never
    read or re-hashed."""


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
    elif repos[0] != FULL_REFERENCE_REPO:
        errors.append(
            f"the {LOCAL_REFERENCE_PROFILE} profile is pinned to repo "
            f"{FULL_REFERENCE_REPO!r}; manifest says {repos[0]!r} — a "
            "different repo is a different model identity and needs a new "
            "reviewed profile, not this one")
    if len(revisions) != 1:
        errors.append(f"the bundle pins ONE revision; manifest mixes: "
                      + ", ".join(revisions))
    elif revisions[0] != FULL_REFERENCE_REVISION:
        errors.append(
            f"the {LOCAL_REFERENCE_PROFILE} profile is pinned to revision "
            f"{FULL_REFERENCE_REVISION!r}; manifest says {revisions[0]!r} — "
            "a new model revision requires a new reviewed profile (the "
            "interface assumptions behind this profile name were reviewed "
            "against the pinned commit only)")
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

    A GENERIC identity function: it pins nothing — it happily digests a
    different repo or revision so digest drift stays demonstrable. Enforcing
    THIS profile's exact pinned identity is
    :func:`validate_local_reference_layout`'s job, not this function's.

    Independent of input ordering and of manifest ordering — re-listing the
    same 24 files yields the same digest; changing *any* repo, revision,
    filename, sha256 or byte size changes it. Rejects non-
    :class:`~qwen_engine.manifest.ModelArtifact` entries and duplicate
    filename identities with a typed :class:`BundleLayoutError` (a silently
    deduplicated digest would look plausible and be wrong). Identity
    metadata only: it is not evidence a model loaded, ran, or produced
    anything good.
    """
    items = list(artifacts)
    if not items:
        raise BundleLayoutError("artifact-set fingerprint needs at least "
                                "one artifact")
    bad = [type(a).__name__ for a in items if not isinstance(a, ModelArtifact)]
    if bad:
        raise BundleLayoutError(
            "artifact-set fingerprint needs ModelArtifact entries; got "
            + ", ".join(bad))
    seen_ids: set[str] = set()
    dupes: list[str] = []
    for a in items:
        if a.identity in seen_ids and a.identity not in dupes:
            dupes.append(a.identity)
        seen_ids.add(a.identity)
    if dupes:
        raise BundleLayoutError(
            "artifact-set fingerprint needs distinct filenames; duplicate "
            "identities would silently collapse in the sorted digest rows: "
            + ", ".join(sorted(dupes)))
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
    shapes and freezes whatever it is given (scalar invariants — profile,
    root/root_resolved consistency, exact pinned repo/revision, 64-hex
    digests — are checked defensively; that is data hygiene, not
    verification). Use :func:`resolve_local_bundle` for the trusted path
    and :func:`project_resolution_report` when a fresh resolver report is
    already in hand.
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

    def preflight_closed_tree(self) -> tuple[str, ...]:
        """Closed-tree check of this bundle's published ``root`` before a
        future ``from_pretrained`` probe may open it: every entry under the
        root must be part of the profile tree (see
        :func:`preflight_closed_tree`). Contents are not read. Never run
        this on a root whose fetch is still in progress — an in-flight
        ``*.partial`` is refused here by design."""
        return preflight_closed_tree(self.root)

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
# Closed-tree preflight (review item 3): a future from_pretrained(component_dir)
# may read files BEYOND the manifest — enumerate the whole tree first
# ---------------------------------------------------------------------------


def _closed_tree_entries() -> dict[str, str]:
    """relative path -> ``"file"`` or ``"dir"``, from the layout table: the
    24 allowlisted files plus the required directory tree leading to them."""
    entries: dict[str, str] = {}
    for filenames in LOCAL_REFERENCE_LAYOUT.values():
        for filename in filenames:
            entries[filename] = "file"
            parent = os.path.dirname(filename)
            while parent:
                entries.setdefault(parent, "dir")
                parent = os.path.dirname(parent)
    return entries


def preflight_closed_tree(root: str | Path) -> tuple[str, ...]:
    """Enumerate EVERY entry under the explicit model ``root`` and refuse
    anything that is not part of the closed profile tree (the 24 allowlisted
    files, the required component directories, and nothing else).

    A future ``from_pretrained(component_dir)`` may open files the manifest
    never listed (leftover ``*.partial`` fetch debris, extra
    config/weight/code files, shadowing ``*.py``); this stdlib preflight is
    what a future local probe calls BEFORE any ML import so the "we load
    exactly the pinned 24 files" claim stays true of the *directory*, not
    just of the manifest. Contents are never read or re-hashed — entries
    are classified by their directory-entry kind only (no symlink
    following: symlinks are refused even at allowlisted paths, deliberately
    *stricter* than the resolver's inside-root symlink tolerance, because a
    link redirects whatever the future loader opens).

    Do **not** run this on a root with an in-progress fetch: a legitimate
    in-flight ``*.partial`` is refused here by design — the preflight is for
    the *settled* tree. Performs no network access, no mkdir, no writes;
    refusals are typed :class:`BundleTreeError` before any ML import.
    Returns the sorted tuple of all checked relative paths.
    """
    root_path = Path(root)
    root_resolved, root_fail = _resolve_or_fail(root_path)
    if root_resolved is None:
        raise BundleTreeError(
            f"model root cannot be resolved: {root_path} ({root_fail})")
    if not root_resolved.is_dir():
        raise BundleTreeError(
            f"model root is not a directory: {root_resolved}")
    expected = _closed_tree_entries()
    seen: set[str] = set()
    stack: list[Path] = [root_resolved]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError as exc:
            raise BundleTreeError(
                f"cannot enumerate directory "
                f"{os.path.relpath(directory, root_resolved) or '.'} under "
                f"the model root: errno {exc.errno} "
                f"({exc.strerror or type(exc).__name__})") from exc
        for entry in entries:
            rel = os.path.relpath(entry.path, root_resolved)
            rel = rel.replace(os.sep, "/")
            is_link = entry.is_symlink()
            is_dir = entry.is_dir(follow_symlinks=False)
            kind = expected.get(rel)
            if kind is None:
                what = ("a symlink (even one pointing inside the root — "
                        "closed-tree policy refuses links outright)"
                        if is_link else "a directory" if is_dir
                        else "a regular file"
                        if entry.is_file(follow_symlinks=False)
                        else "another entry type (special file)")
                raise BundleTreeError(
                    f"{rel}: found {what}, but it is NOT part of the closed "
                    f"{LOCAL_REFERENCE_PROFILE} tree (the {LAYOUT_FILE_COUNT}"
                    f"-file allowlist and its required directories only) — "
                    "a future from_pretrained() over this root could open "
                    "it; remove the stray entry (fetch debris like "
                    "'*.partial' counts) before any ML import")
            if is_link:
                raise BundleTreeError(
                    f"{rel}: an allowlisted path is a symlink; the closed "
                    "tree requires the real entry, not a pointer elsewhere "
                    "(stricter than the resolver by design)")
            if kind == "dir":
                if not is_dir:
                    raise BundleTreeError(
                        f"{rel}: the required component directory is a "
                        f"regular file or special entry, not a directory")
                stack.append(Path(entry.path))
            else:
                if is_dir or not entry.is_file(follow_symlinks=False):
                    raise BundleTreeError(
                        f"{rel}: an allowlisted file path is a "
                        f"{'directory' if is_dir else 'non-regular entry'}, "
                        "not a regular file")
            seen.add(rel)
    missing = sorted(set(expected) - seen)
    if missing:
        raise BundleTreeError(
            "missing from the closed tree under "
            f"{root_resolved}: " + ", ".join(missing))
    return tuple(sorted(expected))


# ---------------------------------------------------------------------------
# The bundle APIs: trusted factory (resolve_manifest itself) + trusted-caller
# projection (documented as trusting its report)
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


def resolve_local_bundle(manifest: ModelManifest,
                         models_dir: str | Path) -> LocalReferenceBundle:
    """**The trusted factory**: run the trusted resolver over the real bytes
    under ``models_dir``, then project the resulting report.

    This is the API to call when no prior resolver run is in hand. Because
    :func:`~qwen_engine.manifest.resolve_manifest` re-reads and re-hashes
    every artifact, this call is deliberately **not** metadata-only: for the
    real 33 GB reference set its cost is a full byte re-verification of that
    set (the same cost as any honest resolve). Callers that already hold a
    fresh resolver report use :func:`project_resolution_report` (which
    *trusts that report*) and avoid the rehash.

    The manifest layout (including the profile's exact pinned repo/revision)
    is validated BEFORE the resolver runs, so an off-profile manifest never
    triggers a 33 GB hash. Refusals are typed and happen before any ML
    import; no network access, no mkdir, no writes.

    Honest limits: re-hashing happens *now*, at this call — it does not
    protect bytes changed afterwards, and this does not close a hostile
    concurrent-filesystem TOCTOU. Actual component load must reverify
    immediately before the library reopens the files.
    """
    validate_local_reference_layout(manifest)
    report = resolve_manifest(manifest, models_dir)
    # project_resolution_report re-checks ok/coherence (the resolver's own
    # negative verdict must never project, whoever built the report).
    return project_resolution_report(manifest, report, models_dir)


def project_resolution_report(manifest: ModelManifest,
                              report: ResolutionReport,
                              models_dir: str | Path) -> LocalReferenceBundle:
    """Project a resolver-shaped report into a :class:`LocalReferenceBundle`.

    **This lower-level API TRUSTS ITS CALLER.** A public
    :class:`~qwen_engine.manifest.ResolutionReport` is a plain dataclass: a
    shape-consistent ``ok=True`` report can be hand-forged, and nothing here
    — positional/field equality with the manifest, coherence checks on
    ``errors``/per-status ``error``/``resolved``, and the cheap regular-file/
    inside-root probes — is cryptographic proof of the file bytes. This
    projection is honestly *not* authenticated evidence; the verification
    lives in :func:`~qwen_engine.manifest.resolve_manifest` (run over real
    bytes — see :func:`resolve_local_bundle`), and a caller using this
    function is asserting "I ran the trusted resolver and this is its
    report". It does not close a hostile concurrent-filesystem TOCTOU:
    actual component load must reverify immediately before the library
    reopens the files.

    What is enforced here: ``manifest`` must satisfy
    :data:`LOCAL_REFERENCE_LAYOUT` and the profile's exact pinned
    repo/revision; ``report`` must be the complete, ``ok``, *internally
    coherent* success claim for that manifest — statuses align positionally,
    every status artifact equals the manifest's, every state is ``present``
    with ``error is None`` and path exactly ``root/filename`` (the caller's
    lexical root), ``report.errors`` is empty, and ``report.resolved``
    equals the role-grouped status paths (keys and tuples, manifest order).
    The bundle **publishes absolute paths under the canonical resolved
    root**, so a future worker started from a different working directory
    can load them even when ``models_dir`` was relative. All refusals are
    typed errors raised before any ML import. No network, no mkdir, no
    writes, no content hashing.
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
