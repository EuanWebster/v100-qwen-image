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
   match a reviewed profile table exactly — the committed relative filenames
   and their role mapping — and carry exactly that profile's pinned identity
   (repo @ revision). Two profiles exist (:data:`PROFILES`): the base
   full-reference set (:data:`LOCAL_REFERENCE_LAYOUT`, 24 files,
   :data:`FULL_REFERENCE_REPO` @ :data:`FULL_REFERENCE_REVISION`) and the
   Turbo transformer-only set (:data:`TURBO_LAYOUT`, 6 files,
   :data:`TURBO_REFERENCE_REPO` @ :data:`TURBO_REFERENCE_REVISION`, which
   reuses the base encoder and VAE — hence its honest label
   :data:`TURBO_ASSET_LABEL`). Internal consistency is NOT enough: any other
   repo or revision (whole-set or mixed) is a different model identity and
   requires a NEW reviewed profile, never a relaxed check here. The layouts
   are explicit immutable tables so they stay reviewable; per-file SHA/byte
   sizes are *not* hardcoded here (the manifest owns those claims; offline
   tests fabricate tiny contents with the same layout).
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
import threading
import time
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

#: The base (full-precision reference) profile. Future variants (GGUF ladder,
#: LoRA, single-file) are *new profiles with their own tables*, never guessed
#: fallbacks inside this one.
LOCAL_REFERENCE_PROFILE = "full-reference"

#: The EXACT upstream identity the base profile is pinned to. A named profile
#: is not "whatever internally consistent repo/revision the manifest happens
#: to declare" — it is *this* repo at *this* revision, so the pinned interface
#: assumptions behind the profile hold. A manifest at any other repo or
#: revision — however internally consistent — is a different model identity
#: and requires a NEW reviewed profile (new name, new table, new review),
#: never a relaxed check here.
FULL_REFERENCE_REPO = "Qwen/Qwen-Image-2.1"
FULL_REFERENCE_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"

#: The Turbo profile: **transformer + scheduler only**. Upstream
#: ``Qwen/Qwen-Image-2.1-Turbo`` also republishes its own processor, text
#: encoder and VAE, but this project deliberately fetches only the Turbo
#: transformer/scheduler and reuses the reviewed base text encoder and VAE
#: (the measured-equal ones; see ``docs/turbo-plan.md`` §1.2/§1.3). The
#: profile therefore pins a manifest for the DiT side alone: it has **no**
#: encoder roles, so an encoder cache key must come from the base bundle.
TURBO_REFERENCE_PROFILE = "turbo-transformer"
TURBO_REFERENCE_REPO = "Qwen/Qwen-Image-2.1-Turbo"
TURBO_REFERENCE_REVISION = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"

#: Honest label for a Turbo install: what is Turbo is the transformer, and
#: nothing else. Anything user-facing that names this profile must say so.
TURBO_ASSET_LABEL = "Turbo transformer + base encoder/VAE"

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

#: The committed Turbo layout: role -> relative filenames. SIX files, THREE
#: roles — the DiT side of the Turbo release and nothing else. There is
#: deliberately no ``processor``/``text_encoder``/``vae`` role here: those
#: three components are reused from the base bundle (see
#: :data:`TURBO_REFERENCE_PROFILE`). Note the role name is ``dit`` in both
#: profiles even though the directory is ``transformer/``; the base profile's
#: role vocabulary is reused so adapter code keeps one key.
TURBO_LAYOUT: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "pipeline": (
        "model_index.json",
    ),
    "scheduler": (
        "scheduler/scheduler_config.json",
    ),
    "dit": (
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    ),
})

#: Component directories per role for the Turbo profile — identical directory
#: NAMES to the base profile for the roles it shares, so a Turbo root is
#: ``pipeline`` at the root plus ``scheduler/`` and ``transformer/``.
TURBO_ROLE_DIRECTORIES: Mapping[str, str] = MappingProxyType({
    "pipeline": ".",
    "scheduler": "scheduler",
    "dit": "transformer",
})

#: The Turbo profile owns no encoder files; the base bundle supplies them.
TURBO_ENCODER_ROLES: tuple[str, ...] = ()

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
# The profile registry: one reviewed table per model identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleProfile:
    """One reviewed on-disk asset profile: its name, its pinned upstream
    identity, its exact layout table, and its component directories.

    A profile is a *reviewed literal*, not a runtime parameter: adding a
    variant means adding a table here and re-reviewing it. Nothing in this
    module ever falls back from one profile to another, and an unknown name
    is a typed refusal (:func:`get_profile`), never "try the other one".

    ``encoder_roles`` is empty for a profile whose manifest does not contain
    the text encoder (the Turbo transformer-only profile reuses the base
    bundle's encoder/VAE), so an encoder-only fingerprint is refused there
    rather than silently digesting nothing.
    """

    name: str
    #: Human-facing description for status/log output. It must stay honest
    #: about what the profile actually installs.
    label: str
    repo: str
    revision: str
    layout: Mapping[str, tuple[str, ...]]
    role_directories: Mapping[str, str]
    encoder_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field in ("name", "label", "repo", "revision"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{field}: must be a non-empty string, got "
                              f"{value!r}")
        if not isinstance(self.layout, Mapping) or not self.layout:
            errors.append("layout: must be a non-empty mapping")
        else:
            if set(self.layout) != set(self.role_directories):
                errors.append(
                    "layout/role_directories: the same roles must appear in "
                    f"both, got layout={sorted(self.layout)} "
                    f"directories={sorted(self.role_directories)}")
            for role, filenames in self.layout.items():
                if not isinstance(filenames, tuple) or not filenames:
                    errors.append(f"layout[{role!r}]: must be a non-empty "
                                  f"tuple, got {filenames!r}")
                elif list(filenames) != sorted(filenames):
                    errors.append(f"layout[{role!r}]: filenames must be in "
                                  "deterministic sorted order")
            all_files = [f for fns in self.layout.values() for f in fns]
            if len(set(all_files)) != len(all_files):
                errors.append("layout: one filename may appear in only one "
                              "role")
            unknown = sorted(set(self.encoder_roles) - set(self.layout))
            if unknown:
                errors.append("encoder_roles: roles not in the layout: "
                              + ", ".join(unknown))
            if len(set(self.encoder_roles)) != len(self.encoder_roles):
                errors.append("encoder_roles: duplicate roles")
        if errors:
            raise BundleLayoutError("invalid bundle profile: "
                                    + "; ".join(errors))

    @property
    def file_count(self) -> int:
        """Total files pinned by the layout (reviewability guard)."""
        return sum(len(v) for v in self.layout.values())

    def expectations(self) -> dict[str, str]:
        """filename -> expected role, from this profile's layout table."""
        expect: dict[str, str] = {}
        for role, filenames in self.layout.items():
            for filename in filenames:
                expect[filename] = role
        return expect

    def closed_tree_entries(self) -> dict[str, str]:
        """relative path -> ``"file"`` or ``"dir"``: the allowlisted files
        plus the required directory tree leading to them."""
        entries: dict[str, str] = {}
        for filenames in self.layout.values():
            for filename in filenames:
                entries[filename] = "file"
                parent = os.path.dirname(filename)
                while parent:
                    entries.setdefault(parent, "dir")
                    parent = os.path.dirname(parent)
        return entries


#: The reviewed base profile (unchanged identity: its fingerprint anchors are
#: asserted by two committed tests).
BASE_PROFILE = BundleProfile(
    name=LOCAL_REFERENCE_PROFILE,
    label="base full-precision reference",
    repo=FULL_REFERENCE_REPO,
    revision=FULL_REFERENCE_REVISION,
    layout=LOCAL_REFERENCE_LAYOUT,
    role_directories=ROLE_DIRECTORIES,
    encoder_roles=ENCODER_ROLES,
)

#: The reviewed Turbo profile: Turbo DiT + Turbo scheduler over the base
#: encoder/VAE (honest label: :data:`TURBO_ASSET_LABEL`).
TURBO_PROFILE = BundleProfile(
    name=TURBO_REFERENCE_PROFILE,
    label=TURBO_ASSET_LABEL,
    repo=TURBO_REFERENCE_REPO,
    revision=TURBO_REFERENCE_REVISION,
    layout=TURBO_LAYOUT,
    role_directories=TURBO_ROLE_DIRECTORIES,
    encoder_roles=TURBO_ENCODER_ROLES,
)

#: name -> profile. Every entry is a reviewed table; there is no registry
#: mutation path and no fallback chain.
PROFILES: Mapping[str, BundleProfile] = MappingProxyType({
    BASE_PROFILE.name: BASE_PROFILE,
    TURBO_PROFILE.name: TURBO_PROFILE,
})


def get_profile(profile: str | BundleProfile) -> BundleProfile:
    """Return the reviewed profile named ``profile`` (or the profile itself).

    An unknown name is a typed :class:`BundleLayoutError` listing the known
    profiles — never a fallback: silently loading base assets because a Turbo
    name was misspelt is exactly the failure this registry exists to stop.
    """
    if isinstance(profile, BundleProfile):
        if PROFILES.get(profile.name) is not profile:
            raise BundleLayoutError(
                f"unknown bundle profile {profile.name!r}: only the reviewed "
                f"registry tables are usable (known: "
                f"{', '.join(sorted(PROFILES))}); a hand-built profile table "
                "is new reviewed code, not a runtime argument")
        return profile
    try:
        return PROFILES[profile]
    except (KeyError, TypeError):
        raise BundleLayoutError(
            f"unknown bundle profile {profile!r} (known: "
            f"{', '.join(sorted(PROFILES))})") from None


# ---------------------------------------------------------------------------
# Layout validation (manifest vs the immutable table — pure data, no disk)
# ---------------------------------------------------------------------------


def validate_layout(manifest: ModelManifest,
                    profile: str | BundleProfile = LOCAL_REFERENCE_PROFILE,
                    ) -> tuple[str, str]:
    """Check the manifest is exactly ``profile``'s set and return the single
    ``(repo, revision)`` pinned for the whole set.

    ``profile`` is a reviewed registry name (:data:`PROFILES`), defaulting to
    the base full-reference profile so every existing caller keeps its
    behaviour. Raises :class:`BundleLayoutError` listing *every* deviation:
    filename-set differences (missing / extra), per-file role mismatches, and
    repo or revision fragmentation. Pure data — touches no disk, imports no
    ML.
    """
    spec = get_profile(profile)
    errors: list[str] = []
    expected = spec.expectations()
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
        errors.append(f"unexpected files for the {spec.name} profile: "
                      + ", ".join(extra))
    for filename, got, want in wrong_role:
        errors.append(f"role mismatch for {filename}: manifest says "
                      f"{got!r}, layout requires {want!r}")
    repos = sorted({a.repo for a in manifest.artifacts})
    revisions = sorted({a.revision for a in manifest.artifacts})
    if len(repos) != 1:
        errors.append(f"the bundle pins ONE repo; manifest mixes: "
                      + ", ".join(repos))
    elif repos[0] != spec.repo:
        errors.append(
            f"the {spec.name} profile is pinned to repo "
            f"{spec.repo!r}; manifest says {repos[0]!r} — a "
            "different repo is a different model identity and needs a new "
            "reviewed profile, not this one")
    if len(revisions) != 1:
        errors.append(f"the bundle pins ONE revision; manifest mixes: "
                      + ", ".join(revisions))
    elif revisions[0] != spec.revision:
        errors.append(
            f"the {spec.name} profile is pinned to revision "
            f"{spec.revision!r}; manifest says {revisions[0]!r} — "
            "a new model revision requires a new reviewed profile (the "
            "interface assumptions behind this profile name were reviewed "
            "against the pinned commit only)")
    if errors:
        raise BundleLayoutError(
            f"{spec.name} layout violation "
            f"({spec.file_count} files expected across "
            f"{len(spec.layout)} roles): " + "; ".join(errors))
    return repos[0], revisions[0]


def validate_local_reference_layout(manifest: ModelManifest) -> tuple[str, str]:
    """Base-profile spelling of :func:`validate_layout` (unchanged public
    contract: this name always means the full-reference profile)."""
    return validate_layout(manifest, LOCAL_REFERENCE_PROFILE)


def validate_turbo_layout(manifest: ModelManifest) -> tuple[str, str]:
    """Turbo-profile spelling of :func:`validate_layout`.

    The Turbo manifest carries **only** the DiT side (model index, scheduler
    config, transformer config/shards/index): encoder and VAE files are not
    part of it, by design — they are reused from the base bundle.
    """
    return validate_layout(manifest, TURBO_REFERENCE_PROFILE)


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
    would lie: neither digest is *one shard* — each covers a whole set.

    **Cache-key scope warning (review item 4).** ``encoder`` is only the
    ASSET component of a future encoder-worker cache key. A production
    worker cache key must additionally include the worker protocol /
    implementation revision and the exact diffusers/transformers pins —
    wiring ``EncoderRequest.model_sha256`` (or any worker cache key) to this
    asset digest ALONE is a contract violation: identical weights under a
    different encoder implementation are NOT a cache hit. The full cache-
    key implementation belongs to the worker milestone, not to this module.
    """

    bundle: str  #: all roles — any file's identity change flips it
    encoder: str | None  #: ENCODER_ROLES only — the ASSET component of the
    #: future encoder cache key (never the whole key; see the scope warning
    #: above). ``None`` for a profile that carries no encoder files (the
    #: Turbo transformer-only profile): there is no encoder digest to report,
    #: and inventing one over an empty set would be a lie.

    def __post_init__(self) -> None:
        fields = (("bundle", self.bundle, False),
                  ("encoder", self.encoder, True))
        errors = [f"{name}: must be a 64-char lowercase hex digest, got "
                  f"{value!r}"
                  for name, value, optional in fields
                  if not (optional and value is None)
                  and not (isinstance(value, str) and len(value) == 64
                           and all(c in "0123456789abcdef" for c in value))]
        if errors:
            raise BundleLayoutError(
                "invalid artifact-set fingerprints: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "version": ARTIFACT_SET_FINGERPRINT_VERSION,
            "bundle_artifact_set_sha256": self.bundle,
        }
        if self.encoder is not None:
            data["encoder_artifact_set_sha256"] = self.encoder
        return data


def reference_bundle_fingerprints(
        manifest: ModelManifest,
        profile: str | BundleProfile = LOCAL_REFERENCE_PROFILE,
        ) -> ArtifactSetFingerprints:
    """Whole-bundle and encoder+processor fingerprints for a manifest that
    satisfies ``profile``'s layout table (validated here first, so a
    mislabelled role can never move a file between the two digests).
    Disk-free: pure manifest identities — usable for cache-key planning with
    nothing downloaded.

    For the Turbo profile ``encoder`` is ``None``: that manifest has no
    encoder files at all, and its encoder cache key must be derived from the
    base bundle's encoder fingerprint (see :data:`TURBO_PROFILE`).
    """
    spec = get_profile(profile)
    repo, revision = validate_layout(manifest, spec)
    del repo, revision  # validated; artifact_set_fingerprint re-derives them
    encoder_rows = [a for a in manifest.artifacts
                    if a.role in spec.encoder_roles]
    return ArtifactSetFingerprints(
        bundle=artifact_set_fingerprint(manifest.artifacts),
        encoder=(artifact_set_fingerprint(encoder_rows)
                 if spec.encoder_roles else None),
    )


# ---------------------------------------------------------------------------
# The projected bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalReferenceBundle:
    """Resolver-verified local files of ONE reviewed profile, projected for
    ``GenerationAdapter.load(config, assets)`` and for future
    ``local_files_only=True`` component probes. ``profile`` names the
    registry entry (:data:`PROFILES`): the base full-reference bundle
    (six roles) or the Turbo transformer-only bundle (three roles, over the
    base encoder/VAE).

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
        # Scalar invariants (review item 5): the *data* this dataclass
        # carries about the named profile is policed on EVERY construction —
        # a direct construction cannot smuggle a foreign identity or a
        # non-digest "digest" into a bundle-shaped object. This is data
        # hygiene, NOT resolver verification: nothing here reads disk, a
        # report or a manifest.
        errors: list[str] = []
        if not isinstance(self.profile, str):
            spec = None
            errors.append(f"profile: must be a registry name (str), got "
                          f"{type(self.profile).__name__}")
        else:
            try:
                spec = get_profile(self.profile)
            except BundleLayoutError as exc:
                spec = None
                errors.append(f"profile: {exc}")
        for name in ("root", "root_resolved"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                errors.append(f"{name}: must be a Path, got "
                              f"{type(value).__name__}")
            elif not value.is_absolute():
                errors.append(f"{name}: must be absolute, got {value}")
        if (isinstance(self.root, Path) and isinstance(self.root_resolved, Path)
                and self.root != self.root_resolved):
            errors.append(
                f"root {self.root} != root_resolved {self.root_resolved}: "
                "the bundle publishes the canonical resolved root, and "
                "records the canonical root as its own resolved form")
        if spec is not None:
            if self.repo != spec.repo:
                errors.append(f"repo: profile {spec.name} pins "
                              f"{spec.repo!r}, got {self.repo!r}")
            if self.revision != spec.revision:
                errors.append(f"revision: profile {spec.name} pins "
                              f"{spec.revision!r}, got {self.revision!r}")
        if not isinstance(self.fingerprints, ArtifactSetFingerprints):
            errors.append("fingerprints: must be ArtifactSetFingerprints, "
                          f"got {type(self.fingerprints).__name__}")
        elif spec is not None:
            if spec.encoder_roles and self.fingerprints.encoder is None:
                errors.append(
                    f"fingerprints.encoder: profile {spec.name} carries "
                    "encoder files, so an encoder digest is required")
            if not spec.encoder_roles and self.fingerprints.encoder is not None:
                errors.append(
                    f"fingerprints.encoder: profile {spec.name} carries no "
                    "encoder files, so it has no encoder digest (the base "
                    "bundle's digest keys the encoder cache)")
        if errors:
            raise BundleLayoutError(
                "invalid LocalReferenceBundle (scalar invariants of the "
                "named profile; no disk/resolver verification implied): "
                + "; ".join(errors))
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
        return preflight_closed_tree(self.root, self.profile)

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


def _closed_tree_entries(profile: str | BundleProfile = LOCAL_REFERENCE_PROFILE,
                         ) -> dict[str, str]:
    """relative path -> ``"file"`` or ``"dir"``, from the selected profile's
    layout table: the allowlisted files plus the required directory tree
    leading to them."""
    return get_profile(profile).closed_tree_entries()


def preflight_closed_tree(root: str | Path,
                          profile: str | BundleProfile = LOCAL_REFERENCE_PROFILE,
                          ) -> tuple[str, ...]:
    """Enumerate EVERY entry under the explicit model ``root`` and refuse
    anything that is not part of the closed ``profile`` tree (its allowlisted
    files, the required component directories, and nothing else).

    A future ``from_pretrained(component_dir)`` may open files the manifest
    never listed (leftover ``*.partial`` fetch debris, extra
    config/weight/code files, shadowing ``*.py``); this stdlib preflight is
    what a future local probe calls BEFORE any ML import so the "we load
    exactly the pinned files" claim stays true of the *directory*, not just
    of the manifest. ``profile`` defaults to the base full-reference set, so
    existing callers keep their behaviour; the Turbo asset root is checked
    with ``profile=TURBO_REFERENCE_PROFILE`` and therefore refuses the
    encoder/VAE files that belong to the base root. Contents are never read
    or re-hashed — entries are classified by their directory-entry kind only
    (no symlink following: symlinks are refused even at allowlisted paths,
    deliberately *stricter* than the resolver's inside-root symlink
    tolerance, because a link redirects whatever the future loader opens).

    Do **not** run this on a root with an in-progress fetch: a legitimate
    in-flight ``*.partial`` is refused here by design — the preflight is for
    the *settled* tree. Performs no network access, no mkdir, no writes;
    refusals are typed :class:`BundleTreeError` before any ML import.
    Entries are streamed through no-follow directory FDs; observed directory
    or root replacement/mutation refuses. This is a settled-tree snapshot,
    not protection for subsequent loads or hostile simultaneous rewrites.
    Returns the sorted tuple of all checked relative paths.
    """
    spec = get_profile(profile)
    root_path = Path(root)
    root_resolved, root_fail = _resolve_or_fail(root_path)
    if root_resolved is None:
        raise BundleTreeError(
            f"model root cannot be resolved: {root_path} ({root_fail})")
    if not root_resolved.is_dir():
        raise BundleTreeError(
            f"model root is not a directory: {root_resolved}")
    expected = _closed_tree_entries(spec)
    seen: set[str] = set()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK

    def signature(info: os.stat_result) -> tuple[int, ...]:
        return (info.st_dev, info.st_ino, info.st_mode,
                info.st_mtime_ns, info.st_ctime_ns)

    def visit(directory_fd: int, prefix: str) -> None:
        before = os.fstat(directory_fd)
        # Stream the directory: the first unexpected entry refuses immediately,
        # instead of allocating/sorting an unbounded list of unrelated files.
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                rel = f"{prefix}/{entry.name}" if prefix else entry.name
                info = os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
                is_link = stat_module.S_ISLNK(info.st_mode)
                is_dir = stat_module.S_ISDIR(info.st_mode)
                kind = expected.get(rel)
                if kind is None:
                    what = ("a symlink (even one pointing inside the root — "
                            "closed-tree policy refuses links outright)"
                            if is_link else "a directory" if is_dir
                            else "a regular file"
                            if stat_module.S_ISREG(info.st_mode)
                            else "another entry type (special file)")
                    raise BundleTreeError(
                        f"{rel}: found {what}, but it is NOT part of the closed "
                        f"{spec.name} tree (the {spec.file_count}"
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
                    # Anchor traversal to the inspected directory, never to a
                    # pathname that could now be a symlink to a different tree.
                    child_fd = os.open(entry.name, flags, dir_fd=directory_fd)
                    try:
                        if signature(os.fstat(child_fd)) != signature(info):
                            raise BundleTreeError(f"{rel}: directory changed before enumeration")
                        visit(child_fd, rel)
                        if signature(os.fstat(child_fd)) != signature(
                                os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)):
                            raise BundleTreeError(f"{rel}: directory path changed during enumeration")
                    finally:
                        os.close(child_fd)
                else:
                    if is_dir or not stat_module.S_ISREG(info.st_mode):
                        raise BundleTreeError(
                            f"{rel}: an allowlisted file path is a "
                            f"{'directory' if is_dir else 'non-regular entry'}, "
                            "not a regular file")
                seen.add(rel)
        if signature(before) != signature(os.fstat(directory_fd)):
            raise BundleTreeError(f"{prefix or '.'}: directory changed during enumeration")

    try:
        root_fd = os.open(root_resolved, flags)
        try:
            initial = os.fstat(root_fd)
            visit(root_fd, "")
            if (signature(initial) != signature(root_resolved.stat(follow_symlinks=False))
                    or root_path.resolve() != root_resolved):
                raise BundleTreeError("model root changed during enumeration")
        finally:
            os.close(root_fd)
    except OSError as exc:
        raise BundleTreeError(
            f"cannot enumerate directory under model root {root_resolved}: "
            f"errno {exc.errno} ({exc.strerror or type(exc).__name__})") from exc
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


#: A successful full-bytes verification by :func:`resolve_local_bundle` is
#: the SAME verification the CLI already ran minutes-to-seconds earlier in
#: the SAME process: :func:`qwen_engine.cli._verify_assets` and the adapter's
#: ``load`` reverify hash the identical bytes with the identical trusted
#: resolver (measured 2026-10-25 on the shipped trees: base 33 GB ≈ 19 s per
#: pass, turbo 14 GB ≈ 9 s per pass, every pass paid twice per run — the
#: Turbo model_initialisation surcharge WAS the second turbo-tree hash).
#: Reusing a verification this recent, keyed on manifest content + profile +
#: the root's filesystem identity, is not a weaker guarantee than re-hashing:
#: a fresh hash never protected against bytes changed afterwards either (see
#: the honest-limits paragraph of :func:`resolve_local_bundle`). Outside the
#: window, or for any different manifest/root/profile identity, and for every
#: REFUSAL, nothing is reused — the trusted resolver runs in full.
RESOLUTION_REUSE_WINDOW_SECONDS = 30.0

#: Clock witness (tests substitute a deterministic one).
_MONOTONIC = time.monotonic
_resolution_memo: dict[tuple[Any, ...], tuple[float, LocalReferenceBundle]] = {}
_resolution_memo_lock = threading.Lock()


def clear_resolution_memo() -> None:
    """Forget every reusable verification (test seam; never needed on a run)."""
    with _resolution_memo_lock:
        _resolution_memo.clear()


def _resolution_key(manifest: ModelManifest, models_dir: str | Path,
                    spec: BundleProfile) -> tuple[Any, ...] | None:
    """Identity under which a full-bytes verification may be reused: profile,
    manifest CONTENT (the artifact-set digest the bundle publishes from the
    very rows the resolver verified), and the root directory's filesystem
    identity (st_dev/st_ino of the fully resolved path — never the lexical
    string alone). ``None`` means uncacheable: the caller must run the
    resolver so its own typed errors surface unchanged."""
    try:
        root = Path(models_dir).resolve()
        info = os.stat(root)
        return (spec.name, artifact_set_fingerprint(manifest.artifacts),
                str(root), info.st_dev, info.st_ino)
    except (OSError, LocalBundleError):
        return None



def prime_resolution_memo(manifest: ModelManifest,
                          models_dir: str | Path,
                          report: ResolutionReport,
                          profile: str | BundleProfile = LOCAL_REFERENCE_PROFILE,
                          ) -> LocalReferenceBundle:
    """Project a FRESH trusted-resolver report and remember the projection as
    this process's verification for :data:`RESOLUTION_REUSE_WINDOW_SECONDS`.

    This is the CLI half of the reuse: :func:`qwen_engine.cli._verify_assets`
    runs the trusted resolver itself, and the adapter's
    :func:`resolve_local_bundle` in ``load`` would otherwise re-hash the
    identical bytes seconds later in the same process (measured: 19 s base +
    9 s turbo, every pass paid twice per run). The projection here runs
    EVERY check :func:`project_resolution_report` runs — ok/coherence/
    positional/file probes — so only a full, honest, successful report can
    ever be remembered; typed refusals propagate and remember nothing. The
    same key/window/expiry rules as :func:`resolve_local_bundle` apply, so
    the adapter hit reuses exactly the bytes the resolver just verified.
    """
    spec = get_profile(profile)
    validate_layout(manifest, spec)
    bundle = project_resolution_report(manifest, report, models_dir, spec)
    key = _resolution_key(manifest, models_dir, spec)
    if key is not None:
        with _resolution_memo_lock:
            _resolution_memo[key] = (_MONOTONIC(), bundle)
    return bundle


def resolve_local_bundle(manifest: ModelManifest,
                         models_dir: str | Path,
                         profile: str | BundleProfile = LOCAL_REFERENCE_PROFILE,
                         ) -> LocalReferenceBundle:
    """**The trusted factory**: run the trusted resolver over the real bytes
    under ``models_dir``, then project the resulting report.

    This is the API to call when no prior resolver run is in hand. Because
    :func:`~qwen_engine.manifest.resolve_manifest` re-reads and re-hashes
    every artifact, this call is deliberately **not** metadata-only: for the
    real 33 GB reference set its cost is a full byte re-verification of that
    set (the same cost as any honest resolve). Callers that already hold a
    fresh resolver report use :func:`project_resolution_report` (which
    *trusts that report*) and avoid the rehash. Within ONE process, an
    identical resolve completed within :data:`RESOLUTION_REUSE_WINDOW_SECONDS`
    is returned from the freshness memo without rehashing (see that constant:
    this is the CLI-verify → adapter-reverify handoff, the identical bytes
    hashed by the identical resolver seconds apart); every other call hashes
    in full, and refusals are never memoized.

    The manifest layout (including the profile's exact pinned repo/revision)
    is validated BEFORE the resolver runs, so an off-profile manifest never
    triggers a 33 GB hash. Refusals are typed and happen before any ML
    import; no network access, no mkdir, no writes.

    ``profile`` selects the reviewed registry table (base by default); the
    Turbo root is resolved with ``profile=TURBO_REFERENCE_PROFILE`` and must
    therefore contain the Turbo DiT tree only.

    Honest limits: re-hashing happens *now*, at this call — it does not
    protect bytes changed afterwards, and this does not close a hostile
    concurrent-filesystem TOCTOU. Actual component load must reverify
    immediately before the library reopens the files.
    """
    spec = get_profile(profile)
    validate_layout(manifest, spec)
    key = _resolution_key(manifest, models_dir, spec)
    if key is not None:
        now = _MONOTONIC()
        with _resolution_memo_lock:
            for stale in [entry_key for entry_key, (when, _) in _resolution_memo.items()
                          if now - when > RESOLUTION_REUSE_WINDOW_SECONDS]:
                del _resolution_memo[stale]
            entry = _resolution_memo.get(key)
            if entry is not None:
                return entry[1]
    report = resolve_manifest(manifest, models_dir)
    # project_resolution_report re-checks ok/coherence (the resolver's own
    # negative verdict must never project, whoever built the report).
    bundle = project_resolution_report(manifest, report, models_dir, spec)
    if key is not None:
        with _resolution_memo_lock:
            _resolution_memo[key] = (_MONOTONIC(), bundle)
    return bundle


def project_resolution_report(manifest: ModelManifest,
                              report: ResolutionReport,
                              models_dir: str | Path,
                              profile: str | BundleProfile = LOCAL_REFERENCE_PROFILE,
                              ) -> LocalReferenceBundle:
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
    the selected ``profile``'s layout table and its exact pinned
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
    spec = get_profile(profile)
    repo, revision = validate_layout(manifest, spec)

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
        role: () for role in spec.layout}
    # Published paths, rooted at the canonical ABSOLUTE root so they stay
    # loadable from any future working directory (relative models_dir safe).
    assets: dict[str, tuple[Path, ...]] = {
        role: () for role in spec.layout}
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
        for role, directory in spec.role_directories.items()
    }
    fingerprints = reference_bundle_fingerprints(manifest, spec)

    return LocalReferenceBundle(
        profile=spec.name,
        root=root_resolved,
        root_resolved=root_resolved,
        repo=repo,
        revision=revision,
        assets=assets,
        component_dirs=component_dirs,
        fingerprints=fingerprints,
    )
