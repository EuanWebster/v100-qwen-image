"""M2.2a offline tests for the local reference-bundle contract
(``qwen_engine.local_bundle``): pure projection of resolver-verified paths.

All fixtures fabricate **tiny fake contents** inside pytest ``tmp_path`` with
the exact full-reference layout (24 files, 6 roles, one repo/revision) and
run them through the **real** ``resolve_manifest`` — no 33 GB, no network, no
heavy imports, no reads or writes of ``models/assets/`` (which may be
concurrently downloading; nothing here touches it). The committed
``models/manifest.json`` is exercised as data only (layout + disk-free
fingerprints).

Review-fix pass (``/tmp/v100-m22a-bundle-review.md``) coverage: relative
roots publish CWD-stable **absolute** paths (including a fresh-child check),
forged reports whose ``errors``/per-status ``error``/``resolved`` mapping
contradict the success claim are refused, and direct construction snapshots
and freezes mutable mappings.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_engine import local_bundle as lb
from qwen_engine.manifest import (
    ArtifactError,
    ArtifactStatus,
    ModelArtifact,
    ModelManifest,
    ResolutionReport,
    STATE_MISSING,
    STATE_PRESENT,
    load_manifest,
    resolve_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "manifest.json"

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"

#: The committed 24 runtime filenames, spelled independently of the module's
#: table so a drift in either direction fails loudly.
EXPECTED_FILES = frozenset({
    "model_index.json",
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
})

EXPECTED_ROLE_COUNTS = {"pipeline": 1, "processor": 9, "scheduler": 1,
                        "text_encoder": 7, "dit": 4, "vae": 2}

#: Anchored identity of the COMMITTED manifest under the bundle fingerprint
#: algorithm (payload per ``artifact_set_fingerprint``: version tag, repo,
#: revision, sorted [filename, sha256, bytes] rows over all 24 files).
#: Second, independent anchor next to test_m2_manifest.IDENTITY_DIGEST.
COMMITTED_BUNDLE_FINGERPRINT = "729fd317ddffd127b68c2acdfd56c1a93612c5c274b9ba61d710b3deeb4eaf73"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _layout_pairs() -> list[tuple[str, str]]:
    return [(role, fn) for role, fns in lb.LOCAL_REFERENCE_LAYOUT.items()
            for fn in fns]


def build_tiny_tree(root: Path, *, repo: str = REPO,
                    revision: str = REVISION) -> ModelManifest:
    """Write one tiny (filename-labelled) file per layout entry and return a
    schema-v1 manifest whose sha256/bytes describe exactly those bytes."""
    artifacts = []
    for role, filename in _layout_pairs():
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"tiny-fake::{filename}".encode("utf-8")
        path.write_bytes(content)
        artifacts.append(ModelArtifact(
            role=role, repo=repo, revision=revision, filename=filename,
            sha256=hashlib.sha256(content).hexdigest(), bytes=len(content)))
    return ModelManifest(schema_version=1, artifacts=tuple(artifacts))


def _manifest_artifacts(manifest: ModelManifest) -> dict[str, ModelArtifact]:
    return {a.identity: a for a in manifest.artifacts}


def _grouped_statuses(statuses: tuple[ArtifactStatus, ...],
                      ) -> dict[str, tuple[Path, ...]]:
    """Role grouping the real resolver would build from these statuses."""
    grouping: dict[str, tuple[Path, ...]] = {}
    for s in statuses:
        grouping[s.artifact.role] = grouping.get(s.artifact.role, ()) + \
            (s.path,)
    return grouping


def _forged_report(manifest: ModelManifest, root: Path,
                   skip: set[str] | None = None,
                   *, errors: tuple = (),
                   resolved: dict[str, tuple[Path, ...]] | None = None,
                   statuses: tuple[ArtifactStatus, ...] | None = None,
                   ) -> ResolutionReport:
    """A hand-built, *internally consistent* 'present' report (NOT from the
    resolver — the bytes are never verified) — used to prove the projection's
    own cheap probes refuse forged success claims. The keyword overrides let
    a test break exactly one coherence dimension (statuses/errors/resolved)."""
    if statuses is None:
        statuses = tuple(
            ArtifactStatus(a, STATE_PRESENT, path=root / a.identity)
            for a in manifest.artifacts if a.identity not in (skip or set()))
    if resolved is None:
        resolved = _grouped_statuses(statuses)
    return ResolutionReport(ok=True, resolved=resolved, statuses=statuses,
                            errors=errors)


def _project(manifest: ModelManifest, root: Path) -> lb.LocalReferenceBundle:
    report = resolve_manifest(manifest, root)
    assert report.ok, [s.to_dict() for s in report.statuses if not s.error is None]
    return lb.project_resolution_report(manifest, report, root)


# ---------------------------------------------------------------------------
# The layout table itself
# ---------------------------------------------------------------------------


def test_layout_table_is_the_committed_full_reference_set():
    assert lb.LAYOUT_FILE_COUNT == 24
    flat = [fn for fns in lb.LOCAL_REFERENCE_LAYOUT.values() for fn in fns]
    assert len(flat) == 24 and set(flat) == EXPECTED_FILES
    counts = {r: len(f) for r, f in lb.LOCAL_REFERENCE_LAYOUT.items()}
    assert counts == EXPECTED_ROLE_COUNTS
    # Deterministic (sorted-filename) order inside every role group.
    for fns in lb.LOCAL_REFERENCE_LAYOUT.values():
        assert list(fns) == sorted(fns)
    # Every non-root file lives under its role's component directory.
    for role, fns in lb.LOCAL_REFERENCE_LAYOUT.items():
        directory = lb.ROLE_DIRECTORIES[role]
        for fn in fns:
            parent = str(Path(fn).parent)
            assert parent in ("" , ".") if directory == "." else parent == directory
    # Upstream layout exception kept visible: role 'dit' lives under 'transformer/'.
    assert lb.ROLE_DIRECTORIES["dit"] == "transformer"


def test_layout_table_is_immutable():
    with pytest.raises(TypeError):
        lb.LOCAL_REFERENCE_LAYOUT["vae"] = ("nope",)  # type: ignore[index]
    with pytest.raises(TypeError):
        lb.ROLE_DIRECTORIES["vae"] = "nope"  # type: ignore[index]


def test_committed_manifest_satisfies_the_layout_disk_free():
    """The committed manifest (read-only) passes layout validation and
    fingerprints WITHOUT touching any model storage."""
    manifest = load_manifest(MANIFEST_PATH)
    repo, revision = lb.validate_local_reference_layout(manifest)
    assert (repo, revision) == (REPO, REVISION)
    fps = lb.reference_bundle_fingerprints(manifest)
    assert fps.bundle != fps.encoder
    for digest in (fps.bundle, fps.encoder):
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_committed_manifest_bundle_fingerprint_anchor():
    manifest = load_manifest(MANIFEST_PATH)
    assert lb.reference_bundle_fingerprints(manifest).bundle == (
        COMMITTED_BUNDLE_FINGERPRINT)


# ---------------------------------------------------------------------------
# Positive projection through the REAL resolver
# ---------------------------------------------------------------------------


def test_positive_projection_of_tiny_bundle(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    bundle = _project(manifest, root)
    abs_root = root.resolve()

    assert bundle.profile == lb.LOCAL_REFERENCE_PROFILE
    assert (bundle.repo, bundle.revision) == (REPO, REVISION)
    # Published root is the canonical ABSOLUTE root (stable across CWDs).
    assert bundle.root == abs_root == bundle.root_resolved
    assert bundle.root.is_absolute()
    assert set(bundle.assets) == set(lb.LOCAL_REFERENCE_LAYOUT)
    for role, fns in lb.LOCAL_REFERENCE_LAYOUT.items():
        paths = bundle.assets[role]
        assert len(paths) == EXPECTED_ROLE_COUNTS[role]
        # Manifest order preserved; paths are literally abs_root/filename.
        assert paths == tuple(abs_root / fn for fn in fns)
        assert all(p.is_absolute() and p.is_relative_to(abs_root)
                   for p in paths)
    # Component dirs are the explicit local targets for future
    # local_files_only=True probes — all under the canonical absolute root.
    assert bundle.pipeline_dir == abs_root
    assert bundle.component_dirs["text_encoder"] == abs_root / "text_encoder"
    assert bundle.component_dirs["dit"] == abs_root / "transformer"
    assert all(d.is_absolute() and d.is_relative_to(abs_root)
               for d in bundle.component_dirs.values())


# ---------------------------------------------------------------------------
# CWD-stable published paths (review fix 1): relative roots must not leak
# relative paths into the bundle
# ---------------------------------------------------------------------------


def test_relative_root_publishes_absolute_paths(tmp_path, monkeypatch):
    project_dir = tmp_path / "proj"
    real_root = project_dir / "models" / "assets"
    manifest = build_tiny_tree(real_root)
    monkeypatch.chdir(project_dir)
    models_dir = "models/assets"  # deliberately RELATIVE, as a CLI would pass
    report = resolve_manifest(manifest, models_dir)
    assert report.ok
    # The resolver's own report stays lexical — the factory still matches
    # status paths against the caller's root/filename spelling...
    assert all(not s.path.is_absolute() for s in report.statuses)
    bundle = lb.project_resolution_report(manifest, report, models_dir)
    # ...but everything the bundle publishes is absolute and canonical.
    assert bundle.root == real_root.resolve() == bundle.root_resolved
    assert bundle.root.is_absolute()
    for role, paths in bundle.assets.items():
        for p in paths:
            assert p.is_absolute(), (role, p)
            assert p.is_file()
            assert p.is_relative_to(bundle.root)
    for role, d in bundle.component_dirs.items():
        assert d.is_absolute() and d.is_dir(), (role, d)
    # A future worker started from a DIFFERENT CWD still loads the same
    # files: every published path resolves from anywhere inside this process.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    for paths in bundle.to_assets().values():
        for p in paths:
            assert p.is_file()
    for d in bundle.component_dirs.values():
        assert d.is_dir()
    assert bundle.pipeline_dir == real_root.resolve()


def test_relative_root_paths_valid_in_fresh_child(tmp_path):
    """A child interpreter launched from ANOTHER working directory can load
    every published path — no ambient-CWD dependency anywhere."""
    project_dir = tmp_path / "proj"
    real_root = project_dir / "models" / "assets"
    manifest = build_tiny_tree(real_root)
    child_cwd = tmp_path / "child-cwd"
    child_cwd.mkdir()
    previous_cwd = Path.cwd()
    try:
        os.chdir(project_dir)
        report = resolve_manifest(manifest, "models/assets")
        assert report.ok
        bundle = lb.project_resolution_report(manifest, report,
                                                 "models/assets")
    finally:
        os.chdir(previous_cwd)
    paths = [str(p) for group in bundle.to_assets().values() for p in group]
    dirs = [str(d) for d in bundle.component_dirs.values()]
    assert len(paths) == 24 and len(dirs) == 6
    code = (
        "import json, os, sys\n"
        "paths, dirs = json.loads(sys.argv[1]), json.loads(sys.argv[2])\n"
        "assert all(os.path.isabs(p) and os.path.isfile(p) for p in paths)"
        ", paths\n"
        "assert all(os.path.isabs(d) and os.path.isdir(d) for d in dirs)"
        ", dirs\n"
        "print('CWD-STABLE-OK', len(paths), len(dirs))\n"
    )
    out = subprocess.run([sys.executable, "-c", code, json.dumps(paths),
                          json.dumps(dirs)],
                         cwd=str(child_cwd), capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert "CWD-STABLE-OK 24 6" in out.stdout


def test_absolute_root_still_publishes_under_the_same_root(tmp_path):
    """Absolute fixtures keep working exactly as before (regression guard
    for fix 1): published root == the canonical absolute root."""
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    bundle = _project(manifest, root)
    assert bundle.root == root.resolve()
    assert bundle.assets["vae"] == tuple(root.resolve() / fn
                                         for fn in lb.LOCAL_REFERENCE_LAYOUT["vae"])


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_assets_are_immutable_with_defensive_copy(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    bundle = _project(manifest, root)
    with pytest.raises(TypeError):
        bundle.assets["vae"] = ()  # type: ignore[index]
    copy = bundle.to_assets()
    assert isinstance(copy, dict) and copy == dict(bundle.assets)
    copy["vae"] = ()  # the copy is plain and mutable; the bundle is untouched
    assert len(bundle.assets["vae"]) == 2
    # Shape matches GenerationAdapter.load(config, assets: Mapping[str, tuple[Path, ...]])
    assert all(isinstance(p, Path)
               for paths in copy.values() for p in paths)


def test_direct_construction_snapshots_and_freezes_mutable_inputs():
    """Review fix 3: a caller passing MUTABLE dicts/lists to the public
    frozen dataclass cannot mutate the 'immutable' bundle afterwards —
    __post_init__ snapshots and freezes the mappings and nested tuples.
    Direct construction performs NO resolver verification (honest promise):
    it shapes and freezes, it does not check disk."""
    path = Path("/nonexistent-because-unverified") / "vae.bin"
    directory = Path("/nonexistent-because-unverified/vae")
    assets = {"vae": [path]}          # mutable LIST inside mutable DICT
    component_dirs = {"vae": directory}
    bundle = lb.LocalReferenceBundle(
        profile=lb.LOCAL_REFERENCE_PROFILE,
        root=Path("/nonexistent-because-unverified"),
        root_resolved=Path("/nonexistent-because-unverified"),
        repo=REPO, revision=REVISION,
        assets=assets, component_dirs=component_dirs,
        fingerprints=lb.ArtifactSetFingerprints(bundle="a" * 64,
                                                encoder="b" * 64))
    # Values tuple-ised and the mappings frozen.
    assert bundle.assets["vae"] == (path,)
    assert isinstance(bundle.assets["vae"], tuple)
    with pytest.raises(TypeError):
        bundle.assets["vae"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        bundle.component_dirs["vae"] = Path("/nope")  # type: ignore[index]
    # Mutating the caller's ORIGINAL containers reaches nothing.
    assets["vae"] = [Path("/evil/replaced")]
    assets["bogus_role"] = ()
    component_dirs["vae"] = Path("/elsewhere")
    component_dirs["extra"] = Path("/elsewhere-2")
    assert bundle.assets["vae"] == (path,)
    assert "bogus_role" not in bundle.assets
    assert bundle.component_dirs["vae"] == directory
    assert "extra" not in bundle.component_dirs
    # Honest invariant: nothing was verified — the fabricated path simply
    # stayed a (non-existent) Path instead of raising here.
    assert not path.exists()


def test_to_dict_is_deterministic_metadata_only(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    bundle = _project(manifest, root)
    a = json.dumps(bundle.to_dict(), sort_keys=True)
    b = json.dumps(_project(manifest, root).to_dict(), sort_keys=True)
    assert a == b
    doc = bundle.to_dict()
    assert doc["fingerprints"]["bundle_artifact_set_sha256"] == bundle.fingerprints.bundle
    assert "loaded" not in json.dumps(doc)  # projection claims nothing about loading


# ---------------------------------------------------------------------------
# Layout refusals (pure data — no disk needed)
# ---------------------------------------------------------------------------


def _manifest_with(**filename_to_role_overrides) -> ModelManifest:
    """Manifest from the layout with per-file role overrides (values: role,
    or None to drop the artifact)."""
    artifacts = []
    for role, filename in _layout_pairs():
        if filename in filename_to_role_overrides:
            new_role = filename_to_role_overrides[filename]
            if new_role is None:
                continue
            role = new_role
        artifacts.append(ModelArtifact(
            role=role, repo=REPO, revision=REVISION, filename=filename,
            sha256="ab" * 32, bytes=1))
    return ModelManifest(schema_version=1, artifacts=tuple(artifacts))


def test_missing_or_unexpected_file_refused_without_disk():
    missing = _manifest_with(**{"vae/config.json": None})
    with pytest.raises(lb.BundleLayoutError, match="missing from manifest"):
        lb.validate_local_reference_layout(missing)
    extra = ModelManifest(
        schema_version=1,
        artifacts=_manifest_with().artifacts + (ModelArtifact(
            role="vae", repo=REPO, revision=REVISION,
            filename="vae/extra_lora.safetensors", sha256="ab" * 32, bytes=1),))
    with pytest.raises(lb.BundleLayoutError, match="unexpected files"):
        lb.validate_local_reference_layout(extra)


def test_wrong_role_file_mapping_refused():
    swapped = _manifest_with(**{"processor/chat_template.jinja": "vae"})
    with pytest.raises(lb.BundleLayoutError, match="role mismatch"):
        lb.validate_local_reference_layout(swapped)


def test_mixed_repo_or_revision_refused():
    arts = list(_manifest_with().artifacts)
    arts[3] = ModelArtifact(**{**arts[3].to_dict(), "revision": "0" * 40})
    with pytest.raises(lb.BundleLayoutError, match="ONE revision"):
        lb.validate_local_reference_layout(
            ModelManifest(schema_version=1, artifacts=tuple(arts)))
    arts = list(_manifest_with().artifacts)
    arts[5] = ModelArtifact(**{**arts[5].to_dict(), "repo": "someone-else/repo"})
    with pytest.raises(lb.BundleLayoutError, match="ONE repo"):
        lb.validate_local_reference_layout(
            ModelManifest(schema_version=1, artifacts=tuple(arts)))


def test_profile_pins_the_exact_reviewed_identity():
    """The named profile is THIS repo at THIS revision — the constants the
    review pinned, spelled independently here so silent edits fail."""
    assert lb.FULL_REFERENCE_REPO == "Qwen/Qwen-Image-2.1"
    assert lb.FULL_REFERENCE_REVISION == (
        "d26bb61231c349cf6b7896fa83353113880e1ba3")
    assert (REPO, REVISION) == (lb.FULL_REFERENCE_REPO,
                                lb.FULL_REFERENCE_REVISION)


def test_all_artifacts_wrong_repo_refused():
    """An internally consistent manifest at ANOTHER repo is NOT the reviewed
    full-reference profile: layout validation refuses it (review item 1),
    so the pinned interface assumptions behind the profile name cannot be
    inherited by an arbitrary consistent repo."""
    wrong = ModelManifest(schema_version=1, artifacts=tuple(
        ModelArtifact(**{**a.to_dict(), "repo": "Someone-Else/Qwen-Image-2.1"})
        for a in _manifest_with().artifacts))
    with pytest.raises(lb.BundleLayoutError, match="pinned to repo"):
        lb.validate_local_reference_layout(wrong)


def test_all_artifacts_wrong_revision_refused():
    """Even a fully consistent, plausible 40-hex revision other than the
    reviewed one is refused: a new model revision requires a NEW reviewed
    profile, not a relaxed check here (review item 1)."""
    wrong = ModelManifest(schema_version=1, artifacts=tuple(
        ModelArtifact(**{**a.to_dict(), "revision": "f" * 40})
        for a in _manifest_with().artifacts))
    with pytest.raises(lb.BundleLayoutError, match="pinned to revision"):
        lb.validate_local_reference_layout(wrong)


# ---------------------------------------------------------------------------
# Report refusals
# ---------------------------------------------------------------------------


def test_incomplete_resolver_report_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    (root / "vae" / "config.json").unlink()
    report = resolve_manifest(manifest, root)
    assert not report.ok
    with pytest.raises(lb.BundleReportError, match="not successful"):
        lb.project_resolution_report(manifest, report, root)


def test_empty_root_report_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    for _, fn in _layout_pairs():
        (root / fn).unlink()
    report = resolve_manifest(manifest, root)
    with pytest.raises(lb.BundleReportError, match="not successful"):
        lb.project_resolution_report(manifest, report, root)


def test_missing_root_refused(tmp_path):
    root = tmp_path / "nowhere"
    manifest = _manifest_with()
    report = _forged_report(manifest, root)
    with pytest.raises(lb.BundlePathError, match="not a directory"):
        lb.project_resolution_report(manifest, report, root)


def test_reordered_statuses_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    report = resolve_manifest(manifest, root)
    reordered = ResolutionReport(
        ok=True, resolved=report.resolved,
        statuses=report.statuses[1:] + report.statuses[:1], errors=())
    with pytest.raises(lb.BundleReportError, match="does not match"):
        lb.project_resolution_report(manifest, reordered, root)


def test_extra_and_short_status_lists_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    report = resolve_manifest(manifest, root)
    pad = report.statuses[0]
    with pytest.raises(lb.BundleReportError, match="statuses but the manifest"):
        lb.project_resolution_report(manifest, ResolutionReport(
            ok=True, resolved=report.resolved,
            statuses=report.statuses + (pad,), errors=()), root)
    with pytest.raises(lb.BundleReportError, match="statuses but the manifest"):
        lb.project_resolution_report(manifest, ResolutionReport(
            ok=True, resolved=report.resolved,
            statuses=report.statuses[:-1], errors=()), root)


def test_forged_status_fields_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    report = resolve_manifest(manifest, root)
    statuses = list(report.statuses)
    victim = statuses[7]
    forged_art = ModelArtifact(**{**victim.artifact.to_dict(),
                                  "sha256": "0" * 64})
    statuses[7] = ArtifactStatus(forged_art, STATE_PRESENT, path=victim.path)
    with pytest.raises(lb.BundleReportError, match="forged"):
        lb.project_resolution_report(manifest, ResolutionReport(
            ok=True, resolved=report.resolved,
            statuses=tuple(statuses), errors=()), root)


def test_forged_status_with_non_present_state_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    report = resolve_manifest(manifest, root)
    statuses = list(report.statuses)
    statuses[13] = ArtifactStatus(statuses[13].artifact, STATE_MISSING)
    with pytest.raises(lb.BundleReportError, match="state"):
        lb.project_resolution_report(manifest, ResolutionReport(
            ok=True, resolved=report.resolved,
            statuses=tuple(statuses), errors=()), root)


def test_status_path_not_matching_root_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    report = resolve_manifest(manifest, root)
    statuses = list(report.statuses)
    statuses[2] = ArtifactStatus(statuses[2].artifact, STATE_PRESENT,
                                 path=root.parent / "elsewhere" / "x")
    with pytest.raises(lb.BundleReportError, match="is not the expected"):
        lb.project_resolution_report(manifest, ResolutionReport(
            ok=True, resolved=report.resolved,
            statuses=tuple(statuses), errors=()), root)


def test_report_from_a_different_root_refused(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_b.mkdir()  # exists but holds nothing: the report's paths must not match
    manifest = build_tiny_tree(root_a)
    report = resolve_manifest(manifest, root_a)
    with pytest.raises(lb.BundleReportError, match="is not the expected"):
        lb.project_resolution_report(manifest, report, root_b)


def test_status_from_other_manifest_reordered_within_same_fileset(tmp_path):
    """A report whose statuses carry the same files but built from a
    reordered manifest copy is refused positionally."""
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    other = ModelManifest(schema_version=1, artifacts=tuple(
        reversed(manifest.artifacts)))
    report = resolve_manifest(other, root)
    with pytest.raises(lb.BundleReportError, match="does not match|statuses but"):
        lb.project_resolution_report(manifest, report, root)


# ---------------------------------------------------------------------------
# Report COHERENCE (review fix 2): ok=True alone never smuggles a broken
# success claim — errors, per-status error, resolved mapping all checked
# ---------------------------------------------------------------------------


def test_nonempty_errors_despite_ok_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    victim = manifest.artifacts[3]
    err = ArtifactError(victim, "hand-forged contradiction")
    with pytest.raises(lb.BundleReportError, match="no errors"):
        lb.project_resolution_report(
            manifest, _forged_report(manifest, root, errors=(err,)), root)


def test_review_example_ok_with_errors_and_empty_resolved_refused(tmp_path):
    """The exact review hand-built shape: ok=True, all-present statuses,
    errors=(some_error,), resolved={} — refused (errors checked first)."""
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    err = ArtifactError(manifest.artifacts[0], "example")
    with pytest.raises(lb.BundleReportError):
        lb.project_resolution_report(manifest, ResolutionReport(
            ok=True,
            statuses=tuple(ArtifactStatus(a, STATE_PRESENT,
                                          path=root / a.identity)
                           for a in manifest.artifacts),
            resolved={}, errors=(err,)), root)


def test_present_status_carrying_error_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    statuses = list(_forged_report(manifest, root).statuses)
    victim = statuses[5]
    statuses[5] = ArtifactStatus(victim.artifact, STATE_PRESENT,
                                 error=ArtifactError(victim.artifact, "forged"),
                                 path=victim.path)
    with pytest.raises(lb.BundleReportError, match="carries error"):
        lb.project_resolution_report(
            manifest,
            _forged_report(manifest, root, statuses=tuple(statuses)), root)


def test_resolved_mapping_must_equal_grouped_status_paths(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    good = _forged_report(manifest, root)
    assert set(good.resolved) == set(lb.LOCAL_REFERENCE_LAYOUT)

    def refute(mutate, match="report.resolved"):
        broken = dict(good.resolved)
        mutate(broken)
        with pytest.raises(lb.BundleReportError, match=match):
            lb.project_resolution_report(
                manifest, _forged_report(manifest, root, resolved=broken),
                root)

    refute(lambda d: d.clear())                                # empty
    refute(lambda d: d.pop("vae"))                             # missing key
    refute(lambda d: d.__setitem__("rogue_role", ()))          # extra key
    refute(lambda d: d.__setitem__("vae", ()))                 # empty group
    refute(lambda d: d.__setitem__(                            # reordered
        "processor", tuple(reversed(d["processor"]))))
    refute(lambda d: d.__setitem__(                            # wrong path
        "dit", (Path(str(d["dit"][0]) + ".evil"),) + d["dit"][1:]))
    # And the real resolver's report for the same root trivially matches —
    # the check polices hand-built reports, not honest ones.
    honest = resolve_manifest(manifest, root)
    assert lb.project_resolution_report(manifest, honest, root).assets


# ---------------------------------------------------------------------------
# Trusted factory vs trusted-caller projection (review item 2): resolver
# provenance is an explicit API split, not an implied property of the
# projection
# ---------------------------------------------------------------------------


def test_trusted_factory_positive_projection(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    bundle = lb.resolve_local_bundle(manifest, root)
    honest = _project(manifest, root)
    assert bundle.to_dict() == honest.to_dict()


def test_trusted_factory_refuses_changed_content_and_projection_is_not_proof(tmp_path):
    """Content changes AFTER the tree was built: the trusted factory runs
    resolve_manifest itself and REFUSES (the rehash sees the tamper), while
    the lower-level projection — which openly trusts its caller's report —
    still projects. That asymmetry IS the contract: a forged shape-
    consistent report is NOT verified evidence (review item 2)."""
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    (root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(
        b"tampered-after-the-original-bytes")
    with pytest.raises(lb.BundleReportError, match="not successful"):
        lb.resolve_local_bundle(manifest, root)
    # A hand-built ok=True report says nothing about bytes; the cheap
    # structural probes pass (file exists, regular, inside root) because the
    # projection never rehashes. Honest, documented, NOT authenticated.
    bundle = lb.project_resolution_report(
        manifest, _forged_report(manifest, root), root)
    assert bundle.assets["vae"]
    tamper = hashlib.sha256(b"tampered-after-the-original-bytes").hexdigest()
    assert not any(a.sha256 == tamper for a in manifest.artifacts)


def test_trusted_factory_refuses_missing_file(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    (root / "processor" / "tokenizer.json").unlink()
    with pytest.raises(lb.BundleReportError, match="not successful"):
        lb.resolve_local_bundle(manifest, root)


def test_trusted_factory_refuses_off_profile_manifest_before_any_hash(tmp_path):
    """Layout validation runs BEFORE resolve_manifest, so an off-profile
    manifest never triggers the (tiny: per-file) hash walk at all."""
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root, revision="e" * 40)  # consistent, wrong
    with pytest.raises(lb.BundleLayoutError, match="pinned to revision"):
        lb.resolve_local_bundle(manifest, root)


def test_api_docstrings_state_the_provenance_split():
    """The lower-level projection must not describe itself as authenticated:
    its contract says it trusts the caller; the factory says it runs the
    trusted resolver itself (and names the TOCTOU limit)."""
    proj_doc = lb.project_resolution_report.__doc__ or ""
    factory_doc = lb.resolve_local_bundle.__doc__ or ""
    assert "TRUSTS ITS CALLER" in proj_doc
    assert "resolve_manifest" in factory_doc
    assert "TOCTOU" in factory_doc and "TOCTOU" in proj_doc


# ---------------------------------------------------------------------------
# Structural probes: forged 'present' statuses must not survive the probe
# ---------------------------------------------------------------------------


def test_escaped_symlink_file_refused_despite_forged_ok(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    real = root / "vae" / "diffusion_pytorch_model.safetensors"
    real.rename(outside / "weights.bin")
    real.symlink_to(outside / "weights.bin")
    # The REAL resolver marks this path_escape -> not ok. A forged 'ok'
    # report must still be refused by the projection's own probe.
    report = resolve_manifest(manifest, root)
    assert not report.ok
    with pytest.raises(lb.BundleReportError):
        lb.project_resolution_report(manifest, report, root)
    with pytest.raises(lb.BundlePathError, match="outside the model root"):
        lb.project_resolution_report(
            manifest, _forged_report(manifest, root), root)


def test_directory_swapped_for_file_refused_despite_forged_ok(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    victim = root / "vae" / "config.json"
    victim.unlink()
    victim.mkdir()
    with pytest.raises(lb.BundlePathError, match="not a regular file"):
        lb.project_resolution_report(
            manifest, _forged_report(manifest, root), root)


def test_deleted_file_refused_despite_forged_ok(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    (root / "text_encoder" / "model-00003-of-00004.safetensors").unlink()
    with pytest.raises(lb.BundlePathError, match="does not exist"):
        lb.project_resolution_report(
            manifest, _forged_report(manifest, root), root)


def test_root_itself_a_regular_file_refused(tmp_path):
    root = tmp_path / "not-a-dir"
    root.write_bytes(b"x")
    manifest = _manifest_with()
    with pytest.raises(lb.BundlePathError, match="not a directory"):
        lb.project_resolution_report(
            manifest, _forged_report(manifest, root), root)


def test_within_root_symlink_allowed_like_the_resolver(tmp_path):
    """HF-cache-style symlinks whose TARGET stays inside the root are legal
    for the resolver; the projection stays consistent (regular file, inside
    root), so it must NOT reject them."""
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    link = root / "processor" / "merges.txt"
    target = root / "processor" / "vocab.json"
    link.unlink()
    link.symlink_to(target)
    # The tiny fixture's manifest sha for merges.txt described the OLD bytes,
    # so the real resolver FAILS on hash mismatch (content changed). Prove
    # both halves: (a) real resolver refuses; (b) the projection's structural
    # probe alone (forged ok report, same root/manifest) accepts the
    # inside-root symlink — consistency with resolver symlink semantics.
    assert not resolve_manifest(manifest, root).ok
    bundle = lb.project_resolution_report(
        manifest, _forged_report(manifest, root), root)
    assert bundle.assets["processor"]


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def test_fingerprints_independent_of_manifest_ordering(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    shuffled_arts = list(reversed(manifest.artifacts))
    other = ModelManifest(schema_version=1, artifacts=tuple(shuffled_arts))
    fps_a = lb.reference_bundle_fingerprints(manifest)
    fps_b = lb.reference_bundle_fingerprints(other)
    assert fps_a.bundle == fps_b.bundle
    assert fps_a.encoder == fps_b.encoder
    # And the projected bundle carries the same digests whatever order.
    assert _project(other, root).fingerprints == fps_a


def test_altered_identity_changes_the_relevant_fingerprint(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    base = lb.reference_bundle_fingerprints(manifest)

    def swap(filename: str, **fields) -> lb.ArtifactSetFingerprints:
        arts = []
        for a in manifest.artifacts:
            arts.append(ModelArtifact(**{**a.to_dict(), **fields})
                          if a.identity == filename else a)
        return lb.reference_bundle_fingerprints(
            ModelManifest(schema_version=1, artifacts=tuple(arts)))

    # An encoder shard's hash flips BOTH digests.
    enc = swap("text_encoder/model-00002-of-00004.safetensors", sha256="cd" * 32)
    assert enc.bundle != base.bundle and enc.encoder != base.encoder
    # The chat template (processor role) is part of the encoder digest.
    tpl = swap("processor/chat_template.jinja", sha256="ef" * 32)
    assert tpl.encoder != base.encoder and tpl.bundle != base.bundle
    # A VAE-only change flips the bundle digest but NOT the encoder one.
    vae = swap("vae/diffusion_pytorch_model.safetensors", bytes=9999999)
    assert vae.bundle != base.bundle and vae.encoder == base.encoder
    # A byte-size (not hash) change still changes identity.
    size = swap("scheduler/scheduler_config.json", bytes=4242)
    assert size.bundle != base.bundle
    # repo/revision changes flip the GENERIC digest function: it is an
    # identity function over whatever it is given (digest drift stays
    # demonstrable), while the PROFILE-level API refuses a different
    # identity before any digest is taken (review item 1).
    rev_arts = tuple(ModelArtifact(**{**a.to_dict(), "revision": "1" * 40})
                     for a in manifest.artifacts)
    rev = ModelManifest(schema_version=1, artifacts=rev_arts)
    assert lb.artifact_set_fingerprint(rev_arts) != base.bundle
    repo_arts = tuple(ModelArtifact(**{**a.to_dict(),
                                       "repo": "someone-else/model"})
                      for a in manifest.artifacts)
    assert lb.artifact_set_fingerprint(repo_arts) != base.bundle
    with pytest.raises(lb.BundleLayoutError, match="pinned to"):
        lb.reference_bundle_fingerprints(rev)


def test_fingerprint_needs_one_repo_one_revision_and_artifacts():
    arts_a = [ModelArtifact(role="vae", repo="r/x", revision="a" * 40,
                            filename="vae/config.json", sha256="ab" * 32, bytes=1)]
    arts_b = [ModelArtifact(role="vae", repo="r/y", revision="a" * 40,
                             filename="vae/diffusion_pytorch_model.safetensors",
                             sha256="ab" * 32, bytes=1)]
    with pytest.raises(lb.BundleLayoutError, match="one repo"):
        lb.artifact_set_fingerprint(arts_a + arts_b)
    with pytest.raises(lb.BundleLayoutError, match="at least one"):
        lb.artifact_set_fingerprint([])


def test_fingerprint_version_tag_changes_digest():
    manifest = load_manifest(MANIFEST_PATH)
    base = lb.reference_bundle_fingerprints(manifest).bundle
    original = lb.ARTIFACT_SET_FINGERPRINT_VERSION
    try:
        lb.ARTIFACT_SET_FINGERPRINT_VERSION = original + ".mutant"
        assert lb.reference_bundle_fingerprints(manifest).bundle != base
    finally:
        lb.ARTIFACT_SET_FINGERPRINT_VERSION = original


def test_model_sha256_is_not_one_shard_documented_by_digest_names(tmp_path):
    """The two digests differ from each individual file hash — honest
    metadata, never a single shard presented as the set."""
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    fps = lb.reference_bundle_fingerprints(manifest)
    individual = {a.sha256 for a in manifest.artifacts}
    assert fps.bundle not in individual and fps.encoder not in individual


# ---------------------------------------------------------------------------
# Import hygiene (fresh interpreter; guard env rides via os.environ)
# ---------------------------------------------------------------------------


def test_module_import_is_ml_free():
    src = str(ROOT / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys, qwen_engine.local_bundle as lb\n"
        "heavy = [m for m in ('torch', 'diffusers', 'transformers')"
        " if m in sys.modules]\n"
        "assert not heavy, heavy\n"
        "assert lb.LAYOUT_FILE_COUNT == 24\n"
        "print('ML-FREE-OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "ML-FREE-OK" in out.stdout


# ---------------------------------------------------------------------------
# Closed-tree preflight (review item 3): a from_pretrained(component_dir)
# must never meet a file that the manifest never listed
# ---------------------------------------------------------------------------

#: Recognized-LOOKING extras: config-style, weight-style and code-style
#: names a future loader could plausibly open even though the manifest does
#: not list them (review item 3).
EXTRA_LOOKALIKE_FILES = [
    "model_index.json.bak",              # stray backup at the root
    "pipeline.py",                       # shadowing code file
    "scheduler/extra_scheduler.json",    # extra config-style file
    "transformer/model.safetensors",     # extra weight-style file
    "vae/diffusion_pytorch_model-00002-of-00002.safetensors",  # fake shard
    "text_encoder/extra/generation_config.json",  # extra directory + file
]


def test_preflight_accepts_the_exact_fake_tree(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    bundle = _project(manifest, root)
    checked = bundle.preflight_closed_tree()
    # Every layout file plus every required component directory was checked.
    flat = {fn for fns in lb.LOCAL_REFERENCE_LAYOUT.values() for fn in fns}
    dirs = {d for d in lb.ROLE_DIRECTORIES.values() if d != "."}
    assert set(checked) == flat | dirs
    assert len(checked) == len(flat) + len(dirs)
    # And the module-level function agrees on the same root.
    assert lb.preflight_closed_tree(root) == checked


def test_preflight_refuses_extra_recognized_looking_files(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    for rel in EXTRA_LOOKALIKE_FILES:
        stray = root / rel
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_bytes(b"not in the manifest")
        with pytest.raises(lb.BundleTreeError,
                           match="NOT part of the closed"):
            lb.preflight_closed_tree(root)


def test_preflight_refuses_leftover_partial_before_any_ml_import(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    debris = (root / "transformer"
              / "diffusion_pytorch_model-00001-of-00002.safetensors.partial")
    debris.write_bytes(b"interrupted fetch debris")
    with pytest.raises(lb.BundleTreeError, match="NOT part of the closed"):
        lb.preflight_closed_tree(root)


def test_preflight_refuses_symlink_even_inside_root(tmp_path):
    """Deliberately stricter than the resolver/projection: an allowlisted
    path reached through a symlink redirects whatever from_pretrained
    opens, so the closed tree refuses links outright."""
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    link = root / "processor" / "merges.txt"
    target = root / "processor" / "vocab.json"
    link.unlink()
    link.symlink_to(target)  # target stays INSIDE the root
    with pytest.raises(lb.BundleTreeError, match="is a symlink"):
        lb.preflight_closed_tree(root)


def test_preflight_refuses_extra_directory_even_when_empty(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    (root / "vae" / "cache").mkdir()
    with pytest.raises(lb.BundleTreeError, match="NOT part of the closed"):
        lb.preflight_closed_tree(root)


def test_preflight_refuses_wrong_entry_kind_and_missing_entry(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    victim = root / "vae" / "config.json"
    victim.unlink()
    victim.mkdir()  # directory where an allowlisted file belongs
    with pytest.raises(lb.BundleTreeError, match="not a regular file"):
        lb.preflight_closed_tree(root)
    victim.rmdir()
    victim.write_bytes(b"x" * 3)
    (root / "scheduler" / "scheduler_config.json").unlink()
    with pytest.raises(lb.BundleTreeError, match="missing from the closed"):
        lb.preflight_closed_tree(root)


def test_preflight_missing_root_refused(tmp_path):
    with pytest.raises(lb.BundleTreeError, match="not a directory"):
        lb.preflight_closed_tree(tmp_path / "nowhere")
