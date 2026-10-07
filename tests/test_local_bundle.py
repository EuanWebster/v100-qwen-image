"""M2.2a offline tests for the local reference-bundle contract
(``qwen_engine.local_bundle``): pure projection of resolver-verified paths.

All fixtures fabricate **tiny fake contents** inside pytest ``tmp_path`` with
the exact full-reference layout (24 files, 6 roles, one repo/revision) and
run them through the **real** ``resolve_manifest`` — no 33 GB, no network, no
heavy imports, no reads or writes of ``models/assets/`` (which may be
concurrently downloading; nothing here touches it). The committed
``models/manifest.json`` is exercised as data only (layout + disk-free
fingerprints).
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


def _forged_report(manifest: ModelManifest, root: Path,
                   skip: set[str] | None = None) -> ResolutionReport:
    """A hand-built 'present' report (NOT from the resolver) — used to prove
    the projection's own cheap probes refuse forged status entries."""
    statuses = tuple(
        ArtifactStatus(a, STATE_PRESENT, path=root / a.identity)
        for a in manifest.artifacts if a.identity not in (skip or set()))
    return ResolutionReport(ok=True, resolved={}, statuses=statuses, errors=())


def _project(manifest: ModelManifest, root: Path) -> lb.LocalReferenceBundle:
    report = resolve_manifest(manifest, root)
    assert report.ok, [s.to_dict() for s in report.statuses if not s.error is None]
    return lb.local_bundle_from_resolution(manifest, report, root)


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

    assert bundle.profile == lb.LOCAL_REFERENCE_PROFILE
    assert (bundle.repo, bundle.revision) == (REPO, REVISION)
    assert bundle.root == root
    assert set(bundle.assets) == set(lb.LOCAL_REFERENCE_LAYOUT)
    for role, fns in lb.LOCAL_REFERENCE_LAYOUT.items():
        paths = bundle.assets[role]
        assert len(paths) == EXPECTED_ROLE_COUNTS[role]
        # Manifest order preserved; paths are literally root/filename.
        assert paths == tuple(root / fn for fn in fns)
        assert all(p.is_relative_to(root) for p in paths)
    # Component dirs are the explicit local targets for future
    # local_files_only=True probes — all under the given root.
    assert bundle.pipeline_dir == root
    assert bundle.component_dirs["text_encoder"] == root / "text_encoder"
    assert bundle.component_dirs["dit"] == root / "transformer"
    assert all(str(d).startswith(str(root)) for d in bundle.component_dirs.values())


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
        lb.local_bundle_from_resolution(manifest, report, root)


def test_empty_root_report_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    for _, fn in _layout_pairs():
        (root / fn).unlink()
    report = resolve_manifest(manifest, root)
    with pytest.raises(lb.BundleReportError, match="not successful"):
        lb.local_bundle_from_resolution(manifest, report, root)


def test_missing_root_refused(tmp_path):
    root = tmp_path / "nowhere"
    manifest = _manifest_with()
    report = _forged_report(manifest, root)
    with pytest.raises(lb.BundlePathError, match="not a directory"):
        lb.local_bundle_from_resolution(manifest, report, root)


def test_reordered_statuses_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    report = resolve_manifest(manifest, root)
    reordered = ResolutionReport(
        ok=True, resolved=report.resolved,
        statuses=report.statuses[1:] + report.statuses[:1], errors=())
    with pytest.raises(lb.BundleReportError, match="does not match"):
        lb.local_bundle_from_resolution(manifest, reordered, root)


def test_extra_and_short_status_lists_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    report = resolve_manifest(manifest, root)
    pad = report.statuses[0]
    with pytest.raises(lb.BundleReportError, match="statuses but the manifest"):
        lb.local_bundle_from_resolution(manifest, ResolutionReport(
            ok=True, resolved=report.resolved,
            statuses=report.statuses + (pad,), errors=()), root)
    with pytest.raises(lb.BundleReportError, match="statuses but the manifest"):
        lb.local_bundle_from_resolution(manifest, ResolutionReport(
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
        lb.local_bundle_from_resolution(manifest, ResolutionReport(
            ok=True, resolved=report.resolved,
            statuses=tuple(statuses), errors=()), root)


def test_forged_status_with_non_present_state_refused(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    report = resolve_manifest(manifest, root)
    statuses = list(report.statuses)
    statuses[13] = ArtifactStatus(statuses[13].artifact, STATE_MISSING)
    with pytest.raises(lb.BundleReportError, match="state"):
        lb.local_bundle_from_resolution(manifest, ResolutionReport(
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
        lb.local_bundle_from_resolution(manifest, ResolutionReport(
            ok=True, resolved=report.resolved,
            statuses=tuple(statuses), errors=()), root)


def test_report_from_a_different_root_refused(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_b.mkdir()  # exists but holds nothing: the report's paths must not match
    manifest = build_tiny_tree(root_a)
    report = resolve_manifest(manifest, root_a)
    with pytest.raises(lb.BundleReportError, match="is not the expected"):
        lb.local_bundle_from_resolution(manifest, report, root_b)


def test_status_from_other_manifest_reordered_within_same_fileset(tmp_path):
    """A report whose statuses carry the same files but built from a
    reordered manifest copy is refused positionally."""
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    other = ModelManifest(schema_version=1, artifacts=tuple(
        reversed(manifest.artifacts)))
    report = resolve_manifest(other, root)
    with pytest.raises(lb.BundleReportError, match="does not match|statuses but"):
        lb.local_bundle_from_resolution(manifest, report, root)


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
        lb.local_bundle_from_resolution(manifest, report, root)
    with pytest.raises(lb.BundlePathError, match="outside the model root"):
        lb.local_bundle_from_resolution(
            manifest, _forged_report(manifest, root), root)


def test_directory_swapped_for_file_refused_despite_forged_ok(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    victim = root / "vae" / "config.json"
    victim.unlink()
    victim.mkdir()
    with pytest.raises(lb.BundlePathError, match="not a regular file"):
        lb.local_bundle_from_resolution(
            manifest, _forged_report(manifest, root), root)


def test_deleted_file_refused_despite_forged_ok(tmp_path):
    root = tmp_path / "assets"
    manifest = build_tiny_tree(root)
    (root / "text_encoder" / "model-00003-of-00004.safetensors").unlink()
    with pytest.raises(lb.BundlePathError, match="does not exist"):
        lb.local_bundle_from_resolution(
            manifest, _forged_report(manifest, root), root)


def test_root_itself_a_regular_file_refused(tmp_path):
    root = tmp_path / "not-a-dir"
    root.write_bytes(b"x")
    manifest = _manifest_with()
    with pytest.raises(lb.BundlePathError, match="not a directory"):
        lb.local_bundle_from_resolution(
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
    bundle = lb.local_bundle_from_resolution(
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
    # repo/revision changes flip everything (validated at layout level too).
    rev = ModelManifest(schema_version=1, artifacts=tuple(
        ModelArtifact(**{**a.to_dict(), "revision": "1" * 40})
        for a in manifest.artifacts))
    assert lb.reference_bundle_fingerprints(rev).bundle != base.bundle


def test_fingerprint_needs_one_repo_one_revision_and_artifacts():
    arts_a = [ModelArtifact(role="vae", repo="r/x", revision="a" * 40,
                            filename="vae/config.json", sha256="ab" * 32, bytes=1)]
    arts_b = [ModelArtifact(role="vae", repo="r/y", revision="a" * 40,
                             filename="vae/config.json", sha256="ab" * 32, bytes=1)]
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
