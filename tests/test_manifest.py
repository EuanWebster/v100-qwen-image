"""Task 1.2 acceptance: versioned model-manifest schema + no-download resolver (R3),
extended by the senior-review fix pass:

* multiple artifacts per **role** (shards/indexes/configs); unique normalized
  **filename** identity; resolved paths grouped by role as tuples;
* validation on **direct dataclass construction** as well as JSON parsing;
* containment-checked, fd-single-open, change-detecting verification;
* typed errors distinguish missing / unreadable / path-escape / non-regular.

Tests block sockets and still expect clean structured results. No network, no
GPU, no weights (R7).
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import stat as stat_module
import sys

import pytest

from qwen_engine import manifest as mf

HEAVY_MODULES = ("torch", "diffusers", "transformers")

REPO = "Qwen/Qwen-Image-2.1"
REV = "0" * 40


def artifact_dict(role="dit", filename=None, repo=REPO, revision=REV,
                  content=b"payload", size=None, sha=None):
    filename = filename or f"{role}/model.safetensors"
    return {
        "role": role,
        "repo": repo,
        "revision": revision,
        "filename": filename,
        "sha256": sha or hashlib.sha256(content).hexdigest(),
        "bytes": len(content) if size is None else size,
    }


def manifest_dict(*artifacts, schema_version=mf.MANIFEST_SCHEMA_VERSION):
    return {"schema_version": schema_version,
            "artifacts": list(artifacts or [artifact_dict()])}


def write_manifest(tmp_path, data, name="manifest.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def materialize(tmp_path, manifest, role="dit", content=b"payload", index=0):
    """Write the real artifact file for one role (optionally one shard) under models_dir."""
    arts = [a for a in manifest.artifacts if a.role == role]
    art = arts[index]
    target = tmp_path / art.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def block_network(monkeypatch):
    import urllib.request  # import *before* patching so ssl's import isn't broken

    def explode(*a, **k):
        raise AssertionError("resolver attempted network access")
    monkeypatch.setattr(socket, "socket", explode)
    monkeypatch.setattr(socket, "create_connection", explode)
    monkeypatch.setattr(urllib.request, "urlopen", explode)


# --- schema / parsing -----------------------------------------------------------


def test_parses_valid_manifest(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    assert m.schema_version == mf.MANIFEST_SCHEMA_VERSION
    assert m.artifacts[0].role == "dit"
    assert m.artifacts[0].repo == REPO


@pytest.mark.parametrize("version", [99, "1", 1.0, True, None, 0])
def test_schema_version_must_be_exact_int_1(tmp_path, version):
    """bools and floats are not ints here; 1.0 == 1 must not sneak through."""
    path = write_manifest(tmp_path, manifest_dict(schema_version=version))
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(path)
    assert "schema_version" in str(exc.value)


def test_malformed_manifest_json_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{nope")
    with pytest.raises(mf.ManifestFormatError):
        mf.load_manifest(path)


def test_invalid_utf8_manifest_is_typed_format_error(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_bytes(b'{"schema_version": 1, "artifacts": "\xff\xfe"}')
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(path)
    assert "UTF-8" in str(exc.value)
    # dry-run status stays a structured dict even for this
    status = mf.manifest_status(path, tmp_path)
    assert status["status"] == "manifest_invalid"


def test_manifest_path_with_nul_is_typed_format_error(tmp_path):
    with pytest.raises(mf.ManifestFormatError):
        mf.load_manifest(tmp_path / "man\x00ifest.json")


@pytest.mark.parametrize("field,bad_artifact", [
    ("sha256", lambda: {**artifact_dict(), "sha256": "nothex"}),
    ("sha256", lambda: {**artifact_dict(), "sha256": hashlib.sha256(b"x").hexdigest() + "\n"}),
    ("sha256", lambda: {**artifact_dict(), "sha256": hashlib.sha256(b"x").hexdigest().upper()}),
    ("bytes", lambda: {**artifact_dict(), "bytes": -5}),
    ("bytes", lambda: {**artifact_dict(), "bytes": True}),
    ("bytes", lambda: {**artifact_dict(), "bytes": 7.0}),
    ("repo", lambda: artifact_dict(repo="")),
    ("revision", lambda: artifact_dict(revision="main")),
    ("revision", lambda: artifact_dict(revision="v1.2.3")),
    ("revision", lambda: artifact_dict(revision="0" * 39)),
    ("filename", lambda: artifact_dict(filename="a/./b")),
    ("filename", lambda: artifact_dict(filename="a\x00b")),
    ("filename", lambda: artifact_dict(filename="a//b")),
    ("filename", lambda: artifact_dict(filename=".")),
])
def test_invalid_artifact_fields_name_the_field(tmp_path, field, bad_artifact):
    path = write_manifest(tmp_path, manifest_dict(bad_artifact()))
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(path)
    assert field in str(exc.value)


def test_unknown_keys_rejected_manifest_and_artifact(tmp_path):
    data = manifest_dict({**artifact_dict(), "checksum": "sha1"})
    data["tole"] = "misc"  # typo'd top-level key
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(write_manifest(tmp_path, data))
    text = str(exc.value)
    assert "tole" in text and "checksum" in text


def test_artifacts_must_be_a_list(tmp_path):
    data = manifest_dict()
    data["artifacts"] = "not-a-list"
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(write_manifest(tmp_path, data))
    assert "artifacts" in str(exc.value)


def test_duplicate_filename_identity_rejected(tmp_path):
    path = write_manifest(tmp_path,
                          manifest_dict(artifact_dict("dit", "a/x"),
                                        artifact_dict("vae", "a/x")))
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(path)
    assert "identity" in str(exc.value)


def test_repeated_roles_are_allowed(tmp_path):
    m = mf.load_manifest(write_manifest(
        tmp_path,
        manifest_dict(artifact_dict("dit", "dit/model-00001.safetensors"),
                      artifact_dict("dit", "dit/model-00002.safetensors"),
                      artifact_dict("dit", "dit/index.json"))))
    assert [a.role for a in m.artifacts] == ["dit", "dit", "dit"]


def test_all_problems_reported_together(tmp_path):
    path = write_manifest(tmp_path, manifest_dict(
        {**artifact_dict(), "sha256": "bad", "bytes": 0}))
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(path)
    text = str(exc.value)
    assert "sha256" in text and "bytes" in text


# --- direct dataclass construction ----------------------------------------------


def test_direct_artifact_construction_validates():
    art = mf.ModelArtifact("dit", REPO, REV, "dit/model.bin",
                           hashlib.sha256(b"payload").hexdigest(), 7)
    assert art.identity == "dit/model.bin"
    with pytest.raises(mf.ManifestFormatError):
        mf.ModelArtifact("dit", REPO, "main", "dit/model.bin",
                         hashlib.sha256(b"payload").hexdigest(), 7)
    with pytest.raises(mf.ManifestFormatError):
        mf.ModelArtifact("dit", REPO, REV, "../escape", "0" * 64, 7)
    with pytest.raises(mf.ManifestFormatError):
        mf.ModelArtifact("dit", REPO, REV, "a\x00b", "0" * 64, 7)
    with pytest.raises(mf.ManifestFormatError):
        mf.ModelArtifact("dit", REPO, REV, "a.bin", "0" * 64, True)


def test_direct_manifest_construction_validates():
    art = mf.ModelArtifact("dit", REPO, REV, "dit/model.bin",
                           hashlib.sha256(b"payload").hexdigest(), 7)
    m = mf.ModelManifest(mf.MANIFEST_SCHEMA_VERSION, (art,))
    assert isinstance(m.artifacts, tuple)
    with pytest.raises(mf.ManifestFormatError):
        mf.ModelManifest(True, (art,))          # bool is not the int 1
    with pytest.raises(mf.ManifestFormatError):
        mf.ModelManifest(1.0, (art,))           # float is not the int 1
    with pytest.raises(mf.ManifestFormatError):
        mf.ModelManifest(mf.MANIFEST_SCHEMA_VERSION, ())
    with pytest.raises(mf.ManifestFormatError):
        mf.ModelManifest(mf.MANIFEST_SCHEMA_VERSION, [art, art])  # same identity
    with pytest.raises(mf.ManifestFormatError):
        mf.ModelManifest(mf.MANIFEST_SCHEMA_VERSION, ["not-an-artifact"])


def test_manifest_immutable_and_artifacts_tuple():
    art = mf.ModelArtifact("dit", REPO, REV, "dit/model.bin",
                           hashlib.sha256(b"payload").hexdigest(), 7)
    m = mf.ModelManifest(mf.MANIFEST_SCHEMA_VERSION, [art])  # list coerced to tuple
    assert isinstance(m.artifacts, tuple)
    with pytest.raises(Exception):
        m.schema_version = 2  # type: ignore[misc]


# --- resolution (verify-only, no downloads) -------------------------------------


def test_resolve_present_artifact_returns_grouped_tuple(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    materialize(tmp_path, m)
    result = mf.resolve_manifest(m, tmp_path)
    assert result.ok
    assert result.resolved["dit"] == (tmp_path / "dit/model.safetensors",)


def test_missing_artifact_is_typed_error_naming_repo_revision_file(tmp_path, monkeypatch):
    block_network(monkeypatch)
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    result = mf.resolve_manifest(m, tmp_path)
    assert not result.ok
    err = result.errors[0]
    assert isinstance(err, mf.ArtifactMissingError)
    text = str(err)
    assert REPO in text and REV in text and "dit/model.safetensors" in text
    assert err.repo == REPO and err.revision == REV and err.filename == "dit/model.safetensors"


def test_size_mismatch_is_typed_error(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    materialize(tmp_path, m, content=b"too short")
    result = mf.resolve_manifest(m, tmp_path)
    err = result.errors[0]
    assert isinstance(err, mf.ArtifactSizeMismatchError)
    assert err.expected_bytes == len(b"payload") and err.actual_bytes == len(b"too short")
    assert "dit/model.safetensors" in str(err) and REPO in str(err)


def test_hash_mismatch_is_typed_error_even_at_right_size(tmp_path):
    content = b"X" * len(b"payload")
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    materialize(tmp_path, m, content=content)
    result = mf.resolve_manifest(m, tmp_path)
    err = result.errors[0]
    assert isinstance(err, mf.ArtifactHashMismatchError)
    assert "dit/model.safetensors" in str(err) and REV in str(err)


def test_file_changed_during_hashing_is_detected(tmp_path, monkeypatch):
    """A writer appending mid-read must never be reported as 'present'."""
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    target = materialize(tmp_path, m)
    real_read = os.read
    state = {"done": False}

    def read_then_mutate(fd, n):
        if not state["done"]:
            state["done"] = True
            with open(target, "ab") as fh:
                fh.write(b"mutated")
        return real_read(fd, n)

    monkeypatch.setattr(os, "read", read_then_mutate)
    result = mf.resolve_manifest(m, tmp_path)
    assert not result.ok
    err = result.errors[0]
    assert isinstance(err, mf.ArtifactSizeMismatchError)
    assert "changed during hashing" in str(err)


def test_unreadable_is_distinguished_from_missing(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    target = materialize(tmp_path, m)
    target.chmod(0o000)
    try:
        result = mf.resolve_manifest(m, tmp_path)
    finally:
        target.chmod(0o644)
    assert not result.ok
    err = result.errors[0]
    assert isinstance(err, mf.ArtifactUnreadableError)
    assert not isinstance(err, mf.ArtifactMissingError)
    assert "errno" in str(err)


def test_non_regular_files_never_block_hashing(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    target = tmp_path / "dit/model.safetensors"
    target.parent.mkdir(parents=True)
    os.mkfifo(target)
    result = mf.resolve_manifest(m, tmp_path)  # would hang if we opened the FIFO
    assert isinstance(result.errors[0], mf.ArtifactUnreadableError)
    assert "FIFO" in str(result.errors[0])
    # a directory at the artifact path likewise never blocks
    os.unlink(target)
    target.mkdir()
    result = mf.resolve_manifest(m, tmp_path)
    assert isinstance(result.errors[0], mf.ArtifactUnreadableError)
    assert "directory" in str(result.errors[0])


def test_symlink_escape_rejected_with_typed_path_error(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict(
        artifact_dict("dit", "dit/link.bin"))))
    outside = tmp_path.parent / (tmp_path.name + "_outside_secret.bin")
    outside.write_bytes(b"payload")
    link = tmp_path / "dit/link.bin"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    result = mf.resolve_manifest(m, tmp_path)
    assert not result.ok
    err = result.errors[0]
    assert isinstance(err, mf.ArtifactPathError)
    assert "escapes" in str(err)
    with pytest.raises(mf.ArtifactPathError):
        mf.resolve_model_paths(m, tmp_path, "dit")


def test_symlink_within_root_is_accepted(tmp_path):
    """HF caches legitimately use in-root links — containment is what matters."""
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict(
        artifact_dict("dit", "dit/link.bin"))))
    real = tmp_path / "store/blob.bin"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"payload")
    link = tmp_path / "dit/link.bin"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)
    result = mf.resolve_manifest(m, tmp_path)
    assert result.ok
    assert result.resolved["dit"] == (link,)
    assert mf.resolve_model_path(m, tmp_path, "dit") == link


def test_shards_grouped_by_role(tmp_path):
    specs = [(f"dit/shard-{i:05}.safetensors", f"part{i}".encode()) for i in range(3)]
    specs.append(("dit/index.json", b"{}"))
    parts = [artifact_dict("dit", fn, content=c) for fn, c in specs]
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict(*parts)))
    for fn, content in specs:
        target = tmp_path / fn
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    result = mf.resolve_manifest(m, tmp_path)
    assert result.ok
    assert len(result.resolved["dit"]) == 4
    paths = mf.resolve_model_paths(m, tmp_path, "dit")
    assert isinstance(paths, tuple) and len(paths) == 4
    assert all(p.is_file() for p in paths)


def test_resolve_model_path_rejects_ambiguous_role(tmp_path):
    specs = [("dit/shard-1", b"one"), ("dit/shard-2", b"two")]
    m = mf.load_manifest(write_manifest(
        tmp_path, manifest_dict(*(artifact_dict("dit", fn, content=c)
                                 for fn, c in specs))))
    for fn, content in specs:
        target = tmp_path / fn
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    with pytest.raises(mf.AmbiguousRoleError) as exc:
        mf.resolve_model_path(m, tmp_path, "dit")
    assert "resolve_model_paths" in str(exc.value)
    assert len(mf.resolve_model_paths(m, tmp_path, "dit")) == 2


def test_all_bad_artifacts_reported_together_resolution(tmp_path):
    m = mf.load_manifest(write_manifest(
        tmp_path,
        manifest_dict(artifact_dict("dit"),
                      artifact_dict("text_encoder", "te/model.bin"),
                      artifact_dict("vae", "vae/model.bin", content=b"correct")),
    ))
    materialize(tmp_path, m, role="dit", content=b"WRONG CONTENT LENGTH")  # size+hash wrong
    materialize(tmp_path, m, role="vae", content=b"correct")  # present, valid
    result = mf.resolve_manifest(m, tmp_path)  # text_encoder missing
    assert not result.ok and len(result.errors) == 2
    kinds = {type(e).__name__ for e in result.errors}
    assert kinds == {"ArtifactSizeMismatchError", "ArtifactMissingError"}
    assert result.errors[0].role == "dit" and result.errors[1].role == "text_encoder"


def test_resolve_model_path_returns_path_when_present(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    target = materialize(tmp_path, m)
    assert mf.resolve_model_path(m, tmp_path, "dit") == target


def test_resolve_model_path_raises_typed_missing(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    with pytest.raises(mf.ArtifactMissingError):
        mf.resolve_model_path(m, tmp_path, "dit")


def test_resolve_model_path_unknown_role(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    with pytest.raises(mf.ManifestError) as exc:
        mf.resolve_model_path(m, tmp_path, "lora")
    assert "lora" in str(exc.value)
    with pytest.raises(mf.ManifestError):
        mf.resolve_model_paths(m, tmp_path, "lora")


def test_resolution_never_touches_network(tmp_path, monkeypatch):
    block_network(monkeypatch)
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict(
        artifact_dict("dit"), artifact_dict("vae", content=b"other"))))
    materialize(tmp_path, m, role="vae", content=b"other")
    result = mf.resolve_manifest(m, tmp_path)
    assert not result.ok and len(result.errors) == 1  # dit missing, still no network


# --- dry-run status dict ----------------------------------------------------------


def test_status_unconfigured_is_deterministic_dict():
    status = mf.manifest_status(None, None)
    assert status["status"] == "unconfigured"
    assert json.dumps(status, sort_keys=True) == json.dumps(
        mf.manifest_status(None, None), sort_keys=True)


def test_status_reports_missing_manifest_file(tmp_path):
    status = mf.manifest_status(tmp_path / "manifest.json", tmp_path)
    assert status["status"] == "manifest_file_missing"


def test_status_reports_per_artifact_state(tmp_path):
    m = mf.load_manifest(write_manifest(
        tmp_path,
        manifest_dict(artifact_dict("dit"),
                      artifact_dict("vae", "vae/model.bin")),
    ))
    materialize(tmp_path, m, role="vae")
    status = mf.manifest_status(tmp_path / "manifest.json", tmp_path)
    assert status["status"] == "incomplete"
    assert status["schema_version"] == mf.MANIFEST_SCHEMA_VERSION
    by_role = {a["role"]: a["status"] for a in status["artifacts"]}
    assert by_role == {"dit": "missing", "vae": "present"}
    roles = [a["role"] for a in status["artifacts"]]
    assert roles == sorted(roles)  # deterministic ordering
    text = json.dumps(status, sort_keys=True)
    assert json.loads(text) == status


def test_status_reports_path_escape_state(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict(
        artifact_dict("dit", "dit/link.bin"))))
    outside = tmp_path.parent / (tmp_path.name + "_outside.bin")
    outside.write_bytes(b"payload")
    link = tmp_path / "dit/link.bin"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    status = mf.manifest_status(tmp_path / "manifest.json", tmp_path)
    assert status["status"] == "incomplete"
    assert status["artifacts"][0]["status"] == "path_escape"


def test_status_ok_when_everything_present(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    materialize(tmp_path, m)
    assert mf.manifest_status(tmp_path / "manifest.json", tmp_path)["status"] == "ok"


def test_manifest_import_pulls_in_no_heavy_modules():
    import qwen_engine.manifest  # noqa: F401
    assert not set(HEAVY_MODULES) & set(sys.modules)


# ---------------------------------------------------------------------------
# Second senior-review pass: Path.resolve failure modes stay structured
# (3.11 may raise RuntimeError on symlink loops / OSError on permissions;
# that behaviour is the WHY of this guard, not an observed 3.11 run.)
# ---------------------------------------------------------------------------


def test_symlink_loop_artifact_is_structured_unreadable(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    loop = root / "dit" / "loop.bin"
    loop.parent.mkdir(parents=True)
    os.symlink("loop.bin", loop)  # resolves to itself on any interpreter
    man = mf.manifest_from_dict(manifest_dict(artifact_dict(
        filename="dit/loop.bin")))
    report = mf.resolve_manifest(man, root)  # must not raise
    assert not report.ok
    assert len(report.statuses) == 1
    st = report.statuses[0]
    assert st.state == mf.STATE_UNREADABLE
    assert isinstance(st.error, mf.ArtifactUnreadableError)
    # dry-run stays structured too:
    (root / "manifest.json").write_text(json.dumps(man.to_dict()))
    status = mf.manifest_status(str(root / "manifest.json"), str(root))
    assert status["status"] == "incomplete"
    assert status["artifacts"][0]["status"] == mf.STATE_UNREADABLE


def test_injected_resolve_runtimeerror_artifact_is_unreadable(tmp_path,
                                                             monkeypatch):
    """CPython 3.11-style RuntimeError from Path.resolve on a symlink loop,
    injected — translated to the typed unreadable channel, never raw."""
    root = tmp_path / "models"
    (root / "dit").mkdir(parents=True)
    (root / "dit" / "model.bin").write_bytes(b"payload")
    man = mf.manifest_from_dict(manifest_dict(artifact_dict(
        filename="dit/model.bin")))
    real_resolve = mf.Path.resolve

    def fake_resolve(self):
        if self.name == "model.bin":
            raise RuntimeError("Symlink loop from 'model.bin'")
        return real_resolve(self)

    monkeypatch.setattr(mf.Path, "resolve", fake_resolve)
    report = mf.resolve_manifest(man, root)
    assert not report.ok
    st = report.statuses[0]
    assert st.state == mf.STATE_UNREADABLE
    assert isinstance(st.error, mf.ArtifactUnreadableError)
    assert "RuntimeError" in str(st.error)


def test_injected_root_resolve_oserror_is_structured_everywhere(tmp_path,
                                                               monkeypatch):
    """A permission-style OSError canonicalizing the ROOT reports every
    artifact unreadable through the report — resolve_manifest and
    manifest_status never raise; resolve_model_paths raises the typed
    ArtifactUnreadableError."""
    root = tmp_path / "models"
    root.mkdir()
    (root / "dit").mkdir()
    (root / "dit" / "model.bin").write_bytes(b"payload")
    man = mf.manifest_from_dict(manifest_dict(artifact_dict(
        filename="dit/model.bin")))
    real_resolve = mf.Path.resolve

    def fake_resolve(self):
        if self == root:
            raise OSError(13, "Permission denied", str(root))
        return real_resolve(self)

    monkeypatch.setattr(mf.Path, "resolve", fake_resolve)
    report = mf.resolve_manifest(man, root)
    assert not report.ok and not report.resolved
    assert all(s.state == mf.STATE_UNREADABLE for s in report.statuses)
    status = mf.manifest_status(None, None)  # unrelated smoke: still typed
    assert status["status"] == "unconfigured"
    with pytest.raises(mf.ArtifactUnreadableError):
        mf.resolve_model_paths(man, root, "dit")
