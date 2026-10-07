"""Task 1.2 acceptance: versioned model-manifest schema + no-download resolver (R3).

The resolver verifies existence / byte size / sha256 and *never downloads*:
tests block sockets and still expect clean structured results. Every typed
error names repo/revision/file. No network, no GPU, no weights (R7).
"""
from __future__ import annotations

import hashlib
import json
import socket
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


def materialize(tmp_path, manifest, role="dit", content=b"payload"):
    """Write the real artifact file for one role under models_dir."""
    art = next(a for a in manifest.artifacts if a.role == role)
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


def test_unknown_schema_version_names_field(tmp_path):
    path = write_manifest(tmp_path, manifest_dict(schema_version=99))
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(path)
    assert "schema_version" in str(exc.value)


def test_malformed_manifest_json_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{nope")
    with pytest.raises(mf.ManifestFormatError):
        mf.load_manifest(path)


@pytest.mark.parametrize("field,bad_artifact", [
    ("sha256", lambda: {**artifact_dict(), "sha256": "nothex"}),
    ("bytes", lambda: {**artifact_dict(), "bytes": -5}),
    ("repo", lambda: artifact_dict(repo="")),
    ("revision", lambda: artifact_dict(revision="")),
])
def test_invalid_artifact_fields_name_the_field(tmp_path, field, bad_artifact):
    path = write_manifest(tmp_path, manifest_dict(bad_artifact()))
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(path)
    assert field in str(exc.value)


def test_artifacts_must_be_a_list(tmp_path):
    data = manifest_dict()
    data["artifacts"] = "not-a-list"
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(write_manifest(tmp_path, data))
    assert "artifacts" in str(exc.value)


def test_duplicate_roles_rejected(tmp_path):
    path = write_manifest(tmp_path,
                          manifest_dict(artifact_dict("dit", "a/x"),
                                        artifact_dict("dit", "b/y")))
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(path)
    assert "role" in str(exc.value)


@pytest.mark.parametrize("filename", ["../escape.bin", "/abs/model.bin"])
def test_unsafe_filenames_rejected(tmp_path, filename):
    path = write_manifest(tmp_path, manifest_dict(artifact_dict(filename=filename)))
    with pytest.raises(mf.ManifestFormatError) as exc:
        mf.load_manifest(path)
    assert "filename" in str(exc.value)


# --- resolution (verify-only, no downloads) -------------------------------------


def test_resolve_present_artifact_returns_path(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    materialize(tmp_path, m)
    result = mf.resolve_manifest(m, tmp_path)
    assert result.ok
    assert result.resolved["dit"] == tmp_path / "dit/model.safetensors"


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


def test_all_bad_artifacts_reported_together(tmp_path):
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


def test_status_ok_when_everything_present(tmp_path):
    m = mf.load_manifest(write_manifest(tmp_path, manifest_dict()))
    materialize(tmp_path, m)
    assert mf.manifest_status(tmp_path / "manifest.json", tmp_path)["status"] == "ok"


def test_manifest_import_pulls_in_no_heavy_modules():
    import qwen_engine.manifest  # noqa: F401
    assert not set(HEAVY_MODULES) & set(sys.modules)
