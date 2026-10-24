"""CPU-guarded tests for the Q8 GGUF sidecar contract (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from qwen_engine import q8_artifact as q8
from qwen_engine.local_bundle import FULL_REFERENCE_REVISION
from qwen_engine.manifest import ManifestError

GGUF_V3 = b"GGUF" + (3).to_bytes(4, "little")
GGUF_V2 = b"GGUF" + (2).to_bytes(4, "little")
NAME = "qwen-image-q8_0.gguf"
DEFAULT_CONTENT = GGUF_V3 + b"payload"


def _same_size_tamper(content: bytes) -> bytes:
    """Same length as ``content``, different bytes, still a v3 GGUF header."""
    tail = bytearray(content[8:]) or bytearray(b"x")
    tail[0] ^= 0x01
    return GGUF_V3 + bytes(tail)


def _sidecar(**over) -> dict:
    doc = {
        "schema": "qwen_engine.q8_artifact.v1",
        "repo": "unsloth/Qwen-Image-2.1-GGUF",
        "revision": "a" * 40,
        "filename": NAME,
        "sha256": "0" * 64,
        "bytes": 1,
        "source_model_revision": FULL_REFERENCE_REVISION,
        "quant_format": "gguf-q8_0",
    }
    doc.update(over)
    return doc


def _write_gguf(dirpath: Path, name: str = NAME,
                content: bytes = DEFAULT_CONTENT) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / name
    path.write_bytes(content)
    return path


def _fix(sidecar_dir: Path, data_dir: Path | None = None, *,
         content: bytes = DEFAULT_CONTENT, **over) -> Path:
    """Write a tiny GGUF into the data dir and a matching sidecar into
    ``sidecar_dir``; return the sidecar path."""
    data_root = data_dir if data_dir is not None else sidecar_dir
    _write_gguf(data_root, content=content)
    doc = _sidecar(sha256=hashlib.sha256(content).hexdigest(),
                   bytes=len(content), **over)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar = sidecar_dir / "q8_artifact.json"
    sidecar.write_text(json.dumps(doc))
    return sidecar


def _resolve(sidecar: Path, models_dir: Path | None = None, **kw):
    return q8.resolve_q8_artifact(sidecar, models_dir, **kw)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_canonical_path_and_identity(tmp_path):
    content = GGUF_V3 + b"tiny weights"
    sidecar = _fix(tmp_path / "data", content=content)
    art = _resolve(sidecar)
    assert art.path == (tmp_path / "data" / NAME).resolve()
    assert art.path.is_absolute()
    assert art.spec.sha256 == hashlib.sha256(content).hexdigest()
    assert art.spec.bytes == len(content)
    assert art.repo == "unsloth/Qwen-Image-2.1-GGUF"
    assert art.revision == "a" * 40
    assert art.source_model_revision == FULL_REFERENCE_REVISION
    assert art.quant_format == "gguf-q8_0"
    assert art.filename == NAME
    # fingerprint: deterministic, 64 hex, independent of ambient paths
    assert art.identity_sha256 == q8.q8_identity_sha256(art.spec)
    assert re.fullmatch(r"[0-9a-f]{64}", art.identity_sha256)


def test_explicit_models_dir_overrides_sidecar_parent(tmp_path):
    sidecar = _fix(tmp_path / "meta", data_dir=tmp_path / "data")
    art = _resolve(sidecar, tmp_path / "data")
    assert art.path.parent == (tmp_path / "data").resolve()
    # sidecar parent alone has no GGUF -> typed missing, not a fallback guess
    with pytest.raises(q8.Q8FileMissingError):
        _resolve(sidecar)


def test_identity_fingerprint_covers_fields_not_paths(tmp_path):
    art_a = _resolve(_fix(tmp_path / "a"))
    art_b = _resolve(_fix(tmp_path / "b"))
    assert art_a.path != art_b.path
    assert art_a.identity_sha256 == art_b.identity_sha256
    # any manifest field change drifts the fingerprint
    doc = json.loads((tmp_path / "a" / "q8_artifact.json").read_text())
    doc["sha256"] = "f" * 64
    assert q8.q8_identity_sha256(
        q8.q8_artifact_spec_from_dict(doc)) != art_a.identity_sha256


def test_to_dict_excludes_path_and_includes_identity():
    spec = q8.Q8ArtifactSpec(**_sidecar())
    art = q8.Q8Artifact(spec=spec, path=Path("/x/y.gguf"),
                        identity_sha256="e" * 64)
    d = art.to_dict()
    assert "path" not in d and "identity_sha256" in d
    assert d["filename"] == NAME


def test_gguf_version2_accepted(tmp_path):
    _fix(tmp_path / "d", content=GGUF_V2 + b"x")
    art = _resolve(tmp_path / "d" / "q8_artifact.json")
    assert art.spec.bytes == len(GGUF_V2 + b"x")


def test_nested_filename_under_data_root(tmp_path):
    data = tmp_path / "data"
    _write_gguf(data / "sub", content=GGUF_V3 + b"nested")
    content = (data / "sub" / NAME).read_bytes()
    doc = _sidecar(filename="sub/" + NAME,
                   sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    sidecar = data / "q8_artifact.json"
    sidecar.write_text(json.dumps(doc))
    art = _resolve(sidecar, data)
    assert art.path == (data / "sub" / NAME).resolve()


# ---------------------------------------------------------------------------
# Strict sidecar JSON
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutate,frag", [
    (lambda d: d.pop("schema"), "schema"),
    (lambda d: d.pop("quant_format"), "quant_format"),
    (lambda d: d.pop("bytes"), "bytes"),
    (lambda d: d.update({"schema": "qwen_engine.q8_artifact.v2"}), "schema"),
    (lambda d: d.update({"quant_format": "gguf-q4_0"}), "quant_format"),
    (lambda d: d.update({"extra": 1}), "unknown"),
    (lambda d: d.update({"repo": "noslash"}), "repo"),
    (lambda d: d.update({"repo": "org/name/extra"}), "repo"),
    (lambda d: d.update({"repo": "org/../evil"}), "repo"),
    (lambda d: d.update({"revision": "main"}), "revision"),
    (lambda d: d.update({"revision": "A" * 40}), "revision"),
    (lambda d: d.update({"sha256": "ABC"}), "sha256"),
    (lambda d: d.update({"sha256": "A" * 64}), "sha256"),
    (lambda d: d.update({"bytes": 0}), "bytes"),
    (lambda d: d.update({"bytes": -5}), "bytes"),
    (lambda d: d.update({"bytes": True}), "bytes"),
    (lambda d: d.update({"bytes": "123"}), "bytes"),
    (lambda d: d.update({"filename": "model.bin"}), r"\.gguf"),
    (lambda d: d.update({"filename": "../escape.gguf"}), "parent-directory"),
    (lambda d: d.update({"filename": "/abs/model.gguf"}), "absolute"),
    (lambda d: d.update({"filename": "sub\\model.gguf"}), "backslash"),
    (lambda d: d.update({"filename": "C:/evil.gguf"}), "drive"),
    (lambda d: d.update({"filename": "./model.gguf"}), r"dot path"),
    (lambda d: d.update({"filename": "a//model.gguf"}), "empty path"),
    (lambda d: d.update({"filename": ""}), "non-empty"),
], ids=lambda v: str(v)[:24])
def test_bad_sidecar_field_refused(tmp_path, mutate, frag):
    doc = _sidecar()
    mutate(doc)
    with pytest.raises(q8.Q8SidecarFormatError) as exc:
        q8.q8_artifact_spec_from_dict(doc)
    assert re.search(frag, str(exc.value))


def test_direct_construction_validates_too():
    with pytest.raises(q8.Q8SidecarFormatError):
        q8.Q8ArtifactSpec(**_sidecar(quant_format="gguf-q8_0 "))
    with pytest.raises(q8.Q8SidecarFormatError):
        q8.Q8Artifact(spec=q8.Q8ArtifactSpec(**_sidecar()),
                      path=Path("relative.gguf"), identity_sha256="0" * 64)


def test_duplicate_json_key_rejected(tmp_path):
    d = tmp_path / "d"
    _fix(d)
    sidecar = d / "q8_artifact.json"
    sidecar.write_text('{"schema": "qwen_engine.q8_artifact.v1", '
                       '"schema": "qwen_engine.q8_artifact.v1"}')
    with pytest.raises(q8.Q8SidecarFormatError, match="duplicate"):
        _resolve(sidecar)


def test_nonfinite_constant_rejected(tmp_path):
    d = tmp_path / "d"
    _fix(d)
    sidecar = d / "q8_artifact.json"
    doc = json.loads(sidecar.read_text())
    sidecar.write_text(sidecar.read_text().replace(
        json.dumps(doc["bytes"]), "NaN"))
    with pytest.raises(q8.Q8SidecarFormatError):
        _resolve(sidecar)


def test_sidecar_size_cap(tmp_path):
    d = tmp_path / "d"
    _fix(d)
    sidecar = d / "q8_artifact.json"
    doc = json.loads(sidecar.read_text())
    doc["repo"] = "o" * (1 << 20)
    sidecar.write_text(json.dumps(doc))
    with pytest.raises(q8.Q8SidecarFormatError):
        _resolve(sidecar)


def test_sidecar_not_json(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    sidecar = d / "q8_artifact.json"
    sidecar.write_text("{not json")
    with pytest.raises(q8.Q8SidecarFormatError, match="not valid JSON"):
        _resolve(sidecar)


def test_sidecar_missing_file_typed(tmp_path):
    with pytest.raises(q8.Q8SidecarFormatError, match="not readable"):
        _resolve(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Source-revision pin
# ---------------------------------------------------------------------------


def test_source_revision_mismatch_refused(tmp_path):
    sidecar = _fix(tmp_path / "d", source_model_revision="b" * 40)
    with pytest.raises(q8.Q8SidecarFormatError, match="source_model_revision"):
        _resolve(sidecar)


def test_expected_source_revision_override(tmp_path):
    sidecar = _fix(tmp_path / "d", source_model_revision="b" * 40)
    art = _resolve(sidecar, tmp_path / "d", expected_source_revision="b" * 40)
    assert art.source_model_revision == "b" * 40


def test_invalid_expected_source_revision_refused(tmp_path):
    sidecar = _fix(tmp_path / "d")
    with pytest.raises(q8.Q8SidecarFormatError, match="expected_source_revision"):
        _resolve(sidecar, tmp_path / "d", expected_source_revision="main")


# ---------------------------------------------------------------------------
# File verification: escape, symlinks, corruption
# ---------------------------------------------------------------------------


def test_filename_escape_refused_at_parse(tmp_path):
    sidecar = _fix(tmp_path / "d", filename="../outside.gguf")
    with pytest.raises(q8.Q8SidecarFormatError, match="parent-directory"):
        _resolve(sidecar)


def test_symlink_escape_refused(tmp_path):
    outside = _write_gguf(tmp_path / "outside")
    data = tmp_path / "data"
    data.mkdir()
    (data / NAME).symlink_to(outside)
    content = outside.read_bytes()
    doc = _sidecar(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    sidecar = data / "q8_artifact.json"
    sidecar.write_text(json.dumps(doc))
    with pytest.raises(q8.Q8FilePathError):
        _resolve(sidecar, data)


def test_symlink_inside_root_allowed(tmp_path):
    data = tmp_path / "data"
    real = _write_gguf(data, name="actual.gguf")
    link = data / NAME
    link.symlink_to(Path("actual.gguf"))
    content = real.read_bytes()
    doc = _sidecar(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    (data / "q8_artifact.json").write_text(json.dumps(doc))
    art = _resolve(data / "q8_artifact.json", data)
    assert art.path == real.resolve()


def test_directory_swap_refused(tmp_path):
    d = tmp_path / "d"
    _fix(d)
    (d / NAME).unlink()
    (d / NAME).mkdir()
    with pytest.raises(q8.Q8FileUnreadableError, match="not a regular file"):
        _resolve(d / "q8_artifact.json")


def test_size_mismatch_refused(tmp_path):
    d = tmp_path / "d"
    _fix(d)
    (d / NAME).write_bytes(GGUF_V3 + b"longer payload")
    with pytest.raises(q8.Q8FileSizeMismatchError):
        _resolve(d / "q8_artifact.json")


def test_hash_mismatch_refused(tmp_path):
    d = tmp_path / "d"
    _fix(d)
    (d / NAME).write_bytes(_same_size_tamper(DEFAULT_CONTENT))
    with pytest.raises(q8.Q8FileHashMismatchError):
        _resolve(d / "q8_artifact.json")


def test_missing_file_refused(tmp_path):
    d = tmp_path / "d"
    _fix(d)
    (d / NAME).unlink()
    with pytest.raises(q8.Q8FileMissingError):
        _resolve(d / "q8_artifact.json")


@pytest.mark.parametrize("content,bad_frag", [
    (b"NOTG" + (3).to_bytes(4, "little") + b"x", "magic"),
    (GGUF_V3[:4] + (9).to_bytes(4, "little") + b"x", "version 9"),
    (GGUF_V3[:4] + (1).to_bytes(4, "little") + b"x", "version 1"),
], ids=["magic", "v9", "v1"])
def test_bad_gguf_header_refused(tmp_path, content, bad_frag):
    sidecar = _fix(tmp_path / "d", content=content)
    with pytest.raises(q8.Q8FileUnreadableError, match=bad_frag):
        _resolve(sidecar)


def test_size_gate_before_header(tmp_path):
    # wrong size AND wrong magic: the size channel reports first
    sidecar = _fix(tmp_path / "d", content=GGUF_V3 + b"orig")
    (tmp_path / "d" / NAME).write_bytes(b"XXXX" + (3).to_bytes(4, "little"))
    with pytest.raises(q8.Q8FileSizeMismatchError):
        _resolve(sidecar)


def test_truncated_gguf_header_size_gate(tmp_path):
    sidecar = _fix(tmp_path / "d", content=GGUF_V3 + b"payload")
    (tmp_path / "d" / NAME).write_bytes(b"GG")
    with pytest.raises(q8.Q8FileSizeMismatchError):
        _resolve(sidecar)


# ---------------------------------------------------------------------------
# Typed-error contract + no side effects
# ---------------------------------------------------------------------------


def test_all_errors_are_manifest_errors():
    for cls in (q8.Q8SidecarFormatError, q8.Q8FileError, q8.Q8FileMissingError,
                q8.Q8FilePathError, q8.Q8FileSizeMismatchError,
                q8.Q8FileHashMismatchError, q8.Q8FileUnreadableError):
        assert issubclass(cls, ManifestError)


def test_error_names_sidecar_identity(tmp_path):
    d = tmp_path / "d"
    _fix(d)
    (d / NAME).write_bytes(_same_size_tamper(DEFAULT_CONTENT))
    with pytest.raises(q8.Q8FileHashMismatchError) as exc:
        _resolve(d / "q8_artifact.json")
    msg = str(exc.value)
    assert "unsloth/Qwen-Image-2.1-GGUF" in msg and NAME in msg
    assert exc.value.repo == "unsloth/Qwen-Image-2.1-GGUF"
    assert exc.value.filename == NAME


def test_resolution_has_no_side_effects(tmp_path):
    d = tmp_path / "d"
    sidecar = _fix(d)
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    mtimes = {p.name: p.stat().st_mtime_ns for p in d.iterdir()}
    _resolve(sidecar)
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before
    assert {p.name: p.stat().st_mtime_ns for p in d.iterdir()} == mtimes


def test_module_imports_are_ml_free():
    import sys
    import qwen_engine.q8_artifact as mod
    src = Path(mod.__file__).read_text()
    for forbidden in ("import torch", "import transformers", "import diffusers"):
        assert forbidden not in src
    assert not {"torch", "diffusers", "transformers"} & set(sys.modules)
