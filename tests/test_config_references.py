"""Reference-id config fields and CLI store ingestion (design §2.1/§2.2 patch).

Pure stdlib + fake store: no PIL, no torch, no network. The CLI ingestion is
exercised against a stub ``qwen_engine.reference_store`` module (the real store
is Q2's and covered by its own tests); these tests pin OUR side: field
validation, verbatim copy semantics, id-only config, dry-run echo.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from qwen_engine import cli
from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.errors import EngineError

ID_A = "a" * 64
ID_B = "b" * 64


# --- config: reference fields -------------------------------------------------


def test_defaults_are_empty_text_to_image():
    config = GenerationConfig()
    assert config.references == () and config.references_dir is None
    assert config.to_dict()["references"] == ()
    round_tripped = GenerationConfig.from_json(config.to_json())
    assert round_tripped == config


def test_valid_references_round_trip_like_job_envelopes(tmp_path):
    config = GenerationConfig(prompt="edit", references=(ID_A, ID_B),
                              references_dir=str(tmp_path))
    assert config.references == (ID_A, ID_B)
    envelope = GenerationConfig.from_json(json.dumps(config.to_dict()))
    assert envelope.references == (ID_A, ID_B) and isinstance(envelope.references, tuple)
    assert envelope.references_dir == str(tmp_path)


def test_duplicate_reference_ids_are_legal(tmp_path):
    # The same image may fill two <imageN> slots (design §2.1).
    config = GenerationConfig(references=(ID_A, ID_A), references_dir=str(tmp_path))
    assert config.references == (ID_A, ID_A)


@pytest.mark.parametrize("references", [
    ("A" * 64,),                      # uppercase is not canonical
    ("a" * 63,), ("a" * 65,),         # not exactly 64 hex
    ("z" * 64,), ("a" * 63 + "g",),   # not hex
    (ID_A, 7), (ID_A, None),          # not strings
    ID_A,                              # a bare id is not a sequence of ids
    tuple(f"{i:064x}" for i in range(11)),  # above REF_MAX
])
def test_invalid_reference_ids_refuse(references):
    with pytest.raises(ConfigError) as excinfo:
        GenerationConfig(references=references, references_dir="/store")
    assert "references" in excinfo.value.fields


def test_references_without_store_directory_refuse():
    with pytest.raises(ConfigError) as excinfo:
        GenerationConfig(references=(ID_A,))
    assert excinfo.value.fields == ("references_dir",)


def test_store_directory_alone_is_legal(tmp_path):
    assert GenerationConfig(references_dir=str(tmp_path)).references_dir == str(tmp_path)


def test_references_dir_follows_path_field_rules(tmp_path):
    file_target = tmp_path / "not-a-dir"
    file_target.write_bytes(b"x")
    with pytest.raises(ConfigError) as excinfo:
        GenerationConfig(references=(ID_A,), references_dir=str(file_target))
    assert excinfo.value.fields == ("references_dir",)
    with pytest.raises(ConfigError):
        GenerationConfig(references=(ID_A,), references_dir="  ")
    assert GenerationConfig(references=(ID_A,),
                            references_dir=tmp_path).references_dir == str(tmp_path)


# --- cli: parser, ingestion, dry-run echo -------------------------------------


def _fake_store(monkeypatch, tmp_path, *, max_bytes=1 << 20, refuse=None):
    """Stub reference_store: id = sha256, verbatim blob copy, typed refusals."""
    calls = []

    class StoreError(EngineError):
        pass

    def put_bytes(root, raw):
        if refuse is not None:
            raise refuse if isinstance(refuse, BaseException) else StoreError(str(refuse))
        reference_id = hashlib.sha256(raw).hexdigest()
        blob = Path(root) / (reference_id + ".img")
        if not blob.exists():
            blob.write_bytes(raw)
        calls.append((str(root), reference_id))
        return reference_id

    fake = types.SimpleNamespace(MAX_REFERENCE_BYTES=max_bytes, put_bytes=put_bytes)
    # Bind BOTH the module slot and the package attribute: ``from qwen_engine
    # import reference_store`` inside the CLI resolves the package attribute
    # once the real module has been imported by any other collector.
    import qwen_engine
    monkeypatch.setitem(sys.modules, "qwen_engine.reference_store", fake)
    monkeypatch.setattr(qwen_engine, "reference_store", fake, raising=False)
    return calls


def test_parser_accepts_repeated_reference_flags():
    args = cli.build_parser().parse_args(
        ["--reference", "one.png", "--reference", "two.png",
         "--references-dir", "store", "--dry-run"])
    assert args.references == ["one.png", "two.png"]
    assert args.references_dir == Path("store")


def _write_inputs(tmp_path):
    first, second = tmp_path / "one.png", tmp_path / "two.png"
    first.write_bytes(b"first-image-bytes")
    second.write_bytes(b"second-image-bytes")
    return first, second


def test_load_config_ingests_files_into_content_ids(tmp_path, monkeypatch):
    first, second = _write_inputs(tmp_path)
    store = tmp_path / "store"
    calls = _fake_store(monkeypatch, tmp_path)
    args = cli.build_parser().parse_args(
        ["--prompt", "edit", "--output-dir", str(tmp_path / "out"),
         "--references-dir", str(store),
         "--reference", str(first), "--reference", str(second),
         "--reference", str(first)])
    config = cli._load_config(args)
    ids = tuple(hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (first, second, first))
    assert config.references == ids
    assert config.references_dir == str(store)
    # Verbatim copies under the content id; duplicates stored once, passed thrice.
    for reference_id in ids:
        blob = store / (reference_id + ".img")
        assert blob.read_bytes() in (b"first-image-bytes", b"second-image-bytes")
    assert calls[-1][1] == ids[-1]


def test_store_defaults_under_cache_dir(tmp_path, monkeypatch):
    first, _ = _write_inputs(tmp_path)
    _fake_store(monkeypatch, tmp_path)
    args = cli.build_parser().parse_args(
        ["--prompt", "edit", "--output-dir", str(tmp_path / "out"),
         "--cache-dir", str(tmp_path / "cache"), "--reference", str(first)])
    config = cli._load_config(args)
    assert config.references_dir == str(tmp_path / "cache" / "references")


def test_reference_without_any_store_target_refuses(tmp_path):
    first, _ = _write_inputs(tmp_path)
    args = cli.build_parser().parse_args(
        ["--prompt", "edit", "--output-dir", str(tmp_path / "out"),
         "--reference", str(first)])
    with pytest.raises(ConfigError) as excinfo:
        cli._load_config(args)
    assert excinfo.value.fields == ("references_dir",)


def test_oversized_reference_file_refuses(tmp_path, monkeypatch):
    first, _ = _write_inputs(tmp_path)
    _fake_store(monkeypatch, tmp_path, max_bytes=4)
    args = cli.build_parser().parse_args(
        ["--prompt", "edit", "--output-dir", str(tmp_path / "out"),
         "--references-dir", str(tmp_path / "store"), "--reference", str(first)])
    with pytest.raises(ConfigError) as excinfo:
        cli._load_config(args)
    assert excinfo.value.fields == ("references",)


def test_missing_reference_file_refuses(tmp_path, monkeypatch):
    _fake_store(monkeypatch, tmp_path)
    args = cli.build_parser().parse_args(
        ["--prompt", "edit", "--output-dir", str(tmp_path / "out"),
         "--references-dir", str(tmp_path / "store"),
         "--reference", str(tmp_path / "gone.png")])
    with pytest.raises(ConfigError) as excinfo:
        cli._load_config(args)
    assert excinfo.value.fields == ("references",)


def test_store_refusal_becomes_exit_two(tmp_path, monkeypatch):
    first, _ = _write_inputs(tmp_path)
    _fake_store(monkeypatch, tmp_path, refuse="reference store is full")
    args = cli.build_parser().parse_args(
        ["--prompt", "edit", "--output-dir", str(tmp_path / "out"),
         "--references-dir", str(tmp_path / "store"), "--reference", str(first)])
    with pytest.raises(ConfigError) as excinfo:
        cli._load_config(args)
    assert excinfo.value.fields == ("references",)


def test_dry_run_echoes_ids_sizes_order(tmp_path, monkeypatch, capsys):
    first, second = _write_inputs(tmp_path)
    _fake_store(monkeypatch, tmp_path)
    config_file = tmp_path / "cfg.json"
    config_file.write_text(json.dumps(
        {"prompt": "edit", "output_dir": str(tmp_path / "out")}))
    code = cli.main(["--config", str(config_file), "--dry-run",
                     "--references-dir", str(tmp_path / "store"),
                     "--reference", str(first), "--reference", str(second)])
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["references"] == [
        {"ordinal": 1, "id": hashlib.sha256(b"first-image-bytes").hexdigest(),
         "bytes": first.stat().st_size},
        {"ordinal": 2, "id": hashlib.sha256(b"second-image-bytes").hexdigest(),
         "bytes": second.stat().st_size}]
    assert document["config"]["references"] == [entry["id"] for entry in
                                                document["references"]]


def test_dry_run_shape_unchanged_without_references(tmp_path, capsys):
    config_file = tmp_path / "cfg.json"
    config_file.write_text(json.dumps(
        {"prompt": "plain", "output_dir": str(tmp_path / "out")}))
    assert cli.main(["--config", str(config_file), "--dry-run"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert "references" not in document
    assert document["config"]["references"] == []
