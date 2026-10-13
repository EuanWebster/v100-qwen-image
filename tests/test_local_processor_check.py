"""Small-file snapshot regressions, no model files/ML/network required."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import replace
from pathlib import Path

import pytest

from qwen_engine.manifest import ModelArtifact

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location("local_processor_check", ROOT / "tools/local_processor_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "snapshot"
    source.mkdir()
    destination.mkdir()
    artifacts = []
    for name, data in (("tokenizer.json", b'{"tokens":[1,2]}'), ("chat_template.jinja", b"raw template")):
        (source / name).write_bytes(data)
        artifacts.append(ModelArtifact(role="processor", repo="test/repo", revision="a" * 40,
                                       filename="processor/" + name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    return tuple(artifacts), source, destination


def test_verified_processor_only_snapshot(checker, tree):
    artifacts, source, destination = tree
    assert checker.copy_processor_files(artifacts, source, destination) == sum(a.bytes for a in artifacts)
    assert sorted(p.name for p in destination.iterdir()) == sorted(p.name for p in source.iterdir())
    assert all((source / p.name).read_bytes() == p.read_bytes() for p in destination.iterdir())


@pytest.mark.parametrize("kind", ["extra", "missing", "symlink", "directory", "fifo"])
def test_closed_flat_inventory_before_copy(checker, tree, kind):
    artifacts, source, destination = tree
    path = source / "tokenizer.json"
    if kind == "extra":
        (source / "unapproved").write_bytes(b"x")
    else:
        path.unlink()
        if kind == "symlink":
            path.symlink_to(source / "chat_template.jinja")
        elif kind == "directory":
            path.mkdir()
        elif kind == "fifo":
            os.mkfifo(path)
    with pytest.raises(ValueError, match="incomplete|unexpected"):
        checker.copy_processor_files(artifacts, source, destination)
    assert not list(destination.iterdir())


@pytest.mark.parametrize("kind", ["same-size-corruption", "truncated", "oversized"])
def test_bad_bytes_never_copied(checker, tree, kind):
    artifacts, source, destination = tree
    data = (source / "tokenizer.json").read_bytes()
    (source / "tokenizer.json").write_bytes(b"x" * len(data) if kind == "same-size-corruption" else
                                           data[:-1] if kind == "truncated" else data + b"x")
    with pytest.raises(ValueError, match="size/hash|bounded regular"):
        checker.copy_processor_files(artifacts, source, destination)
    assert not list(destination.iterdir())


@pytest.mark.parametrize("kind", ["count", "file-bytes", "total-bytes", "duplicate", "role", "nested", "lookalike"])
def test_snapshot_scope_refused_before_io(checker, tree, monkeypatch, kind):
    artifacts, source, destination = tree
    if kind == "count":
        monkeypatch.setattr(checker, "MAX_PROCESSOR_FILES", 1)
    elif kind == "file-bytes":
        monkeypatch.setattr(checker, "MAX_PROCESSOR_FILE", 1)
    elif kind == "total-bytes":
        monkeypatch.setattr(checker, "MAX_PROCESSOR_TOTAL", 1)
    elif kind == "duplicate":
        artifacts = (artifacts[0], artifacts[0])
    elif kind == "role":
        artifacts = (replace(artifacts[0], role="text_encoder"),)
    elif kind == "nested":
        artifacts = (replace(artifacts[0], filename="processor/nested/tokenizer.json"),)
    else:
        artifacts = (object(),)
    monkeypatch.setattr(checker.os, "scandir", lambda *a: pytest.fail("invalid scope accessed disk"))
    with pytest.raises(ValueError, match="scope/bounds|duplicate"):
        checker.copy_processor_files(artifacts, source, destination)


def test_existing_destination_preserved(checker, tree):
    artifacts, source, destination = tree
    (destination / "owned-by-someone-else").write_bytes(b"retain")
    with pytest.raises(ValueError, match="empty"):
        checker.copy_processor_files(artifacts, source, destination)
    assert (destination / "owned-by-someone-else").read_bytes() == b"retain"


def test_later_source_mutation_refused(checker, tree, monkeypatch):
    artifacts, source, destination = tree
    from qwen_engine import _local_io
    original = _local_io.read_bounded_file

    def mutate(path, cap):
        data = original(path, cap)
        if path.name == "tokenizer.json":
            (source / "chat_template.jinja").write_bytes(b"changed late")
        return data

    monkeypatch.setattr(_local_io, "read_bounded_file", mutate)
    with pytest.raises(ValueError, match="changed before read"):
        checker.copy_processor_files(artifacts, source, destination)
    assert not (destination / "chat_template.jinja").exists()
