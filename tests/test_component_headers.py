"""Bounded tensor-header inspection; no ML or real checkpoints needed."""
from __future__ import annotations

import importlib.util
import json
import os
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location("component_cpu_check", ROOT / "tools/component_cpu_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(tmp_path, header=None, *, raw_header=None, body=b"x" * 12):
    if header is None:
        header = {"weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}}
    raw = json.dumps(header).encode() if raw_header is None else raw_header
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + body)
    return path


def test_header_only_no_tensor_body_read(checker, tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    reads = []
    original = checker.os.read
    def read(fd, size):
        result = original(fd, size)
        reads.append(len(result))
        return result
    monkeypatch.setattr(checker.os, "read", read)
    result = checker.read_checkpoint_header(path, path.stat().st_size)
    assert result == {"weight": ((2, 3), "BF16")}
    assert sum(reads) == path.stat().st_size - 12


@pytest.mark.parametrize("size", [None, True, 7, -1, "100"])
def test_invalid_size_before_open(checker, tmp_path, size, monkeypatch):
    monkeypatch.setattr(checker.os, "open", lambda *a: pytest.fail("invalid size opened file"))
    with pytest.raises(ValueError, match="bound"):
        checker.read_checkpoint_header(tmp_path / "missing", size)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "wrong-size", "short-prefix", "header-cap"])
def test_nonregular_wrong_extent_and_header_bound_refuse(checker, tmp_path, kind, monkeypatch):
    path = checkpoint(tmp_path)
    size = path.stat().st_size
    if kind == "symlink":
        peer = path.with_name("peer")
        path.rename(peer)
        path.symlink_to(peer)
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif kind == "directory":
        path.unlink()
        path.mkdir()
    elif kind == "wrong-size":
        size += 1
    elif kind == "short-prefix":
        path.write_bytes(b"short")
    else:
        monkeypatch.setattr(checker, "MAX_HEADER_BYTES", 2)
    with pytest.raises((ValueError, OSError)):
        checker.read_checkpoint_header(path, size)


@pytest.mark.parametrize("header", [None, [], {}, {"__metadata__": {}},
    {"__metadata__": {"format": 1}}, {"w": []}, {"w": {}},
    {"w": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12], "extra": 1}},
    {"w": {"dtype": "BAD", "shape": [2, 3], "data_offsets": [0, 12]}},
    {"w": {"dtype": [], "shape": [2, 3], "data_offsets": [0, 12]}},
    {"w": {"dtype": "BF16", "shape": [True, 6], "data_offsets": [0, 12]}},
    {"w": {"dtype": "BF16", "shape": [0, 6], "data_offsets": [0, 12]}},
    {"w": {"dtype": "BF16", "shape": [1] * 9, "data_offsets": [0, 12]}},
    {"w": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [False, 12]}},
    {"w": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 10]}},
    {"w": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [1, 13]}},
    {"w": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [1, 12]}},
])
def test_bad_header_metadata_never_accepted(checker, tmp_path, header):
    raw = json.dumps(header).encode()
    path = checkpoint(tmp_path, raw_header=raw)
    with pytest.raises(ValueError):
        checker.read_checkpoint_header(path, path.stat().st_size)


@pytest.mark.parametrize("raw", [b'{"w":1,"w":2}', b'{"x":NaN}', b"[" * 2000 + b"0" + b"]" * 2000, b"invalid", b"\xff\xff"])
def test_duplicate_nonfinite_deep_invalid_json_refuse(checker, tmp_path, raw):
    path = checkpoint(tmp_path, raw_header=raw)
    with pytest.raises(ValueError):
        checker.read_checkpoint_header(path, path.stat().st_size)


@pytest.mark.parametrize("kind", ["gap", "overlap", "trailing"])
def test_spans_must_cover_body_without_overlap_or_gaps(checker, tmp_path, kind):
    header = {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
              "b": {"dtype": "BF16", "shape": [4], "data_offsets": [4, 12]}}
    if kind == "gap":
        header["b"] = {"dtype": "BF16", "shape": [3], "data_offsets": [6, 12]}
    elif kind == "overlap":
        header["b"]["data_offsets"] = [2, 10]
    else:
        header["b"] = {"dtype": "BF16", "shape": [3], "data_offsets": [4, 10]}
    path = checkpoint(tmp_path, header)
    with pytest.raises(ValueError, match="spans|body extent"):
        checker.read_checkpoint_header(path, path.stat().st_size)


def test_observed_path_replacement_refused_and_descriptor_closed(checker, tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    size = path.stat().st_size
    original, observed = checker.os.read, []
    def mutate(fd, size):
        result = original(fd, size)
        if not observed:
            observed.append(fd)
            path.rename(path.with_name("retained"))
            checkpoint(tmp_path)
        return result
    monkeypatch.setattr(checker.os, "read", mutate)
    with pytest.raises(ValueError, match="changed"):
        checker.read_checkpoint_header(path, size)
    with pytest.raises(OSError):
        os.fstat(observed[0])
