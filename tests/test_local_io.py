"""Guarded local input snapshots, never downloaded models or external writes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from qwen_engine import _local_io as io


@pytest.mark.parametrize("cap", [True, 0, -1, 1.0, None, sys.maxsize, 10**100])
def test_invalid_cap_refuses_before_path_access(monkeypatch, cap):
    monkeypatch.setattr(io.os, "open", lambda *a, **k: pytest.fail("opened with invalid cap"))
    with pytest.raises(ValueError, match="cap"):
        io.read_bounded_file("does-not-exist", cap)


def test_exact_bound_and_stable_input_symlink(tmp_path):
    real = tmp_path / "input"
    real.write_bytes(b"payload")
    link = tmp_path / "link"
    link.symlink_to(real)
    assert io.read_bounded_file(link, 7) == b"payload"
    with pytest.raises(ValueError, match="bounded regular"):
        io.read_bounded_file(real, 6)


@pytest.mark.parametrize("kind", ["fifo", "directory"])
def test_nonregular_input_does_not_block(tmp_path, kind):
    path = tmp_path / "input"
    if kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    with pytest.raises(ValueError, match="bounded regular"):
        io.read_bounded_file(path, 100)


@pytest.mark.parametrize("change", ["fifo", "symlink"])
def test_final_input_swap_before_open_refused_nonblocking(tmp_path, monkeypatch, change):
    path = tmp_path / "input"
    path.write_bytes(b"payload")
    other = tmp_path / "other"
    other.write_bytes(b"other")
    original = os.open
    def open_swapped(target, flags, *a, **k):
        assert flags & os.O_NONBLOCK and flags & os.O_NOFOLLOW
        path.unlink()
        if change == "fifo":
            os.mkfifo(path)
        else:
            path.symlink_to(other)
        return original(target, flags, *a, **k)
    monkeypatch.setattr(io.os, "open", open_swapped)
    with pytest.raises((ValueError, OSError)):
        io.read_bounded_file(path, 100)


@pytest.mark.parametrize("change", ["grow", "edit", "replace", "alias"])
def test_input_changes_during_read_refused(tmp_path, monkeypatch, change):
    path = tmp_path / "input"
    path.write_bytes(b"payload")
    original = os.read
    changed = False
    def read(fd, size):
        nonlocal changed
        raw = original(fd, size)
        if raw and not changed:
            changed = True
            if change == "grow":
                with path.open("ab") as fh:
                    fh.write(b"x" * 200)
            elif change == "edit":
                before = path.stat()
                path.write_bytes(b"changed")
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            elif change == "replace":
                other = tmp_path / "other"
                other.write_bytes(b"payload")
                other.replace(path)
            else:
                other = tmp_path / "other"
                path.rename(other)
                path.symlink_to(other)
        return raw
    monkeypatch.setattr(io.os, "read", read)
    with pytest.raises(ValueError, match="changed|byte cap"):
        io.read_bounded_file(path, 100)


@pytest.mark.parametrize("error", [OSError("read failed"), KeyboardInterrupt()])
def test_input_descriptor_closed_on_read_failure(tmp_path, monkeypatch, error):
    path = tmp_path / "input"
    path.write_bytes(b"payload")
    def failed(*a):
        raise error
    monkeypatch.setattr(io.os, "read", failed)
    before = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(type(error)) as caught:
        io.read_bounded_file(path, 100)
    assert caught.value is error
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_growing_input_reads_only_cap_plus_one(tmp_path, monkeypatch):
    path = tmp_path / "input"
    path.write_bytes(b"payload")
    original = os.read
    count = 0
    def read(fd, size):
        nonlocal count
        with path.open("ab") as fh:
            fh.write(b"x" * 1000)
        raw = original(fd, size)
        count += len(raw)
        return raw
    monkeypatch.setattr(io.os, "read", read)
    with pytest.raises(ValueError, match="byte cap"):
        io.read_bounded_file(path, 100)
    assert count == 101
