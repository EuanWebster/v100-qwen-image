"""Bounded snapshot reads of explicitly chosen local input files (stdlib-only).

Stable symlinks are intentional inputs and allowed after canonicalization.
This is not an asset-root containment policy or a hostile-filesystem sandbox.
Callers translate OS/value failures to their own typed input error boundary.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

_STAT_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")


def _signature(info):
    return tuple(getattr(info, field) for field in _STAT_FIELDS)


def read_bounded_file(path: str | Path, cap: int) -> bytes:
    if type(cap) is not int or not 1 <= cap < sys.maxsize:
        raise ValueError("input cap must be a representable positive byte count")
    path = Path(path)
    canonical = path.resolve(strict=True)
    fd = os.open(canonical, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
            raise ValueError(f"input must be a bounded regular file ({cap} byte cap)")
        data = bytearray()
        while True:
            block = os.read(fd, min(65536, cap + 1 - len(data)))
            if not block:
                break
            data.extend(block)
            if len(data) > cap:
                raise ValueError("input byte cap exceeded")
        if (len(data) != before.st_size or _signature(before) != _signature(os.fstat(fd))
                or _signature(before) != _signature(canonical.stat(follow_symlinks=False))
                or path.resolve(strict=True) != canonical):
            raise ValueError("input changed during read")
        return bytes(data)
    finally:
        os.close(fd)
