"""R7 CPU-test guard — active only when ``QWEN_CPU_TEST_GUARD=1``.

Installed via ``PYTHONPATH`` (by ``run_cpu_tests.sh`` or explicitly) so it
runs at interpreter startup in the **pytest process and every subprocess**
it spawns, including the fake encoder workers — no per-API monkeypatching:
one :func:`sys.addaudithook` plus a ``sys.meta_path`` finder cover open,
remove, rename, rmdir, replace, symlink, rmtree, utime, chmod…, every
socket connect/sendto/getaddrinfo, and imports, respectively.

Blocked (before any side effect happens):

* writes (open for write/append/create/truncate, remove/rename/replace/
  mkdir/rmdir/rmtree/utime/chmod/…) whose resolved target lies outside the
  configured temp roots (``QWEN_CPU_TMP_ROOTS``, ``os.pathsep``-separated);
  reads — including all repository reads — are never touched;
* network access: socket **construction** for network families (denied at
  ``socket.__new__``, before the OS ``socket()`` call and therefore before
  any fd exists — CPU tests need no network sockets, so this holds even
  when an *outer* sandbox denies socket creation itself), plus connect /
  sendto / getaddrinfo / gethostbyname; AF_UNIX construction is allowed
  (connect/sendto on it are still denied);
* imports of the heavy ML packages (``QWEN_CPU_DENY_IMPORT``), refused by
  a meta-path finder *before any module code executes*, even if a stub is
  on the path.

``/dev`` targets (DEVNULL etc.) are allowed. The hook resolves paths with
``realpath``, so symlinked escapes out of the temp roots are caught.

Scope note: this is an honest-test guard for cooperative CPU tests, not a
security sandbox — a deliberately hostile process could bypass audit hooks
after install (they cannot be removed once added; a re-exec with a stripped
environment is out of scope for this milestone).
"""
from __future__ import annotations

import os
import sys

GUARD_NAME = "qwen-r7-cpu-guard"

GUARDED = os.environ.get("QWEN_CPU_TEST_GUARD") == "1"


def _split_env(name: str) -> tuple:
    return tuple(p for p in (s.strip() for s in
                              os.environ.get(name, "").split(os.pathsep)) if p)


#: Real temp roots every write must stay inside (runner-created run root).
ROOTS = tuple(os.path.realpath(p) for p in _split_env("QWEN_CPU_TMP_ROOTS"))

#: Top-level module names refused before execution (heavy ML boundary).
DENIED_IMPORTS = frozenset(
    n for n in (s.strip()
                for s in os.environ.get("QWEN_CPU_DENY_IMPORT", "")
                .split(",")) if n) or frozenset(
    {"torch", "diffusers", "transformers"})

_WRITE_FLAGS = (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC
                | getattr(os, "O_APPEND", 0) | getattr(os, "O_EXCL", 0))

#: event -> ((path argument index, dir_fd argument index or None), ...).
#: Signatures per the CPython audit-event docs; rmtree/shutil and the
#: concurrent.futures internals delete fd-relatively, so dir_fd matters.
_PATH_EVENTS = {
    "os.mkdir": ((0, 2),),
    "os.rmdir": ((0, 1),),
    "os.removedirs": ((0, None),),
    "os.remove": ((0, 1),), "os.unlink": ((0, 1),),
    "os.rename": ((0, 2), (1, 3)),
    "os.replace": ((0, None), (1, None)),
    "os.symlink": ((1, 2),),
    "os.link": ((0, 2), (1, 3)),
    "os.chmod": ((0, 2),), "os.lchmod": ((0, None),),
    "os.chown": ((0, 4),), "os.utime": ((0, 2),),
    "os.truncate": ((0, None),), "shutil.rmtree": ((0, None),),
}
_NETWORK_EVENTS = frozenset({
    "socket.connect", "socket.connect_ex", "socket.sendto",
    "socket.getaddrinfo", "socket.gethostbyname",
})
#: Construction events: ``socket.__new__`` (3.11+) and the legacy ``socket``
#: name; args are (socket, family, type, protocol) and the hook fires
#: BEFORE the OS socket() call, so a denial leaves no fd behind.
_SOCKET_CREATE_EVENTS = frozenset({"socket", "socket.__new__"})
#: socket.AF_UNIX (stable constant; sitecustomize must not import socket).
_AF_UNIX = 1


class GuardViolation(PermissionError):
    """Raised inside the guarded process when a CPU-test rule would break;
    the offending operation never runs (audit hooks fire beforehand)."""


def _resolve(path, dir_fd=None) -> str | None:
    """Absolute, symlink-resolved target of a possibly fd-relative path;
    ``None`` when it cannot be determined (then the write is denied)."""
    if isinstance(path, int):  # fd-based operation on an already-open file
        return f"/proc/self/fd/{path}"
    p = os.fsdecode(os.fspath(path))
    if os.path.isabs(p):
        return os.path.realpath(p)
    if dir_fd is not None:
        try:  # Linux: the dir_fd's own location (reads of /proc are fine)
            base = os.readlink(f"/proc/self/fd/{int(dir_fd)}")
        except OSError:
            return None
        return os.path.realpath(os.path.join(base, p))
    return os.path.realpath(os.path.abspath(p))


def _within_roots(path, dir_fd=None) -> bool:
    rp = _resolve(path, dir_fd)
    if rp is None:
        return False
    if rp == "/dev" or rp.startswith("/dev" + os.sep):
        return True  # DEVNULL and friends: harmless sink
    if rp.startswith("/proc/self/fd/"):
        return True  # already-open descriptor (its open was checked)
    return any(rp == root or rp.startswith(root + os.sep) for root in ROOTS)


def _check_write(path, what: str, dir_fd=None) -> None:
    if not _within_roots(path, dir_fd):
        shown = os.fsdecode(os.fspath(path)) if not isinstance(path, int) \
            else f"<fd {path}>"
        raise GuardViolation(
            f"{GUARD_NAME}: {what} would write outside the configured CPU "
            f"test temp roots {list(ROOTS)!r}: {shown!r}"
            " (R7: CPU tests write only inside temporary directories)")


def _audit_hook(event: str, args: tuple) -> None:
    if event == "open":
        path, mode, flags = args[0], args[1], args[2]
        writing = isinstance(mode, str) and any(c in mode for c in "wax+")
        if not writing:
            try:
                writing = bool(int(flags) & _WRITE_FLAGS)
            except (TypeError, ValueError):
                writing = False
        if writing:
            dir_fd = args[3] if len(args) > 3 else None
            _check_write(path, "open", dir_fd)
    elif event in _PATH_EVENTS:
        for path_idx, fd_idx in _PATH_EVENTS[event]:
            if path_idx < len(args):
                fd = args[fd_idx] if fd_idx is not None and fd_idx < len(args) \
                    else None
                _check_write(args[path_idx], event, fd)
    elif event in _NETWORK_EVENTS:
        raise GuardViolation(
            f"{GUARD_NAME}: network access denied in CPU tests "
            f"(event {event!r}) — R7 requires a no-network environment")
    elif event in _SOCKET_CREATE_EVENTS:
        family = args[1] if len(args) > 1 else None
        if family != _AF_UNIX:
            raise GuardViolation(
                f"{GUARD_NAME}: network socket construction denied in CPU "
                f"tests (event {event!r}, family {family!r}) — denied at "
                "socket.__new__ before any OS socket/fd exists; R7 "
                "requires a no-network environment")


class _DeniedImportFinder:
    """Refuses heavy ML imports before any module code can run — a stub on
    sys.path is never executed, so there is nothing to undo afterwards."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        top = fullname.partition(".")[0]
        if top in DENIED_IMPORTS:
            raise ImportError(
                f"{GUARD_NAME}: import of {top!r} is denied in CPU tests "
                "(no heavy ML imports outside the real adapter); refused "
                "before any module code executed")
        return None

    # Legacy API, kept so nothing can smuggle a finder-protocol variant.
    def find_module(self, fullname, path=None):  # noqa: ARG002
        self.find_spec(fullname)
        return None


INSTALLED = False


def install() -> None:
    global INSTALLED
    if INSTALLED:
        return
    INSTALLED = True
    if DENIED_IMPORTS:
        sys.meta_path.insert(0, _DeniedImportFinder())
    sys.addaudithook(_audit_hook)


if GUARDED:
    install()
