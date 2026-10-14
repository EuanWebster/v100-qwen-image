#!/usr/bin/env python3
"""Private Linux terminating-command subreaper, scoped to this process only.

The parent owns this wrapper. It owns one explicit browser/test/tool lineage,
adopts orphaned grandchildren and waits them, unlike a non-reaping PID1.
Never wrap a service whose intended job is to leave background processes alive.
Never alter PID1/global subreaper state, reap foreign children or kill an owner
before its descendants are reaped. Protocol waits are bounded; uninterruptible
kernel children retain this owner instead of fabricating successful cleanup.
"""
from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time

TERM_GRACE_S = 1.0
PENDING_NOTICE_S = 5.0
MAX_CHILD_BYTES = 65536
MAX_CHILDREN = 4096


def become_subreaper():
    if sys.platform != "linux":
        raise RuntimeError("lineage owner requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    # PR_SET_CHILD_SUBREAPER applies only to THIS fresh wrapper. The flag is
    # not inherited by Chrome; orphaned descendants return here to be waited.
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("lineage subreaper unavailable")


def direct_children():
    # Single-threaded wrapper; this proc file contains ONLY its own children.
    with open(f"/proc/self/task/{os.getpid()}/children", "rb", buffering=0) as stream:
        raw = stream.read(MAX_CHILD_BYTES + 1)
    if len(raw) > MAX_CHILD_BYTES:
        raise RuntimeError("owned child byte cap")
    values = raw.split()
    if len(values) > MAX_CHILDREN or any(not value.isdigit() or len(value) > 10 for value in values):
        raise RuntimeError("owned child count/identity cap")
    children = tuple(int(value) for value in values)
    if any(not 1 <= pid <= 2**31 - 1 for pid in children) or len(set(children)) != len(children):
        raise RuntimeError("invalid owned child PID")
    return children


def reap(chrome):
    """Return True only when the kernel says this owner has no children."""
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return True
        if pid == 0:
            return False
        if pid == chrome.pid:
            chrome.returncode = os.waitstatus_to_exitcode(status)


def signal_owned(chrome, sig):
    try:
        children = direct_children()
    except (OSError, RuntimeError):
        # Still retain the owner if procfs cannot enumerate adopted children.
        # The unreaped direct Chrome PID remains safe to check/signal.
        children = (chrome.pid,) if chrome.returncode is None else ()
    for pid in children:
        try:
            # WNOWAIT keeps a dead child's PID reserved until our next reap.
            # ECHILD means it isn't ours; never signal a reused foreign PID.
            exited = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if exited is not None:
                continue
            if pid == chrome.pid and chrome.returncode is None:
                os.killpg(pid, sig)  # Chrome's owned session, live unreaped leader
            else:
                os.kill(pid, sig)  # adopted direct child, ownership just confirmed
        except (ChildProcessError, OSError):
            # Permission/kernel errors mean cleanup is still pending. Never
            # exit the subreaper and silently orphan inaccessible descendants.
            continue


def run(command):
    become_subreaper()  # BEFORE launching any descendant
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous[sig] = signal.signal(sig, stop)
    chrome = None
    primary = None
    stopping_at = None
    noticed = False
    try:
        # Pass stderr/stdout through to Node; never capture unbounded output.
        # Chrome owns a separate session, wrapper remains alive to reap it.
        chrome = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        while True:
            if reap(chrome):
                code = chrome.returncode
                return code if code is not None and code >= 0 else (128 - code if code is not None else 1)
            if stopped or chrome.returncode is not None:
                if stopping_at is None:
                    stopping_at = time.monotonic()
                elapsed = time.monotonic() - stopping_at
                signal_owned(chrome, signal.SIGTERM if elapsed < TERM_GRACE_S else signal.SIGKILL)
                if elapsed >= PENDING_NOTICE_S and not noticed:
                    print("lineage-owner: cleanup_pending; retained owned descendants", file=sys.stderr, flush=True)
                    noticed = True
            time.sleep(.02)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if chrome is not None and primary is not None:
            # Preserve ownership even on wrapper errors. Do not exit and hand
            # live/zombie grandchildren to a non-reaping PID1.
            while not reap(chrome):
                signal_owned(chrome, signal.SIGKILL)
                time.sleep(.02)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2 or args[0] != "--" or not args[1] or "\0" in args[1]:
        print("lineage-owner: invalid command", file=sys.stderr)
        return 2
    try:
        return run(args[1:])
    except Exception:
        print("lineage-owner: execution_failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
