"""Actual private process-tree reaping, not browser or foreign-zombie cleanup."""
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/process_lineage_owner.py"
spec = importlib.util.spec_from_file_location("browser_owner", SCRIPT)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper contract")
@pytest.mark.parametrize("mode", ["orphan_exit", "block"])
def test_actual_wrapper_reaps_owned_fork_tree_normal_and_term_ignoring(tmp_path, mode):
    command = [sys.executable, "-B", str(SCRIPT), "--", sys.executable, "-B",
               str(ROOT / "tests/fixtures/browser_owner_child.py"), mode, str(tmp_path)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not all((tmp_path / (name + ".pid")).is_file() for name in ("leader", "child", "grandchild")):
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.01)
        pids = [int((tmp_path / (name + ".pid")).read_text()) for name in ("leader", "child", "grandchild")]
        if mode == "block":
            process.send_signal(signal.SIGTERM)  # ONLY owner, never its group/KILL
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == (0 if mode == "orphan_exit" else 137), stderr.decode()
        assert not stdout and b"cleanup_pending" not in stderr
        assert all(not Path(f"/proc/{pid}").exists() for pid in pids), "all owned live AND zombie PIDs must be reaped"
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=8)
        process.stdout.close()
        process.stderr.close()


def test_wrapper_rejects_missing_command_without_launch():
    process = subprocess.run([sys.executable, "-B", str(SCRIPT)], capture_output=True, timeout=3)
    assert process.returncode == 2 and b"invalid command" in process.stderr


def test_wrapper_reports_missing_executable_without_orphan():
    process = subprocess.run([sys.executable, "-B", str(SCRIPT), "--", "qwen-nonexistent-browser-fixture"],
                             capture_output=True, timeout=3)
    assert process.returncode == 1 and b"execution_failed" in process.stderr


def test_reap_sets_leader_returncode_and_requires_kernel_no_children(monkeypatch):
    values = iter([(5, 0), (0, 0)])
    monkeypatch.setattr(owner.os, "waitpid", lambda *args: next(values))
    chrome = type("Child", (), {"pid": 5, "returncode": None})()
    assert owner.reap(chrome) is False and chrome.returncode == 0
    def empty(*args):
        raise ChildProcessError()
    monkeypatch.setattr(owner.os, "waitpid", empty)
    assert owner.reap(chrome) is True


def test_signals_only_waitid_confirmed_owned_children_and_live_leader_group(monkeypatch):
    calls = []
    chrome = type("Child", (), {"pid": 5, "returncode": None})()
    monkeypatch.setattr(owner, "direct_children", lambda: (5, 6, 7, 8))
    def waitid(kind, pid, flags):
        assert kind == os.P_PID and flags & os.WNOWAIT
        if pid == 7:
            raise ChildProcessError()
        return object() if pid == 8 else None
    monkeypatch.setattr(owner.os, "waitid", waitid)
    monkeypatch.setattr(owner.os, "killpg", lambda pid, sig: calls.append(("group", pid, sig)))
    monkeypatch.setattr(owner.os, "kill", lambda pid, sig: calls.append(("child", pid, sig)))
    owner.signal_owned(chrome, signal.SIGKILL)
    assert calls == [("group", 5, signal.SIGKILL), ("child", 6, signal.SIGKILL)]


def test_signal_permission_fault_retains_ownership(monkeypatch):
    monkeypatch.setattr(owner, "direct_children", lambda: (6,))
    monkeypatch.setattr(owner.os, "waitid", lambda *args: None)
    def refuse(*args):
        raise PermissionError()
    monkeypatch.setattr(owner.os, "kill", refuse)
    chrome = type("Child", (), {"pid": 5, "returncode": 0})()
    owner.signal_owned(chrome, signal.SIGTERM)  # retry loop retains owner; no exit/foreign signal


@pytest.mark.parametrize("raw", [b"0", b"-1", b"1 1", b"abc", b"99999999999", b"2147483648", b"x" * 65537])
def test_children_inventory_bounds_and_pid_validation(monkeypatch, raw):
    monkeypatch.setattr(owner, "open", lambda *args, **kwargs: io.BytesIO(raw), raising=False)
    with pytest.raises(RuntimeError):
        owner.direct_children()


def test_children_inventory_reads_only_own_single_thread_proc_path(monkeypatch):
    def read(path, mode, *, buffering):
        assert path == f"/proc/self/task/{os.getpid()}/children" and mode == "rb" and buffering == 0
        return io.BytesIO(b"10 11\n")
    monkeypatch.setattr(owner, "open", read, raising=False)
    assert owner.direct_children() == (10, 11)


def test_subreaper_setup_is_local_fixed_request_and_refuses_failure(monkeypatch):
    calls = []
    class Function:
        result = 0
        def __call__(self, *args):
            calls.append(args)
            return self.result
    function = Function()
    library = type("Lib", (), {"prctl": function})()
    def load(name, *, use_errno):
        assert name is None and use_errno is True
        return library
    monkeypatch.setattr(owner.ctypes, "CDLL", load)
    owner.become_subreaper()
    assert calls == [(36, 1, 0, 0, 0)]
    function.result = -1
    with pytest.raises(RuntimeError, match="unavailable"):
        owner.become_subreaper()


def test_proc_fault_never_broadens_signals_to_foreign_pids(monkeypatch):
    def absent():
        raise OSError()
    calls = []
    monkeypatch.setattr(owner, "direct_children", absent)
    monkeypatch.setattr(owner.os, "waitid", lambda *args: None)
    monkeypatch.setattr(owner.os, "killpg", lambda pid, sig: calls.append(pid))
    monkeypatch.setattr(owner.os, "kill", lambda *args: pytest.fail("no adopted-child ownership was established"))
    chrome = type("Child", (), {"pid": 5, "returncode": None})()
    owner.signal_owned(chrome, signal.SIGTERM)
    assert calls == [5]
    chrome.returncode = 0
    owner.signal_owned(chrome, signal.SIGTERM)
    assert calls == [5]  # no group signal after leader reaping
