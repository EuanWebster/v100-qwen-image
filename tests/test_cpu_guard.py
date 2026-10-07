"""R7 CPU-test guard self-tests (plan: no network, no GPU, no weights;
writes only inside temporary directories).

Every test here runs in a *child* interpreter with the guard installed via
``PYTHONPATH`` (the same mechanism ``run_cpu_tests.sh`` uses for the fake
worker subprocesses), so the guard is proven to reject outside writes,
network access and heavy imports **before any side effect** — a stub whose
import would write a marker never executes at all. Network socket
*construction* is denied by the guard at ``socket.__new__``, so the network
tests never depend on the OS allowing socket creation (they hold unchanged
in outer sandboxes that deny ``socket()`` itself with a plain
``PermissionError``). One test additionally
runs real fake-worker tests through ``run_cpu_tests.sh`` end to end, and
one proves the parent-process guard live when the suite itself is guarded.

No runtime library behaviour changes: the guard lives in tests/ only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GUARD_DIR = Path(__file__).parent / "cpu_guard"
REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "run_cpu_tests.sh"

GUARDED_RUN = os.environ.get("QWEN_CPU_TEST_GUARD") == "1"


def guarded_env(tmp_path, roots, *, extra_path=None, deny=None):
    env = dict(os.environ)
    path_entries = [str(GUARD_DIR)] + ([str(extra_path)] if extra_path else [])
    if env.get("PYTHONPATH"):
        path_entries.append(env["PYTHONPATH"])
    env.update({
        "PYTHONPATH": os.pathsep.join(path_entries),
        "QWEN_CPU_TEST_GUARD": "1",
        "QWEN_CPU_TMP_ROOTS": os.pathsep.join(str(r) for r in roots),
        "QWEN_CPU_DENY_IMPORT": deny or "torch,diffusers,transformers",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(roots[0] / "tmp"),
        "HOME": str(roots[0] / "home"),
    })
    return env


def run_guarded(tmp_path, code, roots, **kw):
    return subprocess.run([sys.executable, "-B", "-c", code],
                          env=guarded_env(tmp_path, roots, **kw),
                          capture_output=True, text=True, timeout=60,
                          cwd=str(tmp_path))


# --- outside writes fail BEFORE any side effect -------------------------------


def test_outside_write_denied_before_the_file_exists(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside_write.txt"  # NOT among the roots
    code = (
        "import sys\n"
        f"target = {str(outside)!r}\n"
        "try:\n"
        "    open(target, 'w').write('leak')\n"
        "except PermissionError as exc:\n"
        f"    assert 'qwen-r7-cpu-guard' in str(exc), exc\n"
        "    print('DENIED')\n"
        "    sys.exit(0)\n"
        "print('WROTE'); sys.exit(1)\n"
    )
    proc = run_guarded(tmp_path, code, [inside])
    assert proc.returncode == 0 and "DENIED" in proc.stdout, proc.stderr
    assert not outside.exists(), "denied write must leave NO file behind"


def test_outside_remove_rename_mkdir_denied(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"do not delete me")
    code = (
        "import os, sys\n"
        f"victim = {str(victim)!r}\n"
        f"dest = {str(tmp_path / 'renamed.bin')!r}\n"
        f"newdir = {str(tmp_path / 'evil_dir')!r}\n"
        "denied = 0\n"
        "for fn, args in ((os.remove, (victim,)), (os.replace, (victim, dest)),\n"
        "                 (os.mkdir, (newdir,))):\n"
        "    try:\n"
        "        fn(*args)\n"
        "    except PermissionError as exc:\n"
        "        assert 'qwen-r7-cpu-guard' in str(exc), exc\n"
        "        denied += 1\n"
        "sys.exit(0 if denied == 3 else 1)\n"
    )
    proc = run_guarded(tmp_path, code, [inside])
    assert proc.returncode == 0, proc.stderr
    assert victim.exists() and not (tmp_path / "renamed.bin").exists()
    assert not (tmp_path / "evil_dir").exists()


def test_writes_inside_the_roots_work(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    code = (
        "import os, tempfile\n"
        "os.makedirs(os.environ['TMPDIR'], exist_ok=True)\n"
        "tempfile.tempdir = None\n"
        f"p = {str(inside / 'ok.bin')!r}\n"
        "open(p, 'w').write('fine')\n"
        "d = os.path.join(tempfile.gettempdir(), 'x'); os.makedirs(d)\n"
        "import shutil; shutil.rmtree(d)\n"
    )
    proc = run_guarded(tmp_path, code, [inside])
    assert proc.returncode == 0, proc.stderr
    assert (inside / "ok.bin").read_text() == "fine"


# --- network denied -----------------------------------------------------------


def test_network_denied_before_any_packet(tmp_path):
    """Network is denied before anything reaches the OS: resolution first,
    then **construction** — the guard rejects network socket families at
    ``socket.__new__``, so the denial comes from the guard even when an
    outer sandbox refuses ``socket()`` itself, and no fd is ever created.
    ``connect``/``sendto`` hook rejection is verified through explicit
    audit events (and a real AF_UNIX ``connect`` where the OS lets the
    socket exist), so the test never depends on the OS *allowing* network
    socket creation."""
    inside = tmp_path / "inside"
    inside.mkdir()
    code = (
        "import os, socket, sys\n"
        "for fn, args in ((socket.getaddrinfo, ('127.0.0.1', 9)),\n"
        "                 (socket.gethostbyname, ('localhost',))):\n"
        "    try:\n"
        "        fn(*args)\n"
        "    except OSError as exc:\n"
        "        assert 'qwen-r7-cpu-guard' in str(exc), exc\n"
        "    else:\n"
        "        sys.exit(1)\n"
        "def sock_fds():\n"
        "    out = set()\n"
        "    for n in os.listdir('/proc/self/fd'):\n"
        "        try:\n"
        "            if 'socket:' in os.readlink('/proc/self/fd/' + n):\n"
        "                out.add(n)\n"
        "        except OSError:\n"
        "            pass\n"
        "    return out\n"
        "before = sock_fds()\n"
        "try:\n"
        "    socket.socket()\n"
        "except PermissionError as exc:\n"
        "    assert 'qwen-r7-cpu-guard' in str(exc), exc\n"
        "else:\n"
        "    sys.exit(1)\n"
        "assert sock_fds() == before, 'denied construction leaked an fd'\n"
        "for ev in ('socket.connect', 'socket.sendto'):\n"
        "    try:\n"
        "        sys.audit(ev, None, ('127.0.0.1', 9))\n"
        "    except PermissionError as exc:\n"
        "        assert 'qwen-r7-cpu-guard' in str(exc), exc\n"
        "    else:\n"
        "        sys.exit(1)\n"
        "try:\n"
        "    u = socket.socket(socket.AF_UNIX)\n"
        "except PermissionError as exc:\n"
        "    assert 'qwen-r7-cpu-guard' not in str(exc), exc\n"
        "else:\n"
        "    try:\n"
        "        u.connect('/qwen-r7-nonexistent-socket-path')\n"
        "    except PermissionError as exc:\n"
        "        assert 'qwen-r7-cpu-guard' in str(exc), exc\n"
        "    else:\n"
        "        sys.exit(1)\n"
    )
    proc = run_guarded(tmp_path, code, [inside])
    assert proc.returncode == 0, proc.stderr


# --- heavy imports denied before side effects ---------------------------------


def test_heavy_import_denied_even_with_executing_stub(tmp_path):
    """A stub ``torch`` that would write a marker on import is placed on the
    path; the meta-path finder must refuse the name BEFORE executing it, so
    the marker never appears."""
    inside = tmp_path / "inside"
    inside.mkdir()
    stub = tmp_path / "stub"
    (stub / "torch").mkdir(parents=True)
    marker = inside / "torch-imported.marker"
    (stub / "torch" / "__init__.py").write_text(
        f"open({str(marker)!r}, 'w').write('ran')\n")
    code = (
        "import sys\n"
        "try:\n"
        "    import torch\n"
        "except ImportError as exc:\n"
        f"    assert 'qwen-r7-cpu-guard' in str(exc), exc\n"
        "    sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    proc = subprocess.run([sys.executable, "-B", "-c", code],
                          env=guarded_env(tmp_path, [inside], extra_path=stub),
                          capture_output=True, text=True, timeout=60,
                          cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert not marker.exists(), "stub executed — import not denied beforehand"


# --- guarded fake-worker subprocess still works -------------------------------


def test_fake_worker_subprocess_runs_guarded(tmp_path):
    """The guard covers subprocess fake workers: launched with the guard
    environment (as run_cpu_tests.sh does), the fake worker still answers
    normally — its response lands inside the temp roots."""
    inside = tmp_path / "inside"
    inside.mkdir()
    req = inside / "request.json"
    resp = inside / "response.json"
    req.write_text(json.dumps({
        "request_id": "r-1", "prompt": "guard probe", "dtype": "fp16",
        "kwargs": {}}))
    fake = REPO / "tests" / "fake_encoder_worker.py"
    proc = subprocess.run(
        [sys.executable, str(fake), "--request", str(req),
         "--response", str(resp)],
        env=guarded_env(tmp_path, [inside]), capture_output=True, text=True,
        timeout=60, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(resp.read_text())
    assert doc["status"] == "ok" and doc["request_id"] == "r-1"


# --- end-to-end: the runner itself passes ordinary fake-worker tests ----------


def test_cpu_runner_passes_real_fake_worker_tests(tmp_path):
    node_ids = [
        "tests/test_encoder.py::test_worker_plain_encode_and_stderr_capture",
        "tests/test_encoder.py::test_encode_miss_then_hit_without_double_encode",
        "tests/test_manifest.py::test_manifest_import_pulls_in_no_heavy_modules",
    ]
    proc = subprocess.run(["bash", str(RUNNER), *node_ids],
                          capture_output=True, text=True, timeout=300,
                          cwd=str(REPO), env={**os.environ,
                                              "PYTHON": sys.executable})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "3 passed" in proc.stdout


# --- the parent pytest process is guarded live when running guarded -----------


@pytest.mark.skipif(not GUARDED_RUN,
                    reason="only meaningful inside a run_cpu_tests.sh run")
def test_parent_process_guard_blocks_outside_write_live():
    import sitecustomize as guard

    assert guard.GUARDED and guard.INSTALLED
    probe = Path(guard.ROOTS[0]).parent / f"qwen-guard-probe-{os.getpid()}.txt"
    with pytest.raises(PermissionError, match="qwen-r7-cpu-guard"):
        probe.write_text("leak")
    assert not probe.exists()
    # and ordinary temp writes still work in the pytest process itself
    probe2 = Path(guard.ROOTS[0]) / "inside-ok.txt"
    probe2.write_text("fine")
    assert probe2.read_text() == "fine"
    probe2.unlink()
