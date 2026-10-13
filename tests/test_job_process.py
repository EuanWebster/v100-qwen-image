"""Real bounded owned subprocess tests; synthetic backend only via private test redirect."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from qwen_engine import generation_worker as w, job_process as p
from qwen_engine.config import GenerationConfig
from qwen_engine.jobs import JobContext, JobController, JobState, JobCancelled

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    instances, controllers, children = [], [], []
    original = subprocess.Popen
    commands = []

    def redirect(command, *args, **kwargs):
        assert command[:4] == [sys.executable, "-B", "-m", "qwen_engine.generation_worker"]
        assert kwargs["start_new_session"] is True and kwargs["stdin"] == subprocess.DEVNULL
        commands.append(command)
        child = original([*command[:2], str(ROOT / "tests/fixtures/generation_process_child.py"), *command[4:]],
                         *args, **kwargs)
        children.append(child)
        return child

    def create(mode="ok", **options):
        monkeypatch.setenv("QWEN_PROCESS_TEST_MODE", mode)
        monkeypatch.setattr(p.subprocess, "Popen", redirect)
        directory = tmp_path / str(len(instances))
        scratch = directory / "scratch"
        output = directory / "output"
        scratch.mkdir(parents=True)
        output.mkdir()
        executor = p.GenerationProcessExecutor(scratch_dir=scratch, python=sys.executable,
                                              cancel_grace_s=.2, **options)
        instances.append(executor)
        controller = JobController(output, executor)
        controllers.append(controller)
        return executor, controller

    yield create, children, commands
    for controller in controllers:
        assert controller.close(timeout_s=3)
    for executor in instances:
        assert executor.close(timeout_s=3)
        assert executor.active_pid is None and not list(executor.scratch_dir.iterdir())
    assert all(child.poll() is not None for child in children)


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(.01)
    raise AssertionError("fixture did not reach expected boundary")


def test_fixed_worker_success_all_phases_and_fresh_artifact(harness):
    create, children, commands = harness
    executor, controller = create()
    record = controller.submit(GenerationConfig())
    done = controller.wait(record.job_id)
    assert done.state == JobState.SUCCEEDED and done.phase == "save"
    assert done.artifact.filename == "fixture.bin"
    assert len(children) == 1 and children[0].returncode == 0
    assert executor.active_pid is None and not list(executor.scratch_dir.iterdir())
    assert commands[0][4] == "--request" and commands[0][6] == "--response"


@pytest.mark.parametrize("mode", ["block", "ignore_term"])
def test_cancellation_reaps_even_noncooperative_leader(harness, mode):
    create, children, commands = harness
    executor, controller = create(mode)
    record = controller.submit(GenerationConfig())
    wait_for(lambda: controller.get(record.job_id).phase == "denoise")
    controller.cancel(record.job_id)
    done = controller.wait(record.job_id)
    assert done.state == JobState.CANCELLED and children[0].poll() is not None
    assert executor.active_pid is None and not list(executor.scratch_dir.iterdir())


def test_timeout_reaps_worker_and_does_not_report_cancel_without_request(harness):
    create, children, commands = harness
    executor, controller = create("ignore_term", timeout_s=.25)
    record = controller.submit(GenerationConfig())
    done = controller.wait(record.job_id)
    assert done.state == JobState.FAILED and not done.cancel_requested and done.artifact is None
    assert children[0].poll() is not None


@pytest.mark.parametrize("mode", ["stdout_flood", "stderr_flood", "bad_phase", "bad_result", "no_result", "symlink_result", "fifo_result", "oversize_result"])
def test_bad_child_protocol_output_or_special_result_never_succeeds(harness, mode):
    create, children, commands = harness
    executor, controller = create(mode, max_output_bytes=1024)
    record = controller.submit(GenerationConfig())
    done = controller.wait(record.job_id)
    assert done.state == JobState.FAILED and done.artifact is None and done.error_code == "execution_failed"
    assert children[0].poll() is not None


def test_verified_publication_wins_late_cancel(harness):
    create, children, commands = harness
    executor, controller = create("publish_then_wait")
    record = controller.submit(GenerationConfig())
    wait_for(lambda: (Path(record.config.output_dir) / "published.marker").exists())
    controller.cancel(record.job_id)
    done = controller.wait(record.job_id)
    assert done.state == JobState.SUCCEEDED and done.cancel_requested and done.artifact is not None
    assert children[0].returncode == 0


def test_worker_owns_and_reaps_its_encoder_style_descendant(harness):
    create, children, commands = harness
    executor, controller = create("descendant")
    record = controller.submit(GenerationConfig())
    marker = Path(record.config.output_dir) / "descendant.pid"
    wait_for(marker.exists)
    pid = int(marker.read_text())
    controller.cancel(record.job_id)
    assert controller.wait(record.job_id).state == JobState.CANCELLED
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_close_requests_stop_without_pretending_synchronous_reap(harness):
    create, children, commands = harness
    executor, controller = create("block")
    record = controller.submit(GenerationConfig())
    wait_for(lambda: controller.get(record.job_id).phase == "denoise")
    assert not executor.close(timeout_s=0)
    assert controller.wait(record.job_id).state == JobState.FAILED
    assert executor.close() and children[0].poll() is not None
    assert executor.active_pid is None


def test_parent_callback_fault_stops_child_and_preserves_primary(harness):
    create, children, commands = harness
    executor, controller = create("block")
    directory = controller.root / "direct"
    directory.mkdir()
    error = ValueError("callback fault")
    def fail(phase):
        raise error
    context = JobContext("b" * 32, GenerationConfig(output_dir=directory), threading.Event(), fail)
    with pytest.raises(ValueError) as caught:
        executor(context)
    assert caught.value is error and children[0].poll() is not None


def test_real_product_worker_refuses_without_assets_or_ml(tmp_path):
    scratch, output = tmp_path / "scratch", tmp_path / "output"
    scratch.mkdir()
    output.mkdir()
    executor = p.GenerationProcessExecutor(scratch_dir=scratch, python=sys.executable)
    controller = JobController(output, executor)
    try:
        record = controller.submit(GenerationConfig())
        assert controller.wait(record.job_id).state == JobState.FAILED
        assert executor.active_pid is None and not list(scratch.iterdir())
    finally:
        assert controller.close() and executor.close()


@pytest.mark.parametrize("field,value", [("timeout_s", 0), ("timeout_s", True), ("timeout_s", float("inf")),
    ("timeout_s", 10 ** 1000), ("cancel_grace_s", -1), ("cancel_grace_s", 31),
    ("max_output_bytes", 0), ("max_output_bytes", True), ("max_output_bytes", 1024 * 1024 + 1),
    ("python", ""), ("python", "bad\0path")])
def test_invalid_process_setup_before_scratch_io(tmp_path, field, value):
    with pytest.raises(p.GenerationProcessError):
        p.GenerationProcessExecutor(scratch_dir=tmp_path / "absent", **{field: value})


@pytest.fixture
def result_context(tmp_path):
    return JobContext("a" * 32, GenerationConfig(output_dir=tmp_path), threading.Event(), lambda p: None)


def result(context):
    return dict(schema=w.RESULT_SCHEMA, request_id=context.job_id, status="succeeded", exit_code=0,
                output_path=str(Path(context.config.output_dir) / "result.bin"), error_code=None)


@pytest.mark.parametrize("field,value", [("schema", "wrong"), ("request_id", "b" * 32), ("status", "wrong"),
    ("exit_code", True), ("exit_code", 7), ("output_path", "relative"), ("output_path", None),
    ("output_path", []), ("error_code", "arbitrary")])
def test_result_envelope_drift_typed(result_context, field, value):
    document = result(result_context)
    document[field] = value
    with pytest.raises(p.GenerationProcessError):
        p._result(document, result_context, 0)


@pytest.mark.parametrize("document", [[], None, {}, {"secret": "not echoed"}])
def test_nondocument_or_missing_result_fields_typed(result_context, document):
    with pytest.raises(p.GenerationProcessError) as caught:
        p._result(document, result_context, 0)
    assert str(caught.value) == "invalid_result_envelope"


def test_unrequested_child_cancel_is_not_accepted(result_context):
    document = result(result_context)
    document.update(status="cancelled", exit_code=130, output_path=None)
    with pytest.raises(p.GenerationProcessError, match="unrequested"):
        p._result(document, result_context, 130)


@pytest.mark.parametrize("code", [1, 2, 3, 4, 5])
def test_failure_preserves_fixed_category_without_arbitrary_diagnostics(result_context, code):
    document = result(result_context)
    document.update(status="failed", exit_code=code, output_path=None, error_code=w.ERROR_CODES[code])
    with pytest.raises(p.GenerationProcessError) as caught:
        p._result(document, result_context, code)
    assert str(caught.value) == w.ERROR_CODES[code]


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite", "deep", "oversize", "empty"])
def test_owned_response_reader_is_strict_and_bounded(tmp_path, kind):
    path = tmp_path / "response.json"
    data = {"duplicate": b'{"x":1,"x":2}', "nonfinite": b'{"x":NaN}',
            "deep": b"[" * 2000 + b"0" + b"]" * 2000,
            "oversize": b"x" * (w.MAX_RESULT_BYTES + 1), "empty": b""}[kind]
    path.write_bytes(data)
    with pytest.raises(p.GenerationProcessError):
        p._read_result(path)


def test_scratch_root_replacement_refuses_before_spawn(tmp_path, monkeypatch):
    scratch, output = tmp_path / "scratch", tmp_path / "output"
    scratch.mkdir()
    output.mkdir()
    executor = p.GenerationProcessExecutor(scratch_dir=scratch)
    scratch.rename(tmp_path / "retained")
    scratch.mkdir()
    context = JobContext("a" * 32, GenerationConfig(output_dir=output), threading.Event(), lambda p: None)
    monkeypatch.setattr(p.subprocess, "Popen", lambda *a, **kw: pytest.fail("spawn after root replacement"))
    with pytest.raises(p.GenerationProcessError, match="scratch_root_changed"):
        executor(context)
    assert executor.close()


def test_precancel_refuses_without_scratch_or_child(tmp_path, monkeypatch):
    cancel = threading.Event()
    cancel.set()
    executor = p.GenerationProcessExecutor(scratch_dir=tmp_path)
    context = JobContext("a" * 32, GenerationConfig(output_dir=tmp_path), cancel, lambda p: None)
    with pytest.raises(JobCancelled):
        executor(context)
    assert not list(tmp_path.iterdir()) and executor.close()


def test_pending_reap_retains_owner_and_refuses_new_execution(tmp_path, monkeypatch):
    executor = p.GenerationProcessExecutor(scratch_dir=tmp_path)
    signals = []
    waits = []
    class Unreaped:
        pid = 123456  # synthetic; killpg is replaced below, never signals a real PID
        def poll(self):
            return None
        def wait(self, *, timeout):
            waits.append(timeout)
            raise subprocess.TimeoutExpired("synthetic", timeout)
    owned = Unreaped()
    executor._active = owned
    monkeypatch.setattr(p.os, "killpg", lambda pid, signal: signals.append((pid, signal)))
    assert not executor.close(timeout_s=0)
    assert executor.active_pid == owned.pid and waits == [0, 0]
    context = JobContext("a" * 32, GenerationConfig(output_dir=tmp_path), threading.Event(), lambda p: None)
    with pytest.raises(p.GenerationProcessError, match="unavailable"):
        executor(context)
    assert executor._active is owned and len(signals) == 2
    assert not list(tmp_path.iterdir())


def test_reaped_leader_is_never_group_signalled_again(tmp_path, monkeypatch):
    executor = p.GenerationProcessExecutor(scratch_dir=tmp_path)
    class Reaped:
        pid = 123456
        def poll(self):
            return 0
    owned = Reaped()
    executor._active = owned
    monkeypatch.setattr(p.os, "killpg", lambda *a: pytest.fail("signal after ownership ended"))
    executor._stop(owned)
    assert executor.active_pid is None


def test_failure_ack_code_must_match_failure_category(result_context):
    document = result(result_context)
    document.update(status="failed", output_path=None, exit_code=3, error_code="invalid_cancellation_acknowledgment")
    with pytest.raises(p.GenerationProcessError, match="incoherent"):
        p._result(document, result_context, 3)


def test_phase_json_string_brackets_do_not_count_as_nesting():
    assert p._protocol_json(b'{"message":"[[[\\\"{quote}\\\"]]]"}') == {"message": '[[["{quote}"]]]'}
