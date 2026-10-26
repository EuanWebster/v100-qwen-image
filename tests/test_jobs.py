"""CPU-only owned-controller witnesses; executors are explicit fixture fakes."""
from __future__ import annotations

import hashlib
import os
import threading
import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_engine import jobs
from qwen_engine.config import GenerationConfig
from qwen_engine.core import GenerationResult


def saved(context):
    path = Path(context.config.output_dir) / "fixture.bin"
    path.write_bytes(b"synthetic executor artifact, not an image")
    return GenerationResult(str(path))


@pytest.fixture
def harness(tmp_path):
    controllers, releases = [], []

    def create(executor=saved, **kwargs):
        root = tmp_path / ("controller-" + str(len(controllers)))
        root.mkdir()
        controller = jobs.JobController(root, executor, **kwargs)
        controllers.append(controller)
        return controller

    def blocked(*, acknowledge=True):
        entered, release = threading.Event(), threading.Event()
        releases.append(release)

        def executor(context):
            entered.set()
            assert release.wait(3)
            if acknowledge:
                context.check_cancelled()
            return saved(context)

        controller = create(executor)
        record = controller.submit(GenerationConfig())
        assert entered.wait(2)
        return controller, record, release

    yield SimpleNamespace(create=create, blocked=blocked, releases=releases)
    for release in releases:
        release.set()
    for controller in controllers:
        assert controller.close(timeout_s=3), "test left an owned worker running"


def test_real_artifact_only_success_and_immutable_snapshots(harness):
    seen = []

    def execute(context):
        seen.append(context)
        for phase in ("load", "encode", "denoise", "decode", "save"):
            context.report_phase(phase)
        return saved(context)

    controller = harness.create(execute)
    original = GenerationConfig(prompt="preserve exact  text")
    queued = controller.submit(original)
    done = controller.wait(queued.job_id)
    assert queued.state == jobs.JobState.QUEUED and queued.artifact is None
    assert done.state == jobs.JobState.SUCCEEDED and done.phase == "save"
    assert done.config.prompt == original.prompt and original.output_dir is None
    assert done.config.output_dir == str(controller.root / done.job_id)
    assert done.artifact.filename == "fixture.bin"
    data = (Path(done.config.output_dir) / done.artifact.filename).read_bytes()
    assert done.artifact.bytes == len(data) and done.artifact.sha256 == hashlib.sha256(data).hexdigest()
    assert controller.snapshots() == (done,)
    with pytest.raises(FrozenInstanceError):
        done.state = jobs.JobState.QUEUED
    with pytest.raises(jobs.JobError, match="outside running"):
        seen[0].report_phase("load")


def test_queue_cancel_never_executes_and_active_cancel_waits(harness):
    controller, active, release = harness.blocked()
    queued = controller.submit(GenerationConfig())
    cancelled = controller.cancel(queued.job_id)
    assert cancelled.state == jobs.JobState.CANCELLED and cancelled.cancel_requested
    cancelling = controller.cancel(active.job_id)
    assert cancelling.state == jobs.JobState.CANCELLING and not cancelling.terminal
    assert controller.wait(active.job_id, timeout_s=0).state == jobs.JobState.CANCELLING
    assert not controller.close(timeout_s=0)
    assert controller._thread.is_alive()
    release.set()
    assert controller.wait(active.job_id).state == jobs.JobState.CANCELLED
    assert controller.close()
    assert controller.cancel(queued.job_id) == cancelled
    assert not list(Path(queued.config.output_dir).iterdir())


def test_verified_completion_wins_late_cancellation(harness):
    controller, record, release = harness.blocked(acknowledge=False)
    controller.cancel(record.job_id)
    release.set()
    done = controller.wait(record.job_id)
    assert done.state == jobs.JobState.SUCCEEDED and done.cancel_requested and done.artifact is not None


def test_single_worker_serializes_jobs(harness):
    controller, first, release = harness.blocked()
    second = controller.submit(GenerationConfig())
    assert controller.get(first.job_id).state == jobs.JobState.RUNNING
    assert controller.get(second.job_id).state == jobs.JobState.QUEUED
    release.set()
    assert controller.wait(first.job_id).state == jobs.JobState.SUCCEEDED
    assert controller.wait(second.job_id).state == jobs.JobState.SUCCEEDED


def test_history_cap_and_forget_preserve_artifacts(harness):
    controller = harness.create(max_pending=1, max_records=1)
    first = controller.submit(GenerationConfig())
    assert controller.wait(first.job_id).terminal
    with pytest.raises(jobs.JobError, match="history full"):
        controller.submit(GenerationConfig())
    path = Path(first.config.output_dir) / "fixture.bin"
    controller.forget(first.job_id)
    assert path.is_file() and controller.snapshots() == ()
    second = controller.submit(GenerationConfig())
    assert controller.wait(second.job_id).terminal
    with pytest.raises(jobs.JobError, match="unknown"):
        controller.get(first.job_id)


def test_queue_cap_no_directory_on_refused_submission(harness):
    controller, active, release = harness.blocked()
    controller._max_pending = 1
    controller.submit(GenerationConfig())
    before = tuple(controller.root.iterdir())
    with pytest.raises(jobs.JobError, match="queue full"):
        controller.submit(GenerationConfig())
    assert tuple(controller.root.iterdir()) == before
    with pytest.raises(jobs.JobError, match="live"):
        controller.forget(active.job_id)


@pytest.mark.parametrize("pending,records", [(0, 1), (True, 4), (33, 34), (2, 1), (1, 129), (1, True)])
def test_invalid_limits_before_worker_or_io(tmp_path, pending, records):
    with pytest.raises(jobs.JobError, match="limits"):
        jobs.JobController(tmp_path / "absent", saved, max_pending=pending, max_records=records)


@pytest.mark.parametrize("bad", [None, {}, "", 1])
def test_requires_explicit_executor_before_io(tmp_path, bad):
    with pytest.raises(jobs.JobError, match="explicit executor"):
        jobs.JobController(tmp_path / "absent", bad)


@pytest.mark.parametrize("bad", [None, {}, True, GenerationConfig(output_dir="configured"),
                                 GenerationConfig(prompt="a" * (jobs.MAX_REQUEST_BYTES + 1)),
                                 GenerationConfig(prompt="蓝" * (jobs.MAX_REQUEST_BYTES // 2))])
def test_bad_or_oversized_input_before_creating_directory(harness, bad):
    controller = harness.create()
    with pytest.raises(jobs.JobError):
        controller.submit(bad)
    assert controller.snapshots() == () and not list(controller.root.iterdir())


def test_relative_paths_bound_at_submission(harness, tmp_path, monkeypatch):
    controller = harness.create()
    monkeypatch.chdir(tmp_path)
    record = controller.submit(GenerationConfig(models_dir="models/../assets", cache_dir="cache"))
    assert record.config.models_dir == str(tmp_path / "models/../assets")
    assert record.config.cache_dir == str(tmp_path / "cache")
    assert controller.wait(record.job_id).terminal


@pytest.mark.parametrize("value", [True, None, -1, 61, float("inf"), float("nan"), 10 ** 1000])
def test_timeout_validation_does_not_close_or_wait(harness, value):
    controller = harness.create()
    record = controller.submit(GenerationConfig())
    with pytest.raises(jobs.JobError, match="timeout"):
        controller.close(timeout_s=value)
    with pytest.raises(jobs.JobError, match="timeout"):
        controller.wait(record.job_id, timeout_s=value)
    assert not controller._closing


@pytest.mark.parametrize("phase", [None, "bogus", "save", "encode", 3])
def test_invalid_phase_report_fails_without_exposing_exception(harness, phase):
    def execute(context):
        context.report_phase(phase)
        return saved(context)
    controller = harness.create(execute)
    record = controller.submit(GenerationConfig())
    done = controller.wait(record.job_id)
    assert done.state == jobs.JobState.FAILED and done.error_code == "execution_failed"
    assert done.artifact is None


@pytest.mark.parametrize("kind", ["wrong-result", "missing", "empty", "outside", "symlink", "hardlink", "fifo", "directory", "oversize"])
def test_invalid_artifact_never_reports_success(harness, tmp_path, monkeypatch, kind):
    def execute(context):
        path = Path(context.config.output_dir) / "result.bin"
        if kind == "wrong-result":
            return str(path)
        if kind == "outside":
            path = tmp_path / "elsewhere.bin"
            path.write_bytes(b"external")
        elif kind == "symlink":
            peer = path.parent / "peer.bin"
            peer.write_bytes(b"peer")
            path.symlink_to(peer)
        elif kind == "hardlink":
            peer = path.parent / "peer.bin"
            peer.write_bytes(b"peer")
            os.link(peer, path)
        elif kind == "fifo":
            os.mkfifo(path)
        elif kind == "directory":
            path.mkdir()
        elif kind == "empty":
            path.write_bytes(b"")
        elif kind == "oversize":
            path.write_bytes(b"too big")
            monkeypatch.setattr(jobs, "MAX_ARTIFACT_BYTES", 1)
        return GenerationResult(str(path))
    controller = harness.create(execute)
    record = controller.submit(GenerationConfig())
    done = controller.wait(record.job_id)
    assert done.state == jobs.JobState.FAILED and done.artifact is None


def test_executor_failure_sanitized_and_next_job_runs(harness):
    calls = []
    def execute(context):
        calls.append(context.job_id)
        if len(calls) == 1:
            raise RuntimeError("secret must not appear in job state")
        return saved(context)
    controller = harness.create(execute)
    first = controller.submit(GenerationConfig())
    assert controller.wait(first.job_id).error_code == "execution_failed"
    second = controller.submit(GenerationConfig())
    assert controller.wait(second.job_id).state == jobs.JobState.SUCCEEDED



TorchOutOfMemory = type("OutOfMemoryError", (RuntimeError,), {})
TorchOutOfMemory.__module__ = "torch"


@pytest.mark.parametrize("kind,code", [
    ("torch-class", "out_of_gpu_memory"),
    ("torch-subclass-module", "out_of_gpu_memory"),
    ("runtime-cuda-text", "out_of_gpu_memory"),
    ("typed-engine", "out_of_gpu_memory"),
    ("typed-via-cause", "out_of_gpu_memory"),
    ("plain-runtime-text", "execution_failed"),
    ("same-name-other-module", "execution_failed"),
    ("context-only-oom", "execution_failed"),
    ("unrequested-cancel-unchanged", "invalid_cancellation_acknowledgment"),
])
def test_cuda_oom_classifies_typed_without_importing_torch(harness, kind, code):
    from qwen_engine.errors import GenerationCancelled, OutOfGpuMemoryError

    def execute(context):
        if kind == "torch-class":
            raise TorchOutOfMemory("CUDA out of memory. Tried to allocate 2.00 GiB")
        if kind == "torch-subclass-module":
            subclass = type("OutOfMemoryError", (TorchOutOfMemory,), {})
            subclass.__module__ = "torch.cuda"
            raise subclass("CUDA out of memory")
        if kind == "runtime-cuda-text":
            raise RuntimeError("CUDA error: out of memory in cudaMallocAsync")
        if kind == "typed-engine":
            raise OutOfGpuMemoryError("adapter-reported device exhaustion")
        if kind == "typed-via-cause":
            try:
                raise TorchOutOfMemory("CUDA out of memory")
            except TorchOutOfMemory as inner:
                raise RuntimeError("wrapper text") from inner
        if kind == "plain-runtime-text":
            raise RuntimeError("some unrelated cuda kernel fault")
        if kind == "same-name-other-module":
            other = type("OutOfMemoryError", (RuntimeError,), {})
            other.__module__ = "numpy"
            raise other("host allocation failed")
        if kind == "context-only-oom":
            try:
                raise TorchOutOfMemory("CUDA out of memory")
            except TorchOutOfMemory:
                raise ValueError("primary failure with OOM as context only")
        raise GenerationCancelled("cancelled while an OOM frame was in flight")

    released = []
    controller = harness.create(execute, memory_release=lambda: released.append(1))
    record = controller.submit(GenerationConfig())
    done = controller.wait(record.job_id)
    assert done.state == jobs.JobState.FAILED and done.error_code == code
    assert "CUDA" not in repr(done) and "secret" not in repr(done)
    assert released == [1]  # the bc550be failure-boundary release still runs


def test_unrequested_cancellation_is_failure(harness):
    def execute(context):
        raise jobs.JobCancelled()
    controller = harness.create(execute)
    first = controller.submit(GenerationConfig())
    assert controller.wait(first.job_id).error_code == "invalid_cancellation_acknowledgment"


def test_worker_baseexception_marks_all_jobs_failed(harness, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    errors = []
    harness.releases.append(release)
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(args.exc_value))
    def execute(context):
        entered.set()
        assert release.wait(3)
        raise SystemExit("worker exited")
    controller = harness.create(execute)
    first = controller.submit(GenerationConfig())
    assert entered.wait(2)
    second = controller.submit(GenerationConfig())
    release.set()
    assert controller.wait(first.job_id).error_code == "worker_stopped"
    assert controller.wait(second.job_id).error_code == "worker_stopped"
    assert controller.close()
    assert len(errors) == 1 and isinstance(errors[0], SystemExit)
    with pytest.raises(jobs.JobError, match="unavailable"):
        controller.submit(GenerationConfig())


def test_closed_controller_refuses_and_root_descriptor_reaped(harness):
    controller = harness.create()
    fd = controller._root_fd
    assert controller.close() and controller.close()
    with pytest.raises(OSError):
        os.fstat(fd)
    with pytest.raises(jobs.JobError, match="closed"):
        controller.submit(GenerationConfig())


def test_root_replacement_refuses_submission(harness):
    controller = harness.create()
    retained = controller.root.with_name("retained-root")
    controller.root.rename(retained)
    controller.root.mkdir()
    with pytest.raises(jobs.JobError, match="root changed"):
        controller.submit(GenerationConfig())
    assert not list(controller.root.iterdir()) and not list(retained.iterdir())


def test_queued_directory_replacement_refuses_before_executor(harness):
    controller, active, release = harness.blocked()
    queued = controller.submit(GenerationConfig())
    path = Path(queued.config.output_dir)
    path.rename(path.with_name("retained-job"))
    path.mkdir()
    release.set()
    assert controller.wait(active.job_id).state == jobs.JobState.SUCCEEDED
    done = controller.wait(queued.job_id)
    assert done.state == jobs.JobState.FAILED and not list(path.iterdir())


@pytest.mark.parametrize("method", ["get", "cancel", "forget", "wait"])
@pytest.mark.parametrize("job_id", [None, [], "", "x" * 33, "z" * 32, "0" * 32])
def test_unknown_or_invalid_job_identifiers(harness, method, job_id):
    controller = harness.create()
    with pytest.raises(jobs.JobError, match="unknown job"):
        getattr(controller, method)(job_id)


def test_cancellation_race_during_phase_report_is_acknowledged(harness):
    entered, release = threading.Event(), threading.Event()
    harness.releases.append(release)
    def execute(context):
        entered.set()
        assert release.wait(3)
        # Exercise the locked second cancellation check after an earlier
        # context.check_cancelled() could have raced a new cancel request.
        context._report("load")
        return saved(context)
    controller = harness.create(execute)
    record = controller.submit(GenerationConfig())
    assert entered.wait(2)
    controller.cancel(record.job_id)
    release.set()
    assert controller.wait(record.job_id).state == jobs.JobState.CANCELLED


@pytest.mark.parametrize("mutation", ["file-replace", "file-grow", "directory-replace"])
def test_artifact_mutation_detected_and_files_preserved(harness, monkeypatch, mutation):
    original_read = jobs.os.read
    changed = False
    owned = []
    def execute(context):
        result = saved(context)
        owned.append(Path(result.output_path))
        return result
    def read(fd, size):
        nonlocal changed
        data = original_read(fd, size)
        if data and not changed:
            changed = True
            path = owned[0]
            if mutation == "file-replace":
                path.rename(path.with_name("retained-original.bin"))
                path.write_bytes(b"peer bytes")
            elif mutation == "file-grow":
                with path.open("ab") as stream:
                    stream.write(b"growth")
            else:
                path.parent.rename(path.parent.with_name("retained-original-dir"))
                path.parent.mkdir()
                path.write_bytes(b"peer bytes")
        return data
    monkeypatch.setattr(jobs.os, "read", read)
    controller = harness.create(execute)
    record = controller.submit(GenerationConfig())
    done = controller.wait(record.job_id)
    assert changed and done.state == jobs.JobState.FAILED and done.artifact is None
    assert list(controller.root.iterdir())  # no rollback/deletion of originals or peers


def test_thread_start_failure_closes_owned_descriptor(tmp_path, monkeypatch):
    opened = []
    original = jobs.os.open
    def capture(*args, **kwargs):
        fd = original(*args, **kwargs)
        opened.append(fd)
        return fd
    monkeypatch.setattr(jobs.os, "open", capture)
    monkeypatch.setattr(jobs.threading.Thread, "start", lambda self: (_ for _ in ()).throw(RuntimeError("cannot start")))
    with pytest.raises(RuntimeError, match="cannot start"):
        jobs.JobController(tmp_path, saved)
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_duplicate_id_or_existing_directory_never_overwrites(harness, monkeypatch):
    monkeypatch.setattr(jobs.uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    controller = harness.create()
    first = controller.submit(GenerationConfig())
    assert controller.wait(first.job_id).state == jobs.JobState.SUCCEEDED
    path = Path(first.config.output_dir) / "fixture.bin"
    expected = path.read_bytes()
    with pytest.raises(jobs.JobError, match="collision"):
        controller.submit(GenerationConfig())
    controller.forget(first.job_id)
    with pytest.raises(FileExistsError):
        controller.submit(GenerationConfig())
    assert path.read_bytes() == expected


def test_executor_self_close_remains_truthfully_live_until_return(harness):
    entered, release = threading.Event(), threading.Event()
    harness.releases.append(release)
    controllers = []
    observed = []
    def execute(context):
        entered.set()
        assert release.wait(3)
        observed.append(controllers[0].close(timeout_s=0))
        context.check_cancelled()
    controller = harness.create(execute)
    controllers.append(controller)
    record = controller.submit(GenerationConfig())
    assert entered.wait(2)
    release.set()
    assert controller.wait(record.job_id).state == jobs.JobState.CANCELLED
    assert observed == [False] and controller.close()


def test_surrogate_input_is_typed_and_creates_no_job(harness):
    controller = harness.create()
    with pytest.raises(jobs.JobError, match="UTF-8 JSON"):
        controller.submit(GenerationConfig(prompt="\ud800"))
    assert not controller.snapshots() and not list(controller.root.iterdir())


def test_request_snapshot_owns_builtin_values_not_mutable_subclasses(harness):
    class Text(str):
        pass
    class Number(int):
        pass
    controller = harness.create()
    original = GenerationConfig(prompt=Text("exact prompt"), steps=Number(40))
    submitted = controller.submit(original)
    assert type(submitted.config.prompt) is str and type(submitted.config.steps) is int
    assert submitted.config.prompt == original.prompt and submitted.config.steps == original.steps
    assert controller.wait(submitted.job_id).state == jobs.JobState.SUCCEEDED


def test_boundary_collector_fires_once_after_state_for_failed_and_cancelled_only(harness):
    controller_ref = {}
    releases = []
    collected = threading.Event()

    def collect():
        controller = controller_ref["controller"]
        releases.append({
            "states": tuple((r.state, r.error_code) for r in controller.snapshots()),
            "handling_exception": sys.exc_info()[1] is not None,
            "thread": threading.current_thread().name,
        })
        collected.set()

    entered, release = threading.Event(), threading.Event()
    block_entered, block_release = threading.Event(), threading.Event()
    harness.releases.extend([release, block_release])

    def execute(context):
        if context.config.prompt == "fail":
            raise RuntimeError("fixture executor failure")
        if context.config.prompt == "cancel":
            entered.set()
            assert release.wait(3)
            context.check_cancelled()
        if context.config.prompt == "block":
            block_entered.set()
            assert block_release.wait(3)
        return saved(context)

    def recorded(controller, job_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            value = controller.collect_ms(job_id)
            if value is not None:
                return value
            time.sleep(0.005)
        raise AssertionError("boundary release duration was never recorded")

    controller = harness.create(execute, memory_release=collect)
    controller_ref["controller"] = controller

    failed = controller.submit(GenerationConfig(prompt="fail"))
    assert controller.wait(failed.job_id).error_code == "execution_failed"
    assert collected.wait(3) and len(releases) == 1
    assert recorded(controller, failed.job_id) >= 0.0
    # the terminal state and its fixed error code are already recorded when
    # the collector runs, and the worker thread holds no exception being
    # handled: the release is outside every except scope, after bookkeeping
    assert releases[0]["states"] == ((jobs.JobState.FAILED, "execution_failed"),)
    assert releases[0]["handling_exception"] is False
    assert releases[0]["thread"] == "qwen-owned-job-worker"
    collected.clear()

    cancelled = controller.submit(GenerationConfig(prompt="cancel"))
    assert entered.wait(2)
    assert controller.cancel(cancelled.job_id).state == jobs.JobState.CANCELLING
    release.set()
    assert controller.wait(cancelled.job_id).state == jobs.JobState.CANCELLED
    assert collected.wait(3) and len(releases) == 2
    assert recorded(controller, cancelled.job_id) >= 0.0
    assert releases[1]["states"] == ((jobs.JobState.FAILED, "execution_failed"),
                                     (jobs.JobState.CANCELLED, None))
    collected.clear()

    # keep the worker busy so the queued cancellation can never race into
    # execution; successes and never-executed queued cancellations never pay
    blocker = controller.submit(GenerationConfig(prompt="block"))
    assert block_entered.wait(2)
    queued = controller.submit(GenerationConfig(prompt="later"))
    assert controller.cancel(queued.job_id).state == jobs.JobState.CANCELLED
    good = controller.submit(GenerationConfig())
    block_release.set()
    assert controller.wait(blocker.job_id).state == jobs.JobState.SUCCEEDED
    assert controller.wait(good.job_id).state == jobs.JobState.SUCCEEDED
    assert len(releases) == 2
    assert controller.collect_ms(good.job_id) is None
    assert controller.collect_ms(queued.job_id) is None
    controller.forget(failed.job_id)
    assert controller.collect_ms(failed.job_id) is None


@pytest.mark.parametrize("bad", [True, 1, "", [], object()])
def test_non_callable_memory_release_refuses_before_worker_or_io(tmp_path, bad):
    with pytest.raises(jobs.JobError, match="memory release collector"):
        jobs.JobController(tmp_path / "absent", saved, memory_release=bad)
