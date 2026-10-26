"""Persistent job history: per-job records written on terminal states and
strictly rebuilt from disk by a fresh controller (a server restart).

Everything here runs on tmp directories with fake executors: the record is
the only new on-disk state, and a restarted controller must never load,
serve or write anything that a live one would refuse.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from qwen_engine import jobs
from qwen_engine.config import GenerationConfig
from qwen_engine.core import GenerationResult

PNG = b"\x89PNG\r\n\x1a\nsynthetic record fixture"


def saving(context):
    path = Path(context.config.output_dir) / "result.png"
    path.write_bytes(PNG)
    return GenerationResult(str(path))


def failing(context):
    raise RuntimeError("executor exploded: prompt-bearing text never stored")


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "web"
    path.mkdir()
    return path


@pytest.fixture
def controllers():
    started = []

    def build(root, executor=saving, **kwargs):
        controller = jobs.JobController(root, executor, **kwargs)
        started.append(controller)
        return controller

    yield build
    for controller in started:
        controller.close(timeout_s=3)


def record_path(record) -> Path:
    return Path(record.config.output_dir) / jobs.JOB_RECORD_NAME


def stored_document(record) -> dict:
    return json.loads(record_path(record).read_bytes())


def skipped_lines(capsys) -> list[str]:
    return [line for line in capsys.readouterr().err.splitlines()
            if line.startswith("qwen-history: skipped job")]


def test_success_record_round_trips_through_a_restart(root, controllers):
    first = controllers(root)
    done = first.wait(first.submit(GenerationConfig(prompt="persisted prompt", seed=99)).job_id)
    assert done.state is jobs.JobState.SUCCEEDED
    assert first.close()

    stored = stored_document(done)
    assert set(stored) == jobs._RECORD_KEYS
    assert stored["schema"] == jobs.JOB_RECORD_SCHEMA and stored["job_id"] == done.job_id
    assert stored["state"] == "succeeded" and stored["error_code"] is None
    assert stored["artifact"] == {"filename": done.artifact.filename,
                                 "bytes": done.artifact.bytes,
                                 "sha256": done.artifact.sha256}
    assert 0 <= stored["created_ns"] <= stored["finished_ns"]

    second = controllers(root)
    restored = second.snapshots()
    assert [job.job_id for job in restored] == [done.job_id]
    job = restored[0]
    assert job.state is jobs.JobState.SUCCEEDED and job.artifact == done.artifact
    assert job.config.to_dict() == done.config.to_dict()
    assert job.config.prompt == "persisted prompt" and job.config.seed == 99
    assert job.config.output_dir == done.config.output_dir
    assert second.read_artifact(job.job_id) == PNG


@pytest.mark.parametrize("damage,noticed", [("same-size", False), ("resized", True),
                                           ("removed", True)])
def test_damaged_image_entry_stays_visible_but_is_never_served(
        root, controllers, capsys, damage, noticed):
    """A valid record is what history is rebuilt from; the image is only ever
    handed out through the digest gate, so a damaged file cannot be served —
    and the entry stays visible instead of silently disappearing from the UI
    while its job.json is still on disk."""
    first = controllers(root)
    done = first.wait(first.submit(GenerationConfig()).job_id)
    assert first.close()
    image = Path(done.config.output_dir) / done.artifact.filename
    if damage == "same-size":
        image.write_bytes(b"\x00" * len(PNG))
    elif damage == "resized":
        image.write_bytes(PNG + b"extra")
    else:
        image.unlink()

    second = controllers(root)
    assert [job.job_id for job in second.snapshots()] == [done.job_id]
    with pytest.raises((jobs.JobError, OSError)):
        second.read_artifact(done.job_id)
    err = capsys.readouterr().err.splitlines()
    # A valid record is not "skipped": at most the image is unavailable. The
    # startup proof covers size/link only (the sha256 is re-checked on every
    # serve), so same-size substitution is caught at request time instead.
    assert not [line for line in err if line.startswith("qwen-history: skipped job")]
    assert bool([line for line in err if "image no longer matches" in line]) is noticed


@pytest.mark.parametrize("artifact_change", ["other-name", "bigger-size"])
def test_record_whose_artifact_does_not_match_the_file_is_never_served(
        root, controllers, artifact_change):
    first = controllers(root)
    done = first.wait(first.submit(GenerationConfig()).job_id)
    assert first.close()
    document = stored_document(done)
    if artifact_change == "other-name":
        document["artifact"]["filename"] = "other.png"
    else:
        document["artifact"]["bytes"] = len(PNG) + 1
    record_path(done).write_bytes(json.dumps(document).encode())

    second = controllers(root)
    assert [job.job_id for job in second.snapshots()] == [done.job_id]
    with pytest.raises((jobs.JobError, OSError)):
        second.read_artifact(done.job_id)


@pytest.mark.parametrize("payload", [b"", b"{not json", b"[]", b'{"schema": 1}',
                                     b"\xff\xfe\x00", b"[" * 4000])
def test_corrupt_record_is_skipped_with_one_log_line_and_never_fatal(
        root, controllers, capsys, payload):
    first = controllers(root)
    good = first.wait(first.submit(GenerationConfig(prompt="survives")).job_id)
    assert first.close()
    corrupt = root / ("c" * 32)
    corrupt.mkdir()
    (corrupt / jobs.JOB_RECORD_NAME).write_bytes(payload)

    second = controllers(root)
    assert [job.job_id for job in second.snapshots()] == [good.job_id]
    assert second.read_artifact(good.job_id) == PNG
    lines = skipped_lines(capsys)
    assert len(lines) == 1 and "c" * 32 in lines[0]


def test_forgotten_entry_stays_forgotten_and_keeps_the_image(root, controllers):
    first = controllers(root)
    done = first.wait(first.submit(GenerationConfig()).job_id)
    image = Path(done.config.output_dir) / done.artifact.filename
    first.forget(done.job_id)
    assert not record_path(done).exists() and image.read_bytes() == PNG
    assert first.close()

    second = controllers(root)
    assert second.snapshots() == ()
    with pytest.raises(jobs.JobError, match="unknown job"):
        second.get(done.job_id)
    assert image.read_bytes() == PNG


def test_job_without_record_is_not_resurrected_and_gains_nothing(root, controllers):
    abandoned = root / ("a" * 32)  # shape of a job that was running when the app died
    abandoned.mkdir()
    second = controllers(root)
    assert second.snapshots() == ()
    assert list(abandoned.iterdir()) == []


def test_live_job_writes_its_record_only_at_the_terminal_state(root, controllers):
    entered, release = threading.Event(), threading.Event()

    def blocked(context):
        entered.set()
        assert release.wait(3)
        return saving(context)

    controller = controllers(root, blocked)
    record = controller.submit(GenerationConfig())
    assert entered.wait(2)
    assert not record_path(record).exists()  # nothing to resurrect mid-flight
    release.set()
    assert controller.wait(record.job_id).state is jobs.JobState.SUCCEEDED
    assert record_path(record).is_file()


def test_failed_job_record_keeps_only_the_fixed_error_code(root, controllers, capsys):
    first = controllers(root, failing)
    done = first.wait(first.submit(GenerationConfig()).job_id)
    assert done.state is jobs.JobState.FAILED and done.error_code == "execution_failed"
    stored = stored_document(done)
    assert stored["state"] == "failed" and stored["error_code"] == "execution_failed"
    assert stored["artifact"] is None
    assert first.close()

    second = controllers(root)
    restored = second.snapshots()
    assert [job.state for job in restored] == [jobs.JobState.FAILED]
    assert restored[0].error_code == "execution_failed" and restored[0].artifact is None
    assert skipped_lines(capsys) == []


def test_queued_cancellation_is_recorded_and_restored(root, controllers):
    entered, release = threading.Event(), threading.Event()

    def blocked(context):
        entered.set()
        assert release.wait(3)
        context.check_cancelled()
        return saving(context)

    controller = controllers(root, blocked)
    active = controller.submit(GenerationConfig())
    assert entered.wait(2)
    queued = controller.submit(GenerationConfig())
    cancelled = controller.cancel(queued.job_id)
    assert cancelled.state is jobs.JobState.CANCELLED
    stored = stored_document(queued)
    assert stored["state"] == "cancelled" and stored["cancel_requested"] is True
    release.set()
    assert controller.wait(active.job_id).terminal and controller.close()

    second = controllers(root)
    restored = {job.job_id: job for job in second.snapshots()}
    assert set(restored) == {active.job_id, queued.job_id}
    assert restored[queued.job_id].state is jobs.JobState.CANCELLED
    assert restored[queued.job_id].cancel_requested is True


def test_restored_history_is_bounded_to_the_newest_records_and_deletes_nothing(root, controllers):
    first = controllers(root, max_pending=1, max_records=3)
    done = []
    for _ in range(3):
        record = first.submit(GenerationConfig())
        done.append(first.wait(record.job_id))
        time.sleep(0.002)
    live_order = [job.job_id for job in first.snapshots()]
    assert live_order == [record.job_id for record in done]
    assert first.close()

    second = controllers(root, max_pending=1, max_records=2)
    restored = [job.job_id for job in second.snapshots()]
    # Oldest first, exactly the order the live controller accumulated: the
    # newest two survive, the oldest is simply not loaded (never deleted).
    assert restored == live_order[1:]
    for record in done:  # nothing was deleted to make room
        assert record_path(record).is_file()
        assert (Path(record.config.output_dir) / record.artifact.filename).read_bytes() == PNG
    assert second.read_artifact(done[1].job_id) == PNG


@pytest.mark.parametrize("mutate", [
    lambda doc: {**doc, "job_id": "d" * 32},
    lambda doc: {**doc, "schema": "qwen_engine.job_record.v2"},
    lambda doc: {**doc, "state": "running"},
    lambda doc: {**doc, "state": "queued"},
    lambda doc: {**doc, "error_code": "execution_failed"},
    lambda doc: {**doc, "error_code": "free text from an exception"},
    lambda doc: {**doc, "cancel_requested": "no"},
    lambda doc: {**doc, "created_ns": -1},
    lambda doc: {**doc, "finished_ns": 0},
    lambda doc: {**doc, "extra": None},
    lambda doc: {key: value for key, value in doc.items() if key != "config"},
    lambda doc: {**doc, "config": {"prompt": "forged", "output_dir": "/tmp/elsewhere"}},
    lambda doc: {**doc, "config": {"prompt": "forged"}},
    lambda doc: {**doc, "config": {**doc["config"], "prompt": ""}},
    lambda doc: {**doc, "config": {**doc["config"], "steps": 0}},
    lambda doc: {**doc, "config": {**doc["config"], "extra": 1}},
    lambda doc: {**doc, "artifact": None},
    lambda doc: {**doc, "artifact": {**doc["artifact"], "sha256": "z" * 64}},
    lambda doc: {**doc, "artifact": {**doc["artifact"], "filename": "../escape.png"}},
    lambda doc: {**doc, "artifact": {**doc["artifact"], "bytes": 0}},
])
def test_forged_record_is_refused_without_echoing_values(root, controllers, capsys, mutate):
    first = controllers(root)
    done = first.wait(first.submit(GenerationConfig(prompt="genuine")).job_id)
    assert first.close()
    forged_id = "e" * 32
    forged_dir = root / forged_id
    forged_dir.mkdir()
    (forged_dir / done.artifact.filename).write_bytes(PNG)
    document = mutate(stored_document(done))
    document["job_id"] = forged_id if document.get("job_id") != "d" * 32 else document["job_id"]
    (forged_dir / jobs.JOB_RECORD_NAME).write_bytes(json.dumps(document).encode())

    second = controllers(root)
    assert [job.job_id for job in second.snapshots()] == [done.job_id]
    lines = skipped_lines(capsys)
    assert len(lines) == 1 and "genuine" not in lines[0] and "forged" not in lines[0]


def test_forged_artifact_digest_loads_but_is_never_served(root, controllers):
    """Startup proves size only; the digest is re-verified on every read, so a
    record whose bytes were swapped for same-size content cannot be served."""
    first = controllers(root)
    done = first.wait(first.submit(GenerationConfig(prompt="genuine")).job_id)
    assert first.close()
    document = stored_document(done)
    document["artifact"]["sha256"] = "0" * 64
    record_path(done).write_bytes(json.dumps(document).encode())

    second = controllers(root)
    assert [job.job_id for job in second.snapshots()] == [done.job_id]
    with pytest.raises(jobs.JobError, match="changed"):
        second.read_artifact(done.job_id)


def test_records_stay_inside_their_own_job_directories(root, controllers):
    controller = controllers(root, max_pending=1, max_records=2)
    records = [controller.wait(controller.submit(GenerationConfig()).job_id) for _ in range(2)]
    assert sorted(entry.name for entry in root.iterdir()) == sorted(
        Path(record.config.output_dir).name for record in records)
    for record in records:
        directory = Path(record.config.output_dir)
        assert sorted(entry.name for entry in directory.iterdir()) == sorted(
            [jobs.JOB_RECORD_NAME, record.artifact.filename])
        assert not [entry for entry in directory.iterdir() if entry.name.startswith(".")]


def test_record_with_trailing_bytes_is_skipped_not_partially_accepted(root, controllers, capsys):
    first = controllers(root)
    done = first.wait(first.submit(GenerationConfig()).job_id)
    assert first.close()
    raw = record_path(done).read_bytes()
    record_path(done).write_bytes(raw + b"{}")

    second = controllers(root)
    assert second.snapshots() == ()
    assert len(skipped_lines(capsys)) == 1
