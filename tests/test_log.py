"""Task 1.1 acceptance: structured JSONL phase/event logger with injectable
clock/sink, and *no* default writes outside configured paths."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

from qwen_engine.log import PhaseLogger, file_sink

HEAVY_MODULES = ("torch", "diffusers", "transformers")


class ListSink:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, line: str) -> None:
        self.lines.append(line)

    def records(self) -> list[dict]:
        return [json.loads(line) for line in self.lines]


def make_logger(**kw):
    sink = ListSink()
    logger = PhaseLogger(sink=sink, clock=lambda: 123.456, wall_clock=lambda: 1.0, **kw)
    return sink, logger


def test_event_emits_one_jsonl_line_with_core_fields():
    sink, logger = make_logger()
    logger.event("done")
    assert len(sink.lines) == 1
    rec = sink.records()[0]
    assert rec == {"schema_version": 1, "run_id": None, "ts": 1.0,
                   "phase": "engine", "event": "done", "elapsed_ms": None}


def test_run_id_is_injectable_and_appears_on_every_record():
    sink = ListSink()
    logger = PhaseLogger(sink=sink, clock=lambda: 0.0, wall_clock=lambda: 1.0,
                         run_id="run-42")
    with logger.phase("load"):
        logger.event("tick")
    for rec in sink.records():
        assert rec["run_id"] == "run-42"
        assert rec["schema_version"] == 1


def test_reserved_keys_cover_attribution_fields():
    _, logger = make_logger()
    with pytest.raises(ValueError):
        logger.event("done", run_id="forged")
    with pytest.raises(ValueError):
        logger.event("done", schema_version=99)


def test_extra_fields_merge_into_record_and_are_sorted():
    sink, logger = make_logger()
    logger.event("cached", tokens=4096, hits=1)
    line = sink.lines[0]
    assert line == json.dumps(json.loads(line), sort_keys=True)
    rec = sink.records()[0]
    assert rec["tokens"] == 4096 and rec["hits"] == 1


def test_reserved_keys_are_rejected_in_extras():
    _, logger = make_logger()
    with pytest.raises(ValueError):
        logger.event("done", ts=99)


def test_phase_context_emits_start_and_end_with_elapsed():
    ticks = iter([0.0, 0.5])  # start, end
    sink = ListSink()
    logger = PhaseLogger(sink=sink, clock=lambda: next(ticks), wall_clock=lambda: 1.0)
    with logger.phase("denoise"):
        pass
    starts, ends = sink.records()
    assert starts["event"] == "phase_start" and starts["phase"] == "denoise"
    assert starts["elapsed_ms"] == 0.0
    assert ends["event"] == "phase_end" and ends["phase"] == "denoise"
    assert ends["elapsed_ms"] == pytest.approx(500.0)


def test_events_inside_phase_inherit_phase_and_elapsed():
    ticks = itertools.count(0, 1)  # 0, 1, 2 ... seconds
    sink = ListSink()
    logger = PhaseLogger(sink=sink, clock=lambda: float(next(ticks)), wall_clock=lambda: 1.0)
    with logger.phase("denoise"):
        logger.event("step")
    rec = sink.records()[1]
    assert rec["phase"] == "denoise"
    assert rec["event"] == "step"
    assert rec["elapsed_ms"] == pytest.approx(1000.0)


def test_phase_error_is_logged_and_raised():
    sink, logger = make_logger()
    with pytest.raises(RuntimeError):
        with logger.phase("load"):
            raise RuntimeError("boom")
    events = [r["event"] for r in sink.records()]
    assert events == ["phase_start", "phase_error"]
    assert sink.records()[1]["error"] == "RuntimeError"


def test_nested_phases_restore_outer_phase():
    sink, logger = make_logger()
    with logger.phase("outer"):
        with logger.phase("inner"):
            logger.event("deep")
        logger.event("back-out")
    recs = sink.records()
    assert recs[2]["phase"] == "inner"
    assert recs[-1]["phase"] == "outer"


class ExplodingSink:
    """Raises whenever a record whose event is in *fail_on* is emitted."""

    def __init__(self, fail_on, inner=None):
        self.fail_on = set(fail_on)
        self.inner = inner or ListSink()

    def __call__(self, line: str) -> None:
        rec = json.loads(line)
        if rec["event"] in self.fail_on:
            raise OSError("log sink exploded")
        self.inner(line)

    def records(self):
        return self.inner.records()


def test_phase_stack_survives_failed_start_emission():
    """A sink that dies at phase_start must not leak a stack entry: the
    logger keeps working afterwards with a clean phase context."""
    sink = ExplodingSink({"phase_start"})
    logger = PhaseLogger(sink=sink, clock=lambda: 0.0, wall_clock=lambda: 1.0)
    with logger.phase("load"):
        pass
    assert logger._stack == []
    logger.event("after")  # must report the default phase, not "load"
    assert sink.records()[-1]["phase"] == "engine"


def test_failed_phase_error_emission_does_not_mask_primary_exception():
    sink = ExplodingSink({"phase_error"})
    logger = PhaseLogger(sink=sink, clock=lambda: 0.0, wall_clock=lambda: 1.0)
    with pytest.raises(RuntimeError):  # not OSError from the sink
        with logger.phase("load"):
            raise RuntimeError("boom")
    assert logger._stack == []


# --- review fix: completion logging is nonfatal ----------------------------
# phase_start/phase_error sink failures were already guarded; phase_end and
# the human completion line must be guarded the same way, or a successful
# save would exit 1 on a log fault.


def test_failing_phase_end_sink_does_not_fail_completed_phase():
    sink = ExplodingSink({"phase_end"})
    logger = PhaseLogger(sink=sink, clock=lambda: 0.0, wall_clock=lambda: 1.0)
    with logger.phase("save"):  # must NOT raise OSError from the sink
        pass
    assert logger._stack == []  # stack still cleaned up
    logger.event("after")  # logger keeps working afterwards
    assert sink.records()[-1]["phase"] == "engine"


def test_failing_human_completion_sink_does_not_fail_completed_phase():
    class ExplodingHuman:
        def __call__(self, line: str) -> None:
            raise OSError("human sink exploded")

    jsonl = ListSink()
    logger = PhaseLogger(sink=jsonl, human_sink=ExplodingHuman(),
                         clock=lambda: 0.0, wall_clock=lambda: 1.0)
    with logger.phase("save"):  # must NOT raise
        pass
    assert logger._stack == []
    # The machine-readable record was still emitted before the human fault.
    assert [r["event"] for r in jsonl.records()] == ["phase_start", "phase_end"]


def test_body_exception_propagates_despite_failing_completion_sinks():
    """Guards must not swallow the body: with EVERY completion event failing
    at the sink, the primary exception still propagates unchanged and the
    stack is cleaned."""
    sink = ExplodingSink({"phase_end", "phase_error"})

    class ExplodingHuman:
        def __call__(self, line: str) -> None:
            raise OSError("human sink exploded")

    logger = PhaseLogger(sink=sink, human_sink=ExplodingHuman(),
                         clock=lambda: 0.0, wall_clock=lambda: 1.0)
    with pytest.raises(RuntimeError):
        with logger.phase("load"):
            raise RuntimeError("boom")
    assert logger._stack == []


def test_default_logger_writes_nothing_anywhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = PhaseLogger()  # no sink configured
    with logger.phase("load"):
        logger.event("tick")
    assert list(tmp_path.iterdir()) == []  # zero files created by default


def test_file_sink_writes_only_to_configured_path(tmp_path):
    target = tmp_path / "logs" / "engine.jsonl"
    sink = file_sink(target)
    sink('{"a": 1}')
    sink('{"a": 2}')
    assert target.read_text().splitlines() == ['{"a": 1}', '{"a": 2}']
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [target]


def test_relative_file_sink_binds_cwd_without_eager_writes(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    sink = file_sink("logs/engine.jsonl")
    assert not list(first.iterdir())
    monkeypatch.chdir(second)
    sink('{"event":"bound"}')
    assert (first / "logs" / "engine.jsonl").read_text().splitlines() == ['{"event":"bound"}']
    assert not list(second.iterdir())


def test_relative_file_sink_preserves_symlink_dotdot_semantics(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    target = tmp_path / "target" / "deep"
    first.mkdir()
    second.mkdir()
    target.mkdir(parents=True)
    (first / "alias").symlink_to(target, target_is_directory=True)
    monkeypatch.chdir(first)
    sink = file_sink("alias/../logs/engine.jsonl")
    monkeypatch.chdir(second)
    sink('{"event":"lexical"}')
    assert (target.parent / "logs" / "engine.jsonl").read_text().splitlines() == ['{"event":"lexical"}']
    assert not (first / "logs").exists()
    assert not list(second.iterdir())


def test_human_sink_gets_readable_phase_lines(tmp_path):
    human = ListSink()
    ticks = iter([0.0, 2.0])
    logger = PhaseLogger(sink=ListSink(), human_sink=human,
                         clock=lambda: next(ticks), wall_clock=lambda: 1.0)
    with logger.phase("decode"):
        pass
    assert len(human.lines) == 1
    assert "decode" in human.lines[0] and "2000" in human.lines[0]


def test_log_import_pulls_in_no_heavy_modules():
    import qwen_engine.log  # noqa: F401
    assert not set(HEAVY_MODULES) & set(sys.modules)


@pytest.mark.parametrize("point", ["start", "event", "end", "error"])
@pytest.mark.parametrize("fault", [OSError("clock failed"), ValueError("clock failed"),
                                  None, "not a clock", float("nan"), float("inf"), 10**1000])
def test_timing_faults_are_nonfatal_and_do_not_mask_body_errors(point, fault):
    sink, human = ListSink(), ListSink()
    calls = 0
    def clock():
        nonlocal calls
        calls += 1
        if calls == (1 if point == "start" else 2):
            if isinstance(fault, Exception):
                raise fault
            return fault
        return 0.0
    logger = PhaseLogger(sink=sink, human_sink=human, clock=clock, wall_clock=lambda: 1.0)
    primary = RuntimeError("original pipeline failure")
    def body():
        with logger.phase("save"):
            if point == "event":
                logger.event("tick")
            if point == "error":
                raise primary
    if point == "error":
        with pytest.raises(RuntimeError) as caught:
            body()
        assert caught.value is primary
    else:
        body()
    assert logger._stack == []
    recs = sink.records()
    bad = next(r for r in recs if r["event"] == (
        "tick" if point == "event" else "phase_error" if point == "error" else "phase_end"))
    assert bad["elapsed_ms"] is None
    if point in ("start", "end"):
        assert human.lines == ["[save] done (timing unavailable)"]


def test_timing_interrupt_is_not_swallowed():
    def clock():
        raise KeyboardInterrupt()
    logger = PhaseLogger(clock=clock)
    with pytest.raises(KeyboardInterrupt):
        with logger.phase("load"):
            pytest.fail("body after interrupt")
    assert logger._stack == []
