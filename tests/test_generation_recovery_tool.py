"""CPU-only offline tests for the owned recovery diagnostic.

Every scenario collaborator (adapter, probe, timer, gc, image opener) is a
fake injected through ``run_scenarios``; the watchdog poll loop and the VAE
hook subclass are driven directly with stubs. No ML, GPU or real processes.
"""
from __future__ import annotations
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
import weakref

import pytest

from qwen_engine.config import GenerationConfig
from qwen_engine.encoder import EncoderCancelledError
from qwen_engine.errors import EngineError, GenerationCancelled

TOKENS: list = []  # weakrefs to per-cancellation "tensor owner" stand-ins


class OwnerToken:
    """Weakref-able stand-in for a CUDA-tensor-owning object."""


def tool():
    spec = importlib.util.spec_from_file_location(
        "generation_recovery",
        Path(__file__).resolve().parents[1] / "scripts/check_generation_recovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _new_cancellation():
    exc = GenerationCancelled("fixture cancellation")
    exc.owner_token = OwnerToken()
    TOKENS.append(weakref.ref(exc.owner_token))
    return exc


class FakeAdapter:
    """Mirrors the managed protocol; ``mode`` selects the fixture behaviour."""

    def __init__(self, log, tag, mode):
        self.log, self.tag, self.mode = log, tag, mode
        self.calls, self.steps_seen = [], []
        self.event = self.progress = self.decode_arm = None
        self.decode_entered = False
        self.decode_inner_returned_at = None
        self.mid_body = None

    def _note(self, name):
        self.calls.append(name)
        self.log.append(self.tag + name)

    def set_cancel_event(self, event):
        self.event = event
        self._note("bind")

    def set_progress_callback(self, callback):
        self.progress = callback

    def set_decode_arm(self, callback):
        self.decode_arm = callback

    def load(self, config, assets):
        self.steps_seen.append(config.steps)
        self._note("load")

    def _cancelled(self):
        return (self.mode in ("cancel", "leak", "worker") and self.event is not None
                and self.event.is_set())

    def _raise_cancellation(self):
        if self.mode == "worker":
            try:
                raise EncoderCancelledError("fixture worker terminated")
            except EncoderCancelledError:
                raise _new_cancellation() from None
        raise _new_cancellation()

    def _fire_mid_body(self):
        if self.mid_body is not None and self.mode in ("cancel", "leak", "error",
                                                       "worker"):
            self.mid_body.fire_now()  # models a timer expiring INSIDE the body

    def encode(self, config):
        self._note("encode")
        self._fire_mid_body()
        if self.mode == "leak":
            (Path(config.output_dir) / "stray.jpg").write_bytes(b"x")
        if self.mode == "error":
            raise ValueError("fixture fault")
        if self._cancelled():
            self._raise_cancellation()
        return "conditioning"

    def denoise(self, config, conditioning):
        self._note("denoise")
        if self.mode == "fail":
            raise EngineError("fixture follow-up fault")
        if self.progress is not None:
            for completed in (1, 2, 3):
                self.progress(completed, config.steps)
                if self._cancelled():
                    self._raise_cancellation()
        elif self._cancelled():
            self._raise_cancellation()
        return "latents"

    def decode(self, latents):
        self._note("decode")
        if self.decode_arm is not None:
            self.decode_entered = True
            self.decode_arm()  # timer fires immediately under the fake
        self.decode_inner_returned_at = time.monotonic()
        if self._cancelled():
            self._raise_cancellation()
        return "image"

    def save(self, image, config):
        self._note("save")
        if self.mode == "late" and self.mid_body is not None:
            self.mid_body.fire_now()  # cancel requested during the save body
        path = Path(config.output_dir) / "published.jpg"
        path.write_bytes(b"img")
        return path

    def close(self):
        self._note("close")


class FakeTimer:
    """``fire_on="arm"`` fires instantly; ``"body"`` fires only when the
    fixture explicitly calls fire_now() from inside a phase body (the encode
    watchdog shape)."""

    def __init__(self, log, tag, action, delay, fire_on="arm", clean=True):
        self.log, self.tag = log, tag
        self.action, self.delay, self.fire_on = action, delay, fire_on
        self.clean = clean
        self.fired_at = None
        self.stopped = False

    def arm(self):
        self.log.append(self.tag + "arm")
        if self.fire_on == "arm":
            self.fire_now()

    def fire_now(self):
        if self.fired_at is None:
            self.fired_at = time.monotonic()
            self.action()

    def stop(self):
        self.stopped = True
        self.log.append(self.tag + "stop")
        return self.clean


class FakeProbe:
    def __init__(self, log, children=[], stage="worker"):
        self.log, self.count = log, 0
        self.children, self.stage = children, stage

    def snapshot(self):
        self.count += 1
        self.log.append(f"snapshot{self.count}")
        return dict(allocated_bytes=1000 * self.count, reserved_bytes=2000,
                    peak_allocated_bytes=3000, peak_reserved_bytes=4000,
                    num_alloc_retries=self.count, host_rss_bytes=500,
                    host_lifetime_peak_rss_bytes=None,
                    children=self.children, monotonic_s=0.0)

    def child_stage(self):
        return self.stage

    @staticmethod
    def retry_delta(before, after):
        if before is None or after is None:
            return None
        return after - before


def run(tmp_path, modes=None, state=None, children=[], stage="worker",
        resolution=[1024, 1024], timer_clean=True):
    module = tool()
    TOKENS.clear()
    log, adapters, timers = [], [], []
    if state is not None:
        state.update(log=log, adapters=adapters, timers=timers)
    modes = modes or {}

    def adapter_factory(cancel_phase):
        adapter = FakeAdapter(log, f"a{len(adapters)}",
                              modes.get(cancel_phase, "cancel"))
        if cancel_phase == "encode" and timers:
            adapter.mid_body = timers[0]
        adapters.append((cancel_phase, adapter))
        return adapter

    def timer_factory(cancel_phase, action, delay):
        timer = FakeTimer(log, f"t{len(timers)}", action, delay,
                          fire_on="arm" if cancel_phase == "decode" else "body",
                          clean=timer_clean)
        timers.append(timer)
        return timer

    def gc_collect():
        # A REAL gc pass: the cancellation exception (and its owner token)
        # stay reachable only through cycles (e.g. core's own primary-frame
        # cycle) unless the diagnostic itself still holds them — so if any
        # token survives this collect, the diagnostic retained the primary.
        log.append("gc")
        return gc.collect()

    report = module.run_scenarios(
        GenerationConfig(prompt="private prompt", seed=7, steps=40), tmp_path,
        assets={}, adapter_factory=adapter_factory,
        probe=FakeProbe(log, children=children, stage=stage),
        timer_factory=timer_factory, gc_collect=gc_collect,
        image_opener=lambda path: resolution)
    return module, report, log, adapters, timers


def test_full_sequence_orders_cancels_then_recovers(tmp_path):
    _, report, log, adapters, timers = run(tmp_path)
    assert report["stop"] is None and len(report["scenarios"]) == 4
    assert [phase for phase, _ in adapters] == ["encode", "sampling", "decode", "none"]
    by_phase = {entry["cancel_phase"]: entry for entry in report["scenarios"]}
    # without the worker-termination chain the label must NOT over-claim
    assert by_phase["encode"]["outcome"] == "cancelled-in-encode-window"
    assert by_phase["encode"]["encode_child_stage"] == "worker"
    assert not by_phase["encode"]["encoder_child_cancelled"]
    assert by_phase["sampling"]["outcome"] == "cancelled-after-step-2"
    assert by_phase["sampling"]["steps_observed"] == [1, 2]
    decode = by_phase["decode"]
    assert decode["decode_entered"] and decode["decode_cancel_inside"]
    assert decode["outcome"] == "cancelled-in-vae-decode"
    assert "save" not in decode["phases"]
    for entry in report["scenarios"][:3]:
        assert entry["acknowledged"] and entry["published_files"] == 0
        assert entry["primary_error"] == "GenerationCancelled"
        assert entry["before"]["allocated_bytes"] is not None
        assert entry["after_gc"]["children"] == []
        assert entry["retry_delta_after"] == 1
    for phase in ("encode", "decode"):
        assert by_phase[phase]["timer_fired"] and by_phase[phase]["timer_terminated"]
    assert by_phase["sampling"]["timer_fired"] is None
    followup = report["scenarios"][3]
    assert followup["outcome"] == "succeeded" and followup["published_files"] == 1
    assert followup["output_resolution"] == [1024, 1024]
    assert report["cancelled_steps"] == 4 and report["followup_steps"] == 40
    assert [adapter.calls[-1] for _, adapter in adapters] == ["close"] * 4
    assert [phase for phase, adapter in adapters if "save" in adapter.calls] == ["none"]
    assert adapters[0][1].steps_seen == [4] and adapters[3][1].steps_seen == [40]
    # the diagnostic never holds the primary through measurement: every
    # cancellation token is freed by the diagnostic's own gc pass
    assert all(ref() is None for ref in TOKENS)
    assert log.index("snapshot1") < log.index("gc") < log.index("snapshot3")
    assert log.count("gc") == 3 and sum(
        1 for line in log if line.startswith("snapshot")) == 11
    assert len(timers) == 2 and all(t.stopped and t.stop() for t in timers)


def test_encoder_worker_label_requires_worker_argv_and_chain(tmp_path):
    _, report, _, _, _ = run(tmp_path, modes={"encode": "worker"})
    entry = report["scenarios"][0]
    assert entry["outcome"] == "cancelled-in-encoder-worker"
    assert entry["encoder_child_cancelled"] and entry["encode_child_stage"] == "worker"


def test_encoder_worker_label_downgrades_without_argv_evidence(tmp_path):
    _, report, _, _, _ = run(tmp_path, modes={"encode": "worker"}, stage=None)
    entry = report["scenarios"][0]
    assert entry["encoder_child_cancelled"]
    assert entry["encode_child_stage"] is None
    assert entry["outcome"] == "cancelled-in-encode-window"


def test_watchdog_polls_worker_then_times_out_explicitly():
    module = tool()
    fired, waits = [], []
    stages = ["identity", "worker"]
    watchdog = module.WorkerWatchdog(
        lambda: fired.append("worker"), lambda: stages.pop(0) if stages else None,
        delay=0.5, timeout=60.0, interval=0.2, clock=lambda: 0.0,
        wait=lambda s: waits.append(s) or False)
    assert watchdog.poll() is True
    assert fired == ["worker"] and 0.2 in waits and 0.5 in waits
    assert watchdog.fired_at == 0.0 and watchdog.stop() is True
    times = [0.0, 5.0]
    fired2 = []
    watchdog = module.WorkerWatchdog(
        lambda: fired2.append("timeout"), lambda: None, delay=0.5, timeout=1.0,
        interval=0.2, clock=lambda: times.pop(0) if times else 5.0,
        wait=lambda s: False)
    assert watchdog.poll() is True
    assert fired2 == ["timeout"] and watchdog.fired_at == 5.0


def test_decode_hook_installs_and_removes_its_own_override():
    module = tool()
    adapter = module._make_decode_observing_adapter_cls()(trust_unbounded_host=False)
    arms = []
    adapter.set_decode_arm(lambda: arms.append(1))

    class Vae:
        def __init__(self, fail=False):
            self.calls, self.fail = 0, fail

        def decode(self, *args):
            self.calls += 1
            if self.fail:
                raise ValueError("fixture decode failure")
            return "decoded"

    class Pipeline:
        vae = None

    pipeline = Pipeline()
    pipeline.vae = Vae()
    adapter._pipeline = pipeline
    adapter._restore_vae()
    assert "decode" in vars(pipeline.vae)
    assert pipeline.vae.decode(None) == "decoded"
    assert arms == [1] and adapter.decode_entered
    assert adapter.decode_inner_returned_at is not None
    # no diagnostic hook cycle may survive into the gc measurement
    assert "decode" not in vars(pipeline.vae)
    assert "_recovery_probe" not in vars(pipeline.vae)
    pipeline.vae = Vae(fail=True)
    adapter._restore_vae()
    with pytest.raises(ValueError):
        pipeline.vae.decode(None)
    assert "decode" not in vars(pipeline.vae)
    assert "_recovery_probe" not in vars(pipeline.vae)


def test_published_image_on_cancelled_job_refuses_further_scenarios(tmp_path):
    with pytest.raises(Exception) as caught:
        run(tmp_path, modes={"encode": "leak"})
    assert type(caught.value).__name__ == "RecoveryRefusal"
    assert caught.value.reason == "image-published"
    report = json.loads((tmp_path / "recovery-report.json").read_text())
    assert report["stop"] == "image-published"
    assert len(report["scenarios"]) == 1
    assert report["scenarios"][0]["published_files"] == 1


def test_unacknowledged_and_faulty_primaries_stop_with_evidence(tmp_path):
    for mode, primary in (("late", None), ("error", "ValueError")):
        fresh = tmp_path / mode
        fresh.mkdir()
        state = {}
        with pytest.raises(Exception) as caught:
            run(fresh, modes={"encode": mode}, state=state)
        assert caught.value.reason == "cancellation-unacknowledged"
        entry = json.loads((fresh / "recovery-report.json").read_text()
                           )["scenarios"][0]
        assert entry["primary_error"] == primary
        # the encode watchdog was armed and stopped even though the job ended
        # by an unacknowledged/incorrect primary
        assert state["timers"][0].stopped


def test_followup_failure_refuses_after_clean_cancellations(tmp_path):
    with pytest.raises(Exception) as caught:
        run(tmp_path, modes={"none": "fail"})
    assert caught.value.reason == "followup-failed"
    report = json.loads((tmp_path / "recovery-report.json").read_text())
    assert [entry["outcome"] for entry in report["scenarios"][:3]] == [
        "cancelled-in-encode-window", "cancelled-after-step-2",
        "cancelled-in-vae-decode"]


def test_followup_requires_correctly_sized_image(tmp_path):
    for label, resolution in (("unreadable", None), ("wrong-size", [512, 512])):
        fresh = tmp_path / label
        fresh.mkdir()
        with pytest.raises(Exception) as caught:
            run(fresh, resolution=resolution)
        assert caught.value.reason == "followup-failed"
        entry = json.loads((fresh / "recovery-report.json").read_text()
                           )["scenarios"][-1]
        assert entry["published_files"] == 1
        assert entry["output_resolution"] == resolution


def test_live_or_unreadable_children_refuse_completion(tmp_path):
    for label, children, reason in (("live", [999], "children-retained"),
                                    ("unknown", None, "children-unknown")):
        fresh = tmp_path / label
        fresh.mkdir()
        with pytest.raises(Exception) as caught:
            run(fresh, children=children)
        assert caught.value.reason == reason
        assert len(json.loads((fresh / "recovery-report.json").read_text()
                              )["scenarios"]) == 1


def test_unterminated_timer_refuses_completion(tmp_path):
    with pytest.raises(Exception) as caught:
        run(tmp_path, timer_clean=False)
    assert caught.value.reason == "timer-not-terminated"


def test_report_is_bounded_prompt_free_and_no_clobber(tmp_path):
    run(tmp_path)
    text = (tmp_path / "recovery-report.json").read_text()
    assert "private" not in text and str(tmp_path) not in text
    assert json.loads(text)["schema"] == "qwen_engine.generation_recovery.v1"
    module = tool()
    with pytest.raises(FileExistsError):
        module._write_report(tmp_path, {"schema": "x"})


def test_plan_only_never_runs_or_echoes_prompt(tmp_path, capsys):
    module = tool()
    path = tmp_path / "input.json"
    path.write_text(GenerationConfig(prompt="private prompt").to_json())
    out = tmp_path / "benchmark"
    assert module.main(["--config", str(path), "--output-dir", str(out)]) == 0
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc["run"] is False and doc["cancelled_steps"] == 4
    assert doc["followup_steps"] == 40 and doc["seed"] == 0
    assert "private" not in captured.out + captured.err
    assert not out.exists() and not (tmp_path / "recovery-report.json").exists()
    for heavy in ("torch", "diffusers", "transformers"):
        assert heavy not in sys.modules


def test_unfixed_seed_and_existing_output_refused_before_any_run(tmp_path):
    module = tool()
    seedless = tmp_path / "seedless.json"
    seedless.write_text(GenerationConfig(seed=None).to_json())
    assert module.main(["--config", str(seedless), "--output-dir",
                        str(tmp_path / "a"), "--run"]) == 1
    assert not (tmp_path / "a").exists()
    path = tmp_path / "input.json"
    path.write_text(GenerationConfig().to_json())
    assert module.main(["--config", str(path), "--output-dir", str(tmp_path),
                        "--run"]) == 1
