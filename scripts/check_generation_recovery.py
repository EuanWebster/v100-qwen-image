#!/usr/bin/env python3
"""Opt-in real V100 cancellation/recovery diagnostic; default is PLAN ONLY.

Serial real scenarios in ONE process, each with a fresh single-use adapter:
encode cancellation (a bounded child-only watchdog arms the moment the
actual encoder WORKER child starts and settles 0.5 s before cancelling; the
child's own argv classifies "identity" vs "worker" — worker existence proves
the process started, NOT model-encoding progress, so the label is
cancelled-in-encoder-worker), sampling cancellation (normal adapter
.set_progress_callback sets the SAME threading.Event after the actual
completed step 2 and core propagates GenerationCancelled), cancellation
during the actual VAE.decode (diagnostic subclass arms a timer at VAE.decode
function entry and removes its own hook before returning, so no diagnostic
reference cycle enters the measurement), then a successful follow-up job with
the user's own steps. Model loading cost is expected.

Memory snapshots are taken BEFORE the job, AFTER the primary exception has
been converted to string/bool observations (the diagnostic releases its own
traceback reference first, so what survives to the snapshot and the one
explicit diagnostic gc.collect is production retention, not diagnostic
retention; the gc cost is measured, never assumed free). CUDA is read only
after it is already initialized (with one boundary sync per snapshot); the
diagnostic never creates a CUDA context, never calls empty_cache, never
resets peaks, and never signals any process (the owned encoder reaps its own
children; scripts/process_lineage_owner.py reaps this process's lineage).
Retained or unreadable direct children, an unterminated timer, a cancelled
job that published an image, an unacknowledged cancellation, or a follow-up
without one correctly sized image all refuse the run.

Root run command (lineage owner wraps the owned script process):

  PYTHONPATH=src python -B scripts/process_lineage_owner.py -- \
    python -B scripts/check_generation_recovery.py --config CFG.json \
    --output-dir NEW_DIR --run [--attest-unbounded-host]

Each cancellation scenario now measures the production queue boundary in
place: after the AFTER snapshot (retention with no diagnostic GC yet), it
calls the very same release production uses for FAILED/CANCELLED jobs --
qwen_engine.jobs.release_cyclic_references, the JobController default
boundary collector -- snapshots again (after_boundary), and only then runs
the diagnostic's own explicit gc.collect pass. after -> after_boundary is
what the production release frees on the GPU; after_boundary -> after_gc is
residue a second pass would still find. The follow-up (success) scenario
runs neither, exactly as the production queue never collects for a
success. These are additive report fields (boundary_ms, after_boundary,
retry_delta_boundary); the v1 report schema stays compatible.

Observation only: no acceptance, fit or quality claims. See
docs/generation-recovery-check.md.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time

from qwen_engine.bench import host_rss_bytes
from qwen_engine.config import GenerationConfig
from qwen_engine.decode_bench import allocator_retry_delta
from qwen_engine.errors import EngineError
from qwen_engine.jobs import release_cyclic_references

REPORT_SCHEMA = "qwen_engine.generation_recovery.v1"
MAX_REPORT_BYTES = 256 * 1024
#: Cancelled jobs use 4 steps (root observed NaN on 1-step schedules; 2 steps
#: must complete, so cancellation lands mid-sampling). The follow-up keeps the
#: user's own steps.
CANCEL_STEPS = 4
CANCEL_AFTER_STEPS = 2
#: Diagnostic settle between encoder-worker start and the cancel request.
ENCODE_SETTLE_S = 0.5
#: Hard bound on the encode watchdog poll loop (observation bound only).
ENCODE_WATCHDOG_S = 60.0
ENCODE_POLL_S = 0.2
#: The decode timer arms INSIDE VAE.decode, so a short settle is real work.
DECODE_ARM_S = 0.05
SCENARIOS = (("encode-cancel", "encode"), ("sampling-cancel", "sampling"),
             ("decode-cancel", "decode"), ("followup", "none"))


class RecoveryRefusal(Exception):
    """Stop condition with a fixed prompt-free reason code."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def direct_children() -> list[int] | None:
    """Live direct children of the MAIN thread (None = unreadable/unknown)."""
    pid = os.getpid()
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "rb", buffering=0) as stream:
            return [int(v) for v in stream.read(65537).split() if v.isdigit()]
    except OSError:
        return None


def encode_child_stage() -> str | None:
    """Classify this process's live encoder child from its OWN argv:
    "identity" (identity projection worker), "worker" (encoder worker process
    running — its mere existence proves the process started, not that model
    encoding is progressing) or None. Fixed tokens only — argv paths are
    never returned or recorded."""
    for pid in (direct_children() or []):
        try:
            with open(f"/proc/{pid}/cmdline", "rb", buffering=0) as stream:
                argv = stream.read(8192)
        except OSError:
            continue
        if b"qwen_engine.encoder_worker" in argv:
            return "identity" if b"--project-identity" in argv else "worker"
    return None


class Countdown:
    """One-shot cancellable timer; stop() cancels, joins and reports whether
    the timer thread terminated."""

    def __init__(self, action, delay: float):
        self._action, self._delay = action, delay
        self._timer: threading.Timer | None = None
        self.fired_at: float | None = None

    def arm(self) -> None:
        if self._timer is None:
            self._timer = threading.Timer(self._delay, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        self.fired_at = time.monotonic()
        self._action()

    def stop(self) -> bool:
        if self._timer is None:
            return True
        self._timer.cancel()
        self._timer.join(timeout=5.0)
        return not self._timer.is_alive()


class WorkerWatchdog:
    """Bounded child-only watchdog (same arm/stop/fired_at contract as
    Countdown): polls this process's own children until the actual encoder
    WORKER appears, settles ENCODE_SETTLE_S so the cancel request lands while
    the worker runs, then fires ONCE with the stage re-sampled at fire time.
    If no worker shows up before ENCODE_WATCHDOG_S it fires anyway with an
    explicit stage, so a finished job surfaces as unacknowledged rather than
    silently uncancelled. It never signals anything itself. stop() cancels,
    joins and reports termination."""

    def __init__(self, action, stage_probe, delay: float = ENCODE_SETTLE_S,
                 timeout: float = ENCODE_WATCHDOG_S, interval: float = ENCODE_POLL_S,
                 clock=time.monotonic, wait=None):
        self._action, self._stage, self._clock = action, stage_probe, clock
        self._delay, self._timeout, self._interval = delay, timeout, interval
        self._stopping = threading.Event()
        self._wait = wait if wait is not None else self._stopping.wait
        self._thread: threading.Thread | None = None
        self.fired_at: float | None = None

    def arm(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self.poll, daemon=True)
            self._thread.start()

    def poll(self) -> bool:
        deadline = self._clock() + self._timeout
        while not self._stopping.is_set():
            if self._stage() == "worker":
                if self._wait(self._delay):
                    return False
                self._fire()
                return True
            if self._clock() >= deadline:
                self._fire()
                return True
            if self._wait(self._interval):
                return False
        return False

    def _fire(self) -> None:
        self.fired_at = self._clock()
        self._action()

    def stop(self) -> bool:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return self._thread is None or not self._thread.is_alive()


class TorchProbe:
    """Read-only snapshots. torch is observed only when ALREADY loaded and
    CUDA only once it is ALREADY initialized (one boundary synchronize per
    snapshot); a snapshot never creates the CUDA context."""

    def snapshot(self) -> dict:
        out = dict(allocated_bytes=None, reserved_bytes=None,
                   peak_allocated_bytes=None, peak_reserved_bytes=None,
                   num_alloc_retries=None, host_rss_bytes=None,
                   host_lifetime_peak_rss_bytes=None,
                   children=direct_children(), monotonic_s=None)
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize(0)
                    out.update(
                        allocated_bytes=int(torch.cuda.memory_allocated(0)),
                        reserved_bytes=int(torch.cuda.memory_reserved(0)),
                        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
                        num_alloc_retries=int(torch.cuda.memory_stats(0)["num_alloc_retries"]),
                        monotonic_s=round(time.monotonic(), 3))
            except Exception:
                pass
        try:
            out["host_rss_bytes"] = int(Path("/proc/self/statm").read_text()
                                        .split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
        out["host_lifetime_peak_rss_bytes"] = host_rss_bytes()
        return out

    def child_stage(self) -> str | None:
        return encode_child_stage()

    @staticmethod
    def retry_delta(before, after):
        try:
            return allocator_retry_delta({"num_alloc_retries": before},
                                         {"num_alloc_retries": after})
        except EngineError:
            return None


def _make_decode_observing_adapter_cls():
    """Diagnostic subclass (lives in THIS script; production stays frozen):
    wraps the live VAE's ``decode`` so the cancel timer arms at actual
    VAE.decode function entry, not merely at decode-phase entry. The wrapped
    call removes the instance override and its own closure references in a
    finally, BEFORE any memory measurement: the hook (closure -> adapter ->
    pipeline -> VAE -> bound method -> VAE) would otherwise be misattributed
    to production cycles by the diagnostic gc.collect."""
    from qwen_engine.adapters.qwen21_diffusers import Qwen21DiffusersAdapter

    class DecodeObserving(Qwen21DiffusersAdapter):
        def __init__(self, *, trust_unbounded_host=False):
            super().__init__(trust_unbounded_host=trust_unbounded_host)
            self.decode_arm = None
            self.decode_entered = False
            self.decode_inner_returned_at = None

        def set_decode_arm(self, arm) -> None:
            self.decode_arm = arm

        def _restore_vae(self) -> None:
            super()._restore_vae()
            vae = getattr(self._pipeline, "vae", None) if self._pipeline else None
            if vae is None or self.decode_arm is None or "decode" in vars(vae):
                return
            refs = {"outer": self, "inner": vae.decode, "vae": vae}

            def timed_decode(*args, **kwargs):
                outer = refs["outer"]
                outer.decode_entered = True
                outer.decode_arm()
                try:
                    result = refs["inner"](*args, **kwargs)
                    outer.decode_inner_returned_at = time.monotonic()
                    return result
                finally:
                    # Restore the instance before returning to the job (no
                    # original instance "decode" existed, so removal is
                    # exact) and empty the closure's only container: the hook
                    # must leave no adapter->pipeline->VAE->bound-method
                    # cycle for the diagnostic gc measurement to misattribute.
                    vae_ref = refs["vae"]
                    if "decode" in vars(vae_ref):
                        del vae_ref.decode
                    if "_recovery_probe" in vars(vae_ref):
                        del vae_ref._recovery_probe
                    refs.clear()

            vae.decode = timed_decode
            vae._recovery_probe = True

    return DecodeObserving


def _cancelled_outcome(entry, cancel_phase) -> str:
    if cancel_phase == "encode":
        if entry["encode_child_stage"] == "worker" and entry["encoder_child_cancelled"]:
            return "cancelled-in-encoder-worker"
        if entry["encode_child_stage"] == "identity":
            return "cancelled-in-identity-load"
        return "cancelled-in-encode-window"
    if cancel_phase == "sampling":
        if len(entry["steps_observed"]) >= CANCEL_AFTER_STEPS:
            return f"cancelled-after-step-{CANCEL_AFTER_STEPS}"
        return "cancelled-in-sampling-window"
    if entry["decode_entered"]:
        return ("cancelled-in-vae-decode" if entry["decode_cancel_inside"]
                else "cancelled-in-decode-window")
    return "cancelled-before-decode-body"


def _run_scenario(config, out_dir, name, cancel_phase, *, assets, adapter_factory,
                  probe, timer_factory, logger, image_opener, gc_collect,
                  boundary_release):
    from qwen_engine.core import generate
    from qwen_engine.encoder import EncoderCancelledError
    from qwen_engine.errors import GenerationCancelled

    followup = cancel_phase == "none"
    job_dir, cache_dir = out_dir / f"job-{name}", out_dir / f"cache-{name}"
    job_dir.mkdir()
    cache_dir.mkdir()
    cfg = replace(config, output_dir=str(job_dir), cache_dir=str(cache_dir),
                  steps=config.steps if followup else CANCEL_STEPS)
    event = threading.Event()
    phases: list[str] = []
    steps_observed: list[int] = []
    state = dict(child_stage=None, timer=None)

    def action() -> None:
        # Evidence at fire time (classification of this process's own live
        # encoder child), then the job-scoped cancel request; nothing else is
        # signalled.
        if cancel_phase == "encode":
            state["child_stage"] = probe.child_stage()
        event.set()

    if cancel_phase in ("encode", "decode"):
        delay = ENCODE_SETTLE_S if cancel_phase == "encode" else DECODE_ARM_S
        state["timer"] = timer_factory(cancel_phase, action, delay)

    def phase_hook(p: str) -> None:
        phases.append(p)
        if p == "encode" and cancel_phase == "encode":
            state["timer"].arm()

    def progress(completed, total) -> None:
        steps_observed.append(int(completed))
        if int(completed) >= CANCEL_AFTER_STEPS:
            action()

    adapter = adapter_factory(cancel_phase)
    if cancel_phase == "sampling":
        adapter.set_progress_callback(progress)
    if cancel_phase == "decode":
        bind = getattr(adapter, "set_decode_arm", None)
        if bind is None:
            raise RecoveryRefusal("decode-arm-hook-unavailable")
        bind(lambda: state["timer"].arm())

    before = probe.snapshot()
    primary_error = None
    acknowledged = encoder_child_cancelled = False
    result = None
    timer_terminated = None
    try:
        result = generate(cfg, adapter, assets, logger,
                          cancel_event=(None if followup else event),
                          phase_callback=phase_hook)
    except BaseException as exc:  # noqa: BLE001 - observation, not concealment
        primary_error = type(exc).__name__
        acknowledged = isinstance(exc, GenerationCancelled)
        encoder_child_cancelled = _chain_has(exc, EncoderCancelledError)
    finally:
        if state["timer"] is not None:
            timer_terminated = state["timer"].stop()
    # The handler's reference to the exception is cleared at except-block
    # exit and no external reference is retained: the snapshots below see
    # production retention (e.g. cycles core/adapter leave naturally), not
    # the diagnostic keeping a traceback alive. Frames are not otherwise
    # touched — that would be intervention.

    after = probe.snapshot()
    collected = gc_ms = boundary_ms = None
    after_boundary = None
    if not followup:
        # The production queue boundary first: JobController's default
        # FAILED/CANCELLED collector, invoked exactly where production
        # invokes it (primary fully handled, nothing retained here).
        # The following diagnostic pass then only sees what the production
        # release itself could not reclaim.
        start = time.monotonic()
        boundary_release()
        boundary_ms = round((time.monotonic() - start) * 1000, 1)
        after_boundary = probe.snapshot()
        start = time.monotonic()
        collected = int(gc_collect())
        gc_ms = round((time.monotonic() - start) * 1000, 1)
    after_gc = probe.snapshot() if not followup else None
    published_files = sum(1 for _ in job_dir.iterdir())
    children_after = (after_gc or after)["children"]

    timer = state["timer"]
    fired_at = timer.fired_at if timer is not None else None
    inner_returned = getattr(adapter, "decode_inner_returned_at", None)
    entry = dict(
        name=name, cancel_phase=cancel_phase, phases=phases,
        steps_observed=steps_observed,
        primary_error=primary_error, acknowledged=acknowledged,
        encoder_child_cancelled=encoder_child_cancelled,
        encode_child_stage=state["child_stage"] if cancel_phase == "encode" else None,
        timer_fired=fired_at is not None if timer is not None else None,
        timer_terminated=timer_terminated,
        decode_entered=bool(getattr(adapter, "decode_entered", False)) if cancel_phase == "decode" else None,
        decode_cancel_inside=(fired_at is not None and inner_returned is not None
                              and fired_at <= inner_returned) if cancel_phase == "decode" else None,
        children_after=children_after,
        published_files=published_files,
        gc_collected=collected, gc_ms=gc_ms, boundary_ms=boundary_ms,
        retry_delta_after=probe.retry_delta(before["num_alloc_retries"],
                                           after["num_alloc_retries"]),
        retry_delta_boundary=(probe.retry_delta(after["num_alloc_retries"],
                                                after_boundary["num_alloc_retries"])
                              if after_boundary is not None else None),
        retry_delta_gc=(probe.retry_delta(after_boundary["num_alloc_retries"],
                                          after_gc["num_alloc_retries"])
                        if after_gc is not None else None),
        before=before, after=after, after_boundary=after_boundary, after_gc=after_gc,
        outcome=None, stop=None)

    if followup:
        published = next(iter(job_dir.iterdir()), None) if published_files == 1 else None
        resolution = (image_opener(published)
                      if image_opener is not None and published is not None else None)
        entry["output_resolution"] = resolution
        if (primary_error is not None or published is None or result is None
                or resolution != [cfg.width, cfg.height]):
            entry.update(outcome="failed", stop="followup-failed")
        else:
            entry["outcome"] = "succeeded"
    elif not acknowledged:
        entry.update(outcome="not-cancelled", stop="cancellation-unacknowledged")
    elif published_files:
        entry.update(outcome="image-published", stop="image-published")
    else:
        entry["outcome"] = _cancelled_outcome(entry, cancel_phase)

    if entry["stop"] is None and timer_terminated is False:
        entry["stop"] = "timer-not-terminated"
    if entry["stop"] is None and isinstance(children_after, list) and children_after:
        entry["stop"] = "children-retained"
    if entry["stop"] is None and children_after is None:
        entry["stop"] = "children-unknown"
    return entry


def _chain_has(exc, cls) -> bool:
    seen = 0
    while exc is not None and seen < 16:
        if isinstance(exc, cls):
            return True
        exc = exc.__context__ if exc.__context__ is not exc else None
        seen += 1
    return False


def run_scenarios(config, out_dir, *, assets, adapter_factory, probe, timer_factory,
                  logger=None, image_opener=None, gc_collect=gc.collect,
                  boundary_release=release_cyclic_references):
    """Shared production/test seam: serial scenarios with injected
    adapter/probe/timer collaborators. Writes the bounded prompt-free report
    FIRST, then raises RecoveryRefusal when a stop condition fired."""
    out_dir = Path(out_dir)
    report = dict(schema=REPORT_SCHEMA, run=True, seed=config.seed,
                  width=config.width, height=config.height,
                  quant=config.quant.value, cancelled_steps=CANCEL_STEPS,
                  followup_steps=config.steps, scenarios=[], stop=None,
                  children_final=None,
                  scope=("real cancellation/recovery observations only; CUDA "
                         "allocator counters after initialize/sync only, /proc "
                         "lineage; encoder-worker argv labels process presence, "
                         "not encoding progress; no empty_cache; no "
                         "acceptance/fit/quality claim"))
    for name, cancel_phase in SCENARIOS:
        entry = _run_scenario(config, out_dir, name, cancel_phase, assets=assets,
                              adapter_factory=adapter_factory, probe=probe,
                              timer_factory=timer_factory, logger=logger,
                              image_opener=image_opener, gc_collect=gc_collect,
                              boundary_release=boundary_release)
        report["scenarios"].append(entry)
        if entry["stop"] is not None:
            report["stop"] = entry["stop"]
            break
    report["children_final"] = direct_children()
    _write_report(out_dir, report)
    if report["stop"] is not None:
        raise RecoveryRefusal(report["stop"])
    return report


def _write_report(out_dir: Path, report: dict) -> Path:
    text = json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
    if len(text.encode("utf-8")) > MAX_REPORT_BYTES:
        raise RecoveryRefusal("report-unbounded")
    path = out_dir / "recovery-report.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _open_image_resolution(path):
    try:
        from PIL import Image
        with Image.open(path) as image:
            return [int(image.size[0]), int(image.size[1])]
    except Exception:
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="existing CLI GenerationConfig JSON")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new benchmark directory (must not already exist)")
    parser.add_argument("--attest-unbounded-host", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = GenerationConfig.from_json(args.config.read_text())
        if config.seed is None:
            raise ValueError("a fixed seed is required for recovery comparison")
        if args.output_dir.exists():
            raise ValueError("benchmark output directory must be new")
        if not args.run:
            # Plan only: no ML import, no GPU, no writes, no prompt/path echo.
            print(json.dumps(dict(run=False, scenarios=[n for n, _ in SCENARIOS],
                                  cancelled_steps=CANCEL_STEPS,
                                  followup_steps=config.steps, seed=config.seed,
                                  width=config.width, height=config.height,
                                  quant=config.quant.value,
                                  lineage_wrap="scripts/process_lineage_owner.py",
                                  boundary_release="qwen_engine.jobs.release_cyclic_references"),
                             sort_keys=True))
            return 0
        from qwen_engine.cli import _make_adapter, _verify_assets
        from qwen_engine.log import PhaseLogger, file_sink

        assets = _verify_assets(config)  # the CLI's own no-download gate
        args.output_dir.mkdir(parents=True, exist_ok=False)
        logger = PhaseLogger(sink=file_sink(args.output_dir / "engine.jsonl"))
        probe = TorchProbe()
        decode_cls = _make_decode_observing_adapter_cls()

        def adapter_factory(cancel_phase):
            if cancel_phase == "decode":
                return decode_cls(trust_unbounded_host=args.attest_unbounded_host)
            return _make_adapter(trust_unbounded_host=args.attest_unbounded_host)

        def timer_factory(cancel_phase, action, delay):
            if cancel_phase == "encode":
                return WorkerWatchdog(action, probe.child_stage, delay=delay)
            return Countdown(action, delay)

        run_scenarios(config, args.output_dir, assets=assets.resolved,
                      adapter_factory=adapter_factory, probe=probe,
                      timer_factory=timer_factory, logger=logger,
                      image_opener=_open_image_resolution)
        print(json.dumps(dict(run=True, completed=[n for n, _ in SCENARIOS])))
        return 0
    except RecoveryRefusal as exc:
        print(f"recovery check stopped: {exc.reason}", file=sys.stderr)
        return 1
    except Exception as exc:  # never echo config/prompt contents
        print("recovery check refused or failed: " + type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
