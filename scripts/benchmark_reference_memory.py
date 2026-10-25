#!/usr/bin/env python3
"""Reference-conditioned memory matrix (design §4); no GPU work without --run.

Default mode prints a prompt-free PLAN of the matrix cells and measures
nothing. Each ``--run`` cell executes in a FRESH process (the parent only
orchestrates): one ``qwen_engine`` generation per cell under a
``decode_bench.PeakMemoryProbe`` (whole-run NVML + allocator + retries), a
``PerformanceRecorder`` report, per-step progress timestamps, and a fresh
allocator/NVML window around the adapter's ``reference_vae_encode`` stage
(peak reset on entry — the same measurement technique the in-engine
CalibrationCollector uses between phases; no ``empty_cache`` mid-run).

Run through the lineage wrapper so an interrupt reaps the whole tree:

    python -B scripts/process_lineage_owner.py -- \
        python -B scripts/benchmark_reference_memory.py --run ...

Each cell ingests its reference images into a private content store under
the output directory (``reference_store``), so ids in the reports are the
byte-exact images measured. OOM is a RECORDED cell outcome, never a matrix
crash. Reports are bounded, prompt-free JSON; only ids, numbers, phases and
fixed status/type names are recorded — never prompts, paths or argv.

Honest limitations: NVML peaks are device-wide interval samples (sub-interval
spikes can be missed; ``samples``/``interval_ms`` state the coverage); the
KV-cache knob is applied to the attached bridge adapter before denoise and
cross-checked against the adapter's measured facts, a mismatch invalidates
the cell; nothing in this file is a performance claim — root reads the data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_engine.config import ConfigError, GenerationConfig  # noqa: E402
from qwen_engine.decode_bench import (  # noqa: E402
    make_memory_observation, utc_now,
)
from qwen_engine.errors import EngineError  # noqa: E402
from qwen_engine import reference_store  # noqa: E402

MAX_REPORT_BYTES = 256 * 1024
VAE_STAGE = "reference_vae_encode"
NVML_INTERVAL_MS = 50.0


def parse_counts(raw: str) -> tuple[int, ...]:
    values = sorted({int(part) for part in raw.split(",") if part.strip() != ""})
    if not values or any(value < 0 or value > 16 for value in values):
        raise ValueError("counts must be a comma list of 0..16, e.g. 0,1,2,4")
    return tuple(values)


def parse_kv(raw: str) -> tuple[bool, ...]:
    seen: list[bool] = []
    for part in raw.split(","):
        part = part.strip()
        if part in ("on", "off") and (part == "on") not in seen:
            seen.append(part == "on")
    if not seen:
        raise ValueError("kv-cache must list at least one of on/off")
    return tuple(seen)


def _sha256_files(paths: list[Path]) -> tuple[str, ...]:
    ids = []
    for path in paths:
        raw = path.read_bytes()
        ids.append(hashlib.sha256(raw).hexdigest())
    return tuple(ids)


def build_plan(args, config: GenerationConfig, reference_ids: tuple[str, ...]):
    """Deterministic cell list: reference count x KV-cache policy. Counts
    beyond the supplied files cycle them (duplicates are legal by design)."""
    cells = []
    index = 0
    for kv_cache in parse_kv(args.kv_cache):
        for count in parse_counts(args.counts):
            if count and not reference_ids:
                raise ValueError("--counts above zero need --reference files")
            ids = tuple(reference_ids[position % len(reference_ids)]
                        for position in range(count)) if count else ()
            cells.append({"index": index, "references": count,
                          "kv_cache": kv_cache, "reference_ids": list(ids)})
            index += 1
    return {"run": False, "cells": cells, "seed": config.seed,
            "width": config.width, "height": config.height, "steps": config.steps,
            "quant": config.quant.value, "reference_ids": list(reference_ids),
            "interval_ms": args.interval_ms,
            "note": "plan only; run with --run through scripts/process_lineage_owner.py"}


def _write_json_exclusive(path: Path, document) -> None:
    text = json.dumps(document, sort_keys=True, allow_nan=False).encode()
    if len(text) > MAX_REPORT_BYTES:
        raise EngineError("report exceeded the 256 KiB bound")
    with path.open("xb") as handle:
        handle.write(text)


# ---------------------------------------------------------------------------
# cell execution (the only place torch may be imported; reached via --run)
# ---------------------------------------------------------------------------


class _StageWindow:
    """Wraps the adapter module's ``stage`` to measure one named stage.

    Observational wrapper: every other stage delegates untouched. On the
    watched stage it takes allocator peak stats with a fresh peak baseline
    (reset on entry, like CalibrationCollector phases — never empty_cache),
    a delta of ``num_alloc_retries``, wall time, and an NVML device-wide
    sampled peak from a SECOND short-lived NVML session; if NVML is
    unavailable the window degrades to allocator-only and says so.
    """

    def __init__(self, *, torch, interval_ms: float, clock=time.monotonic):
        self._torch = torch
        self._interval_s = interval_ms / 1000.0
        self._clock = clock
        self.window: dict | None = None
        self._nvml_error: str | None = None

    def install(self, stage_host) -> None:
        self._real = stage_host.stage
        stage_host.stage = self

    def uninstall(self, stage_host) -> None:
        stage_host.stage = self._real

    def __call__(self, name, *, cuda=False):
        if name != VAE_STAGE:
            return self._real(name, cuda=cuda)
        return self._window(name, cuda=cuda)

    class _NvmlSampler(threading.Thread):
        def __init__(self, device, interval_s):
            super().__init__(daemon=True)
            self.device, self.interval_s = device, interval_s
            self.stop_event = threading.Event()
            self.peak = None
            self.samples = 0
            self.error: Exception | None = None

        def run(self):
            while not self.stop_event.is_set():
                try:
                    used = self.device.used_bytes()
                except Exception as exc:  # noqa: BLE001 - recorded, not raised
                    self.error = type(exc).__name__
                    return
                if self.peak is None or used > self.peak:
                    self.peak = used
                self.samples += 1
                self.stop_event.wait(self.interval_s)

    def _open_nvml(self):
        try:
            from qwen_engine.calibration import _NvmlDevice, _uuid
            properties = self._torch.cuda.get_device_properties(0)
            device = _NvmlDevice(_uuid(str(properties.uuid)),
                                 int(properties.total_memory))
            if type(device.used_bytes()) is not int:
                device.close()
                self._nvml_error = "nvml_used_bytes_non_int"
                return None
            return device
        except Exception:  # noqa: BLE001 - degrade honestly
            self._nvml_error = "nvml_unavailable"
            return None

    @contextmanager
    def _window(self, name, *, cuda):
        inner = self._real(name, cuda=cuda)
        context = self._torch.cuda
        with inner:
            context.synchronize()
            retries_before = int(context.memory_stats()["num_alloc_retries"])
            context.reset_peak_memory_stats()
            nvml = self._open_nvml()
            sampler = (_StageWindow._NvmlSampler(nvml, self._interval_s)
                       if nvml is not None else None)
            if sampler is not None:
                sampler.start()
            start = self._clock()
            outcome = "succeeded"
            try:
                yield
            except BaseException:
                outcome = "failed"
                raise
            finally:
                wall_s = max(0.0, self._clock() - start)
                try:
                    context.synchronize()
                    if sampler is not None:
                        sampler.stop_event.set()
                        sampler.join(timeout=5.0)
                    allocated = int(context.max_memory_allocated())
                    reserved = int(context.max_memory_reserved())
                    retries = int(context.memory_stats()["num_alloc_retries"])
                    observation = make_memory_observation(
                        peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                        interval_ms=self._interval_s * 1000.0,
                        samples=sampler.samples if sampler is not None else 0,
                        allocator_retries=retries - retries_before,
                        nvml_peak_used_bytes=sampler.peak if sampler is not None else None,
                        nvml_total_bytes=nvml.total_bytes if nvml is not None else None,
                        nvml_driver_version=nvml.driver_version if nvml is not None else None,
                    )
                    self.window = {"stage": name, "status": outcome,
                                   "wall_s": wall_s, **observation.to_dict()}
                    if self._nvml_error is not None:
                        self.window["nvml_note"] = self._nvml_error
                    if sampler is not None and sampler.error is not None:
                        self.window["nvml_note"] = "nvml_sampling_failed_mid_window"
                finally:
                    if nvml is not None:
                        nvml.close()


class CellRunner:
    """Adapter subclass applying the cell's KV knob and capturing facts.

    The knob is set on the attached bridge adapter right before its denoise
    (the point where the value is read); the adapter's own measured
    ``calibration_facts`` are captured at the same instant, so the recorded
    ``use_kv_cache`` is the value that actually executed, never a guess.
    """

    def __init__(self, inner_adapter, *, kv_cache: bool, clock=time.monotonic):
        self._inner = inner_adapter
        self._kv_cache = kv_cache
        self._clock = clock
        self.facts: dict | None = None
        self.step_timestamps: list[float] = []
        self._progress = None

    # -- protocol delegation ------------------------------------------------
    def __getattr__(self, item):
        return getattr(self._inner, item)

    def set_cancel_event(self, event):
        self._inner.set_cancel_event(event)

    def set_progress_callback(self, callback):
        def timed(step, total):
            self.step_timestamps.append(self._clock())
            if callback is not None:
                callback(step, total)
        self._progress = timed
        self._inner.set_progress_callback(timed)

    def load(self, config, assets):
        return self._inner.load(config, assets)

    def encode(self, config):
        return self._inner.encode(config)

    def denoise(self, config, conditioning):
        self._inner.use_kv_cache = self._kv_cache
        self.facts = self._inner.calibration_facts()
        if self.facts.get("use_kv_cache") is not self._kv_cache:
            raise EngineError("kv-cache knob did not reach the executing adapter")
        return self._inner.denoise(config, conditioning)

    def decode(self, latents):
        return self._inner.decode(latents)

    def save(self, image, config):
        return self._inner.save(image, config)

    def close(self):
        self._inner.close()


def default_adapter_factory(*, kv_cache, attest_unbounded_host):
    from qwen_engine.adapters.qwen21_diffusers import Qwen21DiffusersAdapter
    from qwen_engine.cli import _make_adapter
    inner = _make_adapter(trust_unbounded_host=attest_unbounded_host)
    return CellRunner(inner, kv_cache=kv_cache)


def default_stage_host():
    import qwen_engine.adapters.qwen21_diffusers as host
    return host


def default_probe_factory(torch_module):
    from qwen_engine.decode_bench import PeakMemoryProbe
    from qwen_engine.calibration import _uuid
    properties = torch_module.cuda.get_device_properties(0)
    return PeakMemoryProbe(device_uuid=_uuid(str(properties.uuid)),
                           device_total_bytes=int(properties.total_memory),
                           interval_ms=NVML_INTERVAL_MS, torch_module=torch_module)


def execute_cell(spec, *, assets_verifier, adapter_factory, probe_factory,
                 stage_host_getter, torch_module, recorder_factory, clock=time.monotonic):
    """Run one matrix cell end to end and return its record dict."""
    from qwen_engine.core import generate
    from qwen_engine.log import PhaseLogger

    config = GenerationConfig.from_json(Path(spec["config_path"]).read_text())
    config = replace(config,
                     references=tuple(spec["reference_ids"]),
                     references_dir=spec["references_dir"],
                     output_dir=str(Path(spec["output_dir"]).resolve()))
    Path(spec["output_dir"]).mkdir(parents=True, exist_ok=True)
    assets = assets_verifier(config)
    adapter = adapter_factory(kv_cache=bool(spec["kv_cache"]),
                              attest_unbounded_host=bool(spec.get("attest")))
    probe = probe_factory(torch_module)
    stage_host = stage_host_getter()
    window = _StageWindow(torch=torch_module, interval_ms=float(spec["interval_ms"]),
                          clock=clock)
    phases: list[str] = []
    started = clock()
    status, error_type = "succeeded", None
    probe.start()
    window.install(stage_host)
    try:
        with recorder_factory(config) as recorder:
            with recorder.stage("total_job"):
                adapter.set_progress_callback(lambda step, total: None)
                generate(config, adapter, assets, PhaseLogger(),
                         phase_callback=phases.append)
    except BaseException as exc:
        status, error_type = "failed", type(exc).__name__
    finally:
        try:
            sampling = probe.finish()
        except Exception as exc:  # noqa: BLE001 - probe failure is a note, not a crash
            sampling = {"probe_error": type(exc).__name__}
        window.uninstall(stage_host)
        try:
            adapter.close()
        except Exception:  # noqa: BLE001 - close must not mask the cell outcome
            pass
    facts = getattr(adapter, "facts", None) or {}
    stamps = getattr(adapter, "step_timestamps", [])
    deltas = [round((b - a) * 1000.0, 3) for a, b in zip(stamps, stamps[1:])]
    return {
        "index": spec["index"], "references": spec["references"],
        "kv_cache": bool(spec["kv_cache"]), "status": status,
        "error_type": error_type, "phases": phases,
        "wall_s": max(0.0, clock() - started),
        "seed": config.seed, "resolution": [config.width, config.height],
        "steps": config.steps, "quant": config.quant.value,
        "reference_ids": list(spec["reference_ids"]),
        "step_ms": deltas,
        "prompt_tokens": facts.get("prompt_tokens"),
        "joint_tokens": facts.get("joint_tokens"),
        "embedding_cache_hit": facts.get("embedding_cache_hit"),
        "use_kv_cache": facts.get("use_kv_cache"),
        "vae_encode": window.window,
        "sampling": sampling.to_dict() if hasattr(sampling, "to_dict") else sampling,
        "observed_at": utc_now(),
    }


def _cell_main(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    import torch

    from qwen_engine.cli import _verify_assets
    from qwen_engine.performance import PerformanceRecorder

    def recorder_factory(config):
        recorder = PerformanceRecorder(
            config, human_sink=lambda line: print(line, file=sys.stderr))
        return _RecorderHandle(recorder, Path(spec["perf_path"]))

    record = execute_cell(
        spec, assets_verifier=_verify_assets,
        adapter_factory=default_adapter_factory,
        probe_factory=default_probe_factory, stage_host_getter=default_stage_host,
        torch_module=torch, recorder_factory=recorder_factory)
    _write_json_exclusive(Path(spec["cell_path"]), record)
    return 0 if record["status"] == "succeeded" else 1


class _RecorderHandle:
    """PerformanceRecorder activation + exclusive report file handle."""

    def __init__(self, recorder, path: Path):
        self._recorder, self._path = recorder, path
        self._handle = None
        self._token = None

    def __enter__(self):
        self._handle = self._path.open("x", encoding="utf-8")
        self._token = self._recorder.activate()
        self._token.__enter__()
        return self._recorder

    def __exit__(self, *exc):
        try:
            self._recorder.write(self._handle,
                                 "failed" if exc[0] is not None else "succeeded")
        finally:
            self._token.__exit__(*exc)
            self._handle.close()
        return False


# ---------------------------------------------------------------------------
# parent: ingest, spawn, aggregate
# ---------------------------------------------------------------------------


def ingest(output_dir: Path, reference_files: list[Path]) -> tuple[str, tuple[str, ...]]:
    store = output_dir / "store"
    store.mkdir(parents=True, exist_ok=False)
    ids = []
    for path in reference_files:
        ids.append(reference_store.put_bytes(store, path.read_bytes()))
    return str(store), tuple(ids)


def default_spawn(command, log_path: Path):
    with log_path.open("x") as log:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            return proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def process_cells(output_dir: Path, plan, store_dir: str, config_path: Path, *,
                  interval_ms: float, attest: bool, spawn=default_spawn):
    cells = []
    for cell in plan["cells"]:
        directory = output_dir / f"cell-{cell['index']}"
        directory.mkdir()
        spec_path = output_dir / f"cell-{cell['index']}.spec.json"
        cell_path = output_dir / f"cell-{cell['index']}.json"
        spec = {**cell, "config_path": str(config_path.resolve()),
                "references_dir": store_dir, "output_dir": str(directory),
                "interval_ms": interval_ms, "attest": attest,
                "perf_path": str(output_dir / f"cell-{cell['index']}.perf.json"),
                "cell_path": str(cell_path)}
        _write_json_exclusive(spec_path, spec)
        code = spawn([sys.executable, "-B", str(Path(__file__).resolve()),
                      "--cell-spec", str(spec_path)],
                     output_dir / f"cell-{cell['index']}.log")
        if cell_path.exists():
            record = json.loads(cell_path.read_text())
        else:
            record = {"index": cell["index"], "references": cell["references"],
                      "kv_cache": cell["kv_cache"], "status": "crashed",
                      "error_type": f"exit_code_{code}"}
        record["exit_code"] = code
        cells.append(record)
    return cells


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--reference", type=Path, action="append", default=[],
                        help="reference image file (repeatable; ingested into a "
                             "content store under --output-dir)")
    parser.add_argument("--counts", default="0,1,2,4")
    parser.add_argument("--kv-cache", default="on,off")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="new matrix directory (must not already exist)")
    parser.add_argument("--interval-ms", type=float, default=NVML_INTERVAL_MS)
    parser.add_argument("--attest-unbounded-host", action="store_true")
    parser.add_argument("--cell-spec", type=Path, default=None,
                        help=argparse.SUPPRESS)  # internal: fresh-process cell
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.cell_spec is not None:
        if not args.run:
            print("cell mode requires --run", file=sys.stderr)
            return 1
        return _cell_main(args.cell_spec)
    try:
        if args.config is None:
            raise ValueError("--config is required outside cell mode")
        if args.output_dir is None:
            raise ValueError("--output-dir is required outside cell mode")
        config = GenerationConfig.from_json(args.config.read_text())
        if config.seed is None:
            raise ValueError("a fixed seed is required for reproducible measurements")
        if args.output_dir.exists():
            raise ValueError("matrix output directory must be new")
        counts, kv = parse_counts(args.counts), parse_kv(args.kv_cache)
        args.counts, args.kv_cache = ",".join(map(str, counts)), ",".join(
            "on" if value else "off" for value in kv)
        if any(count > 0 for count in counts) and not args.reference:
            raise ValueError("counts above zero need --reference files")
        reference_ids = _sha256_files(args.reference)
        if not args.run:
            plan = build_plan(args, config, reference_ids)
            print(json.dumps(plan, sort_keys=True, allow_nan=False))
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=False)
        store_dir, run_ids = ingest(args.output_dir, args.reference)
        if run_ids != reference_ids:
            raise EngineError("reference content changed between plan and ingest")
        plan = build_plan(args, config, run_ids)
        _write_json_exclusive(args.output_dir / "plan.json", plan)
        cells = process_cells(args.output_dir, plan, store_dir, args.config,
                              interval_ms=args.interval_ms,
                              attest=args.attest_unbounded_host)
        _write_json_exclusive(args.output_dir / "matrix.json", {
            "cells": cells, "seed": config.seed,
            "resolution": [config.width, config.height], "steps": config.steps,
            "quant": config.quant.value, "reference_ids": list(run_ids),
            "interval_ms": args.interval_ms, "observed_at": utc_now()})
        for record in cells:
            print(f"cell {record['index']}: refs={record['references']} "
                  f"kv={'on' if record['kv_cache'] else 'off'} {record['status']}")
        print(f"matrix complete in {args.output_dir}; failures are recorded per cell")
        return 0
    except (ConfigError, ValueError, OSError, RuntimeError, EngineError) as exc:
        # Never echo config/prompt contents from failures; refuse by type only.
        print("reference matrix refused or failed: " + type(exc).__name__,
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
