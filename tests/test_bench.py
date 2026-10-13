"""M1.5 bench schema tests — synthetic runs only, no fabricated measurements.

Covers: benchmark_result.v1 round-trip + field-level validation, CUDA-null
semantics, deterministic aggregation, phase collection, the
calibration_observation.v1 schema with its staleness key, and the
no-heavy-imports boundary.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

import qwen_engine.bench as bm

from qwen_engine.bench import (
    AGGREGATE_SCHEMA,
    BENCHMARK_SCHEMA,
    CALIBRATION_FILE_SCHEMA,
    OBSERVATION_SCHEMA,
    BenchmarkEnvironment,
    BenchmarkResult,
    BenchSchemaError,
    CalibrationLog,
    CalibrationObservation,
    PhaseCollector,
    add_observation,
    aggregate_results,
    capture_environment,
    host_rss_bytes,
    load_calibration,
    load_result,
    matching_observations,
    model_identifiers,
    save_calibration,
    save_result,
)
from qwen_engine.manifest import ModelArtifact, ModelManifest

HEAVY_MODULES = ("torch", "diffusers", "transformers")

REV = "a" * 40
SHA = "b" * 64


@pytest.mark.parametrize("loader", [load_result, load_calibration])
@pytest.mark.parametrize("kind", ["fifo", "directory", "oversized"])
def test_bench_inputs_are_bounded_regular_without_blocking(tmp_path, monkeypatch, loader, kind):
    monkeypatch.setattr(bm, "MAX_BENCH_FILE_BYTES", 64)
    path = tmp_path / "input.json"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b" " * 65)
    with pytest.raises(BenchSchemaError, match="bounded regular"):
        loader(path)


@pytest.mark.parametrize("kind", ["result", "calibration"])
def test_duplicate_bench_json_is_typed_refusal(tmp_path, kind):
    if kind == "result":
        raw = '{"schema":"wrong",' + result().to_json()[1:]
        with pytest.raises(BenchSchemaError, match="duplicate"):
            BenchmarkResult.from_json(raw)
        loader = load_result
    else:
        raw = '{"schema":"wrong",' + CalibrationLog().to_json()[1:]
        loader = load_calibration
    path = tmp_path / "input.json"
    path.write_text(raw)
    with pytest.raises(BenchSchemaError, match="duplicate"):
        loader(path)


@pytest.mark.parametrize("kind", ["result", "calibration"])
def test_bench_write_cap_refuses_before_filesystem(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(bm, "MAX_BENCH_FILE_BYTES", 1)
    target = tmp_path / "uncreated"
    with pytest.raises(BenchSchemaError, match="byte cap"):
        if kind == "result":
            save_result(result(), target)
        else:
            save_calibration(CalibrationLog(), target / "calibration.json")
    assert not target.exists()


@pytest.mark.parametrize("kind", ["result", "calibration"])
def test_bench_exact_write_read_cap_and_stable_symlink(tmp_path, monkeypatch, kind):
    path = save_result(result(), tmp_path) if kind == "result" else save_calibration(CalibrationLog(), tmp_path / "c.json")
    loader = load_result if kind == "result" else load_calibration
    monkeypatch.setattr(bm, "MAX_BENCH_FILE_BYTES", path.stat().st_size)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    assert loader(link) == loader(path)
    monkeypatch.setattr(bm, "MAX_BENCH_FILE_BYTES", path.stat().st_size - 1)
    with pytest.raises(BenchSchemaError):
        loader(path)


@pytest.mark.parametrize("kind", ["result", "calibration"])
def test_bench_setup_failure_closes_owned_descriptor(tmp_path, monkeypatch, kind):
    primary = OSError("fdopen setup failed")
    def fail(*a, **kw):
        raise primary
    monkeypatch.setattr(bm.os, "fdopen", fail)
    before = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(OSError) as caught:
        if kind == "result":
            save_result(result(), tmp_path)
        else:
            save_calibration(CalibrationLog(), tmp_path / "c.json")
    assert caught.value is primary
    assert len(list(Path("/proc/self/fd").iterdir())) == before
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["result", "calibration"])
def test_bench_cleanup_failure_does_not_mask_publish_failure(tmp_path, monkeypatch, kind):
    primary = OSError("publish failed")
    def fail_publish(*a):
        raise primary
    def fail_cleanup(*a, **kw):
        raise PermissionError("cleanup failed")
    monkeypatch.setattr(bm.os, "replace", fail_publish)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(OSError) as caught:
        if kind == "result":
            save_result(result(), tmp_path)
        else:
            save_calibration(CalibrationLog(), tmp_path / "c.json")
    assert caught.value is primary


def test_phase_median_does_not_overflow_finite_values():
    agg = aggregate_results([result("a", phases={"load": 1e308}), result("b", phases={"load": 1e308})])
    assert agg["phases"]["load"]["median"] == 1e308
    json.dumps(agg, allow_nan=False)


@pytest.mark.parametrize("value", [2**53 + 1, 10**400])
def test_memory_median_uses_exact_integer_arithmetic(value):
    agg = aggregate_results([result("a", peak_host_rss_bytes=value), result("b", peak_host_rss_bytes=value + 2)])
    assert agg["peak_host_rss_bytes"]["median"] == value + 1


def test_large_positive_steps_do_not_overflow_derived_timing():
    measured = result(phases={"denoise": 1e308}, steps=10**400)
    assert math.isfinite(measured.denoise_ms_per_step)
    assert measured.denoise_ms_per_step == pytest.approx(1e-92, rel=1e-14, abs=0)


@pytest.mark.parametrize("point", ["start", "end"])
@pytest.mark.parametrize("body_error", [False, True])
def test_collector_clock_fault_never_aborts_body_or_masks_failure(point, body_error):
    calls = 0
    def clock():
        nonlocal calls
        calls += 1
        if calls == (1 if point == "start" else 2):
            raise OSError("clock failed")
        return 0.0
    collector = PhaseCollector(clock=clock)
    primary = RuntimeError("model failed")
    def body():
        with collector.phase("denoise"):
            if body_error:
                raise primary
    if body_error:
        with pytest.raises(RuntimeError) as caught:
            body()
        assert caught.value is primary
    else:
        body()
    assert collector.timings() == {} and collector.unmeasured_phases() == ("denoise",)


def test_repeated_phase_with_missing_timing_never_reports_partial_total():
    ticks = iter((0.0, 1.0, 2.0, float("nan"), 3.0, 4.0))
    collector = PhaseCollector(clock=lambda: next(ticks))
    for _ in range(3):
        with collector.phase("denoise"):
            pass
    assert collector.timings() == {} and collector.unmeasured_phases() == ("denoise",)


@pytest.mark.parametrize("ticks", [(1.0, 0.0), (0.0, float("inf")), (-1e308, 1e308)])
def test_invalid_or_overflowing_clock_durations_are_unmeasured(ticks):
    values = iter(ticks)
    collector = PhaseCollector(clock=lambda: next(values))
    with collector.phase("load"):
        pass
    assert collector.timings() == {} and collector.unmeasured_phases() == ("load",)


def test_calibration_append_alias_preserves_link_and_updates_canonical_target(tmp_path):
    real = save_calibration(CalibrationLog(), tmp_path / "real.json")
    alias = tmp_path / "alias.json"
    alias.symlink_to(real)
    first, second = observation(observed_at=1), observation(observed_at=2)
    add_observation(alias, first)
    add_observation(real, second)
    assert alias.is_symlink()
    assert load_calibration(alias).observations == (first, second) == load_calibration(real).observations


def test_calibration_append_io_cannot_follow_a_later_cwd_change(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    lock_for = bm._calibration_lock
    def moved_after_lock(path):
        lock = lock_for(path)
        monkeypatch.chdir(second)
        return lock
    monkeypatch.setattr(bm, "_calibration_lock", moved_after_lock)
    add_observation("calibration.json", observation())
    assert (first / "calibration.json").is_file() and not (second / "calibration.json").exists()


@pytest.mark.parametrize("kind", ["environment", "result", "model", "observation", "calibration", "cuda-info"])
def test_bench_mixed_unknown_mapping_keys_are_typed(kind):
    if kind == "environment":
        data, action = env().to_dict(), BenchmarkEnvironment.from_dict
    elif kind == "result":
        data, action = result().to_dict(), BenchmarkResult.from_dict
    elif kind == "model":
        data = model_entry()
        action = lambda entry: result(model=(entry,))
    elif kind == "observation":
        data, action = observation().to_dict(), CalibrationObservation.from_dict
    elif kind == "calibration":
        data, action = CalibrationLog().to_dict(), CalibrationLog.from_dict
    else:
        data, action = {}, capture_environment
    data[1], data[None] = "wrong", "wrong"
    with pytest.raises(BenchSchemaError):
        action(data)


def env(**overrides) -> BenchmarkEnvironment:
    base = dict(python="3.13.5", platform="Linux-6.0-x86_64",
                hostname="testbox")
    base.update(overrides)
    return BenchmarkEnvironment(**base)


def model_entry(**overrides) -> dict:
    entry = {"role": "dit", "repo": "Qwen/Qwen-Image-2.1", "revision": REV,
             "filename": "transformer/diffusion_pytorch_model.safetensors",
             "sha256": SHA}
    entry.update(overrides)
    return entry


def result(run_id: str = "run-1", **overrides) -> BenchmarkResult:
    base = dict(
        run_id=run_id,
        environment=env(),
        model=(model_entry(),),
        config={"prompt": "a red cube", "steps": 40, "width": 1024,
                "height": 1024, "seed": 0},
        phases={"load": 1000.0, "encode": 500.0, "denoise": 40000.0,
                "decode": 800.0, "save": 50.0},
        steps=40,
        peak_host_rss_bytes=2_000_000_000,
    )
    base.update(overrides)
    return BenchmarkResult(**base)


def observation(**overrides) -> CalibrationObservation:
    base = dict(observed_at=1_700_000_000.0, phase="denoise_step",
                model_revision=REV, backend="sdpa", quant="fp16",
                residency="resident", tokens=4096, resolution=(1024, 1024),
                peak_gpu_allocated_bytes=14_000_000_000,
                peak_host_rss_bytes=9_000_000_000)
    base.update(overrides)
    return CalibrationObservation(**base)


# ---------------------------------------------------------------------------
# benchmark_result.v1
# ---------------------------------------------------------------------------


def test_schema_tags():
    assert BENCHMARK_SCHEMA == "benchmark_result.v1"
    assert AGGREGATE_SCHEMA == "benchmark_aggregate.v1"
    assert OBSERVATION_SCHEMA == "calibration_observation.v1"
    assert CALIBRATION_FILE_SCHEMA == "qwen_engine.calibration.v1"


def test_result_roundtrip_is_lossless_and_stable():
    r = result()
    assert BenchmarkResult.from_dict(r.to_dict()) == r
    assert r.to_json() == BenchmarkResult.from_json(r.to_json()).to_json()
    doc = json.loads(r.to_json())
    assert doc["schema"] == BENCHMARK_SCHEMA


def test_cuda_fields_stay_null_when_unavailable():
    r = result()  # no CUDA on this machine: recorded null, never fabricated
    doc = r.to_dict()
    assert doc["peak_cuda_allocated_bytes"] is None
    assert doc["peak_cuda_reserved_bytes"] is None
    assert doc["environment"]["torch"] is None
    assert doc["environment"]["cuda"] is None
    assert doc["environment"]["arch_list"] is None
    back = BenchmarkResult.from_dict(doc)
    assert back.peak_cuda_allocated_bytes is None
    assert back.environment.torch is None


def test_denoise_ms_per_step_is_derived_not_stored():
    r = result()
    assert r.denoise_ms_per_step == pytest.approx(1000.0)
    assert "denoise_ms_per_step" not in r.to_dict()
    assert result(steps=None).denoise_ms_per_step is None


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"schema": "benchmark_result.v0"}, "schema"),
        ({"run_id": "  "}, "run_id"),
        ({"environment": None}, "environment"),
        ({"model": ()}, "model"),
        ({"model": (model_entry(revision="main"),)}, "revision"),
        ({"model": (model_entry(sha256="nope"),)}, "sha256"),
        ({"model": (model_entry(filename=""),)}, "filename"),
        ({"phases": {}}, "phases"),
        ({"phases": {"load": -1.0}}, "phases.load"),
        ({"phases": {"load": float("nan")}}, "phases.load"),
        ({"phases": {"load": True}}, "phases.load"),
        ({"phases": {"": 1.0}}, "phases"),
        ({"steps": 0}, "steps"),
        ({"steps": True}, "steps"),
        ({"peak_host_rss_bytes": -5}, "peak_host_rss_bytes"),
        ({"peak_host_rss_bytes": 1.5}, "peak_host_rss_bytes"),
        ({"peak_cuda_reserved_bytes": "12GB"}, "peak_cuda_reserved_bytes"),
    ],
)
def test_result_validation_names_the_field(overrides, field):
    with pytest.raises(BenchSchemaError) as exc:
        result(**overrides)
    assert field in str(exc.value)


def test_result_from_dict_rejects_unknown_keys_and_bad_json():
    doc = result().to_dict()
    doc["peak_vram"] = 99
    with pytest.raises(BenchSchemaError, match="peak_vram"):
        BenchmarkResult.from_dict(doc)
    doc = result().to_dict()
    doc["environment"]["nvidia_smi"] = "lied"
    with pytest.raises(BenchSchemaError, match="nvidia_smi"):
        BenchmarkResult.from_dict(doc)
    with pytest.raises(BenchSchemaError, match="not valid JSON"):
        BenchmarkResult.from_json("{nope")
    with pytest.raises(BenchSchemaError, match="config"):
        result(config={"bad": {1, 2}})


def test_capture_environment_never_fabricates_gpu_fields():
    e = capture_environment()
    assert e.torch is None and e.cuda is None and e.driver is None
    assert e.arch_list is None
    assert e.python and e.platform
    e2 = capture_environment({"torch": "2.x", "cuda": "12.6", "driver": "580.1",
                              "arch_list": ["sm_70"]})
    assert e2.arch_list == ("sm_70",)
    with pytest.raises(BenchSchemaError, match="cuda_info"):
        capture_environment({"gpu_free_bytes": 1})  # not a field we can probe here


def test_host_rss_probe_is_real_or_none(monkeypatch):
    rss = host_rss_bytes()
    if sys.platform.startswith("linux"):
        assert isinstance(rss, int) and rss > 0

        class Boom:
            @staticmethod
            def getrusage(*a):
                raise OSError("no resource here")

        monkeypatch.setattr("qwen_engine.bench.resource", Boom())
        assert host_rss_bytes() is None


# ---------------------------------------------------------------------------
# save/load + aggregation
# ---------------------------------------------------------------------------


def test_result_save_and_load_roundtrip(tmp_path):
    r = result(run_id="run-42")
    path = save_result(r, tmp_path / "benchmarks")
    assert path.name == "run-42.benchmark.json"
    assert not list(path.parent.glob(".bench-*.tmp"))
    assert load_result(path) == r
    with pytest.raises(BenchSchemaError, match="unsafe"):
        save_result(r, tmp_path, run_id="../evil")
    (tmp_path / "bad.json").write_text("{ broken")
    with pytest.raises(BenchSchemaError, match="not valid JSON"):
        load_result(tmp_path / "bad.json")


@pytest.mark.parametrize("bad", [
    "a/b",          # forward slash
    "a\\b",         # backslash (future Windows reinterpretation; Linux target)
    "a\0b",         # NUL
    "../evil",      # traversal
    "..evil",       # leading-dot traversal spelling
    "/", "\\",
])
def test_save_result_refuses_unsafe_run_ids_before_filesystem(tmp_path, bad):
    """save_result refuses slash, backslash, NUL and traversal spellings
    BEFORE creating any directory or file (review fix 4; Linux remains the
    supported target — the backslash refusal is defensive, not a claimed
    Windows platform expansion)."""
    r = result(run_id="run-42")
    target = tmp_path / "benchmarks"
    with pytest.raises(BenchSchemaError, match="unsafe"):
        save_result(r, target, run_id=bad)
    assert not target.exists(), "refusal must precede any filesystem creation"


def test_save_result_accepts_normal_run_id(tmp_path):
    """A normal run id still saves after the unsafe-id guard."""
    r = result(run_id="run-7")
    path = save_result(r, tmp_path / "benchmarks", run_id="run-7")
    assert path.name == "run-7.benchmark.json"
    assert load_result(path).run_id == "run-7"


def synthetic_runs():
    a = result(run_id="a", phases={"load": 1000.0, "denoise": 40000.0},
               steps=40, peak_host_rss_bytes=2_000_000_000)
    b = result(run_id="b", phases={"load": 900.0, "denoise": 44000.0},
               steps=40, peak_host_rss_bytes=2_100_000_000)
    c = result(run_id="c", phases={"load": 1100.0, "denoise": 42000.0},
               steps=None, peak_host_rss_bytes=None,
               peak_cuda_allocated_bytes=15_000_000_000)
    return [a, b, c]


def test_aggregate_is_deterministic_and_order_free():
    runs = synthetic_runs()
    agg = aggregate_results(runs)
    assert agg["schema"] == AGGREGATE_SCHEMA
    assert agg["n_runs"] == 3
    assert agg["run_ids"] == ["a", "b", "c"]
    assert aggregate_results(list(reversed(runs))) == agg
    assert json.dumps(agg, sort_keys=True) == json.dumps(
        aggregate_results(runs), sort_keys=True)
    load = agg["phases"]["load"]
    assert load == {"n": 3, "min": 900.0, "median": 1000.0, "max": 1100.0}
    assert agg["phases"]["denoise"]["median"] == 42000.0
    # per-step derived only where steps were recorded
    assert agg["denoise_ms_per_step"]["n"] == 2
    assert agg["denoise_ms_per_step"]["min"] == pytest.approx(1000.0)


def test_aggregate_keeps_unmeasured_as_null_not_zero():
    agg = aggregate_results(synthetic_runs())
    rss = agg["peak_host_rss_bytes"]
    assert rss["n_measured"] == 2 and rss["max"] == 2_100_000_000
    cuda = agg["peak_cuda_allocated_bytes"]
    assert cuda["n_measured"] == 1 and cuda["max"] == 15_000_000_000
    empty = aggregate_results([])
    assert empty["peak_cuda_reserved_bytes"] == {"n_measured": 0, "max": None,
                                                 "median": None}
    assert empty["denoise_ms_per_step"]["n"] == 0
    with pytest.raises(BenchSchemaError):
        aggregate_results([{"not": "a result"}])


def test_phase_collector_with_fake_clock():
    ticks = iter([0.0, 0.5, 1.0, 3.0, 3.5, 4.0])
    collector = PhaseCollector(clock=lambda: next(ticks))
    with collector.phase("load"):
        pass
    with collector.phase("denoise"):
        pass
    assert collector.timings() == {"denoise": 2000.0, "load": 500.0}
    with pytest.raises(BenchSchemaError):
        collector.phase("")


# ---------------------------------------------------------------------------
# calibration observation schema
# ---------------------------------------------------------------------------


def test_observation_roundtrip_and_staleness_key():
    obs = observation()
    assert CalibrationObservation.from_dict(obs.to_dict()) == obs
    assert obs.staleness_key == (OBSERVATION_SCHEMA, REV, "sdpa", "fp16")
    assert observation(model_revision="f" * 40).staleness_key != obs.staleness_key
    assert observation(backend="efficient").staleness_key != obs.staleness_key
    assert observation(quant="q8").staleness_key != obs.staleness_key


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"schema": "calibration_observation.v0"}, "schema"),
        ({"observed_at": 0}, "observed_at"),
        ({"observed_at": float("inf")}, "observed_at"),
        ({"phase": ""}, "phase"),
        ({"model_revision": "main"}, "model_revision"),
        ({"backend": None}, "backend"),
        ({"quant": 5}, "quant"),
        ({"residency": ""}, "residency"),
        ({"vae_tiled_decode": 1}, "vae_tiled_decode"),
        ({"cfg_duplicated": "yes"}, "cfg_duplicated"),
        ({"tokens": 0}, "tokens"),
        ({"resolution": (1024,)}, "resolution"),
        ({"resolution": (1024, 0)}, "resolution"),
        ({"peak_gpu_allocated_bytes": -1}, "peak_gpu_allocated_bytes"),
    ],
)
def test_observation_validation_names_the_field(overrides, field):
    with pytest.raises(BenchSchemaError) as field_error:
        observation(**overrides)
    assert field in str(field_error.value)


def test_observation_peak_fields_null_means_unmeasured():
    obs = observation(peak_gpu_allocated_bytes=None, peak_gpu_reserved_bytes=None,
                      peak_host_rss_bytes=None)
    assert obs.to_dict()["peak_gpu_allocated_bytes"] is None
    assert CalibrationObservation.from_dict(obs.to_dict()) == obs


def test_calibration_log_roundtrip_and_atomic_save(tmp_path):
    log = CalibrationLog(observations=(observation(), observation(phase="decode")))
    path = tmp_path / "calibration.json"
    save_calibration(log, path)
    assert not list(path.parent.glob(".calib-*.tmp"))
    assert load_calibration(path) == log
    assert log.to_json() == load_calibration(path).to_json()
    doc = log.to_dict()
    doc["schema"] = "qwen_engine.calibration.v0"
    with pytest.raises(BenchSchemaError, match="schema"):
        CalibrationLog.from_dict(doc)
    doc = log.to_dict()
    doc["observations"][0]["surprise"] = 1
    with pytest.raises(BenchSchemaError, match="surprise"):
        CalibrationLog.from_dict(doc)
    with pytest.raises(BenchSchemaError, match="not readable"):
        load_calibration(tmp_path / "missing.json")


def test_add_observation_creates_then_appends(tmp_path):
    path = tmp_path / "models" / "calibration.json"
    log1 = add_observation(path, observation())
    assert len(log1.observations) == 1
    log2 = add_observation(path, observation(phase="decode"))
    assert [o.phase for o in log2.observations] == ["denoise_step", "decode"]


def test_matching_observations_never_interpolates_across_mismatch():
    log = CalibrationLog(observations=(
        observation(),
        observation(model_revision="f" * 40),          # stale revision
        observation(backend="efficient"),              # different backend
        observation(quant="q8"),                       # different quant
        observation(residency="streamed"),             # same key, other field
    ))
    fresh = matching_observations(log, model_revision=REV, backend="sdpa",
                                  quant="fp16")
    assert len(fresh) == 2
    assert all(o.staleness_key == (OBSERVATION_SCHEMA, REV, "sdpa", "fp16")
               for o in fresh)
    assert matching_observations(log, model_revision="0" * 40, backend="sdpa",
                                 quant="fp16") == ()


def test_model_identifiers_from_manifest():
    manifest = ModelManifest(schema_version=1, artifacts=(
        ModelArtifact(role="dit", repo="Qwen/Qwen-Image-2.1", revision=REV,
                      filename="transformer/m.safetensors", sha256=SHA,
                      bytes=10),
        ModelArtifact(role="text_encoder", repo="Qwen/Qwen3-VL",
                      revision="c" * 40, filename="encoder/m.safetensors",
                      sha256="d" * 64, bytes=20),
    ))
    ids = model_identifiers(manifest)
    assert [i["role"] for i in ids] == ["dit", "text_encoder"]
    assert ids[0]["sha256"] == SHA
    r = result(model=ids)
    assert BenchmarkResult.from_dict(r.to_dict()) == r


# ---------------------------------------------------------------------------
# Senior-review hardening: schema tags on deserialization, immutability,
# RSS units, concurrent appends
# ---------------------------------------------------------------------------


def test_from_dict_requires_exact_benchmark_schema(tmp_path):
    doc = json.loads(result().to_json())
    doc.pop("schema")
    with pytest.raises(BenchSchemaError, match="schema"):
        BenchmarkResult.from_dict(doc)
    for tag in ("benchmark_result.v0", "benchmark_result.v2", "v1", None):
        bad = json.loads(result().to_json())
        bad["schema"] = tag
        with pytest.raises(BenchSchemaError, match="schema"):
            BenchmarkResult.from_dict(bad)
    # Never re-labelled, either: a doc that got through stays exact.
    assert BenchmarkResult.from_dict(json.loads(result().to_json())).schema \
        == BENCHMARK_SCHEMA


def test_from_json_and_load_result_reject_missing_or_future_schema(tmp_path):
    doc = json.loads(result().to_json())
    doc["schema"] = "benchmark_result.v2"
    with pytest.raises(BenchSchemaError, match="schema"):
        BenchmarkResult.from_json(json.dumps(doc))
    doc.pop("schema")
    p = tmp_path / "run-1.benchmark.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(BenchSchemaError, match="schema"):
        load_result(p)


def test_observation_from_dict_requires_explicit_schema():
    doc = observation().to_dict()
    doc.pop("schema")
    with pytest.raises(BenchSchemaError, match="schema"):
        CalibrationObservation.from_dict(doc)
    doc["schema"] = "calibration_observation.v2"
    with pytest.raises(BenchSchemaError, match="schema"):
        CalibrationObservation.from_dict(doc)


def test_result_snapshot_survives_external_mutation():
    nested = {"scheduler": {"inner": [1, 2]}}
    cfg = {"prompt": "a red cube", "nested": nested}
    m = model_entry()
    model = [m]
    r = result(config=cfg, model=model)
    before = json.loads(r.to_json())

    nested["scheduler"]["inner"].append("poison")
    nested["extra"] = True
    cfg["steps"] = 999
    m["sha256"] = "e" * 64
    model.append(model_entry(filename="other.safetensors"))

    assert json.loads(r.to_json()) == before
    with pytest.raises(TypeError):
        r.config["injected"] = 1  # type: ignore[index]


def test_saved_and_aggregated_output_ignores_later_mutation(tmp_path):
    cfg = {"prompt": "a red cube", "deep": {"list": [1]}}
    r = result(config=cfg)
    path = save_result(r, tmp_path)
    cfg["deep"]["list"].append(2)
    cfg["deep"] = "gone"
    assert load_result(path).to_dict() == json.loads(r.to_json())
    agg = aggregate_results([r])
    cfg["prompt"] = "mutated"
    assert agg["n_runs"] == 1
    assert json.loads(path.read_text())["config"]["prompt"] == "a red cube"


def test_observation_resolution_snapshot_survives_mutation():
    res = [1024, 1024]
    o = observation(resolution=res)
    before = o.to_dict()
    res[0] = 512
    assert o.to_dict() == before
    assert isinstance(o.resolution, tuple)


@pytest.mark.parametrize("bad_model", [5, (m for m in ()), {"role": "dit"},
                                       "dit", {1, 2}])
def test_invalid_model_sequence_is_schema_error(bad_model):
    with pytest.raises(BenchSchemaError, match="model"):
        result(model=bad_model)


@pytest.mark.parametrize("bad_resolution", [5, "1024", (i for i in (1, 2)),
                                            {1024, 1024}, None])
def test_invalid_resolution_sequence_is_schema_error(bad_resolution):
    with pytest.raises(BenchSchemaError, match="resolution"):
        observation(resolution=bad_resolution)


@pytest.mark.parametrize("bad_config", [
    {"x": float("nan")},
    {"x": float("inf")},
    {1: "int key"},
    {"x": {1, 2}},
])
def test_config_rejects_invalid_json(bad_config):
    with pytest.raises(BenchSchemaError, match="config"):
        result(config=bad_config)


def test_phases_reject_huge_ints_without_overflow_error():
    with pytest.raises(BenchSchemaError, match="phases"):
        result(phases={"load": 10 ** 400})


def test_duplicate_model_identifiers_are_refused():
    entry = model_entry()
    with pytest.raises(BenchSchemaError, match="duplicate"):
        result(model=[entry, dict(entry)])
    # Same role with a DIFFERENT artifact file is fine (identity is the
    # filename, roles may repeat).
    other = model_entry(filename="transformer/diffusion_pytorch_model-2.safetensors")
    ok = result(model=[entry, other])
    assert len(ok.model) == 2


def test_host_rss_platform_units_and_unavailable(monkeypatch):
    import qwen_engine.bench as bench

    class FakeResource:
        RUSAGE_SELF = 0

        def __init__(self, peak):
            self.peak = peak

        def getrusage(self, who):
            return types.SimpleNamespace(ru_maxrss=self.peak)

    monkeypatch.setattr(bench, "resource", None)
    assert host_rss_bytes() is None, "unavailable probe must be null, not 0"

    monkeypatch.setattr(bench, "resource", FakeResource(2048))
    assert host_rss_bytes(platform="linux") == 2048 * 1024, \
        "ru_maxrss is KiB on Linux"
    assert host_rss_bytes(platform="darwin") == 2048, \
        "ru_maxrss is already bytes on macOS"
    assert host_rss_bytes(platform="sunos5") is None, \
        "unknown unit must be null, never a guess"


def test_add_observation_concurrent_appends_keep_all(tmp_path):
    path = tmp_path / "calibration.json"
    n = 8
    barrier = threading.Barrier(n)

    def go(i: int) -> None:
        barrier.wait()
        add_observation(path, observation(phase=f"phase-{i}"))

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    log = load_calibration(path)
    assert len(log.observations) == n, "concurrent appends lost observations"
    assert sorted(o.phase for o in log.observations) == \
        sorted(f"phase-{i}" for i in range(n))


def test_prompt_tokens_optional_and_honest():
    o = observation()
    assert o.prompt_tokens is None, "unknown prompt length stays null"
    assert o.to_dict()["prompt_tokens"] is None
    assert CalibrationObservation.from_dict(o.to_dict()).prompt_tokens is None
    o2 = observation(prompt_tokens=77)
    assert CalibrationObservation.from_dict(o2.to_dict()) == o2
    with pytest.raises(BenchSchemaError, match="prompt_tokens"):
        observation(prompt_tokens=0)
    with pytest.raises(BenchSchemaError, match="prompt_tokens"):
        observation(prompt_tokens=True)
    with pytest.raises(BenchSchemaError, match="prompt_tokens"):
        observation(prompt_tokens=12.5)


def test_calibration_roundtrip_still_lossless_with_prompt_tokens(tmp_path):
    log = CalibrationLog(observations=(observation(),
                                       observation(prompt_tokens=128)))
    path = tmp_path / "calibration.json"
    save_calibration(log, path)
    assert load_calibration(path) == log


# ---------------------------------------------------------------------------
# Boundary: stdlib only
# ---------------------------------------------------------------------------


def test_bench_import_pulls_in_no_heavy_modules():
    import qwen_engine.bench  # noqa: F401

    assert not set(HEAVY_MODULES) & set(sys.modules)


def test_bench_import_in_subprocess_without_heavy_modules():
    code = (
        "import sys, qwen_engine.bench;"
        "bad=[m for m in sys.modules if m.split('.')[0] in "
        f"{HEAVY_MODULES}];"
        "sys.exit(1 if bad else 0)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


# ---------------------------------------------------------------------------
# Second senior-review correction pass: deep freeze, typed validation
# ---------------------------------------------------------------------------


def _frozen_result(**overrides) -> BenchmarkResult:
    base = dict(run_id="run-1", environment=env(),
                model=(model_entry(),),
                config={"nested": {"list": [1, {"deep": 2}]}},
                phases={"load": 10.0, "denoise": 20.0}, steps=4)
    base.update(overrides)
    return BenchmarkResult(**base)


def test_result_is_frozen_all_the_way_down():
    """Mutation must fail THROUGH result.phases and result.model[0], not
    merely through references the caller still owns — and the saved file
    and the aggregate must not move either."""
    result = _frozen_result()

    # Direct mutation through the record itself.
    with pytest.raises(TypeError):
        result.phases["load"] = 999.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.phases["evil"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.model[0]["role"] = "evil"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.model[0]["extra"] = 1  # type: ignore[index]
    with pytest.raises((TypeError, AttributeError)):
        # Frozen all the way down: nested sequences became tuples,
        # mappings became MappingProxyType.
        result.config["nested"]["list"].append("poison")  # type: ignore[index]

    # The caller poisoning its own references afterwards changes nothing.
    # (Rebuild to have caller refs at all: they were snapshotted.)
    phases = {"load": 10.0, "denoise": 20.0}
    model = [model_entry()]
    cfg = {"nested": {"list": [1, {"deep": 2}]}}
    r2 = BenchmarkResult(run_id="run-1", environment=env(), model=tuple(model),
                         config=cfg, phases=phases, steps=4)
    phases["load"] = 0.0
    model[0]["role"] = "evil"
    cfg["nested"]["list"].append("poison")
    assert r2.to_dict() == result.to_dict()

    saved_a = result.to_json()
    try:
        result.phases["load"] = 999.0  # type: ignore[index]
    except TypeError:
        pass
    assert result.to_json() == saved_a
    assert aggregate_results([result]) == aggregate_results(
        [BenchmarkResult.from_json(saved_a)])


def test_saved_and_aggregated_output_stable_across_mutation_attempts(tmp_path):
    result = _frozen_result()
    p1 = save_result(result, tmp_path / "a")
    p2 = save_result(result, tmp_path / "b")
    assert p1.read_bytes() == p2.read_bytes()
    agg_1 = aggregate_results([result])
    for attempt in range(3):
        try:
            result.phases["denoise"] = attempt * 1e6  # type: ignore[index]
            result.model[0]["sha256"] = "f" * 64  # type: ignore[index]
        except TypeError:
            pass
    p3 = save_result(result, tmp_path / "c")
    assert p3.read_bytes() == p1.read_bytes()
    assert aggregate_results([result]) == agg_1


def test_mixed_type_phase_keys_are_a_schema_error_not_type_error():
    with pytest.raises(BenchSchemaError, match="phases"):
        BenchmarkResult(run_id="run-1", environment=env(),
                        model=(model_entry(),),
                        phases={1: 1, "load": 1})


def test_result_config_cycle_is_a_schema_error_not_recursion_error():
    cfg: dict = {"a": {}}
    cfg["a"]["back"] = cfg
    with pytest.raises(BenchSchemaError, match="config"):
        _frozen_result(config=cfg)


def test_calibration_log_validates_before_iterating_and_snapshots():
    # A non-sequence is a typed schema error, never a raw TypeError.
    with pytest.raises(BenchSchemaError, match="observations"):
        CalibrationLog(observations=5)
    with pytest.raises(BenchSchemaError, match="observations"):
        CalibrationLog(observations="obs")

    obs = observation(observed_at=1700000000.0)
    caller_list = [obs]
    log = CalibrationLog(observations=caller_list)
    assert isinstance(log.observations, tuple)
    caller_list.clear()  # caller-owned list: must not empty the frozen log
    assert log.observations == (obs,)
    assert obs.to_dict() in log.to_dict()["observations"]


def test_observation_observed_at_overflow_is_a_schema_error():
    """math.isfinite(10**400) raises OverflowError; observed_at must be
    rejected as a schema error instead."""
    with pytest.raises(BenchSchemaError, match="observed_at"):
        observation(observed_at=10 ** 400)


# --- json parser safety limits at the parse boundary ------------------------


@pytest.mark.parametrize("entry", ["result_from_json", "load_calibration"])
@pytest.mark.parametrize("value", ["9" * 5000, "[" * 2000 + "0" + "]" * 2000],
                         ids=["digit-limit", "deep-nesting"])
def test_bench_json_parse_safety_limits_are_typed(tmp_path, entry, value):
    if value.startswith("9") and not getattr(
            sys, "get_int_max_str_digits", lambda: 0)():
        pytest.skip("integer-string digit limit disabled")
    doc = '{"n": ' + value + "}"
    if entry == "result_from_json":
        with pytest.raises(BenchSchemaError):
            BenchmarkResult.from_json(doc)
    else:
        p = tmp_path / "cal.json"
        p.write_text(doc)
        with pytest.raises(BenchSchemaError):
            load_calibration(p)


@pytest.mark.parametrize("boom", [
    ValueError("exceeds the integer-string digit limit"),
    RecursionError("decoder nesting limit"),
], ids=["injected-ValueError", "injected-RecursionError"])
def test_injected_parser_failures_wrap_as_bench_schema_errors(monkeypatch,
                                                              boom):
    def raiser(*args, **kwargs):
        raise boom

    monkeypatch.setattr("json.loads", raiser)
    with pytest.raises(BenchSchemaError):
        BenchmarkResult.from_json('{"n": 1}')
