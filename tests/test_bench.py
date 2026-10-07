"""M1.5 bench schema tests — synthetic runs only, no fabricated measurements.

Covers: benchmark_result.v1 round-trip + field-level validation, CUDA-null
semantics, deterministic aggregation, phase collection, the
calibration_observation.v1 schema with its staleness key, and the
no-heavy-imports boundary.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

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
