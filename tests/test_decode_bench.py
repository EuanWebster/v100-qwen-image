"""CPU-only guarded tests for the decode benchmark core and script CLI.

These tests never import torch/PIL/diffusers: the core module must stay
ML-free, and the script must refuse to touch the GPU without ``--run``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import runpy
import dataclasses
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
SCRIPT = REPO_ROOT / "scripts" / "benchmark_vae_decode.py"

from qwen_engine.config import ConfigError
from qwen_engine.config import GenerationConfig
from qwen_engine.decode_bench import (
    BENCHMARK_SCHEMA, DEFAULT_GEOMETRIES, MemoryObservation, PeakMemoryProbe,
    alpha_stats, build_benchmark_report, build_snapshot_metadata,
    compare_rgb_bytes, make_memory_observation, read_benchmark_report,
    sha256_file, validate_geometries, validate_geometry, validate_label,
    validate_snapshot_metadata, write_benchmark_report,
)


def test_script_semantic_identity_ignores_paths_but_binds_prompt(tmp_path):
    script = runpy.run_path(str(SCRIPT), run_name='decode_bench_test')
    identity = script['semantic_config_sha256']
    config = GenerationConfig(prompt='a red cube', output_dir=str(tmp_path / 'first'))
    relocated = dataclasses.replace(config, output_dir=str(tmp_path / 'second'),
                                    cache_dir=str(tmp_path / 'cache'))
    assert identity(config) == identity(relocated)
    assert identity(config) != identity(dataclasses.replace(config, prompt='a blue cube'))


def test_capture_wrapper_uses_the_bound_cancellation_event(tmp_path):
    script = runpy.run_path(str(SCRIPT), run_name='decode_bench_test')
    class Inner:
        def encode(self, config): return 'conditioning'
        def denoise(self, config, conditioning): return 'latent'
        def close(self): pass
    event = threading.Event()
    capture = script['CapturingAdapter'](Inner(), tmp_path / 'latents.pt', None, None, event)
    assert capture.encode(None) == 'conditioning'
    event.set()
    with pytest.raises(KeyboardInterrupt):
        capture.encode(None)
    with pytest.raises(KeyboardInterrupt):
        capture.denoise(None, None)


def test_probe_records_actual_allocator_retry_delta():
    from types import SimpleNamespace
    class Cuda:
        retries = 4
        def is_available(self): return True
        def synchronize(self): pass
        def empty_cache(self): pass
        def reset_peak_memory_stats(self): pass
        def memory_stats(self): return {'num_alloc_retries': self.retries}
        def max_memory_allocated(self): return 100
        def max_memory_reserved(self): return 200
    cuda = Cuda()
    probe = PeakMemoryProbe(device_uuid='GPU-test', device_total_bytes=1024,
                            interval_ms=50, torch_module=SimpleNamespace(cuda=cuda), allow_nvml=False)
    probe.start()
    cuda.retries += 2
    observation = probe.finish()
    assert observation.allocator_retries == 2
    assert observation.nvml_peak_used_bytes is None
    assert observation.peak_allocated_bytes == 100
from qwen_engine.errors import EngineError

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


# ---------------------------------------------------------------------------
# ML-free import guarantee
# ---------------------------------------------------------------------------


def test_core_import_is_ml_free():
    code = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "import qwen_engine.decode_bench\n"
        "banned = [m for m in ('torch', 'PIL', 'diffusers', 'transformers', "
        "'psutil', 'numpy') if m in sys.modules]\n"
        "assert not banned, banned\n"
    ) % str(SRC)
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile,stride", [(1024, 768), (768, 576), (512, 384), (16, 16)])
def test_geometry_accepts_reviewed_pairs(tile, stride):
    geometry = validate_geometry(tile, stride)
    assert (geometry.tile, geometry.stride) == (tile, stride)
    assert geometry.label == f"tile{tile}-stride{stride}"


@pytest.mark.parametrize("tile,stride", [
    (1024, 769),          # stride not a multiple of 16
    (1000, 768),          # tile not a multiple of 16
    (768, 1024),          # stride above tile
    (0, 0),               # zero
    (-16, 16),            # negative
    (1024.0, 768),        # float, not exact int
    (True, 768),          # bool is not an int here
    ("1024", 768),        # string is never coerced
    (1024, 8),            # stride below the 16 px blend floor
])
def test_geometry_refusals(tile, stride):
    with pytest.raises(ConfigError):
        validate_geometry(tile, stride)


def test_geometry_list_rules():
    assert validate_geometries(list(DEFAULT_GEOMETRIES)) == DEFAULT_GEOMETRIES
    assert {(g.tile, g.stride) for g in DEFAULT_GEOMETRIES} == {
        (1024, 768), (768, 576), (512, 384)}
    with pytest.raises(ConfigError):
        validate_geometries([])
    with pytest.raises(ConfigError):
        validate_geometries([(1024, 768), (1024, 768)])
    with pytest.raises(ConfigError):
        validate_geometries("1024/768")
    with pytest.raises(ConfigError):
        validate_geometries([(1024,)])


# ---------------------------------------------------------------------------
# Labels (path-injection refusal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["tile1024-stride768", "a", "A1.b-c_9", "x" * 64])
def test_label_accepts_safe_tokens(value):
    assert validate_label(value) == value


@pytest.mark.parametrize("value", [
    "", "-lead", "../etc/passwd", "a/b", "a\\b", "a b", "a\nb", "a;b",
    "x" * 65, "tile1024/stride768", 42, None, "a.b$c",
])
def test_label_refusals(value):
    with pytest.raises(ConfigError):
        validate_label(value)


# ---------------------------------------------------------------------------
# Memory observation
# ---------------------------------------------------------------------------


def test_memory_observation_round_trips_and_distinguishes_sources():
    observation = make_memory_observation(
        peak_allocated_bytes=13 * 1024**3, peak_reserved_bytes=15 * 1024**3,
        nvml_peak_used_bytes=15 * 1024**3 + 2**30, nvml_total_bytes=16 * 1024**3,
        nvml_driver_version="535.129.03", interval_ms=50.0, samples=97,
        allocator_retries=0, device_uuid="GPU-abc")
    document = observation.to_dict()
    assert document["peak_allocated_bytes"] != document["nvml_peak_used_bytes"]
    assert "device-wide" in document["nvml_scope"]
    assert observation.allocator_retries == 0


def test_memory_observation_refusals():
    with pytest.raises(ConfigError):
        make_memory_observation(peak_allocated_bytes=15, peak_reserved_bytes=14,
                                interval_ms=50, samples=10, allocator_retries=0)
    with pytest.raises(ConfigError):
        make_memory_observation(peak_allocated_bytes=15, peak_reserved_bytes=16,
                                interval_ms=0.1, samples=10, allocator_retries=0)
    with pytest.raises(ConfigError):
        make_memory_observation(peak_allocated_bytes=15, peak_reserved_bytes=16,
                                interval_ms=50, samples=-1, allocator_retries=0)
    with pytest.raises(ConfigError):
        make_memory_observation(peak_allocated_bytes=15, peak_reserved_bytes=16,
                                interval_ms=50, samples=10, allocator_retries=-1)
    with pytest.raises(ConfigError):
        make_memory_observation(peak_allocated_bytes=float("nan"),
                                peak_reserved_bytes=16, interval_ms=50,
                                samples=10, allocator_retries=0)


def test_probe_requires_cuda_without_touching_nvml():
    probe = PeakMemoryProbe(device_uuid="GPU-x", device_total_bytes=16 * 1024**3,
                            interval_ms=50.0, torch_module=None, allow_nvml=False)
    # On the CPU guard there is no CUDA; start must refuse, not sample.
    try:
        probe.start()
    except EngineError:
        pass
    else:  # pragma: no cover - only on a real CUDA box
        pytest.skip("CUDA present; refusal path not applicable")


def test_probe_rejects_bad_interval():
    with pytest.raises(ConfigError):
        PeakMemoryProbe(device_uuid="GPU-x", device_total_bytes=1,
                        interval_ms=10_000.0, torch_module=object())
    with pytest.raises(ConfigError):
        PeakMemoryProbe(device_uuid="", device_total_bytes=1,
                        interval_ms=50.0, torch_module=object())


# ---------------------------------------------------------------------------
# Image comparison math
# ---------------------------------------------------------------------------


def test_compare_rgb_bytes_exact_metrics():
    reference = bytes([10, 20, 30] * 4)
    candidate = bytearray(reference)
    candidate[0] = 11
    candidate[7] = 255
    metrics = compare_rgb_bytes(reference, candidate)
    assert metrics["max_channel_delta"] == 235
    assert metrics["per_channel_max_delta"] == [1, 235, 0]
    assert metrics["changed_pixel_fraction"] == 2 / 4
    assert metrics["mean_channel_delta"] == (1 + 235) / 12
    assert metrics["pixels"] == 4


def test_compare_rgb_bytes_refusals():
    with pytest.raises(ConfigError):
        compare_rgb_bytes(b"abc", b"abcd")
    with pytest.raises(ConfigError):
        compare_rgb_bytes(b"", b"")
    with pytest.raises(ConfigError):
        compare_rgb_bytes("abc", "abc")
    with pytest.raises(ConfigError):
        compare_rgb_bytes(b"ab", b"ab", channels=3)
    with pytest.raises(ConfigError):
        compare_rgb_bytes(b"abc", b"abc", channels=5)


def test_changed_fraction_counts_pixels_not_changed_channels():
    metrics = compare_rgb_bytes(bytes(8), bytes([255]*4+[0]*4), channels=4)
    assert metrics['changed_pixel_fraction'] == .5


def test_full_2k_rgba_buffers_are_accepted():
    data = bytes([0, 0, 0, 255]) * (2048 * 2048)
    assert alpha_stats(data)['alpha_min'] == 255
    assert compare_rgb_bytes(data, data, channels=4)['changed_pixel_fraction'] == 0


def test_alpha_stats():
    assert alpha_stats(bytes([0, 0, 0, 255] * 3)) == {
        "alpha_uniform": True, "alpha_min": 255, "alpha_max": 255,
        "alpha_mean": 255.0}
    mixed = alpha_stats(bytes([0, 0, 0, 0, 0, 0, 0, 255]))
    assert mixed["alpha_uniform"] is False and mixed["alpha_mean"] == 127.5
    with pytest.raises(ConfigError):
        alpha_stats(b"abc")


# ---------------------------------------------------------------------------
# Snapshot metadata
# ---------------------------------------------------------------------------


def _metadata(**overrides):
    arguments = dict(seed=42, width=2048, height=2048, steps=4, quant="q8",
                     semantic_config_sha256=SHA_A, artifact_set_sha256=SHA_B,
                     runtime_identity_sha256=SHA_C,
                     capture_geometry="tile1024-stride768",
                     latent_shape=(1, 16384, 64), latent_dtype="float16",
                     latent_sha256="d" * 64)
    arguments.update(overrides)
    return build_snapshot_metadata(**arguments)


def test_snapshot_metadata_binds_run_and_geometry():
    document = _metadata()
    assert document["capture_tile"] == 1024 and document["capture_stride"] == 768
    assert document["latent_shape"] == [1, 16384, 64]
    assert document["schema"] == BENCHMARK_SCHEMA
    expected = {"seed": 42, "artifact_set_sha256": SHA_B}
    assert validate_snapshot_metadata(document, expected=expected) == document


@pytest.mark.parametrize("overrides", [
    dict(seed=-1), dict(seed="42"), dict(width=2040), dict(steps=0),
    dict(semantic_config_sha256="nope"), dict(latent_shape=(1, 16384)),
    dict(latent_shape=(1, 16384, 64, 1)), dict(latent_dtype="fp 16"),
    dict(capture_geometry="../evil"), dict(quant="q8_0 x"),
])
def test_snapshot_metadata_refusals(overrides):
    with pytest.raises(ConfigError):
        _metadata(**overrides)


def test_snapshot_replay_rejects_disagreement_and_shape():
    document = _metadata()
    with pytest.raises(EngineError):
        validate_snapshot_metadata(document, expected={"seed": 43})
    with pytest.raises(EngineError):
        validate_snapshot_metadata(document, expected={"artifact_set_sha256": "e" * 64})
    with pytest.raises(ConfigError):
        validate_snapshot_metadata({**document, "extra": 1}, expected={})
    with pytest.raises(ConfigError):
        validate_snapshot_metadata("not a dict", expected={})
    with pytest.raises(ConfigError):
        validate_snapshot_metadata(document, expected={"nope": 1})


# ---------------------------------------------------------------------------
# Benchmark report: strict JSON, byte caps, geometry recorded
# ---------------------------------------------------------------------------


def _report_document():
    observation = make_memory_observation(
        peak_allocated_bytes=13 * 1024**3, peak_reserved_bytes=15 * 1024**3,
        interval_ms=50.0, samples=42, allocator_retries=0)
    run = {"prompt": "p", "seed": 42, "width": 2048, "height": 2048, "steps": 4,
           "quant": "fp16", "semantic_config_sha256": SHA_A,
           "artifact_set_sha256": SHA_B, "runtime_identity_sha256": SHA_C,
           "device_uuid": "GPU-x", "nvml_error": None}
    profiles = [{
        "label": geometry.label,
        "geometry": {"tile": geometry.tile, "stride": geometry.stride},
        "vae_dtype": "fp32", "observed_at": "2025-01-01T00:00:00+00:00",
        "duration_s": 12.5, "memory": observation,
        "image": {"png_path": f"{geometry.label}.png", "png_sha256": SHA_A,
                  "size": [2048, 2048], "mode": "RGB"},
    } for geometry in DEFAULT_GEOMETRIES]
    return build_benchmark_report(
        run=run,
        latents={"snapshot": "l.pt", "snapshot_sha256": SHA_A,
                 "latent_sha256": SHA_B, "latent_shape": [1, 16384, 64],
                 "latent_dtype": "torch.float16"},
        profiles=profiles, comparisons=[], limitations=["sampled peaks"])


def test_report_records_geometry_and_round_trips(tmp_path):
    report = _report_document()
    path = tmp_path / "report.json"
    written = write_benchmark_report(report, str(path))
    payload = path.read_bytes()
    assert 0 < written <= len(payload) <= 1024**3
    assert b"NaN" not in payload and b"Infinity" not in payload
    reloaded = read_benchmark_report(str(path))
    assert reloaded["schema"] == BENCHMARK_SCHEMA
    geometries = {(p["geometry"]["tile"], p["geometry"]["stride"])
                  for p in reloaded["profiles"]}
    assert geometries == {(g.tile, g.stride) for g in DEFAULT_GEOMETRIES}
    # The custom schema is NOT the production CalibrationObservation shape.
    assert "peak_gpu_reserved_bytes" not in payload.decode()


def test_report_refuses_non_finite_and_malformed_blocks():
    report = _report_document()
    report["profiles"][0]["duration_s"] = float("nan")
    with pytest.raises(ConfigError):
        build_benchmark_report(run=report["run"], latents=report["latents"],
                               profiles=report["profiles"], comparisons=[],
                               limitations=[])
    report = _report_document()
    broken = dict(report["profiles"][0])
    broken["geometry"] = {"tile": 1024, "stride": 900}
    with pytest.raises(ConfigError):
        build_benchmark_report(run=report["run"], latents=report["latents"],
                               profiles=[broken, *report["profiles"][1:]],
                               comparisons=[], limitations=[])
    with pytest.raises(ConfigError):
        build_benchmark_report(run={"prompt": ""}, latents={}, profiles=[],
                               comparisons=[], limitations=[])
    with pytest.raises(ConfigError):
        build_benchmark_report(run=_report_document()["run"],
                               latents={"latent_sha256": "bad"},
                               profiles=report["profiles"], comparisons=[],
                               limitations=[])


def test_report_reader_rejects_foreign_documents(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema": "other", "version": 99}))
    with pytest.raises(EngineError):
        read_benchmark_report(str(path))
    path.write_text('{"schema": "qwen.decode-bench", "version": 1, "NaN": 1}')
    with pytest.raises(EngineError):
        read_benchmark_report(str(path))


def test_sha256_file(tmp_path):
    import hashlib

    path = tmp_path / "blob"
    path.write_bytes(b"z" * 5000)
    assert sha256_file(str(path)) == hashlib.sha256(b"z" * 5000).hexdigest()


# ---------------------------------------------------------------------------
# Script CLI: plan-only default, config refusals, no GPU without --run
# ---------------------------------------------------------------------------


def _run_script(*argv):
    return subprocess.run([sys.executable, str(SCRIPT), *argv],
                          capture_output=True, text=True, timeout=180)


def test_script_plan_mode_writes_nothing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    result = _run_script(
        "--output-dir", str(out),
        "--manifest-path", str(tmp_path / "missing.yaml"),
        "--models-dir", str(tmp_path), "--cache-dir", str(tmp_path),
        "--geometries", "1024/768,768/576,512/384")
    assert result.returncode == 0, result.stderr
    assert list(out.iterdir()) == []
    plan = json.loads(result.stdout)
    assert plan["would_run"] is True
    assert plan["geometries"] == ["tile1024-stride768", "tile768-stride576",
                                  "tile512-stride384"]
    assert "torch" not in result.stdout


def test_script_requires_q8_sidecar_and_rejects_bad_geometry(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    base = ["--output-dir", str(out), "--manifest-path", str(tmp_path / "m.yaml"),
            "--models-dir", str(tmp_path), "--cache-dir", str(tmp_path)]
    missing = _run_script(*base, "--quant", "q8")
    assert missing.returncode == 2
    assert "sidecar" in missing.stderr or "q8-manifest-path" in missing.stderr
    bad = _run_script(*base, "--geometries", "1024/777")
    assert bad.returncode == 2
    assert "geometry" in bad.stderr
    missing_dir = _run_script("--output-dir", str(tmp_path / "nope"),
                              "--manifest-path", str(tmp_path / "m.yaml"),
                              "--models-dir", str(tmp_path),
                              "--cache-dir", str(tmp_path))
    assert missing_dir.returncode == 2
