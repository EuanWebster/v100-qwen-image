"""CPU-only guarded tests for the reference-memory benchmark script.

The script is loaded from scripts/ and exercised through its injected
seams (adapter/probe/stage-host/torch/recorder); no torch, no GPU, no
network. Real engine pieces run unmodified: core.generate, the
PerformanceRecorder, reference_store, the fresh-window measurement math.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from qwen_engine import performance
from qwen_engine.config import GenerationConfig
from qwen_engine.image_output import save_output_image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_reference_memory.py"

spec = importlib.util.spec_from_file_location("benchmark_reference_memory", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules["benchmark_reference_memory"] = module
spec.loader.exec_module(module)

PROMPT = "purple-llama-wearing-a-top-hat"


def write_config(tmp_path: Path, **changes) -> Path:
    config = replace(GenerationConfig(prompt=PROMPT, seed=42, output_dir=str(tmp_path),
                                      width=256, height=256, steps=3), **changes)
    path = tmp_path / "config.json"
    path.write_text(config.to_json())
    return path


def png_bytes(colour):
    stream = io.BytesIO()
    Image.new("RGB", (16, 8), colour).save(stream, format="PNG")
    return stream.getvalue()


class _FakeCuda:
    def __init__(self, retries=(0, 3)):
        self._retries = list(retries)
        self.reads = 0
        self.resets = 0

    def synchronize(self):
        return None

    def memory_stats(self):
        value = self._retries[min(self.reads, len(self._retries) - 1)]
        self.reads += 1
        return {"num_alloc_retries": value}

    def reset_peak_memory_stats(self):
        self.resets += 1

    def max_memory_allocated(self):
        return 4096

    def max_memory_reserved(self):
        return 8192

    def is_initialized(self):
        return True

    def get_device_properties(self, index):
        from types import SimpleNamespace
        return SimpleNamespace(uuid="GPU-fake-uuid", total_memory=16 * 1024**3)


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()


class _FakeProbe:
    def __init__(self, torch_module):
        self.started = False

    def start(self):
        self.started = True

    def finish(self):
        return module.make_memory_observation(
            peak_allocated_bytes=4096, peak_reserved_bytes=8192,
            interval_ms=50.0, samples=7, allocator_retries=2,
            nvml_peak_used_bytes=9_000_000_000, nvml_total_bytes=16 * 1024**3,
            nvml_driver_version="fake-driver")


class OutOfMemoryError(Exception):
    """Named exactly like the torch condition the matrix must record."""


class FakeInner:
    """Five-phase adapter witness with a real save and a watched VAE stage."""

    def __init__(self, *, stage, fail_at=None):
        self._stage = stage
        self._fail_at = fail_at
        self.use_kv_cache = True
        self.facts_kv = True
        self.progress = None
        self.cancel = None
        self.closed = False

    def set_progress_callback(self, callback):
        self.progress = callback

    def set_cancel_event(self, event):
        self.cancel = event

    def load(self, config, assets):
        return None

    def encode(self, config):
        return object()

    def calibration_facts(self):
        return {"prompt_tokens": 17, "negative_prompt_tokens": None,
                "cfg_active": False, "encoder_dtype": "fp16",
                "execution_dtype": "fp16", "use_kv_cache": self.use_kv_cache,
                "embedding_cache_hit": False,
                "references": 2, "joint_tokens": 17 + 2 * 4096 + 256}

    def denoise(self, config, conditioning):
        if self._fail_at == "denoise":
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 9 GiB")
        with self._stage(module.VAE_STAGE):
            pass
        for step in range(config.steps):
            self.progress(step + 1, config.steps)
        return object()

    def decode(self, latents):
        return None

    def save(self, image, config):
        return str(save_output_image(Image.new("RGB", (config.width, config.height)),
                                     config.output_dir, config.output_format.value, 42))

    def close(self):
        self.closed = True


def make_spec(tmp_path, **overrides):
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir(exist_ok=True)
    spec = {"index": 0, "references": 2, "kv_cache": False,
            "reference_ids": ["a" * 64, "b" * 64],
            "config_path": None, "references_dir": str(tmp_path / "store"),
            "output_dir": str(cell_dir), "interval_ms": 50.0,
            "perf_path": str(tmp_path / "cell.perf.json"),
            "cell_path": str(tmp_path / "cell.json")}
    spec.update(overrides)
    return spec


def run_cell(tmp_path, *, fail_at=None):
    config_path = write_config(tmp_path)
    spec = make_spec(tmp_path, config_path=str(config_path))
    host = __import__("types").SimpleNamespace(stage=performance.stage)
    inner = FakeInner(stage=lambda name: host.stage(name), fail_at=fail_at)
    ticks = iter([index * 0.5 for index in range(64)])
    clock = lambda: next(ticks)
    adapter_factory = lambda *, kv_cache, attest_unbounded_host: module.CellRunner(
        inner, kv_cache=kv_cache, clock=clock)
    from qwen_engine.performance import PerformanceRecorder

    def recorder_factory(config):
        recorder = PerformanceRecorder(config)
        return module._RecorderHandle(recorder, Path(spec["perf_path"]))

    record = module.execute_cell(
        spec, assets_verifier=lambda config: {}, adapter_factory=adapter_factory,
        probe_factory=lambda torch_module: _FakeProbe(torch_module),
        stage_host_getter=lambda: host, torch_module=_FakeTorch(),
        recorder_factory=recorder_factory, clock=clock)
    return spec, record


# ---------------------------------------------------------------------------
# plan mode
# ---------------------------------------------------------------------------


def test_plan_is_deterministic_prompt_free_and_bounded(tmp_path, capsys):
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(png_bytes((200, 0, 0)))
    second.write_bytes(png_bytes((0, 0, 200)))
    config = write_config(tmp_path)
    argv = ["--config", str(config), "--reference", str(first), "--reference",
            str(second), "--counts", "0,1,2", "--kv-cache", "on,off",
            "--output-dir", str(tmp_path / "matrix")]
    assert module.main(argv) == 0
    plan_a = json.loads(capsys.readouterr().out)
    assert module.main(argv) == 0
    plan_b = json.loads(capsys.readouterr().out)
    assert plan_a == plan_b and plan_a["run"] is False
    cells = plan_a["cells"]
    assert [(c["references"], c["kv_cache"]) for c in cells] == [
        (0, True), (1, True), (2, True), (0, False), (1, False), (2, False)]
    assert cells[4]["reference_ids"] == [plan_a["reference_ids"][0]]
    assert cells[5]["reference_ids"] == plan_a["reference_ids"][:2]
    text = json.dumps(plan_a)
    assert PROMPT not in text and str(tmp_path) not in text
    assert len(text) < 256 * 1024


def test_plan_refuses_seedless_config(tmp_path, capsys):
    config = write_config(tmp_path, seed=None)
    assert module.main(["--config", str(config), "--output-dir",
                        str(tmp_path / "m"), "--counts", "0"]) == 1
    err = capsys.readouterr().err
    assert err.strip() and PROMPT not in err and str(tmp_path) not in err


@pytest.mark.parametrize("counts,kv", [("0,17", "on"), ("0,", "sometimes"),
                                       ("-1", "on")])
def test_plan_refuses_bad_matrix_axes(tmp_path, capsys, counts, kv):
    config = write_config(tmp_path)
    assert module.main(["--config", str(config), "--output-dir",
                        str(tmp_path / "m"), "--counts", counts,
                        "--kv-cache", kv]) == 1
    assert PROMPT not in capsys.readouterr().err


def test_plan_refuses_existing_output_and_missing_references(tmp_path, capsys):
    config = write_config(tmp_path)
    (tmp_path / "m").mkdir()
    assert module.main(["--config", str(config), "--output-dir",
                        str(tmp_path / "m"), "--counts", "0"]) == 1
    first_err = capsys.readouterr().err
    assert first_err.startswith("reference matrix refused")  # type-name only
    assert str(tmp_path) not in first_err
    assert module.main(["--config", str(config), "--output-dir",
                        str(tmp_path / "m2"), "--counts", "0,1"]) == 1
    second_err = capsys.readouterr().err
    assert second_err.startswith("reference matrix refused")
    assert str(tmp_path) not in second_err


def test_ingest_is_content_addressed_and_deduplicated(tmp_path):
    first, dup, other = (tmp_path / n for n in ("a.png", "a-copy.png", "b.png"))
    raw_a = png_bytes((200, 0, 0))
    first.write_bytes(raw_a)
    dup.write_bytes(raw_a)
    other.write_bytes(png_bytes((0, 0, 200)))
    out = tmp_path / "matrix"
    out.mkdir()
    store, ids = module.ingest(out, [first, dup, other])
    assert ids[0] == ids[1] != ids[2]
    blobs = list(Path(store).glob("*" + module.reference_store.BLOB_SUFFIX))
    assert len(blobs) == 2  # duplicate content produced one blob only
    with pytest.raises(FileExistsError):
        module.ingest(out, [first])


# ---------------------------------------------------------------------------
# cell execution
# ---------------------------------------------------------------------------


def test_execute_cell_success_records_every_required_field(tmp_path):
    spec, record = run_cell(tmp_path)
    assert record["status"] == "succeeded" and record["error_type"] is None
    assert record["phases"] == ["load", "encode", "denoise", "decode", "save"]
    assert record["references"] == 2 and record["kv_cache"] is False
    assert record["prompt_tokens"] == 17
    assert record["joint_tokens"] == 17 + 2 * 4096 + 256
    assert record["use_kv_cache"] is False  # the knob reached the executor
    assert record["step_ms"] == [500.0, 500.0]  # 3 step marks -> 2 intervals
    window = record["vae_encode"]
    assert window["stage"] == "reference_vae_encode" and window["status"] == "succeeded"
    assert window["peak_allocated_bytes"] == 4096
    assert window["peak_reserved_bytes"] == 8192
    assert window["allocator_retries"] == 3  # real num_alloc_retries delta
    assert window["nvml_peak_used_bytes"] is None
    assert window["nvml_note"] == "nvml_unavailable"  # honest CPU degradation
    sampling = record["sampling"]
    assert sampling["nvml_peak_used_bytes"] == 9_000_000_000
    assert sampling["allocator_retries"] == 2
    perf = Path(spec["perf_path"])
    assert perf.is_file() and PROMPT not in perf.read_text()
    assert record["seed"] == 42 and record["resolution"] == [256, 256]


def test_execute_cell_records_oom_as_an_outcome(tmp_path):
    spec, record = run_cell(tmp_path, fail_at="denoise")
    assert record["status"] == "failed"
    assert record["error_type"] == "OutOfMemoryError"
    assert record["sampling"]["samples"] == 7  # probe finished despite OOM
    assert "CUDA out of memory" not in json.dumps(record)  # message never echoed


def test_execute_cell_closes_adapter_and_uninstalls_wrapper(tmp_path):
    config_path = write_config(tmp_path)
    spec = make_spec(tmp_path, config_path=str(config_path))
    host = __import__("types").SimpleNamespace(stage=performance.stage)
    original = performance.stage
    inner = FakeInner(stage=lambda name: host.stage(name))
    from qwen_engine.performance import PerformanceRecorder

    def recorder_factory(config):
        return module._RecorderHandle(PerformanceRecorder(config),
                                      Path(spec["perf_path"]))
    module.execute_cell(
        spec, assets_verifier=lambda config: {},
        adapter_factory=lambda **kw: module.CellRunner(inner, kv_cache=True),
        probe_factory=lambda torch_module: _FakeProbe(torch_module),
        stage_host_getter=lambda: host, torch_module=_FakeTorch(),
        recorder_factory=recorder_factory)
    assert inner.closed
    assert host.stage is original  # wrapper uninstalled from the host
    assert performance.stage is original  # global stage untouched


# ---------------------------------------------------------------------------
# matrix driver
# ---------------------------------------------------------------------------


def test_process_cells_attempts_every_cell_and_records_crashes(tmp_path):
    out = tmp_path / "matrix"
    out.mkdir()
    plan = {"cells": [{"index": 0, "references": 0, "kv_cache": True,
                       "reference_ids": []},
                      {"index": 1, "references": 1, "kv_cache": False,
                       "reference_ids": ["c" * 64]}]}
    seen_commands = []

    def spawn(command, log_path):
        seen_commands.append(command)
        log_path.write_text("")
        index = command[-1].split("cell-")[1].split(".")[0]
        if index == "0":
            module._write_json_exclusive(out / "cell-0.json", {
                "index": 0, "references": 0, "kv_cache": True,
                "status": "failed", "error_type": "OutOfMemoryError"})
            return 1
        return 9  # crashed without writing a cell record

    cells = module.process_cells(out, plan, str(out / "store"), Path("cfg.json"),
                                 interval_ms=50.0, attest=False, spawn=spawn)
    assert [c["status"] for c in cells] == ["failed", "crashed"]
    assert cells[0]["error_type"] == "OutOfMemoryError"
    assert cells[1]["error_type"] == "exit_code_9"
    assert all(cmd[0] == sys.executable and cmd[1] == "-B" and cmd[2] ==
               str(SCRIPT.resolve()) and cmd[3] == "--cell-spec"
               for cmd in seen_commands)
    assert (out / "cell-0.spec.json").is_file()


def test_main_run_writes_plan_and_matrix_with_fake_cells(tmp_path, capsys, monkeypatch):
    config_path = write_config(tmp_path)
    reference = tmp_path / "r.png"
    reference.write_bytes(png_bytes((10, 200, 10)))
    fake_cells = [{"index": 0, "references": 0, "kv_cache": True, "status": "succeeded"},
                  {"index": 1, "references": 1, "kv_cache": True, "status": "failed",
                   "error_type": "OutOfMemoryError"}]
    monkeypatch.setattr(module, "process_cells",
                        lambda *a, **k: [dict(c) for c in fake_cells])
    assert module.main(["--config", str(config_path), "--reference", str(reference),
                        "--counts", "0,1", "--kv-cache", "on",
                        "--output-dir", str(tmp_path / "matrix"), "--run"]) == 0
    out = tmp_path / "matrix"
    matrix = json.loads((out / "matrix.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    assert plan["run"] is False or plan["cells"]  # recorded plan document
    assert [c["status"] for c in matrix["cells"]] == ["succeeded", "failed"]
    text = json.dumps(matrix) + json.dumps(plan)
    assert PROMPT not in text and str(tmp_path) not in text
    assert (out / "store").is_dir()


def test_cell_mode_requires_run(tmp_path, capsys):
    assert module.main(["--cell-spec", str(tmp_path / "nope.json")]) == 1
    assert "--run" in capsys.readouterr().err
