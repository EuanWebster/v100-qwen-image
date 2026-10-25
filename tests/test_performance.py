"""Offline witnesses for diagnostics: no ML imports, GPU or model files."""
import io
import json
import threading
from types import SimpleNamespace

import pytest

from qwen_engine.performance import PerformanceRecorder, current, stage, attach_cuda
from qwen_engine.config import GenerationConfig, ConfigError
from qwen_engine import cli


class Cuda:
    def __init__(self):
        self.calls = []

    def synchronize(self, device):
        self.calls.append("sync")

    def reset_peak_memory_stats(self, device):
        self.calls.append("reset")

    def memory_allocated(self, device):
        return 12

    def memory_reserved(self, device):
        return 40

    def max_memory_allocated(self, device):
        return 24

    def max_memory_reserved(self, device):
        return 50

    def get_device_properties(self, device):
        return SimpleNamespace(name="test-device", major=7, minor=0, total_memory=100)


def test_disabled_does_not_attach_or_measure_cuda():
    assert current() is None
    attach_cuda(object())
    with stage("anything", cuda=True):
        assert current() is None


def test_gpu_stage_synchronizes_only_boundaries_and_distinguishes_memory():
    cuda = Cuda()
    recorder = PerformanceRecorder(clock=iter([3., 5.]).__next__)
    with recorder.activate():
        attach_cuda(SimpleNamespace(cuda=cuda))
        with stage("sampling", cuda=True):
            cuda.calls.append("body")
    assert cuda.calls == ["sync", "reset", "body", "sync"]
    row = recorder.records[0]
    assert row["wall_s"] == 2
    assert row["cuda"]["allocated_bytes"] == 12
    assert row["cuda"]["reserved_bytes"] == 40
    assert row["cuda"]["peak_allocated_bytes"] == 24
    assert row["cuda"]["peak_reserved_bytes"] == 50
    assert recorder.document("succeeded")["device"]["compute_capability"] == [7, 0]


def test_exception_preserved_and_scope_reset_after_failure():
    cuda = Cuda()
    recorder = PerformanceRecorder()
    with recorder.activate():
        attach_cuda(SimpleNamespace(cuda=cuda))
        with pytest.raises(RuntimeError, match="primary"):
            with stage("decode", cuda=True):
                raise RuntimeError("primary")
        with stage("cleanup", cuda=True):
            pass
    assert [r["status"] for r in recorder.records] == ["failed", "succeeded"]
    assert current() is None


def test_measurement_failure_does_not_hide_pipeline_error():
    cuda = Cuda()
    def broken(_):
        raise OSError("private detail")
    cuda.synchronize = broken
    recorder = PerformanceRecorder()
    with recorder.activate():
        attach_cuda(SimpleNamespace(cuda=cuda))
        with pytest.raises(ValueError, match="original"):
            with stage("decode", cuda=True):
                raise ValueError("original")
    assert recorder.errors == ["OSError"]
    assert "private detail" not in json.dumps(recorder.document("failed"))


def test_nested_cpu_total_does_not_reset_cuda_peaks():
    cuda = Cuda()
    recorder = PerformanceRecorder()
    with recorder.activate():
        attach_cuda(SimpleNamespace(cuda=cuda))
        with stage("total"):
            with stage("load", cuda=True):
                pass
    assert cuda.calls.count("reset") == 1
    assert recorder.records[-1]["cuda"] is None


def test_context_does_not_leak_to_other_thread():
    recorder = PerformanceRecorder()
    values = []
    with recorder.activate():
        thread = threading.Thread(target=lambda: values.append(current()))
        thread.start()
        thread.join()
        assert current() is recorder
    assert values == [None]


def test_prompt_and_local_paths_never_appear_in_report():
    recorder = PerformanceRecorder(GenerationConfig(prompt="SECRET PHRASE", negative_prompt="OTHER SECRET"))
    with recorder.activate(), stage("output_saving"):
        pass
    handle = io.StringIO()
    recorder.write(handle, "succeeded")
    raw = handle.getvalue()
    assert "SECRET" not in raw and "prompt" not in raw
    assert json.loads(raw)["metadata"]["seed"] == 0


def test_child_report_merge_has_separate_pid_cpu_memory(tmp_path):
    child = PerformanceRecorder()
    with child.activate(), stage("prompt_encoding"):
        pass
    path = tmp_path / "child.json"
    with path.open("w") as handle:
        child.write(handle, "succeeded")
    parent = PerformanceRecorder()
    parent.import_encoder_report(path)
    assert not parent.errors
    assert parent.records == child.records


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(metadata={"prompt": "private"}),
    lambda d: d["stages"][0].update(prompt="private"),
    lambda d: d["stages"][0].update(stage="unknown"),
    lambda d: d["stages"][0].update(wall_s=-1),
    lambda d: d["stages"][0].update(pid=True),
    lambda d: d["stages"][0].update(host_rss_end_bytes=-1),
])
def test_malformed_child_report_is_not_imported(tmp_path, mutation):
    child = PerformanceRecorder()
    with child.stage("prompt_encoding"):
        pass
    doc = child.document("succeeded")
    mutation(doc)
    path = tmp_path / "child.json"
    path.write_text(json.dumps(doc))
    parent = PerformanceRecorder()
    parent.import_encoder_report(path)
    assert not parent.records and parent.errors


def test_performance_flag_is_opt_in_and_not_config_field():
    args = cli.build_parser().parse_args([])
    assert args.performance_report is None
    assert "performance_report" not in GenerationConfig().to_dict()


def test_collectors_cannot_reset_each_others_peaks(tmp_path):
    with pytest.raises(ConfigError, match="separately"):
        cli._run_generate(GenerationConfig(output_dir=str(tmp_path)), cli.PhaseLogger(),
                          performance_report=tmp_path/"perf.json", calibration_record=tmp_path/"cal.json")
    assert not (tmp_path/"perf.json").exists()


def test_performance_output_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("user data")
    with pytest.raises(ConfigError, match="new writable"):
        cli._run_generate(GenerationConfig(output_dir=str(tmp_path)), cli.PhaseLogger(),
                          performance_report=path)
    assert path.read_text() == "user data"


def test_broken_report_sink_preserves_primary_exception():
    class Broken:
        def write(self, text):
            raise OSError("private disk detail")
    recorder = PerformanceRecorder()
    with pytest.raises(RuntimeError, match="original"):
        try:
            raise RuntimeError("original")
        finally:
            assert recorder.write(Broken(), "failed") is False
    assert recorder.errors == ["report_write_OSError"]


@pytest.mark.parametrize("fail", [None, "load", "encode", "denoise", "decode", "save"])
def test_cli_report_and_cleanup_follow_real_core_lifecycle(tmp_path, fail):
    from test_generation_lifecycle import Adapter
    adapter = Adapter(fail=fail)
    path = tmp_path / "performance.json"
    config = GenerationConfig(output_dir=str(tmp_path))
    if fail:
        with pytest.raises(RuntimeError, match="primary failure"):
            cli._run_generate(config, cli.PhaseLogger(), adapter=adapter, performance_report=path)
    else:
        assert cli._run_generate(config, cli.PhaseLogger(), adapter=adapter, performance_report=path) == 0
    doc = json.loads(path.read_text())
    assert doc["status"] == ("failed" if fail else "succeeded")
    assert doc["stages"][-1]["stage"] == "total_job"
    assert "adapter_cleanup" in [r["stage"] for r in doc["stages"]]
    assert adapter.calls[-1] == "close"
