import importlib.util
import json
from pathlib import Path

from qwen_engine.config import GenerationConfig


def tool():
    spec = importlib.util.spec_from_file_location("generation_benchmark", Path(__file__).resolve().parents[1]/"scripts/benchmark_generation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_plan_never_runs_or_echoes_prompt(tmp_path, capsys):
    path = tmp_path / "input.json"
    path.write_text(GenerationConfig(prompt="private prompt").to_json())
    output = tmp_path / "benchmark"
    assert tool().main(["--config", str(path), "--output-dir", str(output)]) == 0
    raw = capsys.readouterr().out
    assert "private" not in raw
    assert json.loads(raw)["run"] is False
    assert not output.exists()


def test_existing_output_and_unfixed_seed_refused_before_run(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(GenerationConfig(seed=None).to_json())
    assert tool().main(["--config", str(path), "--output-dir", str(tmp_path/"new"), "--run"]) == 1
    path.write_text(GenerationConfig().to_json())
    assert tool().main(["--config", str(path), "--output-dir", str(tmp_path), "--run"]) == 1
    assert not (tmp_path / "run-1").exists()


def test_attention_profile_plans_without_gpu_or_output(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("attention_probe", Path(__file__).resolve().parents[1]/"scripts/profile_generation_attention.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "input.json"
    path.write_text(GenerationConfig(prompt="private prompt").to_json())
    output = tmp_path / "profile.json"
    assert module.main(["--config", str(path), "--output", str(output)]) == 0
    raw = capsys.readouterr().out
    assert "private" not in raw
    assert json.loads(raw)["run"] is False
    assert not output.exists()
