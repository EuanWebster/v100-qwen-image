"""CPU plan-mode tests for scripts/probe_fp16_activations.py (no GPU, no ML)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_fp16_activations.py"


def _plan(extra):
    proc = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--probe-plan",
         "--probe-out", "/tmp/probe.json", "--probe-threshold", "1234.5",
         *extra],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_plan_mode_describes_hooks_streams_and_kv_off():
    doc = _plan(["--reference", "a.png", "--seed", "42"])
    assert doc["schema"] == "qwen-fp16-activation-probe-v1"
    assert doc["mode"] == "plan" and doc["ml_imports"] is False
    assert doc["kv_cache"].startswith("off")
    assert doc["threshold"] == 1234.5 and doc["out"] == "/tmp/probe.json"
    assert doc["streams"] == ["text", "reference_image", "target_image"]
    assert doc["hook_targets"] == [
        "transformer.transformer_blocks[i].",
        "transformer.transformer_blocks[i].attn",
        "transformer.transformer_blocks[i].img_mlp"]
    assert "build_token_metadata" in doc["stream_source"]
    assert "--reference" in doc["passthrough_flags"]


def test_plan_mode_never_echoes_prompt_or_reference_values():
    doc = _plan(["--prompt", "SECRET-PROMPT-TEXT", "--reference", "secret/ref.png"])
    assert "SECRET-PROMPT-TEXT" not in json.dumps(doc)
    assert "secret/ref.png" not in json.dumps(doc)
    assert sorted(doc["passthrough_flags"]) == ["--prompt", "--reference"]
