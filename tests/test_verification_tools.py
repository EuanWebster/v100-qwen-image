"""Verification witnesses must not report PASS when Python strips asserts."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSERTION_TOOLS = ("encoder_identity_check", "encoder_assets_check",
                   "local_processor_check",
                   "component_cpu_check",
                   "synthetic_ml_encoder_check", "synthetic_worker_check",
                   "synthetic_encoder_client_check", "synthetic_encoder_client_child")


@pytest.mark.parametrize("tool", ASSERTION_TOOLS)
@pytest.mark.parametrize("mode", ["-O", "-OO", "environment"])
def test_optimized_verification_refuses_before_work(tmp_path, tool, mode):
    env = dict(os.environ)
    flags = [] if mode == "environment" else [mode]
    env["PYTHONOPTIMIZE"] = "1" if mode == "environment" else ""
    result = subprocess.run([sys.executable, "-B", *flags, str(ROOT / "tools" / (tool + ".py"))],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert "Verification assertions must be enabled" in result.stderr
    assert "PASS" not in result.stdout
    assert "Traceback" not in result.stderr
    assert not list(tmp_path.iterdir())
