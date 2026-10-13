#!/usr/bin/env python3
"""Offline actual CUDA-unavailable refusal witness, not physical host inventory.

No model reads, driver installation, NVML query or kernel is permitted here.
If this container starts seeing usable CUDA, this absence-specific check FAILS
so it cannot silently substitute for the pending real V100 acceptance.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

if sys.flags.optimize:
    raise SystemExit("Verification assertions must be enabled")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from local_processor_check import install_check_guard
    run = Path(tempfile.mkdtemp(prefix="qwen-hardware-refusal-"))
    try:
        install_check_guard(run)
        from qwen_engine import hardware_probe as probe, encoder_identity as ei
        assert not set(ei.ML_MODULE_NAMES) & set(sys.modules)
        def forbidden(*args):
            raise AssertionError("CPU absence check attempted a kernel or NVML query")
        with patch.object(probe, "_kernel", forbidden), patch.object(probe, "_nvml_snapshot", forbidden):
            try:
                probe.probe_v100()
            except probe.HardwareProbeError as exc:
                assert str(exc) == "reviewed_cuda_device_unavailable", str(exc)
            else:
                raise AssertionError("expected actual CUDA-unavailable refusal")
        import torch
        assert torch.cuda.is_initialized() is False
        print("PASS actual reviewed Torch/CUDA device is unavailable in this container; typed refusal")
        print("PASS no model reads, GPU kernel, NVML query or CUDA context initialization")
        print("Hardware absence witness passed; NOT host inventory, V100 execution or memory-fit acceptance")
        return 0
    finally:
        shutil.rmtree(run)


if __name__ == "__main__":
    raise SystemExit(main())
