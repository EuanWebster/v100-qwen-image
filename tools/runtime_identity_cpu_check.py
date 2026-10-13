#!/usr/bin/env python3
"""Real pinned CPU runtime fingerprint witness; no CUDA, weights or fit pass."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

if sys.flags.optimize:
    raise SystemExit("Verification assertions must be enabled")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def checks():
    from qwen_engine import encoder_identity as ei
    from qwen_engine.runtime_identity import observe_generation_runtime, _torch_document
    assert not set(ei.ML_MODULE_NAMES) & set(sys.modules)
    ei.validate_installed_packages(ei.observe_installed_packages())
    import torch
    assert torch.cuda.is_initialized() is False
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU fingerprint attempted CUDA initialization/workspace query")
    with patch.object(torch.cuda, "_lazy_init", forbidden), \
         patch.object(torch.backends.cuda, "cublas_workspace_size", forbidden), \
         patch.object(torch.backends.cuda, "cublaslt_workspace_size", forbidden):
        first = observe_generation_runtime()
        assert first == observe_generation_runtime() and first.cuda_context_initialized is False
        assert len(first.sha256) == 64 and torch.cuda.is_initialized() is False
        doc = _torch_document()
        assert doc["cublas_workspace"] is None and doc["cublaslt_workspace"] is None
        print("PASS real pinned metadata/source/host/CPU/settings/environment identity is stable and ML-lazy")
        print("PASS CPU identity skips CUDA initialization and workspace queries; unknown is not zero")
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2 if threads != 2 else 1)
            changed = observe_generation_runtime()
            assert changed.torch_execution_sha256 != first.torch_execution_sha256 and changed.sha256 != first.sha256
        finally:
            torch.set_num_threads(threads)
        assert observe_generation_runtime() == first
        print("PASS actual Torch thread change invalidates identity; restored settings restore digest")
        name = "PYTORCH_RUNTIME_IDENTITY_FIXTURE"
        previous = os.environ.get(name)
        try:
            os.environ[name] = "private disposable fixture value"
            changed = observe_generation_runtime()
            assert changed.environment_sha256 != first.environment_sha256 and changed.sha256 != first.sha256
            assert "disposable" not in str(changed.to_dict())
        finally:
            if previous is None:
                os.environ.pop(name)
            else:
                os.environ[name] = previous
        assert observe_generation_runtime() == first
        print("PASS relevant process environment change invalidates identity; raw values never exported")
    print("Runtime CPU witness passed; NOT CUDA execution, GPU workspace evidence, resource budget or calibration")
    return 0


def main():
    from local_processor_check import install_check_guard
    run = Path(tempfile.mkdtemp(prefix="qwen-runtime-identity-"))
    try:
        install_check_guard(run)
        return checks()
    finally:
        shutil.rmtree(run)


if __name__ == "__main__":
    raise SystemExit(main())
