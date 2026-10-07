"""Shared fixtures. All paths point at tmp_path — tests never write elsewhere (R7).

When the suite runs through ``run_cpu_tests.sh`` (``QWEN_CPU_TEST_GUARD=1``)
the session fixture below *enforces* the guard's integrity in the pytest
process itself; subprocess fake workers are covered by the same guard
installing itself at interpreter startup via ``PYTHONPATH`` sitecustomize
(see ``tests/cpu_guard/sitecustomize.py`` and ``tests/test_cpu_guard.py``).
"""
from __future__ import annotations

import os

import pytest

from qwen_engine.config import GenerationConfig


@pytest.fixture(scope="session", autouse=True)
def r7_cpu_guard():
    """Fail fast if the CPU guard is announced but not actually wired up."""
    if os.environ.get("QWEN_CPU_TEST_GUARD") != "1":
        return  # plain pytest: this fixture adds no runtime behaviour
    import sitecustomize as guard

    assert getattr(guard, "GUARD_NAME", None) == "qwen-r7-cpu-guard", \
        "QWEN_CPU_TEST_GUARD=1 but the guard sitecustomize is not importable"
    assert getattr(guard, "GUARDED", False) and getattr(guard, "INSTALLED", False), \
        "guard env set but the audit hook was never installed"
    assert guard.ROOTS, "guard enabled but QWEN_CPU_TMP_ROOTS is empty"
    for var in ("TMPDIR", "HOME", "XDG_CACHE_HOME"):
        value = os.environ.get(var)
        assert value, f"CPU runner must redirect {var} into a temp root"
        rp = os.path.realpath(value)
        assert any(rp == root or rp.startswith(root + os.sep)
                   for root in guard.ROOTS), \
            f"{var}={value!r} is outside the CPU test temp roots"


@pytest.fixture
def base_config(tmp_path) -> GenerationConfig:
    """A valid config with every runtime path redirected into tmp_path."""
    return GenerationConfig(
        prompt="a red cube on a table",
        output_dir=str(tmp_path / "output"),
        cache_dir=str(tmp_path / "cache"),
        log_dir=str(tmp_path / "logs"),
        benchmarks_dir=str(tmp_path / "benchmarks"),
    )
