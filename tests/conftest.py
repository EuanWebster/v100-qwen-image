"""Shared fixtures. All paths point at tmp_path — tests never write elsewhere (R7)."""
from __future__ import annotations

import pytest

from qwen_engine.config import GenerationConfig


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
