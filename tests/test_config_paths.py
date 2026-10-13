"""Direct Path inputs normalize before local validation, without resolving."""
from pathlib import Path

import pytest

from qwen_engine.config import GenerationConfig, ConfigError


@pytest.mark.parametrize("field", ["output_dir", "cache_dir", "log_dir", "benchmarks_dir", "models_dir", "manifest_path"])
def test_direct_path_coerces_before_validation(tmp_path, field):
    path = tmp_path / "configured"
    if field == "manifest_path":
        path.write_bytes(b"not read by configuration")
    else:
        path.mkdir()
    config = GenerationConfig(**{field: path})
    assert getattr(config, field) == str(path)
    assert type(getattr(config, field)) is str


@pytest.mark.parametrize("field", ["output_dir", "cache_dir", "log_dir", "benchmarks_dir", "models_dir", "manifest_path"])
def test_nonexistent_relative_path_preserves_lexical_semantics(tmp_path, monkeypatch, field):
    monkeypatch.chdir(tmp_path)
    path = Path("missing/../chosen")
    assert getattr(GenerationConfig(**{field: path}), field) == "missing/../chosen"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["output_dir", "manifest_path"])
def test_wrong_existing_kind_still_has_typed_field_error(tmp_path, field):
    path = tmp_path / "wrong"
    if field == "output_dir":
        path.write_bytes(b"file")
    else:
        path.mkdir()
    with pytest.raises(ConfigError) as caught:
        GenerationConfig(**{field: path})
    assert field in caught.value.fields
