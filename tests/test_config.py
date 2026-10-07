"""Task 1.1 acceptance: frozen GenerationConfig, lossless deterministic JSON,
field-level validation, latent-token math. CPU-only, stdlib-only (M1 criteria 1–2)."""
from __future__ import annotations

import dataclasses
import json
import sys

import pytest

from qwen_engine.config import (
    AttentionBackend,
    ConfigError,
    GenerationConfig,
    OffloadMode,
    Quant,
)

HEAVY_MODULES = ("torch", "diffusers", "transformers")


def _err_fields(exc: pytest.ExceptionInfo[ConfigError]) -> set[str]:
    return {e.field for e in exc.value.errors}


# --- immutability / structure -------------------------------------------------


def test_config_is_frozen(base_config):
    with pytest.raises(dataclasses.FrozenInstanceError):
        base_config.width = 2048  # type: ignore[misc]


def test_to_dict_from_dict_roundtrip_preserves_equality(base_config):
    assert GenerationConfig.from_dict(base_config.to_dict()) == base_config


# --- lossless deterministic JSON ----------------------------------------------


def test_json_roundtrip_is_lossless(base_config):
    again = GenerationConfig.from_json(base_config.to_json())
    assert again == base_config


def test_json_is_key_sorted_and_deterministic(base_config):
    text = base_config.to_json()
    assert text == json.dumps(json.loads(text), sort_keys=True)
    assert text == base_config.to_json()  # deterministic across calls


def test_from_dict_accepts_plain_strings_for_enums():
    cfg = GenerationConfig.from_dict(
        {"prompt": "p", "quant": "q8", "offload": "streamed",
         "attention_backend": "sdpa"}
    )
    assert cfg.quant is Quant.Q8
    assert cfg.offload is OffloadMode.STREAMED
    assert cfg.attention_backend is AttentionBackend.SDPA
    # enum values serialize as plain strings, and round-trip back to members
    assert json.loads(cfg.to_json())["quant"] == "q8"
    assert GenerationConfig.from_json(cfg.to_json()) == cfg


# --- field-level validation (each invalid field named individually) -----------


@pytest.mark.parametrize("field,value", [
    ("quant", "int4"),                       # bad enum
    ("offload", "aggressive"),               # bad enum
    ("attention_backend", "flash"),          # bad enum
    ("seed", -1),                            # negative seed
    ("seed", 2**64),                         # seed range
    ("steps", 0),                            # zero steps
    ("steps", -5),                           # negative steps
    ("width", 1000),                         # not multiple of 32
    ("height", 100),                         # not multiple of 32
    ("width", 32),                           # below minimum dimension
    ("height", 8192),                        # above maximum dimension
    ("prompt", ""),                          # empty prompt
    ("prompt", 42),                          # wrong type
    ("vae_tiled_decode", "yes"),             # wrong type
    ("width", True),                         # bool is not an int here
])
def test_invalid_field_names_field_in_error(field, value):
    data = {} if field == "prompt" else {"prompt": "p"}
    data[field] = value
    with pytest.raises(ConfigError) as exc:
        GenerationConfig.from_dict(data)
    assert field in _err_fields(exc), exc.value.errors
    # the field name appears verbatim in the rendered message too
    assert field in str(exc.value)


def test_enum_error_lists_valid_choices():
    with pytest.raises(ConfigError) as exc:
        GenerationConfig(quant="int4")
    err = next(e for e in exc.value.errors if e.field == "quant")
    for choice in ("fp16", "q8", "q4"):
        assert choice in err.message


def test_multiple_bad_fields_are_all_reported():
    with pytest.raises(ConfigError) as exc:
        GenerationConfig(steps=0, seed=-1, width=1000)
    assert {"steps", "seed", "width"} <= _err_fields(exc)


def test_dimension_error_says_multiple_of_32():
    with pytest.raises(ConfigError) as exc:
        GenerationConfig(height=1000)
    err = next(e for e in exc.value.errors if e.field == "height")
    assert "32" in err.message


def test_unknown_field_rejected_by_name():
    with pytest.raises(ConfigError) as exc:
        GenerationConfig.from_dict({"prompt": "p", "temperture": 0.7})
    assert "temperture" in _err_fields(exc)


def test_malformed_json_is_a_config_error():
    with pytest.raises(ConfigError) as exc:
        GenerationConfig.from_json("{not json")
    assert "config" in _err_fields(exc)


def test_negative_seed_constructed_directly_is_rejected():
    with pytest.raises(ConfigError):
        GenerationConfig(prompt="p", seed=-3)


def test_seed_none_is_valid():
    assert GenerationConfig(prompt="p", seed=None).seed is None


# --- latent token math (C6: /16 spatial compression) ---------------------------


@pytest.mark.parametrize("size,tokens", [
    ((1024, 1024), 4096),
    ((2048, 2048), 16384),
    ((512, 320), 32 * 20),
    ((64, 64), 4 * 4),
])
def test_latent_token_math(size, tokens):
    cfg = GenerationConfig(prompt="p", width=size[0], height=size[1])
    assert cfg.latent_tokens == tokens


def test_token_math_1024_squared_is_exactly_4096():
    assert GenerationConfig(width=1024, height=1024).latent_tokens == 4096


def test_token_math_2048_squared_is_exactly_16384():
    assert GenerationConfig(width=2048, height=2048).latent_tokens == 16384


# --- import boundary: config module is stdlib-only -----------------------------


def test_config_import_pulls_in_no_heavy_modules():
    import qwen_engine.config  # noqa: F401
    assert not set(HEAVY_MODULES) & set(sys.modules)
