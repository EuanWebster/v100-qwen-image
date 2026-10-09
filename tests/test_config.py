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


# --- steps / true-CFG defaults and semantics --------------------------------------


def test_default_steps_is_40_official_qwen_example():
    assert GenerationConfig().steps == 40
    assert json.loads(GenerationConfig().to_json())["steps"] == 40


def test_true_cfg_scale_defaults_to_1():
    assert GenerationConfig().true_cfg_scale == 1.0
    assert GenerationConfig.from_json(GenerationConfig().to_json()).true_cfg_scale == 1.0


@pytest.mark.parametrize("value", [0.5, -1.0, 0, True, "2.0", float("inf"),
                                  float("nan"), None])
def test_invalid_true_cfg_scale_names_field(value):
    with pytest.raises(ConfigError) as exc:
        GenerationConfig(prompt="p", true_cfg_scale=value)
    assert "true_cfg_scale" in _err_fields(exc)


@pytest.mark.parametrize("value", [1, 1.0, 2.5, 1000.0])
def test_valid_true_cfg_scale_accepted(value):
    assert GenerationConfig(prompt="p", true_cfg_scale=value).true_cfg_scale == value


@pytest.mark.parametrize("negative,scale,active", [
    ("", 1.0, False),
    ("ugly", 1.0, False),        # scale 1.0 -> CFG off, negative unused
    ("   ", 2.0, False),         # blank negative prompt counts as absent
    ("", 2.0, False),           # no negative -> nothing to guide against
    ("ugly", 1.01, True),
    ("ugly", 4.5, True),
])
def test_cfg_active_requires_negative_and_scale_above_1(negative, scale, active):
    cfg = GenerationConfig(prompt="p", negative_prompt=negative,
                           true_cfg_scale=scale)
    assert cfg.cfg_active is active


# --- path field validation ---------------------------------------------------------


@pytest.mark.parametrize("field", ["output_dir", "cache_dir", "log_dir",
                                   "benchmarks_dir", "models_dir", "manifest_path"])
@pytest.mark.parametrize("value", ["", "   ", "a\x00b"])
def test_path_fields_reject_blank_whitespace_and_nul(field, value):
    with pytest.raises(ConfigError) as exc:
        GenerationConfig(prompt="p", **{field: value})
    assert field in _err_fields(exc)


def test_dir_field_existing_as_file_is_rejected(tmp_path):
    as_file = tmp_path / "not_a_dir"
    as_file.write_text("x")
    with pytest.raises(ConfigError) as exc:
        GenerationConfig(prompt="p", output_dir=str(as_file))
    assert "output_dir" in _err_fields(exc)
    assert "directory" in str(exc.value)


def test_manifest_path_existing_as_dir_is_rejected(tmp_path):
    with pytest.raises(ConfigError) as exc:
        GenerationConfig(prompt="p", manifest_path=str(tmp_path))
    assert "manifest_path" in _err_fields(exc)
    assert "file" in str(exc.value)


def test_path_validation_never_creates_anything(tmp_path):
    target = tmp_path / "would_be_created"
    GenerationConfig(prompt="p", output_dir=str(target),
                     cache_dir=str(tmp_path / "cache"))
    assert not target.exists()
    assert not (tmp_path / "cache").exists()
    assert list(tmp_path.iterdir()) == []


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


# --- second senior-review pass: overflow-proof numerics ----------------------


@pytest.mark.parametrize("bad", [10 ** 1000, 10 ** 400, -(10 ** 400)])
def test_true_cfg_scale_huge_int_is_config_error_not_overflow(bad):
    """math.isfinite(10**1000) raises OverflowError at float conversion;
    it must surface as a field-level ConfigError (CLI exit 2), never as
    OverflowError (CLI exit 1)."""
    with pytest.raises(ConfigError, match="true_cfg_scale"):
        GenerationConfig(true_cfg_scale=bad)
    with pytest.raises(ConfigError, match="true_cfg_scale"):
        GenerationConfig.from_dict({"true_cfg_scale": bad})


# --- json parser safety limits at the parse boundary ------------------------


@pytest.mark.parametrize("case, doc", [
    ("digit-limit", '{"prompt": "p", "seed": ' + "9" * 5000 + "}"),
    ("deep-nesting",
     '{"prompt": "p", "seed": ' + "[" * 2000 + "0" + "]" * 2000 + "}"),
], ids=["digit-limit", "deep-nesting"])
def test_json_parser_safety_limits_are_typed_config_errors(case, doc):
    if case == "digit-limit" and not getattr(
            sys, "get_int_max_str_digits", lambda: 0)():
        pytest.skip("integer-string digit limit disabled")
    with pytest.raises(ConfigError) as exc:
        GenerationConfig.from_json(doc)
    assert type(exc.value) is ConfigError


@pytest.mark.parametrize("boom", [
    ValueError("exceeds the integer-string digit limit"),
    RecursionError("decoder nesting limit"),
], ids=["injected-ValueError", "injected-RecursionError"])
def test_injected_parser_failures_wrap_as_config_errors(monkeypatch, boom):
    """Portable wrapper probe (identical behaviour on 3.11 and 3.13): any
    ValueError / RecursionError out of the decoder is a typed ConfigError."""
    def raiser(*args, **kwargs):
        raise boom

    monkeypatch.setattr("json.loads", raiser)
    with pytest.raises(ConfigError):
        GenerationConfig.from_json('{"prompt": "p"}')
