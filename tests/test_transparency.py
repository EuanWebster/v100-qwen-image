"""Native transparency CORE tests (config wrapper, encoder projection, CLI).

CPU-only, ML-free. Uses the repo reference manifest exactly like
test_encoder_identity.py; no GPU, no downloads, no tiled-memory claims.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from qwen_engine import cli
from qwen_engine import encoder_identity as ei
from qwen_engine.config import (
    ALPHA_OUTPUT_FORMATS,
    TRANSPARENCY_PROMPT_PREFIX,
    TRANSPARENCY_PROMPT_SUFFIX,
    TRANSPARENCY_WRAPPER_REVISION,
    ConfigError,
    GenerationConfig,
    OutputFormat,
)
from qwen_engine.encoder import cache_key
from qwen_engine.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
WRAPPED = TRANSPARENCY_PROMPT_PREFIX + "a red cube" + TRANSPARENCY_PROMPT_SUFFIX


@pytest.fixture
def manifest():
    return load_manifest(ROOT / "models/manifest.json")


def _cli(*argv):
    return cli.build_parser().parse_args(list(argv))


# --- strict bool + defaults ---------------------------------------------------

def test_default_is_false_and_jpg_unchanged():
    config = GenerationConfig()
    assert config.transparent_background is False
    assert config.output_format is OutputFormat.JPG
    assert config.effective_prompt == config.prompt


@pytest.mark.parametrize("bad", [1, 0, "true", "True", "", None, 0.0, [], {}])
def test_strict_bool_refusals(bad):
    with pytest.raises(ConfigError) as exc:
        GenerationConfig(transparent_background=bad)
    assert "transparent_background" in exc.value.fields


def test_wrapper_revision_constant_shape():
    assert isinstance(TRANSPARENCY_WRAPPER_REVISION, str)
    assert TRANSPARENCY_WRAPPER_REVISION == "qwen-engine.transparent-wrapper.v1"
    assert ALPHA_OUTPUT_FORMATS == {OutputFormat.PNG, OutputFormat.WEBP}


# --- effective prompt ----------------------------------------------------------

def test_effective_prompt_wraps_only_when_flag_true():
    plain = GenerationConfig(prompt="a red cube")
    assert plain.effective_prompt == "a red cube"
    trans = GenerationConfig(prompt="a red cube", transparent_background=True,
                             output_format="png")
    assert trans.prompt == "a red cube"  # original preserved verbatim
    assert trans.effective_prompt == WRAPPED
    assert trans.effective_prompt.startswith(TRANSPARENCY_PROMPT_PREFIX)
    assert trans.effective_prompt.endswith(TRANSPARENCY_PROMPT_SUFFIX)


def test_repeated_calls_never_mutate_or_accumulate():
    trans = GenerationConfig(prompt="a red cube", transparent_background=True,
                             output_format="webp")
    first, second = trans.effective_prompt, trans.effective_prompt
    assert first == second == WRAPPED  # no accumulation
    assert trans.prompt == "a red cube"
    assert trans.to_dict()["prompt"] == "a red cube"


def test_whitespace_preserved_exactly():
    raw = "  spaced   prompt \n"
    trans = GenerationConfig(prompt=raw, transparent_background=True,
                             output_format="png")
    assert trans.effective_prompt == TRANSPARENCY_PROMPT_PREFIX + raw \
        + TRANSPARENCY_PROMPT_SUFFIX
    assert GenerationConfig(prompt=raw).effective_prompt == raw


# --- serialization --------------------------------------------------------------

def test_roundtrip_includes_bool():
    trans = GenerationConfig(prompt="x", transparent_background=True,
                             output_format="png")
    assert json.loads(trans.to_json())["transparent_background"] is True
    assert GenerationConfig.from_json(trans.to_json()) == trans
    assert GenerationConfig.from_dict(trans.to_dict()) == trans


def test_old_json_defaults_false():
    old = json.dumps({"prompt": "legacy", "output_format": "jpg"})
    config = GenerationConfig.from_json(old)
    assert config.transparent_background is False
    assert config.effective_prompt == "legacy"


# --- format policy ----------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["png", "webp"])
def test_alpha_formats_allowed(fmt):
    config = GenerationConfig(transparent_background=True, output_format=fmt)
    assert config.transparent_background is True


def test_jpg_refused_with_transparency():
    with pytest.raises(ConfigError) as exc:
        GenerationConfig(transparent_background=True, output_format="jpg")
    assert "output_format" in exc.value.fields
    assert any("jpg" in str(e) for e in exc.value.errors)
    # Explicit default-JPG construction is the same refusal.
    with pytest.raises(ConfigError):
        GenerationConfig(transparent_background=True)
    # No transparency: JPG default stays perfectly valid.
    assert GenerationConfig(output_format="jpg").output_format is OutputFormat.JPG


# --- encoder projection ------------------------------------------------------------

def test_projection_sends_effective_prompt_with_identity(manifest):
    plain = GenerationConfig(prompt="a red cube")
    trans = GenerationConfig(prompt="a red cube", transparent_background=True,
                             output_format="png")
    req_plain = ei.project_encoder_request(plain, manifest, encoder_dtype="fp16")
    req_trans = ei.project_encoder_request(trans, manifest, encoder_dtype="fp16")
    assert req_trans.prompt == WRAPPED
    assert req_plain.prompt == "a red cube"
    assert cache_key(req_trans) != cache_key(req_plain)  # cache binds actual text
    # Deterministic identity: same config -> same key/evidence.
    again = ei.project_encoder_request(
        replace(trans, output_format="webp"), manifest, encoder_dtype="fp16")
    assert cache_key(again) == cache_key(req_trans)


def test_negative_prompt_and_cfg_unchanged(manifest):
    trans = GenerationConfig(prompt="a red cube", negative_prompt="ugly",
                             true_cfg_scale=2.0, transparent_background=True,
                             output_format="png")
    req = ei.project_encoder_request(trans, manifest, encoder_dtype="fp16")
    assert req.negative_prompt == "ugly"  # never wrapped
    assert req.kwargs["cfg_active"] is True
    assert req.dtype == "fp16"  # dtype/model identity untouched
    off = GenerationConfig(prompt="a red cube", negative_prompt="ugly",
                           true_cfg_scale=1.0, transparent_background=True,
                           output_format="png")
    assert ei.project_encoder_request(off, manifest,
                                      encoder_dtype="fp16").kwargs["cfg_active"] is False


# --- CLI ------------------------------------------------------------------------------

def test_cli_default_none_keeps_jpg():
    config = cli._load_config(_cli())
    assert config.transparent_background is False
    assert config.output_format is OutputFormat.JPG


def test_cli_transparent_auto_png():
    config = cli._load_config(_cli("--transparent", "--prompt", "a red cube"))
    assert config.transparent_background is True
    assert config.output_format is OutputFormat.PNG
    assert config.effective_prompt == WRAPPED


@pytest.mark.parametrize("fmt", ["png", "webp"])
def test_cli_transparent_keeps_existing_alpha_format(fmt):
    assert cli._load_config(_cli("--transparent", "--output-format", fmt)
                            ).output_format.value == fmt


def test_cli_explicit_jpg_refused():
    with pytest.raises(ConfigError) as exc:
        cli._load_config(_cli("--transparent", "--output-format", "jpg"))
    assert "output_format" in exc.value.fields


def test_cli_config_jpg_with_transparent_refused_no_hidden_change(tmp_path):
    # JSON true flag without an alpha format must refuse clearly; the shared
    # config is never silently rewritten.
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"transparent_background": True,
                                "output_format": "jpg"}))
    with pytest.raises(ConfigError) as exc:
        cli._load_config(_cli("--config", str(path)))
    assert "output_format" in exc.value.fields
    # --no-transparent rescues the same file and keeps jpg.
    rescued = cli._load_config(_cli("--no-transparent", "--config", str(path)))
    assert rescued.output_format is OutputFormat.JPG


def test_cli_no_transparent_overrides_config_true_and_keeps_format(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"transparent_background": True,
                                "output_format": "webp"}))
    config = cli._load_config(_cli("--no-transparent", "--config", str(path)))
    assert config.transparent_background is False
    assert config.output_format is OutputFormat.WEBP
    assert config.effective_prompt == config.prompt

