"""ML-free boundaries and review-scope refusals for the default loader.

No torch/diffusers/transformers import, no model weights, no GPU and no
network occur in this module: the loader's heavy imports live inside
``load`` after its policy and manifest gates.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from qwen_engine import cli
from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.errors import EngineError
from qwen_engine.manifest import ManifestError

HEAVY = ("torch", "diffusers", "transformers")


def _config(tmp_path, **overrides):
    values = dict(
        prompt="a red cube",
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        manifest_path=str(tmp_path / "missing-manifest.json"),
        models_dir=str(tmp_path / "models"),
    )
    values.update(overrides)
    return GenerationConfig(**values)


def test_module_import_stays_ml_free_in_a_fresh_interpreter():
    code = (
        "import sys\n"
        "import qwen_engine.adapters.qwen21_diffusers as mod\n"
        "heavy = sorted(set(sys.modules) & {'torch', 'diffusers', 'transformers'})\n"
        "assert not heavy, heavy\n"
        "assert mod.Qwen21DiffusersAdapter.__module__ == "
        "'qwen_engine.adapters.qwen21_diffusers'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True, text=True, env=dict(os.environ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_factory_returns_the_default_loader_without_heavy_imports():
    adapter = cli._make_adapter()
    assert type(adapter).__module__ == "qwen_engine.adapters.qwen21_diffusers"
    assert not set(HEAVY) & set(sys.modules)


@pytest.mark.parametrize("field,value", [
    ("quant", "q8"),
    ("quant", "q4"),
    ("offload", "streamed"),
    ("offload", "offload"),
    ("attention_backend", "eager"),
    ("attention_backend", "efficient"),
])
def test_unreviewed_execution_policy_refuses_before_manifest_or_ml(
        tmp_path, field, value):
    adapter = cli._make_adapter()
    with pytest.raises(ConfigError) as caught:
        adapter.load(_config(tmp_path, **{field: value}), {})
    assert field in caught.value.fields
    assert not set(HEAVY) & set(sys.modules)


@pytest.mark.parametrize("missing", [
    "output_dir", "cache_dir", "manifest_path", "models_dir",
])
def test_missing_runtime_paths_refuse_before_manifest_or_ml(tmp_path, missing):
    adapter = cli._make_adapter()
    values = {missing: None}
    with pytest.raises(ConfigError) as caught:
        adapter.load(_config(tmp_path, **values), {})
    assert missing in caught.value.fields
    assert not set(HEAVY) & set(sys.modules)


def test_missing_manifest_refuses_before_any_ml_import(tmp_path):
    adapter = cli._make_adapter()
    with pytest.raises(ManifestError):
        adapter.load(_config(tmp_path), {})
    assert not set(HEAVY) & set(sys.modules)


def test_phase_calls_before_load_are_refused():
    adapter = cli._make_adapter()
    with pytest.raises(EngineError):
        adapter.encode(GenerationConfig())
    with pytest.raises(EngineError):
        adapter.denoise(GenerationConfig(), None)
    with pytest.raises(EngineError):
        adapter.decode(None)
    with pytest.raises(EngineError):
        adapter.save(None, GenerationConfig())


def test_cancel_event_binding_and_idempotent_close():
    import threading

    adapter = cli._make_adapter()
    with pytest.raises(ConfigError):
        adapter.set_cancel_event("not-an-event")
    adapter.set_cancel_event(threading.Event())
    adapter.set_cancel_event(None)  # re-binding before load is allowed
    adapter.close()
    adapter.close()  # idempotent after partial/never load


def test_second_load_is_refused_after_close():
    adapter = cli._make_adapter()
    adapter.close()
    with pytest.raises(EngineError):
        adapter.load(GenerationConfig(), {})


def test_config_replacement_keeps_policy_refusal_reachable(tmp_path):
    """replace() on a valid config still yields the reviewed-profile errors."""
    config = _config(tmp_path, quant="q8")
    adapter = cli._make_adapter()
    with pytest.raises(ConfigError):
        adapter.load(replace(config, offload="streamed"), {})


# -- operator host-scope attestation (off by default) -----------------------


def test_loader_defaults_to_no_host_scope_attestation():
    adapter = cli._make_adapter()
    assert adapter._trust_unbounded_host is False


def test_loader_accepts_the_explicit_operator_attestation():
    adapter = cli._make_adapter(trust_unbounded_host=True)
    assert adapter._trust_unbounded_host is True


@pytest.mark.parametrize("value", ["yes", 1, 0, None])
def test_loader_refuses_a_non_boolean_attestation(value):
    with pytest.raises(ConfigError, match="explicit boolean"):
        cli._make_adapter(trust_unbounded_host=value)


def test_cli_exposes_the_attestation_flag_by_default_off():
    parser = cli.build_parser()
    assert parser.parse_args([]).attest_unbounded_host is False
    assert parser.parse_args(["--attest-unbounded-host"]).attest_unbounded_host is True


def test_cli_notice_names_the_attestation_when_used(tmp_path, capsys, monkeypatch):
    """The run transcript must show the attestation before generation; the
    adapter factory is stubbed so no model is touched."""
    import json

    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps({"schema": "qwen_engine.manifest.v1",
                                 "artifacts": []}))
    (tmp_path / "models").mkdir()
    captured = {}

    class FakeAdapter:
        def set_cancel_event(self, event):
            pass

    def factory(**kwargs):
        captured.update(kwargs)
        raise EngineError("stop after the notice")

    monkeypatch.setattr(cli, "_make_adapter", factory)
    monkeypatch.setattr(cli, "_verify_assets", lambda config: type(
        "R", (), {"resolved": {}})())
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "prompt": "a red cube", "manifest_path": str(mpath),
        "models_dir": str(tmp_path / "models"),
        "output_dir": str(tmp_path / "out")}))
    assert cli.main(["--config", str(cfg), "--run",
                     "--attest-unbounded-host"]) == 1
    err = capsys.readouterr().err
    assert "attesting that this process tree has no cgroup memory limit" in err
    assert captured == {"trust_unbounded_host": True}


# -- transformer release before the VAE decode ------------------------------


class _RecordingInner:
    def __init__(self, order):
        self.order = order

    def denoise(self, config, conditioning):
        self.order.append("denoise")
        return ["latents"]

    def decode(self, latents):
        self.order.append("decode")
        return "image"


def test_denoise_releases_the_transformer_before_decode():
    """Single-use loader: the transformer is dead weight during decode, and
    releasing it is what lets the untiled (artifact-free) decode fit."""
    from types import SimpleNamespace

    adapter = cli._make_adapter()
    order = []
    adapter._inner = _RecordingInner(order)
    transformer = SimpleNamespace()
    adapter._pipeline = SimpleNamespace(transformer=transformer)

    latents = adapter.denoise(GenerationConfig(), None)
    assert adapter._pipeline.transformer is None
    adapter.decode(latents)
    assert order == ["denoise", "decode"]


def test_release_is_idempotent_without_a_pipeline_or_transformer():
    from types import SimpleNamespace

    adapter = cli._make_adapter()
    adapter._release_denoiser()  # no pipeline at all
    adapter._pipeline = SimpleNamespace(transformer=None)
    adapter._release_denoiser()  # already released
    adapter._pipeline = SimpleNamespace()
    adapter._release_denoiser()  # attribute absent
