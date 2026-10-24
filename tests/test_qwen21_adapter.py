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


def test_q8_requires_explicit_sidecar_before_assets_or_ml(tmp_path):
    with pytest.raises(ConfigError) as caught:
        cli._make_adapter().load(_config(tmp_path, quant="q8"), {})
    assert "q8_manifest_path" in caught.value.fields
    assert not set(HEAVY) & set(sys.modules)


def test_q8_remote_kernel_mode_refuses_before_ml_import(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFUSERS_GGUF_CUDA_KERNELS", "true")
    with pytest.raises(EngineError, match="CUDA_KERNELS=false"):
        cli._make_adapter().load(_config(tmp_path, quant="q8", q8_manifest_path=str(tmp_path / "q8.json")), {})
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


# -- phase-exclusive residency (VAE off the card during denoise) ------------


class _FakeVae:
    def __init__(self, name):
        self.name = name
        self.tiling = False

    def to(self, **kwargs):
        self.moved = kwargs
        return self

    def eval(self):
        return self

    def enable_tiling(self, **kwargs):
        self.tiling = True
        self.tiling_args = kwargs


class _FakeVaeClass:
    def __init__(self, order):
        self.order = order
        self.calls = []

    def from_pretrained(self, path, **kwargs):
        self.order.append("reload-vae")
        self.calls.append((path, kwargs))
        return _FakeVae(path)


class _VaeInner:
    def __init__(self, order, pipeline):
        self.order = order
        self.pipeline = pipeline

    def denoise(self, config, conditioning):
        self.order.append(("denoise", self.pipeline.vae))
        return ["latents"]

    def decode(self, latents):
        self.order.append(("decode", self.pipeline.vae))
        return "image"


def test_denoise_takes_the_vae_off_the_card_and_decode_restores_it():
    from types import SimpleNamespace

    order = []
    adapter = cli._make_adapter()
    pipeline = SimpleNamespace(transformer=SimpleNamespace(), vae=_FakeVae("resident"))
    adapter._pipeline = pipeline
    adapter._inner = _VaeInner(order, pipeline)
    adapter._vae_cls = _FakeVaeClass(order)
    adapter._vae_dir = "/models/vae"
    adapter._torch = SimpleNamespace(float16="fp16", float32="fp32",
                                     device=lambda *a: ("cuda", a[1]))
    adapter._tiled_decode = True

    latents = adapter.denoise(GenerationConfig(), None)
    assert order[0] == ("denoise", None)        # VAE already gone
    assert pipeline.transformer is None         # transformer gone afterwards
    adapter.decode(latents)
    assert order[1] == "reload-vae"
    assert isinstance(order[2][1], _FakeVae)    # decoded with the restored VAE
    assert order[2][1].tiling is True
    assert order[2][1].tiling_args == {
        "tile_sample_min_height": 1024, "tile_sample_min_width": 1024,
        "tile_sample_stride_height": 768, "tile_sample_stride_width": 768}
    path, kwargs = adapter._vae_cls.calls[0]
    assert path == "/models/vae"
    # Tiled decode restores the VAE in fp32: the fp16 tiled path is
    # non-finite at 2048², and the transformer is gone by decode time.
    assert kwargs == {"torch_dtype": "fp32", "local_files_only": True}
    assert order[2][1].moved == {"device": ("cuda", 0), "dtype": "fp32"}


def test_untiled_decode_restores_the_vae_in_fp16():
    from types import SimpleNamespace

    order = []
    adapter = cli._make_adapter()
    pipeline = SimpleNamespace(vae=None)
    adapter._pipeline = pipeline
    adapter._vae_cls = _FakeVaeClass(order)
    adapter._vae_dir = "/models/vae"
    adapter._torch = SimpleNamespace(float16="fp16", float32="fp32",
                                     device=lambda *a: ("cuda", a[1]))
    adapter._tiled_decode = False
    adapter._restore_vae()
    path, kwargs = adapter._vae_cls.calls[0]
    assert path == "/models/vae"
    assert kwargs == {"torch_dtype": "fp16", "local_files_only": True}
    assert pipeline.vae.moved == {"device": ("cuda", 0), "dtype": "fp16"}
    assert pipeline.vae.tiling is False


def test_vae_release_is_idempotent_and_restore_is_skipped_when_present():
    from types import SimpleNamespace

    adapter = cli._make_adapter()
    pipeline = SimpleNamespace(vae=_FakeVae("resident"))
    adapter._pipeline = pipeline
    adapter._vae_cls = None       # no loader available: restore must no-op
    adapter._release_vae()
    assert pipeline.vae is None
    adapter._release_vae()        # already released
    adapter._restore_vae()        # nothing to restore with
    assert pipeline.vae is None


# -- high-token MLP chunking (exact, token-wise) ----------------------------


class _Rows:
    """Minimal ML-free stand-in for a (1, tokens, width) tensor."""

    def __init__(self, rows):
        self.rows = list(rows)

    @property
    def shape(self):
        return (1, len(self.rows), 64)

    def __getitem__(self, item):
        batch, rest = item
        assert batch == slice(None)
        return _Rows(self.rows[rest])


class _RecordingMlp:
    def __init__(self):
        self.calls = []

    def forward(self, hidden):
        self.calls.append(len(hidden.rows))
        return _Rows([row + 1 for row in hidden.rows])


def _chunking_fixture():
    from types import SimpleNamespace
    from qwen_engine.adapters import qwen21_diffusers as qd

    first, second = _RecordingMlp(), _RecordingMlp()
    transformer = SimpleNamespace(transformer_blocks=[
        SimpleNamespace(img_mlp=first), SimpleNamespace(img_mlp=second)])
    fake_torch = SimpleNamespace(
        cat=lambda parts, dim: _Rows(row for part in parts for row in part.rows))
    return qd, transformer, fake_torch, first, second


def test_high_token_denoise_chunks_the_swiglu_mlp_exactly():
    qd, transformer, fake_torch, first, second = _chunking_fixture()
    assert qd._apply_mlp_chunking(fake_torch, transformer, 2048) == 2
    out = first.forward(_Rows(range(5000)))
    assert first.calls == [2048, 2048, 904]
    assert out.rows == [row + 1 for row in range(5000)]
    assert second.calls == []          # wrapped, but not called by this test
    first.forward(_Rows([1, 2]))       # short sequence: one unchunked call
    assert first.calls[-1] == 2


def test_mlp_chunking_is_idempotent_and_refuses_an_unknown_layout():
    qd, transformer, fake_torch, first, _ = _chunking_fixture()
    assert qd._apply_mlp_chunking(fake_torch, transformer, 2048) == 2
    assert qd._apply_mlp_chunking(fake_torch, transformer, 512) == 0
    first.forward(_Rows(range(3000)))
    assert first.calls == [2048, 952]   # original chunk retained, never doubled
    from types import SimpleNamespace
    with pytest.raises(EngineError, match="chunk"):
        qd._apply_mlp_chunking(fake_torch, SimpleNamespace(transformer_blocks=[
            SimpleNamespace()]), 2048)
