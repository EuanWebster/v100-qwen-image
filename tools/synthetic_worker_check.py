#!/usr/bin/env python3
"""Real upstream/codec through worker with tiny injected loading/budget/policy.

No actual processor/checkpoint load, production overrides, GPU, or downloads.
Run separately on each pinned ML interpreter with -B. Reusable test in git;
fixtures alone are temporary. Loading, vision policy and budget seams are
patched ONLY in this disposable test process, never exposed through CLI flags.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

if sys.flags.optimize:
    raise SystemExit("Verification assertions must be enabled; unset PYTHONOPTIMIZE and do not use -O/-OO")


def main():
    run = Path(tempfile.mkdtemp(prefix="qwen-worker-synthetic-"))
    try:
        return checks(run)
    finally:
        shutil.rmtree(run)


def checks(run):
    for name in ("home", "cache", "config", "data", "hf", "tmp"):
        (run / name).mkdir()
    for name, sub in {"HOME": "home", "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
                      "XDG_DATA_HOME": "data", "HF_HOME": "hf", "HF_HUB_CACHE": "hf",
                      "HUGGINGFACE_HUB_CACHE": "hf", "TMPDIR": "tmp"}.items():
        os.environ[name] = str(run / sub)
    tempfile.tempdir = str(run / "tmp")
    os.environ.update(QWEN_CPU_TMP_ROOTS=str(run), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sys.dont_write_bytecode = True
    assert not any(n in sys.modules for n in ("torch", "diffusers", "transformers"))
    spec = importlib.util.spec_from_file_location("_worker_guard", ROOT / "tests/cpu_guard/sitecustomize.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.DENIED_IMPORTS = frozenset()
    attempts = []
    original_audit = guard._audit_hook
    def audit(event, args):
        try:
            original_audit(event, args)
        except guard.GuardViolation:
            attempts.append((event, traceback.extract_stack(limit=10)))
            raise
    guard._audit_hook = audit
    guard.install()
    import socket
    for attempt in (lambda: socket.socket(), lambda: open(ROOT / "must-not-write-worker-probe", "wb")):
        try:
            attempt()
        except guard.GuardViolation:
            pass
        else:
            raise AssertionError("test guard failed before imports")
    assert len(attempts) == 2
    attempts.clear()
    print("PASS socket/write test guards before ML imports")

    sys.path.insert(0, str(ROOT / "src"))
    from qwen_engine import encoder_worker as ew, encoder_assets as ea, encoder_identity as ei
    from qwen_engine.encoder import Blob, EncodedPayload, EncoderSetupError
    from qwen_engine.encoder_production import payload_to_branches
    from qwen_engine.config import GenerationConfig
    from qwen_engine.local_bundle import LOCAL_REFERENCE_LAYOUT, FULL_REFERENCE_REPO, FULL_REFERENCE_REVISION
    from qwen_engine.manifest import ModelArtifact, ModelManifest
    from qwen_engine.mem_preflight import MemorySnapshot, require_encoder_memory_fit
    import hashlib

    observe = ei.observe_installed_packages
    observed = observe()
    assert observed.packages["torchvision"].version is None
    ei.PINNED_TORCHVISION_VERSION = "synthetic-test-vision"
    ei.observe_installed_packages = lambda: replace(observed, packages={
        **observed.packages, "torchvision": ei.PackageObservation("synthetic-test-vision")})
    ei.validate_installed_packages(ei.observe_installed_packages())  # actual other pins/VCS checked
    def memory(size, **kwargs):
        return require_encoder_memory_fit(size, **kwargs, snapshot=MemorySnapshot(
            host_available_bytes=1024 ** 3, cgroup_state="unlimited"))
    ew.require_encoder_memory_fit = ea.require_encoder_memory_fit = memory
    model_root = run / "models"
    model_root.mkdir()
    shards = [Path(n).name for n in LOCAL_REFERENCE_LAYOUT["text_encoder"] if n.endswith(".safetensors")]
    index = dict(metadata={"total_size": 4}, weight_map={f"w{i}": s for i, s in enumerate(shards)})
    rows = []
    for role, names in LOCAL_REFERENCE_LAYOUT.items():
        for name in names:
            raw = json.dumps(index).encode() if name.endswith("model.safetensors.index.json") else name.encode()
            rows.append(ModelArtifact(role, FULL_REFERENCE_REPO, FULL_REFERENCE_REVISION, name,
                                      hashlib.sha256(raw).hexdigest(), len(raw)))
            if role in ea.ENCODER_ROLES:
                path = model_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
    manifest = ModelManifest(1, tuple(rows))
    manifest_path, request_path = run / "manifest.json", run / "request.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    torch_module, models = [], []

    actual_load_classes = ew._load_classes
    def loader_classes():
        assert ew._OFFLINE_INSTALLED and os.environ["HF_HUB_OFFLINE"] == "1"
        if not torch_module:
            torch, _, _ = actual_load_classes()  # actual imports and thread setup
        else:
            import torch
        torch.set_num_threads(1)
        torch_module[:] = [torch]
        class Inputs(SimpleNamespace):
            def to(self, device):
                assert str(device) == "cpu"
                return self
        class Processor:
            tokenizer = SimpleNamespace(encode=lambda _: [999])
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                assert path == str(model_root / "processor")
                assert kwargs["local_files_only"] and kwargs["trust_remote_code"] is False
                return cls()
            def apply_chat_template(self, *a, **k):
                return [[11, 12]]
            def __call__(self, *, text, padding, padding_side, return_tensors):
                assert padding and padding_side == "left" and return_tensors == "pt"
                length = 3 if "negative" in text[0] else 5
                ids = torch.tensor([[11, 12, *range(20, 20 + length)]], dtype=torch.int64)
                return Inputs(input_ids=ids, attention_mask=torch.ones_like(ids))
        class Norm(torch.nn.Module):
            def forward(self, value):
                return value * 2
        class Model(torch.nn.Module):
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                assert path == str(model_root / "text_encoder")
                assert kwargs["device_map"] == {"": "cpu"} and kwargs["weights_only"]
                assert kwargs["use_safetensors"] and not kwargs["trust_remote_code"]
                model = cls()
                model.anchor = torch.nn.Parameter(torch.zeros(1, dtype=kwargs["dtype"]))
                model.model = torch.nn.Module()
                model.model.language_model = torch.nn.Module()
                model.model.language_model.norm = Norm()
                text_config = SimpleNamespace(use_cache=True)
                model.config = SimpleNamespace(use_cache=True, get_text_config=lambda: text_config)
                models.append(model)
                return model
            def forward(self, input_ids, attention_mask, output_hidden_states, **kwargs):
                assert output_hidden_states and torch.is_inference_mode_enabled() and not self.training
                hidden = input_ids.unsqueeze(-1).repeat(1, 1, 4).to(self.anchor.dtype)
                return SimpleNamespace(hidden_states=(self.model.language_model.norm(hidden),))
        return torch, Processor, Model
    ew._load_classes = loader_classes

    for dtype in ("fp16", "bf16", "fp32"):
        for active in (False, True):
            config = GenerationConfig(prompt="abcde", negative_prompt="negative", true_cfg_scale=2 if active else 1)
            req = ei.project_encoder_request(config, manifest, encoder_dtype=dtype)
            request_path.write_text(json.dumps(req.to_request_dict("synthetic-worker")))
            response = run / f"response-{dtype}-{active}.json"
            ew.execute_worker(request_path, response, manifest_path, model_root, headroom_bytes=1024)
            doc = json.loads(response.read_bytes())
            assert doc["status"] == "ok" and doc["request_id"] == "synthetic-worker"
            payload = EncodedPayload(metadata=doc["metadata"], blobs=tuple(
                Blob(b["name"], base64.b64decode(b["data_b64"], validate=True)) for b in doc["blobs"]))
            branches = payload_to_branches(payload, expect_request=req)
            torch = torch_module[0]
            expected = torch.arange(20, 25, dtype=models[-1].anchor.dtype).reshape(1, 5, 1).repeat(1, 1, 4)
            assert torch.equal(branches.positive.embeds, expected)
            assert branches.positive.attention_mask is None and not branches.positive.image_pad_mask.any()
            assert (branches.negative is not None) == active
            assert not models[-1].model.language_model.norm._forward_hooks
            print(f"PASS worker parsing/identity/verification/injected-loader/real-upstream/codec/wire {dtype} cfg={active}")

    # A miniature generated checkpoint exercises the ACTUAL fixed-class
    # from_pretrained API, not just an injected loader returning tensors.
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from safetensors.torch import save_file
    from qwen_engine.encoder_production import encode_with_supplied_model
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, pad_token_id=0,
            rope_parameters={"rope_type": "default", "rope_theta": 500000., "mrope_section": [2, 1, 1]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
            patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
            out_hidden_size=16, num_position_embeddings=16, deepstack_visual_indexes=[]),
        image_token_id=30, video_token_id=31, vision_start_token_id=29, vision_end_token_id=28)
    torch.manual_seed(7)
    fresh = Qwen3VLForConditionalGeneration(config)
    assert sum(p.numel() for p in fresh.parameters()) < 20000
    state = fresh.state_dict()
    names = sorted(state)
    weights = {}
    total_size = 0
    for i, shard in enumerate(shards):
        tensors = {name: state[name].contiguous().clone() for name in names[i::4]}
        save_file(tensors, str(model_root / "text_encoder" / shard), metadata={"format": "pt"})
        weights.update({name: shard for name in tensors})
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())
    assert total_size < 100000
    (model_root / "text_encoder/config.json").write_text(config.to_json_string())
    (model_root / "text_encoder/generation_config.json").write_text("{}")
    (model_root / "text_encoder/model.safetensors.index.json").write_text(
        json.dumps(dict(metadata={"total_size": total_size}, weight_map=weights)))
    rows = [replace(a, sha256=hashlib.sha256((model_root / a.identity).read_bytes()).hexdigest(),
                    bytes=(model_root / a.identity).stat().st_size)
            if a.role in ea.ENCODER_ROLES else a for a in rows]
    manifest = ModelManifest(1, tuple(rows))
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    real_models = []
    processor_class = loader_classes()[1]
    class ActualMiniatureLoader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            model = Qwen3VLForConditionalGeneration.from_pretrained(*args, **kwargs)
            real_models.append(model)
            return model
    ew._load_classes = lambda: (torch, processor_class, ActualMiniatureLoader)
    for dtype in ("fp16", "bf16", "fp32"):
        req = ei.project_encoder_request(GenerationConfig(prompt="abcde", negative_prompt="negative", true_cfg_scale=2),
                                        manifest, encoder_dtype=dtype)
        request_path.write_text(json.dumps(req.to_request_dict("miniature-worker")))
        response = run / f"miniature-{dtype}.json"
        ew.execute_worker(request_path, response, manifest_path, model_root, headroom_bytes=1024)
        doc = json.loads(response.read_bytes())
        actual = payload_to_branches(EncodedPayload(metadata=doc["metadata"], blobs=tuple(
            Blob(b["name"], base64.b64decode(b["data_b64"], validate=True)) for b in doc["blobs"])), expect_request=req)
        baseline = payload_to_branches(encode_with_supplied_model(real_models[-1], processor_class(), req), expect_request=req)
        assert torch.equal(actual.positive.embeds.view(torch.uint8), baseline.positive.embeds.view(torch.uint8))
        assert torch.equal(actual.negative.embeds.view(torch.uint8), baseline.negative.embeds.view(torch.uint8))
        assert actual.positive.embeds.shape == (1, 5, 16) and actual.negative.embeds.shape == (1, 3, 16)
        assert not real_models[-1].training and not real_models[-1].config.get_text_config().use_cache
        assert all(p.device.type == "cpu" for p in real_models[-1].parameters())
        assert not real_models[-1].model.language_model.norm._forward_hooks
        print(f"PASS actual local-only fixed-class miniature checkpoint loader + encoder {dtype}")
    # The single urllib3 IPv6 capability probe is denied by the test guard,
    # before any fd is created, and urllib3 catches the OSError.
    assert len(attempts) == 1 and attempts[0][0] == "socket.__new__", attempts
    assert any(f.filename.endswith("/urllib3/util/connection.py") and f.name == "_has_ipv6"
               for f in attempts[0][1]), attempts
    assert not list(run.glob(".qwen-encoder-response-*.tmp"))
    print("PASS only expected denied urllib3 probe; no forbidden writes; response temps cleaned")
    ei.PINNED_TORCHVISION_VERSION = None
    ei.observe_installed_packages = observe
    try:
        ei.project_encoder_request(GenerationConfig(), manifest, encoder_dtype="fp16")
    except EncoderSetupError as exc:
        assert "no reviewed pin" in str(exc)
    else:
        raise AssertionError("production vision refusal was bypassed")
    print("PASS restored real production policy refuses unlocked/missing torchvision")
    print(f"12 synthetic worker checks passed on Python {sys.version.split()[0]}; miniature generated weights only, no downloaded checkpoint/processor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
