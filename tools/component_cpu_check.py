#!/usr/bin/env python3
"""Pinned DiT/VAE shape and tiny sampling witness, never loads real weights.

Actual verified configs + bounded checkpoint headers are inspected; the full
DiT/VAE are instantiated on META only, then compared to header shapes. Tiny
random CPU models exercise real scheduler/DiT/KV/CFG/VAE code. These are not
reference images, full-model numerics, GPU fitness or an enabled adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import stat
import struct
import sys
import tempfile
from pathlib import Path

if sys.flags.optimize:
    raise SystemExit("Verification assertions must be enabled; unset PYTHONOPTIMIZE and do not use -O/-OO")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAX_HEADER_BYTES = 4 * 1024 * 1024
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "BOOL": 1}


def read_checkpoint_header(path, expected_bytes):
    """Read metadata only, no mmap/tensor-body reads, no special/link files."""
    from qwen_engine._jsonsafe import strict_json_loads
    from qwen_engine._local_io import _signature

    if type(expected_bytes) is not int or expected_bytes < 8:
        raise ValueError("checkpoint size must be an explicit positive bound")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise ValueError("checkpoint must be a regular file of exact expected size")

        def exact(size):
            data = bytearray()
            while len(data) < size:
                block = os.read(fd, min(65536, size - len(data)))
                if not block:
                    raise ValueError("truncated checkpoint header")
                data.extend(block)
            return bytes(data)

        size = struct.unpack("<Q", exact(8))[0]
        if not 2 <= size <= min(MAX_HEADER_BYTES, expected_bytes - 8):
            raise ValueError("checkpoint header byte cap/extent invalid")
        try:
            header = strict_json_loads(exact(size))
        except RecursionError as exc:
            raise ValueError("checkpoint header nesting invalid") from exc
        if not isinstance(header, dict) or not 1 <= len(header) <= 10000:
            raise ValueError("checkpoint header object/count invalid")
        result, spans = {}, []
        for name, item in header.items():
            if name == "__metadata__":
                if not isinstance(item, dict) or any(type(v) is not str for v in item.values()):
                    raise ValueError("checkpoint metadata must contain strings")
                continue
            if (type(name) is not str or not 1 <= len(name) <= 512 or not isinstance(item, dict)
                    or set(item) != {"dtype", "shape", "data_offsets"}
                    or type(item["dtype"]) is not str or item["dtype"] not in _DTYPE_BYTES
                    or not isinstance(item["shape"], list) or len(item["shape"]) > 8
                    or any(type(v) is not int or v <= 0 for v in item["shape"])
                    or not isinstance(item["data_offsets"], list) or len(item["data_offsets"]) != 2
                    or any(type(v) is not int or v < 0 for v in item["data_offsets"])):
                raise ValueError("checkpoint tensor metadata invalid")
            start, end = item["data_offsets"]
            if end - start != math.prod(item["shape"]) * _DTYPE_BYTES[item["dtype"]] or end > expected_bytes - 8 - size:
                raise ValueError("checkpoint tensor extent/shape invalid")
            spans.append((start, end))
            result[name] = (tuple(item["shape"]), item["dtype"])
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError("checkpoint spans overlap or leave gaps")
            cursor = end
        if not result or cursor != expected_bytes - 8 - size:
            raise ValueError("checkpoint tensor body extent invalid")
        if (_signature(before) != _signature(os.fstat(fd))
                or _signature(before) != _signature(Path(path).stat(follow_symlinks=False))):
            raise ValueError("checkpoint changed during header read")
        return result
    finally:
        os.close(fd)


def checks(run, models_dir):
    from local_processor_check import copy_processor_files
    from qwen_engine import encoder_identity as ei
    from qwen_engine._local_io import read_bounded_file
    from qwen_engine._jsonsafe import strict_json_loads
    from qwen_engine.local_bundle import preflight_closed_tree, validate_local_reference_layout
    from qwen_engine.manifest import load_manifest

    assert not set(ei.ML_MODULE_NAMES) & set(sys.modules)
    ei.validate_installed_packages(ei.observe_installed_packages())
    manifest = load_manifest(ROOT / "models/manifest.json")
    validate_local_reference_layout(manifest)
    preflight_closed_tree(models_dir)
    artifacts = {a.identity: a for a in manifest.artifacts}

    def config(name):
        art = artifacts[name]
        raw = read_bounded_file(models_dir / name, art.bytes)
        assert len(raw) == art.bytes and hashlib.sha256(raw).hexdigest() == art.sha256
        return strict_json_loads(raw)

    dit_config, vae_config = config("transformer/config.json"), config("vae/config.json")
    scheduler_config = config("scheduler/scheduler_config.json")
    header_sets = {}
    for role in ("dit", "vae"):
        tensors = {}
        for art in manifest.artifacts:
            if art.role == role and art.identity.endswith(".safetensors"):
                header = read_checkpoint_header(models_dir / art.identity, art.bytes)
                assert not set(header) & set(tensors)
                tensors.update(header)
        header_sets[role] = tensors
    processor_dir = run / "processor"
    processor_dir.mkdir()
    copy_processor_files(tuple(a for a in manifest.artifacts if a.role == "processor"),
                         models_dir / "processor", processor_dir)

    import torch
    from transformers import Qwen3VLProcessor
    from diffusers import (QwenImage21Pipeline, QwenImage21Transformer2DModel,
                           AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler)

    for role, cls, settings in (("dit", QwenImage21Transformer2DModel, dit_config),
                                ("vae", AutoencoderKLQwenImage21, vae_config)):
        with torch.device("meta"):
            model = cls.from_config(settings)
        state = model.state_dict()
        assert all(value.is_meta for value in state.values())
        assert {name: tuple(value.shape) for name, value in state.items()} == {
            name: shape for name, (shape, _) in header_sets[role].items()}
        print(f"PASS actual {role} configs/META state shapes match {len(state)} checkpoint header tensors; no tensor-body load")
        del state, model

    processor = Qwen3VLProcessor.from_pretrained(str(processor_dir), local_files_only=True, trust_remote_code=False)
    torch.manual_seed(7)
    tiny_settings = dict(dit_config, num_layers=1, num_attention_heads=2, attention_head_dim=8,
                         context_in_dim=16, axes_dims_rope=[2, 2, 4])
    tiny = QwenImage21Transformer2DModel.from_config(tiny_settings).eval()
    # No real encoder: hand-built finite tiny embeddings, batch1/H16.
    positive, negative = torch.randn(1, 5, 16), torch.randn(1, 3, 16)
    for dtype in (torch.float32, torch.float16):
        tiny.to(dtype=dtype)
        for active in (False, True):
            for cache in (False, True):
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
                pipe = QwenImage21Pipeline(scheduler=scheduler, vae=None, transformer=tiny,
                                          text_encoder=None, processor=processor)
                pipe.set_progress_bar_config(disable=True)
                steps = []
                def callback(pipeline, index, timestep, tensors):
                    assert torch.isfinite(tensors["latents"]).all()
                    steps.append(index)
                    return {}
                with torch.inference_mode():
                    result = pipe(prompt=None, height=64, width=64, num_inference_steps=2,
                        true_cfg_scale=2 if active else 1, prompt_embeds=positive.to(dtype),
                        negative_prompt_embeds=negative.to(dtype) if active else None,
                        generator=torch.Generator(device="cpu").manual_seed(11), output_type="latent",
                        use_kv_cache=cache, callback_on_step_end=callback)
                assert result.images.shape == (1, 16, 64) and result.images.dtype == dtype
                assert result.images.device.type == "cpu" and torch.isfinite(result.images).all()
                assert steps == [0, 1]
                print(f"PASS tiny RANDOM CPU upstream sampling {dtype} CFG={active} KV={cache}, callbacks/latent shape/finite")
    tiny_vae = AutoencoderKLQwenImage21.from_config(dict(vae_config, base_dim=4,
        decoder_base_dim=4, dim_mult=[1, 1, 1, 1, 1], num_res_blocks=1)).eval()
    tiny.to(dtype=torch.float32)
    pipe = QwenImage21Pipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config),
                              vae=tiny_vae, transformer=tiny, text_encoder=None, processor=processor)
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        decoded = pipe(prompt=None, height=64, width=64, num_inference_steps=2,
            prompt_embeds=positive, generator=torch.Generator(device="cpu").manual_seed(11),
            output_type="pt").images
    assert decoded.shape == (1, 4, 64, 64) and torch.isfinite(decoded).all()
    assert decoded.device.type == "cpu" and decoded.dtype == torch.float32
    print("PASS tiny RANDOM CPU VAE decode/latent normalization/RGBA tensor output, NOT an accepted reference image")
    print("Component checks passed; no real weights, reference images, model fidelity, GPU fit or generation adapter acceptance")
    return 0


def main():
    from local_processor_check import install_check_guard
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/assets")
    args = parser.parse_args()
    models_dir = args.models_dir.resolve(strict=True)
    run = Path(tempfile.mkdtemp(prefix="qwen-components-synthetic-"))
    try:
        install_check_guard(run)
        return checks(run, models_dir)
    finally:
        shutil.rmtree(run)


if __name__ == "__main__":
    raise SystemExit(main())
