#!/usr/bin/env python3
"""Offline tiny RANDOM component bridge witness; no real model weight reads.

Full-identity payloads contain synthetic embeddings. Bundle construction below
is deliberately metadata-only and is NOT fresh asset verification/provenance.
The trusted supplied API remains disconnected from the default loader/web.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

if sys.flags.optimize:
    raise SystemExit("Verification assertions must be enabled")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def checks(run, models_dir):
    from local_processor_check import copy_processor_files
    from qwen_engine import encoder_identity as ei, encoder_production as ep
    from qwen_engine._local_io import read_bounded_file
    from qwen_engine._jsonsafe import strict_json_loads
    from qwen_engine.adapters.supplied_qwen21 import SuppliedQwen21Adapter
    from qwen_engine.config import GenerationConfig
    from qwen_engine.core import generate
    from qwen_engine.encoder_client import ProductionEncoding
    from qwen_engine.encoder import EncoderProtocolError, EncoderRequestError
    from qwen_engine.errors import EngineError, GenerationCancelled
    from qwen_engine.local_bundle import (
        LOCAL_REFERENCE_PROFILE, LOCAL_REFERENCE_LAYOUT, ROLE_DIRECTORIES,
        LocalReferenceBundle, reference_bundle_fingerprints, validate_local_reference_layout,
    )
    from qwen_engine.manifest import load_manifest

    assert not set(ei.ML_MODULE_NAMES) & set(sys.modules)
    ei.validate_installed_packages(ei.observe_installed_packages())
    manifest = load_manifest(ROOT / "models/manifest.json")
    validate_local_reference_layout(manifest)
    bundle = LocalReferenceBundle(profile=LOCAL_REFERENCE_PROFILE,
        root=models_dir, root_resolved=models_dir,
        repo=manifest.artifacts[0].repo, revision=manifest.artifacts[0].revision,
        assets={role: tuple(models_dir / name for name in names) for role, names in LOCAL_REFERENCE_LAYOUT.items()},
        component_dirs={role: models_dir / name for role, name in ROLE_DIRECTORIES.items()},
        fingerprints=reference_bundle_fingerprints(manifest))
    artifacts = {a.identity: a for a in manifest.artifacts}

    def config(name):
        artifact = artifacts[name]
        raw = read_bounded_file(models_dir / name, artifact.bytes)
        assert len(raw) == artifact.bytes and hashlib.sha256(raw).hexdigest() == artifact.sha256
        return strict_json_loads(raw)

    dit_config, vae_config = config("transformer/config.json"), config("vae/config.json")
    scheduler_config = config("scheduler/scheduler_config.json")
    processor_dir = run / "processor"
    processor_dir.mkdir()
    copy_processor_files(tuple(a for a in manifest.artifacts if a.role == "processor"),
                         models_dir / "processor", processor_dir)
    import torch
    from PIL import Image
    from transformers import Qwen3VLProcessor
    from diffusers import (QwenImage21Pipeline, QwenImage21Transformer2DModel,
                           AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler)

    processor = Qwen3VLProcessor.from_pretrained(str(processor_dir), local_files_only=True, trust_remote_code=False)

    class SyntheticProvider:
        """Only this witness constructs synthetic content; product cannot select it."""
        def __init__(self, *, stale=False, overflow=False):
            self.stale, self.overflow, self.calls = stale, overflow, []

        def encode(self, generation, *, encoder_dtype, cancel_event):
            self.calls.append((generation, encoder_dtype, cancel_event))
            if self.stale:
                generation = replace(generation, prompt="different fixture prompt")
            request = ei.project_encoder_request(generation, manifest, encoder_dtype=encoder_dtype)
            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[encoder_dtype]
            def branch(seq, padded):
                embeds = torch.randn(1, seq, 16, dtype=dtype)
                if self.overflow:
                    embeds.fill_(100000)
                mask = None if not padded else torch.tensor([[1] * (seq - 1) + [0]], dtype=torch.int64)
                return ep.EncoderBranch(embeds, mask, torch.zeros(1, seq, dtype=torch.bool))
            payload = ep.serialize_branches(ep.validate_production_request(request),
                ep.BranchBundle(branch(5, True), branch(3, False) if generation.cfg_active else None))
            return ProductionEncoding(request, payload)

    def adapter(*, dtype="fp32", encoder_dtype="fp32", cache=True, provider=None):
        torch.manual_seed(7)
        dit = QwenImage21Transformer2DModel.from_config(dict(dit_config, num_layers=1,
            num_attention_heads=2, attention_head_dim=8, context_in_dim=16, axes_dims_rope=[2, 2, 4])).eval()
        vae = AutoencoderKLQwenImage21.from_config(dict(vae_config, base_dim=4,
            decoder_base_dim=4, dim_mult=[1, 1, 1, 1, 1], num_res_blocks=1)).eval()
        execution = {"fp16": torch.float16, "fp32": torch.float32}[dtype]
        dit.to(dtype=execution)
        vae.to(dtype=execution)
        pipe = QwenImage21Pipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config),
            vae=vae, transformer=dit, text_encoder=None, processor=processor)
        pipe.set_progress_bar_config(disable=True)
        return SuppliedQwen21Adapter(pipeline=pipe, encoder=provider or SyntheticProvider(),
            manifest=manifest, bundle=bundle, encoder_dtype=encoder_dtype,
            execution_dtype=dtype, use_kv_cache=cache)

    output = run / "images"
    output.mkdir()
    base = GenerationConfig(prompt="synthetic component fixture", width=64, height=64,
                            steps=2, seed=11, output_dir=output)
    for active in (False, True):
        for cache in (False, True):
            for enc_dtype in ("fp32", "fp16", "bf16"):
                generation = replace(base, negative_prompt="synthetic negative" if active else "",
                                     true_cfg_scale=2 if active else 1)
                supplied = adapter(encoder_dtype=enc_dtype, cache=cache)
                phases = []
                result = generate(generation, supplied, bundle.to_assets(), phase_callback=phases.append)
                with Image.open(result.output_path) as image:
                    assert image.format == "PNG" and image.mode == "RGBA" and image.size == (64, 64)
                    image.load()
                assert phases == ["load", "encode", "denoise", "decode", "save"]
                assert supplied.actual_seed == 11 and supplied._state == "closed" and supplied._pipeline is None
                print(f"PASS tiny RANDOM bridge PNG, encoder={enc_dtype} execution=fp32 CFG={active} KV={cache}")

    supplied = adapter()
    result = generate(replace(base, seed=None), supplied, bundle.to_assets())
    assert type(supplied.actual_seed) is int and 0 <= supplied.actual_seed < 2**64
    assert Path(result.output_path).is_file()
    print("PASS random seed records actual owned generator seed")

    # Same supplied components + exact conditioning/generator: compare the
    # separated phases to the upstream's unsplit PIL path, not a second fake.
    supplied = adapter()
    supplied.load(base, bundle.to_assets())
    branches = supplied.encode(base)
    with torch.inference_mode():
        expected = supplied._pipeline(prompt=None, height=64, width=64, num_inference_steps=2,
            prompt_embeds=branches.positive.embeds,
            prompt_embeds_mask=branches.positive.attention_mask,
            generator=torch.Generator(device="cpu").manual_seed(11),
            output_type="pil", use_kv_cache=True).images[0]
    actual = supplied.decode(supplied.denoise(base, branches))
    assert actual.mode == expected.mode and actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()
    supplied.close()
    print("PASS tiny split sampler/normalization/VAE pixels exactly match unsplit upstream PIL path")

    for provider, error, dtype in ((SyntheticProvider(stale=True), EncoderRequestError, "fp32"),
                                    (SyntheticProvider(overflow=True), EngineError, "fp16")):
        before = set(output.iterdir())
        supplied = adapter(dtype=dtype, provider=provider)
        try:
            generate(base, supplied, bundle.to_assets())
        except error:
            pass
        else:
            raise AssertionError("stale identity or conversion overflow accepted")
        assert supplied._state == "closed" and set(output.iterdir()) == before
    print("PASS stale child identity / fp32-to-fp16 overflow refuse before saving")

    supplied = adapter()
    supplied.load(base, bundle.to_assets())
    branches = supplied.encode(base)
    branches.positive.image_pad_mask[0, 0] = True
    try:
        supplied.denoise(base, branches)
    except EncoderProtocolError:
        pass
    else:
        raise AssertionError("mutated image conditioning accepted")
    finally:
        supplied.close()
    print("PASS mutable conditioning revalidated immediately before sampling")

    from unittest.mock import patch
    original = QwenImage21Pipeline.__call__
    event = threading.Event()
    seen = []
    def cancel_in_sampler(pipe, *args, **kwargs):
        callback = kwargs["callback_on_step_end"]
        def cancelling(*values):
            seen.append(values[1])
            event.set()
            return callback(*values)
        kwargs["callback_on_step_end"] = cancelling
        return original(pipe, *args, **kwargs)
    before = set(output.iterdir())
    supplied = adapter()
    with patch.object(QwenImage21Pipeline, "__call__", cancel_in_sampler):
        try:
            generate(base, supplied, bundle.to_assets(), cancel_event=event)
        except GenerationCancelled:
            pass
        else:
            raise AssertionError("sampler cancellation accepted partial success")
    assert seen == [0] and supplied._state == "closed" and set(output.iterdir()) == before
    print("PASS actual upstream step callback aborts cancellation; no partial PNG")

    supplied = adapter()
    supplied.load(base, bundle.to_assets())
    branches = supplied.encode(base)
    latents = supplied.denoise(base, branches)
    vae = supplied._pipeline.vae
    before = set(output.iterdir())
    with patch.object(vae, "decode", return_value=(torch.full((1, 4, 1, 64, 64), float("nan")),)):
        try:
            supplied.decode(latents)
        except EngineError:
            pass
        else:
            raise AssertionError("nonfinite VAE output accepted")
    supplied.close()
    assert set(output.iterdir()) == before
    print("PASS nonfinite VAE output refuses image conversion/publication")

    supplied = adapter()
    supplied.load(base, bundle.to_assets())
    image = supplied.decode(supplied.denoise(base, supplied.encode(base)))
    target = output / ("0" * 32 + ".png")
    target.write_bytes(b"owned collision fixture")
    from types import SimpleNamespace
    from qwen_engine.encoder import EncoderSetupError
    with patch("qwen_engine.adapters.supplied_qwen21.uuid.uuid4", return_value=SimpleNamespace(hex="0" * 32)):
        try:
            supplied.save(image, base)
        except EncoderSetupError:
            pass
        else:
            raise AssertionError("existing image overwritten")
    supplied.close()
    assert target.read_bytes() == b"owned collision fixture" and not list(output.glob("*.tmp"))
    print("PASS PNG filename collision preserves existing bytes; no temporary leftovers")
    print("Supplied bridge witness passed; NOT actual weights, GPU fit, quality, default loader or web activation")
    return 0


def main():
    from local_processor_check import install_check_guard
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/assets")
    args = parser.parse_args()
    models_dir = args.models_dir.resolve(strict=True)
    run = Path(tempfile.mkdtemp(prefix="qwen-supplied-synthetic-"))
    try:
        install_check_guard(run)
        return checks(run, models_dir)
    finally:
        shutil.rmtree(run)


if __name__ == "__main__":
    raise SystemExit(main())
