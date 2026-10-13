#!/usr/bin/env python3
"""Offline CPU vision/real-processor witness; never loads encoder weights.

Run with each pinned ML interpreter and -B. Only the nine processor assets
are read: bounded snapshots are hash-verified against the reviewed manifest
and copied into an owned disposable fixture before transformers loads them.
Writes stay in that fixture; sockets are denied before ML imports. A tiny
synthetic model tests real tokenization/upstream extraction/codec integration,
NOT real checkpoint tensors, memory fit, generation quality or CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

if sys.flags.optimize:
    raise SystemExit("Verification assertions must be enabled; unset PYTHONOPTIMIZE and do not use -O/-OO")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAX_PROCESSOR_FILE = 16 * 1024 * 1024
MAX_PROCESSOR_TOTAL = 32 * 1024 * 1024
MAX_PROCESSOR_FILES = 16


def copy_processor_files(artifacts, source: Path, destination: Path):
    """Copy only bounded verified flat files to a caller-owned empty fixture.

    Cooperative settled-filesystem check, not a hostile rewriting sandbox.
    The caller validates the fixed full-reference manifest before this helper.
    No transformer/model directory is inspected, even when it is incomplete.
    """
    from qwen_engine._local_io import read_bounded_file, _signature

    from qwen_engine.manifest import ModelArtifact

    if (not isinstance(artifacts, tuple) or not 1 <= len(artifacts) <= MAX_PROCESSOR_FILES
            or any(not isinstance(a, ModelArtifact) for a in artifacts)
            or any(a.role != "processor" or Path(a.identity).parent != Path("processor")
                   or not 1 <= a.bytes <= MAX_PROCESSOR_FILE for a in artifacts)
            or sum(a.bytes for a in artifacts) > MAX_PROCESSOR_TOTAL):
        raise ValueError("processor snapshot exceeds reviewed scope/bounds")
    names = {Path(a.identity).name for a in artifacts}
    if len(names) != len(artifacts):
        raise ValueError("duplicate processor file")
    source = source.resolve(strict=True)

    def inventory():
        result = {}
        with os.scandir(source) as entries:
            for entry in entries:
                if entry.name not in names or not entry.is_file(follow_symlinks=False):
                    raise ValueError("processor directory has unexpected/link/special entries")
                result[entry.name] = _signature(entry.stat(follow_symlinks=False))
        if set(result) != names:
            raise ValueError("processor directory is incomplete")
        return result

    before = inventory()
    directory = _signature(source.stat())
    if list(destination.iterdir()):
        raise ValueError("processor snapshot destination must be empty")
    for artifact in artifacts:
        name = Path(artifact.identity).name
        path = source / name
        if _signature(path.stat(follow_symlinks=False)) != before[name]:
            raise ValueError("processor file changed before read")
        data = read_bounded_file(path, artifact.bytes)
        if len(data) != artifact.bytes or hashlib.sha256(data).hexdigest() != artifact.sha256:
            raise ValueError("processor file size/hash mismatch")
        if _signature(path.stat(follow_symlinks=False)) != before[name]:
            raise ValueError("processor file changed during read")
        with (destination / name).open("xb") as stream:
            stream.write(data)
    if inventory() != before or _signature(source.stat()) != directory:
        raise ValueError("processor directory changed during snapshot")
    return sum(a.bytes for a in artifacts)


def install_check_guard(run: Path):
    """Use the existing cooperative test guard, allowing ML imports only."""
    sys.dont_write_bytecode = True
    os.environ.update(QWEN_CPU_TMP_ROOTS=str(run), PYTHONDONTWRITEBYTECODE="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for variable in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
                     "HF_HOME", "HF_HUB_CACHE", "TMPDIR"):
        path = run / variable.lower()
        path.mkdir()
        os.environ[variable] = str(path)
    tempfile.tempdir = str(run / "tmpdir")
    spec = importlib.util.spec_from_file_location("_processor_guard", ROOT / "tests/cpu_guard/sitecustomize.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.DENIED_IMPORTS = frozenset()
    guard.install()
    import socket
    try:
        socket.socket()
    except guard.GuardViolation:
        pass
    else:
        raise AssertionError("socket guard failed")
    try:
        (ROOT / "must-not-write-processor-probe").write_bytes(b"forbidden")
    except guard.GuardViolation:
        pass
    else:
        raise AssertionError("repository write guard failed")


def checks(run, models_dir):
    from qwen_engine import encoder_identity as ei, encoder_production as ep
    from qwen_engine.config import GenerationConfig
    from qwen_engine.local_bundle import validate_local_reference_layout
    from qwen_engine.manifest import load_manifest

    assert not set(ei.ML_MODULE_NAMES) & set(sys.modules)
    ei.validate_installed_packages(ei.observe_installed_packages())
    manifest = load_manifest(ROOT / "models/manifest.json")
    validate_local_reference_layout(manifest)
    processor_dir = run / "processor"
    processor_dir.mkdir()
    artifacts = tuple(a for a in manifest.artifacts if a.role == "processor")
    copied = copy_processor_files(artifacts, models_dir / "processor", processor_dir)
    print(f"PASS {len(artifacts)} real processor files size/hash verified and snapshotted ({copied} bytes)")

    import torch
    import torchvision
    from transformers import Qwen3VLProcessor
    from diffusers import QwenImage21Pipeline

    assert torch.__version__ == ei.PACKAGE_VERSIONS["torch"]
    assert torchvision.__version__ == ei.PACKAGE_VERSIONS["torchvision"]
    selected = torchvision.ops.nms(torch.tensor([[0., 0., 4., 4.], [0., 0., 3., 3.], [6., 6., 8., 8.]]),
                                   torch.tensor([.9, .8, .7]), .5)
    assert selected.tolist() == [0, 2] and selected.device.type == "cpu"
    resized = torchvision.transforms.v2.functional.resize(torch.zeros(3, 8, 8), [4, 4])
    assert resized.shape == (3, 4, 4) and resized.device.type == "cpu"
    print("PASS native CPU NMS and vision resize with exact reviewed Torch/vision builds")
    processor = Qwen3VLProcessor.from_pretrained(str(processor_dir), local_files_only=True, trust_remote_code=False)
    assert type(processor) is Qwen3VLProcessor
    assert processor.tokenizer.__class__.__name__ == "Qwen2Tokenizer"
    assert processor.image_processor is not None and processor.video_processor is not None
    print("PASS actual local-only fixed-class processor, tokenizer, image and video processor loaded offline")

    class Norm(torch.nn.Module):
        def forward(self, value):
            return value * 2

    class TinyModel(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.norm = Norm()
            self.inputs = []

        def forward(self, input_ids, attention_mask, output_hidden_states, **kwargs):
            assert output_hidden_states and torch.is_inference_mode_enabled()
            assert input_ids.device.type == attention_mask.device.type == "cpu"
            assert set(kwargs) <= {"mm_token_type_ids"}
            self.inputs.append((input_ids.clone(), attention_mask.clone()))
            hidden = (input_ids % 1000).unsqueeze(-1).repeat(1, 1, 4).to(self.anchor.dtype)
            return SimpleNamespace(hidden_states=(self.model.language_model.norm(hidden),))

    config = GenerationConfig(prompt=" A painted sign reading CAFÉ — 蓝色天空 ",
                              negative_prompt="blurred text, misspelled letters", true_cfg_scale=2)
    for dtype, torch_dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)):
        request = ei.project_encoder_request(config, manifest, encoder_dtype=dtype)
        model = TinyModel(torch_dtype).eval()
        reference = QwenImage21Pipeline(scheduler=None, vae=None, transformer=None,
                                       text_encoder=model, processor=processor)
        system = [{"role": "system", "content": [{"type": "text", "text": reference.sys_prompt}]}]
        assert reference._drop_idx == len(processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])
        payload = ep.encode_with_supplied_model(model, processor, request)
        branches = ep.payload_to_branches(payload, expect_request=request)
        assert len(model.inputs) == 2
        for prompt, branch, (ids, mask) in zip((config.prompt, config.negative_prompt),
                                               (branches.positive, branches.negative), model.inputs):
            direct = processor(text=[reference.prompt_template_t2i.format(prompt)], padding=True,
                               padding_side="left", return_tensors="pt")
            assert torch.equal(ids, direct.input_ids) and torch.equal(mask, direct.attention_mask)
            retained = ids[mask.bool()][reference._drop_idx:]
            expected = (retained % 1000).to(torch_dtype).reshape(1, -1, 1).repeat(1, 1, 4)
            assert torch.equal(branch.embeds, expected) and branch.attention_mask is None
            assert branch.image_pad_mask.dtype == torch.bool and not branch.image_pad_mask.any()
        assert not model.model.language_model.norm._forward_hooks
        print(f"PASS real processor + upstream raw template/drop/masks/pre-norm + codec {dtype}; synthetic model ONLY")
    print("Processor/CPU vision checks passed; real checkpoint fidelity, memory fit, CUDA and generation NOT proved")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/assets")
    args = parser.parse_args()
    models_dir = args.models_dir.absolute()
    run = Path(tempfile.mkdtemp(prefix="qwen-real-processor-"))
    try:
        install_check_guard(run)
        return checks(run, models_dir)
    finally:
        shutil.rmtree(run)


if __name__ == "__main__":
    raise SystemExit(main())
