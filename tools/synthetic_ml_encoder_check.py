#!/usr/bin/env python3
"""Real pinned upstream prompt code and codec with tiny synthetic inputs.

Run using .review-envs/py311/bin/python -B or .review-envs/py313/bin/python -B.
Reusable checks live in git; only fixtures/caches are temporary. Socket and
write guards are proven before ML imports. No checkpoint/tokenizer is loaded.
Missing dependencies exit 2; wrong pins and failed checks exit 1.
"""
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ("torch", "safetensors", "diffusers", "transformers"):
        try:
            md.version(name)
        except md.PackageNotFoundError:
            print(f"SKIP: {name} unavailable in {sys.executable}")
            return 2
    sys.dont_write_bytecode = True
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    run = Path(tempfile.mkdtemp(prefix="qwen-encoder-synthetic-"))
    try:
        return checks(run)
    finally:
        shutil.rmtree(run)


def checks(run):
    for name in ("home", "cache", "config", "data", "hf", "tmp"):
        (run / name).mkdir()
    for env, sub in {"HOME": "home", "XDG_CACHE_HOME": "cache",
                     "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
                     "HF_HOME": "hf", "HF_HUB_CACHE": "hf",
                     "HUGGINGFACE_HUB_CACHE": "hf", "TMPDIR": "tmp"}.items():
        os.environ[env] = str(run / sub)
    tempfile.tempdir = str(run / "tmp")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PYTHONDONTWRITEBYTECODE="1", QWEN_CPU_TMP_ROOTS=str(run))
    assert not any(m in sys.modules for m in ("torch", "diffusers", "transformers"))
    spec = importlib.util.spec_from_file_location(
        "_encoder_review_guard", ROOT / "tests/cpu_guard/sitecustomize.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.DENIED_IMPORTS = frozenset()
    attempts = []
    original_audit = guard._audit_hook

    def audit(event, args):
        try:
            original_audit(event, args)
        except guard.GuardViolation:
            attempts.append((event, [(f.filename, f.name) for f in traceback.extract_stack(limit=10)],
                             args[1] if event == "socket.__new__" else None))
            raise

    guard._audit_hook = audit
    guard.install()
    import socket
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except guard.GuardViolation:
        pass
    else:
        raise AssertionError("network guard failed before ML imports")
    try:
        with open(ROOT / "must-not-write-review-probe", "wb"):
            pass
    except guard.GuardViolation:
        pass
    else:
        raise AssertionError("write guard failed before ML imports")
    assert len(attempts) == 2
    attempts.clear()
    sys.path.insert(0, str(ROOT / "src"))
    from qwen_engine.encoder import (
        Blob, EncodedPayload, ComponentIdentity, EncoderImplementationIdentity,
        EncoderRequest, EncoderProtocolError, EncoderRequestError,
        PINNED_DIFFUSERS_COMMIT, PINNED_TRANSFORMERS_COMMIT)
    assert md.version("torch") == "2.14.1+cu126"
    assert md.version("safetensors") == "0.8.0"
    for name, commit in (("diffusers", PINNED_DIFFUSERS_COMMIT),
                         ("transformers", PINNED_TRANSFORMERS_COMMIT)):
        direct = json.loads(md.distribution(name).read_text("direct_url.json"))
        assert direct["vcs_info"]["commit_id"] == commit
    import torch
    from diffusers import QwenImage21Pipeline
    import qwen_engine.encoder_production as ep
    torch.set_num_threads(1)
    # urllib3 imports a local IPv6 capability check. Its socket construction
    # is refused before any OS fd exists and urllib3 catches the denial.
    import_probe_denied = bool(attempts)
    if attempts:
        assert len(attempts) == 1, attempts
        event, frames, family = attempts[0]
        assert event == "socket.__new__" and family == socket.AF_INET6, attempts
        assert any(path.endswith("/urllib3/util/connection.py") and name == "_has_ipv6"
                   for path, name in frames), attempts
        attempts.clear()
    count = 0

    def passed(label):
        nonlocal count
        count += 1
        print(f"PASS {label}", flush=True)

    passed("socket/write denials proven before pinned ML imports")
    if import_probe_denied:
        passed("urllib3 IPv6 capability probe denied before OS socket creation")

    class Inputs(SimpleNamespace):
        def to(self, device):
            assert str(device) == "cpu"
            return self

    class Processor:
        def __init__(self, multimodal=True):
            self.multimodal = multimodal
            self.tokenizer = SimpleNamespace(encode=lambda _: [999])
            self.calls = []

        def apply_chat_template(self, *args, **kwargs):
            return [[11, 12]]  # synthetic prefix, no real tokenizer

        def __call__(self, *, text, padding, padding_side, return_tensors):
            assert padding is True and padding_side == "left" and return_tensors == "pt"
            self.calls.append(text)
            lengths = [4 if "bbbb" in t else 2 if "aa" in t else
                       3 if "negative" in t else 5 if "abcde" in t else 1 for t in text]
            maximum = max(lengths) + 2
            ids, masks = [], []
            for length in lengths:
                row = [11, 12] + list(range(20, 20 + length))
                pad = maximum - len(row)
                ids.append([0] * pad + row)
                masks.append([0] * pad + [1] * len(row))
            out = Inputs(input_ids=torch.tensor(ids, dtype=torch.int64),
                         attention_mask=torch.tensor(masks, dtype=torch.int64))
            if self.multimodal:
                out.mm_token_type_ids = torch.zeros_like(out.input_ids)
            return out

    class Norm(torch.nn.Module):
        def forward(self, value):
            return value * 2

    class Model(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.norm = Norm()
            self.calls, self.fail = [], False

        def forward(self, input_ids, attention_mask, output_hidden_states, **kwargs):
            assert output_hidden_states and torch.is_inference_mode_enabled()
            self.calls.append(kwargs)
            hidden = input_ids.unsqueeze(-1).repeat(1, 1, 4).to(self.anchor.dtype)
            hidden = self.model.language_model.norm(hidden)
            if self.fail:
                raise ValueError("synthetic model failure")
            return SimpleNamespace(hidden_states=(hidden,))

    def request(dtype="fp16", active=True, **over):
        sha, rev = "b" * 64, "a" * 40
        args = dict(prompt="abcde", negative_prompt="negative", dtype=dtype,
                    kwargs={"cfg_active": active}, model_repo="Qwen/Qwen-Image-2.1",
                    model_revision=rev, processor=ComponentIdentity(sha256=sha),
                    tokenizer=ComponentIdentity(revision=rev),
                    chat_template=ComponentIdentity(sha256=sha),
                    implementation_identity=EncoderImplementationIdentity(
                        encoder_artifact_set_sha256=sha,
                        worker_protocol_revision="worker-protocol.v1",
                        worker_implementation_revision="synthetic.v1",
                        prompt_extraction_revision="upstream-pinned.v1",
                        diffusers_commit=PINNED_DIFFUSERS_COMMIT,
                        transformers_commit=PINNED_TRANSFORMERS_COMMIT,
                        torch_build="cu126", torch_version=md.version("torch"),
                        safetensors_version=md.version("safetensors"),
                        torchvision_version="absent"))
        args.update(over)
        return EncoderRequest(**args)

    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    for dtype, torch_dtype in dtypes.items():
        for active in (False, True):
            model, processor = Model(torch_dtype), Processor()
            req = request(dtype, active)
            payload = ep.encode_with_supplied_model(model, processor, req)
            bundle = ep.payload_to_branches(payload, expect_request=req)
            expected = torch.arange(20, 25, dtype=torch_dtype).reshape(1, 5, 1).repeat(1, 1, 4)
            assert torch.equal(bundle.positive.embeds, expected)  # pre-norm, not doubled
            assert bundle.positive.attention_mask is None
            assert not bundle.positive.image_pad_mask.any()
            assert (bundle.negative is not None) == active
            assert len(model.calls) == (2 if active else 1)
            assert all("mm_token_type_ids" in c for c in model.calls)
            assert not model.model.language_model.norm._forward_hooks
            if active:
                assert bundle.negative.embeds.shape == (1, 3, 4)
            passed(f"real upstream encode/codec {dtype} cfg={active}, norm hook removed")
        req = request(dtype, False)
        vals = torch.tensor([0.0, -0.0, 0.125, -2.0], dtype=torch_dtype).reshape(1, 1, 4)
        for mask in (None, torch.ones(1, 1, dtype=torch.int64)):
            bundle = ep.BranchBundle(ep.EncoderBranch(vals, mask, torch.zeros(1, 1, dtype=torch.bool)))
            payload = ep.serialize_branches(ep.validate_production_request(req), bundle)
            recovered = ep.payload_to_branches(payload, expect_request=req)
            assert torch.equal(recovered.positive.embeds.view(torch.uint8), vals.view(torch.uint8))
            assert (recovered.positive.attention_mask is None) == (mask is None)
            passed(f"bitexact codec {dtype}, nullable mask={mask is None}")

    model, processor = Model(torch.float16), Processor(multimodal=False)
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, transformer=None,
                              text_encoder=model, processor=processor)
    with torch.inference_mode():
        embeds, mask, image_mask = pipe.encode_prompt(
            ["aa", "bbbb"], device=torch.device("cpu"), num_images_per_prompt=1)
        assert embeds.shape == (2, 4, 4)
        assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
        assert not image_mask.any() and "mm_token_type_ids" not in model.calls[-1]
        pipe.encode_prompt("", device=torch.device("cpu"), num_images_per_prompt=1)
        assert processor.calls[-1] == [pipe.prompt_template_t2i.format(" ")]
    passed("upstream unequal internal batch, blank normalization, optional multimodal IDs")
    model.fail = True
    try:
        with torch.inference_mode():
            pipe.encode_prompt("aa", device=torch.device("cpu"), num_images_per_prompt=1)
    except ValueError:
        pass
    else:
        raise AssertionError("synthetic upstream failure did not propagate")
    assert not model.model.language_model.norm._forward_hooks
    passed("upstream norm hook removed on model failure")

    req = request(active=False)
    payload = ep.encode_with_supplied_model(Model(torch.float16), Processor(), req)
    data = payload.blobs[0].data

    def refused(label, raw, match=None, template=None):
        template = payload if template is None else template
        try:
            ep.payload_to_branches(
                EncodedPayload(metadata=template.metadata, blobs=(Blob("tensors", raw),)),
                expect_request=req)
        except EncoderProtocolError as exc:
            if match:
                assert match in str(exc), str(exc)
            passed(label)
        else:
            raise AssertionError(f"{label} accepted")

    refused("truncated bytes", data[:-1])
    refused("header cap", (2 ** 40).to_bytes(8, "little") + data[8:])
    length = int.from_bytes(data[:8], "little")
    base = 8 + length
    header = json.loads(data[8:base])
    start, end = header["positive.embeds"]["data_offsets"]
    nan_bytes = torch.full(((end - start) // 2,), float("nan"), dtype=torch.float16).view(torch.uint8).numpy().tobytes()
    refused("nonfinite tensor data", data[:base + start] + nan_bytes + data[base + end:], "non-finite")

    def framed(text, body=data[base:]):
        raw = text.encode("utf-8")
        return len(raw).to_bytes(8, "little") + raw + body

    for label, field, value in (
        ("bool shape refused before load", "shape", [True, 5, 4]),
        ("oversized shape refused before load", "shape", [1, 2 ** 100, 4]),
        ("shape rank refused before load", "shape", [1, 1, 5, 4]),
        ("unsupported dtype refused before load", "dtype", "F64"),
        ("out-of-range offsets refused before load", "data_offsets", [0, len(data)]),
        ("shape/span disagreement refused before load", "shape", [1, 4, 4]),
    ):
        altered = json.loads(data[8:base])
        altered["positive.embeds"][field] = value
        refused(label, framed(json.dumps(altered)))
    altered = json.loads(data[8:base])
    altered["positive.image_pad_mask"]["data_offsets"] = [start, start + 5]
    refused("overlapping spans refused before load", framed(json.dumps(altered)))
    refused("trailing data refused before load", data + b"x")
    refused("duplicate JSON keys refused before load", framed('{"x":{},"x":{}}'))
    refused("deep header refused before load", framed('[' * 100 + '0' + ']' * 100))
    altered = json.loads(data[8:base])
    altered["__metadata__"]["cache_key"] = "f" * 64
    refused("safetensors identity disagreement refused before load", framed(json.dumps(altered)))
    start, end = header["positive.image_pad_mask"]["data_offsets"]
    refused("image conditioning mask refused after load",
            data[:base + start] + b"\x01" * (end - start) + data[base + end:], "all-False")

    masked = ep.serialize_branches(ep.validate_production_request(req), ep.BranchBundle(
        ep.EncoderBranch(torch.ones(1, 2, 4, dtype=torch.float16),
                         torch.ones(1, 2, dtype=torch.int64),
                         torch.zeros(1, 2, dtype=torch.bool))))
    mask_data = masked.blobs[0].data
    mask_base = 8 + int.from_bytes(mask_data[:8], "little")
    mask_header = json.loads(mask_data[8:mask_base])
    start, end = mask_header["positive.attention_mask"]["data_offsets"]
    for value, label, match in ((0, "zero-valid attention mask refused after load", "no valid token"),
                                (2, "nonbinary attention mask refused after load", "binary")):
        raw = value.to_bytes(8, "little") * ((end - start) // 8)
        refused(label, mask_data[:mask_base + start] + raw + mask_data[mask_base + end:],
                match, template=masked)
    try:
        ep.encode_with_supplied_model(Model(torch.float32), Processor(), req)
    except EncoderProtocolError:
        passed("upstream dtype mismatch rejected without a cast")
    else:
        raise AssertionError("dtype mismatch accepted")
    try:
        ep.validate_production_request(request(dtype="fp64"))
    except EncoderRequestError:
        passed("invalid production request rejected")
    else:
        raise AssertionError("invalid request accepted")
    assert not attempts, f"unexpected forbidden operations: {attempts}"
    passed("no network or writes outside fixture root during checks")
    print(f"{count} synthetic checks passed on Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
