#!/usr/bin/env python3
"""FP16 activation-overflow probe for the Qwen-Image-2.1 transformer (GPU, opt-in).

Reference-edit outputs look over-sharpened/HDR-grainy on the V100 while T2I is
clean; the suspect is fp16 activation overflow/saturation on the long joint
(text+vision+reference+target) sequence that upstream runs in bf16. For ONE
run (root executes; same CLI flags as a normal run, KV cache forced OFF like
output/gpu/kvoff_run.py), this registers forward hooks on every transformer
block AND its ``attn``/``img_mlp`` submodules (the block output clips to
+-65504 in fp16, so the submodules show the true pre-clip values), and per
forward call records, for the text / reference-image / target-image token
slices separately: max |x|, count of inf, count of NaN, count of |x| over a
threshold (default 30000). The slices come from the pinned
``QwenImage21Transformer2DModel.build_token_metadata`` itself — never a
re-implementation. A no-reference run (omit --reference) is the T2I baseline.
Writes a compact JSON summary (no prompts). ``--probe-plan`` prints the hook
plan and exits without importing anything ML (CPU-safe).
"""
from __future__ import annotations

import argparse
import json
import sys

SCHEMA = "qwen-fp16-activation-probe-v1"
PROBE_PREFIX = "transformer.transformer_blocks[i]."


def _plan_document(threshold: float, out: str, passthrough: list[str]) -> dict:
    return {
        "schema": SCHEMA, "mode": "plan", "ml_imports": False,
        "kv_cache": "off (SuppliedQwen21Adapter.__init__ patched, kvoff_run pattern)",
        "threshold": threshold, "out": out,
        "passthrough_flags": sorted(t for t in passthrough if t.startswith("-")),
        "hook_targets": [PROBE_PREFIX + suffix for suffix in ("", "attn", "img_mlp")],
        "per_call_record": "max_abs, inf, nan, over_threshold per stream",
        "streams": ["text", "reference_image", "target_image"],
        "stream_source": ("pinned build_token_metadata(image_pad_mask, img_shapes): "
                          "image_ids<0 text, 0<=id<last reference blocks, last block target"),
        "baseline": "omit every --reference flag for the T2I comparison run",
        "caveat": ("text slice includes right-padding kept out of attention only; "
                   "block-output hook is post-clip, attn/img_mlp hooks are pre-clip"),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--probe-plan", action="store_true")
    ap.add_argument("--probe-out", default="fp16_probe.json")
    ap.add_argument("--probe-threshold", type=float, default=30000.0)
    known, passthrough = ap.parse_known_args(argv)
    if known.probe_plan:
        print(json.dumps(_plan_document(known.probe_threshold, known.probe_out,
                                        passthrough), indent=2))
        return 0

    import torch  # noqa: F401  (GPU path only)

    from qwen_engine.adapters import supplied_qwen21 as supplied
    from qwen_engine.cli import main as cli_main

    holder: dict = {"threshold": known.probe_threshold, "records": {},
                    "handles": [], "calls": 0}

    orig_init = supplied.SuppliedQwen21Adapter.__init__

    def _init(self, *args, **kwargs):  # kvoff_run.py: force the KV cache off
        kwargs["use_kv_cache"] = False
        orig_init(self, *args, **kwargs)

    def _slices(tensor):
        masks = holder.get("masks")
        if not masks or tensor.dim() != 3 or tensor.shape[1] != masks["seq_len"]:
            return {}
        out = {}
        for name, mask in masks.items():
            if name == "seq_len":
                continue
            piece = tensor.detach()[:, mask]
            magnitude = piece.abs()
            peak = float(magnitude.max()) if magnitude.numel() else 0.0
            if peak != peak or peak == float("inf"):
                peak = None  # strict-JSON safe; the inf/nan counts carry truth
            out[name] = [peak, int(torch.isinf(piece).sum()),
                         int(torch.isnan(piece).sum()),
                         int((magnitude > known.probe_threshold).sum())]
        return out

    def make_hook(key):
        def hook(_module, _inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            holder["records"].setdefault(key, []).append(
                {"call": holder["calls"], "timestep": holder.get("timestep"),
                 "streams": _slices(tensor)})
        return hook

    def _install(pipeline):
        if holder["handles"]:
            return  # single run: hooks exist already, never double-register
        model = pipeline.transformer
        cls = type(model)
        orig_btm = cls.build_token_metadata

        def build_token_metadata(image_pad_mask, img_shapes):
            image_ids, target_token_mask = orig_btm(image_pad_mask, img_shapes)
            if "masks" not in holder:
                ids = image_ids.detach()
                target = target_token_mask.detach().bool()
                holder["masks"] = {
                    "seq_len": int(ids.shape[0]),
                    "text": ids < 0,
                    "reference_image": (ids >= 0) & ~target,
                    "target_image": target,
                }
            return image_ids, target_token_mask

        cls.build_token_metadata = staticmethod(build_token_metadata)

        def record_timestep(_module, args, kwargs):
            timestep = kwargs.get("timestep", args[2] if len(args) > 2 else None)
            holder["timestep"] = [float(v) for v in timestep.flatten()]
            holder["calls"] += 1

        holder["handles"].append(
            model.register_forward_pre_hook(record_timestep, with_kwargs=True))
        for index, block in enumerate(model.transformer_blocks):
            for suffix, module in (("", block), ("attn", block.attn),
                                   ("img_mlp", block.img_mlp)):
                key = f"block{index}" + (f".{suffix}" if suffix else "")
                holder["handles"].append(module.register_forward_hook(make_hook(key)))
        holder["blocks"] = len(model.transformer_blocks)

    orig_denoise = supplied.SuppliedQwen21Adapter.denoise

    def _denoise(self, config, conditioning):
        _install(self._pipeline)
        return orig_denoise(self, config, conditioning)

    supplied.SuppliedQwen21Adapter.__init__ = _init
    supplied.SuppliedQwen21Adapter.denoise = _denoise
    try:
        rc = cli_main(list(passthrough))
    finally:
        for handle in holder["handles"]:
            handle.remove()
        masks = holder.get("masks", {})
        document = {
            "schema": SCHEMA, "mode": "run", "kv_cache": "off",
            "threshold": known.probe_threshold, "forwards": holder["calls"],
            "blocks": holder.get("blocks"),
            "stream_token_counts": {name: int(mask.sum())
                                    for name, mask in masks.items()
                                    if name != "seq_len"},
            "records": holder["records"],
        }
        with open(known.probe_out, "w", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
