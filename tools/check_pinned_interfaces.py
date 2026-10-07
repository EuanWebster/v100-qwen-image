#!/usr/bin/env python3
"""Pinned-upstream interface contract checker (task M0.2).

A *pure-stdlib, offline, CPU-only* source-signature checker. It parses the two
pinned upstream checkouts with ``ast`` and asserts that the exact interface
facts recorded in ``docs/interfaces.md`` still hold at those commits — the
cheap tripwire that tells us an interface drifted before anyone wires the
real adapter (M4). It imports NOTHING heavy (no torch/diffusers/transformers,
no network, no model files) and may run on any machine that merely *has* the
source trees.

Inputs are explicit — the trees are external pinned artifacts, never assumed:

    python tools/check_pinned_interfaces.py --diffusers-src DIR --transformers-src DIR
    # or env QWEN_DIFFUSERS_SRC / QWEN_TRANSFORMERS_SRC (defaults: the M0.1
    # audit checkout locations, which exist only on machines that ran M0)

Exit codes: 0 = every contract holds; 1 = at least one contract FAILED;
2 = an input tree is missing/unreadable (that is NOT a pass — callers in the
guarded CPU suite turn this into an explicit skip, never a silent success).

Contracts mirror docs/interfaces.md sections; each check prints one line.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

# ---------------------------------------------------------------------------
# tiny check harness

RESULTS: list[tuple[str, str, str]] = []  # (status, name, detail)


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(("PASS" if ok else "FAIL", name, detail))


def find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def find_func(node: ast.AST, name: str) -> ast.FunctionDef | None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name:
            return child
    return None


def param_map(fn: ast.FunctionDef) -> dict[str, ast.expr | None]:
    """arg name -> default expr (None if no default). Positional/kw merged."""
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args]
    defaults: list[ast.expr | None] = [None] * (len(names) - len(a.defaults)) + list(a.defaults)
    for kw, d in zip(a.kwonlyargs, a.kw_defaults):
        names.append(kw.arg)
        defaults.append(d)
    return dict(zip(names, defaults))


def const_of(expr: ast.expr | None):
    """Literal value of a simple default, or a stable marker string when the
    default is an expression we do not evaluate."""
    if expr is None:
        return "<required>"
    try:
        return ast.literal_eval(expr)
    except (ValueError, SyntaxError):
        return "<expr:" + type(expr).__name__ + ">"


def read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def sig(fn: ast.FunctionDef) -> tuple[list[str], dict]:
    names = [x.arg for x in fn.args.posonlyargs + fn.args.args]
    return names, {n: const_of(d) for n, d in param_map(fn).items()}


# ---------------------------------------------------------------------------
# contracts (docs/interfaces.md section → checks)

def check_pipeline(path: str) -> None:
    src = read(path)
    tree = ast.parse(src)
    cls = find_class(tree, "QwenImage21Pipeline")
    check("pipeline: class QwenImage21Pipeline exists", cls is not None)
    if cls is None:
        return

    init = find_func(cls, "__init__")
    names, defaults = sig(init)
    check(
        "ctor: (scheduler, vae, text_encoder, processor, transformer, sample_sigmas=None)",
        names[:6] == ["self", "scheduler", "vae", "text_encoder", "processor", "transformer"]
        and "sample_sigmas" in defaults and defaults["sample_sigmas"] is None,
        f"got {names[1:7]}",
    )
    check("ctor: sys_prompt is the documented constant",
          'sys_prompt = "Comprehend and analyze the provided prompt."' in src)
    check("ctor: raw t2i template attribute (NOT apply_chat_template formatting)",
          "self.prompt_template_t2i" in src and "self.prompt_template_ti2i" in src)
    check("ctor: _drop_idx derived from tokenized system message", "self._drop_idx = len(sys_tokens[0])" in src)

    enc = find_func(cls, "encode_prompt")
    names, defaults = sig(enc)
    check(
        "encode_prompt: exact parameter contract incl. image_pad_mask=None",
        names == ["self", "prompt", "image", "device", "num_images_per_prompt",
                  "prompt_embeds", "prompt_embeds_mask", "image_pad_mask"]
        and defaults["num_images_per_prompt"] == 1 and defaults["image_pad_mask"] is None,
        f"got {names}",
    )
    # v1 depends on the external-embeds T2I path reconstructing a zero image
    # pad mask; if upstream starts *requiring* it, this is drift.
    check("encode_prompt: external embeds without image rebuild a zero image_pad_mask",
          "image_pad_mask = prompt_embeds.new_zeros(prompt_embeds.shape[:2], dtype=torch.bool)" in src)
    # Documented hazard: embeds+mask repeat for num_images_per_prompt, the
    # image pad mask does NOT. If that ever changes, the hazard note must go.
    embeds_repeat = "prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)" in src
    mask_repeat = "prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)" in src
    imgmask_repeat = "image_pad_mask = image_pad_mask.repeat" in src
    check("hazard (documented, unfixed): image_pad_mask NOT repeated by num_images_per_prompt",
          embeds_repeat and mask_repeat and not imgmask_repeat)

    call = find_func(cls, "__call__")
    names, defaults = sig(call)
    expected = {"true_cfg_scale": 1.0, "num_inference_steps": 40, "sigmas": None,
                "num_images_per_prompt": 1, "negative_prompt": None,
                "negative_prompt_embeds": None, "negative_prompt_embeds_mask": None,
                "output_resolution": 1024, "use_kv_cache": True, "output_type": "pil",
                "prompt_embeds": None, "prompt_embeds_mask": None}
    drift = {k: (defaults.get(k, "<missing>"), v) for k, v in expected.items() if defaults.get(k, "<missing>") != v}
    check("call: pinned defaults (steps=40, scale=1.0, res=1024, kv=True, None-negatives)",
          not drift, f"drift {drift}")
    check("call: no image_pad_mask parameter (v1 documented limitation)",
          "image_pad_mask" not in names)

    check("CFG gate: active only when scale>1 AND a negative is non-None",
          "has_neg_prompt = negative_prompt is not None or negative_prompt_embeds is not None" in src
          and "do_true_cfg = true_cfg_scale > 1 and has_neg_prompt" in src)
    check("sigmas precedence: call sigmas > config sample_sigmas > linspace(steps)",
          "sigmas = self.config.sample_sigmas" in src
          and "np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)" in src)
    check("KV: gated on use_kv_cache AND transformer.config.causal_condition",
          "cache_enabled = use_kv_cache and self.transformer.config.causal_condition" in src)
    check("KV: cond/neg caches constructed fresh per __call__",
          "cond_cache = QwenImage21KVCache(num_blocks) if cache_enabled else None" in src
          and "neg_cache = QwenImage21KVCache(num_blocks) if cache_enabled and do_true_cfg else None" in src)
    # raw RMSNorm-neutralizing hook must stay (transformers pin still ties
    # hidden_states[-1]); removal without the transformers-side fix is drift.
    check("pre-RMSNorm hook: forward hook returning args[0] with finally-remove",
          "handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])" in src
          and "handle.remove()" in src)
    check("decode: per-channel latents_std/latents_mean (no scalar scale factor)",
          "latents = latents * latents_std + latents_mean" in src
          and "self.vae.config.latents_mean" in src)


def check_transformer(path: str) -> None:
    src = read(path)
    tree = ast.parse(src)
    attn = find_class(tree, "QwenImage21Attention")
    check("transformer: QwenImage21Attention exists", attn is not None)
    if attn is not None:
        defaults = {t.targets[0].id: t.value
                    for t in attn.body
                    if isinstance(t, ast.Assign) and len(t.targets) == 1
                    and isinstance(t.targets[0], ast.Name)}
        dflt = defaults.get("_default_processor_cls")
        avail = defaults.get("_available_processors")
        check("transformer: default processor is the segmented QwenImage21AttnProcessor",
              isinstance(dflt, ast.Name) and dflt.id == "QwenImage21AttnProcessor",
              ast.dump(dflt) if dflt else "missing")
        if isinstance(avail, (ast.List, ast.Tuple)):
            ids = [e.id for e in avail.elts if isinstance(e, ast.Name)]
            check("transformer: processors = [segmented, flex-optional]",
                  ids == ["QwenImage21AttnProcessor", "QwenImage21FlexAttnProcessor"], str(ids))
        else:
            check("transformer: processors = [segmented, flex-optional]", False, "missing")

    model = find_class(tree, "QwenImage21Transformer2DModel")
    check("transformer: model class exists", model is not None)
    if model is not None:
        init = find_func(model, "__init__")
        _, defaults = sig(init)
        check("transformer config: in_channels=64, num_layers=32, causal_condition=True",
              defaults.get("in_channels") == 64 and defaults.get("num_layers") == 32
              and defaults.get("causal_condition") is True,
              str({k: defaults.get(k) for k in ("in_channels", "num_layers", "causal_condition")}))
    check("transformer: img_mask layout read from row 0 only",
          "image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)" in src)
    check("transformer: slot-to-latent expansion factor is 4 (2x2)",
          "_IMG_TOKENS_PER_SLOT = 4" in src)
    # segmented processor MUST keep forcing backend=None on the prefill calls
    # (env-var/context-manager backends would be silently ignored otherwise —
    # docs/interfaces.md §6 documents that trap; a "fix" changes semantics).
    seg = find_class(tree, "QwenImage21AttnProcessor")
    body = read(path).split("class QwenImage21AttnProcessor")[1].split("class QwenImage21Attention")[0] \
        if seg else ""
    check("segmented processor: prefill dispatch calls use backend=None explicitly",
          body.count("backend=None") >= 2)


def check_vae(path: str) -> None:
    src = read(path)
    tree = ast.parse(src)
    cls = find_class(tree, "AutoencoderKLQwenImage21")
    check("vae: AutoencoderKLQwenImage21 exists (specific class, not generic QwenImage VAE)",
          cls is not None)
    if cls is not None:
        _, defaults = sig(find_func(cls, "__init__"))
        check("vae: class default z_dim=64 (checkpoint config is authoritative)",
              defaults.get("z_dim") == 64, str(defaults.get("z_dim")))


def check_gguf(path: str) -> None:
    src = read(path)
    check("gguf: CUDA kernels opt-in via env, default FALSE",
          'os.getenv("DIFFUSERS_GGUF_CUDA_KERNELS", "false")' in src)
    check("gguf: capability gate is major>=7 (sm_70 NOT excluded; must not become >7)",
          "torch.cuda.get_device_capability()[0] >= 7" in src
          and "torch.cuda.get_device_capability()[0] > 7" not in src)
    check("gguf: kernels absent -> ops=None (native fallback), no import-time Hub call",
          "ops = None" in src)
    check("gguf: fused MMQ/MMVQ remain disabled (dequantize fallback only)",
          "# so we disabled it now." in src
          and "# If there is no available MMQ kernel, fallback to dequantize" in src)
    lin = src.split("class GGUFLinear")[1] if "class GGUFLinear" in src else ""
    check("gguf: native linear dequantizes then casts to compute_dtype",
          "weight = dequantize_gguf_tensor(self.weight)" in lin
          and "weight = weight.to(self.compute_dtype)" in lin)


def check_singlefile(map_path: str, util_path: str) -> None:
    src = read(map_path)
    check("singlefile: QwenImage21Transformer2DModel registered",
          '"QwenImage21Transformer2DModel"' in src
          and "convert_qwen_image21_transformer_checkpoint_to_diffusers" in src)
    util = read(util_path)
    fn = find_func(ast.parse(util), "convert_qwen_image21_transformer_checkpoint_to_diffusers")
    check("singlefile: converter exists (transformer-only; NO pipeline entry)",
          fn is not None)
    if fn is not None:
        fsrc = ast.get_source_segment(util, fn) or ""
        check("singlefile: converter strips model.diffusion_model. prefix and splits fused gate_up",
              '"model.diffusion_model."' in fsrc and "gate_up" in fsrc and "chunk(2, dim=0)" in fsrc)


def check_transformers_side(path: str) -> None:
    src = read(path)
    tree = ast.parse(src)
    cls = find_class(tree, "Qwen3VLTextModel")
    check("text-encoder: Qwen3VLTextModel exists", cls is not None)
    if cls is None:
        return
    fwd = find_func(cls, "forward")
    deco = [ast.unparse(d) for d in fwd.decorator_list]
    # Contract: outputs are captured with the DEFAULT tie_last_hidden_states
    # behaviour, i.e. hidden_states[-1] is the NORMALIZED last_hidden_state —
    # which is why the pipeline installs the pre-RMSNorm hook. If upstream
    # ships huggingface/transformers#48087 and decorates with
    # tie_last_hidden_states=False, the pinning story changes and
    # docs/interfaces.md §2 must be re-surveyed: that is drift, not silence.
    check("text-encoder: @capture_outputs WITHOUT tie_last_hidden_states=False "
          "(pre-RMSNorm hook still required)",
          any(d == "capture_outputs" for d in deco)
          and not any("tie_last_hidden_states=False" in d for d in deco),
          f"decorators: {deco}")


# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diffusers-src", default=os.environ.get("QWEN_DIFFUSERS_SRC",
                      "/tmp/v100_m0_ml/src/diffusers"))
    ap.add_argument("--transformers-src", default=os.environ.get("QWEN_TRANSFORMERS_SRC",
                      "/tmp/v100_m0_ml/src/transformers"))
    args = ap.parse_args(argv)

    d = os.path.join(args.diffusers_src, "src", "diffusers")
    t = os.path.join(args.transformers_src, "src", "transformers")
    files = {
        "pipeline": os.path.join(d, "pipelines", "qwenimage21", "pipeline_qwenimage21.py"),
        "transformer": os.path.join(d, "models", "transformers", "transformer_qwenimage21.py"),
        "vae": os.path.join(d, "models", "autoencoders", "autoencoder_kl_qwenimage21.py"),
        "gguf": os.path.join(d, "quantizers", "gguf", "utils.py"),
        "sf_map": os.path.join(d, "loaders", "single_file_model.py"),
        "sf_util": os.path.join(d, "loaders", "single_file_utils.py"),
        "tvl": os.path.join(t, "models", "qwen3_vl", "modeling_qwen3_vl.py"),
    }
    missing = [p for p in files.values() if not os.path.isfile(p)]
    if missing:
        print("CHECKER NOT RUN — pinned source tree(s) missing (input is explicit; "
              "this is NOT a pass):")
        for p in missing:
            print(f"  missing: {p}")
        return 2

    check_pipeline(files["pipeline"])
    check_transformer(files["transformer"])
    check_vae(files["vae"])
    check_gguf(files["gguf"])
    check_singlefile(files["sf_map"], files["sf_util"])
    check_transformers_side(files["tvl"])

    failed = 0
    for status, name, detail in RESULTS:
        if status == "FAIL":
            failed += 1
        print(f"[{status}] {name}" + (f" — {detail}" if detail and status == "FAIL" else ""))
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} pinned-interface contracts hold")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
