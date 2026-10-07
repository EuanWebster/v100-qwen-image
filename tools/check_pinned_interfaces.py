#!/usr/bin/env python3
"""Pinned-upstream interface contract checker (task M0.2).

Pure-stdlib (argparse/ast/os/sys), offline, CPU-only: parses the supplied
upstream source trees with ``ast`` and reports the interface contracts
recorded in ``docs/interfaces.md``. It imports nothing heavy (no
torch/diffusers/transformers), opens no sockets, reads no model files.

Every check carries an evidence kind:

  code    decided from real AST nodes — class/method presence, signature
          shape and defaults, real assignment expressions, forward-hook
          registration with finally-remove. Comments and docstrings can
          NEVER satisfy a ``code`` check.
  marker  a lightweight substring tripwire over the source TEXT. A marker
          PASS is NOT proof of runtime behaviour, numerical fidelity or
          absence of network calls; it only says the surveyed textual
          shape is still present. Behavioural proof belongs to
          adapter-level tests (M4+), not to this checker.

Inputs are explicit — there are NO implicit task-specific defaults:

    python tools/check_pinned_interfaces.py --diffusers-src DIR --transformers-src DIR
    # or env QWEN_DIFFUSERS_SRC / QWEN_TRANSFORMERS_SRC (required if flags absent)

REVISION LIMITATION: the trees are caller-provided. This checker does NOT
verify they are at the pinned commits recorded in docs/interfaces.md;
pinned-revision verification (e.g. ``git -C <tree> rev-parse HEAD``
compared with the recorded hashes) is a separate, required step.

Exit codes:
  0  every check passed
  1  at least one check FAILED (contract drift, or a source file that
     cannot be parsed — a SyntaxError is contract drift, not an input error)
  2  an input was not specified, or a required file is missing /
     unreadable / not valid UTF-8. Exit 2 is NOT a pass.

``main()`` keeps no module-level state: every invocation starts fresh.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

_TRY_NODES = tuple(getattr(ast, n) for n in ("Try", "TryStar") if hasattr(ast, n))


class InputError(Exception):
    """A required input is missing, unreadable or not valid UTF-8."""


class Report:
    """Result rows for ONE checker invocation (status, kind, name, detail)."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def check(self, kind: str, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append(("PASS" if ok else "FAIL", kind, name, detail))


def read_file(path: str) -> str:
    try:
        if not os.path.isfile(path):
            raise InputError(f"{path}: not a regular file")
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise InputError(f"{path}: cannot read ({exc.strerror or exc})") from exc
    except UnicodeDecodeError as exc:
        raise InputError(f"{path}: not valid UTF-8 ({exc})") from exc


def parse_source(src: str, key: str, rep: Report) -> ast.Module | None:
    try:
        return ast.parse(src)
    except (SyntaxError, ValueError) as exc:
        rep.check("code", f"{key}: source parses", False,
                  f"{type(exc).__name__}: {exc.msg} (line {exc.lineno})")
        return None


# ---------------------------------------------------------------------------
# small AST helpers

def find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
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
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    pos_defaults: list[ast.expr | None] = (
        [None] * (len(a.posonlyargs) + len(a.args) - len(a.defaults)) + list(a.defaults))
    out = dict(zip([x.arg for x in a.posonlyargs + a.args], pos_defaults))
    out.update({k.arg: d for k, d in zip(a.kwonlyargs, a.kw_defaults)})
    return out


def const_of(expr: ast.expr | None):
    if expr is None:
        return "<required>"
    try:
        return ast.literal_eval(expr)
    except (ValueError, SyntaxError):
        return "<expr:" + type(expr).__name__ + ">"


def _subsequence(sub: list[str], seq: list[str]) -> bool:
    it = iter(seq)
    return all(x in it for x in sub)


def check_signature(rep: Report, label: str, fn: ast.FunctionDef,
                    pos_names: list[str], required: set[str],
                    defaults_expected: dict) -> None:
    """Calling-shape contract: no positional-only args, no *args/**kwargs,
    documented positional order preserved, required-parameter set exact
    (catches newly required params/kwargs), documented defaults unchanged."""
    a = fn.args
    posonly = [x.arg for x in a.posonlyargs]
    args = [x.arg for x in a.args]
    kwonly = {k.arg: d for k, d in zip(a.kwonlyargs, a.kw_defaults)}
    shape_ok = (not posonly and a.vararg is None and a.kwarg is None
                and _subsequence(pos_names, args))
    rep.check("code", f"{label}: calling shape (no posonly/*args/**kwargs, "
                      "documented order)", shape_ok,
              f"posonly={posonly} vararg={a.vararg.arg if a.vararg else None} "
              f"kwarg={a.kwarg.arg if a.kwarg else None} args={args}")
    defaulted = set(args[len(args) - len(a.defaults):]) if a.defaults else set()
    required_actual = (set(args) - defaulted) | {k for k, d in kwonly.items() if d is None}
    rep.check("code", f"{label}: no newly-required / no-longer-required parameters",
              required_actual == required, f"required={sorted(required_actual)}")
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    defaults = {n: const_of(d) for n, d in param_map(fn).items()}
    drift = {k: (defaults.get(k, "<missing>"), v) for k, v in defaults_expected.items()
             if defaults.get(k, "<missing>") != v}
    missing_names = [n for n in pos_names if n not in names]
    rep.check("code", f"{label}: pinned defaults, no dropped parameters",
              not drift and not missing_names, f"drift={drift} missing={missing_names}")


def assigns_named(node: ast.AST, name: str):
    for n in ast.walk(node):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name):
            yield n.value


def has_call_attr(node: ast.AST, attr: str) -> bool:
    return any(isinstance(m, ast.Call) and isinstance(m.func, ast.Attribute)
               and m.func.attr == attr for m in ast.walk(node))


def _is_is_not_none(v: ast.expr, names: set[str]) -> bool:
    return (isinstance(v, ast.Compare) and len(v.ops) == 1
            and isinstance(v.ops[0], ast.IsNot)
            and isinstance(v.comparators[0], ast.Constant)
            and v.comparators[0].value is None
            and isinstance(v.left, ast.Name) and v.left.id in names)


def _is_zero_index(x) -> bool:
    if isinstance(x, ast.Constant):
        return x.value == 0
    index = getattr(ast, "Index", None)
    if index is not None and isinstance(x, index):
        return _is_zero_index(x.value)
    return False


def hook_targets(node: ast.AST) -> set[str]:
    """Names assigned from `....norm.register_forward_hook(lambda ...: args[0])`."""
    out = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == "register_forward_hook"
                and any(isinstance(m, ast.Attribute) and m.attr == "norm"
                        for m in ast.walk(n.value.func))
                and any(isinstance(a, ast.Lambda) and isinstance(a.body, ast.Subscript)
                        and isinstance(a.body.value, ast.Name) and a.body.value.id == "args"
                        and _is_zero_index(a.body.slice) for a in n.value.args)):
            out.add(n.targets[0].id)
    return out


def finally_removes(node: ast.AST, targets: set[str]) -> bool:
    if not targets:
        return False
    for n in ast.walk(node):
        if isinstance(n, _TRY_NODES) and n.finalbody:
            for stmt in n.finalbody:
                for m in ast.walk(stmt):
                    if (isinstance(m, ast.Call) and isinstance(m.func, ast.Attribute)
                            and m.func.attr == "remove"
                            and isinstance(m.func.value, ast.Name)
                            and m.func.value.id in targets):
                        return True
    return False


# ---------------------------------------------------------------------------
# contracts (docs/interfaces.md sections -> checks)

CTOR_NAMES = ["self", "scheduler", "vae", "text_encoder", "processor",
              "transformer", "sample_sigmas"]
CTOR_REQUIRED = {"self", "scheduler", "vae", "text_encoder", "processor", "transformer"}
ENC_NAMES = ["self", "prompt", "image", "device", "num_images_per_prompt",
             "prompt_embeds", "prompt_embeds_mask", "image_pad_mask"]
ENC_DEFAULTS = {"image": None, "device": None, "num_images_per_prompt": 1,
                "prompt_embeds": None, "prompt_embeds_mask": None,
                "image_pad_mask": None}
CALL_NAMES = ["self", "prompt", "image", "negative_prompt", "true_cfg_scale",
              "height", "width", "num_inference_steps", "sigmas",
              "num_images_per_prompt", "generator", "latents", "prompt_embeds",
              "prompt_embeds_mask", "negative_prompt_embeds",
              "negative_prompt_embeds_mask", "output_type", "return_dict",
              "attention_kwargs", "callback_on_step_end",
              "callback_on_step_end_tensor_inputs", "output_resolution",
              "use_kv_cache"]
CALL_DEFAULTS = {"prompt": None, "true_cfg_scale": 1.0,
                 "num_inference_steps": 40, "sigmas": None,
                 "num_images_per_prompt": 1, "negative_prompt": None,
                 "negative_prompt_embeds": None, "negative_prompt_embeds_mask": None,
                 "output_resolution": 1024, "use_kv_cache": True,
                 "output_type": "pil", "prompt_embeds": None,
                 "prompt_embeds_mask": None}


def check_pipeline(src: str, rep: Report) -> None:
    tree = parse_source(src, "pipeline", rep)
    if tree is None:
        return
    rep.check("marker", "pipeline: sys_prompt constant text",
              'sys_prompt = "Comprehend and analyze the provided prompt."' in src)
    rep.check("marker", "pipeline: raw prompt-template attributes present",
              "self.prompt_template_t2i" in src and "self.prompt_template_ti2i" in src)
    rep.check("marker", "pipeline: _drop_idx from tokenized system message",
              "self._drop_idx = len(sys_tokens[0])" in src)

    cls = find_class(tree, "QwenImage21Pipeline")
    rep.check("code", "pipeline: class QwenImage21Pipeline exists", cls is not None)
    if cls is None:
        return

    init = find_func(cls, "__init__")
    rep.check("code", "pipeline: __init__ defined", init is not None)
    if init is not None:
        check_signature(rep, "pipeline __init__", init, CTOR_NAMES,
                        CTOR_REQUIRED, {"sample_sigmas": None})

    enc = find_func(cls, "encode_prompt")
    rep.check("code", "pipeline: encode_prompt defined", enc is not None)
    if enc is not None:
        check_signature(rep, "pipeline encode_prompt", enc, ENC_NAMES,
                        {"self", "prompt"}, ENC_DEFAULTS)
        zm = next((v for v in assigns_named(enc, "image_pad_mask")
                   if has_call_attr(v, "new_zeros")), None)
        rep.check("code", "encode_prompt: external-embeds path assigns a rebuilt "
                          "zero image_pad_mask (new_zeros call)", zm is not None)
        pe = any(has_call_attr(v, "repeat") for v in assigns_named(enc, "prompt_embeds"))
        pm = any(has_call_attr(v, "repeat") for v in assigns_named(enc, "prompt_embeds_mask"))
        im = any(has_call_attr(v, "repeat") for v in assigns_named(enc, "image_pad_mask"))
        rep.check("code", "documented hazard: embeds/mask repeat by "
                          "num_images_per_prompt, image_pad_mask does NOT "
                          "(real assignments)", pe and pm and not im)

    call = find_func(cls, "__call__")
    rep.check("code", "pipeline: __call__ defined", call is not None)
    if call is None:
        return
    check_signature(rep, "pipeline __call__", call, CALL_NAMES,
                    {"self"}, CALL_DEFAULTS)
    names = [x.arg for x in call.args.posonlyargs + call.args.args + call.args.kwonlyargs]
    rep.check("code", "__call__: no image_pad_mask parameter "
                      "(v1 documented limitation)", "image_pad_mask" not in names)

    # CFG gate: real assignments in real code — a mention in a comment or
    # docstring must NOT satisfy this.
    h = next((v for v in assigns_named(call, "has_neg_prompt")
              if isinstance(v, ast.BoolOp) and isinstance(v.op, ast.Or)), None)
    neg_names = ({c.left.id for c in h.values
                  if _is_is_not_none(c, {"negative_prompt", "negative_prompt_embeds"})}
                 if h is not None else set())
    rep.check("code", "CFG gate: has_neg_prompt assigned from both "
                      "negative_* 'is not None' checks (real code)",
              neg_names == {"negative_prompt", "negative_prompt_embeds"},
              f"found {sorted(neg_names)}")
    d = next((v for v in assigns_named(call, "do_true_cfg")
              if isinstance(v, ast.BoolOp) and isinstance(v.op, ast.And)), None)
    gt = any(isinstance(c, ast.Compare) and len(c.ops) == 1
             and isinstance(c.ops[0], ast.Gt)
             and isinstance(c.left, ast.Name) and c.left.id == "true_cfg_scale"
             and isinstance(c.comparators[0], ast.Constant)
             and c.comparators[0].value in (1, 1.0)
             for c in (d.values if d is not None else []))
    hn = any(isinstance(c, ast.Name) and c.id == "has_neg_prompt"
             for c in (d.values if d is not None else []))
    rep.check("code", "CFG gate: do_true_cfg assigned as "
                      "(true_cfg_scale > 1) and has_neg_prompt (real code)",
              d is not None and gt and hn)

    rep.check("marker", "sigmas precedence text: call sigmas > config "
                        "sample_sigmas > linspace(steps)",
              "sigmas = self.config.sample_sigmas" in src
              and "np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)" in src)
    ce = next((v for v in assigns_named(call, "cache_enabled")
               if isinstance(v, ast.BoolOp) and isinstance(v.op, ast.And)), None)
    ce_ok = (ce is not None
             and any(isinstance(m, ast.Name) and m.id == "use_kv_cache" for m in ast.walk(ce))
             and any(isinstance(m, ast.Attribute) and m.attr == "causal_condition"
                     for m in ast.walk(ce)))
    rep.check("code", "KV: cache_enabled assigned as use_kv_cache AND "
                      "transformer.config.causal_condition (real code)", ce_ok)
    rep.check("marker", "KV: cond/neg cache construction text",
              "cond_cache = QwenImage21KVCache(num_blocks) if cache_enabled else None" in src
              and "neg_cache = QwenImage21KVCache(num_blocks) if cache_enabled "
                  "and do_true_cfg else None" in src)

    # pre-RMSNorm hook: registration AND finally-remove must be real AST code
    # (the transformers pin still ties hidden_states[-1]; removal without the
    # transformers-side fix is drift).
    targets = hook_targets(cls)
    rep.check("code", "pre-RMSNorm hook: forward hook returning args[0] is "
                      "registered on the text-model norm in real code",
              bool(targets), f"targets={sorted(targets)}")
    rep.check("code", "pre-RMSNorm hook: handle.remove() inside a finally "
                      "block in real code", finally_removes(cls, targets))

    rep.check("marker", "decode text: per-channel latents_std/latents_mean",
              "latents = latents * latents_std + latents_mean" in src
              and "self.vae.config.latents_mean" in src)


def check_transformer(src: str, rep: Report) -> None:
    tree = parse_source(src, "transformer", rep)
    if tree is None:
        return
    attn = find_class(tree, "QwenImage21Attention")
    rep.check("code", "transformer: QwenImage21Attention exists", attn is not None)
    if attn is not None:
        defaults = {t.targets[0].id: t.value
                    for t in attn.body
                    if isinstance(t, ast.Assign) and len(t.targets) == 1
                    and isinstance(t.targets[0], ast.Name)}
        dflt = defaults.get("_default_processor_cls")
        avail = defaults.get("_available_processors")
        rep.check("code", "transformer: default processor is the segmented "
                          "QwenImage21AttnProcessor",
                  isinstance(dflt, ast.Name) and dflt.id == "QwenImage21AttnProcessor",
                  ast.unparse(dflt) if dflt is not None else "missing")
        if isinstance(avail, (ast.List, ast.Tuple)):
            ids = [e.id for e in avail.elts if isinstance(e, ast.Name)]
        else:
            ids = []
        rep.check("code", "transformer: processors = [segmented, flex-optional]",
                  ids == ["QwenImage21AttnProcessor", "QwenImage21FlexAttnProcessor"],
                  str(ids))

    model = find_class(tree, "QwenImage21Transformer2DModel")
    rep.check("code", "transformer: model class exists", model is not None)
    if model is not None:
        init = find_func(model, "__init__")
        rep.check("code", "transformer: model __init__ defined", init is not None)
        if init is not None:
            check_signature(rep, "transformer model __init__", init,
                            ["self", "patch_size", "in_channels", "out_channels",
                             "num_layers", "attention_head_dim",
                             "num_attention_heads", "context_in_dim", "mlp_ratio",
                             "axes_dims_rope", "eps", "causal_condition"],
                            {"self"},
                            {"in_channels": 64, "num_layers": 32,
                             "causal_condition": True})
    rep.check("marker", "transformer text: img_mask layout read from row 0 only",
              "image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)" in src)
    rep.check("marker", "transformer text: slot-to-latent expansion factor is 4 (2x2)",
              "_IMG_TOKENS_PER_SLOT = 4" in src)

    seg = find_class(tree, "QwenImage21AttnProcessor")
    n_none = 0
    if seg is not None:
        for c in ast.walk(seg):
            if isinstance(c, ast.Call) and any(
                    k.arg == "backend" and isinstance(k.value, ast.Constant)
                    and k.value.value is None for k in c.keywords):
                n_none += 1
    rep.check("code", "segmented processor: >=2 dispatch calls pass "
                      "backend=None explicitly (real call nodes, not text)",
              n_none >= 2, f"count={n_none}")


def check_vae(src: str, rep: Report) -> None:
    tree = parse_source(src, "vae", rep)
    if tree is None:
        return
    cls = find_class(tree, "AutoencoderKLQwenImage21")
    rep.check("code", "vae: AutoencoderKLQwenImage21 exists (specific class, "
                      "not generic QwenImage VAE)", cls is not None)
    if cls is None:
        return
    init = find_func(cls, "__init__")
    rep.check("code", "vae: __init__ defined", init is not None)
    if init is not None:
        defaults = {n: const_of(d) for n, d in param_map(init).items()}
        rep.check("code", "vae: class default z_dim=64 (checkpoint config is "
                          "authoritative)", defaults.get("z_dim") == 64,
                  str(defaults.get("z_dim", "<missing>")))


def check_gguf(src: str, rep: Report) -> None:
    tree = parse_source(src, "gguf utils", rep)
    if tree is None:
        return
    rep.check("marker", "gguf text: CUDA kernels opt-in via env, default false",
              'os.getenv("DIFFUSERS_GGUF_CUDA_KERNELS", "false")' in src)
    rep.check("marker", "gguf text: capability gate is major>=7 (sm_70 NOT "
                        "excluded; must not become >7)",
              "torch.cuda.get_device_capability()[0] >= 7" in src
              and "torch.cuda.get_device_capability()[0] > 7" not in src)
    rep.check("marker", "gguf text: 'ops = None' present (native fallback path; "
                        "marker only — NOT proof of no network calls)",
              "ops = None" in src)
    rep.check("marker", "gguf text: fused MMQ/MMVQ disabled comments present",
              "# so we disabled it now." in src
              and "# If there is no available MMQ kernel, fallback to dequantize" in src)
    lin = src.split("class GGUFLinear")[1] if "class GGUFLinear" in src else ""
    rep.check("marker", "gguf text: native linear dequantize + compute_dtype cast",
              "weight = dequantize_gguf_tensor(self.weight)" in lin
              and "weight = weight.to(self.compute_dtype)" in lin)


def check_singlefile(map_src: str, util_src: str, rep: Report) -> None:
    map_tree = parse_source(map_src, "singlefile map", rep)
    if map_tree is None:
        return
    rep.check("marker", "singlefile text: QwenImage21Transformer2DModel registered",
              '"QwenImage21Transformer2DModel"' in map_src
              and "convert_qwen_image21_transformer_checkpoint_to_diffusers" in map_src)
    util_tree = parse_source(util_src, "singlefile util", rep)
    if util_tree is None:
        return
    fn = next((n for n in ast.walk(util_tree)
               if isinstance(n, ast.FunctionDef)
               and n.name == "convert_qwen_image21_transformer_checkpoint_to_diffusers"),
              None)
    rep.check("code", "singlefile: converter function exists (transformer-only; "
                      "NO pipeline entry)", fn is not None)
    if fn is not None:
        fsrc = ast.get_source_segment(util_src, fn) or ""
        rep.check("marker", "singlefile text: converter strips "
                            "model.diffusion_model. prefix, splits fused gate_up",
                  '"model.diffusion_model."' in fsrc and "gate_up" in fsrc
                  and "chunk(2, dim=0)" in fsrc)


def check_transformers_side(src: str, rep: Report) -> None:
    tree = parse_source(src, "text-encoder (qwen3_vl)", rep)
    if tree is None:
        return
    cls = find_class(tree, "Qwen3VLTextModel")
    rep.check("code", "text-encoder: Qwen3VLTextModel exists", cls is not None)
    if cls is None:
        return
    fwd = find_func(cls, "forward")
    rep.check("code", "text-encoder: Qwen3VLTextModel.forward exists", fwd is not None)
    if fwd is None:
        return
    deco = [ast.unparse(d) for d in fwd.decorator_list]
    # Contract: @capture_outputs with the DEFAULT tie_last_hidden_states
    # behaviour (hidden_states[-1] is the NORMALIZED last_hidden_state), which
    # is why the pipeline installs the pre-RMSNorm hook. If upstream ships the
    # tie_last_hidden_states=False change, the pinning story changes: drift.
    rep.check("code", "text-encoder: @capture_outputs WITHOUT "
                      "tie_last_hidden_states=False (pre-RMSNorm hook still "
                      "required)",
              any(d == "capture_outputs" for d in deco)
              and not any("tie_last_hidden_states=False" in d for d in deco),
              f"decorators: {deco}")


# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diffusers-src", default=None,
                    help="diffusers checkout root (or env QWEN_DIFFUSERS_SRC)")
    ap.add_argument("--transformers-src", default=None,
                    help="transformers checkout root (or env QWEN_TRANSFORMERS_SRC)")
    args = ap.parse_args(argv)

    d = args.diffusers_src or os.environ.get("QWEN_DIFFUSERS_SRC")
    t = args.transformers_src or os.environ.get("QWEN_TRANSFORMERS_SRC")
    unset = [label for label, v in (("--diffusers-src / QWEN_DIFFUSERS_SRC", d),
                                    ("--transformers-src / QWEN_TRANSFORMERS_SRC", t))
             if not v]
    if unset:
        print("CHECKER NOT RUN — input not specified (this is NOT a pass). No "
              "implicit defaults exist; pass flags or set the env vars:")
        for label in unset:
            print(f"  unset: {label}")
        return 2

    files = {
        "pipeline": os.path.join(d, "src", "diffusers", "pipelines",
                                 "qwenimage21", "pipeline_qwenimage21.py"),
        "transformer": os.path.join(d, "src", "diffusers", "models", "transformers",
                                    "transformer_qwenimage21.py"),
        "vae": os.path.join(d, "src", "diffusers", "models", "autoencoders",
                            "autoencoder_kl_qwenimage21.py"),
        "gguf": os.path.join(d, "src", "diffusers", "quantizers", "gguf", "utils.py"),
        "sf_map": os.path.join(d, "src", "diffusers", "loaders", "single_file_model.py"),
        "sf_util": os.path.join(d, "src", "diffusers", "loaders", "single_file_utils.py"),
        "tvl": os.path.join(t, "src", "transformers", "models", "qwen3_vl",
                            "modeling_qwen3_vl.py"),
    }
    missing = [p for p in files.values() if not os.path.isfile(p)]
    if missing:
        print("CHECKER NOT RUN — required source file(s) missing (input trees "
              "are explicit; this is NOT a pass):")
        for p in missing:
            print(f"  missing: {p}")
        return 2

    sources = {}
    try:
        for key, path in files.items():
            sources[key] = read_file(path)
    except InputError as exc:
        print(f"CHECKER NOT RUN — {exc} (this is NOT a pass)")
        return 2

    rep = Report()
    check_pipeline(sources["pipeline"], rep)
    check_transformer(sources["transformer"], rep)
    check_vae(sources["vae"], rep)
    check_gguf(sources["gguf"], rep)
    check_singlefile(sources["sf_map"], sources["sf_util"], rep)
    check_transformers_side(sources["tvl"], rep)

    failed = sum(1 for r in rep.rows if r[0] == "FAIL")
    code_n = sum(1 for r in rep.rows if r[1] == "code")
    for status, kind, name, detail in rep.rows:
        line = f"[{status}][{kind}] {name}"
        print(line + (f" — {detail}" if detail and status == "FAIL" else ""))
    print(f"\n{len(rep.rows) - failed}/{len(rep.rows)} checks passed "
          f"({code_n} code-level AST checks, {len(rep.rows) - code_n} source markers)")
    print("NOTE: 'source marker' PASSes are lightweight textual tripwires, NOT "
          "proof of runtime behaviour, numerical fidelity or absence of network "
          "calls; behavioural proof belongs to adapter-level tests.")
    print("NOTE: these are caller-provided source trees. This checker does NOT "
          "verify them against the pinned revisions in docs/interfaces.md — "
          "pinned-revision verification is a separate required step "
          "(e.g. git -C <tree> rev-parse HEAD vs the recorded hashes).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
