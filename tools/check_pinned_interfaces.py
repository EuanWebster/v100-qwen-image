#!/usr/bin/env python3
"""Pinned-upstream interface contract checker (task M0.2).

Pure-stdlib (argparse/ast/os/sys), offline, CPU-only: parses the supplied
upstream source trees with ``ast`` and reports the interface contracts
recorded in ``docs/interfaces.md``. It imports nothing heavy (no
torch/diffusers/transformers), opens no sockets, reads no model files.

Every check carries an evidence kind:

  code    decided from real AST nodes — class/method presence, signature
          shape and defaults, real assignment expressions (including a
          UNIQUE-assignment requirement where the contract pins one
          binding), forward-hook registration anchored to the DIRECT
          encoder forward call (exact call target — not a method call
          merely containing the encoder's name) running inside the try
          whose finally removes that same
          registered handle. Comments and docstrings can
          NEVER satisfy a ``code`` check. Caveat that applies to BOTH kinds:
          AST structure is NOT runtime behaviour — a passing ``code`` check
          says the surveyed code SHAPE is present in the file, not that the
          interpreter reaches or executes it. Behavioural proof belongs to
          adapter-level tests (M4+).
  marker  a lightweight substring tripwire over the source TEXT. A marker
          PASS is NOT proof of runtime behaviour, numerical fidelity or
          absence of network calls; it only says the surveyed textual
          shape is still present. Behavioural proof belongs to
          adapter-level tests (M4+), not to this checker.

Required inputs are read as regular, nonblocking/no-follow snapshots capped
at 2 MiB per source file; stable explicit source aliases remain supported.
Definitions must be unique/direct and methods synchronous; hook/processor
targets and default literal types must match the surveyed source shape.
These checks still are NOT general binding/control-flow analysis.

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
import stat
import sys

_TRY_NODES = tuple(getattr(ast, n) for n in ("Try", "TryStar") if hasattr(ast, n))
MAX_SOURCE_BYTES = 2 * 1024 * 1024


class InputError(Exception):
    """A required input is missing, unreadable or not valid UTF-8."""


class Report:
    """Result rows for ONE checker invocation (status, kind, name, detail)."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def check(self, kind: str, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append(("PASS" if ok else "FAIL", kind, name, detail))


def read_file(path: str) -> str:
    """Bounded regular-file snapshot; stable explicit source aliases are allowed."""
    try:
        original = os.path.abspath(path)
        canonical = os.path.realpath(original, strict=True)
        fd = os.open(canonical, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_SOURCE_BYTES:
                raise InputError(f"{path}: source must be a regular file <= {MAX_SOURCE_BYTES} bytes")
            data = bytearray()
            while block := os.read(fd, min(65536, before.st_size + 1 - len(data))):
                data.extend(block)
                if len(data) > before.st_size:
                    raise InputError(f"{path}: source grew during read")
            def signature(info):
                return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                        info.st_mtime_ns, info.st_ctime_ns)
            if (len(data) != before.st_size or signature(before) != signature(os.fstat(fd))
                    or signature(before) != signature(os.stat(canonical, follow_symlinks=False))
                    or os.path.realpath(original, strict=True) != canonical):
                raise InputError(f"{path}: source changed during read")
            return data.decode("utf-8")
        finally:
            primary = sys.exception()
            try:
                os.close(fd)
            except OSError:
                if primary is None:
                    raise
    except OSError as exc:
        raise InputError(f"{path}: cannot read ({exc.strerror or exc})") from exc
    except UnicodeDecodeError as exc:
        raise InputError(f"{path}: not valid UTF-8 ({exc})") from exc


def parse_source(src: str, key: str, rep: Report) -> ast.Module | None:
    try:
        return ast.parse(src)
    except SyntaxError as exc:
        rep.check("code", f"{key}: source parses", False,
                  f"{type(exc).__name__}: {exc.msg} (line {exc.lineno})")
        return None
    except ValueError as exc:
        # ast.parse raises a bare ValueError for inputs a SyntaxError does not
        # cover (e.g. a source string containing a null byte). ValueError has
        # no .msg/.lineno — report it without touching those attributes.
        rep.check("code", f"{key}: source parses", False,
                  f"{type(exc).__name__}: {exc}")
        return None
    except RecursionError:
        rep.check("code", f"{key}: source parses", False, "parser recursion budget exceeded")
        return None


# ---------------------------------------------------------------------------
# small AST helpers

def find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    return _unique_definition(tree, name, (ast.ClassDef,))


def find_func(node: ast.AST, name: str) -> ast.FunctionDef | None:
    # Async methods return coroutines and are not the surveyed synchronous ABI.
    return _unique_definition(node, name, (ast.FunctionDef,))


def _unique_definition(scope: ast.AST, name: str, kinds):
    """One direct definition, not a nested/duplicate/directly overwritten lookalike.

    This remains a shape tripwire, not general binding/control-flow analysis.
    """
    body = getattr(scope, "body", ())
    matches = [child for child in body if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
               and child.name == name]
    if len(matches) != 1 or not isinstance(matches[0], kinds):
        return None
    for stmt in body:
        targets = (stmt.targets if isinstance(stmt, ast.Assign)
                   else [stmt.target] if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) else [])
        if any(isinstance(target, ast.Name) and target.id == name
               for root in targets for target in ast.walk(root)):
            return None
    return matches[0]


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
             if type(defaults.get(k)) is not type(v) or defaults.get(k, "<missing>") != v}
    missing_names = [n for n in pos_names if n not in names]
    rep.check("code", f"{label}: pinned defaults, no dropped parameters",
              not drift and not missing_names, f"drift={drift} missing={missing_names}")


def assigns_named(node: ast.AST, name: str):
    for n in ast.walk(node):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name):
            yield n.value


def assignments_to(node: ast.AST, name: str) -> list[ast.expr]:
    """Every value assigned to the local name ``name`` inside ``node``:
    plain/chained/tuple-target Assigns, AnnAssign with value, AugAssign
    (read-modify-write overwrite) and NamedExpr walrus. Used where the
    contract pins EXACTLY ONE binding — a second assignment anywhere in the
    walked body (even in a branch) is drift, even when the FIRST assignment
    has the documented shape (a next()-style scan would miss it). AST shape
    only: the checker cannot know which assignments an interpreter reaches."""
    vals: list[ast.expr] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    vals.append(n.value)
                    break
                if isinstance(t, (ast.Tuple, ast.List)) and any(
                        isinstance(e, ast.Name) and e.id == name for e in t.elts):
                    vals.append(n.value)
                    break
        elif isinstance(n, ast.AnnAssign):
            if isinstance(n.target, ast.Name) and n.target.id == name \
                    and n.value is not None:
                vals.append(n.value)
        elif isinstance(n, ast.AugAssign):
            if isinstance(n.target, ast.Name) and n.target.id == name:
                vals.append(n.value)
        elif isinstance(n, ast.NamedExpr):
            if isinstance(n.target, ast.Name) and n.target.id == name:
                vals.append(n.value)
    return vals


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
        return type(x.value) is int and x.value == 0
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
                and isinstance(n.value.func.value, ast.Attribute)
                and n.value.func.value.attr == "norm"
                and isinstance(n.value.func.value.value, ast.Name)
                and n.value.func.value.value.id == "text_model"
                and len(n.value.args) == 1 and not n.value.keywords
                and any(isinstance(a, ast.Lambda) and isinstance(a.body, ast.Subscript)
                        and [p.arg for p in a.args.args] == ["module", "args", "output"]
                        and not (a.args.posonlyargs or a.args.kwonlyargs or a.args.defaults
                                 or a.args.vararg or a.args.kwarg)
                        and isinstance(a.body.value, ast.Name) and a.body.value.id == "args"
                        and _is_zero_index(a.body.slice) for a in n.value.args)):
            if len(assignments_to(node, n.targets[0].id)) == 1:
                out.add(n.targets[0].id)
    return out


def _is_direct_encoder_forward(call: ast.Call) -> bool:
    """EXACT-call predicate for the text-encoder forward in the surveyed
    shape: the pipeline invokes the encoder as the DIRECT call
    ``self.text_encoder(...)`` (attribute target exactly ``self`` .
    ``text_encoder`` — the real pinned form); synthetic fixtures use the
    direct bare-name call ``text_model(...)``. NOTHING else is a forward:
    attribute/method calls that merely CONTAIN those words —
    ``self.text_encoder.eval()``, ``text_model.norm.register_forward_hook(...)``,
    ``text_model.norm(...)``, helper calls — are NOT forwards, and neither
    are dotted lookalikes such as ``self.text_model(...)`` or
    ``helper.text_model(...)``. A rename of the call target is drift — the
    FAIL detail shows the hook targets that WERE found, which is the point
    of the tripwire. Exact AST shape only: no control-flow or data-flow
    analysis, no execution claim."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return (func.attr == "text_encoder"
                and isinstance(func.value, ast.Name) and func.value.id == "self")
    return isinstance(func, ast.Name) and func.id == "text_model"


def encoder_forward_in_protected_try(node: ast.AST, targets: set[str]) -> bool:
    """True iff SOME try block in ``node`` has (a) its finally body calling
    ``remove()`` on one of the registered-hook ``targets`` and (b) a DIRECT
    text-encoder forward call (``_is_direct_encoder_forward`` —
    ``self.text_encoder(...)`` or bare ``text_model(...)``; an
    attribute/method call like ``self.text_encoder.eval()`` is NOT a
    forward) inside its own protected body (handlers, else and finally
    bodies do NOT count). This anchors the surveyed relationship —
    registration and a removal somewhere in the method are NOT enough; the
    forward must run under the protection that removes that same handle.
    AST shape only: this proves the surveyed SHAPE is present, not that the
    interpreter executes it, and not full exception safety (e.g. it cannot
    see raises between registration and try entry)."""
    if not targets:
        return False
    for n in ast.walk(node):
        if not isinstance(n, _TRY_NODES) or not n.finalbody:
            continue
        if not any(isinstance(m, ast.Call) and isinstance(m.func, ast.Attribute)
                   and m.func.attr == "remove"
                   and isinstance(m.func.value, ast.Name)
                   and m.func.value.id in targets
                   for stmt in n.finalbody for m in ast.walk(stmt)):
            continue
        for stmt in n.body:
            for m in ast.walk(stmt):
                if isinstance(m, ast.Call) and _is_direct_encoder_forward(m):
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
    # docstring must NOT satisfy this. EXACTLY the two documented operands
    # (extra ones are semantic drift even when both names appear —
    # `... or True`, `... and False` must FAIL), and EXACTLY ONE assignment
    # to the target in __call__: a later reassignment/overwrite of the same
    # name is drift even though a next()-style scan would still find the
    # documented first assignment. Shape tripwire, not runtime/dataflow proof.
    h_all = assignments_to(call, "has_neg_prompt")
    h = h_all[0] if len(h_all) == 1 and isinstance(h_all[0], ast.BoolOp) \
        and isinstance(h_all[0].op, ast.Or) else None
    neg_names = ([n for c in (h.values if h is not None else [])
                  for n in ("negative_prompt", "negative_prompt_embeds")
                  if _is_is_not_none(c, {n})])
    rep.check("code", "CFG gate: has_neg_prompt assigned EXACTLY ONCE in "
                      "__call__ (extra assignments/overwrites are drift) and "
                      "from EXACTLY the two negative_* 'is not None' checks, "
                      "no extra operands (real code)",
              len(h_all) == 1 and h is not None and len(h.values) == 2
              and sorted(neg_names) == ["negative_prompt",
                                        "negative_prompt_embeds"],
              f"assignments={len(h_all)}, "
              f"operands={len(h.values) if h is not None else 0}, "
              f"found {sorted(neg_names)}")

    def _scale_gt(c: ast.expr) -> bool:
        return (isinstance(c, ast.Compare) and len(c.ops) == 1
                and isinstance(c.ops[0], ast.Gt)
                and isinstance(c.left, ast.Name) and c.left.id == "true_cfg_scale"
                and isinstance(c.comparators[0], ast.Constant)
                and type(c.comparators[0].value) in (int, float)
                and c.comparators[0].value in (1, 1.0))

    def _is_hn(c: ast.expr) -> bool:
        return isinstance(c, ast.Name) and c.id == "has_neg_prompt"

    d_all = assignments_to(call, "do_true_cfg")
    d = d_all[0] if len(d_all) == 1 and isinstance(d_all[0], ast.BoolOp) \
        and isinstance(d_all[0].op, ast.And) else None
    dv = d.values if d is not None else []
    rep.check("code", "CFG gate: do_true_cfg assigned EXACTLY ONCE in "
                      "__call__ (extra assignments/overwrites are drift) as "
                      "EXACTLY (true_cfg_scale > 1) and has_neg_prompt, no "
                      "extra operands (real code)",
              len(d_all) == 1 and len(dv) == 2
              and ((_scale_gt(dv[0]) and _is_hn(dv[1]))
                   or (_scale_gt(dv[1]) and _is_hn(dv[0]))),
              f"assignments={len(d_all)}, operands={len(dv)}")

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

    # pre-RMSNorm hook: registration AND removal must be real AST code in the
    # ACTUAL prompt-encoding method (upstream pins it there), not just
    # anywhere in the class — a hook parked in dead code would be structure
    # without the surveyed call-site context. The relationship is anchored:
    # the DIRECT text-encoder forward call (exact call target per
    # _is_direct_encoder_forward — a method/attribute call such as
    # self.text_encoder.eval() is NOT a forward) must run INSIDE the try
    # whose finally removes that same registered handle — a registration
    # plus a removal sitting in
    # some unrelated finally does NOT protect the forward. The transformers
    # pin still ties hidden_states[-1]; removal without the transformers-side
    # fix is drift. AST structure is NOT runtime-behaviour proof and this
    # check does NOT prove execution or full exception safety (see module
    # docstring).
    pem = find_func(cls, "_get_qwen_prompt_embeds")
    rep.check("code", "pipeline: _get_qwen_prompt_embeds defined "
                      "(norm-hook scope anchor)", pem is not None)
    targets = hook_targets(pem) if pem is not None else set()
    rep.check("code", "pre-RMSNorm hook: forward hook returning args[0] is "
                      "registered on the text-model norm in real code inside "
                      "_get_qwen_prompt_embeds",
              bool(targets), f"targets={sorted(targets)}")
    rep.check("code", "pre-RMSNorm hook: direct encoder forward call (exact "
                      "call target, not a method call merely containing the "
                      "encoder's name) is inside the try whose finally "
                      "removes that registered handle "
                      "(real code inside _get_qwen_prompt_embeds; shape only "
                      "— not an execution or exception-safety proof)",
              encoder_forward_in_protected_try(pem, targets)
              if pem is not None else False,
              f"targets={sorted(targets)}")

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
        default_values = assignments_to(attn, "_default_processor_cls")
        available_values = assignments_to(attn, "_available_processors")
        dflt = default_values[0] if len(default_values) == 1 else None
        avail = available_values[0] if len(available_values) == 1 else None
        rep.check("code", "transformer: default processor is the segmented "
                          "QwenImage21AttnProcessor",
                  isinstance(dflt, ast.Name) and dflt.id == "QwenImage21AttnProcessor",
                  ast.unparse(dflt) if dflt is not None else "missing")
        if isinstance(avail, (ast.List, ast.Tuple)):
            ids = [e.id for e in avail.elts if isinstance(e, ast.Name)]
        else:
            ids = []
        rep.check("code", "transformer: processors = [segmented, flex-optional]",
                  ids == ["QwenImage21AttnProcessor", "QwenImage21FlexAttnProcessor"]
                  and len(avail.elts) == 2,
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
    seg_call = find_func(seg, "__call__") if seg is not None else None
    if seg_call is not None:
        for c in ast.walk(seg_call):
            if (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    and c.func.id == "dispatch_attention_fn" and any(
                    k.arg == "backend" and isinstance(k.value, ast.Constant)
                    and k.value.value is None for k in c.keywords)):
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
    fn = find_func(util_tree, "convert_qwen_image21_transformer_checkpoint_to_diffusers")
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
    for checker, keys in (
        (check_pipeline, ("pipeline",)), (check_transformer, ("transformer",)),
        (check_vae, ("vae",)), (check_gguf, ("gguf",)),
        (check_singlefile, ("sf_map", "sf_util")), (check_transformers_side, ("tvl",)),
    ):
        try:
            checker(*(sources[key] for key in keys), rep)
        except RecursionError:
            rep.check("code", f"{','.join(keys)}: AST analysis fits recursion budget",
                      False, "AST analysis recursion budget exceeded")

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
    print("NOTE: 'code' PASSes are AST-structure facts about the source files "
          "only — this checker never executes upstream code, so no check here "
          "proves what any interpreter would actually do at runtime.")
    print("NOTE: these are caller-provided source trees. This checker does NOT "
          "verify them against the pinned revisions in docs/interfaces.md — "
          "pinned-revision verification is a separate required step "
          "(e.g. git -C <tree> rev-parse HEAD vs the recorded hashes).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
