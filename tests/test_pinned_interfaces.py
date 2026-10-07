"""Pinned-interface contract checker tests (task M0.2).

Two parts, deliberately separated:

1. ALWAYS-RUN synthetic regressions — self-contained fixture source trees
   written into pytest's tmp dir. They exercise the checker's own contract
   logic (fresh state per invocation, missing methods, signature drift,
   calling-kind changes, SyntaxError, input errors, AST-vs-comment
   evidence) with NO external checkout and no ML/network imports.

2. An OPTIONAL external-tree check, explicitly SKIPPED (never silently
   passed) unless QWEN_DIFFUSERS_SRC / QWEN_TRANSFORMERS_SRC are provided.
   A skip says "the pinned-interface contract was NOT checked on this
   run", never "it passed" (pinned commits: docs/interfaces.md — and note
   the checker itself never verifies tree revisions; see its NOTEs).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_pinned_interfaces.py"

_diffusers = os.environ.get("QWEN_DIFFUSERS_SRC")
_transformers = os.environ.get("QWEN_TRANSFORMERS_SRC")


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_pinned_interfaces", CHECKER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# synthetic fixture sources (pure-text; never imported/executed)

PIPELINE = '''
class QwenImage21Pipeline:
    def __init__(self, scheduler, vae, text_encoder, processor, transformer,
                 sample_sigmas=None):
        sys_prompt = "Comprehend and analyze the provided prompt."
        self.prompt_template_t2i = T2I_TEMPLATE
        self.prompt_template_ti2i = TI2I_TEMPLATE
        self._drop_idx = len(sys_tokens[0])

    def _get_qwen_prompt_embeds(self, prompt, device):
        handle = text_model.norm.register_forward_hook(
            lambda module, args, output: args[0])
        try:
            outs = text_model(input_ids=input_ids)
        finally:
            handle.remove()
        return outs

    def encode_prompt(self, prompt, image=None, device=None,
                      num_images_per_prompt=1, prompt_embeds=None,
                      prompt_embeds_mask=None, image_pad_mask=None):
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
        image_pad_mask = prompt_embeds.new_zeros(prompt_embeds.shape[:2],
                                                 dtype=torch.bool)
        return prompt_embeds, prompt_embeds_mask, image_pad_mask

    def __call__(self, prompt=None, image=None, negative_prompt=None,
                 true_cfg_scale=1.0, height=None, width=None,
                 num_inference_steps=40, sigmas=None, num_images_per_prompt=1,
                 generator=None, latents=None, prompt_embeds=None,
                 prompt_embeds_mask=None, negative_prompt_embeds=None,
                 negative_prompt_embeds_mask=None, output_type="pil",
                 return_dict=True, attention_kwargs=None,
                 callback_on_step_end=None,
                 callback_on_step_end_tensor_inputs=None,
                 output_resolution=1024, use_kv_cache=True):
        has_neg_prompt = negative_prompt is not None or negative_prompt_embeds is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        cache_enabled = use_kv_cache and self.transformer.config.causal_condition
        cond_cache = QwenImage21KVCache(num_blocks) if cache_enabled else None
        neg_cache = QwenImage21KVCache(num_blocks) if cache_enabled and do_true_cfg else None
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        sigmas = self.config.sample_sigmas
        latents_mean = self.vae.config.latents_mean
        latents = latents * latents_std + latents_mean
        return latents
'''

TRANSFORMER = '''
class QwenImage21AttnProcessor:
    def __call__(self, attn, query, key, value):
        h1 = dispatch_attention_fn(query, key, value, backend=None)
        h2 = dispatch_attention_fn(query, key, value, backend=None)
        return h1, h2

class QwenImage21Attention:
    _default_processor_cls = QwenImage21AttnProcessor
    _available_processors = [QwenImage21AttnProcessor, QwenImage21FlexAttnProcessor]

class QwenImage21Transformer2DModel:
    def __init__(self, patch_size=1, in_channels=64, out_channels=64,
                 num_layers=32, attention_head_dim=128, num_attention_heads=32,
                 context_in_dim=4096, mlp_ratio=3, axes_dims_rope=(16, 56, 56),
                 eps=1e-06, causal_condition=True):
        self.in_channels = in_channels

image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
_IMG_TOKENS_PER_SLOT = 4
'''

VAE = '''
class AutoencoderKLQwenImage21:
    def __init__(self, in_channels=16, z_dim=64):
        self.z_dim = z_dim
'''

GGUF = '''
import os

use_cuda = os.getenv("DIFFUSERS_GGUF_CUDA_KERNELS", "false") == "true"
cap_ok = torch.cuda.get_device_capability()[0] >= 7
ops = None
# so we disabled it now.
# If there is no available MMQ kernel, fallback to dequantize

class GGUFLinear:
    def forward(self, x):
        weight = dequantize_gguf_tensor(self.weight)
        weight = weight.to(self.compute_dtype)
        return x @ weight
'''

SF_MAP = '''
CONFIG_MAP = {
    "QwenImage21Transformer2DModel":
        convert_qwen_image21_transformer_checkpoint_to_diffusers,
}
'''

SF_UTIL = '''
def convert_qwen_image21_transformer_checkpoint_to_diffusers(state_dict):
    prefix = "model.diffusion_model."
    gate_up = state_dict[prefix + "ff.gate_up"]
    gate, up = chunk(2, dim=0)
    return gate, up
'''

TVL = '''
class Qwen3VLTextModel:
    @merge_with_config_defaults
    @capture_outputs
    def forward(self, hidden_states):
        return hidden_states
'''

_GOOD = {
    "pipeline": PIPELINE, "transformer": TRANSFORMER, "vae": VAE,
    "gguf": GGUF, "sf_map": SF_MAP, "sf_util": SF_UTIL, "tvl": TVL,
}


def _write_tree(base: Path, **overrides) -> tuple[str, str]:
    droot, troot = base / "diffusers", base / "transformers"
    layout = {
        "pipeline": ("d", "pipelines/qwenimage21/pipeline_qwenimage21.py"),
        "transformer": ("d", "models/transformers/transformer_qwenimage21.py"),
        "vae": ("d", "models/autoencoders/autoencoder_kl_qwenimage21.py"),
        "gguf": ("d", "quantizers/gguf/utils.py"),
        "sf_map": ("d", "loaders/single_file_model.py"),
        "sf_util": ("d", "loaders/single_file_utils.py"),
        "tvl": ("t", "models/qwen3_vl/modeling_qwen3_vl.py"),
    }
    contents = dict(_GOOD, **overrides)
    for key, (which, rel) in layout.items():
        pkg = "diffusers" if which == "d" else "transformers"
        path = (droot if which == "d" else troot) / "src" / pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = contents[key]
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data, encoding="utf-8")
    return str(droot), str(troot)


def _run(mod, base, **overrides) -> tuple[int, str, str]:
    import contextlib
    import io
    d, t = _write_tree(base, **overrides)
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        rc = mod.main(["--diffusers-src", d, "--transformers-src", t])
    return rc, buf_out.getvalue(), buf_err.getvalue()


# --- 1. always-run synthetic regressions (no external trees needed) ---------

def test_synthetic_good_tree_passes(tmp_path):
    rc, out, err = _run(_load_checker(), tmp_path)
    assert rc == 0, out + err
    assert "NOT RUN" not in out
    assert "checks passed" in out


def test_fresh_results_fail_then_pass_same_module(tmp_path):
    """No state may leak between main() invocations on one loaded module:
    a failing run must not poison the next passing run (and vice versa)."""
    mod = _load_checker()
    broken = tmp_path / "broken"
    rc_bad, out_bad, _ = _run(mod, broken, tvl='class Qwen3VLTextModel:\n    pass\n')
    assert rc_bad == 1 and "FAIL" in out_bad
    good = tmp_path / "good"
    rc_ok, out_ok, err_ok = _run(mod, good)
    assert rc_ok == 0, out_ok + err_ok
    assert "[FAIL]" not in out_ok, f"prior failure leaked into fresh run:\n{out_ok}"
    assert "NOT RUN" not in out_ok


def test_missing_method_fails_cleanly_no_crash(tmp_path):
    """Removed forward()/init/__call__ must produce exit 1 + a FAIL line,
    never an uncaught AttributeError (the old sig(None) class of crash)."""
    mod = _load_checker()
    cases = {
        "tvl": 'class Qwen3VLTextModel:\n    @capture_outputs\n    def not_forward(self):\n        return 1\n',
        "pipeline": 'class QwenImage21Pipeline:\n    def other(self):\n        return 1\n',
        "vae": 'class AutoencoderKLQwenImage21:\n    def encode(self):\n        return 1\n',
    }
    for key, src in cases.items():
        rc, out, err = _run(mod, tmp_path / key, **{key: src})
        assert rc == 1, f"{key}: rc={rc}\n{out}\n{err}"
        assert "Traceback" not in err and "AttributeError" not in err, err
        assert "FAIL" in out


def test_newly_required_parameter_is_drift(tmp_path):
    bad = PIPELINE.replace(
        "processor, transformer,\n                 sample_sigmas=None):",
        "processor, transformer, rope_theta,\n                 sample_sigmas=None):")
    assert bad != PIPELINE
    rc, out, err = _run(_load_checker(), tmp_path, pipeline=bad)
    assert rc == 1, out + err
    assert "newly-required" in out


def test_kwonly_required_parameter_is_drift(tmp_path):
    bad = PIPELINE.replace(
        "prompt_embeds_mask=None, image_pad_mask=None):",
        "prompt_embeds_mask=None, *, image_pad_mask):")
    assert bad != PIPELINE
    rc, out, err = _run(_load_checker(), tmp_path, pipeline=bad)
    assert rc == 1, out + err
    assert "newly-required" in out


def test_changed_calling_kind_is_drift(tmp_path):
    """Positional-only parameters break keyword calling — a kind change."""
    bad = PIPELINE.replace(
        "processor, transformer,\n                 sample_sigmas=None):",
        "processor, transformer,\n                 sample_sigmas=None, /):")
    assert bad != PIPELINE
    rc, out, err = _run(_load_checker(), tmp_path, pipeline=bad)
    assert rc == 1, out + err
    assert "calling shape" in out


def test_syntax_error_is_contract_failure_exit1(tmp_path):
    rc, out, err = _run(_load_checker(), tmp_path,
                        pipeline="class QwenImage21Pipeline:\n  def broken(:\n")
    assert rc == 1, out + err
    assert "source parses" in out and "Traceback" not in err


def test_missing_file_returns2_not_a_pass(tmp_path):
    mod = _load_checker()
    d, t = _write_tree(tmp_path)
    (Path(d) / "src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py").unlink()
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main(["--diffusers-src", d, "--transformers-src", t])
    assert rc == 2 and "NOT RUN" in buf.getvalue()


def test_unreadable_file_returns2(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("running as root: file permissions cannot make a file unreadable")
    mod = _load_checker()
    d, t = _write_tree(tmp_path)
    p = Path(d) / "src/diffusers/models/transformers/transformer_qwenimage21.py"
    p.chmod(0o000)
    try:
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main(["--diffusers-src", d, "--transformers-src", t])
        assert rc == 2 and "NOT RUN" in buf.getvalue()
        assert "cannot read" in buf.getvalue()
    finally:
        p.chmod(0o644)


def test_non_utf8_file_returns2(tmp_path):
    rc, out, err = _run(_load_checker(), tmp_path,
                        pipeline=b"class QwenImage21Pipeline:\n    x = \xff\xfe\xba\xdd ?\n")
    assert rc == 2, out + err
    assert "NOT RUN" in out and "UTF-8" in out


def test_inputs_are_explicit_no_implicit_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN_DIFFUSERS_SRC", raising=False)
    monkeypatch.delenv("QWEN_TRANSFORMERS_SRC", raising=False)
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _load_checker().main([])
    assert rc == 2 and "NOT RUN" in buf.getvalue()
    assert "implicit defaults" in buf.getvalue()


def test_marker_text_in_comments_docstrings_is_not_code(tmp_path):
    """CFG assignment and norm-hook registration/removal mentioned ONLY in
    comments/docstrings must FAIL the code-level checks — even though the
    marker strings are all present — and a following clean run on the SAME
    loaded module must pass again (fresh state)."""
    mod = _load_checker()
    commented = PIPELINE.replace(
        "        handle = text_model.norm.register_forward_hook(\n"
        "            lambda module, args, output: args[0])\n",
        "        # handle = text_model.norm.register_forward_hook("
        "lambda module, args, output: args[0])\n")
    commented = commented.replace("            handle.remove()",
                                  "            pass  # handle.remove()")
    commented = commented.replace(
        "        has_neg_prompt = negative_prompt is not None or "
        "negative_prompt_embeds is not None\n"
        "        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt\n",
        '        """docs say: has_neg_prompt = negative_prompt is not None or\n'
        '        negative_prompt_embeds is not None; do_true_cfg = '
        'true_cfg_scale > 1 and has_neg_prompt"""\n')
    assert "has_neg_prompt = negative_prompt is not None" in commented  # text survives
    rc, out, err = _run(mod, tmp_path / "commented", pipeline=commented)
    assert rc == 1, out + err
    fail_lines = [l for l in out.splitlines() if l.startswith("[FAIL]")]
    for name in ("CFG gate: has_neg_prompt", "CFG gate: do_true_cfg",
                 "forward hook returning args[0]", "finally"):
        assert any(name in l for l in fail_lines), \
            f"{name} passed on comment-only evidence:\n{out}"
    rc2, out2, err2 = _run(mod, tmp_path / "clean")
    assert rc2 == 0, out2 + err2
    assert "[FAIL]" not in out2


def test_cfg_gate_rejects_extra_boolean_operands(tmp_path):
    """Both documented operands present is NOT enough: an extra constant
    (`... or True`, `... and False`) changes CFG semantics and must FAIL."""
    mod = _load_checker()
    cases = {
        "or_true": ("has_neg_prompt = negative_prompt is not None or "
                    "negative_prompt_embeds is not None",
                    "has_neg_prompt = negative_prompt is not None or "
                    "negative_prompt_embeds is not None or True",
                    "CFG gate: has_neg_prompt"),
        "and_false": ("do_true_cfg = true_cfg_scale > 1 and has_neg_prompt",
                      "do_true_cfg = true_cfg_scale > 1 and has_neg_prompt and False",
                      "CFG gate: do_true_cfg"),
        "and_dup": ("do_true_cfg = true_cfg_scale > 1 and has_neg_prompt",
                    "do_true_cfg = true_cfg_scale > 1 and has_neg_prompt and has_neg_prompt",
                    "CFG gate: do_true_cfg"),
    }
    for i, (label, (old, new, probe)) in enumerate(cases.items()):
        bad = PIPELINE.replace(old, new)
        assert bad != PIPELINE, label
        rc, out, err = _run(mod, tmp_path / label, pipeline=bad)
        assert rc == 1, f"{label}: rc={rc}\n{out}\n{err}"
        fail_lines = [l for l in out.splitlines() if l.startswith("[FAIL]")]
        assert any(probe in l for l in fail_lines), \
            f"{label}: {probe} passed despite extra operand:\n{out}"


def test_cfg_gate_rejects_later_overwrite(tmp_path):
    """A next()-style scan finds the documented FIRST assignment and would
    let a later reassignment of the same target evade the gate. The contract
    is a UNIQUE assignment to has_neg_prompt / do_true_cfg in __call__; any
    extra assignment or overwrite (even inside a branch) must FAIL. Shape
    tripwire only — the checker proves nothing about which line a running
    interpreter executes."""
    mod = _load_checker()
    cases = {
        "overwrite_has_neg_prompt": (
            "        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt\n",
            "        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt\n"
            "        if negative_prompt is not None:\n"
            "            has_neg_prompt = negative_prompt is not None\n",
            "CFG gate: has_neg_prompt"),
        "overwrite_do_true_cfg": (
            "        cache_enabled = use_kv_cache and "
            "self.transformer.config.causal_condition\n",
            "        do_true_cfg = do_true_cfg and use_kv_cache\n"
            "        cache_enabled = use_kv_cache and "
            "self.transformer.config.causal_condition\n",
            "CFG gate: do_true_cfg"),
    }
    for label, (old, new, probe) in cases.items():
        bad = PIPELINE.replace(old, new)
        assert bad != PIPELINE, label
        rc, out, err = _run(mod, tmp_path / label, pipeline=bad)
        assert rc == 1, f"{label}: rc={rc}\n{out}\n{err}"
        fail_lines = [l for l in out.splitlines() if l.startswith("[FAIL]")]
        assert any(probe in l for l in fail_lines), \
            f"{label}: {probe} passed despite a second assignment:\n{out}"


def test_valueerror_on_parse_is_clean_failure_exit1(tmp_path, monkeypatch):
    """parse_source must survive a bare ValueError from ast.parse (no
    .msg/.lineno on ValueError) without AttributeError. On modern CPython
    the null-byte input is reported as SyntaxError (lineno=None), so the
    ValueError path is forced by patching ast.parse on the loaded checker."""
    rc, out, err = _run(_load_checker(), tmp_path,
                        pipeline="class QwenImage21Pipeline:\n    x = \x00\n")
    assert rc == 1, out + err
    assert "source parses" in out
    assert "Traceback" not in err and "AttributeError" not in err, err

    mod = _load_checker()

    def boom(src, *a, **k):
        raise ValueError("synthetic: no msg/lineno attributes here")

    monkeypatch.setattr(mod.ast, "parse", boom)
    rep = mod.Report()
    mod.check_pipeline(PIPELINE, rep)  # must not raise AttributeError
    fails = [r for r in rep.rows if r[0] == "FAIL" and "source parses" in r[2]]
    assert len(fails) == 1, rep.rows
    assert "ValueError" in fails[0][3], fails[0]


def test_norm_hook_must_live_in_prompt_encoding_method(tmp_path):
    """Hook registration + finally-remove parked in a DIFFERENT method of
    the same class is not the surveyed contract — scoped checks must FAIL."""
    mod = _load_checker()
    moved = PIPELINE.replace(
        "    def _get_qwen_prompt_embeds(self, prompt, device):\n",
        "    def _get_qwen_prompt_embeds(self, prompt, device):\n"
        "        return text_model(input_ids=input_ids)\n"
        "\n"
        "    def _dead_code(self):\n")
    assert moved != PIPELINE
    rc, out, err = _run(mod, tmp_path / "moved", pipeline=moved)
    assert rc == 1, out + err
    fail_lines = [l for l in out.splitlines() if l.startswith("[FAIL]")]
    for name in ("forward hook returning args[0]", "finally"):
        assert any(name in l for l in fail_lines), \
            f"{name} passed outside _get_qwen_prompt_embeds:\n{out}"


def test_encoder_forward_must_be_inside_the_hook_protected_try(tmp_path):
    """Registration + a removal in SOME finally is not the surveyed
    relationship: the text-encoder forward must run inside the try whose
    finally removes that same registered handle. Two negatives:
      (a) forward moved OUT of the protected try (the finally now guards
          unrelated work only) — registration still passes, the anchored
          forward check must FAIL;
      (b) the finally removes a DIFFERENT name than the one registered —
          the forward is inside a try, but not one that removes THE handle.
    Static shape only: this is not proof of execution or exception safety."""
    mod = _load_checker()

    forward_out = PIPELINE.replace(
        "        try:\n"
        "            outs = text_model(input_ids=input_ids)\n"
        "        finally:\n"
        "            handle.remove()\n",
        "        outs = text_model(input_ids=input_ids)\n"
        "        try:\n"
        "            pass  # unrelated work is what the finally now guards\n"
        "        finally:\n"
        "            handle.remove()\n")
    assert forward_out != PIPELINE

    other_handle = PIPELINE.replace("            handle.remove()",
                                    "            other_handle.remove()")
    assert other_handle != PIPELINE

    for label, bad in (("forward_outside_try", forward_out),
                       ("removes_different_name", other_handle)):
        rc, out, err = _run(mod, tmp_path / label, pipeline=bad)
        assert rc == 1, f"{label}: rc={rc}\n{out}\n{err}"
        fail_lines = [l for l in out.splitlines() if l.startswith("[FAIL]")]
        assert any("encoder forward" in l and "finally" in l
                   for l in fail_lines), \
            f"{label}: anchored forward/finally check passed:\n{out}"
        assert not any("forward hook returning args[0]" in l
                       for l in fail_lines), \
            f"{label}: registration wrongly failed:\n{out}"


def test_method_calls_are_not_the_forward_in_the_protected_try(tmp_path):
    """The anchored check matches the DIRECT forward call only
    (self.text_encoder(...) surveyed / bare text_model(...) fixture).
    Two lead-repro negatives:
      (a) the protected try contains ONLY self.text_encoder.eval() — a
          method call whose target merely CONTAINS the encoder's name —
          and the real self.text_encoder(...) forward sits AFTER the
          try/finally. A broad dotted-segment matcher wrongly called this
          a forward; the exact-call predicate must FAIL it;
      (b) the protected try contains only method calls around the hook
          (register_forward_hook / norm on text_model), direct forward
          outside. Registration still passes; the anchored forward check
          must FAIL. Static shape only."""
    mod = _load_checker()
    old_try = ("        try:\n"
               "            outs = text_model(input_ids=input_ids)\n"
               "        finally:\n"
               "            handle.remove()\n")

    eval_only = PIPELINE.replace(
        old_try,
        "        try:\n"
        "            self.text_encoder.eval()  # method call, not a forward\n"
        "        finally:\n"
        "            handle.remove()\n"
        "        outs = self.text_encoder(**forward_kwargs)\n"
        "        # ^ real forward AFTER the try/finally: unprotected\n")
    assert eval_only != PIPELINE

    method_only = PIPELINE.replace(
        old_try,
        "        try:\n"
        "            text_model.norm.register_forward_hook(\n"
        "                lambda module, args, output: args[0])\n"
        "            text_model.norm(hidden_states)\n"
        "        finally:\n"
        "            handle.remove()\n"
        "        outs = text_model(input_ids=input_ids)\n"
        "        # ^ direct forward AFTER the try/finally: unprotected\n")
    assert method_only != PIPELINE

    for label, bad in (("eval_only_try_forward_outside", eval_only),
                       ("method_calls_only_in_try", method_only)):
        rc, out, err = _run(mod, tmp_path / label, pipeline=bad)
        assert rc == 1, f"{label}: rc={rc}\n{out}\n{err}"
        fail_lines = [l for l in out.splitlines() if l.startswith("[FAIL]")]
        assert any("encoder forward" in l and "finally" in l
                   for l in fail_lines), \
            f"{label}: method call satisfied the anchored forward check:\n{out}"
        assert not any("forward hook returning args[0]" in l
                       for l in fail_lines), \
            f"{label}: registration wrongly failed:\n{out}"


def test_direct_encoder_forward_classifier_exact_calls():
    """Unit negatives/positives for the exact-call classifier used by the
    anchored try check: only the surveyed DIRECT call targets are
    forwards; attribute/method calls and dotted lookalikes are not."""
    import ast

    mod = _load_checker()

    def _call(src):
        return next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call))

    for positive in ("outputs = self.text_encoder(**forward_kwargs)",
                     "outs = text_model(input_ids=input_ids)",
                     "return self.text_encoder(input_ids, output_hidden_states=True)"):
        assert mod._is_direct_encoder_forward(_call(positive)), positive
    for negative in ("self.text_encoder.eval()",
                     "text_model.norm.register_forward_hook(f)",
                     "text_model.norm(hidden_states)",
                     "self.text_encoder.norm(hidden_states)",
                     "self.text_model(input_ids=input_ids)",
                     "helper.text_model(input_ids=input_ids)",
                     "encoder(input_ids=input_ids)",
                     "self.text_encoder[0](input_ids)",
                     "get_encoder()(input_ids)"):
        assert not mod._is_direct_encoder_forward(_call(negative)), negative


# --- 2. optional external pinned-tree check (explicit skip when env absent) -

@pytest.mark.skipif(
    not (_diffusers and _transformers),
    reason="external pinned source trees NOT provided (QWEN_DIFFUSERS_SRC / "
           "QWEN_TRANSFORMERS_SRC unset): the pinned-interface contract was not "
           "checked on this run — this skip is not a pass (see docs/interfaces.md; "
           "tree revisions also need separate pin verification)",
)
def test_external_pinned_trees(capsys):
    mod = _load_checker()
    rc = mod.main(["--diffusers-src", _diffusers, "--transformers-src", _transformers])
    out = capsys.readouterr().out
    assert rc == 0, f"pinned-interface drift or input problem (exit {rc}):\n{out}"
    assert "NOT RUN" not in out, f"source trees missing/invalid:\n{out}"
