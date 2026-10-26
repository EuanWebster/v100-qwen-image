#!/usr/bin/env python3
"""Differential trace for the 1024x1024 reference-edit artifact (q3 task 2).

Runs the normal CLI in-process (pattern: ``output/gpu/kvoff_run.py``) with
read-only wrappers installed on the pinned pipeline and transformer, recording
every layout/schedule value the root wants to diff between a 1024x1024
(artifact) and a 1024x1088 (clean) Turbo edit run. Nothing is modified in
``src/`` or in diffusers: every wrapper calls through to the original and only
observes. Output is one compact JSON object (no prompts) written to
``--trace-out`` even when the run fails.

Usage (from the repo root, worktree ``src`` first on the path)::

    PYTHONPATH=src python scripts/trace_edit_inputs.py --trace-out trace-1024.json \
        <normal CLI args...>

Field names are stable and the dump is ``sort_keys`` so two traces diff well.

The only copied logic is :func:`rope_index_lists`, a verbatim transcription of
``QwenImage21Rope.forward``'s index-construction loop
(``diffusers/models/transformers/transformer_qwenimage21.py:677-709``, pinned
0.41.0.dev0); it only summarises the tensors the rope module computes from its
arguments, so it cannot change behaviour. If that file changes, re-transcribe
before trusting ``rope_forward[].index_summary``.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import math
import sys
from collections import Counter
from typing import Any

TRACE_FLAG = "--trace-out"
REDACT_FLAGS = frozenset({"--prompt", "--negative"})

# (import path, attribute) pairs install_hooks() wraps. Kept as data so the
# CPU plan-mode test can pin the surface without importing torch/diffusers.
HOOK_TARGETS: tuple[tuple[str, str], ...] = (
    ("diffusers.pipelines.qwenimage21.pipeline_qwenimage21", "QwenImage21Pipeline.__call__"),
    ("diffusers.pipelines.qwenimage21.pipeline_qwenimage21", "QwenImage21Pipeline.encode_prompt"),
    ("diffusers.pipelines.qwenimage21.pipeline_qwenimage21", "QwenImage21Pipeline.prepare_latents"),
    ("diffusers.pipelines.qwenimage21.pipeline_qwenimage21", "retrieve_timesteps"),
    ("diffusers.pipelines.qwenimage21.pipeline_qwenimage21", "calculate_shift"),
    ("diffusers.models.transformers.transformer_qwenimage21", "QwenImage21Transformer2DModel.forward"),
    ("diffusers.models.transformers.transformer_qwenimage21", "QwenImage21Transformer2DModel.build_token_metadata"),
    ("diffusers.models.transformers.transformer_qwenimage21", "QwenImage21Rope.forward"),
    ("diffusers.models.transformers.transformer_qwenimage21", "_qwenimage21_prefix_segments"),
    ("diffusers.image_processor", "VaeImageProcessor.resize"),
)

#: Attribute labels the wrappers add to the trace document (pinned by tests).
TRACE_SECTIONS: tuple[str, ...] = (
    "pipeline_call",
    "encode_prompt",
    "prepare_latents",
    "timestep_setup.retrieve",
    "timestep_setup.calculate_shift",
    "image_resize",
    "transformer_forward",
    "token_metadata",
    "rope_forward",
    "attention_segments",
    "denoise_steps",
)


def plan() -> tuple[tuple[str, str], ...]:
    """Return the hook surface (pure; used by the CPU plan-mode test)."""
    return HOOK_TARGETS


def redact_argv(argv: list[str]) -> list[str]:
    """Mask prompt values so the trace never contains the prompt."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            out.append("<redacted>")
            continue
        flag, sep, _value = token.partition("=")
        if flag in REDACT_FLAGS and sep:
            out.append(flag + "=<redacted>")
            continue
        out.append(token)
        skip = token in REDACT_FLAGS
    return out


# ---------------------------------------------------------------------------
# Pure summarisers (unit-tested on CPU with plain lists)
# ---------------------------------------------------------------------------


def rope_index_lists(
    img_shapes: list[tuple[int, int, int]], is_image_token: list[bool]
) -> tuple[list[int], list[int], list[int]]:
    """Verbatim copy of the index loop in ``QwenImage21Rope.forward`` (:677-709).

    Returns the (frame, height, width) position id lists over the joint
    sequence; the source scatters the grid rows onto clones of the frame list
    at ``image_pad_mask`` positions, mirrored here on the index lists.
    """
    frame_index: list[int] = []
    image_height_index: list[int] = []
    image_width_index: list[int] = []
    cursor, position = 0, 0
    total_len = len(is_image_token)

    for _, height, width in img_shapes:
        block_start = is_image_token.index(True, cursor)
        text_len = block_start - cursor
        frame_index.extend(range(position, position + text_len))
        position += text_len

        cursor = block_start + height * width
        frame_index.extend([position] * (height * width))
        position += max(height, width)

        image_height_index.extend(
            [h for h in range(-(height - height // 2), height // 2) for _ in range(width)]
        )
        image_width_index.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])

    if cursor < total_len:
        frame_index.extend(range(position, position + total_len - cursor))

    height_index = list(frame_index)
    width_index = list(frame_index)
    image_positions = [i for i, v in enumerate(is_image_token) if v]
    if len(image_positions) != len(image_height_index):
        raise ValueError(
            f"mask marks {len(image_positions)} image tokens but img_shapes describes "
            f"{len(image_height_index)}"
        )
    for pos, h, w in zip(image_positions, image_height_index, image_width_index):
        height_index[pos] = h
        width_index[pos] = w
    return frame_index, height_index, width_index


def _minmax_unique(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    return {"min": min(values), "max": max(values), "unique": len(set(values))}


def summarize_rope_indices(
    img_shapes: list[tuple[int, int, int]], is_image_token: list[bool]
) -> dict[str, Any]:
    """Per-block (frame, height, width) id summaries over the joint sequence."""
    frame, height, width = rope_index_lists(img_shapes, is_image_token)
    image_positions = [i for i, v in enumerate(is_image_token) if v]
    blocks: list[dict[str, Any]] = []
    offset = 0
    for block_index, (_, h, w) in enumerate(img_shapes):
        count = h * w
        positions = image_positions[offset : offset + count]
        offset += count
        blocks.append(
            {
                "block": block_index,
                "shape": [1, h, w],
                "tokens": count,
                "frame": _minmax_unique([frame[i] for i in positions]),
                "height": _minmax_unique([height[i] for i in positions]),
                "width": _minmax_unique([width[i] for i in positions]),
            }
        )
    text_positions = [i for i, v in enumerate(is_image_token) if not v]
    return {
        "total_len": len(is_image_token),
        "image_tokens": len(image_positions),
        "text_tokens": len(text_positions),
        "text_frame": _minmax_unique([frame[i] for i in text_positions]),
        "blocks": blocks,
    }


def summarize_token_metadata(ids: list[int], target_mask: list[bool]) -> dict[str, Any]:
    """Summarise ``build_token_metadata`` outputs (id -> token counts, target block)."""
    counts = Counter(ids)
    text_count = counts.pop(-1, 0)
    target_ids = sorted({ids[i] for i, m in enumerate(target_mask) if m})
    target_positions = [i for i, m in enumerate(target_mask) if m]
    return {
        "seq_len": len(ids),
        "text_tokens": text_count,
        "blocks": [{"id": block_id, "tokens": tokens} for block_id, tokens in sorted(counts.items())],
        "target_block_ids": target_ids,
        "target_tokens": len(target_positions),
        "target_positions": [min(target_positions), max(target_positions)] if target_positions else [],
    }


def summarize_segments(segments: list[tuple[int, int, bool]]) -> list[list[Any]]:
    return [[start, end, bool(is_text)] for start, end, is_text in segments]


# ---------------------------------------------------------------------------
# Tensor helpers (torch duck-typed; module never imports torch itself)
# ---------------------------------------------------------------------------


def tensor_stats(t: Any) -> dict[str, Any] | None:
    if t is None:
        return None
    try:
        f = t.detach().float()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "mean": float(f.mean()),
            "std": float(f.std()),
            "absmax": float(f.abs().amax()),
            "nonfinite": int((~f.isfinite()).sum()),
        }
    except Exception as exc:  # observation must never break the run
        return {"error": f"{type(exc).__name__}: {exc}"}


def mask_info(mask: Any) -> dict[str, Any] | None:
    if mask is None:
        return None
    try:
        m = mask.bool()
        row0 = m.reshape(-1, m.shape[-1])[0] if m.ndim > 1 else m
        return {
            "shape": list(mask.shape),
            "len": int(m.shape[-1]),
            "true_total": int(m.sum()),
            "true_row0": int(row0.sum()),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def tensor_shape(t: Any) -> Any:
    if t is None or not hasattr(t, "shape"):
        return None
    return {"shape": list(t.shape), "dtype": str(t.dtype)}


def tensor_list(t: Any) -> Any:
    if t is None or not hasattr(t, "tolist"):
        return None
    try:
        return t.detach().float().tolist()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Trace document
# ---------------------------------------------------------------------------


class TraceDoc:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def add(self, section: str, record: dict[str, Any]) -> None:
        self.data.setdefault(section, []).append(record)

    def note_failure(self, where: str, exc: BaseException) -> None:
        self.add("hook_failures", {"where": where, "error": f"{type(exc).__name__}: {exc}"})

    def dump(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(self.data, handle, sort_keys=True, indent=1)


# ---------------------------------------------------------------------------
# Hook installation (imports torch/diffusers; call only in a GPU environment)
# ---------------------------------------------------------------------------


def _wrap(doc: TraceDoc, owner: Any, attr: str, label: str, recorder: Any) -> str:
    """Replace owner.attr with a call-through observer; return its label.

    ``recorder(doc, args, kwargs, original)`` calls ``original`` itself and
    never raises after doing so, so a recording failure can never double-run an
    observed function (``prepare_latents`` reseeds the sampler on re-entry).
    """
    static = isinstance(inspect.getattr_static(owner, attr), staticmethod)
    original = getattr(owner, attr)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return recorder(doc, args, kwargs, original)

    wrapper.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(owner, attr, staticmethod(wrapper) if static else wrapper)
    return label


def _call_recorder(doc: TraceDoc, section: str, project: Any):
    """Recorder that calls the original, then records ``project(args, kwargs, result)``."""

    def recorder(_doc: TraceDoc, args: Any, kwargs: Any, original: Any) -> Any:
        result = original(*args, **kwargs)
        try:
            record = project(args, kwargs, result)
            if record is not None:
                doc.add(section, record)
        except BaseException as exc:
            doc.note_failure(section, exc)
        return result

    return recorder


def install_hooks(doc: TraceDoc) -> list[str]:
    """Install every wrapper; returns the labels that were successfully bound."""
    installed: list[str] = []
    seq = {"call": 0}

    # -- pipeline module -----------------------------------------------------
    try:
        pipe_mod = importlib.import_module("diffusers.pipelines.qwenimage21.pipeline_qwenimage21")
    except BaseException as exc:
        doc.note_failure("import:pipeline", exc)
        pipe_mod = None
    if pipe_mod is not None:
        pipeline_cls = pipe_mod.QwenImage21Pipeline
        calculate_dimensions = pipe_mod.calculate_dimensions

        def project_call(args: tuple, kwargs: dict, result: Any) -> dict[str, Any]:
            self = args[0]
            kw = dict(kwargs)
            if len(args) > 1:
                kw.setdefault("prompt", args[1])  # only observed caller passes keywords
            images = kw.get("image")
            resolution = kw.get("output_resolution")
            if resolution is None:
                resolution = 1024  # pipeline default (:533); absent kwarg recorded separately
            sizes = None
            recomputed = None
            if images is not None:
                sizes = [list(img.size) if hasattr(img, "size") else tensor_shape(img) for img in images]
                recomputed = []
                for img in images:
                    if hasattr(img, "size"):
                        w, h = img.size
                        iw, ih, _ = calculate_dimensions(resolution * resolution, w / h)
                        recomputed.append([iw, ih])
            pos = kw.get("prompt_embeds")
            neg = kw.get("negative_prompt_embeds")
            scheduler_cfg = {
                key: value
                for key, value in dict(self.scheduler.config).items()
                if not key.startswith("_") and isinstance(value, (int, float, str, bool, list, type(None)))
            }
            return {
                "height": kw.get("height"),
                "width": kw.get("width"),
                "num_inference_steps": kw.get("num_inference_steps"),
                "sigmas_kwarg": kw.get("sigmas"),
                "output_resolution": kw.get("output_resolution", "<absent: pipeline default 1024>"),
                "true_cfg_scale": kw.get("true_cfg_scale"),
                "do_true_cfg": bool(kw.get("true_cfg_scale", 1) > 1)
                and (kw.get("negative_prompt") is not None or neg is not None),
                "use_kv_cache": kw.get("use_kv_cache"),
                "prompt_is_none": kw.get("prompt") is None,
                "num_images_per_prompt": kw.get("num_images_per_prompt"),
                "output_type": kw.get("output_type"),
                "image_count": None if images is None else len(images),
                "image_sizes": sizes,
                "input_image_sizes_recomputed": recomputed,
                "prompt_embeds": tensor_shape(pos),
                "prompt_embeds_mask": mask_info(kw.get("prompt_embeds_mask")),
                "negative_prompt_embeds": tensor_shape(neg),
                "vae_scale_factor": int(self.vae_scale_factor),
                "scheduler_class": type(self.scheduler).__name__,
                "scheduler_config": scheduler_cfg,
                "pipeline_sample_sigmas": getattr(self.config, "sample_sigmas", None),
                "transformer_causal_condition": bool(self.transformer.config.causal_condition),
                "attention_processors": sorted(
                    {type(block.attn.processor).__name__ for block in self.transformer.transformer_blocks}
                ),
            }

        def project_encode(args: tuple, kwargs: dict, result: Any) -> dict[str, Any]:
            embeds, embeds_mask, pad_mask = result
            return {
                "image_passed": kwargs.get("image") is not None,
                "image_count": None if kwargs.get("image") is None else len(kwargs["image"]),
                "prompt_is_none": kwargs.get("prompt") is None,
                "prompt_embeds_in": tensor_shape(kwargs.get("prompt_embeds")),
                "image_pad_mask_in": mask_info(kwargs.get("image_pad_mask")),
                "embeds": tensor_shape(embeds),
                "embeds_mask": mask_info(embeds_mask),
                "image_pad_mask_out": mask_info(pad_mask),
            }

        def project_prepare(args: tuple, kwargs: dict, result: Any) -> dict[str, Any]:
            self, vae_images = args[0], args[1]
            latents, image_latents = result
            gen = args[8] if len(args) > 8 else kwargs.get("generator")
            return {
                "height": args[4] if len(args) > 4 else kwargs.get("height"),
                "width": args[5] if len(args) > 5 else kwargs.get("width"),
                "batch": args[2] if len(args) > 2 else kwargs.get("batch_size"),
                "num_channels_latents": args[3] if len(args) > 3 else kwargs.get("num_channels_latents"),
                "dtype": str(args[6]) if len(args) > 6 else str(kwargs.get("dtype")),
                "vae_images": [tensor_shape(v) for v in vae_images] if vae_images is not None else None,
                "generator_seed": None if gen is None else int(gen.initial_seed()),
                "latents": tensor_stats(latents),
                "image_latents": tensor_stats(image_latents),
                "vae_scale_factor": int(self.vae_scale_factor),
            }

        installed.append(
            _wrap(doc, pipeline_cls, "__call__", "pipeline.__call__", _call_recorder(doc, "pipeline_call", project_call))
        )
        installed.append(
            _wrap(
                doc, pipeline_cls, "encode_prompt", "pipeline.encode_prompt",
                _call_recorder(doc, "encode_prompt", project_encode),
            )
        )
        installed.append(
            _wrap(
                doc, pipeline_cls, "prepare_latents", "pipeline.prepare_latents",
                _call_recorder(doc, "prepare_latents", project_prepare),
            )
        )

        def project_retrieve(args: tuple, kwargs: dict, result: Any) -> dict[str, Any]:
            scheduler = args[0]
            timesteps, steps = result
            return {
                "scheduler_class": type(scheduler).__name__,
                "num_inference_steps_arg": args[1] if len(args) > 1 else kwargs.get("num_inference_steps"),
                "sigmas": kwargs.get("sigmas"),
                "mu": kwargs.get("mu"),
                "timesteps": tensor_list(timesteps),
                "num_inference_steps_returned": steps,
            }

        def project_shift(args: tuple, kwargs: dict, result: Any) -> dict[str, Any]:
            return {
                "image_seq_len": args[0] if args else kwargs.get("image_seq_len"),
                "base_image_seq_len": args[1] if len(args) > 1 else kwargs.get("base_image_seq_len"),
                "max_image_seq_len": args[2] if len(args) > 2 else kwargs.get("max_image_seq_len"),
                "base_shift": args[3] if len(args) > 3 else kwargs.get("base_shift"),
                "max_shift": args[4] if len(args) > 4 else kwargs.get("max_shift"),
                "mu": float(result),
            }

        installed.append(
            _wrap(doc, pipe_mod, "retrieve_timesteps", "retrieve_timesteps",
                  _call_recorder(doc, "timestep_setup.retrieve", project_retrieve))
        )
        installed.append(
            _wrap(doc, pipe_mod, "calculate_shift", "calculate_shift",
                  _call_recorder(doc, "timestep_setup.calculate_shift", project_shift))
        )

    # -- image processor ------------------------------------------------------
    try:
        ip_mod = importlib.import_module("diffusers.image_processor")

        def project_resize(args: tuple, kwargs: dict, result: Any) -> dict[str, Any]:
            images = args[1] if len(args) > 1 else kwargs.get("images")

            def sizes(obj: Any) -> Any:
                if hasattr(obj, "size"):  # single PIL image (pipeline path)
                    return [list(obj.size)]
                try:
                    return [list(img.size) for img in obj]
                except Exception:
                    return tensor_shape(obj)

            return {
                "input_sizes": sizes(images),
                "requested_height": kwargs.get("height"),
                "requested_width": kwargs.get("width"),
                "output_sizes": sizes(result),
            }

        installed.append(
            _wrap(doc, ip_mod.VaeImageProcessor, "resize", "image_processor.resize",
                  _call_recorder(doc, "image_resize", project_resize))
        )
    except BaseException as exc:
        doc.note_failure("import:image_processor", exc)

    # -- transformer module ----------------------------------------------------
    try:
        tr_mod = importlib.import_module("diffusers.models.transformers.transformer_qwenimage21")
    except BaseException as exc:
        doc.note_failure("import:transformer", exc)
        tr_mod = None
    if tr_mod is not None:
        model_cls = tr_mod.QwenImage21Transformer2DModel

        def project_forward(args: tuple, kwargs: dict, result: Any) -> dict[str, Any]:
            hidden = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
            ehs = kwargs.get("encoder_hidden_states", args[2] if len(args) > 2 else None)
            timestep = kwargs.get("timestep", args[3] if len(args) > 3 else None)
            img_shapes = kwargs.get("img_shapes", args[4] if len(args) > 4 else None)
            img_mask = kwargs.get("img_mask", args[5] if len(args) > 5 else None)
            seq["call"] += 1
            target_tokens = None
            try:
                target_tokens = math.prod(img_shapes[0][-1])
            except Exception:
                pass
            out = result[0] if isinstance(result, tuple) else getattr(result, "sample", result)
            noise_target = None
            if target_tokens and hasattr(out, "shape") and out.ndim == 3:
                noise_target = tensor_stats(out[:, -target_tokens:, :])
            return {
                "call_index": seq["call"],
                "branch": "cond" if seq["call"] % 2 == 1 else "uncond",
                # parity is only a guess (CFG alternates cond/uncond); the tensor
                # identity is the ground truth for single-branch (turbo) runs.
                "encoder_hidden_states_ptr": None if ehs is None else ehs.data_ptr(),
                "hidden_states": tensor_shape(hidden),
                "encoder_hidden_states": tensor_shape(ehs),
                "encoder_hidden_states_mask": mask_info(kwargs.get("encoder_hidden_states_mask")),
                "timestep": tensor_list(timestep),
                "img_shapes": [[list(shape) for shape in sample] for sample in img_shapes] if img_shapes else None,
                "img_mask": mask_info(img_mask),
                "kv_cache_mode": kwargs.get("kv_cache_mode"),
                "kv_cache_present": kwargs.get("kv_cache") is not None,
                "target_tokens": target_tokens,
                "noise_pred_full": tensor_stats(out),
                "noise_pred_target_slice": noise_target,
            }

        def project_metadata(args: tuple, kwargs: dict, result: Any) -> dict[str, Any]:
            ids, target_mask = result
            mask = args[0] if args else kwargs.get("image_pad_mask")
            shapes = args[1] if len(args) > 1 else kwargs.get("img_shapes")
            return {
                "image_pad_mask_in": mask_info(mask),
                "img_shapes": [list(shape) for shape in shapes],
                "image_ids": summarize_token_metadata(ids.tolist(), target_mask.bool().tolist()),
            }

        def project_rope(args: tuple, kwargs: dict, result: Any) -> dict[str, Any]:
            img_shapes = args[1] if len(args) > 1 else kwargs.get("img_shapes")
            mask = args[2] if len(args) > 2 else kwargs.get("image_pad_mask")
            mask_list = mask.bool().tolist()
            if mask.ndim != 1:
                mask_list = mask_list[0]
            return {
                "img_shapes": [list(shape) for shape in img_shapes],
                "image_pad_mask": mask_info(mask),
                "freqs_dtype": str(result.dtype),
                "freqs_shape": list(result.shape),
                "index_summary": summarize_rope_indices([tuple(shape) for shape in img_shapes], mask_list),
            }

        def project_segments(args: tuple, kwargs: dict, result: Any) -> dict[str, Any]:
            image_ids = args[0] if args else kwargs.get("image_ids")
            return {
                "call_index": seq["call"],
                "seq_len": int(image_ids.shape[0]),
                "prefix_len": args[1] if len(args) > 1 else kwargs.get("prefix_len"),
                "segments": summarize_segments(result),
            }

        installed.append(
            _wrap(doc, model_cls, "forward", "transformer.forward", _call_recorder(doc, "transformer_forward", project_forward))
        )
        installed.append(
            _wrap(doc, model_cls, "build_token_metadata", "transformer.build_token_metadata",
                  _call_recorder(doc, "token_metadata", project_metadata))
        )
        installed.append(
            _wrap(doc, tr_mod.QwenImage21Rope, "forward", "rope.forward", _call_recorder(doc, "rope_forward", project_rope))
        )
        installed.append(
            _wrap(doc, tr_mod, "_qwenimage21_prefix_segments", "prefix_segments",
                  _call_recorder(doc, "attention_segments", project_segments))
        )

    # -- per-step latents: intercept the bridge's callback_on_step_end ---------
    if pipe_mod is not None:
        wrapped_call = pipe_mod.QwenImage21Pipeline.__call__  # the recorder installed above

        @functools.wraps(wrapped_call)
        def step_capture_call(self: Any, *args: Any, **kwargs: Any) -> Any:
            callback = kwargs.get("callback_on_step_end")
            if callback is not None and not getattr(callback, "_q3_trace", False):

                @functools.wraps(callback)
                def trace_callback(pipe: Any, index: int, timestep: Any, tensors: dict, *cargs: Any, **ckw: Any) -> Any:
                    try:
                        doc.add(
                            "denoise_steps",
                            {
                                "step_index": int(index),
                                "timestep": tensor_list(timestep),
                                "latents": tensor_stats(tensors.get("latents")),
                                "tensor_keys": sorted(tensors),
                            },
                        )
                    except BaseException as exc:
                        doc.note_failure("step_callback", exc)
                    return callback(pipe, index, timestep, tensors, *cargs, **ckw)

                trace_callback._q3_trace = True  # type: ignore[attr-defined]
                kwargs["callback_on_step_end"] = trace_callback
            return wrapped_call(self, *args, **kwargs)

        pipe_mod.QwenImage21Pipeline.__call__ = step_capture_call
        installed.append("pipeline.callback_on_step_end")

    return installed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def extract_trace_out(argv: list[str]) -> tuple[str, list[str]]:
    path = None
    rest: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == TRACE_FLAG:
            if index + 1 >= len(argv):
                raise SystemExit(f"{TRACE_FLAG} requires a path")
            path = argv[index + 1]
            index += 2
            continue
        if token.startswith(TRACE_FLAG + "="):
            path = token.split("=", 1)[1]
            index += 1
            continue
        rest.append(token)
        index += 1
    if path is None:
        raise SystemExit(f"{TRACE_FLAG} PATH is required, then normal CLI args")
    return path, rest


def main(argv: list[str]) -> int:
    trace_path, cli_argv = extract_trace_out(argv)
    doc = TraceDoc()
    doc.data["argv"] = redact_argv(cli_argv)
    doc.data["hook_targets"] = [[mod, attr] for mod, attr in plan()]
    try:
        doc.data["hooks_installed"] = install_hooks(doc)
    except BaseException as exc:
        doc.note_failure("install_hooks", exc)
        doc.data["hooks_installed"] = []
    exit_code: Any = 0
    try:
        from qwen_engine.cli import main as cli_main

        exit_code = cli_main(cli_argv)
    except SystemExit as exc:  # argparse / cli exits
        exit_code = exc.code
    except BaseException as exc:
        doc.note_failure("cli", exc)
        exit_code = f"cli raised {type(exc).__name__}: {exc}"
        raise
    finally:
        doc.data["exit_code"] = exit_code if isinstance(exit_code, (int, str, type(None))) else str(exit_code)
        doc.data["record_counts"] = {key: len(value) for key, value in doc.data.items() if isinstance(value, list)}
        doc.dump(trace_path)
    return exit_code if isinstance(exit_code, int) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
