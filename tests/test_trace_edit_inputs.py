"""CPU plan-mode tests for scripts/trace_edit_inputs.py (no torch/diffusers)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trace_edit_inputs.py"
_spec = importlib.util.spec_from_file_location("trace_edit_inputs", SCRIPT)
trace = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(trace)


def test_plan_surface_is_stable():
    targets = trace.plan()
    assert len(targets) == len(set(targets)), "duplicate hook targets"
    attrs = {attr for _mod, attr in targets}
    assert attrs == {
        "QwenImage21Pipeline.__call__",
        "QwenImage21Pipeline.encode_prompt",
        "QwenImage21Pipeline.prepare_latents",
        "retrieve_timesteps",
        "calculate_shift",
        "QwenImage21Transformer2DModel.forward",
        "QwenImage21Transformer2DModel.build_token_metadata",
        "QwenImage21Rope.forward",
        "_qwenimage21_prefix_segments",
        "VaeImageProcessor.resize",
    }
    for mod, _attr in targets:
        assert mod.startswith("diffusers."), mod
    # one extra install step (the callback_on_step_end interception) over targets
    assert len(trace.TRACE_SECTIONS) == len(targets) + 1


def test_rope_index_lists_single_block():
    frame, height, width = trace.rope_index_lists([(1, 2, 2)], [False, True, True, True, True])
    assert frame == [0, 1, 1, 1, 1]
    assert height == [0, -1, -1, 0, 0]
    assert width == [0, -1, 0, -1, 0]


def test_rope_index_lists_two_blocks_and_trailing_text():
    # text | 2x2 block | text | 1x2 block | text
    mask = [False, True, True, True, True, False, True, True, False]
    frame, height, width = trace.rope_index_lists([(1, 2, 2), (1, 1, 2)], mask)
    # first block frame freezes at 1 (one text token before), position advances by max(2,2)
    # second text token gets frame 3; second block freezes at 4; trailing text continues.
    assert frame == [0, 1, 1, 1, 1, 3, 4, 4, 6]
    assert height == [0, -1, -1, 0, 0, 3, -1, -1, 6]
    assert width == [0, -1, 0, -1, 0, 3, -1, 0, 6]


def test_rope_index_lists_rejects_mask_shape_mismatch():
    try:
        trace.rope_index_lists([(1, 2, 2)], [False, True, True])
    except ValueError as exc:
        assert "img_shapes" in str(exc)
    else:
        raise AssertionError("mismatched mask/grid must raise ValueError")


def test_summarize_rope_indices_blocks():
    summary = trace.summarize_rope_indices([(1, 2, 2)], [False, True, True, True, True])
    assert summary["image_tokens"] == 4 and summary["text_tokens"] == 1
    assert summary["text_frame"] == {"min": 0, "max": 0, "unique": 1}
    block = summary["blocks"][0]
    assert block["shape"] == [1, 2, 2] and block["tokens"] == 4
    assert block["frame"] == {"min": 1, "max": 1, "unique": 1}
    assert block["height"] == {"min": -1, "max": 0, "unique": 2}
    assert block["width"] == {"min": -1, "max": 0, "unique": 2}


def test_summarize_token_metadata():
    ids = [-1, -1, -1, 0, 0, 1, 1, 1]
    target = [False] * 5 + [True] * 3
    out = trace.summarize_token_metadata(ids, target)
    assert out["seq_len"] == 8 and out["text_tokens"] == 3
    assert out["blocks"] == [{"id": 0, "tokens": 2}, {"id": 1, "tokens": 3}]
    assert out["target_block_ids"] == [1]
    assert out["target_tokens"] == 3 and out["target_positions"] == [5, 7]


def test_summarize_segments():
    assert trace.summarize_segments([(0, 3, True), (3, 5, 0)]) == [[0, 3, True], [3, 5, False]]


def test_redact_argv_masks_prompts():
    argv = ["--prompt", "a secret sentence", "--negative=neg", "--width", "1024"]
    assert trace.redact_argv(argv) == [
        "--prompt",
        "<redacted>",
        "--negative=<redacted>",
        "--width",
        "1024",
    ]


def test_extract_trace_out_splits_cli():
    path, rest = trace.extract_trace_out(["--trace-out", "/tmp/t.json", "--width", "1024"])
    assert path == "/tmp/t.json" and rest == ["--width", "1024"]
    path, rest = trace.extract_trace_out(["--trace-out=/tmp/u.json"])
    assert path == "/tmp/u.json" and rest == []
    for bad in (["--width", "1024"], ["--trace-out"]):
        try:
            trace.extract_trace_out(bad)
        except SystemExit:
            continue
        raise AssertionError(f"missing {trace.TRACE_FLAG} must exit: {bad}")


def test_trace_doc_dumps_sorted_single_json(tmp_path):
    doc = trace.TraceDoc()
    doc.add("zeta", {"b": 2, "a": 1})
    doc.add("alpha", {"z": [1, 2]})
    doc.add("zeta", {"c": 3})
    out = tmp_path / "trace.json"
    doc.dump(str(out))
    text = out.read_text()
    loaded = json.loads(text)  # exactly one JSON document
    assert list(loaded) == sorted(loaded)
    assert [list(rec) for rec in loaded["zeta"]] == [["a", "b"], ["c"]]
    assert loaded["zeta"][0] == {"a": 1, "b": 2}
    doc.note_failure("somewhere", RuntimeError("boom"))
    doc.dump(str(out))
    assert json.loads(out.read_text())["hook_failures"][0]["where"] == "somewhere"


def test_tensor_helpers_tolerate_foreign_objects():
    assert trace.tensor_stats(None) is None
    assert trace.mask_info(None) is None
    assert trace.tensor_shape(None) is None
    assert trace.tensor_list(object()) is None
    assert "error" in trace.tensor_stats(object())
    assert "error" in trace.mask_info(object())


def test_recorder_calls_through_and_survives_project_errors():
    doc = trace.TraceDoc()

    class Owner:
        calls = 0

        @staticmethod
        def static_target(value: int) -> int:
            Owner.calls += 1
            return value * 2

    def good(value: int) -> int:
        return Owner.static_target(value)

    label = trace._wrap(
        doc, Owner, "static_target", "static",
        trace._call_recorder(doc, "seen", lambda args, kwargs, result: {"result": result}),
    )
    assert label == "static" and Owner.static_target(21) == 42
    assert doc.data["seen"] == [{"result": 42}]

    # a failing projection must not suppress the result nor re-run the target
    trace._wrap(
        doc, Owner, "static_target", "static2",
        trace._call_recorder(doc, "bad", lambda *_a: (_ for _ in ()).throw(RuntimeError("project boom"))),
    )
    assert Owner.static_target(5) == 10 and Owner.calls == 2
    failures = doc.data["hook_failures"]
    assert failures[-1]["where"] == "bad" and "project boom" in failures[-1]["error"]
