"""The 2K FP16 long-prompt warning, EXECUTED in node over the real app.js.

2K at FP16 fits only short prompts (measured 2026-10-27 on the V100 at
2752x1536: 500 prompt tokens fit, 600 ran out of memory; Q8 fit 1,500). The
page warns before a job is sent, from the prompt's character count.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "qwen_engine" / "web_static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _js_function(js: str, name: str) -> str:
    m = re.search(rf"function {re.escape(name)}\([\s\S]*?\n  \}}", js)
    assert m, f"{name} not found in app.js"
    return m.group(0)


def _js_var(js: str, name: str) -> str:
    m = re.search(rf"^  var {re.escape(name)} = [^;]*;", js, re.M)
    assert m, f"{name} not found in app.js"
    return m.group(0)


def _run(tmp_path: Path, cases: list[dict]) -> list[dict]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    js = _read("app.js")
    script = "\n".join([
        "var NODES = {};",
        "function $(id) { return NODES[id]; }",
        "function setText(id, text) { NODES[id].textContent = text; }",
        _js_function(js, "pyStrip"),
        _js_var(js, "MEMORY_WARNING_PIXELS"),
        _js_var(js, "MEMORY_WARNING_PROMPT_CHARS"),
        _js_function(js, "syncMemoryWarning"),
        "var out = [];",
        f"JSON.parse({json.dumps(json.dumps(cases))}).forEach(function (c) {{",
        "  NODES = { 'memory-warning': { hidden: true, textContent: '' },",
        "            width: { value: c.width }, height: { value: c.height },",
        "            prompt: { value: c.prompt }, quant: { value: c.quant } };",
        "  syncMemoryWarning();",
        "  out.push({ hidden: NODES['memory-warning'].hidden,",
        "             text: NODES['memory-warning'].textContent });",
        "});",
        "process.stdout.write(JSON.stringify(out));",
    ])
    path = tmp_path / "memory_warning.js"
    path.write_text(script, encoding="utf-8")
    done = subprocess.run([node, str(path)], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


LONG = "x" * 1601
LIMIT = "x" * 1600


def test_the_warning_shows_only_for_2k_fp16_with_a_long_prompt(tmp_path):
    cases = [
        {"width": "2752", "height": "1536", "quant": "fp16", "prompt": LONG},
        {"width": "1536", "height": "2752", "quant": "fp16", "prompt": LONG},
        {"width": "2048", "height": "2048", "quant": "fp16", "prompt": LONG},
        {"width": "2752", "height": "1536", "quant": "fp16", "prompt": LIMIT},
        {"width": "2752", "height": "1536", "quant": "q8", "prompt": LONG},
        {"width": "1024", "height": "1024", "quant": "fp16", "prompt": LONG},
        {"width": "1344", "height": "768", "quant": "fp16", "prompt": LONG},
        {"width": "", "height": "1536", "quant": "fp16", "prompt": LONG},
        {"width": "2752", "height": "1536", "quant": "fp16", "prompt": "  " + LIMIT + "  "},
    ]
    shown = [not item["hidden"] for item in _run(tmp_path, cases)]
    assert shown == [True, True, True, False, False, False, False, False, False]


def test_the_warning_names_the_count_and_both_ways_out(tmp_path):
    case = {"width": "2752", "height": "1536", "quant": "fp16", "prompt": LONG}
    text = _run(tmp_path, [case])[0]["text"]
    assert "1601 characters" in text and "Q8" in text and "1,600 characters" in text
    cleared = _run(tmp_path, [dict(case, quant="q8")])[0]["text"]
    assert cleared == ""


def test_the_warning_is_refreshed_on_every_form_change_and_sits_above_generate():
    js = _read("app.js")
    assert "syncMemoryWarning();" in _js_function(js, "renderValidation")
    html = _read("index.html")
    warning = html.index('id="memory-warning"')
    assert warning < html.index('id="generate-btn"')
    tag = html[html.rindex("<", 0, warning):html.index(">", warning)]
    assert "hidden" in tag and 'role="status"' in tag
