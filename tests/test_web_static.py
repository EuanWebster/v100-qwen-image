"""W1 guarded tests for the packaged static web workspace assets.

Pure text/structure checks on ``src/qwen_engine/web_static/`` — no browser,
no network, no ML imports, no writes. The assets must stay original,
self-contained and honest-by-construction (PLAN W-milestone pivot,
docs/web-interface.md). W1 performs NO network requests at all: the earlier
speculative capability probe was removed after review, and these tests assert
it stays removed. Real-Chrome behavioural checks live in
``scripts/chrome_browser_check.mjs`` — that script is NOT part of this guarded
suite and requires ``CHROME_PATH``.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import qwen_engine
from qwen_engine.config import (
    AttentionBackend,
    GenerationConfig,
    OffloadMode,
    Quant,
)

WEB_DIR = Path(qwen_engine.__file__).resolve().parent / "web_static"
ASSETS = ("index.html", "app.css", "app.js")

# Nothing in the product files may name an external origin (fonts, CDN,
# analytics) — the page is local-only.
EXTERNAL_ORIGIN = re.compile(r"(?:src|href)\s*=\s*[\"']https?://|\bhttps?://[a-z0-9.-]+\.[a-z]{2,}", re.I)


def _read(name: str) -> str:
    path = WEB_DIR / name
    assert path.is_file(), f"missing packaged static asset: {path}"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"static asset is empty: {path}"
    return text


def test_assets_exist_and_are_packaged_under_the_package():
    for name in ASSETS:
        text = _read(name)
        assert WEB_DIR.is_relative_to(Path(qwen_engine.__file__).resolve().parent)
        assert text  # non-empty (checked in _read)


def test_no_external_origins_or_subresource_integrity_risk():
    for name in ASSETS:
        match = EXTERNAL_ORIGIN.search(_read(name))
        assert match is None, f"{name} references an external origin: {match}"


def test_html_wires_only_local_assets():
    html = _read("index.html")
    assert 'href="app.css"' in html and 'src="app.js"' in html
    assert "<form" in html and 'id="generate-btn"' in html


def test_js_never_uses_innnerhtml_and_keeps_seed_a_decimal_string():
    js = _read("app.js")
    assert "innerHTML" not in js
    assert "insertAdjacentHTML" not in js
    assert "document.write" not in js
    # full uint64 range handled via BigInt, not Number coercion
    assert "BigInt" in js
    assert "18446744073709551615" in js
    assert "crypto.getRandomValues" in js


def test_seed_grammar_is_bounded_before_bigint_without_truncation():
    js = _read("app.js")
    # parseSeed enforces a trimmed grammar of at most 20 characters BEFORE
    # the digit regex and any BigInt call (documented policy: longer strings,
    # even all-leading-zero ones, are rejected on length, keeping the user's
    # exact text — an inline error explains it).
    m = re.search(r"function parseSeed\(str\)\s*\{(.*?)\n  \}", js, re.S)
    assert m, "parseSeed not found"
    body = m.group(1)
    assert ".trim()" in body
    assert r"/^\d+$/" in body
    assert body.index("length > SEED_MAX_DIGITS") < body.index(r"/^\d+$/"), \
        "length bound must precede the digit regex"
    assert body.index("length > SEED_MAX_DIGITS") < body.index("BigInt"), \
        "length bound must precede BigInt"
    assert "SEED_MAX_DIGITS = SEED_MAX.toString().length" in js
    # maxlength silently TRUNCATES an over-length paste into a DIFFERENT
    # valid seed — it must not exist; parseSeed is the only enforcement.
    html = _read("index.html")
    m = re.search(r"<input id=\"seed\"[^>]*>", html)
    assert m and "maxlength" not in m.group(0), \
        "seed input must NOT carry maxlength (silent truncation risk)"
    assert "nothing was truncated" in js.lower() or "kept exactly as entered" in js.lower(), \
        "over-length input needs an explicit inline explanation"


def test_seed_row_is_a_full_row_with_short_helper_and_accessible_range():
    html = _read("index.html")
    seed_block = re.search(r'<div class="field seed-field">.*?\n {10}</div>',
                           html, re.S)
    assert seed_block, "seed must be its own full-width field row"
    assert 'id="steps"' not in seed_block.group(0), "seed must not share the Steps row"
    assert "Leave at 0 or choose Random." in html
    # precise range explanation retained accessibly (title attribute)
    assert "18446744073709551615" in html and "title=" in html
    css = _read("app.css")
    assert re.search(r"\.seed-row input\s*{[^}]*min-width:\s*0", css)


def test_generate_is_always_disabled_with_no_capability_probe_anywhere():
    # W1 removed the speculative /api/v1/capability probe entirely: the page
    # makes no network requests under ANY protocol, and Generate is disabled
    # unconditionally. No test here may promise an endpoint or a ready state.
    for name in ASSETS:
        text = _read(name)
        assert "api/v1" not in text, f"{name} still references a W1 endpoint"
        assert "capability" not in text.lower(), f"{name} still probes capability"
        assert "location.protocol" not in text
    js = _read("app.js")
    assert not re.search(r"\bfetch\s*\(", js), "W1 must not call fetch()"
    assert re.search(r"generate\.disabled\s*=\s*true;\s*//\s*unconditional", js)
    assert "disabled = false" not in js, "nothing may enable Generate in W1"
    lowered = js.lower().replace("readystate", "")
    assert "ready" not in lowered, "no ready-state semantics may survive in W1 app.js"
    html = _read("index.html")
    m = re.search(r'id="generate-btn"[^>]*', html)
    assert m and "disabled" in m.group(0), "Generate must start disabled"
    # disabled visual state is NOT the gate: handlers refuse unconditionally
    assert "function refuseSubmit" in js
    assert js.count("refuseSubmit()") >= 2, "submit AND keydown must refuse"
    assert "Nothing was submitted" in js


def test_form_snapshot_uses_exact_config_enums_and_keys():
    js = _read("app.js")
    # UI wire snapshot mirrors GenerationConfig keys; seed stays a STRING
    m = re.search(r"seed:\s*([^\n]+)", js)
    assert m and "seedWireString" in m.group(1) and "Number" not in m.group(1)
    assert "ui-wire/w1" in js
    m = re.search(r"function formSnapshot\(\)\s*\{.*?return\s*\{(.*?)\};", js, re.S)
    assert m, "formSnapshot not found"
    keys = set(re.findall(r"^\s*(\w+):", m.group(1), re.M))
    config_fields = {f.name for f in dataclasses.fields(GenerationConfig)}
    assert keys - {"kind"} <= config_fields, \
        f"snapshot keys outside GenerationConfig fields: {keys - config_fields}"
    assert keys - {"kind"} == config_fields - {
        "output_dir", "cache_dir", "log_dir", "benchmarks_dir",
        "models_dir", "manifest_path",
    }, "snapshot must carry every non-path GenerationConfig field, no aliases"

    # Selects expose EXACTLY the config.py enum values — no invented options,
    # no aliases, compared against the real Python enums.
    html = _read("index.html")
    for select_id, enum_cls in (("quant", Quant), ("offload", OffloadMode),
                                ("attention", AttentionBackend)):
        m = re.search(rf'<select id="{select_id}".*?</select>', html, re.S)
        assert m, f"select {select_id} missing"
        values = set(re.findall(r'value="([^"]+)"', m.group(0)))
        assert values == {member.value for member in enum_cls}, \
            f"{select_id} options diverge from {enum_cls.__name__}"
    for invented in ("candidate-a", "candidate-b", 'value="on"',
                     'value="off"', 'value="default"'):
        assert invented not in html


def test_validation_marks_aria_invalid_on_every_control():
    js = _read("app.js")
    m = re.search(r"function renderValidation\((?:errors)?\w*\)\s*\{(.*?)\n  \}",
                  js, re.S)
    assert m, "renderValidation not found"
    body = m.group(1)
    for control in ("steps", "seed", "cfg", "prompt", "width", "height"):
        assert f"'{control}'" in body, f"renderValidation misses {control}"
    assert "setAriaInvalid" in body
    # dimension error must mark BOTH dimension controls
    assert body.count("!!errors.dims") == 2
    html = _read("index.html")
    # both dimension inputs share the dims error description
    assert html.count('aria-describedby="dims-error"') == 2


def test_prompt_required_like_config_but_never_alerted_on_load():
    # config.py validates prompt as a non-empty trimmed string; the form
    # mirrors that truthfully, gated so the error only appears after the
    # user edited the prompt or attempted submit — never a first-load alert.
    js = _read("app.js")
    m = re.search(r"function validateForm\(\)\s*\{(.*?)\n  \}", js, re.S)
    assert m and r"$('prompt').value.trim()" in m.group(1), \
        "validateForm must check the prompt (mirrors config.py)"
    assert "promptTouched" in js
    m = re.search(r"function renderValidation\((?:errors)?\w*\)\s*\{(.*?)\n  \}",
                  js, re.S)
    assert m and "promptTouched" in m.group(1), \
        "prompt error must be gated on user interaction"
    assert "auto-fill" in _read("index.html").lower() or "auto-filled" in _read("index.html").lower(), \
        "the page must state it never auto-fills the prompt"
    html = _read("index.html")
    assert 'id="prompt-error"' in html and 'aria-describedby="prompt-help prompt-error"' in html


def test_routing_focuses_headings_only_on_user_hashchange():
    js = _read("app.js")
    assert 'addEventListener(\'hashchange\', routeOnHashchange)' in js
    assert "heading.focus()" in js
    m = re.search(r"function initialRoute\(\)\s*\{(.*?)\n  \}", js, re.S)
    assert m and ".focus()" not in m.group(1), "no autofocus on initial load"
    m = re.search(r"function routeOnHashchange\(\)\s*\{(.*?)\n  \}", js, re.S)
    assert m, "routeOnHashchange not found"
    assert "name === ''" in m.group(1), \
        "empty hash must route to Create (back/forward from hash-less start)"
    assert "hasOwnProperty.call(VIEWS, name)" in m.group(1), \
        "unknown non-empty hashes (#main) must not reroute to Create"
    html = _read("index.html")
    for heading in ("create-heading", "history-heading", "settings-heading"):
        assert f'id="{heading}" tabindex="-1"' in html or \
               'tabindex="-1"' in re.search(
                   rf'<h2[^>]*id="{heading}"[^>]*>', html).group(0), \
            f"{heading} needs tabindex=-1 to be focusable"


def test_no_storage_or_service_apis_in_w1():
    js = _read("app.js")
    for forbidden in ("localStorage", "sessionStorage", "indexedDB",
                      "navigator.serviceWorker"):
        assert forbidden not in js, f"W1 must not use {forbidden}"


def test_privacy_wording_is_factual_about_locality():
    html = _read("index.html")
    m = re.search(r"<dt>Privacy</dt><dd>(.*?)</dd>", html, re.S)
    assert m
    privacy = " ".join(m.group(1).split())
    assert "no third-party requests" in privacy
    assert "analytics" in privacy
    assert "static hosting" in privacy and "enforces nothing" in privacy, \
        "must not suggest static hosting enforces locality"


def test_assets_import_nothing_python_and_module_stays_ml_free():
    # The static tree carries no Python modules; importing the package must
    # remain torch/diffusers/transformers-free (boundary invariant).
    assert not list(WEB_DIR.glob("*.py"))
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, "-c",
         "import qwen_engine, sys;"
         "assert not [m for m in sys.modules if m.split('.')[0] in "
         "('torch','diffusers','transformers')]"],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
