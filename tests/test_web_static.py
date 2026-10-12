"""W1 guarded tests for the packaged static web workspace assets.

Pure text/structure checks on ``src/qwen_engine/web_static/`` — no browser,
no network, no ML imports, no writes. The assets must stay original,
self-contained and honest-by-construction (PLAN W-milestone pivot,
docs/web-interface.md).
"""
from __future__ import annotations

import re
from pathlib import Path

import qwen_engine

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


def test_starts_disconnected_with_disabled_generate():
    html = _read("index.html")
    assert "Not connected" in html
    m = re.search(r'id="generate-btn"[^>]*', html)
    assert m and "disabled" in m.group(0), "Generate must start disabled"
    js = _read("app.js")
    # capability probe is fixed-path and protocol-gated (no file:// fetch)
    assert "'/api/v1/capability'" in js
    assert "location.protocol" in js


def test_form_snapshot_uses_exact_config_enums_and_keys():
    js = _read("app.js")
    # UI wire snapshot mirrors GenerationConfig keys; seed stays a STRING
    m = re.search(r"seed:\s*([^\n]+)", js)
    assert m and "seedWireString" in m.group(1) and "Number" not in m.group(1)
    for key in ("negative_prompt", "true_cfg_scale", "attention_backend",
                "vae_tiled_decode"):
        assert key in js, f"snapshot key {key} missing"
    assert "ui-wire/w1" in js
    # Selects expose exactly the config.py enum values — no invented options.
    html = _read("index.html")
    offload = re.search(r'<select id="offload".*?</select>', html, re.S)
    attention = re.search(r'<select id="attention".*?</select>', html, re.S)
    quant = re.search(r'<select id="quant".*?</select>', html, re.S)
    assert offload and attention and quant
    for value in ("resident", "streamed", "offload"):
        assert f'value="{value}"' in offload.group(0)
    for value in ("auto", "sdpa", "efficient", "eager"):
        assert f'value="{value}"' in attention.group(0)
    for value in ("fp16", "q8", "q4"):
        assert f'value="{value}"' in quant.group(0)
    for invented in ("candidate-a", "candidate-b", 'value="on"',
                     'value="off"', 'value="default"'):
        assert invented not in html


def test_no_storage_or_service_apis_in_w1():
    js = _read("app.js")
    for forbidden in ("localStorage", "sessionStorage", "indexedDB",
                      "navigator.serviceWorker"):
        assert forbidden not in js, f"W1 must not use {forbidden}"


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
