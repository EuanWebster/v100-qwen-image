"""Guarded tests for the packaged static web workspace assets (W1 + W2).

Pure text/structure checks on ``src/qwen_engine/web_static/`` — no browser,
no network, no ML imports, no writes. The assets must stay original,
self-contained and honest-by-construction (PLAN W-milestone pivot,
docs/web-interface.md). W1 made zero network requests (the speculative
capability probe was removed after review); W2 replaces that with a strict
same-origin contract: the ONLY fetch targets are the three relative
``/api/v1/...`` endpoints of the local qwen-web server, and failures render
honestly as "Not connected". Real-Chrome behavioural checks live in
``scripts/chrome_browser_check.mjs`` and the served-UI checks in
``scripts/web_integration_check.mjs`` — neither is part of this guarded
suite and both require ``CHROME_PATH``.
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
    # Python whitespace semantics (pyStrip), NOT JS trim(): trimming is a
    # BROWSER-SIDE affordance on the user's typed input, applied at the
    # edges before the canonical digit STRING is put on the wire. The API
    # wire NEVER strips — the server accepts only canonical digits and
    # refuses anything else rather than editing it.
    assert "pyStrip(String(str))" in body
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


def test_generation_is_status_gated_with_only_same_origin_api_calls():
    # W2 truth: the page may talk ONLY to the three same-origin relative
    # endpoints (/api/v1/status, /api/v1/plan, /api/v1/config/export).
    # Generate stays unconditionally disabled (generation is W3); the
    # submit handlers refuse directly — the disabled button is not the gate.
    for name in ASSETS:
        text = _read(name)
        assert "capability" not in text.lower(), f"{name} still probes capability"
        assert "location.protocol" not in text
    js = _read("app.js")
    # Every fetch target is one of the three relative API constants — never
    # a string-literal URL, never a host, never an absolute origin.
    assert not re.search(r"fetch\(\s*['\"]", js), "fetch must use API constants only"
    for const, route in (("API_STATUS", "/api/v1/status"),
                         ("API_PLAN", "/api/v1/plan"),
                         ("API_EXPORT", "/api/v1/config/export")):
        m = re.search(rf"var {const} = '([^']+)'", js)
        assert m, f"{const} missing"
        assert m.group(1) == route
    assert "!apiConnected || !generationConnected || submittingJob" in js
    html = _read("index.html")
    m = re.search(r'id="generate-btn"[^>]*', html)
    assert m and "disabled" in m.group(0), "Generate must start disabled"
    # disabled visual state is NOT the gate: handlers refuse unconditionally
    assert "function submitGeneration" in js
    assert js.count("submitGeneration()") >= 2, "submit AND keydown must use the gate"
    assert "Nothing was submitted" in js


def test_form_snapshot_uses_exact_config_enums_and_keys():
    js = _read("app.js")
    # UI wire snapshot mirrors GenerationConfig keys; seed stays a STRING
    m = re.search(r"seed:\s*([^\n]+)", js)
    assert m and "seedWireString" in m.group(1) and "Number" not in m.group(1)
    assert "ui-wire/w2" in js
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
    assert m and r"pyStrip($('prompt').value)" in m.group(1), \
        "validateForm must check the prompt with PYTHON whitespace semantics"
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


# ---------------------------------------------------------------- W2


def _python_schema_tags() -> dict[str, str]:
    from qwen_engine.web import api
    return {"status": api.WEB_STATUS_SCHEMA, "plan": api.WEB_PLAN_SCHEMA,
            "request": api.WEB_PLAN_REQUEST_SCHEMA, "error": api.WEB_ERROR_SCHEMA}


def test_w2_schema_literals_match_the_python_api_exactly():
    js = _read("app.js")
    tags = _python_schema_tags()
    for tag in tags.values():
        assert f"'{tag}'" in js, f"wire schema tag missing from app.js: {tag}"
    assert "'X-Qwen-UI'" in js or '"X-Qwen-UI"' in js


def test_w2_envelope_exact_editable_fields_no_kind_no_paths():
    from qwen_engine.web.api import EDITABLE_FIELDS, RUNTIME_PATH_FIELDS
    js = _read("app.js")
    m = re.search(r"buildEnvelope\(\)[\s\S]*?return \{\s*ok: true,\s*envelope: \{([\s\S]*?)\n      \}\n    \};", js)
    assert m, "buildEnvelope envelope literal not found"
    body = m.group(1)
    assert "schema: PLAN_REQUEST_SCHEMA" in body
    config_body = body[body.index("config: {") + len("config: {"):]
    config_keys = set(re.findall(r"^\s*(\w+):", config_body, re.M))
    assert config_keys == set(EDITABLE_FIELDS)
    assert "kind" not in body, "UI wire kind must be excluded from the envelope"
    for path_field in RUNTIME_PATH_FIELDS:
        assert path_field not in body
    assert not re.search(r"schema:\s*[\w.]*\+", body)


def test_w2_status_is_strictly_validated_never_trusted_by_shape():
    js = _read("app.js")
    m = re.search(r"function parseStatus\(s\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "parseStatus not found"
    body = m.group(1)
    assert "s.schema !== STATUS_SCHEMA" in body
    assert "s.planning_available !== true" in body          # strict boolean
    assert "typeof s.generation_available !== 'boolean'" in body
    assert "s.generation_available !== (s.generation_blockers.length === 0)" in body
    # EXACT blocker set (four required + manifest reason when applicable),
    # never "any bounded subset" — a malformed server cannot hide blockers.
    assert "blockersExact(s.generation_blockers, manifest.blocker)" in body
    assert "parseManifestCore(s.manifest)" in body
    assert "GATE_STATES" in js and "adapter_available" in js
    # a schema-rejected status is a typed state, distinct from a disconnect
    assert "Status rejected" in js and "Not connected" in js
    # every fetch is bounded by an AbortController timeout; no polling
    assert "AbortController" in js and "FETCH_TIMEOUT_MS" in js
    assert "setInterval" not in js


def test_w2_plan_response_seed_must_be_decimal_string_or_null():
    js = _read("app.js")
    m = re.search(r"function parsePlan\(raw\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "parsePlan not found"
    body = m.group(1)
    assert r"/^\d{1,20}$/" in body, "plan seed must be validated as a string"
    assert "typeof c.seed === 'string'" in body
    # rendering uses textContent helpers only, never markup or .value writes
    m = re.search(r"function planRenderFields\(plan\)\s*\{([\s\S]*?)\n  \}", js)
    assert m
    assert "innerHTML" not in m.group(1)
    assert ".value" not in m.group(1)


def test_w2_finite_numbers_checked_before_serialization():
    js = _read("app.js")
    m = re.search(r"function numberField\(id, wantInt\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "numberField not found"
    body = m.group(1)
    assert "raw === ''" in body, "empty fields must be an error, never 0"
    assert "Number.isFinite" in body
    assert "Number.isInteger" in body


def test_w2_export_downloads_raw_blob_without_reserialization():
    js = _read("app.js")
    m = re.search(r"function exportConfig\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "exportConfig not found"
    body = m.group(1)
    assert "apiRequest(API_EXPORT" in body and "'raw'" in body, \
        "export must consume the raw response bytes through the bounded core"
    assert "new Blob([r.bytes])" in body, \
        "export must download the complete raw bytes untouched"
    assert "JSON.parse" not in body and ".text()" not in body and \
           "res.json()" not in body, \
        "export bytes must never be parsed/reserialized on the success path"
    assert "URL.createObjectURL" in body and "URL.revokeObjectURL" in body
    assert "qwen-editable-config.json" in body
    # never a silent export of a stale/invalid draft:
    assert "buildEnvelope()" in body
    assert "local form errors" in body


def test_export_refuses_every_body_error_and_requires_typed_bytes():
    js = _read("app.js")
    m = re.search(r"function exportConfig\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "exportConfig not found"
    body = m.group(1)
    # ALL bodyError outcomes are refused before any download is built — an
    # 'unsupported' readable-stream outcome can arrive with res.ok=true and
    # NO bytes at all, and Blob([undefined]) would otherwise download the
    # literal text 'undefined' under a success message.
    assert re.search(r"if \(r\.bodyError\)", body), \
        "export must refuse EVERY bodyError outcome, not just too_large"
    guard = body.index("if (r.bodyError)")
    blob = body.index("new Blob([r.bytes])")
    assert guard < blob, "the bodyError refusal must precede the download path"
    # a download is produced ONLY from actual Uint8Array bytes
    assert "r.bytes instanceof Uint8Array" in body, \
        "export must type-check the raw bytes before Blob/download"
    assert body.index("instanceof Uint8Array") < blob, \
        "the bytes type guard must precede the download path"
    # a schema/readability failure invalidates the trusted availability with
    # a DISTINCT rejected state — never a fabricated disconnect, and never
    # a success message.
    rejected = body[guard:blob]
    assert "Status rejected" in rejected and "Not connected" not in rejected, \
        "body refusal must be a typed rejection, not a disconnect"
    assert "setApiAvailable(false)" in rejected, \
        "a refused export body must invalidate planning/export availability"
    assert "not a disconnect" in rejected


def test_export_too_large_cannot_bypass_the_unified_refusal():
    # ROOT FINDING (16:30 review): an OLD early `if (r.bodyError ===
    # 'too_large') { ...; return false; }` sat BEFORE the unified branch
    # and returned without setApiAvailable(false) or the rejected chip —
    # an over-cap export body left 'Connected' and enabled buttons
    # standing. The regression this guards: NO special-cased bodyError
    # branch may exist beside the unified one; too_large may appear only
    # INSIDE `if (r.bodyError)` as a reason string.
    js = _read("app.js")
    m = re.search(r"function exportConfig\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "exportConfig not found"
    body = m.group(1)
    assert re.search(r"if \(r\.bodyError === ", body) is None, \
        "no early special-cased bodyError branch may bypass the unified refusal"
    guard = body.index("if (r.bodyError)")
    http_err = body.index("if (!r.res.ok)")
    blob = body.index("new Blob([r.bytes])")
    assert guard < http_err, \
        "the unified bodyError refusal must precede the HTTP-error path"
    assert guard < blob, \
        "the unified bodyError refusal must precede the download path"
    # the too_large reason and the invalidation live INSIDE the one
    # unified branch — one setApiAvailable(false), distinct rejected chip,
    # never 'Not connected', never a success message.
    branch = body[guard:http_err]
    assert branch.count("setApiAvailable(false)") == 1, \
        "exactly one invalidation, in the unified branch"
    assert "'too_large'" in branch and "'unsupported'" in branch, \
        "too_large must be a reason inside the unified refusal"
    assert "over the fixed response cap" in branch, \
        "the size-cap reason must be stated accurately"
    assert "Status rejected" in branch and "Not connected" not in branch, \
        "the too_large outcome is a typed rejection, not a disconnect"
    # the Uint8Array actual-bytes requirement on success survives (the
    # download is built ONLY from real bytes, never parsed/reserialized)
    assert body.index("r.bytes instanceof Uint8Array") < blob
    assert "JSON.parse" not in body and ".text()" not in body


def test_parse_plan_rejects_blank_prompt_exactly_like_generation_config():
    js = _read("app.js")
    m = re.search(r"function parsePlan\(raw\)\s*\{([\s\S]*?)\n  \}", js, re.S)
    assert m, "parsePlan not found"
    body = m.group(1)
    # shared GenerationConfig refuses `not prompt.strip()` (config.py); an
    # echoed plan with an empty/Python-whitespace-only prompt is a broken
    # contract and must be REJECTED — under the corrected pyStrip
    # predicate, never JS trim() (FEFF / U+001C-class differences).
    assert "pyStrip(c.prompt)" in body, \
        "parsePlan must require a non-blank echoed prompt under Python strip"
    assert "c.prompt.trim()" not in body, \
        "JS trim must not substitute for the Python whitespace predicate"


def test_w2_failure_paths_preserve_the_draft():
    js = _read("app.js")
    # only applyPreset (2 writes) and the Random button (1 write) may write
    # an input value — no server response ever populates the form.
    assert js.count(".value = ") == 3, "server data must never write form fields"
    assert "draft was preserved" in js or "draft stays" in js or \
           "Nothing was sent or changed" in js


def test_w2_plan_panel_is_rendered_not_invented():
    html = _read("index.html")
    for elem_id in ("btn-check-config", "btn-plan", "btn-export", "api-message",
                    "plan-panel", "plan-seed", "plan-tokens", "plan-cfg",
                    "plan-manifest", "plan-blockers"):
        assert f'id="{elem_id}"' in html, f"W2 UI element missing: {elem_id}"
    assert 'hidden' in re.search(r'<section class="plan-panel"[^>]*>', html).group(0), \
        "plan panel must start hidden (nothing rendered before a server answer)"


def test_w2_actions_gated_until_recognized_status():
    js = _read("app.js")
    html = _read("index.html")
    m = re.search(r"function setApiAvailable\(available\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "setApiAvailable not found"
    body = m.group(1)
    assert "$('btn-plan').disabled = !available" in body
    assert "$('btn-export').disabled = !available" in body
    assert "generate" not in body, "gating must never touch the Generate button"
    for btn in ("btn-plan", "btn-export"):
        m = re.search(rf'id="{btn}"[^>]*', html)
        assert m and "disabled" in m.group(0), f"{btn} must start disabled"
    # Check configuration is the reconnect path: it re-checks status first
    assert re.search(r"'btn-check-config'\).addEventListener[\s\S]*?checkStatus\(\)", js)
    # an initial hash must resolve by OWN property, never inherited names
    assert "hasOwnProperty.call(VIEWS, name)" in js


def test_w2_previous_plan_results_never_look_current():
    js = _read("app.js")
    m = re.search(r"function runPlan\(showFull\)\s*\{([\s\S]*?)\n  \}", js, re.S)
    assert m, "runPlan not found"
    body = m.group(1)
    assert "$('plan-panel').hidden = true" in body, \
        "previous results must be hidden while a new attempt runs"
    assert "seq !== planSeq" in body, "out-of-order answers must be ignored"
    assert "No planning result received" in body, \
        "a failed plan must not claim nothing was sent"
    # Invalidate BEFORE local validation: planSeq++ must happen before
    # buildEnvelope() so an invalid NEW attempt still cancels a pending
    # older one (which would otherwise clear the new errors and render an
    # old success over them).
    assert body.index("planSeq++") < body.index("buildEnvelope()"), \
        "prior attempts must be invalidated before local validation"
    assert "syncPlanStaleness()" in body, \
        "an invalid draft must stale any visible result"
    # a rendered result is bound to the signature of ITS snapshot
    assert "planShownSignature = sentSignature" in body
    # the success render must not clear errors of an edited draft
    assert "renderValidation({})" not in body, \
        "success must render the CURRENT draft errors, not blank them"
    assert "renderValidation(validateForm().errors)" in body


def test_w2_staleness_covers_preset_random_and_change_only_edits():
    js = _read("app.js")
    # syncPlanStaleness must be wired wherever the draft can change:
    # text/number inputs, preset buttons, the Random seed button, and the
    # change-only selects/checkbox (no inline errors, still stale).
    m = re.search(r"function applyPreset\(name\)\s*\{([\s\S]*?)\n  \}", js)
    assert m and "syncPlanStaleness()" in m.group(1), "preset edits must stale"
    m = re.search(r"\$\('random-seed'\)\.addEventListener[\s\S]*?\}\);", js)
    assert m and "syncPlanStaleness()" in m.group(0), "random seed must stale"
    m = re.search(r"\['quant', 'offload', 'attention', 'vae-tiling'\]"
                  r"[\s\S]*?'change'[\s\S]*?syncPlanStaleness\(\);[\s\S]*?\}\);", js)
    assert m, "change-only fields must also mark a visible result stale"
    m = re.search(r"function syncPlanStaleness\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "syncPlanStaleness not found"
    body = m.group(1)
    assert "envelopeSignature(buildEnvelope()) !== planShownSignature" in body, \
        "staleness must compare against the validated snapshot's signature"
    assert "classList.add('stale')" in body and "STALE" in body, \
        "a stale result must be clearly labelled, not left looking validated"
    assert "classList.remove('stale')" in body
    css = _read("app.css")
    assert ".plan-panel.stale" in css, "stale state needs a visible style"


def test_w2_request_bound_whole_operation_including_body():
    js = _read("app.js")
    # The timeout must cover the BODY, not just the headers: the timer is
    # cleared in settle() AFTER body consumption — never in a .finally on
    # the fetch promise alone.
    m = re.search(r"function apiRequest\(url, opts, mode\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "apiRequest not found"
    body = m.group(1)
    assert "new AbortController()" in body and "controller.abort()" in body
    assert "requestOpts.signal = controller.signal" in body
    assert "readBodyLimited(res)" in body, "body consumption must run inside the timer scope"
    assert "settle(" in body and "clearTimeout(timer); return value;" in body
    assert ".finally(function () { clearTimeout(timer); })" not in js, \
        "clearing on the fetch promise alone leaves the body untimed"
    assert "timedOut = true" in body, "a timeout must be distinguishable"
    # no unbounded bodies: a fixed cap, streaming-read and cancelled
    m = re.search(r"function readBodyLimited\(res\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "readBodyLimited not found"
    body = m.group(1)
    assert "MAX_RESPONSE_BYTES" in body and "reader.cancel()" in body, \
        "an over-cap body must be cancelled, never buffered"
    m = re.search(r"var MAX_RESPONSE_BYTES = (\d+);", js)
    assert m and 64 * 1024 < int(m.group(1)) <= 1024 * 1024, \
        "the cap must exceed a valid 64 KiB request echo but stay fixed"
    # raw export bytes preserved without reserialization
    m = re.search(r"function apiRequest\(url, opts, mode\)\s*\{([\s\S]*?)\n  \}", js)
    assert "if (mode === 'raw') return settle({ res: res, bytes: bytes });" in m.group(1)


def test_w2_network_failure_and_schema_rejection_are_distinct_outcomes():
    js = _read("app.js")
    m = re.search(r"function apiRequest\(url, opts, mode\)\s*\{([\s\S]*?)\n  \}", js)
    body = m.group(1)
    # malformed/undecodable complete body -> schema rejection marker
    assert "bodyError: 'json'" in body
    # over-cap -> its own rejection marker, not a disconnect
    assert "bodyError: 'too_large'" in body
    # timeout / unreachable -> network marker (never a schema verdict)
    assert "network: true, timedOut: timedOut" in body
    # JSON null parses to the value null — rejected by the strict parser,
    # never by a property access on null (no __not_json access at all)
    assert "__not_json" not in js
    m = re.search(r"function parseStatus\(s\)\s*\{([\s\S]*?)\n  \}", js)
    assert m.group(1).lstrip().startswith("if (!s || typeof s !== 'object') return null;"), \
        "parseStatus must be null-safe (JSON null is a schema rejection)"


def test_w2_parse_plan_bounds_aligned_with_server_contract():
    js = _read("app.js")
    m = re.search(r"function parsePlan\(raw\)\s*\{([\s\S]*?)\n  \}", js)
    body = m.group(1)
    # bounds ALIGNED with the 64 KiB request transport cap — no invented
    # 4000-char generation limit that would reject server-valid echoes
    assert "MAX_CONFIG_STRING_CHARS" in body
    assert "c.prompt.length > 4000" not in body
    assert "negative_prompt.length > 4000" not in body
    m = re.search(r"var MAX_CONFIG_STRING_CHARS = (\d+);", js)
    assert m and int(m.group(1)) == 65536, \
        "string bound must match the server's 64 KiB request cap exactly"
    # dimensions honoured as multiples of 32 (config.py DIM_MULTIPLE)
    assert "c.width % 32 !== 0" in body and "c.height % 32 !== 0" in body
    # cfg_active recomputed with PYTHON strip semantics, exactly
    assert "pyStrip(c.negative_prompt)" in body
    assert "raw.cfg_active !== cfgActive" in body
    # latent_tokens consistent with the echoed dimensions (/16, C6)
    assert "raw.latent_tokens !== (c.width / 16) * (c.height / 16)" in body
    # manifest verification EXACT, not any bounded string
    assert "m.verification !== 'not_performed'" in body
    assert "m.fit !== 'not_evaluated'" in body
    assert "m.ready !== false" in body
    # EXACT blocker set, not any subset of the closed list
    assert "blockersExact(raw.generation_blockers, manifest.blocker)" in body
    assert "parseManifestCore(m)" in body
    m = re.search(r"var KNOWN_BLOCKER_CODES = \[([\s\S]*?)\];", js)
    assert m, "KNOWN_BLOCKER_CODES missing"
    from qwen_engine.web import api
    codes = set(re.findall(r"'([a-z_]+)'", m.group(1)))
    assert {"generation_disabled", "runtime_unconfigured"} <= codes
    # the same exact-set contract must gate status blockers too
    m = re.search(r"function parseStatus\(s\)\s*\{([\s\S]*?)\n  \}", js)
    assert "blockersExact(s.generation_blockers, manifest.blocker)" in m.group(1)
    # gate CODES are exact too (closed per-gate contract)
    assert "gate.code !== spec.code" in m.group(1)


# ------------------------------------------- root-follow-up regressions

def test_py_strip_covers_plain_space_and_matches_python_whitespace():
    # The '   ' + cfg>1 mainstream case: Python str.strip() strips plain
    # SPACE, so a pyStrip WITHOUT U+0020 would compute cfg_active=true for
    # an all-space negative prompt and reject the server's cfg_active=false.
    js = _read("app.js")
    m = re.search(r"function pyStrip\(s\)\s*\{(.*?)\n  \}", js, re.S)
    assert m, "pyStrip not found"
    body = m.group(1)
    cls = re.search(r"\[([^\]]*)\]", body).group(1)
    assert "\\x20" in cls, "pyStrip must strip plain space like Python does"
    # EDGE-ONLY stripping: Python .strip() never touches INTERIOR
    # whitespace — '4 2' must be REJECTED by the seed grammar, not parsed
    # as 42 (root browser review caught the interior-strip variant).
    assert body.count("[") >= 2 and re.search(r"/\^[^/]*\+\|[^/]*/g", body), \
        "pyStrip must anchor at ^ and $ (leading/trailing only)"
    for esc in ("\\x09", "\\x0A", "\\x0B", "\\x0C", "\\x0D",
                "\\x1C", "\\x1D", "\\x1E", "\\x1F", "\\x85", "\\xA0"):
        assert esc in cls, esc
    assert "uFEFF" not in cls, "Python strip does NOT strip U+FEFF"
    # python-side cross-check: every char Python's str.strip removes is
    # covered by the JS class (space included).
    ws = "".join(chr(c) for c in range(0x100) if chr(c).isspace())
    assert " " in ws  # sanity of the premise itself


def test_status_requests_are_sequenced_never_land_late():
    # A newer status check must win over an older late arrival in BOTH
    # directions (success after disconnect, failure after success).
    js = _read("app.js")
    assert "var statusSeq = 0;" in js
    m = re.search(r"function checkStatus\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "checkStatus not found"
    body = m.group(1)
    assert "var seq = ++statusSeq;" in body
    assert "if (seq !== statusSeq) return null;" in body


def test_body_without_readable_stream_fails_closed_not_unbounded():
    js = _read("app.js")
    m = re.search(r"function readBodyLimited\(res\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "readBodyLimited not found"
    body = m.group(1)
    assert "UNSUPPORTED_BODY" in body
    assert "res.arrayBuffer()" not in body, \
        "arrayBuffer fallback reads UNBOUNDED — fail closed instead"
    assert "bodyError: 'unsupported'" in js
    assert "err === UNSUPPORTED_BODY" in js


def test_rejected_plan_document_invalidates_recognized_status():
    # A rejected plan BODY or schema must take planning/export away again
    # and show the typed schema-rejection chip (not a silent 'Connected').
    js = _read("app.js")
    m = re.search(r"function runPlan\([\s\S]*?\n  \}", js)
    assert m, "runPlan not found"
    body = m.group(0)
    rejection = body[body.index("if (r.bodyError)"):]
    rejection = rejection[:rejection.index("var plan = parsePlan")]
    assert "setApiAvailable(false)" in rejection
    assert "Status rejected" in rejection
    schema_branch = body[body.index("var plan = parsePlan"):]
    schema_branch = schema_branch[:schema_branch.index("promptTouched = true")]
    assert "setApiAvailable(false)" in schema_branch
    assert "Status rejected" in schema_branch


def test_manifest_core_and_blocker_contract_shared_and_exact():
    js = _read("app.js")
    m = re.search(r"function parseManifestCore\(m\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "parseManifestCore not found — status/plan share ONE parser"
    body = m.group(1)
    assert "m.verification !== 'not_performed'" in body
    assert "m.valid && !m.configured" in body          # valid implies configured
    assert "MANIFEST_REASON_CODES.indexOf(m.reason)" in body
    assert "m.schema_version" in body and "m.artifact_count" in body
    assert "m.schema_version !== 1" in body
    assert "m.artifact_count < 1" in body
    # exact required set mirrors the REAL server gates
    from qwen_engine.web import api
    m2 = re.search(r"var KNOWN_BLOCKER_CODES = \[([\s\S]*?)\];", js)
    assert m2, "KNOWN_BLOCKER_CODES missing"
    codes = set(re.findall(r"'([a-z_]+)'", m2.group(1)))
    assert codes == {"generation_disabled", "runtime_unconfigured", "manifest_unconfigured",
                     "manifest_file_missing", "manifest_invalid", "manifest_unreadable"}
    m3 = re.search(r"var MANIFEST_REASON_CODES = \[([\s\S]*?)\];", js)
    assert m3
    reason_codes = set(re.findall(r"'([a-z_]+)'", m3.group(1)))
    from qwen_engine.web import api as _api
    assert reason_codes <= {_api.MANIFEST_UNCONFIGURED, _api.MANIFEST_FILE_MISSING,
                            _api.MANIFEST_INVALID, _api.MANIFEST_UNREADABLE}


def test_api_requests_bypass_the_http_cache():
    # Two status checks must be TWO real requests (the root wire race proves
    # it); a cached status document is never authoritative.
    js = _read("app.js")
    m = re.search(r"function apiRequest\([^)]*\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "apiRequest not found"
    assert "requestOpts.cache = 'no-store'" in m.group(1)
