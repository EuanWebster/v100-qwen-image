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
import json
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
    # W2 truth: the page may talk ONLY to the same-origin relative endpoints
    # (/api/v1/status, /api/v1/plan, /api/v1/config/export, and — R4 — the
    # rewrite route). Generate stays unconditionally disabled (generation is
    # W3); the submit handlers refuse directly — the disabled button is not
    # the gate.
    #
    # The "capability" ban is about BROWSER capability probing (navigator.gpu,
    # WebGPU/WebNN feature detection): the server's rewrite status block is
    # named for what it is (a rewriter state), never "capability", so this
    # assertion keeps its original force.
    for name in ASSETS:
        text = _read(name)
        assert "capability" not in text.lower(), f"{name} still probes capability"
        assert "location.protocol" not in text
    js = _read("app.js")
    # Every fetch target is one of the four relative API constants — never
    # a string-literal URL, never a host, never an absolute origin.
    assert not re.search(r"fetch\(\s*['\"]", js), "fetch must use API constants only"
    for const, route in (("API_STATUS", "/api/v1/status"),
                         ("API_PLAN", "/api/v1/plan"),
                         ("API_EXPORT", "/api/v1/config/export"),
                         ("API_REWRITE", "/api/v1/rewrite")):
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
    # Path fields belong to the SERVER's runtime; use the API's own closed
    # sets instead of a literal list so a new path field cannot silently pass,
    # and so an announced deferral (recipe — docs/turbo-plan.md §6) is the
    # only other class of field the wire may omit.
    from qwen_engine.web.api import DEFERRED_WIRE_FIELDS, RUNTIME_PATH_FIELDS
    assert keys - {"kind"} == config_fields - set(RUNTIME_PATH_FIELDS) - set(DEFERRED_WIRE_FIELDS), \
        "snapshot must carry every non-path, non-deferred GenerationConfig field, no aliases"

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
    for forbidden in ("sessionStorage", "indexedDB",
                      "navigator.serviceWorker"):
        assert forbidden not in js, f"W1 must not use {forbidden}"
    assert "localStorage.setItem(THEME_STORAGE_KEY, theme)" in js
    assert js.count("localStorage.setItem(") == 1


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
    # Explicit import is the only server response allowed to populate fields.
    assert js.count("applyImportedPlan(plan);") == 1
    assert "draftRevision !== revisionAtPick" in js
    assert js.index("var plan = parsePlan(r.json);") < js.index("applyImportedPlan(plan);")
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
    m = re.search(r"\['quant', 'offload', 'attention', 'vae-tiling', 'output-format'\]"
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


# ------------------------------------------------------ transparency (D1)


def test_transparency_control_markup_is_labelled_and_described():
    html = _read("index.html")
    block = re.search(r'<div class="field transparency-field">.*?\n          </div>',
                      html, re.S)
    assert block, "transparency field block missing"
    body = block.group(0)
    assert 'id="transparent"' in body and 'name="transparent_background"' in body
    assert 'type="checkbox"' in body
    assert '<label for="transparent">Transparent background</label>' in body
    assert 'aria-describedby="transparency-help transparency-note"' in body
    assert 'id="transparency-help"' in body and 'id="transparency-note"' in body
    # The effective prompt is display-only: no input control, starts hidden.
    effective = re.search(r'<p id="effective-prompt-field"[^>]*>', body)
    assert effective and "hidden" in effective.group(0)
    assert '<span id="effective-prompt"' in body
    assert 'readonly' not in body   # it is not an editable field at all
    # The format error id the JS writes must exist, and the select points at it.
    assert 'id="format-error"' in html
    select = re.search(r'<select id="output-format"[^>]*>', html).group(0)
    assert 'aria-describedby="format-error"' in select
    # Plain-language explanation names the jpg restriction.
    assert "JPEG cannot store transparency" in body


def test_transparency_wrapper_literals_match_config_py_exactly():
    from qwen_engine.config import (TRANSPARENCY_PROMPT_PREFIX,
                                    TRANSPARENCY_PROMPT_SUFFIX,
                                    TRANSPARENCY_WRAPPER_REVISION)
    js = _read("app.js")

    def js_string(value: str) -> str:
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

    assert f"var TRANSPARENT_PREFIX = {js_string(TRANSPARENCY_PROMPT_PREFIX)};" in js
    assert f"var TRANSPARENT_SUFFIX =\n    {js_string(TRANSPARENCY_PROMPT_SUFFIX)};" in js
    # The revision constant is cited so a later wrapper bump is traceable.
    assert TRANSPARENCY_WRAPPER_REVISION in js
    from qwen_engine.config import ALPHA_OUTPUT_FORMATS
    assert set(re.findall(r"'(\w+)'", re.search(r"var ALPHA_FORMATS = \[([^\]]*)\]", js).group(1))) \
        == {fmt.value for fmt in ALPHA_OUTPUT_FORMATS}


def test_transparency_checkbox_switches_jpg_to_png_and_explains_it():
    js = _read("app.js")
    m = re.search(r"function onTransparencyChange\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "onTransparencyChange not found"
    body = m.group(1)
    assert "ALPHA_FORMATS.indexOf($('output-format').value) === -1" in body
    assert "$('output-format').value = 'png'" in body
    assert "switched from jpg to png" in body
    # Unticking never rewrites the format: the switch is gated on .checked.
    switch = body[body.index("$('transparent').checked"):]
    assert "= 'jpg'" not in body and "= 'webp'" not in body
    assert switch.index("$('transparent').checked") < switch.index("value = 'png'")
    # Wired as a real handler, and validated/staled like every other edit.
    assert "$('transparent').addEventListener('change', onTransparencyChange);" in js
    assert "syncPlanStaleness();" in body
    # The explanation is a persistent note, not a one-frame flash.
    assert "transparencySwitchNote" in js


def test_transparency_state_renders_effective_prompt_and_format_reason():
    js = _read("app.js")
    m = re.search(r"function syncTransparency\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "syncTransparency not found"
    body = m.group(1)
    assert "TRANSPARENT_PREFIX + $('prompt').value + TRANSPARENT_SUFFIX" in body
    assert "$('effective-prompt-field').hidden = !on" in body
    # The prompt FIELD itself is never rewritten by the wrapper.
    assert "$('prompt').value =" not in body
    # The mismatch reason is stated, never auto-corrected silently here.
    assert "JPEG cannot keep transparency" in body
    # Wired into the validation pass (so aria-invalid/format errors agree).
    m = re.search(r"function renderValidation\(errors\)\s*\{(.*?)\n  \}", js, re.S)
    assert m and "syncTransparency();" in m.group(1)
    assert "setText('format-error', errors.format || '')" in m.group(1)
    assert "setAriaInvalid('output-format', !!errors.format)" in m.group(1)


def test_transparency_local_validation_mirrors_the_server_rule():
    js = _read("app.js")
    m = re.search(r"function validateForm\(\)\s*\{(.*?)\n  \}", js, re.S)
    assert m, "validateForm not found"
    assert "transparencyFormatMismatch()" in m.group(1)
    assert "errors.format" in m.group(1)
    m = re.search(r"function transparencyFormatMismatch\(\)\s*\{(.*?)\n  \}", js, re.S)
    assert m, "transparencyFormatMismatch not found"
    assert "ALPHA_FORMATS.indexOf($('output-format').value) === -1" in m.group(1)


def test_plan_parsing_requires_transparent_boolean_and_alpha_format():
    js = _read("app.js")
    m = re.search(r"function parsePlan\(raw\)\s*\{([\s\S]*?)\n  \}", js)
    assert m
    body = m.group(1)
    assert "'transparent_background'" in body, "parsePlan exact-key list must include it"
    assert "typeof c.transparent_background !== 'boolean'" in body
    assert "ALPHA_FORMATS.indexOf(c.output_format) === -1" in body


def test_parse_job_accepts_13_keys_and_keeps_legacy_shapes_honest():
    js = _read("app.js")
    m = re.search(r"function parseJob\(job\)\s*\{([\s\S]*?)\n    return job;\n  \}", js)
    assert m, "parseJob not found"
    body = m.group(0)
    assert "ckeys.length === 13" in body
    assert "ckeys.length === 12" in body and "ckeys.length === 11" in body
    assert "!hasKey('transparent_background')" in body
    assert "c.transparent_background = false" in body
    assert "typeof c.transparent_background !== 'boolean'" in body


def test_job_error_code_gate_matches_the_python_public_map():
    """The UI's closed error_code list and the web layer's PUBLIC_ERROR_CODES
    map are ONE contract; drift means a real failure silently renders as an
    invalid job (server codes are fixed literals, prompt-free)."""
    from qwen_engine.web.generation import PUBLIC_ERROR_CODES
    js = _read("app.js")
    m = re.search(r"if \(\[null,([^\]]*?)\]\.indexOf\(job\.error_code\)", js, re.S)
    assert m, "parseJob error_code gate not found"
    assert set(re.findall(r"'([a-z_]+)'", m.group(1))) == set(PUBLIC_ERROR_CODES)


def test_gpu_oom_failure_message_is_plain_language_and_actionable():
    js = _read("app.js")
    m = re.search(r"out_of_gpu_memory: '([^']*)' \+\s*\n?\s*'([^']*)'", js)
    assert m, "out_of_gpu_memory message not found"
    assert "".join(m.groups()) == ("The GPU ran out of memory for this job. "
                                   "Try Q8, a smaller size, fewer reference images, "
                                   "or Turbo.")
    # only the closed map may replace the generic sentence; no server text
    assert "failureMessages[job.error_code] ||" in js


def test_recipe_field_mirrors_the_config_gates():
    html = _read("index.html")
    js = _read("app.js")
    css = _read("app.css")
    from qwen_engine.config import Recipe, TURBO_STEPS

    # The UI's Turbo step count and recipe names are the SAME values the
    # server validates against.
    assert f"var TURBO_STEPS = {TURBO_STEPS};" in js
    m = re.search(r"var RECIPES = \[([^\]]*)\];", js)
    assert m, "RECIPES not found"
    assert re.findall(r"'([a-z]+)'", m.group(1)) == [r.value for r in Recipe]
    # One radio group, both values, Quality checked by default.
    assert re.search(r'<fieldset[^>]*id="recipe-field"[^>]*>', html)
    for value in ("base", "turbo"):
        assert f'id="recipe-{value}" name="recipe" type="radio" value="{value}"' in html
    assert re.search(r'id="recipe-base"[^>]*checked', html, re.S)
    assert 'id="recipe-help"' in html and 'id="recipe-note"' in html
    assert 'id="recipe-error"' in html
    # The note is a live region and is referenced by the locked controls so
    # the reason for the lock reaches assistive tech.
    note = re.search(r'<p id="recipe-note"[^>]*>', html).group(0)
    assert 'role="status"' in note
    for control in ("steps", "recipe-base", "recipe-turbo"):
        tag = re.search(rf'<input id="{control}"[^>]*', html, re.S)
        assert tag and "recipe-note" in tag.group(0), f"#{control} must describe the recipe note"
    assert re.search(r'<input id="steps"[^>]*aria-describedby="steps-error recipe-note"',
                     html, re.S)
    # Plan panel echoes the recipe.
    assert 'id="plan-recipe"' in html
    # Locks are STYLED (visible-but-inert), never just hidden.
    assert ".radio-row" in css and ".radio-field" in css
    assert "input:disabled" in css and "select:disabled" in css


def test_apply_recipe_mode_locks_and_restores_the_gated_controls():
    js = _read("app.js")
    body = _js_function(js, "applyRecipeMode")
    # Turbo pins exactly what config.py refuses to see otherwise.
    assert "steps.value = String(TURBO_STEPS);" in body
    assert "cfg.value = '1.0';" in body
    assert "quant.value = 'fp16';" in body
    # All four gated controls are locked, and the reason is written down.
    for control in ("steps", "cfg", "negative", "quant"):
        assert f"{control}.disabled = turbo;" in body, f"{control} must be locked under Turbo"
    assert "recipe-note" in body and "hidden = !turbo" in body
    # Leaving Turbo restores the user's own values instead of keeping pins.
    assert "recipeSaved" in body and "steps.value = recipeSaved.steps;" in body
    assert "cfg.value = recipeSaved.cfg;" in body
    assert "quant.value = recipeSaved.quant;" in body
    # An imported plan supplies the recipe's own values: no capture.
    assert "keepValues" in body
    m = re.search(r"function applyImportedPlan\(plan\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "applyImportedPlan not found"
    imported = m.group(1)
    assert "applyRecipeMode(true)" in imported
    # Order matters: the recipe is applied after the file's values so the
    # locks match the file, not a leftover draft.
    assert imported.index("setRecipeControl(c.recipe)") > imported.index("c.quant")
    assert "recipeSaved = null;" in imported


def test_recipe_contract_is_mirrored_by_parse_plan_and_parse_job():
    js = _read("app.js")
    body = _js_function(js, "recipeConsistent")
    assert "PLAN_ENUMS.recipe.indexOf(c.recipe) === -1" in body
    assert "c.steps === TURBO_STEPS" in body
    assert "c.quant === 'fp16'" in body
    assert "true_cfg_scale > 1" in body
    # Both parsers refuse an echo the server could not have produced.
    assert "recipeConsistent(c)" in _js_function(js, "parsePlan")
    assert "recipeConsistent(c)" in _js_function(js, "parseJob")
    # The history label distinguishes the two recipes.
    assert "job.config.recipe === 'turbo' ? ' · Turbo' : ''" in js
    # Server refusals are mapped onto the recipe's own fields.
    assert "recipe: 'recipe-error'" in js and "recipe: 'recipe-field'" in js
    for fn in ("runPlan", "submitGeneration"):
        assert "applyServerFieldErrors(" in _js_function(js, fn), \
            f"{fn} must render per-field server refusals"


def test_recipe_radios_are_wired_and_locked_on_load():
    js = _read("app.js")
    assert 'document.querySelectorAll(\'input[name="recipe"]\')' in js
    # The group handler re-derives the locks, re-renders errors and stales
    # any visible plan result — the same contract as every other field.
    m = re.search(r"input\[name=\"recipe\"\]'\),([\s\S]*?)\}\);\n", js)
    assert m and "applyRecipeMode();" in m.group(1)
    assert "renderValidation(validateForm().errors);" in m.group(1)
    assert "syncPlanStaleness();" in m.group(1)
    # Locks exist on first paint, not only after the first change event.
    assert "applyRecipeMode();   // lock state matches" in js
    # The envelope carries the recipe value (not a derived boolean).
    assert "recipe: currentRecipe()," in js
    assert "recipe: snap.recipe," in js


def test_import_applies_transparent_background_without_auto_switch():
    js = _read("app.js")
    m = re.search(r"function applyImportedPlan\(plan\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "applyImportedPlan not found"
    body = m.group(1)
    assert "$('transparent').checked = c.transparent_background;" in body
    # An imported, server-validated plan is applied verbatim: no local rewrite.
    assert "onTransparencyChange" not in body
    assert body.index("$('output-format').value = c.output_format;") < \
        body.index("$('transparent').checked = c.transparent_background;")


def test_plan_panel_shows_the_transparency_row():
    html = _read("index.html")
    assert 'id="plan-transparency"' in html
    js = _read("app.js")
    m = re.search(r"function planRenderFields\(plan\)\s*\{([\s\S]*?)\n  \}", js)
    assert m and "setText('plan-transparency'" in m.group(1)
    assert "plan.config.transparent_background" in m.group(1)


def test_alpha_results_are_previewed_on_a_checkerboard_only():
    css = _read("app.css")
    rule = re.search(r"\.history-card img \{([^}]*)\}", css)
    assert rule and "repeating-conic-gradient" in rule.group(1), \
        "transparent thumbnails need a checkerboard behind them"
    assert 'data-theme="paper"' in css and ".history-card img" in css
    # Preview only: nothing in the CSS or JS writes pixels into the saved file.
    js = _read("app.js")
    assert "canvas" not in js and "toBlob" not in js
    # The transparency row has its own visible styles.
    assert ".transparency-field" in css and ".effective-prompt-text" in css


# ---------------------------------------------------------------------------
# W4 — reference images (design §2.4 / §5)
#
# The upload route takes RAW file bytes; the JSON routes cap bodies at
# 64 KiB while one reference may be 32 MiB, so base64-in-JSON is not an
# option and the tests refuse it outright. Conditioning order is the
# DISPLAYED order, so the envelope must carry the ids in list order.
# ---------------------------------------------------------------------------


def _js_function(js: str, signature: str) -> str:
    """Body of a top-level ``function <signature>(...) { ... }`` in app.js.

    The file is 2-space indented and every top-level function closes on a
    line that is exactly ``  }``; the regex is anchored there so a nested
    block can never end the match early.
    """
    m = re.search(rf"function {re.escape(signature)}\([\s\S]*?\n  \}}", js)
    assert m, f"{signature} not found in app.js"
    return m.group(0)


def test_w4_constants_mirror_the_python_caps_and_schema():
    from qwen_engine import reference_store
    from qwen_engine.config import REF_MAX
    from qwen_engine.web import api

    js = _read("app.js")
    assert f"var MAX_REFERENCES = {REF_MAX};" in js
    assert f"var MAX_REFERENCE_BYTES = {reference_store.MAX_REFERENCE_BYTES};" in js or \
        f"32 * 1024 * 1024" in js
    assert f"var MAX_REFERENCE_DIM = {reference_store.MAX_REFERENCE_DIM};" in js
    assert f"var REFERENCE_SCHEMA = '{api.WEB_REFERENCE_SCHEMA}';" in js
    # The upload route is the server's own path, relative (same-origin only).
    assert "var API_REFERENCES = '/api/v1/references';" in js
    # The id is a content digest the server also validates: lowercase only.
    assert r"[0-9a-f]{64}" in js
    assert "REFERENCE_ACCEPT = 'image/png,image/jpeg,image/webp'" in js


def test_w4_editable_keys_match_the_server_set_exactly():
    from qwen_engine.web.api import EDITABLE_FIELDS

    js = _read("app.js")
    m = re.search(r"var EDITABLE_KEYS = \[([\s\S]*?)\];", js)
    assert m, "EDITABLE_KEYS not found"
    keys = re.findall(r"'([a-z_]+)'", m.group(1))
    assert len(keys) == len(set(keys)) == 15, keys
    assert set(keys) == set(EDITABLE_FIELDS), (
        f"EDITABLE_KEYS diverge from api.EDITABLE_FIELDS: "
        f"{set(keys) ^ set(EDITABLE_FIELDS)}")
    assert keys[-1] == "references", "references is the newest (last) wire field"
    assert keys.index("recipe") == keys.index("steps") + 1, \
        "recipe sits with the sampling fields, right after steps"


def test_w4_reference_fieldset_markup_is_labelled_and_announced():
    html = _read("index.html")
    m = re.search(r'<fieldset[^>]*id="reference-field"[^>]*>([\s\S]*?)</fieldset>', html)
    assert m, "reference fieldset missing"
    block = m.group(0)
    assert "<legend>Reference images</legend>" in block
    # Plain-language help states the numbering contract and that size comes
    # from the preset, not the images.
    assert "image 1, image 2" in block
    assert "Output size comes from the preset" in block
    # File input: accepted types mirror the server's formats, multi-select,
    # hidden (the visible button opens it) — never a drag-only affordance.
    assert 'accept="image/png,image/jpeg,image/webp"' in block
    assert " multiple" in block and "hidden" in block
    for control in ("add-reference", "reference-file", "reference-list",
                    "reference-count", "reference-empty", "reference-status",
                    "reference-limit"):
        assert f'id="{control}"' in block, f"missing #{control}"
    assert 'aria-describedby="reference-help reference-status"' in block
    assert 'role="status"' in block and 'aria-live="polite"' in block
    assert 'aria-label="Reference images, in prompt order"' in block
    # The disabled Add button is never the only guard, and there is no
    # draggable attribute anywhere: reordering is button-only.
    assert "draggable" not in block


def test_w4_upload_sends_raw_bytes_never_base64_or_json():
    js = _read("app.js")
    body = _js_function(js, "uploadReference")
    assert "Content-Type': 'application/octet-stream'" in body
    assert "'X-Qwen-UI': '1'" in body
    assert "body: bytes" in body
    # The file is read as an ArrayBuffer and wrapped in a Uint8Array; the
    # page never base64-encodes it and never JSON-wraps the bytes.
    reader = _js_function(js, "readFileBytes")
    assert "readAsArrayBuffer" in reader
    assert "new Uint8Array(bytes)" in reader
    assert "JSON.stringify" not in body
    # No base64 path exists at all: no encoder, no data-URL reader.
    for forbidden in ("btoa(", "atob(", "readAsDataURL", "toDataURL", "FileReaderSync"):
        assert forbidden not in js, f"app.js must not use {forbidden}"
    # The upload is gated on a recognized status: writes need one.
    add = _js_function(js, "addReferenceFiles")
    assert "if (!apiConnected)" in add
    # Config import reuses the same bounded read path — one convention.
    limited = _js_function(js, "readFileLimited")
    assert "readFileBytes(file, MAX_IMPORT_FILE_BYTES)" in limited


def test_w4_upload_refusals_are_typed_and_bounded():
    js = _read("app.js")
    body = _js_function(js, "uploadReference")
    # Status codes reach the user as the server's own closed code pattern,
    # truncated, never a raw message of unbounded length.
    assert r"/^[a-z0-9_]{1,64}$/" in body
    assert "payload.message.slice(0, 200)" in body
    assert "setReferenceStatus(" in body
    # A schema-invalid 201 is a contract rejection: availability is dropped,
    # never rendered as a successful upload.
    assert "parseReferenceResponse(r.json)" in body
    assert "setApiAvailable(false)" in body
    # Content addressing: the same bytes are the same id, added once.
    assert "referenceIds().indexOf(ref.id)" in body


def test_w4_hard_cap_is_a_stated_stop_not_a_silent_drop():
    js = _read("app.js")
    add = _js_function(js, "addReferenceFiles")
    assert "MAX_REFERENCES - references.length" in add
    assert "files.slice(0, room)" in add
    assert "were not uploaded" in add or "not uploaded" in add
    # Room is computed BEFORE any bytes are sent: no over-cap upload can
    # even start.
    assert add.index("var room = MAX_REFERENCES - references.length") < \
        add.index("return step()")
    controls = _js_function(js, "updateReferenceControls")
    assert "references.length >= MAX_REFERENCES" in controls
    assert "$('add-reference').disabled" in controls


def test_w4_parse_reference_response_is_a_closed_shape():
    js = _read("app.js")
    body = _js_function(js, "parseReferenceResponse")
    assert body.count("return null") >= 6
    assert "Object.keys(raw).length !== 6" in body
    for field in ("width", "height", "has_alpha", "bytes"):
        assert field in body
    for cap in ("MAX_REFERENCE_DIM", "MAX_REFERENCE_BYTES"):
        assert cap in body, f"{cap} not enforced on the response"


def test_w4_parse_plan_and_job_keep_explicit_legacy_branches():
    js = _read("app.js")
    plan = _js_function(js, "parsePlan")
    # Current shape first, then the four explicitly-gated older shapes (the
    # recipe era, then references, transparency and output_format).
    assert "var want = EDITABLE_KEYS;" in plan
    assert "have.length === want.length" in plan
    for n in (14, 13, 12, 11):
        assert f"have.length === {n}" in plan, f"parsePlan misses the {n}-key branch"
    # Every legacy branch is gated on the recipe key too, so a 14-key echo
    # that HAS recipe (i.e. some other key missing) is a rejection.
    assert "!hasKey('recipe')" in plan
    assert "c.recipe = 'base';" in plan
    assert "c.references = [];" in plan
    assert "Object.keys(c).length !== want.length" in plan
    assert "Array.isArray(c.references)" in plan and \
        "c.references.length > MAX_REFERENCES" in plan
    assert "REFERENCE_ID_RE.test(c.references[i])" in plan

    job = _js_function(js, "parseJob")
    for n in (15, 14, 13, 12, 11):
        assert f"ckeys.length === {n}" in job, f"parseJob misses the {n}-key branch"
    assert "!hasKey('recipe')" in job
    assert "c.recipe = 'base';" in job
    assert "c.references = [];" in job
    assert "REFERENCE_ID_RE.test(c.references[r])" in job
    # An over-cap list is refused, never truncated.
    assert "c.references.length > MAX_REFERENCES" in job


def test_w4_reference_ids_are_validated_but_duplicates_are_legal():
    js = _read("app.js")
    plan = _js_function(js, "parsePlan")
    assert "typeof c.references[i] !== 'string'" in plan
    # config.py allows repeating an id (the same image cited twice), so the
    # parser must not de-duplicate or unique-filter the list.
    assert "new Set(" not in plan
    assert ".filter(" not in plan.split("c.references")[0]


def test_w4_import_round_trips_reference_ids_in_file_order():
    js = _read("app.js")
    body = _js_function(js, "applyImportedPlan")
    assert "references = c.references.map(" in body
    # Order is the FILE's order, and the display metadata is honestly
    # unknown (the export carries ids only).
    assert "id: id, width: null, height: null, has_alpha: null, bytes: null" in body
    assert "renderReferences()" in body


def test_w4_plan_panel_shows_the_reference_count():
    html = _read("index.html")
    assert 'id="plan-references"' in html
    js = _read("app.js")
    body = _js_function(js, "planRenderFields")
    assert "setText('plan-references'" in body
    assert "plan.config.references.length" in body


def test_w4_envelope_and_render_agree_on_order():
    js = _read("app.js")
    snapshot = _js_function(js, "formSnapshot")
    assert "references: referenceIds()" in snapshot
    envelope = re.search(r"buildEnvelope\(\)[\s\S]*?transparent_background: snap\."
                         r"transparent_background,\s*\n\s*references: snap\.references",
                         js)
    assert envelope, "buildEnvelope must forward snap.references verbatim"
    # The list is renumbered from POSITION on every render — never stored —
    # so a remove/reorder cannot leave a stale ordinal.
    render = _js_function(js, "renderReferences")
    assert "referenceCard(references[i], i)" in render, \
        "renderReferences must number cards from their list position"
    assert "'Image ' + (index + 1)" in _js_function(js, "referenceCard")
    assert "renderReferences();" in _js_function(js, "moveReference")
    assert "renderReferences();" in _js_function(js, "removeReference")
    # Every reorder/remove also bumps the draft revision via staleness.
    for fn in ("moveReference", "removeReference"):
        assert "syncPlanStaleness()" in _js_function(js, fn), f"{fn} must stale plans"


def test_w4_reorder_buttons_are_keyboard_accessible_and_end_aware():
    js = _read("app.js")
    render = _js_function(js, "referenceCard")
    assert "button.type = 'button'" in render
    assert "setAttribute('aria-label', buttons[b][2])" in render
    assert "'Move reference image ' + (index + 1) + ' up'" in render
    # The ends are honestly unavailable, not silent no-ops.
    assert "button.disabled = index === 0" in render
    assert "button.disabled = index === references.length - 1" in render
    # Focus survives the re-render (the list is rebuilt every time).
    assert "want.focus()" in _js_function(js, "focusReferenceButton")


def test_w4_submit_surfaces_reference_unavailable():
    js = _read("app.js")
    submit = _js_function(js, "submitGeneration")
    assert "r.json.code === 'reference_unavailable'" in submit
    assert "referenceNumberFromError(" in submit
    helper = _js_function(js, "referenceNumberFromError")
    # Only the ORDINAL is extracted from server text, and only when it
    # matches the documented message; anything else falls back safely.
    assert r"/^reference (\d{1,2}) is not available/" in helper
    assert "MAX_REFERENCES" in helper


def test_w4_reference_thumbnails_use_the_checkerboard_background():
    css = _read("app.css")
    rule = re.search(r"\.reference-card img \{([^}]*)\}", css)
    assert rule and "repeating-conic-gradient" in rule.group(1), \
        "alpha reference thumbnails need a checkerboard behind them"
    assert re.search(r':root\[data-theme="paper"\] \.reference-card img \{([^}]*)\}',
                     css) and "repeating-conic-gradient" in \
        re.search(r':root\[data-theme="paper"\] \.reference-card img \{([^}]*)\}',
                  css).group(1)
    for selector in (".reference-field", ".reference-list", ".reference-actions",
                     ".reference-status", ".reference-ordinal", ".reference-meta"):
        assert selector in css, f"missing {selector} styles"


def test_w4_reference_cards_wrap_instead_of_forcing_a_wider_viewport():
    """A 4-column grid over ordinal+thumb+meta+3 buttons overflowed 320px.

    The browser check measures the symptom (innerWidth widening past the
    emulated 320px); this pins the layout choice that prevents it.
    """
    css = _read("app.css")
    card = re.search(r"\.reference-card \{([^}]*)\}", css)
    assert card, "missing .reference-card rule"
    assert "display: flex" in card.group(1) and "flex-wrap: wrap" in card.group(1), \
        "reference cards must wrap; a rigid grid widens the layout viewport"
    assert "grid-template-columns" not in card.group(1)
    meta = re.search(r"\.reference-meta \{([^}]*)\}", css)
    assert meta and "min-width: 0" in meta.group(1), \
        "the metadata cell must be allowed to shrink for overflow-wrap to apply"
    actions = re.search(r"\.reference-actions \{([^}]*)\}", css)
    assert actions and "flex: 0 0 auto" in actions.group(1), \
        "action buttons keep their hit size and wrap as a unit"


def test_w4_batch_end_rerenders_so_remove_is_enabled_again():
    """Cards are rendered while uploadingReference is true.

    Only refreshing the toolbar at the batch end left every Remove button
    permanently disabled (found by screenshot, now pinned two ways).
    """
    js = _read("app.js")
    add = _js_function(js, "addReferenceFiles")
    assert add.count("renderReferences();") == 2, \
        "both the success and failure paths must re-render the card list"
    assert "button.disabled = uploadingReference" in _js_function(
        js, "referenceCard")


def test_w4_a_pruned_reference_states_the_thumbnail_is_gone():
    """A 422 names the ordinal; the card must not stay a silent broken icon.

    The card keeps its LIST POSITION (the envelope must show the order the
    user would submit), so the absence is stated on the card and in the
    aria-live status line instead.
    """
    render = _js_function(_read("app.js"), "referenceCard")
    assert "img.addEventListener('error'" in render
    assert "reference-missing" in render
    assert "thumbnail unavailable" in render
    assert "setReferenceStatus(" in render
    # `var i` is function-scoped: an error handler that reads the loop
    # variable narrates one past the end of the list. The ordinal must be
    # read from the live DOM position instead.
    # The handler BODY only: the buttons loop later in renderReferences uses
    # `i` correctly (synchronously, before any re-render).
    start = render.index("img.addEventListener('error'")
    handler = render[start:render.index("});", start)]
    assert "pos + 1" in handler
    # Parent-relative, so the card needs no closure over the render list.
    assert "indexOf.call(item.parentNode.children, item)" in handler
    assert "(i + 1)" not in handler
    assert "isConnected" in handler, "discarded cards must not narrate"
    css = _read("app.css")
    assert ".reference-card.reference-missing" in css
    rule = re.search(r"\.reference-card\.reference-missing \{([^}]*)\}", css)
    # Every theme defines --error; no theme defines a --warning token.
    assert "--error" in rule.group(1)
    for token in ("--bg:", "--text:", "--border:", "--accent:", "--error:"):
        assert css.count(token) >= 3, f"{token} must be defined in all 3 themes"


def test_w4_each_card_gets_its_own_closure_scope():
    """`var` is function-scoped: building cards inside one loop function
    left every error handler bound to the LAST card's item/img/meta.

    Each card is now built by its own function call, so the handler's
    bindings are per-card. A report of the wrong ordinal/card traced
    directly to this.
    """
    js = _read("app.js")
    render = _js_function(js, "renderReferences")
    card = _js_function(js, "referenceCard")
    # The DOM nodes are created inside referenceCard, never the render loop.
    for creation in ("document.createElement('li')", "document.createElement('img')"):
        assert creation in card, f"referenceCard must create {creation}"
        assert creation not in render, \
            f"{creation} in renderReferences re-creates the shared-scope bug"
    # The handler must not read the render loop's variable at all.
    assert "img.addEventListener('error'" in card
    assert "list.children" not in card


def test_test_surface_is_exposed_for_the_browser_checks():
    js = _read("app.js")
    surface = js[js.index("window.QwenStudio = {"):]
    # The version is a cache-buster/attribution marker, not a contract: assert
    # its SHAPE so a deliberate bump never fails this check.
    assert re.search(r"version: 'w[0-9]+-[a-z0-9-]+'", surface), "version marker missing"
    for name in ("referenceIds", "addReferenceFiles", "removeReference",
                 "moveReference", "parseReferenceResponse", "MAX_REFERENCES",
                 "parseRewriteBlock", "parseRewriteAnswer", "setRewriterState",
                 "presetForRatio", "requestRewrite", "applyProposal",
                 "keepOriginal", "rewriteState"):
        assert f"{name}: {name}" in surface, f"QwenStudio must expose {name}"


def test_w5_rewrite_ui_is_a_suggestion_flow_not_an_automatic_edit():
    """The Rewrite action must never change the draft on its own.

    Pins the four load-bearing properties (plan R4): the proposal block is a
    separate, initially-hidden region; Apply/Edit are the ONLY writers of
    ``prompt`` from the rewrite module; the fetch carries the separate long
    timeout; and an unavailable rewriter is hidden, not disabled-and-explained.
    """
    html = _read("index.html")
    js = _read("app.js")

    # -- markup: the block starts hidden and is a labelled group ----------
    block = re.search(r'<div id="rewrite-proposal".*?\n {14}</div>', html, re.S)
    assert block, "the proposal block must exist"
    block = block.group(0)
    assert "hidden" in block.split(">", 1)[0], "proposal starts hidden"
    assert 'tabindex="-1"' in block, "proposal must be focusable for a11y"
    assert 'role="group"' in block
    assert 'aria-labelledby="rewrite-proposal-heading"' in block
    for node in ("rewrite-original", "rewrite-suggested", "rewrite-ratio",
                 "rewrite-apply", "rewrite-edit", "rewrite-keep",
                 "rewrite-apply-ratio", "rewrite-error"):
        assert f'id="{node}"' in block, f"proposal must contain #{node}"
    # The button starts hidden AND disabled: an absent rewriter is not a
    # broken one, so there is nothing to press.
    button = re.search(r'<button[^>]*id="rewrite-btn"[^>]*>', html).group(0)
    assert "hidden" in button and "disabled" in button
    assert 'id="rewrite-hint"' in html
    # The help copy names the action and that it is opt-in.
    help_text = re.search(r'<p id="prompt-help".*?</p>', html, re.S).group(0)
    assert "Rewrite" in help_text and "Apply" in help_text

    # -- the ONLY prompt writes from the rewrite module -------------------
    rewrite_region = js[js.index("function setRewriterState"):
                        js.index("function apiErrorPayloadFromBytes")]
    for fn in ("applyProposal", "editProposal"):
        body = _js_function(js, fn)
        assert "$('prompt').value = proposal.prompt;" in body
    # Showing a proposal never writes the textarea.
    assert "$('prompt').value =" not in _js_function(js, "showProposal")
    assert "$('prompt').value =" not in _js_function(js, "requestRewrite")
    assert "$('prompt').value =" not in rewrite_region.split("function applyProposal")[0]

    # -- separate, longer timeout; shared one untouched -------------------
    assert re.search(r"var REWRITE_TIMEOUT_MS = \d+;", js)
    assert "REWRITE_TIMEOUT_MS" in _js_function(js, "requestRewrite")
    assert "timeoutMs: REWRITE_TIMEOUT_MS" in js
    assert int(re.search(r"var REWRITE_TIMEOUT_MS = (\d+);", js).group(1)) >= 240000
    assert int(re.search(r"var FETCH_TIMEOUT_MS = (\d+);", js).group(1)) <= 30000
    # The route constant is the only rewrite URL, and it is relative.
    assert re.search(r"var API_REWRITE = '/api/v1/rewrite';", js)
    assert re.search(r"var REWRITE_REQUEST_SCHEMA = 'qwen_engine\.web_rewrite_request\.v1';", js)
    assert re.search(r"var REWRITE_SCHEMA = 'qwen_engine\.web_rewrite\.v1';", js)

    # -- stale discard: BOTH the revision and the echoed text -------------
    req = _js_function(js, "requestRewrite")
    assert "answer.draft_revision !== promptRevision" in req
    assert "answer.original_prompt !== $('prompt').value" in req
    # A suggestion dies when the prompt text moves — but a width/steps change
    # must NOT throw it away, so the counter is its own.
    assert re.search(r"var promptRevision = 0;", js)
    assert re.search(r"var draftRevision = 0;", js), \
        "the plan/import revision must stay separate from the prompt revision"
    # syncPlanStaleness owns draftRevision; it must NOT bump the prompt one,
    # or a resolution change would silently discard a good suggestion.
    staleness = _js_function(js, "syncPlanStaleness")
    assert "draftRevision++" in staleness
    assert "promptRevision" not in staleness

    # -- no model-generated text is ever treated as HTML ------------------
    for fn in ("showProposal", "renderRewriteError", "setRewriterState",
               "requestRewrite", "bumpPromptRevision"):
        body = _js_function(js, fn)
        assert "innerHTML" not in body, f"{fn} must not assign innerHTML"
        assert "insertAdjacentHTML" not in body, f"{fn} must not insert HTML"
