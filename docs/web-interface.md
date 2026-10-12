# Web interface design — local Qwen Image Studio (W milestones)

Dated 2026-10-12 (user pivot: the **web UI is the primary interface**, replacing
the planned/deferred desktop GUI; the engine and CLI are kept unchanged).
This document is the narrow design record for W1; PLAN.md §W carries the
milestone list.

## What FreeVideo is — and is not

The FreeVideo web studio (`FlashML-org/FreeVideo` `web/studio.js`,
`web/view_navigation.js`, `web/library.js`, inspected read-only by root+peer)
was used **only as a visual/UX reference**: header + two columns (left
controls / right stable preview), progressive disclosure, measured stat cards,
and a creations thumbnail history. **No FreeVideo code or assets are vendored;
there is no ComfyUI dependency anywhere in this project** — FreeVideo's
division of labour (ComfyUI owns execution) is *not* copied; our future W2/W3
server talks to our own engine only. Only T2I-relevant patterns were adapted;
video/editor/nodes/LoRA/sharing/batch/queue/automatic-download/cloud features
are out of scope by directive.

## Files and boundary

* `src/qwen_engine/web_static/index.html|app.css|app.js` — original,
  self-contained (no framework, no CDN, no external fonts/analytics). Under
  W2 the page makes exactly three kinds of request, all **same-origin and
  relative**: `/api/v1/status` (on load), `/api/v1/plan` and
  `/api/v1/config/export` (on explicit user actions). No other network code
  exists. Opening the files over `file://` still works: the single status
  fetch fails and the chip honestly says **"Not connected"** — static
  hosting remains disconnected from any engine.
* `src/qwen_engine/web/{api.py,server.py,__main__.py}` — W2 server: `api.py`
  is the **pure contract module** (schemas, validation, projections — no
  sockets, no filesystem); `server.py` is the stdlib transport (loopback bind,
  header policy, exact route allowlist, static serving, typed errors).
  Runnable as `qwen-web` or `python -m qwen_engine.web`; `--help` prints
  without binding, reading or importing ML. Generation is not implemented
  anywhere (W3).
* `tests/test_web_static.py` — guarded text/structure checks (no browser, no
  network, ML-free; asserts originality-relevant invariants: no external
  origins, no `innerHTML`, BigInt seed handling with a ≤ 20-digit bounded
  grammar, the strict same-origin endpoint allowlist, always-disabled
  Generate, `aria-invalid` wiring, no storage/service APIs, exact
  `config.py` enum values compared against the real Python enums, and the
  W2 envelope/blob/export invariants).
* `tests/test_web_api.py`, `tests/test_web_server_guard.py` — guarded
  contract/policy tests: pure dispatch on bytes (status/plan/export happy
  paths, seed grammar, hostile JSON, leak prevention), the pure transport
  policy functions, `--help`, entrypoint/packaging declarations. No sockets,
  no ML, no writes.
* `scripts/chrome_browser_check.mjs` — **real-Chrome behavioural check** over
  `file://` (dependency-free Node, global WebSocket CDP, `$CHROME_PATH`,
  throwaway profile, own-child process-group cleanup). **Not** part of the
  guarded suite; requires `CHROME_PATH`.
* `scripts/web_integration_check.mjs` — **real-socket integration check**,
  also separate from the guarded suite: spawns its OWN app-server child on an
  ephemeral 127.0.0.1 port with a tiny temp manifest, runs raw-HTTP checks
  (static, traversal/target refusal, Host/Origin/Sec-Fetch/CORS/UI-header/
  length/Expect limits, full 64-bit seed round-trip, byte-exact export,
  hostile JSON) and cleans up only its own process group. The real-Chrome
  served-UI leg is **opt-in** (`QWEN_RUN_CHROME=1` + `CHROME_PATH`).
* **Not in W2 (by design):** job manager, generation endpoints, progress,
  History wiring, anything that touches model weights. Those are W3.

## Visual hierarchy

Full-browser dark charcoal shell (`#16181b`), fine 1px borders, one warm
restrained accent (amber), high-contrast type on the system font stack.
Header: product name **Qwen Image Studio**, Create/History navigation,
Settings, and a connection chip. Desktop layout is a 380 px controls column
plus a flexible preview column; under ~940 px it stacks with no horizontal
overflow. Visible focus rings, labels for every control, `aria-live` error
text paired with `aria-invalid="true|false"` on every validated control
(prompt, steps, seed, CFG, and **both** dimension inputs for the shared
dimension error), skip link, `aria-current` on the active route,
`aria-pressed` on presets. View routing is hash-driven with `tabindex="-1"`
headings focused **only** on user-driven `hashchange` (never on initial
load); back/forward restore views — an **empty hash routes to Create** (so
Back from History after a hash-less start restores Create), while an unknown
non-empty hash such as the skip link's `#main` keeps the current view
instead of rerouting History/Settings back to Create.

## Honesty model (the load-bearing part)

* W1 had **no server to talk to and made no network requests at all** (the
  speculative `/api/v1/capability` probe was removed in the 2026-10-12 review
  because it enabled Generate on a speculative `ready: true`). W2 replaces
  that with a **strict same-origin contract**: the page connects to
  `GET /api/v1/status` and trusts it only after validating the schema tag,
  the literal booleans (`generation_available` must be exactly `false`, or
  the whole document is rejected), bounded gate/blocker shapes — a response
  it does not fully validate is reported as a typed **"Status rejected"**
  state, distinct from **"Not connected"** (no server / timeout), and never
  as ready. Every fetch carries an `AbortController` timeout; there is no
  polling. Plan and Export start **disabled** and are only offered after a
  recognized status (Check configuration doubles as the reconnect action);
  any disconnect or schema rejection takes them away again. The status can
  only ever say **planning available, generation not available**: every
  generation gate is explicitly unsatisfied (adapter not implemented,
  hardware not probed, assets not verified, calibration **not verified for
  this server** — *unknown is never rendered as absent or verified*), and
  the server's own `generation_available` boolean is displayed, never
  inferred. Generate remains **always disabled** — the W2 plan/export
  endpoints do not generate anything; the submit handlers still enforce that
  directly (`preventDefault` + "Nothing was submitted"). Plan results never
  masquerade as current: the panel is hidden while a new attempt runs,
  out-of-order answers are ignored, a failed attempt says "no planning
  result received; your draft is unchanged; nothing was generated", and a
  draft edited during a wait is labelled as a submitted snapshot. Whether
  opened over `file://` or a static server the page is still disconnected;
  locality is enforced by the `qwen-web` server binding loopback only —
  static hosting enforces nothing.
  Nothing is pretend: no fake ready state, no synthetic results, no fake
  jobs, no invented timings. Stat cards show **em dash / Unknown**, never 0.
  Plan values are rendered **only from an authoritative server answer**, via
  `textContent`, after successful validation; a failed plan/export attempt
  preserves the draft and says so.
* The browser checks (dimensions 64–4096 multiples of 32; steps 1–1000; finite
  CFG ≥ 1; seed grammar) are **lightweight form validation**, labelled as such
  in the UI — explicitly *not* a RAM/model fit check. The future status/plan
  backend would own authoritative `GenerationConfig` validation and runtime
  paths (Settings says so). The prompt check mirrors `config.py` truthfully
  (`prompt` must be a non-empty trimmed string) and is surfaced only after
  the user edits the prompt or attempts submit — never a first-load alert,
  and nothing is ever auto-filled. Generate itself stays disabled regardless.
* No draft autosave, no storage APIs in W1 (Settings explains). Any future
  draft persistence must be opt-in and store only validated form values.

## Form model

* Prompt first: blank textarea, helpful placeholder, **nothing silently
  generated or auto-filled**; used exactly as typed (`textContent` only — no
  `innerHTML` injection paths anywhere). Like `GenerationConfig` in
  `config.py`, a prompt must not be empty; the lightweight form check says
  so accessibly after editing or submit — not on first load.
* Presets: Square 1024×1024 / Landscape 1344×768 / Portrait 768×1344
  (`aria-pressed`), each driving width/height and the preview's true
  aspect-ratio frame (kept honest by the ratio-aware `max-width` cap).
  Manual dimension edits **recompute the exact preset match**: typing
  1344×768 re-presses Landscape, typing any non-preset size clears every
  `aria-pressed` — no stale preset state.
* Steps default **40**; seed gets its **own full row** (never squeezed beside
  Steps so all 20 digits stay inspectable at the 380 px desktop side panel and
  at 375/320 mobile) with text default **"0"** and an explicit **Random**
  action. The field deliberately has **no `maxlength`**: maxlength would
  silently truncate a pasted 21-digit value into a *different* valid seed;
  `parseSeed` rejects the original text instead and says so inline. Helper
  text is "Leave at 0 or choose Random."; the precise range explanation lives
  in the input's `title` (and here) — not crammed into the visible helper.
* Advanced (collapsible): negative prompt; **CFG default 1.0**; width/height;
  quantization **fp16 + q8/q4 candidates explicitly marked unvalidated**;
  offload **`resident`/`streamed`/`offload`** and attention
  **`auto`/`sdpa`/`efficient`/`eager`** — the exact `OffloadMode` /
  `AttentionBackend` values declared in `qwen_engine/config.py` (no invented
  options; nothing is remapped browser-side); VAE tiling checkbox.
* `QwenStudio.formSnapshot()` keys mirror `GenerationConfig`
  (`negative_prompt`, `true_cfg_scale`, `attention_backend`,
  `vae_tiled_decode`, …); the one deliberate difference is the UI-only
  decimal-**STRING** seed — the snapshot is a UI *display* format. The actual
  wire format is `buildEnvelope()` → `{schema, config}`: same editable keys,
  seed still a decimal string, `kind` deliberately **excluded** (it is UI
  metadata, not config). No aliases, no remapping, no runtime-path fields.
* **Seed semantics:** the full uint64 range `0–18446744073709551615` is kept
  as a **decimal string** and handled with `BigInt` internally — never coerced
  through JS `Number`. Grammar, enforced in `parseSeed` **before** any BigInt:
  trimmed, ASCII decimal digits only, **at most 20 characters — length is
  checked FIRST** — so longer digit strings are rejected on length alone,
  including ones whose leading zeros would make the value small (documented
  policy; the input keeps the user's exact text and explains the rejection —
  nothing is truncated, which is why there is no `maxlength`) — then the
  `[0, 2^64−1]` range.
  Random seeds come from `crypto.getRandomValues` (8
  bytes → exact 64 bits). Null-vs-explicit-random semantics belong to the W2
  API contract, not to invented frontend behaviour.
* `QwenStudio.formSnapshot()` produces the **UI wire format** (seed is a
  STRING) and is *not* the engine's `GenerationConfig` JSON; exporting a
  compatible config is W2's job once an authoritative serializer exists.
* Ctrl/Meta+Enter runs the same unconditional refusal as the submit event:
  validation is shown, then the handler states plainly that nothing was
  submitted — it can never pretend a job exists.

## W2 — server, contract and security model

### Running it

```
qwen-web                          # or: python -m qwen_engine.web
qwen-web --port 8901 --manifest /path/manifest.json --models-dir /path/models
```

Binds **127.0.0.1:8765** by default (never an arbitrary interface, never port
3081 — that is the review harness, not this product). `--host` accepts only
the literal `127.0.0.1`; `::1` is refused with an explicit message because an
IPv6 loopback bind needs its own actually-tested milestone. The page must be
opened at `http://127.0.0.1:<port>/` — and that URL is only reachable
*inside* the serving container: from any other machine nothing is exposed
(report the fact; the server makes no claim about why a given host is or is
not reachable). `--help` performs no bind, no read and no ML import.

Runtime paths are **server startup flags only** — `--manifest`,
`--models-dir`, `--output-dir`, `--cache-dir`, `--log-dir`,
`--benchmarks-dir`. Omitted flags mean *unconfigured*, never a silent default
write location; each given path is validated through `GenerationConfig` and
canonicalized **once at startup** (bad path → exit 2, before any bind). A
browser JSON body can never set or override any of them: any runtime-path
field appearing in a request body is a typed `unknown_config_field` refusal,
not a strip.

### GET /api/v1/status (`qwen_engine.web_status.v1`)

`planning_available: true`, `generation_available: false`, the four explicit
gates above, their union as `generation_blockers` (plus a `manifest_*` code
when the manifest is missing/invalid), and a **metadata-only** manifest block
(configured / valid / schema_version / artifact_count /
`verification: "not_performed"` / typed reason when not valid). The status
path never calls `resolve_manifest` or `manifest_status` — verifying weights
means stream-hashing gigabytes, which no UI poll may ever do — and never
opens a weight file, loads a processor or imports torch.

### POST /api/v1/plan (`qwen_engine.web_plan_request.v1` → `qwen_engine.web_plan.v1`)

Exact envelope `{schema, config}` — extra or missing envelope keys are
refused. `config` carries **exactly** the editable subset of
`GenerationConfig`: `prompt, negative_prompt, width, height, steps,
true_cfg_scale, seed, quant, offload, attention_backend, vae_tiled_decode`;
anything else (including any runtime path) → typed refusal. The body merges
with the server's startup paths and shares `GenerationConfig.from_dict` —
one validator, not two. **Seed wire format:** a decimal **string** (`null` =
server-chosen later) of 1–20 ASCII digits, parsed to an int and range-checked
to uint64 — never a JSON float, never `Number`, never JS-rounded; the plan
and status responses keep the seed as that decimal string. Internally the
server calls the existing `core.plan` on the merged config, but returns a
**web projection** — the raw core document (which contains runtime paths and
a numeric seed) is never emitted. The projection adds `cfg_active`,
`latent_tokens`, the honest manifest state (`fit: "not_evaluated"`,
`ready: false`, `verification: "not_performed"`) and the unknown hardware /
calibration records.

### POST /api/v1/config/export

Same validated envelope; responds with an attachment
(`qwen-editable-config.json`) whose bytes are the canonical JSON of exactly
the editable fields — runtime paths **omitted** (an exported machine's
folders are not part of a portable config; import will be designed with them
explicitly), and the seed an **exact uint64 JSON number** (Python serializes
it losslessly; browsers would not). The browser downloads the raw attachment
as a `Blob` and never `JSON.parse`/reserializes it. Import is deliberately
**not** implemented yet.

### Hostile-input policy

Duplicate JSON object keys (at any depth) → `duplicate_json_key`; `NaN` /
`Infinity` literals and overflow floats (`1e999`) → typed refusal; nesting
beyond the shared depth limit → typed; int literals that cannot be parsed →
typed; lone surrogates in any string → `invalid_text`; invalid UTF-8 →
`invalid_utf8`. Every error is `qwen_engine.web_error.v1` with status
400/403/405/411/413/417/422/500, a short message and field list — never a
traceback, never a filesystem path, never an internal module name.

### Transport hardening

Literal-loopback bind; **one** `Host` header that must name the bound literal
(or the `localhost` alias) **and the actual bound port** — userinfo,
whitespace, leading-zero ports, over-long digit ports, extra values and any
other name are refused (digit strings are length-bounded **before** any
`int()` conversion, so an arbitrary-length digit header is a typed 400, never
a `ValueError` → 500; no DNS is consulted, which is also why `same-site` is
denied: another port on loopback would be same-site but not same-origin).
Every route — including method refusals (`OPTIONS`/`HEAD`/`PUT`/`DELETE`/
`PATCH`) — passes the Host and Origin/site policy **first**; every method
that then lands on a known route with the wrong method gets a 405 with a
correct `Allow` header and **no CORS headers are ever emitted** (OPTIONS
grants nothing). A foreign `Origin` or a cross-site/cross-origin/same-site
`Sec-Fetch-Site` is denied with 403 on every route. Every POST requires the
exact `X-Qwen-UI: 1` header (a cross-origin form post cannot set it without a
preflight, and no preflight is granted) and an exact `application/json`
content type (optionally `; charset=utf-8`; anything else — form encodings
included — is refused 415 **before the body is read**).
`Content-Length` must appear once as plain ASCII digits with no sign, space or
leading zeros, and is checked against the 65536-byte cap **before any body
byte is read**; `Transfer-Encoding` (incl. chunked) → 400; **every `Expect`
extension, including `100-continue`, is refused 417** — the server never
emits an early 100 before Host/Origin/length validation and never reads an
unsupported Expect body. Finite socket/read timeouts apply before header
parsing; every response closes the connection. Routing uses the **raw wire
request target** (Python's own parser folds a leading `//` in the normalized
path — the raw target is checked too, so a real `//x` request is a typed 400,
never accidentally served) and is an exact origin-form allowlist — absolute
targets, query strings, fragments, any percent-encoding, dot segments and
unknown routes are refused (400/404) and **no path is ever translated to a
filesystem path**; there is no directory listing. Requests are served by a
**bounded pool of persistent worker threads** (four workers, small bounded
queue, lazy start, typed 503 when saturated, serial fallback if the box
cannot start a thread at all) — no thread is ever spawned per connection.
Responses carry `Cache-Control: no-store` (API), `nosniff`,
`Referrer-Policy: no-referrer`, CORP `same-origin`, `X-Frame-Options: DENY`
and a CSP of `default-src 'none'; script-src 'self'; style-src 'self';
connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none';
form-action 'self'`; JSON serialization uses `allow_nan=False`. Static assets
are three literal routes with fixed MIME types. A configured runtime path
that stops validating **after** startup (e.g. a dir later replaced by a
file) produces the generic `server_runtime_configuration` error — never a
message embedding the private path. Manifest unreadability reasons are a
closed code set (`manifest_unreadable` etc.); arbitrary caller strings never
reach the wire.

## Verification evidence

**2026-10-12 (W1 original, commit `89041f9`)** — real headless Chrome 154:
screenshots at 1440×1000 and 375×812 (`/tmp/w1-*.png`, not product assets);
scripted DOM checks (seed BigInt edges, unique crypto random seeds, snapshot
seed-is-string, validation rejects, preset/aria state, preview ratio, hash
routing, Generate gated only by the then-present capability probe) — all
pass; no console errors; mobile `scrollWidth` ≤ viewport. Guarded CPU suite
780 passed + 1 skipped.

**2026-10-12 (review fixes, this update)** — guarded CPU suite on this
container: **786 passed + 1 skipped** (baseline 780+1 plus 6 new/expanded
asset tests; 14 asset tests total) with `bash run_cpu_tests.sh` (default
interpreter), and **787 passed, 0 skipped** with the pinned-env invocation
(`PYTHON=/tmp/qwen-review.bIFHcI/venv/bin/python PYTHONPATH=/work/src
QWEN_DIFFUSERS_SRC=… QWEN_TRANSFORMERS_SRC=… bash run_cpu_tests.sh` — the
one previously-skipped test runs under the pinned sources). Both
interpreters on this box are Python 3.13.5; **no Python 3.11 binary exists
here**, so the "3.11" leg of the standing protocol could not be executed on
this container and is recorded as not-run rather than assumed. ML-free,
socket-free, writes only under a fresh temp root.

Real-Chrome behaviour: `scripts/chrome_browser_check.mjs` is shipped and
runs its own checks (seed grammar incl. length/non-ASCII, snapshot keys,
preset↔manual sync, ratio, nav/focus/`#main`, aria-invalid, always-disabled
with zero HTTP probes, overflow at 1440/375/320), **but it did not pass
green in this container at fix time**: the sandbox's cgroup pids controller
(512) is saturated by ~390 unreaped zombie processes leaked by earlier
sessions (their parent PID 1 is `sleep` and never reaps), so Chrome fails
`pthread_create` before loading the page. The script now detects exactly
this and exits 3 with an honest diagnosis instead of hanging — an
environment problem, not a product failure. The static suite encodes the
same invariants as machine checks. The browser script must be run (and its
output attached) on a clean sandbox before W2 acceptance.

