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
  self-contained (no framework, no CDN, no external fonts/analytics, **no
  network requests of any kind**). Runnable today over `file://` or any
  static HTTP server — both are equally disconnected from any engine.
* `tests/test_web_static.py` — guarded text/structure checks (no browser, no
  network, ML-free; asserts originality-relevant invariants: no external
  origins, no `innerHTML`, BigInt seed handling with a ≤ 20-digit bounded
  grammar, absence of any endpoint/probe code, always-disabled Generate,
  `aria-invalid` wiring, no storage/service APIs in W1, exact `config.py`
  enum values compared against the real Python enums).
* `scripts/chrome_browser_check.mjs` — **real-Chrome behavioural check**,
  separate from the guarded main suite: dependency-free Node (global
  WebSocket CDP against `$CHROME_PATH`), throwaway profile, own-child
  cleanup, no screenshots written. It is **not** a guard for
  `run_cpu_tests.sh` and requires `CHROME_PATH`.
* **Not in W1 (by design):** any server, job manager, generation path,
  dependency/entrypoint/package change. Those are W2/W3 in PLAN.md. The
  packaging question (setuptools package-data for `web_static`) is deferred to
  W2, when the server that would serve it exists.

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

* The page has **no server to talk to and makes no network requests at all**:
  the earlier speculative `/api/v1/capability` probe was **removed entirely**
  in the 2026-10-12 review because it enabled Generate on a speculative
  `ready: true` while the submit path did nothing. Whether opened over
  `file://` or served by static HTTP, the page is equally disconnected —
  static hosting enforces nothing. Generate is **always disabled** with the
  visible reason "no generation adapter or job API is implemented", and the
  submit handlers enforce that directly (`preventDefault` + an honest
  "Nothing was submitted" message) — the disabled button is a *visual state*,
  not the gate. Nothing is pretend: no fake ready state, no synthetic
  results, no fake jobs, no placeholder thumbnails, no invented quality
  presets, no card/memory/asset/timing claims. Stat cards show **em dash /
  Unknown**, never 0. Result actions are disabled with no empty links. W2 may
  add a reviewed status/plan contract — never generation.
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
  decimal-**STRING** seed — the snapshot is a wire format, **not** an exported
  `GenerationConfig` JSON document. The W2 API converts a validated decimal
  to a Python int; authoritative config import/export is W2's job.
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

