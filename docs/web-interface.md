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
  self-contained (no framework, no CDN, no external fonts/analytics). Runnable
  today over `file://` or any future same-origin static server.
* `tests/test_web_static.py` — guarded text/structure checks (no browser, no
  network, ML-free; asserts originality-relevant invariants: no external
  origins, no `innerHTML`, BigInt seed handling, fixed-path protocol-gated
  capability probe, no storage/service APIs in W1).
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
text, skip link, `aria-current` on the active route, `aria-pressed` on
presets.

## Honesty model (the load-bearing part)

* The page **starts and stays "Not connected"** with Generate **disabled** and
  a visible reason ("…no engine server to talk to yet… Generation not
  available."). Nothing is pretend: no fake ready state, no synthetic results,
  no fake jobs, no placeholder thumbnails, no invented quality presets, no
  card/memory/asset/timing claims. Stat cards show **em dash / Unknown**, never
  0. Result actions are disabled with no empty links.
* Capability: the *only* network call is a fixed-path, same-origin
  `GET /api/v1/capability` probe, executed **only when served over
  http(s)** — never under `file://`, never with anything derived from user
  input. Generate enables only when that endpoint reports `ready: true` (W2).
  Prompt text and form validity are then local gates on top.
* The browser checks (dimensions 64–4096 multiples of 32; steps 1–1000; finite
  CFG ≥ 1; seed grammar) are **lightweight form validation**, labelled as such
  in the UI — explicitly *not* a RAM/model fit check. The future status/plan
  backend owns authoritative `GenerationConfig` validation and runtime paths
  (Settings says so).
* No draft autosave, no storage APIs in W1 (Settings explains). Any future
  draft persistence must be opt-in and store only validated form values.

## Form model

* Prompt first: blank textarea, helpful placeholder, **nothing silently
  generated or auto-filled**; used exactly as typed (`textContent` only — no
  `innerHTML` injection paths anywhere).
* Presets: Square 1024×1024 / Landscape 1344×768 / Portrait 768×1344
  (`aria-pressed`), each driving width/height and the preview's true
  aspect-ratio frame.
* Steps default **40**; seed text field default **"0"** plus an explicit
  **Random** action.
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
  through JS `Number`. Random seeds come from `crypto.getRandomValues` (8
  bytes → exact 64 bits). Null-vs-explicit-random semantics belong to the W2
  API contract, not to invented frontend behaviour.
* `QwenStudio.formSnapshot()` produces the **UI wire format** (seed is a
  STRING) and is *not* the engine's `GenerationConfig` JSON; exporting a
  compatible config is W2's job once an authoritative serializer exists.
* Ctrl/Meta+Enter submits only when the real submit path exists and Generate
  is genuinely enabled — never an imaginary job.

## Verification evidence (2026-10-12)

Real headless Chrome (`/usr/local/bin/chrome-headless-shell` 154):
screenshots at 1440×1000 and 375×812 (`/tmp/w1-*.png`, not product assets);
scripted DOM checks (seed BigInt edges incl. `2^64−1` accept / `2^64` reject,
20 unique crypto random seeds, snapshot seed-is-string, validation rejects,
preset/aria state, preview ratio, hash routing, Generate stays disabled even
with a valid prompt) — all pass; no page console errors; mobile product page
`scrollWidth` ≤ viewport. Guarded CPU suite: **780 passed + 1 skipped** on
both 3.11 and 3.13 (baseline 772+1 plus 8 new asset tests), ML-free.
