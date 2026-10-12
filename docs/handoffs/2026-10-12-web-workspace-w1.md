# W1 web workspace done — static Create+History+Settings shell + pivot docs

Agent: Qwen (dsh, sole product-code writer). 2026-10-12. Baseline `0a8c13e`
(docs) / `08993bc` (JSON), both root-accepted. Task file:
`/tmp/v100-web-workspace-task.md`.

Completed:
* Original static web workspace `src/qwen_engine/web_static/{index.html,
  app.css,app.js}` — FreeVideo-inspired layout only (header + left controls /
  right stable preview, stat cards, creations history); no FreeVideo
  code/assets, no ComfyUI. Honest "Not connected"/disabled Generate, BigInt
  decimal-string seeds with `crypto` random, presets, advanced progressive
  disclosure, lightweight form validation only (server remains the
  authoritative validator). No server/jobs/generation (W2/W3), no
  dependency/entrypoint/package changes, no storage APIs, no drafts.
* `tests/test_web_static.py` — 8 guarded ML-free text/structure tests.
* Root in-flight fix applied within session (per
  `docs/handoffs/2026-10-12-web-pivot-progress.md`): offload/attention selects
  now expose the exact `config.py` enum values (`resident/streamed/offload`,
  `auto/sdpa/efficient/eager`), snapshot keys mirror `GenerationConfig`
  (`negative_prompt/true_cfg_scale/attention_backend/vae_tiled_decode`) with
  the UI-only decimal-STRING seed; no invented option values, no suggestion
  semantics.
* Pivot docs: PLAN.md (Goal, architecture row for `web_static/`, W milestones,
  open-question 4 resolved, execution note) + `docs/web-interface.md`.
* Commits the two pre-existing root admin handoffs — the Turbo assessment
  (`docs/handoffs/2026-10-10-turbo-assessment.md`) and the 2026-10-12
  web-pivot progress note — **as found** (untracked → tracked, content
  unchanged, per directive).

Verification: guarded full suite 780 passed + 1 skipped on Python 3.11.15
(.venv) and 3.13.5 (uv cached env, offline) — baseline 772+1 skip plus the 8
new tests; ML-free import invariant held. Real headless Chrome 154
(`$CHROME_PATH`): screenshots 1440×1000 and 375×812 + scripted DOM checks via
a /tmp harness page (details in docs/web-interface.md) — all pass, no page
console errors, no mobile horizontal overflow, preview ratio geometrically
exact. Screenshots lived only under /tmp.

Preserved as found: `sandbox.yaml` (still the one dirty non-task file, host
3081 — untouched, uncommitted), `models/assets/`, ML locks, services.

Next: root visual review of /tmp screenshots (or re-shoot on request) →
W2 tiny same-origin capability server (also owns packaging package-data and
GenerationConfig export) → W3 generation wiring, gated on engine milestones.
The worker-tensor engine milestone remains next after this frontend pivot,
unchanged and not discarded.
