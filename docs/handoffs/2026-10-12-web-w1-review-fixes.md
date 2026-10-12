# Handoff — W1 review fixes (2026-10-12)

Writer: sole Qwen product writer (dsh), per `/tmp/v100-web-w1-fixes-task.md`.
Input: root's real-Chrome review (1440/375/320; stale preset state) + peer
review (no-op Generate blocker). Scope: W1 static workspace only. No amend or
rewrite of existing commits; `sandbox.yaml` left dirty-uncommitted as found;
assets/ML locks/services untouched; no installs, no network, no model loads.

## What changed (product: commit `9315cd6`, tests+script: commit `9937c8e`)

1. **Blocking — speculative probe removed.** `/api/v1/capability`, the
   `fetch` call, protocol gating and all ready/enabled logic are gone.
   W1 makes zero network requests under any protocol (file:// and static
   HTTP are both honestly disconnected). Generate is **always disabled**
   with the reason "no generation adapter or job API is implemented";
   the submit and Ctrl/Meta+Enter handlers enforce unavailability
   themselves (`preventDefault` + "Nothing was submitted…") — the
   disabled button is a visual state, not the gate. Misleading comments
   and the test promising the endpoint were removed. W2 = reviewed
   status/plan contract, never generation (PLAN.md corrected).
2. **Preset state.** Manual width/height edits recompute the exact preset
   match and `aria-pressed`: exact 1344×768 → Landscape pressed; any
   non-preset size clears all. Preview keeps its true-ratio `max-width`
   cap.
3. **Seed bounds before BigInt.** `parseSeed`: trimmed, ASCII digits only,
   **length ≤ 20 enforced before `BigInt`** (so >20-digit strings,
   including leading-zero ones, are rejected by documented policy), then
   the uint64 range. `maxlength="20"` hardens typing only. Full uint64 max
   stays an exact decimal STRING; no Number coercion.
4. **aria-invalid.** `renderValidation` sets `aria-invalid` true/false on
   steps, seed, cfg and **both** dimension inputs (shared dims error).
   No new prompt-required gate: W1 simply never enables Generate.
5. **Seed row.** Seed moved to its own full-width row (`min-width:0`,
   compact non-shrinking Random button) so all 20 digits are inspectable
   at the 380 px side panel and 375/320 mobile. Helper shortened to
   "Leave at 0 or choose Random."; precise range retained in the input
   `title` and this doc set.
6. **Hash navigation.** `tabindex="-1"` view headings focused on
   user-driven `hashchange` only — no initial autofocus. Unknown hashes
   (skip-link `#main`) no longer reroute History/Settings back to Create;
   back/forward restore views.
7. **Privacy wording.** Factual: page makes no third-party requests/fonts/
   analytics; static hosting enforces nothing; locality is enforced only
   by the planned W2 server.

## Proof (honest counts)

* Static asset tests: **8 → 13**; full guarded CPU suite **785 passed +
  1 skipped** (`bash run_cpu_tests.sh`, default python) and **786 passed,
  0 skipped** with the pinned env command (`PYTHON=/tmp/qwen-review.bIFHcI/
  venv/bin/python PYTHONPATH=/work/src QWEN_DIFFUSERS_SRC=…
  QWEN_TRANSFORMERS_SRC=… bash run_cpu_tests.sh`). Both interpreters on
  this container are **3.13.5; no 3.11 binary exists here**, so the 3.11
  leg could not be executed — recorded as not-run, not assumed. The task's
  "773/772" baseline did not reproduce here either: the pre-fix baseline on
  this box was 780+1skip (measured before any edit).
* **Real Chrome: could NOT be run green in this container.** ~390 zombie
  processes leaked by earlier sessions (parent PID 1 = `sleep`, never
  reaps) saturate the cgroup pids limit (512; ~21–40 free slots vs the
  ~45 Chrome needs), causing `pthread_create: EAGAIN` before page load.
  This predates this session (chrome processes were already ~73 min old at
  session start; killing the live ones turned them into more zombies).
  `scripts/chrome_browser_check.mjs` (dependency-free Node + global
  WebSocket CDP against `$CHROME_PATH`; tmp profile + own-child cleanup; no
  screenshots) preflights exactly this condition and exits 3 with a clear
  environment diagnosis — verified in this state. **Action for root: run
  the script on a clean sandbox and attach its output before W2.** The
  script is explicitly NOT a guard for `run_cpu_tests.sh`, which stays
  static/ML-free/socket-free (and passed as above).

## Next

Root re-review of the fixed W1 (fresh sandbox for the Chrome leg), then W2
per `/tmp/v100-web-api-task.md`: reviewed same-origin status/plan contract +
package-data; never generation.
