# fix-complete-done — item11 correction (2026-10-09)

Agent: local Qwen (DeepSeek Harness), sole product writer. Fresh narrowly
scoped session per root's instruction; no todo_write, no vault writes
(read-only), no network/GPU/model load/dependency install. Vault
unreachable → this local handoff fallback.

## Done

- Only item11 of /tmp/v100-full-review-fixes.md: core.generate's final
  `logger.event("complete")` after a verified save could raise on a sink
  that fails only on complete (root probe: out.png saved, OSError after).
- Fix commit **28e93ee** (scoped to the two allowed files):
  - `src/qwen_engine/core.py` — completion emission wrapped in
    `except Exception: pass` like log.py's phase completion guards;
    field names are fixed non-reserved literals and all primary
    config/pipeline/save failures raise earlier, so only logging-plumbing
    faults can be swallowed; the result is still returned. Docstring synced.
  - `tests/test_cli.py` — +2 regressions: direct `generate` with a
    complete-only failing sink (saved file, result returned, earlier
    machine records intact, clean stack) and end-to-end CLI `--run` exit 0,
    real saved output, no traceback. Guard stays completion-only: existing
    all-sink-failing failing-save tests still pass unchanged.
- Prior work not repeated: review-fix1 39f8031 phase-end guards kept as-is;
  its five tests verified present exactly once and passing; duplicate-name
  check clean.

## Test evidence (this run, post-commit)

- `bash run_cpu_tests.sh tests/test_cli.py tests/test_log.py` → **63 passed**
  (baseline before this change: 61).
- Full guarded `bash run_cpu_tests.sh` → **660 passed, 1 skipped**
  (658+1 before this change).
- Root probe `/tmp/v100-complete-log-review.py` → `{"generate_result":
  "success", "saved_files": ["out.png"]}` (was OSError before the fix).

## Untouched / next

- User-owned `sandbox.yaml` left dirty-but-untouched; models/assets untouched.
- Not started (root to schedule next): items 2–10 of the full-review list
  (cache path/size hardening, run-id path hardening, fetch hard-link and
  staging-ancestor preflight, dependencies.md wording, response/cache cap
  representability, cgroup-hierarchy memory preflight, FIFO response read).
