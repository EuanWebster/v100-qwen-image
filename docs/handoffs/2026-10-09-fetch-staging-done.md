# Fetch staging integrity (review items 5+6) — done 2026-10-09

Agent: Qwen (DSH), sole product writer. Task `/tmp/v100-fix-fetch-task.md`
(root queue item 3): `/tmp/v100-full-review-fixes.md` items 5 (HIGH
resume-through-hardlink corruption) and 6 (staged-path ancestor collision).
Baseline `6490ad2`; product commit **b680404**; docs/models.md on top.
Fake tiny transports/fixtures only — no network, GPU, real asset changes or
installs. Vault untouched per task (read-only local fallback); root's
in-flight `docs/handoffs/2026-10-09-review-queue.md` edit left unstaged as
found; the sole user-owned dirty `sandbox.yaml` untouched.

Completed:

- `_resume_offset` refuses `st_nlink > 1` BEFORE any read/transport/write —
  empty, partial-length and byte-complete stagings refuse identically
  (exit 6, `UnsafePathError`); every name and byte (and mtime/nlink)
  preserved, nothing published. Root's `/tmp/v100-fetch-hardlink-review.py`
  now reports rc 6, 0 requests, `verified_final_changed false`,
  victim `b"prefix"`.
- `_open_partial` rechecks `st_nlink` on the OPENED FD after `fstat` and
  before `ftruncate`/`lseek`/`write`; the race regression links the partial
  from inside the fake transport's `get()` (between preflight and open) and
  both names stay `b"prefix"`, no final.
- Atomic publish unchanged and still valid: its temporary second link is
  created only after the full SHA256 rehash; failed-unlink debris is inert
  (a later fetch refuses multiply-linked partials; removing the stale
  `*.partial` name loses nothing while the final stands). Documented in
  the tool docstring and `docs/models.md`.
- `_check_manifest_filename_collisions` now also refuses (usage error, exit
  2, BOTH plan and fetch, either manifest order, before mkdir/network/
  write) a staged path landing on another artifact's ancestor directory
  (`a/x` + `a/x.partial/y`). No schema change; normal siblings and the
  committed 24-entry manifest stay accepted (read-only acceptance test).

Evidence: focused `tests/test_fetch_tool.py` 99 passed (+9 functions/+5
param cases); `test_manifest.py` + `test_local_bundle.py` 114 passed;
guarded full suite `run_cpu_tests.sh` rc 0 — 705 passed + 1 skipped
(691+1 before). Real interrupted partial untouched (6,302,990,336 B,
nlink 1, Oct 7 mtime); `models/` otherwise untouched; sandbox.yaml
preserved.

Next: root independent review/reproduction before acceptance; queue item 4
(memory preflight cgroup hierarchy) follows per root's queue.
