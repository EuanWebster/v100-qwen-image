# finish-cache-done — cache draft completion (2026-10-09)

Agent: local Qwen (DeepSeek Harness), sole product writer. Fresh narrowly
scoped session per /tmp/v100-finish-cache-task.md; no todo_write, no vault
writes (read-only), no network/GPU/model load/dependency install/real
assets. Vault unreachable → local handoff fallback.

## Done

- Task: finish the interrupted exec6842 cache draft (base HEAD 9ec3613,
  dirty encoder.py was the previous draft, kept as-is) — no redesign.
- Commit **555346d** (scoped to the two allowed files):
  - `src/qwen_engine/encoder.py` — the one required root correction:
    `_canonicalize_dir` resolves the configured path **once** at
    construction (`os.path.realpath`) and stores that canonical absolute
    path for BOTH file I/O and the lock registry, so a later retarget of
    an explicitly chosen root symlink can no longer move cache writes to
    a new root under the OLD root's mutation lock. Typed blank/NUL/
    invalid-path errors and no-mkdir-at-construction preserved. Comment
    wording fixed: entry vs base64-response sizes differ by header JSON;
    the shared 256 MiB default is a policy bound, not a mathematical
    guarantee. Rest of the draft (bounded reads, ownership refusals,
    contained quarantine dir, absolute paths) unchanged; response code
    unchanged; no hostile-concurrent-filesystem claims anywhere.
  - `tests/test_encoder.py` — +12 test functions / +20 cases: external
    symlink and hardlink entries → miss with target bytes/mtime untouched;
    external quarantine-dir symlink → refused, evidence stays at the
    entry name and the listing remains accurate; FIFO squat → refused
    without blocking; fstat ownership branch (geteuid shim) → miss;
    `max_entry_bytes` validation matrix (bool/float/str/≤0/10**100/
    sys.maxsize refused, sys.maxsize-1 accepted); exact-cap roundtrip;
    cap+1 entry → miss + quarantine with ZERO parser calls; over-cap put
    → typed refusal before any buffer construction (`to_entry_bytes`
    never called, nothing written); root-symlink retarget regression
    (original resolved root receives the entry, new root nothing,
    same-root instances share the lock); relative-CWD cache/scratch
    stability. No duplicate test function names.
- Independent verification of the correction: root's
  /tmp/v100-cache-root-snapshot-review.py now prints
  `original_root_received_entry=true, redirected_root_received_entry=false`
  (same-root lock sharing asserted in the new regression test; the probe's
  third field compares two genuinely different roots post-fix).

## Test evidence (this run, post-commit)

- `bash run_cpu_tests.sh tests/test_encoder.py` → **173 passed** (153 before).
- `bash run_cpu_tests.sh` (guarded full) → **680 passed, 1 skipped**
  (660+1 at HEAD~1, verified by collect-only in a throwaway worktree).

## Untouched / next

- `sandbox.yaml` user edit left dirty and unstaged; models/assets untouched.
- Root's note `docs/handoffs/2026-10-09-stage1-senior-review.md` updated
  with actual progress (that note's earlier 09:37 root edit committed as
  found in this docs commit).
- Next per root's queue: /tmp/v100-fix-response-task.md,
  /tmp/v100-fix-fetch-task.md, /tmp/v100-fix-memory-task.md; then
  worker-tensors, then stdlib worker wiring. Cache corrections await
  root's review.
