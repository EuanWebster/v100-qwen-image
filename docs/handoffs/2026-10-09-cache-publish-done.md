# Cache publication + quarantine-ownership task — done 2026-10-09

Agent: Qwen (DSH), fresh sole writer. Task `/tmp/v100-fix-cache-publish-task.md`
(plus root's 10:50 added temp-collision invariant in the same file and the
review-queue note). Baseline **2531a92**. One scoped code/test commit
**7e0456d** (`src/qwen_engine/encoder.py` + `tests/test_encoder.py`); this
docs milestone carries the authorised fetch-prose correction, root's admin
note + `next-worker-contract.md`, and this handoff. No network/GPU/install/
GPU/assets/real weights; `sandbox.yaml` untouched and unstaged.

Fixed (all three defects + persistence boundaries):

1. **Publish target ownership** (`EmbeddingCache.put`): under the cache-wide
   root mutex and BEFORE any temp file is created, the target is lstat-ed.
   Absent → normal new publish; existing → replaced only if regular, singly
   linked, current-user-owned (ordinary owned overwrite, incl. corrupt-entry
   → fresh-valid replacement, which the stale-corrupt-publish race test
   depends on). Foreign-owned at any link count, own multiply linked,
   symlink and nonregular targets are refused with a typed `EncoderError`;
   name/bytes/inode/mtime/nlink preserved, no temp, no quarantine move, no
   unlink. The genuine workflow (get preserves a foreign name on miss, the
   worker path's `put` then replaced it) and the direct-put nlink decrement
   are closed. No cross-process/hostile-FS security claim; checked once near
   publish; atomic publish + LRU transaction, size preallocation refusal and
   all lifecycle/race protections unchanged.
2. **Persistence boundaries typed**: root `mkdir` and target-lstat failures
   raise `EncoderError` (file-as-root previously raised raw
   `FileExistsError`; the squatted file is neither deleted nor
   mkdir-repaired); temp cleanup runs only when this call actually created
   the temp (failed O_EXCL open leaves a pre-existing same-name file
   untouched — the deterministic-uuid collision probe row now green) and a
   failing cleanup can no longer replace the primary typed publish failure.
3. **Quarantine UID before branch** (`_quarantine`): the regular-file
   current-user ownership refusal now runs BEFORE the `read_ident` branch —
   the verified-read identity tuple omits the uid, so a regular file that
   became foreign-owned after the corrupt read was movable on the
   identity-matching path. Replaced-new-valid-entry identity protections
   unchanged.

Tests: 186 → 197 in `tests/test_encoder.py` (fake UID/transports/temp
fixtures only, no chown): direct-put refusal × {foreign single, foreign
multi, own multi, symlink, FIFO} with full preservation + no-new-artifacts;
owned-overwrite and absent-target publish unchanged; fake-worker
get→miss→put typed refusal; prior-identity + fake-UID-change quarantine
regression; file-as-root; injected cleanup-OSError; uuid-collision temp.

Evidence: root probes run READ-ONLY, unchanged —
`/tmp/v100-cache-publish-review.py` all seven rows now green (foreign_single/
foreign_linked/owned_linked/worker_miss_publication/quarantine_read_identity_
owner_change/root_creation_failure typed-preserved;
failed_exclusive_open_preserves_existing_temp name+bytes preserved after the
10:50 invariant fix — it was false in the pre-fix baseline);
`/tmp/v100-cache-ownership-review.py` and `/tmp/v100-cache-review-probe.py`
green. Guarded focused encoder suite 197 passed (3.11.15); guarded full suite
**753 passed + 1 skipped, rc 0** (was 742+1skip; +11 here). Correction
(root acceptance 2026-10-09 11:00): the earlier claim that Python 3.13 pytest
is absent in-sandbox was **wrong** — `/tmp/qwen-review.bIFHcI/venv/bin/python`
(3.13.5, pytest 9.1.1) works with `PYTHONPATH=/work/src` plus the pinned
source env vars; root's independent full runs were 753+1skip on 3.11.15 and
754 passed on pinned 3.13, and this writer later reproduced the 3.13 full
suite directly (fix-json task, 08993bc).

Fetch prose correction (docs-milestone-only, per task): the overspecific
"reverse order … hits a file where the other artifact must mkdir" direction
claim replaced with "creates a staging-file/directory conflict" in
`tools/fetch_models.py` module docstring, `_check_manifest_filename_collisions`
docstring + UsageError rationale, and the matching comment in
`tests/test_fetch_tool.py`. os.link+unlink publish description and the
conservative either-order refusal policy unchanged; no behaviour change.

Next per queue: root independent re-review of this commit; then
`/tmp/v100-fix-json-task.md`. Whole cache still NOT accepted by root.
