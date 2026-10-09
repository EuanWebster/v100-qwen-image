# Cache ownership follow-up — done (Qwen, sole writer)

Task /tmp/v100-fix-cache-ownership-task.md, baseline d58a07e/555346d. Both
reproduced bugs fixed in d726f3d (encoder.py + test_encoder.py only).

1. Foreign-owner miss: a foreign-owned REGULAR entry was a miss but the
   generic OSError routing moved it into quarantine. get() now refuses on
   the lstat BEFORE opening — no read, no touch, no rename, evidence and
   name preserved; _quarantine re-checks lstat ownership on the
   regular-single-link branch so the refusal is never undone. Symlink /
   multi-link / nonregular squats still move aside by name (no external
   bytes/mtime/nlink change).
2. LRU: counted every entry-shaped name, followed links for mtime and
   unlinked blindly — an entry hardlinked outside was "evicted", outside
   nlink 2->1. Counting, ordering and eviction now use no-follow lstat;
   only regular, singly linked, current-user-owned names count toward the
   cap or are evicted (unsafe names use no budget, are never deleted,
   shape rechecked before each unlink). entries() docstring made honest:
   name listing, no content-validation claim.

Evidence: root reproduction /tmp/v100-cache-ownership-review.py now reports
`{"lru_hardlink": {"outside_nlink_before": 2, "outside_nlink_after": 2,
"linked_name_removed": false}, "foreign_owner_refusal":
{"refused_entry_moved": false}}` (baseline was nlink 1 / moved true);
focused guarded `bash run_cpu_tests.sh tests/test_encoder.py` 174 passed
(173 before; +1 LRU linked-name+nlink test incl. valid entries still
evicted under cap, foreign-owner test reworked to preserved-evidence
assertions). Valid hits, same-root/key concurrency, ordinary LRU and
protect-new unchanged.

Finding: the shared vault is NO LONGER read-only — `memctl handoff`
succeeded (vault commit 6e00db0, push pending credentials). Prior "vault
read-only so local fallback" notes are stale; root should re-confirm the
intended vault policy. sandbox.yaml (user edit) and models/assets
untouched. next: root re-review with independent probes; then
/tmp/v100-fix-response-task.md.
