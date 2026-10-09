# Finish-small-docs task — done 2026-10-09

Agent: Qwen (DSH), fresh sole writer. Task `/tmp/v100-finish-small-docs-task.md`
(follow-up to `current36485`, which ended 0 prematurely after its three good
code commits — `78fd52a` bench runID guard, `79528b8` cache READ ownership
ordering, `3ba1150` honest trusted-unlimited reason — with no doc work).
Baseline **3ba1150**. Docs/comment changes ONLY: no control flow, no
manifest/asset/requirements/lock/install/source-pin changes, no
network/GPU/install/real weights; the user-owned `sandbox.yaml` untouched and
unstaged. The three code commits were NOT redone.

Completed (the task's two corrections):

1. **`docs/dependencies.md` torchvision honesty.** The old text framed
   missing `torchvision` as a hypothetical future optional need. It is now
   stated as a **known runtime blocker**: imports are clean without it
   (`Qwen3VLProcessor` + pipeline module, verified 2026-10-07), but the
   actual socket-denied local `Qwen3VLProcessor.from_pretrained` probe
   failed because its `Qwen3VLVideoProcessor` import chain requires
   `torchvision`; both pinned ML envs lack it; the tokenizer-only local
   load succeeded (Qwen2Tokenizer). Explicitly: **no** encoder parity, RAM
   fit or GPU proof. No pin/lock change made.
2. **Fetch staged-ancestor collision explanation.** The "neither manifest
   order can ever download both" claim for `a/x` + `a/x.partial/y` was
   false and is corrected everywhere it appeared (module docstring,
   `_check_manifest_filename_collisions` docstring + staged-ancestor
   `UsageError` rationale, `tests/test_fetch_tool.py` review-fix-6 comment,
   `docs/models.md` preflight paragraph, dated appendix to the committed
   `2026-10-09-fetch-staging-done.md`). Correct description: publish is
   `os.link` of the verified partial then `os.unlink` of the staging name
   (this fetcher never uses `os.replace`/`os.rename`), so an `a/x`-first
   run removes `a/x.partial` and `a/x.partial/y` can then download; the
   reverse order — or a later repair while a staging file still occupies
   that path — conflicts with the directory. The conservative both-order
   static refusal is KEPT, now described as an order-independent
   REORDER/REPAIR SAFETY policy. The plain file-vs-directory case
   (`a` and `a/b`) genuinely can never both download and its wording is
   unchanged.

NOT claimed fixed (separate queued task, per root's 10:39/10:40 notes): the
cache **publication** defects (`EncoderWorker.encode` foreign get→miss→put
replacement, direct-put nlink drop through an existing hardlink, and the
`_quarantine` `read_ident`-foreign ownership branch remaining uncovered).
`/tmp/v100-fix-cache-publish-task.md` runs before fix-json; the cache is
not accepted.

Evidence (all guarded via `run_cpu_tests.sh`, socket-denied, ML-imports
denied, writes only in temp roots; Python 3.11.15 `/work/.venv`):

- focused `tests/test_fetch_tool.py`: **99 passed** (comment-only change;
  behavior and assertions unchanged);
- ML-free import-boundary `tests/test_cpu_guard.py` + `tests/test_cli.py`:
  **53 passed** (dependencies-relevant heavy-import path);
- full suite on the final tree: **742 passed + 1 skipped, rc 0** (counts
  unchanged by this docs-only delta);
- `py_compile` of `tools/fetch_models.py` clean.

Also committed: root's in-flight `docs/handoffs/2026-10-09-review-queue.md`
administrative note as found, with the accurate completion evidence
appended. Vault untouched this run (read-only local fallback per root's
access note).

Next: root launches `/tmp/v100-fix-cache-publish-task.md`, then
`/tmp/v100-fix-json-task.md`.
