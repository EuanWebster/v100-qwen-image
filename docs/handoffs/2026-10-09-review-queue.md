# Senior review queue — 2026-10-09 09:49 UTC

Agent: Codex/root. Shared vault physically read-only; local handoffs substitute
until memctl can lock normally. Qwen via DSH writes all product code; root
reviews and coordinates at most three active participants. No permission
requests, real model load, network/install or GPU actions. sandbox.yaml is the
sole user-owned dirty file and is never staged/edited by project work.

Accepted completion-log fixes: phase-end guard39f8031 and final complete-event
guard28e93ee, docs9ec3613; root independently focused63passed and complete-only
failure reproduction now succeeds with out.png saved.

Cache draft committed555346d, docsd58a07e. Root independently focused173passed;
Qwen full guarded680+1skip. Original symlink/hardlink mtime and quarantine
redirection probes now safe. Root canonical-path probe now reports original
root true / redirected false / actual-root mutation-lock sharing true (probe
corrected to compare actual root, not intentionally different new root).
Cache NOT accepted yet: peer+root reproduced LRU decrementing outside hardlink
nlink2->1 and foreign-owner miss still moving entry. Fresh narrowly scoped
Qwen ownership follow-up started after exec84887 finished; use latest exec
session ID in root tool history. Task /tmp/v100-fix-cache-ownership-task.md.

Next order (one fresh headless process at a time, validate commit+tests before
acceptance; never accept exit0 alone):
1. Review ownership follow-up, independent root probes/tests.
2. /tmp/v100-fix-response-task.md: unrepresentable/giant buffered-read cap,
   nonregular/FIFO/symlink post-exit response hang; stdout/stderr excerpt read
   growth/FIFO as same transport boundary. Root2B response + sys.maxsize-1
   gives raw OverflowError; log growth fixture reads1,048,580B vs16,000 promised.
3. /tmp/v100-fix-fetch-task.md: hardlinked staging corrupts verified final,
   plus staging-path ancestor collisions; fake transport/tiny files ONLY.
4. /tmp/v100-fix-memory-task.md: inherited finite caps/parent current usage,
   unknown counters/host, malformed paths/units, zero caps, integer arithmetic;
   scoped visible Linux cgroupv2, explicit namespace/root limitations failclosed.
5. /tmp/v100-fix-small-task.md: backslash runID and honest missing-torchvision
   processor-load evidence. Requirements/model pins unchanged.
6. Independent full guarded suites3.11/3.13 and final correction review.
7. /tmp/v100-worker-tensors-task.md: ONLY lazy upstream encode + safe codec,
   separate synthetic pinned-ML tests; then request/preflight/production-worker
   wiring from /tmp/v100-m22b-worker-task.md in another fresh scoped task.
8. Review all new code, return concrete fixes to Qwen, repeat accepted loops.

Useful root scratch diagnostics /tmp/v100-{cache-review-probe,
cache-root-snapshot-review,cache-ownership-review,fetch-hardlink-review,
response-cap-review,response-index-bound-review,log-bound-review,
memory-review-probe,upstream-encoder-synthetic-probe,complete-log-review}.py.
All operate on tiny isolated temp fixtures (upstream probe synthetic ML only).

Harness reliability: correction exec49219 forgot its commit, duplicated tests,
then recovered; root Ctrl-C ended it130 after clean product tree. Cache exec6842
ended0 via provider stop with incoherent57token final, no commit/tests. Fresh
completion84887 eventually committed, tests passed, docs done. Durable tool
JSON intact; no evidence context cap caused problems. Stale reasoning/long
planning is model continuity issue. Fresh commit-sized tasks, no todo_write,
targeted reads and external outcome checks. Do not change harness/server.

Hardware/runtime gates: V100 not installed;19/24 final assets verified earlier
plus6,302,990,336B recoverable transformerpartial; DNSunavailable; torchvision
missing exactML311/313; encoder checkpoint17,534,339,488B exceeds16GiB cap.
No real-weight parity, generated image, measured model-fit or GPU claims.
Existing pinned ML envs /tmp/v100_m0_ml/venv-ml311 and venv-ml313; source dirs
/tmp/v100_m0_ml/src/diffusers and transformers. Main ML-free pytest .venv311;
3.13 /tmp/qwen-review.bIFHcI/venv/bin/python requires PYTHONPATH=/work/src
and QWEN_DIFFUSERS_SRC/QWEN_TRANSFORMERS_SRC env for external checker test.

Commit this administrative root note at the next docs milestone. It records
work/queue, not authority to broaden the task or bypass sandbox permissions.

09:54 UTC: in-flight ownership follow-up draft fixes LRU linkcount and
single-link foreign-owner refusal. Root extended ownership probe: foreign-
owned MULTIPLY-LINKED regular entry still moves because link check precedes
UID refusal. Queued tiny guard-ordering correction/test in fix-small-task.md
(separate scoped encoder commit), so response hardening can proceed first.
Do not treat complete cache ownership accepted until that final branch passes.

09:55 UTC: ownership follow-up committed d726f3d + docsd509930; root inspected
the changes and independently reran174guarded encoder tests (result below/
tool history), extended foreign-linked branch still pending fix-small task.
Response/log hardening fresh Qwen exec30717 now active. Queue unchanged.

Vault access clarification, directly verified by root: Qwen reported memctl
handoff vault6e00db0; root read that actual Git commit (7event lines). Root's
own memctl brief STILL fails Errno30 creating .git/memctl.lock. Thus access
differs by execution context, not proven 'vault no longer read-only' globally.
Do not edit vault policy or request credentials/push changes. Root continues
local handoff fallback; normal agent memory operations must follow AGENTS and
their actual permissions, never a path/tool workaround for denied access.

Response/log-boundary task done (queue item2): baseline d509930, product
commit 6351c2f + this docs milestone. Constructor rejects cap+1>sys.maxsize
incl10**100 typed; response read now chunked<=1MiB/cap+1 total on ONE
O_NOFOLLOW|O_NONBLOCK-opened fstat-validated regular descriptor (symlink
ELOOP, FIFO/dir refusal, no post-exit block, swap-race-safe, growth/sizes/
typed-errors preserved); _read_excerpt same boundary with bounded read
(old1,048,580B-vs16,000B growth-at-read fixed). Focused184 passed; guarded
full691+1skip7.7s no hangs; SIGALRM watchdogs on FIFO tests. Root probes:
cap-review→EncoderSetupError, index-bound→typed protocol error, log-bound
probe inert (patched Path.open no longer used; os.read-patch regression
added). No vault writes this run; sandbox.yaml untouched. Awaiting root
independent review before acceptance; queue otherwise unchanged.
