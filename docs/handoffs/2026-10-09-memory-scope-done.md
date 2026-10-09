# 2026-10-09 — memory preflight scope-gate correction done (review2)

Completed /tmp/v100-fix-memory-review2.md as fresh Qwen sole writer,
building on (not restarting) the interrupted writer42728 draft; its valid
hierarchy/counter/host/unit/integer fixes preserved.

**Product (9c89e7a, `src/qwen_engine/mem_preflight.py` + tests):**

1. **Missing `memory.max` at ANY visible level refuses** — ENOENT does not
   prove the controller is disabled (missing/mismapped directory possible;
   no controller evidence read). Never silently skipped; already-seen
   finite caps named truthfully in the refusal.
2. **THE NEW GATE — visible-root completeness is never assumed**: a cgroup
   namespace `/` can itself be a non-root cgroup under a tighter hidden
   ancestor, so **real autodiscovery returns unknown/refuse regardless of
   the visible root's `memory.max` contents** (present `max` or finite).
   An honest, documented fail-closed gate: the default deployment
   legitimately refuses every real load until trusted complete
   host-budget evidence exists. The visible walk still reports the
   tightest minimum `limit - current` (sibling usage included, pure
   selection helper + per-level source notes) but only as an **advisory
   visible availability — an UPPER bound** true availability that hidden
   ancestors can only tighten (root's 10:22 terminology clarification;
   required bytes are the allocation LOWER bound — the two are never
   conflated). The measured 17,534,339,488 B > visible 16 GiB case thus
   survives the gate as an explicit **guaranteed-refusal** reason.
   An explicitly injected finite/unlimited `MemorySnapshot` stays TRUSTED
   caller evidence for pure budget arithmetic/pass tests (documented);
   `budget()` never consults the advisory field. No root-is-complete
   default, no production bypass, no broad mount/namespace parser.
3. **Giant `MemAvailable`** (unit-correct 5000-digit kB) caught LOCALLY as
   unknown at CPython's int-str conversion safety limit — interpreter
   limit never raised, value never approximated.

**Evidence:** root+peer probes fully green —
`/tmp/v100-memory-scope-review.py` → missing_intermediate=refuse,
unproven_namespace_root_unlimited/finite=refuse, giant_host_counter=unknown;
`/tmp/v100-memory-review-probe.py` all four originals unchanged (refuse/
refuse/MemoryPreflightError/true). Tests 40→67: read-path tests fail closed
on unknown mount-root scope; pass/arithmetic tests use trusted injected
snapshots; hierarchy tests still demonstrate minimum EACH finite
max-current + truthful provenance notes. Guarded focused **67 passed** on
3.11.15 and pinned-source 3.13.5; guarded full on final code **732 passed
+ 1 skip** rc 0 (3.11) and **733 passed** rc 0 (3.13 pinned). Real-sandbox
read-only diagnostic (stdlib only): refuse / `cgroup_state unknown` /
advisory visible availability ≈ 5.8 GiB / guaranteed-refusal reason — the
honest gate live.

**Not done / next:** no production loader or worker calls this yet; trusted
complete host-budget evidence for real passes remains root's call; no
mount/namespace parser. No vault writes this run (local fallback);
`sandbox.yaml`, `models/assets/`, manifest and fetcher untouched. Awaiting
root independent review before acceptance.
