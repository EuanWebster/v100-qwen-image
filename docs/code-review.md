# CPU code/design review — current pass complete

Codex direct review, 2026-10-13. This is a coverage/findings ledger, not a
claim that the whole project is accepted or that generation works. All prior
hardware/real-model/memory gates remain. The separate2026-10-13 vision
milestone updates dependency/lock/policy and accepts real offline processor
loading ([vision-review.md](vision-review.md)); earlier review rows describe
their historical scope. No service changes, Qwen jobs or generation
activation. Later CPU milestones add an internal tested controller
and pure memory policy, and complete the previously approved base asset fetch;
their scoped acceptance is documented in job-controller.md, memory-policy.md
and component-cpu-review.md. Current guarded suites are recorded there/in the
handoff, without relabelling historical counts as present proof.

## Corrections verified

1. **Runtime/cache identity** (`e11c370`): four direct package pins missed
   transitive runtime drift. Now validate all63locked versions/sources and
   fingerprint Python patch/full optional distribution metadata, including
   canonical complete direct_url provenance. Ambiguous duplicate names and
   corrupt/bounded inventory refused. Lock-policy consistency is tested.
   Both guarded suites1480/2skip; actual metadata6/6 and miniature worker12/12
   on each Py3.11.15/3.13.5. Generic opaque cache requires fresh production
   identity BEFORE lookup; parent/child projection boundary added in item8.
2. **Logger timing**: startup/elapsed clock failures could abort work or mask
   a primary body error despite nonfatal logging policy. Timing now becomes
   null/unavailable on ordinary failures/invalid/nonfinite values; stack and
   primary exception remain intact. Interrupts are not swallowed.
3. **Generic manifest filesystem reads**: stat-before-blocking-open could
   hang on a replaced FIFO, growing files had no initial-size read bound,
   and replaced paths could be reported verified. Canonical in-root targets
   now open via anchored no-follow directory FDs and nonblocking no-follow
   file FDs; hashing bounded by initial size+1; opened/published/path/root
   checks detect observed replacement/mutation. Deliberate stable in-root
   symlinks still work. Verification is a snapshot, not protection against
   later or arbitrary hostile simultaneous filesystem changes.
4. **Manifest input parsing**: unbounded/blocking input reads and duplicate/
   nonfinite JSON now refuse under a1MiB regular-file/identity-checked read;
   deliberate stable manifest symlinks still work. Non-string mapping keys
   produce typed format errors, not mixed-key sorting crashes.
5. **Worker diagnostics**: broad response-context wrapper relabelled model/
   inference RuntimeError/ValueError/OSError as bad output destinations.
   Only destination acquisition is wrapped now; body failures retain identity.

Corrections2–5:54new guarded regressions; both full suites1534passed+2skip
on Py3.11.15/3.13.5. Regression checks cover real fixture FIFO/replacement,
parent symlink refusal before any read, descriptor cleanup, growth read cap,
mtime-reset mutation, alias/root replacement, strict input boundaries,
nonfatal clocks and preserved worker errors. No live assets are mutated.

6. **Config input boundary**: CLI config was a blocking/unbounded file read,
   and CLI/GenerationConfig JSON silently accepted duplicate keys. The
   stdlib `_local_io` snapshot helper now supplies the same bounded regular
   no-follow/nonblocking final-FD read to config and manifest inputs. CLI
   config cap1MiB, mutation/identity checks and strict duplicate/nonfinite/
   overflowing-float JSON; invalid inputs remain exit2, never an adapter
   invocation. Stable symlinks and valid override semantics preserved.
  38new regressions; both full guarded suites1572passed+2skip each.
7. **Generic encoder/cache**: response/cache-header duplicate JSON now refuses
   at the strict decoder, preserving stderr/quarantine behavior. Mixed-key
   request mappings report typed errors. Blob-name duplicate detection is now
   linear rather than quadratic (operation-count regression). Per-key locks
   are weakly registered and retained by active callers/waiters, not forever
   by every past prompt. Root state intentionally remains process-lived so a
   retained lock stays shared even after its original cache instance is
   dropped. Cache temp cleanup now runs on interrupts as well as OSError,
   preserving the old published entry and exclusive-name ownership rules.
  21new regressions; generic encoder222/222, both full guarded suites
  1593passed+2skip and actual miniature worker12/12 each on both Python versions.
8. **Selected-child identity before production cache**: new ProductionEncoder
   completes a metadata-only seed in a fresh selected interpreter before every
   cache hit/miss. Parent never substitutes its package facts; checks request
   coherence/encoder-manifest fingerprint and wire budget. Model-root access
   and actual memory/asset gates remain in the separate loading child. Shared
   stdlib codec inspection validates hits and misses before return/store;
   full tensor values still require the consumer decoder. Harmless own
   editable dist-info+egg-info duplication is excluded from unused metadata;
   executing source still hashed; ML duplicate names still refuse.
   49new tests; both full suites1642/2skip. Actual cross-version child checks8/8
   each direction with real metadata but PRIVATE synthetic vision/payload seams;
   no real model/fit claim. Actual identity6, helper36 and miniature worker12
   passed per interpreter. See docs/encoder-client.md for contracts and limits.
9. **Memory snapshot/counter integrity**: diagnostic notes now own an immutable
   string-only snapshot; invalid supplied snapshot types refuse typed. Duplicate
   MemAvailable lines and sign-prefixed/underscored/non-ASCII cgroup counters
   are unknown, not advisory numeric guesses.19new tests; both full suites
   1661passed+2skip. Actual narrow-verifier memory-refusal checks3/3 each.
   Root-completeness refusal and exact pure budget arithmetic unchanged;
   no real-memory approval or public verifier/worker override added.
10. **Benchmark/calibration**: strict16MiB bounded snapshot input/output;
    duplicate JSON and mixed-key schema errors typed; atomic-write raw-FD/
    cleanup error handling; overflow-safe finite phase medians and exact
    integer byte medians; huge-step timing derivation; nonfatal clocks with
    explicit unmeasured phases, never partial/zero invented totals; canonical
    append I/O aligned with its lock, preserving aliases/CWD stability.
    36new tests; benchmark132, both full suites1697passed+2skip. No calibration
    evidence/production fit claim. Contracts/limits: docs/benchmark-review.md.
11. **Local bundle closed-tree preflight**: eager sorted directory listings
    could allocate unbounded unrelated entries; path-based traversal could
    follow a component replaced after inspection, and entry stat faults
    escaped the typed error boundary. Enumeration now streams through
    anchored no-follow directory FDs, checks inspected/opened/published
    directory and root snapshots, closes owned handles on refusal, and
    reports I/O faults as BundleTreeError. Stable root aliases still work.
    Eight new regressions; focused bundle suite65passed; full guarded suites
    1705passed+2skip each on Python3.11.15/3.13.5. This is a
    settled-tree structural snapshot, not byte verification or protection
    for a later loader against hostile simultaneous rewriting.
12. **Fetcher staging/transport integrity**: absolute root bound before
    callbacks; anchored no-follow directory creation and partial I/O;
    nonblocking bounded rehash with identity checks; empty partial reuse
    and same-size replacement refusal; publication from opened verified
    inode rather than a replaceable staging name; owned cleanup with honest
    outcomes and non-masking close faults; bounded ASCII Content-Range and
    declared-body span, strict byte chunks and sanitized encoding errors;
    file/directory durability reporting.30new offline regressions,
    focused fetch129passed; full guarded suites1735passed+2skip each on
    Python3.11.15/3.13.5. No real download/resume. Linux/proc-FD and
    settled-tree/concurrency limits explicit in docs/fetch-review.md.
13. **Web API/server/frontend**: strict manifest metadata JSON; owned frozen
    startup summary and closed public fields/reasons with consistent blockers;
    initial-size/cap/identity-checked startup read; typed programmatic keys;
    exactly schema-v1/nonempty frontend metadata; full editable request/plan
    echo equality before current/stale labels; idle-worker shutdown fallback
    and bounded auxiliary port conversion.25new regressions; focused web142;
    full guarded1762passed+2skip each. Real Chrome105/105 against each Python
    backend, separate static file://61/61; raw framing7 and lifecycle6 pure
    checks. Complete contracts and local-server limits: docs/web-review.md.
14. **Configured log path** (`adcb96d`): relative sinks previously followed
    later CWD changes. Bind CWD when constructed, keeping lazy writes and
    deliberate symlink/`..` filesystem semantics (no lexical collapse or
    symlink sandbox claim). Two new regressions; logger49passed, covered by
    both full suites1762/2skip.
15. **Interface-source audit** (`9a526b8`): bounded2MiB regular snapshot reads
    and typed parser/analysis recursion failures; unique direct synchronous
    definitions, typed pinned literals, exact norm/lambda/index/handle and
    dispatch/processor-list checks.33new tests; focused52/1skip; actual cached
    upstream HEADs separately verified against both pins, tracked-source
    diffs empty, checker53/53. AST/marker evidence is not behavior proof.
16. **Persistent setup and witnesses** (`482008e`, `16e0ba7`): preflight all
    fixed targets before sync, refuse aliases/nonisolated/wrong interpreters,
    add non-synchronizing --check (actual three envs passed); six assertion
    witnesses refuse -O/-OO/PYTHONOPTIMIZE before work.17+18new tests.
    The stale footer assertion left after17e5b7c's post-suite diagnostic
    polish was found by this round's full run and updated to the accurate
    pure-check disclaimer; no product/browser failure was suppressed.
17. **Browser lifecycle**: first CDP timeout/cleanup abort clears every
    pending exchange, and successful child/browser races clear losing
    timers rather than unnecessarily keeping Node alive. Nine pure checks
    pass (three new). Fresh real Chrome served105 against each Python backend
    and standalone61 pass;
    final guarded1830passed+2skip each on Python3.11.15/3.13.5.

Tooling correction contracts and evidence: docs/tooling-review.md. Current
runtime blockers and exact next implementation gates: docs/runtime-readiness.md.

Final actual pinned checks this round: identity6 and actual-memory refusal3
per interpreter; synthetic helper/codec36 and miniature four-shard worker12
per interpreter; selected-child/cache8 per cross-version direction. Private
test seams only, real vision/memory refusals restored; no real model/fit
evidence. All diagnostics completed, no background implementation jobs.

## Coverage / next work

Fully read this pass: encoder_identity + runtime policy, encoder_worker, encoder_client,
encoder_production (complete helper/codec), encoder_assets, mem_preflight, bench,
manifest, log, config, CLI, core, adapters/base, _jsonsafe, _local_io, local_bundle,
fetch tool and
generic encoder identity/request/payload, cache and worker lifecycle/response
code; complete web API/server/static JS/CSS/HTML, errors and package/entrypoint
modules. Identity/worker policy and
source/asset/codec integration were also exercised by prior CPU milestones;
that evidence does not substitute for reviewing all remaining source.

The current production-source pass and focused diagnostic/setup/browser-helper
audit are complete; this is not final project acceptance. Next resolve the
reviewed vision dependency and complete memory-budget evidence/capacity;
then actual processor/model fidelity and adapter/job implementation with
hardware-gated integration/numerics/calibration. See runtime-readiness.md.
Real processor, memory scope, reviewed vision dependency and hardware gates
remain. No real assets were rehashed or downloaded by this review.
Keep reusable review evidence and environment setup in the repo/.review-envs;
only disposable test fixtures belong in /tmp.

Known generic limits to preserve honestly: root registry scales with configured
roots and remains process-lived, not every prompt; captured child log **reads**
are bounded, but scratch stdout/stderr **writes** are not disk-quota-limited;
timeout/reaping provides the current ordinary child lifecycle. Cache/filesystem
snapshots are not guarantees against hostile simultaneous rewriting.
The local HTTP socket timeout is an idle timeout, not a total slow-trickle
request deadline; finite shutdown joins retain any still-live daemon refs.

## 2026-10-13/14 unattended integration and recovery review

Re-read the interrupted workspace and preserved the unrelated user-owned
`sandbox.yaml`. Scoped implementation/review commits through`0e80749` add:

- Config Path coercion correction; managed phase lifecycle/cleanup and
  cancellation while preserving verified saves and primary failures.
- Fixed offline generation worker/executor protocol, bounded pipes/results,
  own-group stop/reaping, late coherent-save semantics, retained retryable
  pipe/scratch ownership and replacement-peer refusal. No web activation.
- Trusted supplied Qwen21 component bridge: full-identity/value/mask decoding,
  explicit conditioning conversion, upstream sampler/VAE/normalization/PNG,
  finite checks, real callback cancellation and no-clobber output. FP32 stays
  CPU-witness-only. It loads no weights and grants no asset/resource proof.
- Runtime identity including host/CPU/affinity, source/packages, actual
  precision/thread/allocator/backend settings and bounded relevant env.
  CPU leaves CUDA workspace/binding unknown; initialized CUDA settings bind
  to exact logical-index/UUID. Actual CUDA success acceptance remains pending.
- Explicit V100 diagnostic groundwork: pinned build/profile checks, tiny
  synchronized FP16 kernel, UUID-matched typed NVML whole-device counters and
  balanced library reference cleanup. Real native/GPU success remains pending.

Read the new modules and their integration/failure boundaries. Review fixes
include post-reap scratch/pipe retention, cleanup not masking committed worker
results, explicit current-device binding and rejecting unreviewed FP32 GPU
execution. Offline tiny FP16/FP32 component witnesses run each pinned ML
interpreter; split sampler/VAE pixels exactly match the unsplit upstream path
for the same tiny random components. Default-threshold tiling is actually
entered at320px, not merely toggled. No actual base checkpoint body was loaded.
Real processor, helper36, miniature-worker12, cross-child/cache8, metadata6,
actual-memory-refusal3, CPU runtime4 and CUDA-absence2 checks pass each direction.

Final process audit found an additional browser grandchild zombie leak despite
passing CDP/direct-child checks. The resumed work implements a private Linux
subreaper wrapper, owned-descendant waiting and no hard owner kill/parent exit.
17Python regressions and10pure JS checks, real standalone61 and served105 each,
with no new zombies in controlled snapshots. Existing zombies remain and were
not globally killed/reparented. Details: [process-ownership.md](process-ownership.md).

Current guarded full suites2435passed+2skipped each on3.11.15/3.13.5.
All reusable environments/scripts/tests are persistent; ordinary disposable
fixtures alone live in temporary directories. No cloud model, Qwen harness,
subagent, driver/service/config mutation or fake generation activation.

This is CPU integration/review acceptance, **not completed image generation**.
The default local loader, resource/peak collectors/calibration and live web jobs
remain pending. Actual V100 bring-up, native ABI/numerics/VRAM/PCIe/quality
require real hardware; full CPU encoder fidelity also requires sufficient
complete applicable RAM evidence (17.53GB weights exceed observed16GiB cap).
Current blockers/sequence: [runtime-readiness.md](runtime-readiness.md).
