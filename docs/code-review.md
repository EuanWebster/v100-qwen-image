# CPU code/design review — in progress

Codex direct review, 2026-10-13. This is a coverage/findings ledger, not a
claim that the whole project is accepted or that generation works. All prior
hardware/real-processor/memory/torchvision gates remain. No model downloads,
dependency/lock changes, service changes, Qwen jobs or generation activation.

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

## Coverage / next work

Fully read this pass: encoder_identity + runtime policy, encoder_worker, encoder_client,
encoder_production (complete helper/codec), encoder_assets, mem_preflight, bench,
manifest, log, config, CLI, core, adapters/base, _jsonsafe, _local_io and
generic encoder identity/request/payload, cache and worker lifecycle/response
code. Identity/worker policy and
source/asset/codec integration were also exercised by prior CPU milestones;
that evidence does not substitute for reviewing all remaining source.

Next: fix confirmed benchmark issues before acceptance: blocking/unbounded
result/calibration reads, duplicate JSON and mixed-key mapping errors, median
overflow/byte rounding, phase-timing faults masking body exceptions, canonical
calibration append lock paths disagreeing with lexical I/O after CWD changes.
Then finish local_bundle, fetch tool, web API/server and real-browser UI.
Keep reusable review evidence and environment setup in the repo/.review-envs;
only disposable test fixtures belong in /tmp.

Known generic limits to preserve honestly: root registry scales with configured
roots and remains process-lived, not every prompt; captured child log **reads**
are bounded, but scratch stdout/stderr **writes** are not disk-quota-limited;
timeout/reaping provides the current ordinary child lifecycle. Cache/filesystem
snapshots are not guarantees against hostile simultaneous rewriting.
