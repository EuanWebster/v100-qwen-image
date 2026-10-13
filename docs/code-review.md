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
   identity BEFORE lookup; parent/child projection boundary remains pending.
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

## Coverage / next work

Fully read this pass: encoder_identity + runtime policy, encoder_worker,
manifest, log, config, CLI, core and adapters/base. Identity/worker policy and
source/asset/codec integration were also exercised by prior CPU milestones;
that evidence does not substitute for reviewing all remaining source.

Next confirmed gap: CLI config uses an unbounded blocking read; both config
JSON entrypoints accept duplicate keys. Fix input boundaries without changing
override/default semantics. Then production parent/child identity-before-cache
API and review the generic encoder transport/cache (read only partial so far),
production helper/codec, encoder assets, memory preflight, local bundle,
benchmark schemas/persistence, fetch tool, web API/server and real-browser UI.
Keep reusable review evidence and environment setup in the repo/.review-envs;
only disposable test fixtures belong in /tmp.
