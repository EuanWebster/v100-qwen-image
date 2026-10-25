# Cancellation/recovery diagnostic (opt-in, real V100)

`scripts/check_generation_recovery.py` runs four serial real jobs in ONE
process with fresh single-use adapters to observe what cancellation actually
leaves behind: encoder-worker cancellation, sampling cancellation,
cancellation inside the actual `VAE.decode`, and a successful follow-up job.
The hypothesis under test is that cancellation residue (production reference
cycles, allocator state) survives job end; the tool measures, changes no
production code and makes no acceptance, fit or quality claim.

## Boundaries

Default invocation is PLAN ONLY: config parse and validation, no ML import,
no GPU access, no writes, no prompt/path echo. `--run` is explicit and the
only path that imports the CLI asset gate (`_verify_assets`), the CLI adapter
factory (`_make_adapter`) and the heavy stack. `--config` takes an existing
CLI `GenerationConfig` JSON with a fixed seed; `--output-dir` must be a NEW
benchmark directory; `--attest-unbounded-host` is forwarded verbatim. The
follow-up keeps the user's `steps` (40 for quality); cancelled jobs use 4
diagnostic steps (root observed a NaN 1-step schedule, and 2 completed steps
must be possible before sampling cancellation). A fresh full generation on
this machine is under ~30 s after encode entry (encoder request ~15.8 s, 4
steps ~4 s, decode ~0.5 s), so long fixed arm delays would miss the window
entirely. Model loading cost per scenario is expected; root executes GPU runs.

## Root run (lineage-wrapped)

    PYTHONPATH=src python -B scripts/process_lineage_owner.py -- \
      python -B scripts/check_generation_recovery.py --config CFG.json \
      --output-dir NEW_DIR --run [--attest-unbounded-host]

The owned `process_lineage_owner.py` wraps the script process so this
process's own lineage is reaped terminatingly. The diagnostic never signals
anything (no ps-wide kills), never calls `empty_cache` and never resets CUDA
peaks. The existing encoder boundary terminates and reaps its OWN children on
cancellation; the script only records its `/proc` direct children around each
event.

## Scenarios and evidence

1. `encode-cancel`: a bounded child-only **watchdog** (polls at most 0.2 s,
   hard bound 60 s) arms when the encode phase starts and fires the moment
   the live encoder WORKER child (this process's own child whose argv names
   `qwen_engine.encoder_worker` without `--project-identity`) appears, after
   a 0.5 s diagnostic settle; the stage is re-classified at fire time. If no
   worker appears before the bound it fires anyway with an explicit stage,
   so a finished job surfaces as unacknowledged instead of silently
   uncancelled. The argv classification proves only that the worker PROCESS
   started (import/load vs encoding are indistinguishable), so the strongest
   label is `cancelled-in-encoder-worker` (worker live at fire AND
   `EncoderCancelledError` in the primary's context chain — that error is
   only produced when the cancel event terminated a running worker child);
   a live `--project-identity` child gives `cancelled-in-identity-load`;
   anything else `cancelled-in-encode-window`. No "actual encoding" is ever
   claimed. A fresh private embedding cache per job makes the encode a real
   cache miss (no prompt mutation).
2. `sampling-cancel`: the normal `adapter.set_progress_callback` sets the
   SAME Event after the actual completed step 2; `core.generate` propagates
   `GenerationCancelled`. Observed step numbers are recorded.
3. `decode-cancel`: a script-local diagnostic subclass wraps
   `_restore_vae` so the cancel timer arms at actual `VAE.decode` FUNCTION
   entry. The wrapped call removes the instance-level override and its own
   closure references in a `finally` (there was no original instance
   `decode`, so removal is exact): the hook closure otherwise forms its own
   adapter→pipeline→VAE→bound-method cycle that the gc measurement would
   misattribute to production. `cancelled-in-vae-decode` requires the timer
   to have fired before the wrapped call returned.
4. `followup`: a fresh adapter completes the user's own steps; success
   requires exactly one published file AND an actual readback of its PIL
   resolution matching the configured width/height. An unreadable or
   wrong-sized image is `followup-failed`, never a silent success.

## Observation methodology

The primary exception is converted to observations (type name,
acknowledged bool, `EncoderCancelledError` context-chain bool) inside the
handler, and the diagnostic retains NO reference to it: after the try
statement exits, the traceback is released, so the AFTER snapshot and the
one explicit diagnostic `gc.collect()` measure production retention (cycles
core/adapter leave naturally), not the diagnostic holding its own traceback.
Frames are not otherwise cleared — that would be intervention. The gc's
collected count and wall cost are recorded; the cost is measured, never
assumed free. Snapshots carry CUDA allocated/reserved (current +
process-lifetime peaks), `num_alloc_retries` with actual before→after and
after→gc deltas via the existing `allocator_retry_delta`, current RSS,
lifetime peak RSS and `/proc` direct children. CUDA is read only when
`torch.cuda.is_initialized()` is already true, with one boundary
`synchronize` per snapshot (after-job observation only; no in-loop sync, no
context creation, no `empty_cache`). CUDA fields are honest nulls until then.

## Stop conditions and report

The run refuses (exit 1, fixed reason code, partial report retained) on
`cancellation-unacknowledged`, `image-published` (cancelled job's directory
not empty), `followup-failed`, `timer-not-terminated` (a cancel timer whose
thread did not terminate after cancel+join), `children-retained` (this
process's `/proc` direct children still live after a scenario — nothing is
killed) or `children-unknown` (unreadable `/proc` children: unknown is never
treated as empty). The bounded (256 KiB) prompt-free `recovery-report.json`
is written before any refusal is raised and contains only scenario outcomes,
phase/step sequences, error TYPE names, the numeric snapshots, child PID
lists and fixed evidence tokens — never config, paths or prompts.

## CPU tests

`tests/test_generation_recovery_tool.py` drives the same
`run_scenarios(...)` seam with injected fake adapters/probes/timers/gc (no
ML, no GPU) plus direct stub-driven unit tests of the watchdog poll loop and
the VAE hook install/remove contract. They cover: sequencing and cancel
ordering, mid-body vs boundary timer semantics, per-scenario timer cancel +
join + termination reporting, snapshot ordering around the measurement, and
refcount-release proof: each cancellation carries a weakref'd fixture
"tensor owner" token that may legitimately survive BEFORE the diagnostic GC
(core's own primary-frame cycle is the measured phenomenon), must be gone
AFTER the real `gc.collect()` the fixture performs, and would survive it if
the diagnostic still held the exception; conservative encoder labels (worker
label only with argv AND chain evidence), hook override removal on success
and failure paths, all stop
reasons including children/timer refusals, follow-up resolution strictness,
report prompt-freedom and no-clobber, and the plan-only no-import/no-write
gate. Fixture passes are not GPU evidence; only a root `--run` report counts
as observation.
