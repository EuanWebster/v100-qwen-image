# Performance work handoff — 25 October 2026

**Work is stopped at Euan's request. Do not resume implementation without a new
request.** This handoff records the completed batch; the wider optimisation
project and feature roadmap remain unfinished.

## Repository and running application

- Development worktree: `/home/euan/Sandbox/v100-qwen-image-optimisation`.
- Branch: `feature/v100-performance-memory-optimisation`.
- Implementation/results checkpoint: `3ba0161`; this handoff is a subsequent
  documentation-only commit on that branch.
- Draft PR: https://github.com/EuanWebster/v100-qwen-image/pull/1
- No merge or deployment. Remote `main` remains `65d1533`; original checkout
  `/home/euan/Sandbox/v100-qwen-image` remains at `96abea6`.
- Original user changes in `sandbox.yaml`, `agreed-logos/` and `out/` were left
  untouched. Do not reset/stash/overwrite them.
- App listeners checked at handoff: port8765 on localhost, LAN and Tailscale.
  URLs: http://127.0.0.1:8765, http://192.168.68.103:8765,
  http://100.98.64.62:8765. The app uses the original checkout.

## Completed in this batch

Optional synchronized stage timing and CUDA allocated/reserved/current/peak
metrics, separate CPU encoder child timings/RSS, repeat-run aggregation,
fresh/reused-process benchmarking, real attention-dispatch profiling,
same-latent VAE comparisons, saved-image comparisons and cancellation/recovery
diagnostics. Reports omit private prompt contents where implemented by the new
performance/recovery/image tools; older decode capture reports include the fixed
benchmark prompt. Do not publish private run artefacts without reviewing them.

Confirmed production defect fixed on the branch: the pinned base schedule's
single-step case produces non-finite values. `85c9d21` refuses it before model
construction. CLI asset verification retains its existing precedence.

Important commits:

| Commit | Change |
|---|---|
| `8691cee` | Optional inference/encoder-child stage metrics |
| `d2262f0` | Reviewed Qwen aggregation and repeated-run driver |
| `4032e13` | Actual denoiser attention-dispatch diagnostic |
| `85c9d21` | Early single-step schedule refusal |
| `6e78d87` | Reviewed TheHive DeepSeek image comparison tool |
| `81bd17b` | Reviewed Qwen cancellation/recovery diagnostic |
| `3ba0161` | Final evidence, regression results and test-order fix |

## Verified results

- Final guarded suite: **2863 passed, 2 skipped, 27.52s**. Skips require separate
  ML-test interpreters and external pinned source trees; they are not passes.
- Three same-process FP16/1024-square/40-step/seed42 jobs completed. Live CUDA
  allocations returned to exactly8,519,680bytes (8.125MiB) after each job.
  Reserved allocator blocks remained8.49GiB without growth.
- All three PNGs match one another and the historical FP16 anchor pixel-for-pixel.
- Cache-hit total engine time: fresh-process median71.007s (n=3), reused-process
  median68.567s (n=2). Models are still recreated per job; this is a baseline,
  not a new model-retention speedup. Existing embedding-cache miss was86.263s.
- Actual four-step FP16 denoising observed efficient SDPA dispatch. CPU profiling
  identifies ATen operators, not CUDA kernel names or valid speed comparisons.
- Fresh Q8/2048-square/40-step capture reproduced the historical reference PNG
  exactly. FP32 decode1024/768:15.565GiB sampled peak, one allocator retry;
  768/576:9.704GiB, zero retries. A separate fresh process decoding only768/576
  used9.690GiB, took6.966s including VAE load and reproduced candidate pixels.
- Candidate versus legacy geometry: mean channel delta0.502/255, max36/255.
  One landscape is insufficient to accept a new quality default. Production
  tiles remain1024/768; resolution and precision defaults were not reduced.

## Highest-priority finding for the next session

The real recovery diagnostic acknowledged encoder-worker, sampling and actual
VAE-decode cancellation, published no cancelled images, reaped owned timers and
children, and completed a normal40-step follow-up in the same process.

However, **sampling cancellation retained14,258,056,704 live CUDA bytes until
explicit diagnostic GC**, then8,519,680bytes. Decode cancellation retained
700,331,008bytes before GC, then8,519,680bytes. The diagnostic releases its own
exception reference and removes its VAE hook before measuring.

This establishes delayed release through naturally retained frames/cycles, not
an uncollectable leak. The next job was tested after explicit GC, so failure of
an uncollected next production job is not established. **Deterministic production
cleanup is not fixed.** Investigate cleanup after the exception has fully unwound
(including the serial queue boundary), then repeat real cancellation/follow-up
checks. Do not collect while deliberately retaining a traceback, or add
`empty_cache()`/GC inside sampling loops.

The active web path runs jobs in a serial **server thread**, recreating model
objects per job. It does not launch a new generation process for each web job.
A separate owned-process executor exists but is not selected by the web path.

## Evidence and commands

Ignored local evidence under the development worktree:
`output/performance-validation/`, especially `reused-process-fp16/`,
`fresh-process-fp16/`, `decode-2k-fresh/`, `decode-2k-768-isolated/`,
`recovery-fp16/` and adjacent comparison JSON reports. Preserve original images
and latents; never commit heavyweight artefacts or weights.

```bash
cd /home/euan/Sandbox/v100-qwen-image-optimisation
memctl brief v100-qwen-image-performance
PYTHON=/home/euan/Sandbox/v100-qwen-image/.venv/bin/python \
  PYTHONPATH=src bash run_cpu_tests.sh -rs
```

Use that interpreter with `PYTHONPATH=src` for this worktree. The shared virtual
environment's editable install otherwise points at the original checkout.
GPU scripts require explicit `--run`; wrap terminating GPU jobs with
`scripts/process_lineage_owner.py`. Check the live app queue/GPU ownership before
starting tests. Preserve services and never run competing GPU jobs. Saved latent
replay binds source/runtime/artifact identities: do not edit package source while
a capture/replay is running or bypass mismatches afterward.

## Agent setup and preferences

User requested visible workers in the current Herdr tab. Created panes:
`wZ:p8` (`qwen-recovery`, OMP) and `wZ:p9` (`deepseek-quality`, OpenCode), beside
parent Codex`wZ:p2`. Both workers finished and were left idle/open. IDs are a
dated snapshot: inspect live state before use.

- Qwen: `omp --model radiance/Qwen3.8 --thinking xhigh`, local port8079.
  **Euan requires XHigh for future Qwen sessions.** At most two local Qwen workers.
- Ratchet prelude was enabled using `/tmp/omp-ratchet-check.yml`; no full ratchet
  evaluation flow or approvals were initialized. It is an LLM-evaluation feature,
  not an automatic engine-code optimisation mode.
- DeepSeek: OpenCode model
  `hive-ai/deepseek-ai/deepseek-v4.1-flash-1017`. Do not substitute the quota-limited
  OpenCode free endpoint or another model without direction.
- Herdr status labels reported idle even during visible work. Check pane output,
  file timestamps, tests and actual activity regularly. The initial OMP600-second
  limit prevented a later follow-up; it was restarted with saved context at
  XHigh without that limit. No workers remain actively generating.

## Remaining work, only after the user resumes

Production cancellation/error cleanup; instrumentation-off overhead comparison;
1536/wide/tall/genre coverage and repeated2K; FP16 high-resolution acceptance;
CPU-weight cache/persistent-worker comparison; production GPU memory admission
without fabricated calibration. The accounting helper exists but is not wired.

Transparency, Turbo8-step, reference editing, multiple references and prompt
rewriting remain on the separate feature roadmap. A private28-test transparency
core exists under the original checkout's ignored feature-build workspace; it is
not integrated. Q4 stays removed at the user's request.

Read `docs/performance-results.md`, `docs/performance-audit.md`,
`docs/performance-reports.md`, `docs/generation-recovery-check.md`,
`docs/image-quality-benchmark.md` and `docs/next-features-plan-2026-10-25.md`.
Keep the PR draft until outstanding work is reviewed. Never merge automatically.
