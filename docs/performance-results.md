# Performance and reliability results (work in progress)

2026-10-25. Development branch:
`feature/v100-performance-memory-optimisation`. Remote baseline `65d1533`.
Existing application checkout `96abea6` remains unchanged. This report describes
an ongoing engineering pass, not completion of the entire optimisation plan.

## Implemented and verified

Optional stage instrumentation (`8691cee`), repeated-run aggregation and a
benchmark driver (`d2262f0`), and a real denoiser attention-dispatch probe
(`4032e13`). Metrics are disabled by default and contain no prompt contents.
Allocator live/reserved counters are separate; CPU encoder subprocess timings
and PID memory are distinct from the generation process. The existing encoder
cache, FP16/Q8 modes, output formats, dimensions and queue/API schemas remain.

A confirmed single-step schedule defect is refused early (`85c9d21`): the pinned
base scheduler emitted an invalid divide and sampling returned non-finite values
in a real V100 run. The adapter now returns a `steps` configuration error before
reading model weights/importing ML. Dry-run configuration remains compatible;
normal 40-step generation is unchanged. Regression: 90 focused tests passed.

## Repeated FP16 baseline and lifecycle

Same cube prompt as the existing FP16 anchor, 1024 square, seed42, 40 steps,
no negative CFG, PNG, original untiled decode, same model/runtime environment.
The reused-process mode creates and closes models for every job; it is **not**
model or CPU-weight retention. Measurements below are total engine job times
after the first CLI asset verification, not shell startup-to-exit times.

| Mode / embedding cache | Runs | Median seconds | Range seconds | Population std dev |
|---|---:|---:|---:|---:|
| Fresh Python process, hit | 3 | 71.007 | 70.976–71.275 | 0.134 |
| Reused Python process, hit | 2 | 68.567 | 68.430–68.704 | 0.137 |
| Reused Python process, miss | 1 | 86.263 | 86.263 | 0 (single sample) |

This establishes a baseline distinction between process and embedding-cache
warmth; it is not a demonstrated new optimisation. Three reused-process jobs
returned to exactly **8,519,680 bytes (8.125 MiB) live CUDA allocations** after
adapter cleanup. Reserved memory remained 9,116,319,744 bytes (8.49 GiB), without
growth. Cached allocator blocks alone are not leaked tensors. This small success
series does not establish cancellation/OOM recovery or long-duration stability.

All three reused-process images were **pixel-identical** to one another and to
the pre-instrumentation FP16 anchor (maximum and mean channel delta0). Raw
reports/images remain outside Git in the development worktree's
`output/performance-validation/{reused-process-fp16,fresh-process-fp16}/`.
The anchor is in the original checkout's
`output/q8-validation-2026-10-25/fp16-1k/`.

In the two reused-process cache-hit runs, median model initialisation was24.386s,
transformer loading7.210s, encoder request0.123s, sampling42.223s, VAE reload0.716s
and decode0.477s. Initialisation includes nested loading and verification; these
durations overlap and must not be summed. A redundant initial VAE load measured
about0.33s plus0.1s disposal; removing it would also require changing validated
component attachment. No such change was accepted for a speculative sub-1% gain.

## Actual V100 attention dispatch

The four-step diagnostic (FP16, 1024 square, native/auto attention) completed and
observed `aten::_scaled_dot_product_efficient_attention`160times, alongside
`aten::scaled_dot_product_attention`160times. This identifies efficient SDPA
operator dispatch in the real denoiser. CPU activity profiling does not trace
CUDA kernel names, and its overhead makes those times unsuitable as benchmarks.
Evidence: `output/performance-validation/attention-dispatch-4step.json`.
No attention backend or dependency change was introduced.

## 2K decode evidence and remaining acceptance

Historical Q8/2048 square/40-step/seed42 mountain-lake latents were decoded with
three FP32 geometries in one process:

| Tile / stride | Profile seconds, including VAE load | Sampled device peak GiB | Allocation retries |
|---|---:|---:|---:|
| 1024 / 768 (production reference) | 6.586 | 15.55 | 1 |
| 768 / 576 | 7.038 | 9.69 | 0 |
| 512 / 384 | 7.239 | 5.48 | 0 |

768/576 saves roughly5.86GiB in this diagnostic at an added0.45s. Mean absolute
channel delta was0.502/255, maximum36/255. The 1024 replay matched the original
generation PNG exactly. A 1:1 crop of detailed tree/mist boundaries showed no
obvious seams, but one scene is insufficient to accept a new quality default.
Warm library caches can affect replay results; a fresh-source capture, isolated
candidate replay and broader shape/genre review are still required. Production
geometry remains1024/768. There is no silent resolution/precision reduction.

## Open work and rollback

Cancellation/failed-job cleanup, repeated2K and 1024/1536/2048/wide/tall geometry
coverage, instrumentation-off overhead comparisons, CPU-weight retention and
persistent-worker alternatives remain under investigation. Production GPU
memory admission is still not wired; the accounting helper is diagnostic
groundwork, not an active admission guarantee. Fresh worker fallback remains
available through the CLI/process executor; the current web path uses a serial
server thread with fresh model objects per job, not a generation process per job.

Run the full guarded offline suite after final integration. Baseline was2788pass,
2skip; an intermediate instrumented run was2828pass,2skip. Later focused checks
passed, but that intermediate full count is not a final release result.

Remove measurement flags for the original default execution path. Revert
`85c9d21` to remove the early one-step refusal; revert the instrumentation/tool
commits to remove diagnostics. No model weight, dependency, driver, AMD service
or default tile/precision change needs rolling back. The branch is independently
reviewable and must not be merged automatically.
