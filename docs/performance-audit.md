# V100 optimisation audit and baseline

Audit started 2026-10-25. Remote baseline: `65d1533b44d8682e2f0182b3a9319e8b31805941`.
Work is isolated on `feature/v100-performance-memory-optimisation` in a separate
worktree. The original checkout remains at `96abea6`; its user changes and
running application are untouched. Five existing diagnostic commits were
cherry-picked, ending at `161416d`. No production optimisation is accepted yet.

## Actual architecture

The active web path is **not a fresh generation subprocess per request**.
`web/generation.py:42` calls `generate()` in the serial `JobController` worker
thread (`jobs.py:162,363`), inside the long-lived server. CLI jobs run in their
invoking process. `job_process.py` and `generation_worker.py` provide a separate
owned-process executor, but the current web executor does not select it.

1. CPU verifies local assets and pinned runtime. The adapter loads the FP16 or
   Q8 transformer onto CUDA, then an FP16 VAE, scheduler and processor
   (`adapters/qwen21_diffusers.py:265-376`). No network downloads.
2. Prompt -> fresh CPU identity-projection subprocess -> validated embedding
   cache lookup (`encoder_client.py:132`). A miss starts another CPU process,
   loads the Qwen3-VL encoder in BF16 on **CPU**, encodes, returns bounded CPU
   tensors and exits (`encoder_worker.py:207,228`). CPU BF16 is not V100 BF16.
   Encoder child termination/reaping is owned by `encoder.py:1588,1861`.
3. CPU embeddings are transferred to CUDA in the supplied adapter's denoise
   stage (`adapters/supplied_qwen21.py:204`). The VAE reference is removed and
   Python cycles collected before sampling. The transformer produces FP16
   latents. Above 4096 latent tokens the token-wise MLP uses chunks of 2048.
4. Transformer reference is removed and cycles collected. CUDA latents survive
   into decode. The VAE is loaded again: FP32 for tiled decode, FP16 otherwise.
   Tiled geometry is 1024 pixels / stride 768. No dimension substitution.
5. CUDA VAE decoding -> CPU PIL conversion -> requested JPG/PNG/WebP. Core
   drops its intermediates and invokes adapter close in a `finally` block
   (`core.py:151`). Queue serialization is one worker per controller.

Allocated tensors and allocator-reserved blocks are different. Removing a model
does not immediately return cached blocks to the driver. Existing production
code does not indiscriminately call `empty_cache()`.

## Findings (line references at baseline 161416d)

| Severity / confidence | Location | Finding and recommended action |
|---|---|---|
| Medium, confirmed overhead | `adapters/qwen21_diffusers.py:314,387,427` | VAE is loaded, discarded before sampling, then reloaded. Measure separately; evaluate deferred loading without weakening supplied-component validation. |
| High, confirmed coverage gap | `web/generation.py:42`, `core.py:151`, `adapters/qwen21_diffusers.py:475` | Active jobs share the server process, contrary to the roadmap's fresh-process description. Adapter close drops references but does not collect the final VAE's possible cycles. Audit repeated jobs before proposing retention; never claim process exit cleans each web job. |
| High, confirmed missing integration | `phase_resources.py`, `run_preflight.py`, `web/generation.py:42` | Exact memory-policy machinery exists but is not wired into production generation. CPU encoder admission is active. Do not describe the helper as a working GPU admission gate or invent evidence for arbitrary prompt lengths. |
| High, measured near-capacity risk | `adapters/qwen21_diffusers.py:87,412` | Existing square 2K tiled decode sampled 15.55 GiB and one allocator retry. Investigate 768/576, preserving old geometry until quality and fresh-process validation pass. |
| Medium, confirmed observability gap | `core.py:137`, `log.py`, `bench.py` | Five coarse wall-clock phases combine loading/cleanup and decoding/conversion. Existing calibration is a separate explicit tool. Add optional substage synchronized metrics without changing queue protocol. |
| Medium, confirmed cancellation boundary | `adapters/supplied_qwen21.py:248-270` | Decode checks cancellation around a monolithic VAE call, not between tiles. Cancellation is cooperative and may wait for decode. Do not acknowledge completion or start the next job early. |
| Low, hypothesis requiring profiling | `adapters/supplied_qwen21.py:233` | Per-step finite checks may synchronize. They are correctness safeguards, not proven waste; retain them unless a measured safe replacement is established. |
| Medium, verification pending | `adapters/supplied_qwen21.py:141` | Native Diffusers SDPA is requested; this alone does not identify the actual CUDA kernel selected by PyTorch. Profile V100 dispatch before claiming FlashAttention or another implementation. |

No confirmed numerical corruption or queue concurrency defect was found in this
initial inspection. Lifecycle/OOM recovery still requires real repeated-job tests.

## Baseline evidence

Isolated-branch offline suite: **2788 passed, 2 skipped, 24.94 s**. No failures.
Command: `PYTHON=<original>/.venv/bin/python PYTHONPATH=src bash run_cpu_tests.sh`.
Raw log retained locally at `/tmp/v100-optimisation-baseline-tests.log`.

Installed environment: Python 3.13.14; Torch 2.14.1+cu126; Diffusers
0.41.0.dev0; Transformers 5.17.0; Accelerate 1.15.0; Safetensors 0.8.0;
GGUF 0.19.0; Pillow 12.3.0; pytest 9.1.1. Model/runtime identities remain
validated by the existing pinned manifests. GPU observed: Tesla V100-SXM2-16GB,
16384 MiB, driver 580.178.04, PCIe generation 3 width 4. No GPU library imports
or weights are needed by the offline test runner.

Historical measured same-latent Q8 2048-square / 40-step / seed-42 landscape
profiles, captured before this isolated branch, are retained in the original
checkout's `output/decode-profiles-2026-10-25/landscape-2k/`. They include real
latents, original images, digests and `decode-bench.json`:

| Tile / stride | Decode profile seconds | Sampled device peak GiB | Allocator retries |
|---|---:|---:|---:|
| 1024 / 768 | 6.586 | 15.551 | 1 |
| 768 / 576 | 7.038 | 9.690 | 0 |
| 512 / 384 | 7.239 | 5.479 | 0 |

These are single diagnostic profiles including VAE loading, **not repeated
medians or end-to-end speedups**. Model/source identities are in the report;
source changes invalidate calibration reuse. Warm library caches can influence
the replay. Mean channel differences from 1024/768 were 0.502/255 and 0.754/255.
These numbers do not establish visual equivalence. Full-size quality inspection,
fresh-process confirmation, 1536-square, wide/tall shapes, repeated jobs and
FP16 comparisons remain outstanding. Weights/images/large reports stay out of Git.

The feature roadmap in `next-features-plan-2026-10-25.md` remains intact.
Transparency, Turbo, editing and rewriting are separate from this optimisation
branch. Model reuse is an experiment after instrumentation and lifecycle checks.
