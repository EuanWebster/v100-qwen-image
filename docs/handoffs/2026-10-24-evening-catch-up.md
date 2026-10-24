# Catch-up — working Qwen CLI and measured memory

2026-10-24, Codex. Euan asked to catch up with another model's substantial
work. This is an orientation update, not a new implementation milestone or
a full code review. The application baseline inspected was `7c5bf71`.

## What supersedes the earlier handoff

- The CLI's default `Qwen21DiffusersAdapter` now loads verified local weights
  and generates real Qwen-Image-2.1 images on the V100.
- The project's host `.venv` was rebuilt with the locked ML stack and pytest;
  it is no longer the earlier Torch-free host environment. Current inspected
  versions: torch 2.14.1+cu126, torchvision 0.29.1+cu126, Diffusers
  0.41.0.dev0, Transformers 5.17.0.
- Runs are on the **host**, using `--attest-unbounded-host` to supply explicit
  operator evidence about cgroup scope. Available RAM is still read at the
  gate. This does not establish an unbounded project-container budget.
- NVIDIA ComfyUI remains installed with its independent CUDA environment,
  but was stopped with user approval for later testing. Its service is
  currently inactive; the V100 was idle at 0 MiB used / 37°C when checked.

## Working implementation

The default loader currently supports unquantized FP16, resident transformer,
native SDPA, and a separate CPU encoder subprocess with embedding cache.
Q8/Q4 and streaming are not implemented in this loader.

Memory changes that made real generation work:

1. Phase-exclusive residency: release the VAE before sampling, then release
   the transformer before restoring the VAE for decode. `gc.collect()` is
   required to release Diffusers' module-reference cycles.
2. At 1024², untiled VAE decode is the quality default. Earlier small tiled
   decode left visible dashed artifacts.
3. Above 4096 image tokens, token-wise image MLP work is split into 2048-token
   chunks to reduce intermediate memory while retaining the mathematical
   operation.
4. Experimental 2K requires `--tiled-decode`, using 1024-pixel tiles,
   768-pixel strides and FP32 VAE execution. The earlier FP16 tiled 2K path
   produced non-finite output.

## Measured results and quality

| Configuration | Recorded wall time | Denoising | Measured device memory |
| --- | --- | --- | --- |
| 1024², 40 steps | 107–109 s across six quality cases | 42.8–43.0 s | Exact collector run: 15.247 GiB |
| 2048², 20 steps, tiled decode | 3m39s | 149.5 s | Exact collector run: 15.698 GiB |

These are recorded runs, not a matched ComfyUI Qwen comparison. Initial
warm-cache exploratory runs varied from 1m27s to 1m42s. Verification/model
loading and CPU encoding account for substantial time outside denoising.

The fixed six-prompt suite covers object fidelity, text, composition,
counting, style and lighting. All six runs exited successfully. The recorded
agent review is five passes and one partial: the lighting image lacks the
requested long shadows. Human acceptance remains pending. Determinism and
fresh/warm embedding-cache equivalence were measured for a fixed 512² case;
the 1024² object anchor also remained byte-identical across later changes.

Read:

- [Generation measurements](../generation-measurements-2026-10-24.md).
- [Fixed quality suite and review](../quality-suite-2026-10-24.md).
- [Calibration collector and policy use](../calibration-collector-2026-10-24.md).

## Calibration state

`--calibration-record PATH` collects five phase observations, sampling
device-wide NVML use by CUDA UUID and owned-process-tree RSS every 50 ms,
alongside Torch peaks and measured runtime/artifact/conditioning/cache facts.
Two real runs produced ten exact observations. The pure `assess_memory`
consumer was exercised against them, including refusal on steps/runtime
identity drift and a 1 GiB GPU reserve.

**The collector is not yet a generation preflight/job gate.** The CLI records
measurements but does not call `assess_memory` before loading/generating.
Both observed configurations fail the demonstrated policy with 1 GiB GPU
headroom; 2K leaves only about 0.30 GiB of raw device memory. Sampling can
miss sub-50ms peaks, and summed process RSS can double-count shared pages.

## Remaining integration

- Wire fresh resources and exact calibration evidence into generation
  preflight/job decisions; define a usable reserve policy without inventing
  unsupported fit claims.
- Web generation is still disabled. Its closed server/browser status
  contract still uses `adapter_not_implemented`; that explanation is stale
  now that the CLI adapter exists. Update server gate/blocker data, browser
  validator/copy and tests together, then integrate actual owned job execution.
- Complete human quality review, broader numerical/quality coverage and any
  fair Qwen-versus-ComfyUI speed comparison before claiming superiority.
- PNGs, reports and exact calibration JSON currently live in `/tmp`; they
  were present during this catch-up but are not durable across cleanup/reboot.

## What Codex checked in this catch-up

Read the new commit history, implementation/CLI/calibration paths, updated
plan and measurement records. Inspected the six-image contact sheet and
independently verified all six PNG hashes against the report. Verified the
calibration file checksum and its ten phase records, installed package
versions, GPU idle state and inactive ComfyUI service. Did not rerun heavy
generation or the full regression suite. Latest prior recorded suite:
**2532 passed, 2 skipped**.

Preserve the user-owned `sandbox.yaml` port edit. No application code,
services, models or GPU settings were changed during the catch-up.
