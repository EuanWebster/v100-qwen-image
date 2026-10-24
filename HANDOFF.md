# Handoff: V100 Qwen-Image project

> **Latest checkpoint: STOPPED at user request, 2026-10-24.** Read
> [the Q8 checkpoint](docs/q8-checkpoint-2026-10-24.md). Experimental Q8 passed a
> real V100 smoke; guarded CPU suite2644passed/2skipped. Full-recipe quality/2K
> benchmarks and production preflight integration remain. The app is restored.
> Pause until **2026-10-25 01:50 GMT/UTC** unless Euan explicitly resumes sooner.
> The scheduled desktop reminder does not automatically resume Codex.

> **Current state, evening 2026-10-24:** The hand-rolled CLI now produces real
> Qwen-Image-2.1 images on the V100: 1024²/40 steps in roughly 107–109 seconds,
> and experimental 2048²/20 steps in 3m39s. Exact memory observations are
> recorded; exact calibration preflight is not wired into generation. W3 now
> supports real web generation, Preview, session History, cancellation and
> measured timings. A real browser-driven 1024²/20-step run took 86.7 seconds.
> Executable `./start-web.sh` now binds localhost, LAN192.168.68.103 and
> Tailscale100.98.64.62 on8765 with one shared queue; see
> [LAN/Tailscale startup](docs/lan-tailscale.md).
> **Roadmap expanded 2026-10-24:** editing with one/up to ten references,
> native transparent output, official eight-step Turbo, all seven upstream
> aspect ratios and optional reviewed prompt rewriting are now planned in
> PLAN M6.2/M8/W4. Q8 + calibrated memory preflight remain next GPU work.
> See [the upstream/local-Qwen review](docs/upstream-feature-roadmap-2026-10-24.md).
> NVIDIA ComfyUI remains installed but stopped for engine testing.
> Read [W3 instructions](docs/web-generation-w3.md),
> [the W3 handoff](docs/handoffs/2026-10-24-w3-generation.md)
> and run `memctl brief v100-qwen-image` before continuing.
>
> The text below is the historical Oct-7 snapshot. Its "Implementation has not
> started" and hardware-absence statements apply to that date only. Engine and
> web work has since landed. See the current catch-up for the measured CLI
> results and remaining quality, preflight and web integration work.

Updated 2026-10-07 by Codex. Euan is ending the host session and will reopen
Codex inside this project's sandbox. Review and requirements interview are
complete. Implementation has not started.

## Resume here

1. Read `AGENTS.md`, then run `memctl brief v100-qwen-image`.
2. Read this handoff before acting on `PLAN.md`: the plan contains assumptions
   superseded by the review and Euan's answers below.
3. Check the sandbox's actual Python environment, model mounts, GitHub access,
   and GPU visibility. Host observations and credentials do not prove container
   availability. The project expects work inside the container at `/work`.
4. Recommended next work is to revise the plan around these findings and build
   a minimal foundation. The V100 is not needed to begin coding; it is needed
   to validate GPU execution, memory budgets, and performance.

## Confirmed requirements

- A single Tesla V100 16GB is a firm requirement at this time.
- Maximum image quality takes priority. Improve speed where possible; Euan has
  not specified a hard generation-time limit.
- Start with text-to-image and expand later. Editing and RGBA workflows are
  not required for the first usable version.
- All discussed image categories may be needed: photographic portraits,
  landscapes, illustrations, product images, and text-heavy designs. Use a
  balanced quality evaluation set rather than optimizing for one genre.
- Preserve existing services if possible. Do not assume the AMD GPUs or the
  machine's entire RAM capacity are available to this project.
- The existing draft's 1024×1024 default and optional 2K mode remain the scope;
  2K feasibility has not been demonstrated.
- Euan expects to install the V100 on 2026-10-07, using an M.2-to-PCIe 4.0 x4
  adapter. Installation was not confirmed before this handoff.
- Cooling and power are handled by Euan.

## Feasibility assessment and required plan corrections

1024×1024 is plausible with suitable precision, quantization and memory
management. No V100 generation was run, so speed, quality and 2K feasibility
remain unmeasured. The engine/CLI split, upstream model implementation,
encoder subprocess and embedding cache are sound starting points.

### Connection and weight residency

NVIDIA specifies the PCIe V100 as PCIe 3.0 x16. With a conventional direct-wired
M.2 x4 adapter, expect a maximum negotiated link of PCIe 3.0 x4, approximately
3.94 GB/s theoretical per direction before transaction overhead. This is an
inference, not a measurement of Euan's setup. Check actual link generation,
width, host-to-device throughput, and possible shared-chipset contention once
installed.

Repeated block streaming may be expensive on this connection. Establish a
numerically sound higher-precision/offloaded reference, then compare resident
higher-quality quants such as Q8 against it. Do not select Q4 or a distilled
sampler as the quality default without comparison. Keep streaming available
when it is necessary for quality or capacity.

### Upstream interfaces and dependencies

- The correct class is `QwenImage21Pipeline`, not `QwenImage2Pipeline`.
- At review time, Qwen documents a Git install of Diffusers and Transformers
  >=5.17. Resolve and pin an exact tested dependency set and model revisions.
- The plan's assertion that Volta requires PyTorch <=2.10 is wrong. Official
  CUDA 12.6 wheels retain Volta support beyond 2.10; PyTorch announced 2.14 as
  the last release with published cu126 wheels. Verify the chosen wheel's
  compiled architectures and actual CUDA execution, not only GPU detection.
- Reviewed Diffusers source has a non-Flex SDPA attention processor for 2.1,
  a Qwen21 single-file checkpoint mapping, and native PyTorch GGUF
  dequantization. Test the exact checkpoint with these before writing custom
  ComfyUI-based integration. These source features are not a proven V100 run.
- A successful ComfyUI GGUF load does not validate Diffusers execution.
  stable-diffusion.cpp is a possible alternative engine, not automatically a
  drop-in loader for a Diffusers pipeline.

### Attention, numerics and memory

- Confirmed image-token counts are 4096 at 1024² and 16384 at 2048². Prompt
  tokens, conditioning and caches also affect the memory budget.
- Move attention compatibility and FP16 stability checks before the main
  engine build. A dense FP32 attention-score tensor for 32 heads and 16384
  image tokens alone is 32 GiB. Efficient backend selection is load-bearing.
- VAE tiling reduces decode memory; it does not solve denoising-attention OOMs.
- FP32 normalization does not guarantee that all FP16 intermediate operations
  remain finite. Check representative intermediate tensors and compare
  against a higher-precision reference.
- Distinguish BF16 checkpoint storage from execution precision: accepting
  BF16 weights for conversion and rejecting BF16 checkpoints are contradictory
  requirements in the draft. The CPU encoder's dtype is a separate decision;
  V100 limitations do not force FP16 CPU execution.
- The VRAM guard needs phase/backend/conditioning-aware budgets, headroom,
  current available memory and a conservative policy for uncalibrated cases.
  A linear fit to successful runs does not guarantee that OOMs cannot occur.
- Budget available host RAM, including loading/conversion peaks and existing
  services. Do not budget against installed RAM alone.

### Sampling and validation

- The draft's generic four-step Lightning dependency is not an identified,
  validated Qwen-Image-2.1 recipe. Start with the full base sampling recipe.
- A concrete optional acceleration candidate is
  `Viggle/Qwen-Image-2.1-viggle-turbo` v0.3: six steps, prescribed sigmas and
  scheduler settings. Its card recommends keeping the runtime LoRA unmerged
  or merging in FP32 before quantization, contradicting the draft's blanket
  "merge wins" claim. Inspect the exact current recipe if selecting it.
- A non-grey PNG is not evidence of correct model execution. SSIM >=0.95
  between distilled and base-model generations is not a sound quality gate.
  Use numerical comparisons for equivalent execution paths and human
  evaluation for prompt adherence, text, anatomy, composition and detail.
- Validate the encoder's actual upstream prompt formatting, hidden-state
  extraction and masks. Matching embedding shapes alone is insufficient.

## Suggested implementation order

1. Correct the plan and resolve exact upstream versions, weights and interfaces.
2. Build configuration, CLI, model manifests/path resolution, logging, encoder
   subprocess/cache plumbing and a benchmark harness. Use meaningful,
   lightweight CPU tests. Large model smoke tests need a fresh RAM budget.
3. After installation, verify host/container GPU access, CUDA execution and
   PCIe bandwidth. Driver bring-up is in the original project scope but no
   driver installation, host configuration change or reboot occurred here.
4. Prove attention, precision and loading at small resolution, then produce
   an acceptable 1024² image using the base recipe. Measure each phase.
5. Compare resident quantization and streaming while preserving quality and
   service coexistence; calibrate the memory guard from those observations.
6. Attempt 2K and optional acceleration after the baseline is established.

## Verified session state

- Authenticated GitHub access worked from the host to both
  `The-Deflection-Engine/ltx-2.5-rocm-generator` (public) and
  `EuanWebster/ltx-build-archive` (private). Read the public `ltx_engine.py`,
  including `run_subprocess_logged`, `_embed_cache_key`, `vram_profile`,
  `fit_vram_model`, `record_vram_observation`, group offloading, and guarded
  pinned-host cache cleanup. Do not assume host GitHub auth exists in sandbox.
- Earlier read-only host checks on 2026-10-07 showed no NVIDIA device in
  `lspci`, two Radeon AI PRO R9700s plus integrated graphics, Ryzen 7 7800X3D,
  kernel `7.0.0-38-generic`, 122 GiB RAM total and approximately 43 GiB available.
  These observations can change, particularly after today's installation.
- No application code, dependency installation, model download or GPU tests
  were performed. `PLAN.md` remains the original draft, despite its stale
  "uncommitted" label: it was already committed as `df59ce7`.
- There is a pre-existing user edit in `sandbox.yaml`: host port 3082 -> 3081.
  It was preserved and is deliberately excluded from this handoff commit.
- Shared memory task slug is `v100-qwen-image`. The final state and session
  handoff point to this document for detail.

## Source references checked during review

- [Original LTX project](https://github.com/The-Deflection-Engine/ltx-2.5-rocm-generator)
- [Private LTX archive](https://github.com/EuanWebster/ltx-build-archive)
- [Qwen-Image-2.1 upstream requirements and examples](https://github.com/QwenLM/Qwen-Image-2.1)
- [Diffusers Qwen21 pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)
- [Diffusers Qwen21 transformer and attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py)
- [Diffusers GGUF documentation](https://huggingface.co/docs/diffusers/quantization/gguf)
- [PyTorch cu126 retirement announcement](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
- [Viggle Turbo model and exact recipe](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)
- [NVIDIA V100 PCIe product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/Tesla-V100-PCIe-Product-Brief.pdf)

These references were inspected on 2026-10-07; upstream main branches are not
dependency pins. No benchmark claim in this handoff implies a local GPU test.
