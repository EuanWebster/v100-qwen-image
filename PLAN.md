# PLAN: v100-qwen-image

Hand-rolled Qwen-Image-2.1 generation pipeline for a Tesla V100 16GB, structured like
the LTX-2.5 ROCm build (engine / front-end split, subprocess encoder, VRAM calibration).

Status: **uncommitted draft, awaiting Euan's approval.**

---

## Goal

Generate images with Qwen-Image-2.1 at 1024×1024 (2K opt-in) on a single 16GB Volta
card, with a headless CLI front-end and a VRAM guard that refuses configs before they
OOM.

---

## Current context / assumptions

### From Euan (answered 2026-10-06, this session — these ARE the spec)

1. **"Hand-rolled" = LTX-style engine, diffusers underneath.** Asked how deep the
   hand-rolling goes, he asked "what will give the best results?" — the answer is
   option 1, and this plan adopts it. Reasoning, so the decision is on the record:
   generation quality is set by the weights, the quant, the numerics and the sampler —
   **not** by who writes the DiT forward. The block-causal attention mask is the single
   highest-risk component to hand-roll (a subtly wrong mask produces plausible garbage,
   not an error); diffusers implements it correctly and its attention is mathematically
   identical. Hand-rolling the forward buys zero image quality and a large bug surface.
   What IS hand-rolled, because that's where the LTX engineering paid off: the engine
   layer — streaming offload, subprocess encoder, embedding cache, VRAM fit model, CLI.
2. **Hardware bring-up is in scope as Milestone 0** ("Yes — driver install + container
   GPU passthrough are Milestone 0").
3. **1024×1024 default, 2K opt-in and slow** ("drives the VRAM budget and whether
   tiling is mandatory" — accepted as written).

### From other user-supplied material

- **The LTX-2.5 ROCm build** (`The-Deflection-Engine/ltx-2.5-rocm-generator`, private
  archive `EuanWebster/ltx-build-archive`). Patterns ported verbatim in spirit, read
  from `ltx_engine.py` (2860 lines) this session:
  - `run_subprocess_logged` — encoder/prompt-enhancer run in a throwaway subprocess so
    they cost no VRAM during generation.
  - `_embed_cache_key` — embedding cache keyed on (model, prompt, negative).
  - `vram_profile` / `fit_vram_model` / `record_vram_observation` — measured VRAM
    calibration: fit peak-GB against effective tokens, refuse configs that won't fit.
  - Streaming block offload + the lesson that `torch.cuda.empty_cache()` frees zero
    pinned host memory (`torch._C._host_emptyCache()` does).
  - Engine never imports the UI. Keep that.
- **Model facts** (model card + third-party writeups, **verify at first download**,
  see M1): Qwen-Image-2.1 released 2026-09-20. DiT: 7B, 32 single-stream layers,
  block-causal attention (token-level causal for text, chunk-level for image blocks,
  KV-cached across denoise steps), 14.2 GB bf16. Text encoder: Qwen3-VL 8B, 17.5 GB
  bf16. VAE: RGBA, 64-channel, 16× spatial compression, ~0.7–1.4 GB depending on
  source. Native 2K, seven aspect ratios.

### Verified against the box (2026-10-06)

| Fact | Check that proved it |
|---|---|
| V100 NOT in the box yet; no NVIDIA driver installed | `lspci` → 2× AMD Navi 48 + Raphael iGPU only; `nvidia-smi` not found; no `/proc/driver/nvidia` |
| Host is a ROCm box (2× Radeon AI PRO R9700 run vLLM) | `lspci` + `vllm serve` process on :8080 |
| 122 GB RAM, ~946 GB free on `/` | `free -h`, `df -h` |
| `~/models` exists, 312 GB used — weights go here | `du -sh ~/models` |
| Sandbox container: Python 3.13.5, **no torch yet** | `podman exec … python3 -c "import torch"` → ModuleNotFoundError |
| Container reaches host `:8079` on `127.0.0.1` | `curl` inside container → HTTP 200 on `/v1/models` |
| ComfyUI exists at `~/ComfyUI` with Qwen-Image blueprints | `ls ~/ComfyUI/blueprints` — reference/Plan B only, this plan does not use it |

### Where I corrected / extended Euan's answers

1. **Quant = GGUF Q4_K_M, not FP8.** All "16GB → FP8 sweet spot" advice assumes modern
   cards. V100 (sm_70) has **no FP8 and no bf16 hardware**. FP8 storage-quant (cast per
   layer to fp16) is possible; Q4_K_M + torch-fallback dequant is the tested path on
   Volta. Blocks: M4.
2. **Volta numerics: fp16 only, with fp32 guards.** bf16 checkpoints are cast to fp16;
   layer-norm/statistics stay fp32 to dodge fp16 overflow. This is a known-good ComfyUI
   pattern, ported here. Blocks: M5.
3. **PyTorch pinned to cu126 ≤2.10.** CUDA 12.8+ wheels deprecated sm_70, 2.11 dropped
   it, CUDA 13 has no sm_70 kernels at all. The pin is load-bearing: a future
   `pip install -U` silently produces a torch that cannot see the card. Blocks: M0.
4. **Text encoder does NOT use Qwen's official low-VRAM builds** (`int8_convrot`,
   `w4a8`): they need Triton INT8 kernels, which crash/corrupt on sm_70. The encoder
   runs bf16→fp16 **on CPU** inside the subprocess, or GPU GGUF Q4 if measured faster.
   Blocks: M3.
5. **Torch-version choice forces the container to diverge from the dsh image** — the
   project image is an LLM-harness chain; torch cu126 (~2.5 GB) goes in a project venv
   at `/work/.venv`, not baked into the chain. Blocks: M0.

---

## Architecture

`qwen_engine.py` owns the pipeline (load → encode-cache → denoise → VAE decode → save)
and imports nothing UI-related. `cli_gen.py` is the only front-end in v1 (Tk GUI
deferred — YAGNI). Components: Qwen3-VL encoder in a subprocess (host RAM, outside the
VRAM budget), 32-layer DiT resident as Q4 GGUF dequanted to fp16 with streaming block
offload as the fallback mode, VAE + decode on GPU with tiled decode for 2K. A
`vramfit.py` module records (tokens, peak GB) observations into `calibration.json` and
answers "does this config fit 16 GB?" **before** the run starts.

---

## Milestones & tasks

Conventions: work happens **inside the container on `/work`** (`enter.sh
v100-qwen-image`), one git commit per task, tests with `pytest` (add as a pinned dev
dep up front — LEM lesson: unpinned `.venv` installs vanish on folder moves).

### M0 — Hardware bring-up (host, then container)

> Euan said cooling and power are sorted. Software only.

- **0.1** Pre-flight: confirm Secure Boot state (`mokutil --sb-state`) and that the
  kernel is 7.0.0-38. Decide signing vs MOK enrollment vs `--disable-signing`.
  *Verify:* command output recorded in `docs/bringup.md`.
- **0.2** Install the NVIDIA driver **proprietary branch supporting Volta** (driver
  580.x is the last series with sm_70 support — pin it, don't take `*-server-open`;
  open kernel modules never supported Volta). Reboot.
  *Verify:* `nvidia-smi` lists the V100 16GB; `nvidia-smi -q -d TEMP` under
  `cuda_samples` load stays < 85 °C (proves the cooling claim, per his note).
- **0.3** Container GPU access: install `libnvidia-container1` + toolkit, generate the
  podman CDI spec (`nvidia-ctk cdi generate`). Add `device: nvidia.com/gpu=all` (or the
  CDI device) to `sandbox.yaml`, `projctl up v100-qwen-image`.
  *Verify:* `podman exec proj-v100-qwen-image nvidia-smi -L` prints the V100.
- **0.4** Project venv + pinned torch: `uv venv /work/.venv && uv pip install
  torch==2.7.* --index-url …/whl/cu126` + `diffusers`, `transformers`, `accelerate`,
  `sentencepiece`, `pytest` (dev pin). Add `requirements.lock` to the repo.
  *Verify:* `python -c "import torch; print(torch.cuda.get_device_capability(0))"` →
  `(7, 0)`; `torch.cuda.get_device_name(0)` → Tesla V100.

### M1 — Weights

- **1.1** Download to `~/models/qwen-image-2.1/`: DiT bf16 (~14.2 GB), Qwen3-VL 8B
  encoder bf16 (~17.5 GB), VAE. Check hashes against the model card; record actual
  filenames + sizes in `docs/models.md` (**the model-facts table above is unverified
  until this task runs**).
  *Verify:* `du -sh` matches the card ±5 %.
- **1.2** GGUF Q4_K_M quant of the DiT with city96's `convert-gguf.py` (llama.cpp
  fork for diffusion); fp16 conversion of encoder + VAE. Script `scripts/fetch_models.py`
  makes the whole thing idempotent (skip-if-present + hash check).
  *Verify:* Q4 file ~4.6 GB; loads via ComfyUI-GGUF's loader class in a one-line probe.

### M2 — Engine skeleton

- **2.1** `qwen_engine.py`: config dataclass (prompt, size, steps, seed, quant, offload
  mode, device paths) + `resolve_model_path` + logging. No model code yet.
  Test: config round-trips JSON. **TDD, commit.**
- **2.2** `latent_tokens(width,height)` + `effective_tokens` ported from LTX (single
  latent grid here — no temporal term; spatial /16 per the 2.1 VAE — confirm against
  diffusers' `QwenImage2Pipeline` at first read).
  Test: 1024² → 4096 tokens; 2048² → 16384. **Commit.**
- **2.3** `cli_gen.py`: argparse front-end, calls `engine.generate(config)`; prints the
  LTX-style phase log. *Verify:* `--dry-run` prints the resolved plan, generates nothing.
- **2.4** First smoke run on **CPU** (fp16 weights via `low_cpu_mem_usage`, 512², 4
  steps with Lightning LoRA) — proves weights, mask and pipeline end-to-end with zero
  driver variables. *Verify:* a non-black, non-grey PNG (test: RGB std-dev > 5).
  This de-risks M5: by then, only the GPU path is new.

### M3 — Encoder subprocess (LTX's `run_subprocess_logged` pattern)

- **3.1** `subprocess_entry` port: parent writes request JSON → child loads Qwen3-VL
  (fp16, `device_map=cpu`) → encodes → writes embed tensor + mask to
  `cache/` → exits. Parent never pays its VRAM. Test: round-trip a prompt, shapes match
  `QwenImage2Pipeline`'s expected `prompt_embeds`. **Commit.**
- **3.2** `_embed_cache_key` port (model hash, prompt, negative, kwargs);
  cache in `cache/embeds/`, LRU cap 64. Negative-prompt-must-not-pollute-key bug:
  port the fix, test both key variants. **Commit.**
  *Note:* 2.1 KV-caches text/image conditioning across steps anyway; the disk cache
  still saves ~40 s/run of CPU encoding — measure, keep if >10 s.

### M4 — GPU path: quant + offload

- **4.1** GGUF load path on GPU: Q4 weights resident, dequant to **fp16** per module —
  the Triton kernel must be disabled/fallback on sm_70 (the known Volta landmine).
  Spike task: time both fallback paths, keep the faster in `dequant_hook.py`.
  *Verify:* 1024² × 4 steps under, say, 4 min (calibrate; record baseline).
- **4.2** Streaming block offload as fallback mode (`offload=stream`): 32 layers
  ping-ponging, `pin_memory` groups, `torch._C._host_emptyCache()` on teardown (the LTX
  lesson, verbatim). *Verify:* peak GPU < 8 GB with the encoder on CPU; output
  pixel-identical to resident mode at fixed seed (tolerance: max abs diff < 1e-2 fp16).
- **4.3** `vramfit.py`: port `record_vram_observation` / `fit_vram_model`;
  `fits(tokens, quant, mode) -> bool|None` (None = no data yet, proceed with warning).
  TDD on the fit math. **Commit.**

### M5 — Sampling & numerics

- **5.1** fp16 cast at load + fp32 layernorm/statistics guard; negative test: bf16
  tensors must raise at load (`bf16_checkpoint_available()` ported, returns False and
  the loader refuses). **Commit.**
- **5.2** Lightning LoRA 4-step path: `merge` vs `add` measured, merge wins (fixed
  weights, zero per-step cost). Steps/CFG defaults table in `presets.py`
  (1024²/4-step quality-preset, 2K opt-in). *Verify:* SSIM ≥ 0.95 vs 40-step at 3 seeds
  on the same 3 prompts.
- **5.3** Attention backend: xformers or torch SDPA (`--use-pytorch-cross-attention`
  equivalent); benchmark the three, set default from measurement, not folklore.

### M6 — Resolution modes & guard

- **6.1** 1024 default; aspect-ratio presets from the card's table; 2K presets gated
  behind `--hi-res` with VAE tiled decode (`--tiled-decode`, tile=512 default).
  *Verify:* 2K run peak < 15.5 GB or the run is refused with the fitted-GB estimate.
- **6.2** VRAM guard wired into the CLI: before load, `fits()` decides
  resident/stream/refuse. Test: a fabricated 2K-resident config → refuse, exit 3.

### M7 — Bench, calibrate, document

- **7.1** Benchmark matrix (1024/2K × resident/stream × LoRA/no-LoRA × 3 seeds) →
  `benchmarks/`, feeds `calibration.json`. **The thresholds in 4.x/6.x are starting
  values; this task sets the real ones.**
- **7.2** `README.md` (what fits, what it costs, commands) + `docs/bringup.md` (V100
  driver/CDI trail so the next Volta box is a copy-paste).

---

## Tests / validation

- Full: `cd /work && .venv/bin/pytest -q` (expect ≥1 test per numbered commit task;
  count lands during the build — the plan's contract is *every* task has one).
- E2E golden: 3 fixed prompts × seed 42 at 1024²; store reference PNGs after the first
  human-approved run; CI check = SSIM ≥ 0.98 vs reference (catches silent mask/numerics
  regressions — the failure mode this whole plan is allergic to).
- Negative: bf16 checkpoint → refuses; 2K-resident config → refuses; missing V100 →
  clean error naming the CUDA pin, not a stack trace.

## Risks, tradeoffs, open questions

| Risk | Mitigation |
|---|---|
| GGUF dequant Triton kernels break on sm_70 (known Volta landmine) | 4.1 spikes both fallback paths before anything is built on top; Plan B = stable-diffusion.cpp's GGUF loader as the quant backend |
| fp16 overflow in DiT (no bf16 on Volta) | fp32 stats (5.1); golden-PNG SSIM test catches drift |
| Driver 580.x vs kernel 7.0 / Secure Boot | 0.1 pre-flight before touching anything |
| CUDA 13-era `pip -U` silently kills the card | versions locked in `requirements.lock`; loader asserts `capability==(7,0)` at startup |
| 2K attention activations exceed 16 GB | tiling + guard (M6); 1024 default per spec |
| diffusers `QwenImage2Pipeline` is young code | 2.4 CPU smoke run validates it before any GPU work; upstream-pin in lockfile |
| Model-card component sizes are third-party claims | 1.1 replaces the table with measured numbers |

**Open questions (none block M0):**
1. LoRA quality-vs-4-step tolerance is a taste call — M7.1 outputs go to Euan for the
   golden set. (Blocks 7.2 only.)
2. Whether dsh-UI/GUI front-end ever happens — deliberately out (YAGNI; CLI per spec).

## Execution notes

*(empty — appended during the build, one line per deviation from this plan, per the
standing rule that executed plans stay honest.)*
