# PLAN: v100-qwen-image (revised)

Hand-rolled Qwen-Image-2.1 generation pipeline for a Tesla V100 16GB, structured like
the LTX-2.5 ROCm build (engine / front-end split, subprocess encoder, VRAM calibration).

Status: **current executable plan, revised 2026-10-07.** This revision supersedes the
original draft (committed as `df59ce7`) and incorporates every correction and confirmed
requirement in `HANDOFF.md` (committed as `08bde4b`). The supersession ledger in
*Corrections applied* below records what changed and why.

---

## Goal

Generate high-quality images with Qwen-Image-2.1 at 1024×1024 on a single 16GB Volta
card (2K as an experimental opt-in whose feasibility is **not yet demonstrated**), with a
headless CLI front-end and a conservative VRAM guard that refuses configs likely to OOM.
Maximum image quality takes priority; speed is improved where possible, with no
user-specified hard generation-time limit.

---

## Confirmed requirements (spec — from Euan, confirmed 2026-10-07 in the review)

1. A single Tesla V100 16GB is a **firm requirement** at this time.
2. **Maximum image quality takes priority.** Improve speed where possible; there is no
   hard generation-time limit.
3. **Text-to-image first.** Editing and RGBA workflows are not required for the first
   usable version.
4. All discussed categories may be needed — photographic portraits, landscapes,
   illustrations, product images, text-heavy designs. Evaluate on a **balanced genre
   set**, never optimized for one genre.
5. **Preserve existing services.** Do not assume the AMD GPUs or the machine's full RAM
   are available to this project.
6. 1024×1024 default; 2K is an **experimental opt-in**, feasibility unproven.
7. Hardware bring-up (driver install + container GPU passthrough) is in scope, done by
   the project at the software layer; physical installation, cooling and power are
   Euan's.
8. "Hand-rolled" = the **LTX-style engine layer over upstream model code** (diffusers
   underneath), per Euan's "what will give the best results?" — quality is set by
   weights, quant, numerics and sampler, not by who writes the DiT forward. Hand-rolled:
   streaming offload, subprocess encoder, embedding cache, VRAM guard, CLI. Not
   hand-rolled: the DiT forward / attention mask (upstream implements them; hand-rolling
   buys zero quality and adds a large bug surface).

## Environment snapshot

Read-only checks **inside this sandbox on 2026-10-07** (replaces the stale 2026-10-06
host table; host observations do not prove container state):

| Fact | Evidence |
|---|---|
| GPU **not visible in this sandbox**; availability unknown, not assumed | no `nvidia-smi`/`lspci` binaries; container has no GPU device access configured yet |
| V100 physical install **not confirmed** | Euan expected it on 2026-10-07 via M.2→PCIe 4.0 x4 adapter; unverified at handoff |
| Python 3.13.5, **no torch installed** | `python3 --version`; `import torch` → ModuleNotFoundError |
| RAM 122 GiB total, **~41 GiB available**; existing services run on this host | `free -h` — budget against *available*, not installed |
| ~944 GB free on `/work` | `df -h` |
| `~/models` **not mounted** in the sandbox | `ls ~/models` → missing; model path plan needed at M2 |
| Outbound HTTPS to GitHub reachable from sandbox | `api.github.com` → HTTP 200; **private-repo auth in-sandbox unverified** |
| `sandbox.yaml` has a user-owned uncommitted edit (host port 3082→3081) | `git diff` — preserve it; project work must not commit or modify this file |

Host-side facts from the earlier review (date 2026-10-07, may change — especially after
the V100 install): Ryzen 7 7800X3D, 2× Radeon AI PRO R9700 + iGPU, kernel
`7.0.0-38-generic`, no NVIDIA device in host `lspci` *at that time*. Re-verify from the
host after installation; do not treat as current.

## Corrections applied (supersession ledger vs. the `df59ce7` draft)

Every item below was corrected in the review (`HANDOFF.md` §"Feasibility assessment and
required plan corrections"); the ledger exists so nobody re-imports a dead claim.

| # | Draft claim | Corrected position |
|---|---|---|
| C1 | Pipeline class `QwenImage2Pipeline` | Correct class is **`QwenImage21Pipeline`** (diffusers `pipelines/qwenimage21/`). |
| C2 | "Volta requires PyTorch ≤2.10"; pin torch 2.7.* | **Wrong.** Official CUDA 12.6 wheels retain sm_70 support beyond 2.10; PyTorch announced **2.14 as the last release publishing cu126 wheels**. Pin an exact tested wheel and verify `torch.cuda.get_arch_list()` + real CUDA execution — not just device detection. |
| C3 | GGUF Q4_K_M resident as the quality default | **Rejected as a default.** Q4 is a capacity-fallback candidate only. Quality baseline = full-precision/higher-quality reference; compare resident **Q8** (and other higher-quality quants) against it and pick from measurement + human review. |
| C4 | 4-step Lightning LoRA in presets and the CPU smoke | **Rejected.** The generic 4-step Lightning dependency is not an identified, validated Qwen-Image-2.1 recipe. Default = **full base sampling recipe**. One concrete *optional* accelerator candidate: `Viggle/Qwen-Image-2.1-viggle-turbo` v0.3 (6 steps, prescribed sigmas/scheduler; its card says keep the LoRA **unmerged or merged in FP32 before quantization** — contradicting the draft's blanket "merge wins"). Inspect the exact current recipe before adopting. |
| C5 | "bf16 tensors must raise at load" (BF16 rejected) | **Contradictory in the draft; resolved:** BF16 checkpoints are *accepted for conversion* (→ fp16 for GPU execution); V100's lack of BF16 hardware does not forbid BF16 storage, and does **not** force FP16 CPU execution either — the CPU encoder's dtype is a separate, measured decision. |
| C6 | VAE tiling as the mitigation for 2K attention OOM | **Misconception removed.** VAE tiling reduces *decode* memory only; it does nothing for denoising-attention memory. Confirmed token counts: **4096 at 1024², 16384 at 2048²**; a dense FP32 attention-score tensor (32 heads × 16384 tokens) alone is **32 GiB**. Attention-backend selection is load-bearing at 2K; 2K stays experimental. |
| C7 | SSIM ≥0.95 distilled-vs-base as the quality gate; non-black PNG as smoke proof | **Rejected.** SSIM between distilled and base generations is not a sound quality gate, and a non-grey PNG proves nothing about model correctness. Use **numerical comparison for equivalent execution paths** (e.g. resident vs streamed at fixed seed) and **human evaluation** for prompt adherence, text, anatomy, composition, detail. A golden-image regression check may compare the *same config against its own reference* — never across recipes as a quality verdict. |
| C8 | Linear `fit_vram_model` fit → "refuses configs that won't OOM"; 2K-refused thresholds asserted up front | **No uncalibrated guarantees.** The guard needs phase/backend/conditioning-aware budgets, headroom, a check of *currently available* memory, and a conservative policy for uncalibrated cases. A fit to successful runs cannot guarantee no-OOM. Host RAM likewise must be budgeted from *available* RAM including load/conversion peaks and existing services — not installed RAM. |
| C9 | Verified box table asserting GPU absence/driver state as settled | **GPU availability is an assumption to verify, never a premise.** All GPU-dependent steps are explicitly gated on a *confirmed, measured* card. PCIe note: NVIDIA specifies the PCIe V100 as **Gen3 x16**; via a conventional direct-wired M.2 x4 adapter the best negotiated link is **Gen3 x4 ≈ 3.94 GB/s theoretical per direction** — an inference, not a measurement of Euan's setup. Measure link gen/width and H2D throughput once installed; repeated block streaming may be expensive on this link. |
| C10 | ComfyUI-GGUF loader probe as the load-path validation | **Insufficient.** A successful ComfyUI GGUF load does not validate Diffusers execution; stable-diffusion.cpp is a possible *alternative engine*, not a drop-in loader for a Diffusers pipeline. Reviewed diffusers `main` already has a non-Flex SDPA attention processor for 2.1, a Qwen21 single-file checkpoint mapping, and native PyTorch GGUF dequantization — test the exact checkpoint with **those** before writing custom ComfyUI-based integration. None of this is a proven V100 run yet. |
| C11 | Dependency plan "torch 2.7.* + diffusers/transformers from PyPI" | Qwen documents a **Git install of Diffusers** and **Transformers ≥5.17** at review time. Resolve and pin an **exact** dependency set *and model revisions* — upstream `main` is not a pin. |
| C12 | Encoder validation = embedding shapes match | **Insufficient.** Validate the encoder's actual upstream **prompt formatting, hidden-state extraction and masks** against upstream behavior; shape equality alone can hide subtle conditioning errors. |
| C13 | "2K run peak < 15.5 GB" style thresholds written into tasks | Kept only as explicit *placeholders* to be replaced by measurement (C8/C9); no milestone may claim a memory or speed number it has not measured. |

Unchanged, still-standing model facts (third-party claims — **verify at first download,
task 2.1**): Qwen-Image-2.1, DiT 7B / 32 layers (~14.2 GB BF16); text encoder Qwen3-VL
8B (~17.5 GB BF16); VAE ~0.7–1.4 GB; native 2K support. The draft's block-causal
attention description is a model-card/third-party claim: read the actual attention/mask
semantics in the pinned diffusers source (task 0.2) and build against *that*, not the
card.

---

## Architecture

`qwen_engine.py` owns the pipeline (load → encode-cache → denoise → VAE decode → save)
and imports nothing UI-related. `cli_gen.py` is the only v1 front-end (GUI deferred,
YAGNI).

- **Encoder:** Qwen3-VL in a throwaway subprocess (`run_subprocess_logged` pattern) so
  it costs no VRAM; runs on host CPU; execution dtype is a measured choice (C5), with
  the *available-RAM* budget as the constraint.
- **Embedding cache:** keyed on (model, prompt, negative, kwargs) — port LTX's
  `_embed_cache_key` *with* the negative-prompt key fix; measure whether the disk cache
  is worth it (upstream KV-caches conditioning across steps anyway).
- **DiT:** resident quant chosen by measurement, expected quality default **Q8 GGUF →
  fp16 dequant**; streaming block offload (LTX pattern, incl.
  `torch._C._host_emptyCache()` on pinned-host teardown) kept as the capacity/quality
  fallback — costed against the measured PCIe link (C9).
- **Attention:** backend selected by benchmark (efficient/SDPA path from upstream
  diffusers, not custom code); the dense-FP32 path is understood to be prohibitive above
  1024² (C6).
- **Memory guard:** `vramfit.py` records phase-wise observations and answers "does this
  config fit *right now*?" with headroom and a conservative uncalibrated policy (C8).

Source patterns ported from `ltx_engine.py` (public
`The-Deflection-Engine/ltx-2.5-rocm-generator`, private `EuanWebster/ltx-build-archive`):
`run_subprocess_logged`, `_embed_cache_key`, `vram_profile` / `fit_vram_model` /
`record_vram_observation` (as the *starting skeleton* for C8's guard), group offloading,
guarded pinned-host cache cleanup, and "engine never imports the UI".

---

## Milestones & tasks

Conventions: work happens **inside the container on `/work`**; one git commit per task;
tests with `pytest` as a pinned dev dep from day one (LEM lesson: unpinned `.venv`
installs vanish on folder moves). Python torch stack goes in a project venv at
`/work/.venv`, not baked into the harness image chain. **`sandbox.yaml` is user-owned —
project commits must not touch it**; any device-access change it needs is raised for
Euan to apply.

### M0 — Pin dependencies and interfaces (executable now, no GPU needed)

- **0.1** Resolve and lock the exact dependency set in `requirements.lock`: candidate
  torch from the cu126 wheel series (**≤2.14**, the last publishing cu126 — C2);
  diffusers and transformers at **exact git commits** per Qwen's documented requirement
  (Git-installed diffusers; Transformers ≥5.17 at review time — C11); pin model repo
  revisions too. *Verify:* lockfile committed; `torch.cuda.get_arch_list()` contains
  `sm_70` (this check needs only the wheel, not the card).
- **0.2** Interface survey against the pinned checkouts → `docs/interfaces.md`: confirm
  `QwenImage21Pipeline` signature and `prompt_embeds`/mask contract (C1/C12), the
  attention processors incl. the non-Flex SDPA path (C6/C10), the Qwen21 single-file
  GGUF checkpoint mapping and native GGUF dequant (C10), and the real attention/mask
  semantics (supersedes the card's block-causal description).
- **0.3** Lightweight CPU-only tests committed as the foundation lands (config
  round-trip, token math, cache keys, CLI dry-run). No large-model smoke until a fresh
  *available-RAM* budget check (C8: ~41 GiB available at snapshot, services running).

### M1 — Foundation without the card (executable now)

- **1.1** `qwen_engine.py` skeleton: config dataclass (prompt, size, steps, seed, quant,
  offload mode, paths), `resolve_model_path` against the manifest from 2.1, structured
  logging. Test: config round-trips JSON. **TDD, commit.**
- **1.2** `cli_gen.py`: argparse front-end → `engine.generate(config)`; LTX-style phase
  log; `--dry-run` prints the resolved plan (including the guard's answer) and generates
  nothing. **Commit.**
- **1.3** Encoder subprocess plumbing (port `run_subprocess_logged`) + embedding cache
  (port `_embed_cache_key`, LRU cap, **negative-key fix tested both variants**).
  Cache dir `cache/embeds/`. Test: request round-trip and key semantics without loading
  the full encoder. **Commit.**
- **1.4** Benchmark harness skeleton (phase timers → `benchmarks/`, observation records
  → `calibration.json` schema). **Commit.**

### M2 — Weights and model-side verification (needs disk + a RAM budget check)

- **2.1** Decide the in-sandbox model path (sandbox has no `~/models` mount); download
  DiT, Qwen3-VL encoder, VAE at **pinned revisions**; verify hashes; record actual
  filenames/sizes in `docs/models.md`, replacing the unverified model-facts table.
  *Verify:* measured sizes vs card within ±5 %; revisions recorded.
- **2.2** Load-path probes (CPU, small): Diffusers native GGUF loader + Qwen21
  single-file mapping loads the exact checkpoint; encoder fidelity check against
  **upstream prompt formatting / hidden-state extraction / masks**, not just shapes
  (C10/C12). A CPU smoke generation at ≤512² with the **base recipe at a low step count**
  is plumbing smoke only — explicitly *not* quality evidence (C7).

### M3 — GPU bring-up (gated: blocked until the V100 install is confirmed)

> Gate: physical installation, cooling and power are Euan's; not confirmed at handoff.
> Software only once the card is confirmed present on the host.

- **3.1** Pre-flight: Secure Boot state (`mokutil --sb-state`), kernel version; decide
  signing vs MOK vs `--disable-signing`. Record in `docs/bringup.md`.
- **3.2** Install the proprietary NVIDIA driver branch that still supports Volta
  (580.x series at review time — re-verify it is still the last sm_70 branch; not
  `*-server-open`). Reboot. *Verify:* `nvidia-smi` lists the V100 16GB; thermals sane
  under load (cooling is Euan's claim — verify, don't assume).
- **3.3** Container GPU access: `libnvidia-container` + CDI spec for podman; the device
  entry must be added to `sandbox.yaml` **by Euan** (user-owned file). *Verify:*
  `nvidia-smi -L` inside the sandbox prints the V100; then prove **real CUDA execution**
  (a kernel run, not just detection — C2) from `/work/.venv` torch.
- **3.4** Connection measurement: negotiated PCIe gen/width and host→device throughput;
  check shared-chipset contention. Record in `docs/bringup.md`; feeding these numbers
  into the streaming cost model (C9).

### M4 — Numerics, attention, and the first good 1024² image

- **4.1** Precision validation: BF16 checkpoints accepted and converted; GPU execution
  fp16 with fp32 norms/statistics as a *guard, not a guarantee* — check representative
  intermediate tensors for finiteness against a higher-precision reference (C5/C7).
  Encoder CPU dtype decided by measurement (C5). **Commit.**
- **4.2** Attention backend benchmark at small and 1024² scale, incl. a measured dense
  vs efficient memory profile consistent with the 32 GiB dense estimate (C6); default
  set from measurement. *Verify:* documented choice + numbers.
- **4.3** First acceptable 1024×1024 image with the **full base sampling recipe** (no
  distilled defaults — C4); measure and record every phase. Gate = **human evaluation**
  on the balanced genre set (portraits, landscapes, illustration, product, text-heavy).

### M5 — Quantization, memory strategy, and the guard (calibrated, not asserted)

- **5.1** Reference vs quant ladder: higher-precision/offloaded reference first, then
  resident **Q8** and other higher-quality quants compared against it (numerical
  comparison on equivalent execution paths + human eval on the genre set). Q4 only as a
  capacity fallback with the quality delta documented (C3).
- **5.2** Streaming block offload mode where it preserves quality or is needed for
  capacity; cost it against the measured link (M3.4). Equivalence test: streamed vs
  resident at fixed seed, max abs diff within fp16 tolerance — this is an *equivalent
  path* check, the legitimate use of numerical diffing (C7).
- **5.3** `vramfit.py` guard wired into the CLI: phase/backend/conditioning-aware
  budgets, headroom, reads currently available memory, conservative refuse for
  uncalibrated configs, and a host-RAM budget that accounts for conversion peaks and
  existing services (C8). Tests: fabricated 2K-resident config → refuse (exit 3);
  missing V100 → clean error naming the CUDA pin, not a stack trace. Thresholds come
  from calibration, never asserted (C13).

### M6 — 2K experimental and optional acceleration (after baseline)

- **6.1** 2K attempt behind `--hi-res`: attention backend from 4.2 is load-bearing
  here (16384 tokens); VAE tiling (`--tiled-decode`) is offered as a *decode-memory*
  knob only (C6). No pre-asserted GB ceiling; the guard from 5.3 decides per-run from
  calibrated data.
- **6.2** Optional accelerator study — `Viggle/Qwen-Image-2.1-viggle-turbo` v0.3 (6
  steps, prescribed sigmas/scheduler; LoRA unmerged or FP32-merged pre-quantization per
  its card) — evaluated against the base recipe by human eval on the genre set; ships
  only as an opt-in preset if quality holds (C4/C7).

### M7 — Bench, calibrate, document

- **7.1** Benchmark matrix (1024 [×2K if viable] × resident/stream × base/[turbo if
  adopted] × multiple seeds) → `benchmarks/`, feeding `calibration.json`. **All
  placeholder thresholds elsewhere are replaced here.**
- **7.2** `README.md` (what fits at what cost, commands) + `docs/bringup.md` (V100
  driver/CDI/PCIe trail for the next Volta box).

---

## Tests / validation

- Full: `cd /work && .venv/bin/pytest -q`; every numbered task ships a test.
- Equivalent-path numerical diffs (resident vs streamed, quant vs reference) — sound
  use of numerical comparison. **No cross-recipe SSIM quality gates** (C7).
- Golden regression: fixed prompts × fixed seed per **config** vs its own stored
  reference after first human-approved run — catches silent mask/numerics regressions;
  scoped to same-config comparison only.
- Encoder contract test: prompt formatting, hidden-state extraction and masks match the
  pinned upstream implementation (C12).
- Negative: 2K-resident config beyond calibrated budget → refuse, exit 3; missing V100 →
  clean error naming the CUDA pin. BF16 inputs must **not** be rejected at the conversion
  boundary (C5).
- Quality decisions: human evaluation on the balanced genre set (requirement 4), never
  one genre, never "PNG isn't grey" (C7).

## Risks, tradeoffs, open questions

| Risk | Mitigation |
|---|---|
| GGUF dequant Triton kernels break on sm_70 | Spike fallback paths on the real card before building on top; stable-diffusion.cpp = separate *alternative engine* option, not a diffusers drop-in (C10) |
| fp16 overflow in DiT (no BF16 on Volta) | fp32-stat guards + intermediate-tensor checks vs a higher-precision reference (C5); golden same-config regression |
| Driver branch vs kernel 7.0 / Secure Boot | 3.1 pre-flight before touching anything |
| `pip -U` silently lands a wheel without sm_70 | `requirements.lock` + startup assert of `capability==(7,0)` **and** arch-list/CUDA execution proof, not detection alone (C2) |
| PCIe link far slower than assumed | measure M3.4 before choosing resident-vs-stream strategies (C9) |
| 2K attention activations exceed 16 GB | backend choice (4.2) + calibrated guard (5.3); VAE tiling is not the answer (C6); 2K stays experimental |
| diffusers 2.1 support is young code | 0.2/0.3 CPU tests + 2.2 probes before GPU work; exact git pins in lockfile (C11) |
| Model-card sizes/behavior are third-party claims | 2.1/0.2 replace them with measured facts and source reading |
| Host RAM headroom smaller than assumed (services running) | budget from *available* memory incl. conversion peaks; 41 GiB at snapshot (C8) |

**Open questions (none block M0/M1):**
1. Which resident quant survives the quality ladder (5.1) — measurement + Euan's eye,
   not folklore.
2. Whether Viggle turbo quality holds as an opt-in preset (6.2).
3. In-sandbox model storage path/mount (2.1).
4. GUI front-end — deliberately out (YAGNI; CLI per spec).

## Execution notes

*(empty — appended during the build, one line per deviation from this plan, per the
standing rule that executed plans stay honest.)*
