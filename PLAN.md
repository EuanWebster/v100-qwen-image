# PLAN: v100-qwen-image (revised)

Hand-rolled Qwen-Image-2.1 generation pipeline for a Tesla V100 16GB, structured like
the LTX-2.5 ROCm build (engine / front-end split, subprocess encoder, VRAM calibration).

Status: **current executable plan, revised 2026-10-07; senior-review fix pass on
`f6325d0`.** This revision supersedes the original draft (committed as `df59ce7`) and
incorporates every correction and confirmed requirement in `HANDOFF.md` (committed as
`08bde4b`) — all of those corrections remain in force. The supersession ledger in
*Corrections applied* below records what changed and why; the senior-review ledger
(R1–R7) records the structural fixes applied on top of `f6325d0`: package boundaries
instead of an engine monolith, explicit CPU-only acceptance criteria, a pinned model
manifest contract, a memory-policy descriptor, a hardened encoder cache, versioned
benchmark schemas, and fresh-environment test requirements.

---

## Goal

Generate high-quality images with Qwen-Image-2.1 at 1024×1024 on a single 16GB Volta
card (2K as an experimental opt-in whose feasibility is **not yet demonstrated**), with a
headless CLI front-end and a conservative memory policy that refuses anything it cannot
clear from calibration (R4/M5.3 — no uncalibrated guarantees, per C8).
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

### Senior-review fix ledger (on `f6325d0` — all C1–C13 corrections above stand)

| # | Finding on `f6325d0` | Fix in this revision |
|---|---|---|
| R1 | `qwen_engine.py` was a monolith mixing config, loading, logging and CLI | Explicit **package boundaries** (see *Architecture*): immutable config/validation, manifest + no-download resolver, pipeline **adapter protocol** with heavy ML imports deferred inside the eventual `Qwen21DiffusersAdapter`, structured phase/event logging, CLI, memory policy, encoder worker/cache, benchmark schema. |
| R2 | CPU-only milestone acceptance criteria were vague | M1 now ships **testable acceptance criteria**: config JSON round-trip + field-level validation, token math (1024² = 4096, 2048² = 16384), dimensions multiple-of-32, `import` + `--help` with no torch/models installed, deterministic JSON `--dry-run` reporting unresolved hardware/assets/calibration without importing heavy packages, malformed config → exit 2, actual-run missing assets → documented nonzero exit without traceback. |
| R3 | Model "manifest from 2.1" was undefined | **Versioned, pinned manifest contract** (`models/manifest.json`): per-artifact repo / revision / filename / sha256 / byte-size; resolver verifies and **never silently downloads** — a missing artifact is an error naming what to fetch, not an implicit fetch. |
| R4 | Guard was "phase/backend/conditioning-aware" but the decision inputs were undefined | **Memory-policy descriptor** with named fields (phase, tokens/resolution, prompt length, CFG duplication, attention backend, quant, residency/offload, VAE mode, model revision, current free memory + headroom), **stale-calibration invalidation**, conservative-unknown refusal, and **2K refused until calibrated** unless an explicit expert-override flag is set (logged). |
| R5 | Cache key was (model, prompt, negative, kwargs); worker lifecycle unspecified | **Encoder cache key v2** adds processor/tokenizer/chat-template revisions or content hashes and encoder dtype (image identities + order when conditioning on images arrives); atomic cache writes; corrupt entry = miss + quarantine; concurrency-safe; worker timeout/cancel/error propagation and stderr capture; temp cleanup; **fake-worker tests** without any model. |
| R6 | Benchmark output was "phase timers → benchmarks/" | **Versioned benchmark/result schemas**: per-phase wall time, peak host RSS, peak CUDA allocated/reserved *when available* (null when not — never fabricated), plus environment identifiers (torch/CUDA/driver/arch, host, python) and model identifiers (repo/revision/hash). |
| R7 | CPU tests did not state their environment | CPU test suites must pass in a **fresh environment with no network, no GPU, no model weights** and must **write nothing outside temporary directories** (all cache/benchmark/log paths relocatable via config). |

Unchanged, still-standing model facts (third-party claims — **verify at first download,
task 2.1**): Qwen-Image-2.1, DiT 7B / 32 layers (~14.2 GB BF16); text encoder Qwen3-VL
8B (~17.5 GB BF16); VAE ~0.7–1.4 GB; native 2K support. The draft's block-causal
attention description is a model-card/third-party claim: read the actual attention/mask
semantics in the pinned diffusers source (task 0.2) and build against *that*, not the
card.

---

## Architecture (package boundaries — R1)

There is **no `qwen_engine.py` monolith**. The engine is a small package with explicit
boundaries, so config validation, manifest resolution, dry-run planning, the CLI and all
CPU tests never touch a heavy ML import:

| Module | Responsibility | May import |
|---|---|---|
| `qwen_engine/config.py` | **Immutable** (frozen dataclass) generation config + field-level validation: dimensions multiple of 32, resolution→token math (1024² = 4096, 2048² = 16384 — C6), steps/seed ranges, quant/offload/backend enums, paths. Lossless JSON round-trip. | stdlib |
| `qwen_engine/manifest.py` | **Versioned model-manifest schema + `resolve_model_path`** — pinned repo/revision/artifact/sha256/size contract, existence/size verification, **no-download policy** (R3; schema in M2.1). | stdlib |
| `qwen_engine/adapters/base.py` | **Pipeline adapter protocol** the engine core calls (`plan`, `load`, `encode`, `denoise`, `decode`, `save`) plus plan/dry-run types. The core runs end-to-end against a fake adapter in tests. | config/manifest/memory/log types |
| `qwen_engine/adapters/qwen21_diffusers.py` | The eventual `Qwen21DiffusersAdapter` — the **only** module that imports torch / diffusers / transformers, and those imports are deferred *inside* the adapter (lazy), executed only on an actual run. Does not exist yet; stubbing it at M1 is a boundary violation the import test catches. | torch, diffusers, transformers |
| `qwen_engine/log.py` | Structured **phase/event logging**: JSONL events `{ts, phase, event, elapsed_ms, counters…}` to a configurable sink, plus human-readable phase lines. Shared by engine, CLI and bench. | stdlib |
| `qwen_engine/memory.py` | **Memory-policy** descriptor + verdict logic (R4; detail in M5.3): "does this config fit *right now*?" from calibrated observations, current free memory, and headroom; conservative on unknown; stale calibration invalidated (C8). (Supersedes the draft's standalone `vramfit.py` naming.) | config/manifest/log types |
| `qwen_engine/encoder.py` | Encoder **subprocess worker + embedding cache** (port `run_subprocess_logged`; key v2, atomic writes, quarantine, timeout/cancel, fake-worker tests — R5; detail in M1.4). | stdlib |
| `qwen_engine/bench.py` | **Versioned benchmark/result schemas** (R6; detail in M1.5): phase timings, peak host RSS, peak CUDA memory when available, environment and model identifiers; feeds `calibration.json`. | stdlib |
| `qwen_engine/cli.py` | The only v1 front-end (GUI deferred, YAGNI): argparse → validate config → plan via core/adapter protocol → `generate`; implements the **exit-code contract** (M1 acceptance criteria). `cli_gen.py` at the repo root stays as a thin entry-point shim. | the above only |

**Boundary invariant (tested):** `python -c "import qwen_engine"` and
`cli_gen.py --help` / `--dry-run` must succeed on a machine with **no torch, no
diffusers, no transformers and no model weights** installed. Heavy ML imports live only
inside `Qwen21DiffusersAdapter` and the encoder worker script, reached only by an actual
run. The engine package imports nothing UI-related (LTX rule preserved).

- **Encoder:** Qwen3-VL in a throwaway subprocess (`run_subprocess_logged` pattern) so
  it costs no VRAM; runs on host CPU; execution dtype is a measured choice (C5), with
  the *available-RAM* budget as the constraint. Worker lifecycle hardening per R5 (M1.4).
- **Embedding cache:** key **v2** — model repo+revision, processor/tokenizer/chat-template
  revisions or content hashes, prompt, negative prompt, all encoding kwargs, encoder dtype
  (+ ordered image identities when image conditioning exists; R5). Port LTX's
  `_embed_cache_key` *with* the negative-prompt key fix as the base, then extend; measure
  whether the disk cache is worth it (upstream KV-caches conditioning across steps anyway).
- **DiT:** resident quant chosen by measurement, expected quality default **Q8 GGUF →
  fp16 dequant**; streaming block offload (LTX pattern, incl.
  `torch._C._host_emptyCache()` on pinned-host teardown) kept as the capacity/quality
  fallback — costed against the measured PCIe link (C9).
- **Attention:** backend selected by benchmark (efficient/SDPA path from upstream
  diffusers, not custom code); the dense-FP32 path is understood to be prohibitive above
  1024² (C6).
- **Memory guard:** `memory.py` (R4) answers "does this config fit *right now*?" from a
  named policy descriptor, headroom and calibrated observations, with a
  conservative unknown-policy (C8); detail in M5.3.

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
  revisions too, recorded in the versioned `models/manifest.json` contract (R3, M2.1) —
  never downloaded implicitly. *Verify:* lockfile committed;
  `torch.cuda.get_arch_list()` contains `sm_70` (this check needs only the wheel, not
  the card).
- **0.2** Interface survey against the pinned checkouts → `docs/interfaces.md`: confirm
  `QwenImage21Pipeline` signature and `prompt_embeds`/mask contract (C1/C12), the
  attention processors incl. the non-Flex SDPA path (C6/C10), the Qwen21 single-file
  GGUF checkpoint mapping and native GGUF dequant (C10), and the real attention/mask
  semantics (supersedes the card's block-causal description).
- **0.3** Lightweight CPU-only tests committed as the foundation lands (config
  round-trip, token math, cache keys, CLI dry-run). Test rule (R7): every CPU test must
  pass in a **fresh environment with no network, no GPU and no model weights**, and must
  **write nothing outside temporary directories** — all cache/benchmark/log/output paths
  are config-relocatable and tests point them at `tmp_path`. No large-model smoke until a
  fresh *available-RAM* budget check (C8: ~41 GiB available at snapshot, services
  running).

### M1 — Foundation without the card (executable now)

Build the package skeleton from *Architecture*; the real adapter does not exist yet —
everything below runs against the adapter protocol with fakes.

- **1.1** `config.py` + `log.py`: frozen generation-config dataclass (prompt, negative,
  size, steps, seed, quant, offload mode, attention backend, VAE tiling flag, paths,
  cache/bench/log dirs) with field-level validation and lossless JSON round-trip;
  structured phase/event logger. **Commit.**
- **1.2** `manifest.py`: versioned manifest schema + `resolve_model_path` implementing
  the **no-download contract** (R3, schema finalized in 2.1): artifact present & size
  matches → path; absent/mismatch → structured error naming repo/revision/file.
  **Commit.**
- **1.3** `adapters/base.py` adapter protocol + engine core `generate()`/`plan()` over
  the protocol; `cli.py` front-end: argparse, `--dry-run`, `--config`, exit-code
  contract. Heavy imports reachable only through the (not yet written)
  `Qwen21DiffusersAdapter`. **Commit.**
- **1.4** `encoder.py`: subprocess worker (port `run_subprocess_logged`) + embedding
  cache. Requirements (R5):
  - **Key v2** (version-tagged schema): model repo+revision; processor, tokenizer and
    chat-template revisions *or content hashes*; prompt; negative prompt; **all**
    encoding kwargs; encoder dtype; and — once image conditioning exists — image
    identities (content hashes) *in order*. Port `_embed_cache_key` with the
    negative-key fix as the base; **negative-key fix tested both variants**; LRU cap.
    Cache dir `cache/embeds/` (config-relocatable).
  - **Atomic writes** (temp file + `os.replace`); a corrupt/unreadable entry is a **cache
    miss**, and the bad file is **quarantined** (moved aside), never silently overwritten
    in place.
  - **Concurrency safety**: atomic rename means readers never see partial entries;
    per-key in-process locking so parallel requests don't double-encode or corrupt.
  - **Worker lifecycle**: timeout + cancel propagate to the child (terminate → kill),
    worker error/exit status surfaces as a typed engine error, **stderr captured** into
    the event log (not swallowed), temp files and pinned-host resources cleaned up on
    every path.
  - Tests use a **fake encoder worker** (a scripted stub) — no model, exercising hit,
    miss, corrupt-entry quarantine, timeout, and error propagation. **Commit.**
- **1.5** `bench.py`: benchmark harness skeleton (phase timers → `benchmarks/`,
  observation records → `calibration.json`). **Versioned result schema**
  `benchmark_result.v1` (R6): schema version; per-phase wall time (load / encode /
  denoise-per-step / decode / save); **peak host RSS**; **peak CUDA allocated/reserved
  when CUDA is available — recorded as `null` when it is not, never fabricated**;
  environment identifiers (python, torch/CUDA/driver versions, `get_arch_list()`,
  hostname); model identifiers (manifest repo/revision/sha256 per artifact); and an echo
  of the generation config. `calibration.json` gets a matching versioned observation
  schema whose records carry the fields the memory policy keys staleness on (M5.3).
  **Commit.**

**M1 acceptance criteria (all CPU-only, each an actual test):**

1. Config **JSON round-trip is lossless**; **field-level validation** rejects each
   invalid field individually with the field named in the error (bad enum, negative
   seed, zero steps, …).
2. **Token math**: 1024² → **4096** image tokens and 2048² → **16384** asserted exactly
   (C6), plus arbitrary in-range sizes; **width and height must be multiples of 32** —
   anything else is a validation error naming the offending dimension.
3. **Import and `--help` without torch/models**: importing `qwen_engine` and running
   `cli_gen.py --help` succeed with torch, diffusers, transformers and all weights
   *absent* (test asserts none of these modules end up in `sys.modules`).
4. **Deterministic JSON `--dry-run`**: `--dry-run` prints stable, key-sorted JSON —
   resolved config, manifest resolution status, guard verdict — and explicitly reports
   **unresolved hardware / assets / calibration** as such (never pretending a value was
   measured); it must not import heavy packages and must not generate or write outside
   temp dirs.
5. **Malformed config → exit 2** with per-field errors on stderr, no traceback.
6. **Actual run with missing/unverifiable assets → documented nonzero exit (4)** naming
   the unresolved artifact and the manifest entry — no traceback, and no download
   attempted (R3).

Exit-code contract (tested): `0` success · `2` invalid/malformed config · `3` memory
guard refusal (M5.3) · `4` missing/unverified model asset · `5` required hardware absent
or unverified (e.g. no confirmed V100 — error names the CUDA pin, per 5.3) · `1`
unexpected (logged with context, traceback only into the log sink).

### M2 — Weights and model-side verification (needs disk + a RAM budget check)

- **2.1** Decide the in-sandbox model path (sandbox has no `~/models` mount). Define the
  **model manifest contract** (R3) as committed, versioned `models/manifest.json`:
  `schema_version` plus, per artifact (DiT, Qwen3-VL encoder, VAE, any LoRA),
  `{role, repo, revision (commit SHA), filename, sha256, bytes}`. `manifest.py` resolves
  against it and **never silently downloads** — the resolver only verifies; fetching is a
  separate, explicit operator action (a documented `fetch_models.py` run), and a missing
  or mismatched artifact exits 4 naming repo/revision/file (M1 criterion 6). Manifest
  changes are git commits, so every run's model identity is reproducible from
  `benchmark_result.v1` model identifiers (R6). Then download DiT, Qwen3-VL encoder, VAE
  at the **pinned revisions**; verify hashes; record actual filenames/sizes in
  `docs/models.md` (backed by the manifest), replacing the unverified model-facts table.
  *Verify:* manifest committed with schema version; measured sizes vs card within ±5 %;
  revisions recorded; resolver test proves no network call on the missing-asset path.
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
- **5.3** `memory.py` policy wired into the CLI and the adapter (R4). Every guard query
  carries a **policy descriptor**: phase (load / encode / denoise-step / decode / save);
  token count + resolution (C6); prompt token length; **CFG duplication** (does the
  negative prompt double the batch); attention backend; quant format; residency mode
  (resident / streamed / offload); VAE mode (tiled decode on/off — decode memory only,
  C6); pinned model revision; **current free GPU and host memory** with configured
  headroom. Rules:
  - **Stale-calibration invalidation:** observations are keyed by model revision,
    backend, quant and schema version; any mismatch makes them stale — ignored, never
    interpolated across.
  - **Conservative unknown:** no fresh matching calibration → refuse (or offer only the
    most conservative known-safe mode), never guess (C8).
  - **2K refusal until calibrated:** descriptor above 1024²-class token counts → refuse
    (exit 3) while calibration is missing/stale, unless an explicit
    `--expert-override` flag is set — an acknowledged escape hatch for 6.1 experiments,
    recorded in the event log and benchmark record.
  - Host-RAM budget from *available* memory incl. conversion peaks and existing services
    (C8). Tests: fabricated 2K-resident config with empty/stale calibration → refuse
    (exit 3), and refuse is lifted only by matching calibration or the override flag
    (flag is logged); stale-revision observation is ignored; missing V100 → clean error
    naming the CUDA pin (exit 5), not a stack trace. Thresholds come from calibration,
    never asserted (C13).

### M6 — 2K experimental and optional acceleration (after baseline)

- **6.1** 2K attempt behind `--hi-res`: attention backend from 4.2 is load-bearing
  here (16384 tokens); VAE tiling (`--tiled-decode`) is offered as a *decode-memory*
  knob only (C6). No pre-asserted GB ceiling; the guard from 5.3 decides per-run from
  calibrated data — and while 2K calibration is still missing, each 2K experiment runs
  explicitly under `--expert-override` (logged), per R4.
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
- **Fresh-environment CPU suite (R7):** the M0.3/M1 tests must pass in a clean
  environment with **no network, no GPU and no model weights** (and torch/diffusers/
  transformers not importable — the import test asserts this), and must **write nothing
  outside temporary directories** — a fixture redirects cache/benchmark/log/output paths
  to `tmp_path` and fails the run on any write elsewhere in the repo or home.
- **CLI contract tests** (M1 acceptance criteria): JSON round-trip + field-level
  validation; token math 1024² = 4096 and 2048² = 16384; multiple-of-32 dimension
  validation; `import` + `--help` without torch/models; deterministic key-sorted JSON
  `--dry-run` reporting unresolved hardware/assets/calibration; malformed config → exit
  2; actual run with missing assets → exit 4, no traceback, no download attempt.
- Encoder worker tests via a **fake worker**: key-v2 semantics (negative-key fix tested
  both variants), atomic write, corrupt-entry miss + quarantine, concurrent access,
  timeout/cancel and stderr capture, temp cleanup (R5).
- Equivalent-path numerical diffs (resident vs streamed, quant vs reference) — sound
  use of numerical comparison. **No cross-recipe SSIM quality gates** (C7).
- Golden regression: fixed prompts × fixed seed per **config** vs its own stored
  reference after first human-approved run — catches silent mask/numerics regressions;
  scoped to same-config comparison only.
- Encoder contract test: prompt formatting, hidden-state extraction and masks match the
  pinned upstream implementation (C12).
- Negative: 2K-resident config with missing/stale calibration → refuse, exit 3 (only
  matching calibration or logged `--expert-override` lifts it, R4); missing V100 →
  clean error naming the CUDA pin, exit 5. BF16 inputs must **not** be rejected at the
  conversion boundary (C5).
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

- 2026-10-07 senior-review fix pass (manifest commit `36abafc` + config/core/CLI/log
  commit): multi-artifact roles with filename identity + `resolve_model_paths`;
  construction-time manifest validation; containment-checked single-fd resolver with
  regular-file and change detection (missing ≠ unreadable, typed path escapes).
- Same pass: default `steps=40` (official Qwen example); `true_cfg_scale` (finite
  ≥1.0, default 1.0), CFG active only when negative prompt present AND scale > 1;
  CLI `--true-cfg-scale`; path fields reject blank/NUL/file-dir mismatches without
  creating anything; invalid-UTF-8 config → exit 2.
- Same pass: adapter `load(config, assets)` receives verified role-grouped paths;
  core refuses unconfigured `output_dir` before load and verifies the saved regular
  file stays inside `output_dir`; adapter `plan` method + dead `PHASES` removed
  (core planning adapter-independent this milestone).
- Same pass: `--run`/`--dry-run` exclusive (2); asset errors keep exit 4 (checked
  first); `_make_adapter` maps only the exact absent-adapter ModuleNotFoundError;
  CLI handlers survive dead log sinks; logger records carry `schema_version` +
  injectable `run_id`, phase stack leak-free on failed start emission, log faults
  never mask primaries. `license = "MIT"` removed (none selected); build backend
  pinned `setuptools==84.0.0`; uv lock in sync. 182 CPU tests pass on Python 3.13;
  Python 3.11 execution still outstanding (M0/R7 fresh-env rule).
- 2026-10-07 task 1.4 (`encoder.py`, commit `8f1bcc4`): worker command is
  **required** (no default heavy worker at M1) with `{python}/{request}/{response}`
  placeholders; worker stdio redirects to scratch files (no pipe deadlock, output
  never lost across poll slices); cancel is cooperative via injected
  `threading.Event`, timeout path is terminate→wait→kill. Entry format
  `embed_entry.v1` = canonical JSON header line + raw **opaque** blob bytes —
  tensor payloads join later from the worker side without the cache importing
  torch. Corrupt entries quarantine under `<cache>/embeds/quarantine/`; LRU cap
  128 by mtime with refresh-on-hit; same-key in-process locks. 64 fake-worker
  tests; suite 246.
- 2026-10-07 task 1.5 (`bench.py`, commit `258385f`): `benchmark_result.v1`
  records phase ms, peak host RSS (stdlib `resource`; real bytes or `null`), and
  peak CUDA allocated/reserved **as null unless an adapter supplies them** —
  `bench.py` itself never imports torch, so unmeasured fields cannot be
  fabricated; denoise-per-step is derived, never stored. Aggregation
  `benchmark_aggregate.v1` is order-free and keeps unmeasured ≠ 0.
  `calibration_observation.v1` staleness key = (schema, model revision, backend,
  quant) for M5.3; `calibration.json` writes are atomic. 50 synthetic tests;
  suite 296; heavy-import boundary still clean in package and CLI dry run.
- 2026-10-07 senior-review correction pass on top of `6955b92`, two scoped
  commits. `cbecacf` (encoder): shared `_jsonsafe` deep-freeze means
  `EncoderRequest.kwargs`/`EncodedPayload.metadata` snapshots and
  `Blob` bytes are immutable all the way down — one request ⇒ one key ⇒ one
  wire document, immune to later caller mutation; direct invalid inputs
  (non-`ComponentIdentity` components, non-`Blob` entries, non-string JSON
  keys, NaN/Inf numbers) are typed `EncoderRequestError`; NaN/Inf
  `timeout_s`/`cancel_grace_s` rejected (NaN had silently disabled the
  timeout). Cache mutations/quarantine/LRU now run under one in-process
  mutex per canonical root with per-key locks shared per root: the
  deterministic max_entries=1 concurrent-put race (both puts snapshot, then
  evict each other → empty cache) is gone, and two instances on one root
  cannot double-encode; atomic publication kept. Worker reaps live children
  on **every** exit path before scratch cleanup — including
  `KeyboardInterrupt` during the wait (`Popen.wait` is sleep-based, so the
  interrupted teardown reaps via `os.waitpid`); returncode is checked
  before any response read; scratch/request/stdio setup failures are typed
  and clean up. Every failure branch emits a terminal
  `encoder_worker_failed` event carrying captured stderr, typed failures
  keep `.stderr`, dead log sinks can neither mask errors nor orphan
  workers, and stderr excerpts are bounded tail reads. Strict result wire:
  unknown result/error/blob fields and non-finite JSON constants are typed
  protocol errors; entry headers the same (corrupt = miss + quarantine).
  `929e0ef` (bench): `from_dict`/`from_json`/`load_result` require the
  exact explicit `benchmark_result.v1` (and explicit observation schema) —
  no silent re-labelling of missing/old/future tags; records deep-freeze
  config/model/phases/resolution so post-construction mutation cannot
  change saved output or aggregation; invalid direct constructions
  (NaN/Inf or non-string-keyed config, non-sequence model/resolution,
  10**400-ms phases, duplicate model identifiers) are `BenchSchemaError`,
  never raw `TypeError`/`OverflowError`. `resource` import is guarded;
  `host_rss_bytes` is documented as the parent-process lifetime peak,
  multiplies KiB→bytes only on Linux (macOS already bytes, unknown units →
  null). `add_observation` is synchronized per canonical path in-process
  (cross-process not required at this milestone — documented);
  `matching_observations` documented as staleness-key prefilter only, M5.3
  must still match phase/tokens/resolution/residency/CFG/VAE/prompt length;
  optional `prompt_tokens` added (null = unknown). No memory guard
  implemented early. 56 new tests; suite 352 on Python 3.13, fake workers
  only — no GPU/model measurements claimed.
