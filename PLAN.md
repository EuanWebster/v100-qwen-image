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
local **web workspace** as the primary front-end (2026-10-12 pivot — replaces the
deferred desktop GUI; the headless CLI stays; design in `docs/web-interface.md`)
and a conservative memory policy that refuses anything it cannot
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
| V100 physical install **confirmed NOT installed** (user, 2026-10-07) | supersedes the earlier "expected today / unverified at handoff" row: the M.2→PCIe 4.0 x4 adapter install has not happened — all GPU-gated milestones (M3+) stay closed |
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

Model facts, upgraded from card claims to **pinned upstream metadata** at the
M2.1 manifest commit (task 2.1 *preparation* — expected sizes from the pinned
manifest identities, **not local measurements** at that time): Qwen-Image-2.1,
DiT 7B / 32 layers — expected weight shards
14,230,315,061 B (~14.2 GB, BF16 *declared* storage); text encoder Qwen3-VL
8B — expected 17,534,409,013 B (~17.5 GB); VAE — expected 1,350,991,591 B
(~1.35 GB, at the top of the old 0.7–1.4 GB card range); all 24 runtime files
33,131,601,593 B. **Asset status since then:** root's explicitly approved
pinned fetch into `models/assets/` ended 2026-10-07 21:53 UTC with a timeout
on the first transformer shard; an independent offline resolver/plan found
**19 of 24 final files present and size+SHA256-verified, 5 missing, 0 bad**
(processor and text-encoder groups complete; recoverable
`transformer/diffusion_pytorch_model-00001-of-00002.safetensors.partial` =
6,302,990,336 B; remaining expected transfer after it 9,278,315,946 B) —
first-download hash verification of the LFS identities is therefore
**partly done** (details: `docs/models.md` §3/§7). Separately recorded: a
socket-denied processor-only load reached the pinned transformers code but
failed for missing `torchvision`; a tokenizer-only local load succeeded
(Qwen2Tokenizer, 151,669 entries, 9 tokens for root's sample); dependency
work is a later scoped milestone. "Native 2K support" remains a model-card claim; the draft's
block-causal attention description is a model-card/third-party claim: the
actual attention/mask semantics were read in the pinned diffusers source
(task 0.2 done — `docs/interfaces.md`) and the build follows *that*, not the
card. Declared dtypes are config metadata; actual tensor dtypes, CPU/GPU
execution dtype and 16 GB fit stay unverified until measured (M2.2/M3/M4).

---

## Architecture (package boundaries — R1)

There is **no `qwen_engine.py` monolith**. The engine is a small package with explicit
boundaries, so config validation, manifest resolution, dry-run planning, the CLI and all
CPU tests never touch a heavy ML import:

| Module | Responsibility | May import |
|---|---|---|
| `qwen_engine/config.py` | **Immutable** (frozen dataclass) generation config + field-level validation: dimensions multiple of 32, resolution→token math (1024² = 4096, 2048² = 16384 — C6), steps/seed ranges, quant/offload/backend enums, paths. Lossless JSON round-trip. | stdlib |
| `qwen_engine/manifest.py` | **Versioned model-manifest schema + `resolve_model_path`** — pinned repo/revision/artifact/sha256/size contract, existence/size verification, **no-download policy** (R3; schema in M2.1). | stdlib |
| `qwen_engine/local_bundle.py` | **Local reference-bundle contract (M2.2a)** — trusted factory `resolve_local_bundle` runs `resolve_manifest` itself (explicit full-rehash cost) before projecting into role-grouped adapter `assets` plus explicit component dirs under an explicit model root; lower-level `project_resolution_report` **trusts its caller's report** (never authenticated evidence). Enforces the committed 24-file full-reference layout table and the profile's **exact pinned repo/revision**, plus a coherence-checked success claim (`errors`/per-status `error`/`resolved` mapping); publishes **absolute, CWD-stable paths**; closed-tree `preflight_closed_tree` refuses any entry beyond the allowlist before a future `from_pretrained`; construction-time scalar invariants + defensive freezing; version-tagged artifact-set fingerprints (encoder digest = **asset component only** of the future worker cache key). No downloads, no ML imports. | manifest types, stdlib |
| `qwen_engine/adapters/base.py` | **Pipeline adapter protocol** the engine core calls (`load`, `encode`, `denoise`, `decode`, `save`) plus dry-run status types. Deliberately **no `plan` method** — dry-run planning is adapter-independent (`core.plan()` builds the document itself). The core runs end-to-end against a fake adapter in tests. | config/manifest/memory/log types |
| `qwen_engine/adapters/qwen21_diffusers.py` | The eventual `Qwen21DiffusersAdapter` — the **only** module that imports torch / diffusers / transformers, and those imports are deferred *inside* the adapter (lazy), executed only on an actual run. Does not exist yet; stubbing it at M1 is a boundary violation the import test catches. | torch, diffusers, transformers |
| `qwen_engine/log.py` | Structured **phase/event logging**: JSONL events `{ts, phase, event, elapsed_ms, counters…}` to a configurable sink, plus human-readable phase lines. Shared by engine, CLI and bench. | stdlib |
| `qwen_engine/memory.py` | **Memory-policy** descriptor + verdict logic (R4; detail in M5.3): "does this config fit *right now*?" from calibrated observations, current free memory, and headroom; conservative on unknown; stale calibration invalidated (C8). (Supersedes the draft's standalone `vramfit.py` naming.) | config/manifest/log types |
| `qwen_engine/encoder.py` | Encoder **subprocess worker + embedding cache** (port `run_subprocess_logged`; key **v3** — v2 plus the optional immutable `EncoderImplementationIdentity` (M2.2b foundation): encoder artifact-set digest (**asset component only**, never the whole key), worker protocol/implementation + prompt-extraction revisions, exact diffusers/transformer commits, torch build/version, safetensors/torchvision versions; generic fake requests may carry none, a production worker must require one; atomic writes, quarantine, timeout/cancel, bounded worker responses, fake-worker tests — R5; detail in M1.4/M2.2b). | stdlib |
| `qwen_engine/mem_preflight.py` | **Host-memory preflight for the forthcoming CPU encoder child (M2.2b)**: conservative stdlib check, before any ML import/weight read/loader call, that checkpoint bytes + explicit dtype expansion (integer ceiling division) + explicit runtime headroom fit the **tighter** of host `MemAvailable` (validated `kB` unit; conversion-limit-exceeding digits are unknown) and this process's cgroup v2 state. A real read walks **every visible ancestor** but is fail-closed **unknown**: a missing `memory.max` at any level proves nothing about the controller, and the visible root's completeness is unproven (a namespace `/` can hide a tighter ancestor), so real autodiscovery refuses regardless of root file contents — the tightest visible `limit - current` travels only as an **advisory visible availability** (an UPPER bound on true availability a hidden ancestor can only tighten) with truthful per-level source notes. An explicitly injected finite/unlimited `MemorySnapshot` is **trusted caller evidence** for pure budget arithmetic (finite cap never substituted away by host RAM; unknown host always refuses; absent finite current is never zero; zero cap is a zero budget). Passing is a conservative prerequisite, **not measured model-fit evidence**; the default deployment may legitimately refuse all real loads until trusted complete host-budget evidence exists. Scoped to Linux cgroup v2 — not the M5.3 `memory.py` policy and not a general resource framework. | stdlib |
| `qwen_engine/bench.py` | **Versioned benchmark/result schemas** (R6; detail in M1.5): phase timings, peak host RSS, peak CUDA memory when available, environment and model identifiers; feeds `calibration.json`. | stdlib |
| `qwen_engine/cli.py` | The engine's CLI front-end (kept after the 2026-10-12 web pivot; the desktop-GUI idea is superseded by the web workspace — see the W milestones): argparse → validate config → plan via `core.plan()` (adapter-independent) → `generate`; implements the **exit-code contract** (M1 acceptance criteria). `cli_gen.py` at the repo root stays as a thin entry-point shim. | the above only |
| `qwen_engine/web_static/` | **Original static web workspace** (W milestones; design in `docs/web-interface.md`): self-contained HTML/CSS/JS — no framework, no CDN, no external fonts. Since W2 the page makes **only same-origin, relative API calls** (`/api/v1/status`, `/api/v1/plan`, `/api/v1/config/export`) to the local `qwen-web` server — no other network code exists. The earlier "no network requests at all" statement is HISTORICAL W1 evidence (W1 shipped without a server; the speculative capability probe was removed in the 2026-10-12 review). Opened over `file://` or without a server the page shows honest disconnected states. Generate is unconditionally disabled — no generation adapter or job API exists. FreeVideo's web studio was a visual/UX reference only — no code or assets copied, no ComfyUI involvement. | `qwen_engine/web/` (W2 server, stdlib-only) |

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

> Gate: physical installation, cooling and power are Euan's. **Confirmed not
> installed** (user, 2026-10-07 — consistent with the host `lspci` observation).
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
  token count + resolution (C6); prompt token length; **CFG activity** (guidance
  on/off — in this pipeline CFG runs as *separate* cond/uncond forwards with their
  own cache branches, NOT necessarily a doubled batch; cost it as extra forwards,
  not `2×batch`); attention backend; quant format; residency mode
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

### W — Web workspace milestones (2026-10-12 pivot: the web UI is the primary interface; CLI + engine unchanged)

Visual/UX reference only (inspected, not vendored): FlashML-org/FreeVideo `web/`
studio layout — header + left controls / right stable preview, progressive
disclosure, stat cards, creations history. Engine stays ComfyUI-free.

- **W1 (done 2026-10-12, commit `89041f9`; review fixes same day)** — polished
  **static** workspace:
  `src/qwen_engine/web_static/{index.html,app.css,app.js}` + guarded text tests.
  Starts honestly "Not connected"; Generate disabled with a visible reason; no
  server, no jobs, no fake results. Full-browser dark shell, Create/History/
  Settings routing, BigInt decimal-string seeds (`crypto` random), presets,
  lightweight form validation only (server stays authoritative). Design:
  `docs/web-interface.md`.
  **2026-10-12 root/peer review fixes** (see
  `docs/handoffs/2026-10-12-web-w1-review-fixes.md`): the speculative
  `/api/v1/capability` probe is **removed entirely** — W1 makes zero network
  requests and Generate is unconditionally disabled (submit handlers refuse
  directly; the disabled button is not the gate); manual dimension edits
  recompute exact preset `aria-pressed` (exact 1344×768 → Landscape, non-preset
  clears all); `parseSeed` bounds trimmed decimals to ≤ 20 chars **before
  the digit regex and any BigInt** (leading-zero strings > 20 rejected by
  documented policy; full uint64 max kept as an exact string) with **no HTML
  `maxlength`** — a later root finding showed maxlength silently truncates an
  over-length paste into a *different* seed; the original text is kept and
  rejected with an inline explanation; `renderValidation` sets `aria-invalid`
  on prompt/steps/seed/cfg/**both** dimension inputs (the prompt check mirrors
  `config.py`'s non-empty requirement — a root correction of the brief's
  initial claim — surfaced only after editing or submit, never on load); the
  seed gets a full row (all 20 digits inspectable at 380/375/320)
  with helper "Leave at 0 or choose Random." and the precise range retained in
  title/docs; hash nav focuses the new active `tabindex="-1"` heading on
  user/hashchange only, **empty hash routes to Create** (back from History
  after a hash-less start restores Create) and unknown non-empty hashes
  (skip-link `#main`) no longer reroute History/Settings back to Create;
  privacy wording is factual (static hosting
  enforces nothing; locality is enforced by the planned W2 server). New real
  Chrome check script `scripts/chrome_browser_check.mjs` (no dependencies;
  Node built-in WebSocket CDP; **not** part of the guarded suite; requires
  `CHROME_PATH`).
- **W2 (implemented 2026-10-12 — review pending root acceptance, this update)** — runnable **local server + authoritative
  config/plan API**, stdlib-only and ML-free:
  `qwen_engine/web/{api.py,server.py,__main__.py}` + `qwen-web` script +
  `python -m qwen_engine.web`; assets now shipped via setuptools package-data
  and served same-origin. Binds **literal 127.0.0.1 only** (default port 8765;
  `::1` refused until an actually-tested IPv6 milestone; never the harness
  port 3081). Runtime paths (`--manifest/--models-dir/--output-dir/…`) are
  **server-startup-only flags**, canonicalized once, can never come from
  browser JSON, and are omitted entirely when unset (no default writes).
  `GET /api/v1/status` → `qwen_engine.web_status.v1` (planning true /
  generation false with the four honest gates adapter-not-implemented,
  hardware-not-probed, assets-unverified, calibration-unknown; manifest
  **metadata only** — never `resolve_manifest`/`manifest_status`, which would
  stream-hash gigabytes). `POST /api/v1/plan` → exact envelope
  `qwen_engine.web_plan_request.v1` with **only** the 11 editable
  `GenerationConfig` fields, seed as a decimal **string** wire (≤20 ASCII
  digits, uint64 range, never a float/bool/JS number), merged with server
  runtime paths, answered with the **web projection** `qwen_engine.web_plan.v1`
  (never the raw `core.plan` doc — no paths, filenames, hashes or tracebacks
  in any response). `POST /api/v1/config/export` returns the editable fields
  as a raw attachment with the seed as an **exact uint64 JSON number**
  (import is W3-later, not implemented). Typed errors everywhere
  (`qwen_engine.web_error.v1`, 400/413/417/403/405/500, no tracebacks;
  duplicate JSON keys, NaN/Infinity, overflow floats, deep nesting, lone
  surrogates and invalid UTF-8 all refused with typed codes). Transport
  hardening: single exact Host bound to the literal+port, foreign
  Origin/cross-site Sec-Fetch-Site denied on every route, no CORS ever
  (OPTIONS = plain 405), exact `X-Qwen-UI: 1` write gate, Content-Length
  ≤ 65536 enforced before read, no Transfer-Encoding, unsupported Expect →
  417, finite socket timeouts, exact origin-form allowlist routing (no
  `translate_path`, no directory listing), defensive headers + CSP
  `default-src 'none'; connect-src 'self'`. The front end now validates the
  strict status, renders authoritative plans via `textContent` only, and
  downloads export bytes as a raw blob **without ever parsing/reserializing**
  them (seed strings never pass through `Number`). Guarded tests:
  `tests/test_web_api.py`, `tests/test_web_server_guard.py` (pure dispatch,
  policy functions, entrypoint — no sockets, no ML, no writes); the
  real-socket smoke suite is the **separate** command
  `scripts/web_integration_check.mjs` (spawns its own ephemeral-port server,
  Chrome leg strictly opt-in). CLI unchanged; generation stays W3. Design +
  evidence: `docs/web-interface.md`.
- **W3** — generation wiring: job submit/status endpoints over the real engine
  path (`core` + adapter), real results into the preview + History, honest
  timings. Gated on the engine milestones (M3/M4) — the UI never invents output.

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
4. ~~GUI front-end — deliberately out (YAGNI; CLI per spec).~~ **Resolved
   2026-10-12:** pivot to a local web workspace as the primary interface
   (W milestones above; `docs/web-interface.md`). CLI and engine kept.

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
  no Python 3.11 run had been performed **as of that pass** (R7; first exact 3.11
  evidence is dated below under the 2026-10-07 R7 portability note).
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
- 2026-10-07 second senior-review correction pass on top of `ef73df3`, three
  scoped commits, suite 394 passed + 1 conditional skip on Python 3.13 (fake
  workers only; no GPU/model measurements claimed; **no Python 3.11 run at the
  time of that pass** — superseded by the 2026-10-07 R7 portability note below).
  `a5826cd`: `BenchmarkResult` snapshots
  config/model/phases deeply (mapping proxies all the way down — mutation
  *through* `result.phases` / `result.model[i]` now raises and cannot move
  saved output or aggregation); `_jsonsafe.freeze_json` detects container
  cycles by recursion-stack identity (shared acyclic subtrees still legal)
  and bounds depth — cyclic kwargs/metadata/config become path-specific
  typed errors, never `RecursionError`; `CalibrationLog` validates the
  observations container before iterating and snapshots to a tuple;
  mixed-type phase keys report `BenchSchemaError` (no pre-validation
  `sorted()`); `10**400` timeouts / `observed_at` become typed setup/schema
  errors via guarded float coercion; `EmbeddingCache.get` quarantine +
  LRU touch run under the root mutex with a stat-**identity check** before
  the move — the confirmed race where a valid entry published after a
  stale corrupt read got quarantined is closed (event-coordinated
  regression; no barrier inside the mutex). `579d6f9`: the real CLI refuses
  an unconfigured `output_dir` **after** asset verification and **before**
  adapter creation (`core.require_run_config`), so the genuinely absent M4
  adapter can no longer mask config exit 2 with exit 1;
  `true_cfg_scale=10**1000` is a field-level `ConfigError` (exit 2), not
  `OverflowError` (exit 1); manifest `Path.resolve` failures (3.11 may
  raise `RuntimeError` on symlink loops; `OSError` on permissions —
  translated via injected-error tests, **not** an observed-3.11 claim)
  become `ArtifactUnreadableError`/all-unreadable reports, dry-run stays
  structured. `02f0210` (R7): `run_cpu_tests.sh` +
  `tests/cpu_guard/sitecustomize.py` — an audit-hook + meta-path guard that
  activates at interpreter startup in the pytest process **and every
  subprocess fake worker**: writes outside the temp run root (incl.
  fd-relative paths resolved through `/proc/self/fd`), all socket
  connect/sendto/getaddrinfo and heavy ML imports (even executing stubs)
  are refused *before* any side effect; HOME/XDG/TMPDIR/basetemp are fresh
  temp dirs, bytecode and pytest-cache writes disabled, repo reads
  allowed; self-tests prove each denial and that the runner passes real
  fake-worker tests end to end (also verified: guarded full suite = 395
  passed, no `__pycache__`/`.pytest_cache` touched). Exact fresh offline
  command: `bash run_cpu_tests.sh` (any pytest args pass through). Limits:
  the guard is an honest-test harness, not a security sandbox (a hostile
  process re-execing with a stripped environment bypasses it — out of
  scope); no runtime library behaviour changed for R7.
- 2026-10-07 R7 portability fix + exact Python 3.11 evidence: the second
  reviewer's outer sandbox denies **socket creation itself** (plain
  `PermissionError` at `socket()`/`socket.__new__`), so the pre-fix
  `test_network_denied_before_any_packet` — which constructed a socket
  outside its `try` — failed there on **both** Python 3.11.15 and 3.13.5
  (394 passed + that 1 failure, inside that sandbox; 395 outside it). Fix:
  the audit guard now denies **network socket construction** on the
  `socket.__new__` (and legacy `socket`) audit event — fired *before* the
  OS `socket()` call, so no fd exists and the guard's own `GuardViolation`
  surfaces ahead of any outer-policy denial — for every family except
  `AF_UNIX` (local construction allowed; connect/sendto/getaddrinfo/
  gethostbyname remain denied for all families, no skipping, no
  weakening). The self-test no longer depends on the OS allowing socket
  creation: it asserts the guard's denial at construction plus **no fd
  leak**, verifies `socket.connect`/`socket.sendto` hook rejection via
  explicit `sys.audit` events, and exercises a real AF_UNIX `connect`
  when the OS permits it (outer-policy denial tolerated there).
  **Exact evidence (verified 2026-10-07, this session, *outside* the
  reviewer's socket-denying sandbox):** interpreter
  `/tmp/qwen-review.bIFHcI/venv311/bin/python`, `3.11.15 (main, Jul 23
  2026, 14:43:45) [Clang 22.1.3]`, package built/installed offline in a
  clean 3.11.15 venv (`qwen-engine --help` works, no ML deps):
  `PYTHONPATH=/work/src PYTHON=<that python> bash run_cpu_tests.sh` →
  **395 passed**; unguarded `PYTHONPATH=/work/src python -m pytest -q` →
  394 passed + 1 skipped (the guarded-parent test). Same results on
  3.13.5 (`/tmp/qwen-review.bIFHcI/venv/bin/python`): 395 guarded, 394+1
  plain. Audit-event ordering (denial before fd creation) verified on
  both interpreters. *Inside* the reviewer's socket-denying sandbox the
  fixed test needs no successful socket creation — the guard fires first
  by construction — and R7's no-network promise now holds there too; an
  in-sandbox re-run by the reviewer remains the confirming step.
- 2026-10-07 M0.1 fix pass (lead review queue `/tmp/v100-m0-review.md`, on
  baseline `8c55f3b`): the lock is recompiled from a corrected
  `requirements.in` — (a) **transformers now installs from the exact git
  commit** `856157a2f3e9594954310df18fdccc31ffddebe9` (what annotated tag
  v5.17.0 dereferences to, verified via `git ls-remote …refs/tags/v5.17.0^{}`;
  the tag object itself is never pinned), honoring M0.1's "exact git commits"
  for *both* diffusers and transformers instead of the PyPI wheel; installed
  `direct_url.json` records the commits. (b) **Index scoping fixed**: resolve
  and install now run with `--index-strategy unsafe-best-match` (CLI-only
  flag); uv's default first-index strategy let the torch extra index mask
  PyPI and pinned its stale mirrors (certifi 2022.12.7, requests 2.28.1,
  urllib3 1.26.13). The non-torch lock set now provably equals a pure-PyPI
  resolve (verified by diff), torch stays `2.14.1+cu126` (identical wheel
  hashes; `+cu126` exists only on the official cu126 index), and torch's
  CUDA subtree stays 12.x via its own constraints (`cuda-bindings<13,>=12.9.4`).
  (c) **networkx 3.6.1 pinned** for interpreter uniformity (3.7 is
  requires-python ≥3.12), same rationale as numpy. The two per-interpreter
  compiles are now byte-identical (the old markupsafe hand-merge is gone).
  (d) All reproduction/install commands corrected (`uv pip sync --python …`,
  no `pip --require-hashes` — pip cannot hash VCS requirements; uv enforces
  every wheel hash and anchors the two git entries by commit SHA) and
  re-executed verbatim in fresh `/tmp/v100_m0_ml/venv-ml311|313` envs:
  sm_70 in `get_arch_list()` on both, `QwenImage21Pipeline` imports,
  `uv pip check` clean (63 pkgs), `direct_url.json` commits match.
  `/work/.venv` and the `/tmp/qwen-review.bIFHcI` audit envs remain ML-free;
  guarded CPU suite re-run in fresh pytest-only venvs: **395 passed** on
  3.11.15 and 3.13.5. No model weights, GPU, driver or service change; the
  real-CUDA gate (§5 of docs/dependencies.md, M3) remains PENDING.
- 2026-10-07 **M0.2 interface survey** (this commit, on baseline `315705d`):
  `docs/interfaces.md` surveys the exact pinned interfaces (pipeline
  ctor/`encode_prompt`/`__call__`, raw templates + `_drop_idx` + pre-RMSNorm
  hook + masks, CFG-inactive `None` semantics, sigmas precedence, VAE/latent
  scales, local-only loading, default segmented processor vs. Flex and real
  dispatch-backend selection, singlefile Qwen21 mapping + GGUF native fallback
  vs. opt-in remote kernels at capability **≥7**, per-call KV lifecycle and
  shared-pipeline call serialization). v1 stays T2I (user requirement) with one
  image per call (our v1 scope choice); the
  unfixed upstream `image_pad_mask` × `num_images_per_prompt` replication
  hazard is **documented, not patched** (no upstream patch). Added
  `tools/check_pinned_interfaces.py` (stdlib-only AST source-signature
  contract checker, offline, no ML imports): all contracts pass against the
  verified pinned trees and it fails loudly on deliberate signature drift
  (defaults, CFG gate, ≥7 gate, default processor swap, `tie_last_hidden_states`
  arrival); missing input trees exit 2 = "not run", never a silent pass — the
  optional `tests/test_pinned_interfaces.py` skips explicitly without the
  env-provided trees. (Check-count "36/36" in this note is superseded by the
  final-polish note below: now 53 checks, 38 AST-code + 15 markers.) PLAN
  fixes: stale `adapter.plan` references corrected
  (protocol has no `plan`; `core.plan()` is adapter-independent — matches the
  shipped code) and the V100 install row now records the user-confirmed fact
  **card NOT installed** (all pre-existing decisions/corrections preserved).
  Dependency docs honesty fix: `requirements.in` no longer claims uv "builds
  the VCS wheels and enforces their recorded hashes" — VCS entries carry **no
  hash** (commit-SHA source anchor; registry wheels carry the hashes), and the
  isolated VCS *build* tooling is not captured by the runtime lock, so installs
  are not advertised as bit-reproducible builds.
- 2026-10-07 **M0.2 final polish** (this commit; accepted M1 `fd3ba15` and
  M0.1 `315705d` untouched): checker hardening — CFG-gate AST checks now require
  **exactly** the two documented operands (`... or True` / `... and False` is
  drift; regression added); `parse_source` handles a bare `ValueError` from
  `ast.parse` (no `.msg`/`.lineno`) as a clean FAIL + exit 1, never an
  AttributeError (regression added); the pre-RMSNorm-hook registration and
  `finally`-remove checks are scoped to `_get_qwen_prompt_embeds` (a hook
  parked in another method is drift; regression added); checker output now
  states explicitly that a "code" PASS is AST **structure**, never runtime
  behaviour proof. Checker total: **53 checks = 38 AST-code + 15 markers,
  53/53 exit 0** against the pinned trees (was 36/36 in the M0.2 notes).
  `docs/interfaces.md` corrections: duplicated evidence-ledger material
  deduplicated (top taxonomy now defers to the single §9 ledger); stale 36/36
  count fixed; "no tensor was ever run" replaced with the truthful "no model
  loading, no inference performed — NOT a claim that zero tensor code executed
  anywhere"; `GGUFQuantConfig` → **`GGUFQuantizationConfig`**; §6 backend truth
  — decode passes the *processor-specific* `_attention_backend` (class default
  `None`), prefill passes literal `backend=None` to the global registry, and
  no automatic-Flash-fallback-on-Volta is implied; CFG documented as separate
  forwards/cache branches, not a doubled batch (PLAN §5.3 descriptor updated
  to match); T2I (user requirement) separated from one-image-per-call (our v1
  scope choice); clickable commit-permalinks added per section; the mangled §1
  constructor/template/image-token paragraphs were repaired by **mechanically
  copying** the pinned `__init__` source bytes (L212–L232) into a verbatim
  fenced block via a generated patch — the raw token literals never passed
  through the writing model's context and are never retyped by hand; §9 now
  lists checker/tests/input/pin-verification limitations truthfully (revision
  verification remains a separate manual step; the guarded suite never imports
  torch/diffusers/transformers and no model loading/inference has been
  performed anywhere in this project to date). `tests/test_pinned_interfaces.py`:
  15 always-run synthetic regressions (+3 this pass) plus the explicit-skip
  external check. Evidence: guarded suite (R7) **410 passed + 1 skip** without
  `QWEN_{DIFFUSERS,TRANSFORMERS}_SRC` and **411 passed** with them, on both
  3.11.15 and 3.13.5; standalone checker 53/53 exit 0 on the pinned trees
  (commits re-verified by `git rev-parse HEAD` the same day). No weights, GPU,
  driver, service or sandbox-infra changes. M2 manifest is the next separate
  assignment.
- 2026-10-07 **M0.2 final-review fix pass** (this commit; sole writer Qwen,
  lead queue `/tmp/v100-m02-final-review.md`; accepted M1 `fd3ba15` / M0.1
  `315705d` untouched; the M0.2 polish commits `f354df2`/`7408115` stand,
  full-M0.2 acceptance is root's call after this pass): checker hardening —
  CFG-gate checks now require **exactly one assignment** to
  `has_neg_prompt`/`do_true_cfg` in `__call__` (a later overwrite, including
  in a branch, AugAssign or walrus, is drift; a next()-style scan would miss
  it — regression added per target, verified to fail on the pre-fix checker);
  the pre-RMSNorm-hook check is **anchored**: the text-encoder forward call
  must sit inside the try whose `finally` removes that same registered
  handle — registration + a removal in some unrelated `finally` no longer
  passes (forward-outside-protected-try and removes-a-different-name
  regressions added; static shape only — no execution or exception-safety
  claim). `docs/interfaces.md` §6: removed the hardware-observed SDPA
  wording — no GPU is installed; the default registry selects the native
  SDPA implementation and the actual backend/kernel on sm_70 stays
  unverified [U]; a user-selected backend is used as configured (nothing
  surveyed shows automatic fallback). §9 states the CFG-uniqueness and
  hook-anchoring limitations explicitly (AST shape tripwires, not runtime
  proofs). Checker total unchanged: **53 checks = 38 AST-code + 15 markers,
  53/53 exit 0** on the pinned trees (both Pythons; pinned commits
  `c6df88a5`/`856157a2` re-verified by `git rev-parse HEAD` this pass).
  Synthetic regressions now **17** (+2). Evidence: guarded suite (R7)
  **412 passed + 1 skip** without `QWEN_{DIFFUSERS,TRANSFORMERS}_SRC` and
  **413 passed** with them, on both 3.11.15 and 3.13.5 (up from 410+1/411 —
  exactly the two new regressions). The mechanically-copied §1 constructor
  block re-verified byte-equal to pinned L212–L232 after indentation removal
  (nothing retyped; no raw control-token literal entered any model context);
  the §4 CFG display block is token-equal to pinned L677–L686 (alignment
  whitespace only — pre-existing presentation, untouched). No weights, GPU,
  driver, service or heavy-dependency changes; `sandbox.yaml` user edit left
  uncommitted. M2 manifest is the next separate assignment.
- 2026-10-07 **M0.2 residual hook-call fix** (this commit; root repro
  `/tmp/v100-m02-hook-repro.md`; sole writer Qwen; accepted M1 `fd3ba15` /
  M0.1 `315705d` and the prior M0.2 commits stand): the anchored norm-hook
  check's forward matcher was too broad — it accepted ANY call whose dotted
  target *contained* a `text_encoder`/`text_model` path segment, so a
  protected try containing only `self.text_encoder.eval()` satisfied it
  while the real forward sat outside. Replaced with a tiny **exact-call
  predicate** (`_is_direct_encoder_forward`): a forward is the direct call
  `self.text_encoder(...)` (surveyed real form) or the direct bare-name
  call `text_model(...)` (synthetic fixture form); attribute/method calls
  merely containing those words (`self.text_encoder.eval()`,
  `text_model.norm.register_forward_hook(...)`, `...norm(...)`) and dotted
  lookalikes (`self.text_model(...)`, `helper.text_model(...)`) are NOT
  forwards. Deliberately NOT expanded into control-flow/data-flow
  verification — same AST-shape / no-runtime-proof limits as before (a
  raise between registration and try entry stays out of scope).
  Regressions +2: eval-only protected try with the actual direct forward
  after the try/finally plus method-call-only variants (e2e), and a
  classifier positive/negative unit test; both new tests verified to FAIL
  on the pre-fix checker (`d8f2e77`). The real pinned source
  (`outputs = self.text_encoder(**forward_kwargs)` inside the try,
  `handle.remove()` in finally) keeps matching. Constructor snippet and raw
  token literals untouched (never quoted or retyped). Checker total
  unchanged: **53 checks = 38 AST-code + 15 markers, 53/53 exit 0** on the
  pinned trees, both Pythons (pinned commits `c6df88a5`/`856157a2`
  re-verified by `git rev-parse HEAD` this pass). Synthetic regressions
  now **19**. Evidence: guarded suite (R7) **414 passed + 1 skip** without
  `QWEN_{DIFFUSERS,TRANSFORMERS}_SRC` and **415 passed** with them, on both
  3.11.15 and 3.13.5 (up from 412+1/413 — exactly the two new tests). No
  weights, GPU, driver, service or heavy-dependency changes; `sandbox.yaml`
  user edit still uncommitted. Awaiting root M0.2 acceptance; M2 manifest
  stays a separate assignment.
- 2026-10-07 **M2.1 manifest PREPARATION** (this commit; sole writer Qwen, task
  `/tmp/v100-m2-manifest-task.md`; accepted M0.2 `dad7ee0` / M1 `fd3ba15` /
  M0.1 `315705d` untouched): committed schema-v1 `models/manifest.json` —
  **all 24 runtime files** of `Qwen/Qwen-Image-2.1` at immutable revision
  `d26bb61…e1ba3`, upstream subdirectory layout, deterministic filename-sorted
  order; roles model_index.json=`pipeline`, processor/*=`processor`,
  scheduler/*=`scheduler`, text_encoder/*=`text_encoder`, transformer/*=`dit`,
  vae/*=`vae`; upstream README/license/.gitattributes/preview excluded from the
  runtime set (linked from `docs/models.md`). Identities come verbatim from the
  root-verified reference `/tmp/v100-manifest-expected.json` — 16 small regular
  Git files hashed from bounded root fetches + 8 pinned upstream LFS file
  identities (tokenizer.json + 7 safetensors; Git blob SHA1 ≠ file SHA256); the
  canonical-identity digest recomputed from the committed file matches the
  root reference `14147c16…e6ba3`, and size/sha/rename tamper probes each flip
  it. Storage decision: gitignored `models/assets/` only (manifest itself
  committed; config path defaults stay unset; explicit
  `--manifest models/manifest.json --models-dir models/assets` examples are
  dry-run/config-only, not a runnable adapter or fit-proven preset). **No
  downloads, no weights, no network, no model load, no GPU** — expected totals
  (24 files 33,131,601,593 B; 7 weight shards 33,115,613,408 B) are upstream
  metadata, not local measurements; the model-facts paragraph above now carries
  those expected sizes. `tests/test_m2_manifest.py`: 11 offline tests — strict
  schema-v1 parse, independent committed canonical-identity **digest constant**
  (no /tmp dependency at test runtime), exact 24-filename/role fixtures,
  metadata byte totals, extras exclusion, multi-shard roles retained
  (`AmbiguousRoleError` before disk access), deterministic all-missing dry-run
  against fresh temp roots, actual-run **exit 4** naming all 24
  files/repo/revision with no traceback, no adapter import, no network
  (fresh-interpreter + in-process socket-poisoned). Evidence: guarded suite
  (R7) **425 passed + 1 skip** without `QWEN_{DIFFUSERS,TRANSFORMERS}_SRC` and
  **426 passed** with them, on both 3.11.15 and 3.13.5 (up from 414+1/415 —
  exactly the 11 new tests); pinned-interface checker unchanged **53/53 exit
  0** both Pythons (commits `c6df88a5`/`856157a2` re-verified via
  `git rev-parse HEAD`). **Still pending, not done here:** explicit operator
  download of the 33.1 GB set + local sha256/size confirmation; M2.2 CPU
  load-path/fidelity probes; the M4 adapter; M3 GPU (card not installed); M5
  calibration — M2 is *partially* prepared, and none of it is blocked on code
  design solely because the GPU is absent. `sandbox.yaml` user edit still
  uncommitted. Awaiting root review.
- 2026-10-07 **M2.1 fetch tool: CODE + OFFLINE TESTS only** (this commit; sole
  writer Qwen, task `/tmp/v100-m2-fetch-task.md`, root acceptance `511e282`
  respected as the starting point; accepted M1 `fd3ba15` / M0.1 `315705d` /
  M0.2 `dad7ee0` untouched): committed `tools/fetch_models.py` — the explicit
  operator asset fetcher for the committed schema-v1 manifest. Pure stdlib
  (plus the pure-stdlib `qwen_engine.manifest` parser); default/`--help`/plan
  paths are read-only and **socket-free**; the ONLY network/write path is an
  explicit `--fetch` with explicit `--manifest`/`--models-dir` (no implicit
  defaults, no ambient HF cache, no branch fallback — the manifest's pinned
  40-hex revision is the only one the parser accepts). Fixed HTTPS official
  Hub (`https://huggingface.co/<repo>/resolve/<rev>/<file>`, host constant,
  URL-safe repo segments, non-HTTPS redirects refused, signed redirect
  queries stripped from every message); bounded 1 MiB streaming with the
  manifest byte count enforced as an upper bound *while reading*; publish
  only on exact size+SHA256 via atomic **no-clobber** `os.link` of
  `<file>.partial` (never rename/replace over an existing final); honest
  Range resume (append only on 206 + `Content-Range` proven to start at the
  requested offset; Range ignored/misstated → refused, partial untouched;
  final verification always rehashes the whole file from disk, so a corrupt
  partial can never publish); existing verified finals skipped, existing
  bad-identity finals refused never-deleted/overwritten; symlinked paths on
  the write path refused; bounded disk estimate vs fresh `statvfs` free
  printed before any transfer (disk only — **no RAM/GPU fit claim**);
  tool-specific exit codes (0/2/3/4/5/6, documented in `--help` and distinct
  from the engine contract). `tests/test_fetch_tool.py`: **31** offline
  fake-stream tests (injected transport — zero sockets, zero real weights,
  writes only in `tmp_path`): happy path incl. >1 MiB stream, already-good
  skip, mismatched-final never-touched, truncated/oversized/bad-hash/
  gzip-encoded refusals, interrupt→resume with real Range headers, ignored
  Range and four forged/wrong `Content-Range` shapes refused untouched,
  oversized partial, symlinked partial / symlinked write-path dir /
  symlinked final refusals, disk-space refusal before any action,
  publish-no-clobber unit, signed-query error sanitation, manifest branch
  + multi-segment-repo refusals, fresh-child offline plan/help, no-heavy-
  import checks. Engine manifest/resolver/dry-run/no-download semantics and
  the 24-entry pinned manifest are **unchanged**. Evidence: guarded suite
  (R7) **456 passed + 1 skip** without `QWEN_{DIFFUSERS,TRANSFORMERS}_SRC`
  and **457 passed** with them, on both 3.11.15 and 3.13.5 (425+1/426 + the
  31 new tests); pinned-interface checker unchanged **53/53 exit 0**. **No
  real 33.1 GB transfer, no network request, no weights, no model load, no
  GPU/driver/service change** — running `--fetch` against the live Hub is a
  separate lead/operator decision after root review; download ≠ fit (M5) ≠
  GPU evidence (M3). `sandbox.yaml` user edit still uncommitted. Awaiting
  root review.
- 2026-10-07 **M2.1 fetcher fix pass (root review corrections)** (this commit;
  sole writer Qwen, review `/tmp/v100-m2-fetch-review.md` applied in full, on
  top of `e21b93a`/`67cf078`): `tools/fetch_models.py` — (1) an explicit
  `--models-dir` root that **is** a symlink, or sits under an existing
  symlinked path component, is refused BEFORE any mkdir/network/write
  (the resolver's resolved-root boundary can otherwise be redirected
  outside; absent parent components are NOT refused by assumption);
  (2) resume now requires a **numeric** `Content-Range` whose total equals
  the manifest size exactly — wildcard `*` totals refused — and a coherent
  `start <= end < total` span (reversed spans refused); partial untouched;
  (3) diagnostics hardened: `_clean_url` renders only scheme + host/port +
  path (userinfo stripped, default ports collapsed, IPv6 bracketed);
  `HTTPError` reason, `URLError`/`OSError` reason text and untrusted
  `Content-Range` values are never echoed (types only) — sentinel tests
  prove a signed URL in a reason or header cannot leak; (4) partials are
  opened `O_NOFOLLOW` (where available) after the `lstat`, new ones created
  `O_CREAT|O_EXCL` (a concurrently appeared partial is refused, never
  truncated; a non-regular partial path is refused), the opened FD is
  re-verified regular at exactly the resume size before writing, and every
  `os.write` loops until the chunk is fully written (short writes
  completed); (5) a partial already at exactly the manifest byte length is
  **rehashed** — valid → published with no request at all (never an
  impossible Range from EOF), invalid → integrity failure, never published,
  partial removed as unrecoverable; scratch `/tmp/v100-m2-fetch-task.md`
  reference removed from the production docstring. `tests/test_fetch_tool.py`
  **31 → 49** offline tests (injected fake streams, zero sockets, writes
  only in `tmp_path`): symlinked root + symlinked ancestor refusals with
  external dir untouched and transport never called, legitimate missing
  multi-component root still created, complete-partial publish (no network)
  and complete-partial wrong-hash integrity refusal, `O_NOFOLLOW` symlink /
  `O_EXCL` concurrent / size-drift / non-regular `_open_partial` units,
  `_write_all` short-write unit + end-to-end capped-`os.write` fetch,
  wildcard + reversed `Content-Range` params, userinfo/query/reason/header
  sentinel tests. Engine manifest/resolver/dry-run/no-download semantics and
  the 24-entry pinned manifest **unchanged**; `docs/models.md` §4 contract
  synced. Evidence: guarded suite (R7) **474 passed + 1 skip** without
  `QWEN_{DIFFUSERS,TRANSFORMERS}_SRC` and **475 passed** with them, on both
  3.11.15 (uv-cached interpreter, pytest 9.1.1 — same pytest as the 3.13
  venv) and 3.13.5 (`/work/.venv`); pinned-interface checker unchanged
  **53/53 exit 0** on both Pythons, both trees re-verified at
  `c6df88a5`/`856157a2` via `git rev-parse HEAD`. **No real 33.1 GB
  transfer, no network request, no weights, no model load, no GPU/driver/
  service change** — `--fetch` against the live Hub stays a separate
  lead/operator decision; download ≠ fit (M5) ≠ GPU evidence (M3).
  `sandbox.yaml` user edit still uncommitted (deliberately untouched).
  Awaiting root re-review.

- 2026-10-07 **M2.1 fetcher review2 (path-normalization bypass)** (this commit;
  sole writer Qwen, review `/tmp/v100-m2-fetch-review2.md` applied in full, on
  accepted `bd596d8`/`51608ef`): closed the symlink+`..` escape —
  `_check_root_symlinks`'s `os.path.abspath` collapses `..` **lexically**, so
  with `in/link` → `out/deep` the root `in/link/../assets` inspected
  symlink-free `in/assets` while `makedirs`/`open` wrote under `out/assets`
  (A/B probe on the pre-fix tree: rc 0, 1 transport call, external
  `out/assets/a/x.bin` written). `do_fetch` now calls
  `_check_root_no_dotdot(root)` before ANY normalization/mkdir/network/write
  (exit 6, no writes anywhere, transport never called); the read-only
  offline plan is deliberately unaffected. `artifact_url` now also refuses
  whole `.`/`..` repo namespace/name segments (character-legal but they
  collapse inside the URL path; `[A-Za-z0-9._-]` and the two-segment
  contract otherwise unchanged). `tests/test_fetch_tool.py` **49 → 64**
  offline tests: reproduced bypass end-to-end (abspath vs resolve disagree,
  fake transport, refusal pre-network, both trees untouched), plain `..`
  refusal, exact `_check_root_no_dotdot` parametrization (`a/./b` legal,
  any `..` refused), read-only plan unaffected by the new refusal,
  dot-repo malformed-manifest regressions (plan exit 2, fetch exit 2 with
  transport never called, no file written) and dotted-but-valid repo names.
  All prior safety behaviour and `sandbox.yaml` (user-owned, untouched)
  unchanged; engine semantics and the 24-entry manifest unchanged. No
  network, no weights, no model load, no GPU, no service change.

- 2026-10-07 **M2.1 fetcher review3 (final/staging-name collision preflight)**
  (this commit; sole writer Qwen, review `/tmp/v100-m2-fetch-review3.md`
  applied in full, on accepted `810a9ee`): schema-v1 rejects only duplicate
  *final* filenames, so a valid manifest may list both `foo` and
  `foo.partial`; pre-fix A/B probe (fake resume-capable transport, no
  network) proved the destructive path — `--fetch` skipped the verified
  final `foo.partial`, `_resume_offset` mistook it for `foo`'s partial,
  appended over it and unlinked it on the hash mismatch (rc 5, 1 transport
  call, final DELETED). `load_manifest_checked` now runs a static
  `_check_manifest_filename_collisions` for BOTH plan and `--fetch`: any
  artifact whose staged partial path `<final>.partial` equals another
  artifact's final filename (either order, any depth) is a typed usage
  error (exit 2) naming both filenames before any mkdir/network/write
  (post-fix probe: rc 2, 0 transport calls, final byte-for-byte intact,
  mtime unchanged); a filename that is simultaneously another's directory
  (`a` and `a/b`) is refused the same way as the review's suggested
  usability preflight. `tests/test_fetch_tool.py` **64 → 85** offline
  tests: required end-to-end refusal in BOTH manifest orders with the
  verified final preserved byte-for-byte, pre-mkdir refusal, refusal even
  when every artifact verifies (static check), read-only plan typed
  usage error, no-false-positive fetches (lone `foo.partial` final;
  cross-directory `.partial` names), file-vs-directory refusal, and exact
  checker parametrization. The committed 24-entry manifest contains no
  `.partial` filenames (verified `grep -c` = 0) — the fix hardens the
  reusable tool contract, not current data. All earlier fixes, engine
  no-download semantics and `sandbox.yaml` (user-owned, untouched)
  unchanged; the review2 "plan unaffected" note referred to the
  `--models-dir` `..` refusal — manifest-content checks now gate both
  paths identically. No network, no weights, no model load, no GPU, no
  service change.
- 2026-10-07 **M2.2a part 1: local reference-bundle contract — implemented and
  offline-tested** (this commit; sole writer Qwen, task
  `/tmp/v100-m22a-bundle-task.md`, on accepted `a434855`):
  `src/qwen_engine/local_bundle.py` (stdlib-only, ML-free) projects an already
  `resolve_manifest`-verified local file set into the adapter
  `load(config, assets)` shape — role-grouped immutable paths in manifest
  order, explicit component dirs (`dit` → `transformer/`, pipeline = root)
  for future `local_files_only=True` probes — under an **explicit** model
  root only; never a repo id, never an ambient HF cache, never a second
  downloader or rehash of 33 GB. The committed full-reference layout is an
  explicit immutable table (24 files: pipeline 1, processor 9, scheduler 1,
  text_encoder 7, dit 4, vae 2; one pinned repo+revision); per-file SHA/sizes
  stay in the manifest so offline tests use tiny fake contents. A report is
  accepted only if complete, `ok` and **positionally and field-for-field
  equal** to the manifest (`report.resolved` never trusted; paths re-derived
  as `root/filename` and cheap-probed: regular file, resolves inside the
  root — so missing/extra/reordered/forged statuses, escaped symlinks and
  file-vs-directory swaps are refused with typed errors — `LocalBundleError`
  under `ManifestError`, exit 4 — before any ML import). Version-tagged
  canonical **artifact-set fingerprints** (SHA256 over sorted
  repo/revision/filename/sha256/bytes rows) give a whole-bundle digest and an
  encoder+processor digest for future cache keys (any encoder shard/index/
  config/tokenizer/chat-template change flips the encoder digest); digests
  are identity metadata, not load or quality proof. `tests/test_local_bundle.py`
  adds **31** offline tests (tiny 24-file fixture through the **real**
  resolver; order-independent fingerprints; identity-change → relevant-digest
  mapping incl. VAE-only change leaving the encoder digest intact; committed
  manifest layout + anchored bundle digest; every refusal class; inside-root
  symlink tolerated like the resolver; fresh-interpreter ML-free import);
  writes only to `tmp_path` — `models/assets/` (root's concurrent pinned
  fetch) never inspected or touched. Evidence: guarded suite **541 passed +
  1 skip** without / **542 passed** with pinned source envs on 3.11.15
  (`/work/.venv`) and 3.13.5 (`/tmp/qwen-review.bIFHcI/venv`,
  `PYTHONPATH=/work/src`), pytest 9.1.1; pinned-interface checker **53/53
  exit 0** both Pythons, trees re-verified `c6df88a5`/`856157a2`.
  **Pending, unchanged:** 24 real-file hash verification, CPU component
  load, encoder fidelity, adapter generation, RAM/VRAM fit and GPU all
  remain unverified until measured. `manifest.py`, `models/manifest.json`,
  fetcher, engine no-download semantics and `sandbox.yaml` untouched; no
  network, no model load, no GPU, no service change.
- 2026-10-07 **M2.2a review-fix pass: local reference-bundle contract — three
  scoped corrections** (root review `/tmp/v100-m22a-bundle-review.md`; sole
  writer Qwen, on `9f3d0d6`): **(1) CWD-stable paths** — status paths are
  still matched against the caller's *lexical* `root/filename`, but every
  path the bundle **publishes** (`root`, `assets`, `component_dirs`,
  `pipeline_dir`) is absolute under the canonical resolved root, so a
  relative `models_dir` no longer hands a worker launched from another CWD
  unloadable relative paths (evidence: in-process CWD change + fresh child
  interpreter started in a different directory stats all 24 files and 6
  component dirs); absolute-root fixtures unchanged. **(2) Coherent success
  claim** — a report is accepted only when `errors` is empty, every present
  status carries `error is None`, and `report.resolved` equals the
  role-grouped status paths (keys and tuples, manifest order); the
  hand-built `ok=True + errors=(e,) + resolved={}` shape and every
  missing/extra/empty-group/reordered/wrong-path `resolved` variant are
  refused with typed `BundleReportError` — documented honestly as *shape*
  checks, not cryptographic re-verification (forging bytes still means
  defeating the trusted resolver itself). **(3) Defensive immutability** —
  `LocalReferenceBundle.__post_init__` snapshots and freezes `assets` /
  `component_dirs` and their nested path tuples, so mutable containers
  passed to the public frozen dataclass cannot mutate the "immutable"
  bundle afterwards; the class docstring states plainly that direct
  construction performs **no** resolver verification.
  `tests/test_local_bundle.py` **31 → 39** offline tests (relative root +
  CWD change + fresh child from another CWD; absolute-root regression;
  forged `errors` / per-status `error` / `resolved`-mapping classes incl.
  the review example; direct-construction mutation); the forged-report
  fixture is now internally consistent so the probe tests actually probe.
  `docs/models.md` (gates row + section 8) and this PLAN module row synced
  to the same contract-only precision — no load/fidelity claims added.
  Evidence: guarded suite **549 passed + 1 skip** without / **550 passed**
  with pinned source envs, on 3.11.15 (`/work/.venv`) and 3.13.5
  (`/tmp/qwen-review.bIFHcI/venv`, `PYTHONPATH=/work/src`), all rc 0;
  pinned-interface checker **53/53 exit 0** both Pythons, trees re-verified
  `c6df88a5`/`856157a2` via `git rev-parse HEAD`. **Pending, unchanged:**
  24 real-file hash verification, CPU component load, encoder fidelity,
  adapter generation, RAM/VRAM fit and GPU remain unverified until
  measured; asset-download status stays root's to confirm.
  `models/assets/` (root's concurrent pinned transfer), `manifest.py`,
  `models/manifest.json`, the fetcher and `sandbox.yaml` untouched; no
  network, no weights read, no model load, no GPU, no service change.
- 2026-10-07 **M2.2a bundle review2: pinned profile identity, resolver
  provenance, closed-tree preflight, cache-key scope, constructor
  invariants + factual asset status** (root review
  `/tmp/v100-m22a-bundle-review2.md`; sole writer Qwen, on `4c95ea1` +
  `0251420`): **(1)** the named `full-reference` profile now enforces **this
  model revision** — new constants `FULL_REFERENCE_REPO`
  (`Qwen/Qwen-Image-2.1`) and `FULL_REFERENCE_REVISION`
  (`d26bb61231c349cf6b7896fa83353113880e1ba3`) checked by
  `validate_local_reference_layout`; internally consistent manifests at any
  other repo or revision are refused (a new model revision requires a NEW
  reviewed profile), while `artifact_set_fingerprint` stays a generic
  identity function that demonstrably drifts on repo/revision change;
  **(2)** honest resolver provenance as an API split — trusted factory
  `resolve_local_bundle` (validates layout + pinned identity first, then
  runs `resolve_manifest` itself: explicit full-rehash cost, never a silent
  metadata call) vs lower-level `project_resolution_report`, documented as
  **trusting its caller's report** (a public dataclass report can be
  forged; shape checks are not authenticated evidence); both state the
  hostile-filesystem TOCTOU limit (component load must reverify immediately
  before library reopen); **(3)** stdlib closed-tree preflight
  `preflight_closed_tree` (function + bundle method) enumerating every
  entry under an explicit root and refusing extras with typed
  `BundleTreeError` — lookalike config/weight/code files, empty dirs,
  leftover `*.partial` debris, symlinks even inside-root — no contents
  read; opt-in, never on an actively downloading root; **(4)** the encoder
  fingerprint is documented as ONLY the asset component of the future
  worker cache key (worker protocol/implementation revision + exact
  diffusers/transformers pins required; wiring `EncoderRequest.model_sha256`
  to the asset digest alone is a contract violation; full key = worker
  milestone); **(5)** public direct constructors police scalar invariants
  (profile, absolute+consistent `root`/`root_resolved`, exact pinned
  repo/revision, 64-hex digests) with typed errors — data hygiene, not
  resolver verification — and `artifact_set_fingerprint` rejects
  non-ModelArtifact and duplicate identities instead of a silently
  collapsed plausible digest. **Factual asset status** (was stale prose):
  root's explicitly approved pinned fetch ended 2026-10-07 21:53 UTC with a
  timeout on the first transformer shard; an independent offline
  resolver/plan found 19 of 24 final files present and size+SHA256-verified,
  5 missing, 0 bad; recoverable
  `transformer/diffusion_pytorch_model-00001-of-00002.safetensors.partial`
  = 6,302,990,336 B; remaining expected transfer after it 9,278,315,946 B;
  processor and text-encoder groups complete. Separately: a socket-denied
  processor-only load reached the pinned transformers code but failed for
  missing `torchvision`; a tokenizer-only local load succeeded
  (Qwen2Tokenizer, 151,669 entries, 9 tokens for root's sample) —
  dependency work is a later scoped milestone. `docs/models.md` (status
  paragraph, §2–§8) and this PLAN (model-facts paragraph + module row)
  synced; no claim that no network transfer has ever occurred, and no full
  load/fidelity/fit claim added. `tests/test_local_bundle.py` **39 → 57**
  offline tiny-file tests (all six items incl. content-change factory
  refusal vs forged lower-level report, preflight refusal classes, scalar
  invariant matrix). Evidence: guarded suite **567 passed + 1 skip**
  without / **568 passed** with pinned source envs on 3.11.15
  (`/work/.venv`) and 3.13.5 (`/tmp/qwen-review.bIFHcI/venv`,
  `PYTHONPATH=/work/src`), all rc 0; pinned-interface checker **53/53 exit
  0** both Pythons, trees re-verified `c6df88a5`/`856157a2` via
  `git rev-parse HEAD`. **Pending, unchanged:** the remaining 5 files +
  transfer resume (root/operator decision; default sandbox DNS denied),
  real-asset resolve, CPU component load, encoder fidelity, adapter,
  RAM/VRAM fit, GPU. `models/assets/` (incl. the interrupted partial),
  `manifest.py`, `models/manifest.json`, the fetcher, `sandbox.yaml`
  (user-owned) and `models/` assets untouched and unread by this pass; no
  network, no model load, no GPU, no service change.
- 2026-10-09 **M2.2b stage 1 item 1: production encoder implementation
  identity in the generic protocol/cache — foundation implemented and
  offline-tested** (sole writer Qwen, task
  `/tmp/v100-m22b-foundation-task.md`, on accepted `3ce32fd`; shared vault
  read-only this session — `docs/handoffs/2026-10-09-resume.md` is the local
  fallback, no memctl writes, `sandbox.yaml` user edit untouched):
  `qwen_engine/encoder.py` gains the validated immutable
  **`EncoderImplementationIdentity`** record — encoder artifact-set SHA256
  (the **asset component** of the future worker cache key, never the whole
  key), worker protocol + implementation revisions, prompt-extraction
  algorithm revision, exact 40-hex diffusers/transformers commits
  (constants `PINNED_DIFFUSERS_COMMIT` `c6df88a5…650ce` /
  `PINNED_TRANSFORMERS_COMMIT` `856157a2…ebe9` anchor the known source pins;
  installed-package verification against them is a LATER loader step),
  torch build + version, safetensors version and torchvision version (an
  explicit absence marker is the honest value while torchvision is missing —
  never null, never fabricated). Folded into `EncoderRequest` as the
  optional `implementation_identity` with a deliberate schema bump: request
  wire `qwen_engine.encoder_request.v2`, key
  `qwen_engine.embed_cache_key.v3`; all model/component/prompt/negative/
  dtype/kwargs/image key fields and the deep-freeze guarantee preserved
  (the record is a frozen string record; `to_dict`/`key_document` return
  fresh documents, so no caller-side mutation can move key or wire).
  Existing generic fake requests may explicitly carry **no** identity
  (key-document `implementation_identity: null`; absent parses identically
  to null); a future production worker must require one — this is identity
  **plumbing**, not any claim that a production encoder ran. Parsing is
  strict through the existing typed errors (unknown keys, missing fields,
  explicit nulls, non-hex commits/digest, blank/whitespace/non-string
  revisions, non-mapping). `tests/test_encoder.py` **107 → 141** offline
  tests (schema tags + pin constants anchored; exact wire/key roundtrip;
  every one of the 10 fields changes the key; 17-class malformed-value
  matrix; strict identity and request-level parsing refusals; frozen-record
  + poisoned-snapshot mutation cannot move key or wire; generic request
  without identity unchanged). Evidence: guarded suite **601 passed + 1
  skip** without / **602 passed** with pinned source envs on 3.11.15
  (`/work/.venv`) and 3.13.5 (`/tmp/qwen-review.bIFHcI/venv`,
  `PYTHONPATH=/work/src`), rc 0; pinned-interface checker exit 0, trees
  re-verified `c6df88a5`/`856157a2` via `git rev-parse HEAD`; the skip is
  the documented external-tree skip, never counted as a pass. **This proves
  nothing about real loads:** encoder load, safetensors payload, numeric
  parity, torchvision dependency, adapter and GPU stay pending; no network,
  no GPU, no model load, no permission request.
- 2026-10-09 **M2.2b stage 1 item 2: bounded worker responses — foundation
  implemented and offline-tested** (sole writer Qwen, same task; corrects
  the item-1 entry's encoder test count to **107 → 141**):
  `EncoderWorker` gained the validated **`max_response_bytes`** (positive
  int, not bool; typed `EncoderSetupError`; documented default
  `DEFAULT_MAX_RESPONSE_BYTES = 256 MiB` — a real result's base64 blobs are
  a few MiB, so the bound leaves orders of magnitude of headroom while
  hard-capping what a child can push into engine RAM through one response
  file). Enforcement is double by design: a **pre-read `stat`** refuses an
  already-oversized file **before opening it** (typed
  `EncoderProtocolError` naming actual size and cap — no read, no JSON, no
  base64), and the read itself is **bounded to `cap + 1` bytes**, so a file
  that was small at stat time but **grows before/during the read** is
  refused as "grew past" the cap rather than decoded unbounded. No new
  transport, no pickle; child reaping, terminal failure events and stderr
  diagnostics on every failure branch are untouched. `tests/test_encoder.py`
  **141 → 153** (typed cap matrix incl. bool/float/zero/negative/None;
  default constant anchored; end-to-end tiny-cap refusal with a perfectly
  valid but oversized fake response; **exact byte boundary** accepted via a
  new fake-worker `_response_bytes_file` recorder — same request and
  request_id yield byte-identical responses, so cap == size passes and
  size-1 refuses; stat-path refusal on a pre-created oversized file with the
  exact size in the message; deterministic stat-vs-read growth simulation
  through a procfs file that stats 0 bytes but reads longer than the cap,
  no race, no sleep). Evidence: guarded suite **613 passed + 1 skip**
  without / **614 passed** with pinned source envs on 3.11.15 and 3.13.5,
  rc 0; ordinary fake-worker behaviour unchanged (full prior suite intact).
  This bounds one file's size — it is **not** a memory-fit or load claim;
  real encoder load, payload and fidelity remain pending. No network, no
  GPU, no model load, no permission request; `models/assets/`, manifest,
  fetcher and `sandbox.yaml` untouched.
- 2026-10-09 **M2.2b stage 1 item 3: host-memory preflight for the CPU
  encoder child — foundation implemented and offline-tested** (sole writer
  Qwen, same task): new stdlib-only `qwen_engine/mem_preflight.py`, scoped
  to the documented Linux cgroup v2 target (NOT the future M5.3 `memory.py`
  policy, NOT a general resource framework). Reads host `MemAvailable` and
  this process's cgroup v2 `memory.max`/`memory.current` (deepest-existing
  `memory.max` on the walk to the cgroup root; `max` = unlimited; no `0::`
  line = **unknown**, never assumed unlimited), applies the **tighter
  applicable budget** — with a finite cap the decision is bounded by
  `max - current` even when host RAM looks plentiful (the cap is never
  substituted away), while an unknown constrained state **refuses**:
  unreadable/unparseable/negative counters, a cap without its usage
  counter, missing `MemAvailable` under an unlimited cap → refusal, never
  a safety assertion. Required bytes = checkpoint total × explicit
  dtype-expansion ratio (rounded up, shrink clamped to the checkpoint
  total) + explicit non-defaulted runtime headroom. `require_encoder_memory_fit`
  raises typed `MemoryPreflightError` (memory-refusal exit-3 family)
  **before any ML import, weight read or loader call**. Passing is stated
  — module, report field `conservative_precondition` and reason text — as
  a **conservative prerequisite, not measured model-fit evidence**; it
  alone never proves a load fits. `tests/test_mem_preflight.py`: **40**
  offline tests on injected tiny proc/cgroup fixtures only — the real
  sandbox scenario (17,534,339,488 B checkpoint vs the 17,179,869,184 B
  16 GiB cap refuses both with fp32 expansion and bare bf16), tighter-host
  and tighter-cgroup selection, unlimited `max`, exact budget boundary,
  current-at-cap zero room, nested-cgroup walk (deepest wins, fallback to
  root), every invalid-counter/unknown-state class, dtype arithmetic
  (ceil-doubling, shrink clamp), input validation matrix, snapshot
  injection that never touches the paths, JSON-safe tagged report, no
  real proc reads, no asset reads, ML-free import. Real-sandbox read-only
  diagnostic (2026-10-09, stdlib only, no model or asset reads): snapshot
  `host 32,903,032,832 B / finite cap 17,179,869,184 B / current
  10,792,022,016 B` → **refuse** (budget 6,387,847,168 B ≪ required
  37,216,162,624 B) — the preflight now says the known-out-loud verdict
  before anything heavy would run. Evidence: guarded suite **653 passed +
  1 skip** without / **654 passed** with pinned source envs on 3.11.15 and
  3.13.5, rc 0. Real encoder load, safetensors payload, numeric parity,
  adapter, measured RAM/VRAM fit and GPU remain pending their actual
  dependencies; no network, no GPU, no model load, no permission request;
  `models/assets/`, manifest, fetcher and `sandbox.yaml` untouched.
 - 2026-10-09 **M2.2b item 9 senior-review correction: fail-closed cgroup
   scope gate + typed giant MemAvailable** (sole writer Qwen, review2 task
   after deliberate interruption of the first correction draft; valid
   draft hierarchy/counter/host/unit/integer fixes preserved): root+peer
   probes (`/tmp/v100-memory-review-probe.py`,
   `/tmp/v100-memory-scope-review.py`) reproduced that the draft still
   proceeded with a **missing intermediate** `memory.max` and with a
   **present** root file whose completeness was unproven, plus a raw
   `ValueError` on a unit-correct 5000-digit `MemAvailable`. Corrections:
   **(1)** a missing `memory.max` at ANY visible level is unknown —
   `ENOENT` does not prove the controller is disabled (missing/mismapped
   directory possible, no controller evidence read) — never silently
   skipped, with already-seen finite caps still named truthfully;
   **(2)** visible-root completeness is never assumed: a namespace `/` can
   be a non-root cgroup under a tighter hidden ancestor, so real
   **autodiscovery returns unknown/refuse regardless of the visible
   root's `memory.max` contents** (present finite or `max`) — an honest
   fail-closed gate, documented; the default deployment may legitimately
   refuse every real load until trusted complete host-budget evidence
   exists; the walk still computes the tightest visible minimum of
   `limit - current` (sibling usage included, pure selection helper +
   truthful per-level source notes) but only as an **advisory visible
   availability** — an UPPER bound on true availability that hidden
   ancestors can only tighten (root's 10:22 clarification: required bytes
   are the allocation lower bound; never label observed availability a
   lower bound) — the actual 17,534,339,488 B > visible 16 GiB refusal
   evidence survives the gate as an explicit guaranteed-refusal reason;
   **(3)** `MemAvailable` digits beyond the interpreter's int-conversion
   safety limit are caught LOCALLY as unknown (interpreter limit never
   raised, value never approximated). An explicitly injected
   finite/unlimited `MemorySnapshot` remains **trusted caller evidence**
   for pure budget arithmetic/pass tests (documented); `budget()` never
   consults the advisory field; no root-is-complete default, no production
   bypass, no broad mount/namespace parser.
   `tests/test_mem_preflight.py` **40 → 67**: read-path tests fail closed
   on unknown root scope; pass/arithmetic tests use trusted injected
   snapshots; hierarchy tests still demonstrate the minimum of EACH finite
   `limit - current` and truthful provenance notes. Evidence: both root
   probes fully green; guarded focused **67 passed** on 3.11.15 and on
   pinned-source 3.13.5; guarded full **732 passed + 1 skip** (3.11) and
   **733 passed** (3.13 pinned), rc 0. Real-sandbox read-only diagnostic
   (stdlib only, no model/asset reads): the checkpoint request now
   **refuses** with `cgroup_state unknown`, the advisory visible
   availability (16 GiB cap minus current) and the truthful
   lower-bound-above-upper-bound guaranteed-refusal reason — the honest
   gate live. No network, no GPU, no permissions;
   `models/assets/`, manifest, fetcher and `sandbox.yaml` untouched.

- 2026-10-12 **W-milestone pivot (user directive)**: primary interface is now
  the local web workspace (was "GUI out, CLI per spec"); engine and CLI kept.
  W1 shipped as static original assets (`web_static/`, FreeVideo-inspired
  layout only — no code/asset/ComfyUI reuse) with honest disconnected states;
  server + generation deferred to W2/W3. No dependency, entrypoint or
  packaging changes; `sandbox.yaml` and ML locks untouched.

- 2026-10-13 **M2.2b CPU boundary follow-ons (Codex direct implementation,
  after the user ended Qwen writing)**: supplied upstream helper/codec accepted
  8eefa81; manifest/request/executing-runtime identity accepted 172cbc8; narrow
  actual-memory-gated 16-file verifier accepted 0390f0d; local-only CPU worker
  now connects these boundaries with strict bounded reads, safe index references,
  fixed-class local CPU/safetensors loading, offline socket guard, bounded wire
  response and atomic no-clobber publication. Full guarded suites 1160/2skip on
  Python3.11.15/3.13.5; real pinned worker checks12/12 each, including actual
  generated miniature checkpoint loads in all three dtypes (no downloaded real
  processor/checkpoint, no GPU). Full-project review, runtime/transitive cache
  identity audit and production activation are still pending; torchvision and
  actual memory prerequisites continue to refuse. Web generation stays disabled.
  One stdlib-only worker entrypoint added, ML locks and user sandbox.yaml unchanged.
  Review envs/runtime/cache and reusable checks persist under /work, not /tmp;
  disposable generated fixtures alone are temporary. See docs/encoder-worker.md.

- 2026-10-13 **CPU readiness progression (Codex direct work)**: exact
  torchvision0.29.1+cu126 joins the hashed runtime lock/unified64-package
  policy; all prior63pins/hashes retained. Native CPU vision operations and
  actual local nine-file processor loading accepted on both pinned Python
  versions (docs/vision-review.md), not full encoder fidelity. Internal
  stdlib-only bounded job controller and exact phase/runtime/device memory
  policy groundwork are tested, NOT wired to CLI/web and NOT a production
  executor or fit override (docs/job-controller.md, docs/memory-policy.md).
  Previously approved base fetch resumed/completed: all24files verified,
  33,131,601,593B, settled29-entry tree/no partials. Actual config/full META
  DiT/VAE shapes match checkpoint headers; tiny random CPU sampler/CFG/KV/VAE
  checks pass (docs/component-cpu-review.md), not accepted real images.
  Actual encoder memory still refuses17.53GB weights against16GiB cap and
  unproven cgroup scope. Real adapter/process executor/calibration, full
  model fidelity and V100/numerics/quality acceptance remain pending;
  Generate stays disabled and user-owned sandbox.yaml remains untouched.

- 2026-10-24 **host test environment rebuilt + CPU baseline (step 1 of testing
  against the installed card)**: the NVIDIA ComfyUI service was stopped by user
  approval (GPU back to 0MiB used); the project `.venv` was recreated on the
  host with uv (CPython 3.13.14) from `requirements.lock` (64 packages:
  torch2.14.1+cu126, torchvision0.29.1+cu126, diffusers0.41.0.dev0,
  transformers5.17.0 at git856157a2, sm_70 in `get_arch_list()`), pytest9.1.1,
  and an editable install of this package. Guarded R7 suite: **2435 passed +
  2 skipped** (no torch/diffusers/transformers imported by tests, writes only
  in temp). Real CUDA execution from the locked environment on the V100:
  cc(7,0), FP16 512x512 matmul max abs deviation 0.050 vs an FP32 CPU
  reference, FP32 elementwise exact, 22MiB reserved; no model load and no
  generation. The default loader is still absent (`--run` exits 1 by design).
  One test correction: `test_actual_fresh_child_package_refusal_happens_before_cache`
  had assumed the ambient child interpreter lacked torchvision (true on
  2026-10-13, false now that the reviewed lock installs it); it now uses a
  deterministic wrapper interpreter that hides only the vision wheel at the
  metadata boundary, with the production worker unchanged. Found by the fresh
  environment run, not a product regression. Real-model fidelity, encoder
  RAM/fit, VRAM calibration and quality acceptance remain pending.

- 2026-10-24 **default loader implemented + real V100 component proof (step 2
  of testing against the installed card)**: `adapters/qwen21_diffusers.py`
  (`Qwen21DiffusersAdapter`) is now the adapter behind `cli._make_adapter()`.
  Import stays ML-free; `load()` refuses any non-`fp16`/non-`resident`/eager-
  or-efficient-attention policy and any missing `manifest_path`/`models_dir`/
  `output_dir`/`cache_dir` *before* a manifest read or ML import, re-resolves
  the local bundle (never trusts a caller path list), checks the pinned runtime
  (CUDA 12.6, sm_70, V100-16GB band), loads processor/transformer/VAE/scheduler
  locally one component at a time (move to GPU, then read the next, to bound
  host peak), and delegates the five phases to `SuppliedQwen21Adapter`. Asset
  comparison resolves the CLI's lexical paths against the bundle's canonical
  absolutes. Tests: `tests/test_qwen21_adapter.py` (ML-free boundary, policy
  and lifecycle refusals) plus updated `test_cli.py` ordering/factory tests.
  Guarded suite: **2452 passed + 2 skipped**.
  Real run on the card (`--run`, 512x512, 4 steps): 24 assets verified, bundle
  resolved, runtime identity bound, transformer+VAE resident FP16, then the
  documented deliberate refusal at the encoder's host-memory gate (exit 3,
  `cgroup memory constraint state unknown` -> `refuse`); GPU returned to 0MiB.
  Component-level proof with synthetic conditioning (`/tmp/opencode/qwen_component_smoke.py`):
  denoise 4 steps ~1.0s, VAE decode OOMs **without** tiling at 512x512
  (resident 14,806MiB peak) and succeeds with `vae.enable_tiling()`; PNG saved.
  That is load/kernel/sampling/decode evidence only — not prompt fidelity, not
  an accepted image, and `--tiled-decode` is required on this 16GB profile.

- 2026-10-24 **first real text-to-image images on the V100 + tiled-decode
  quality fix**: `--attest-unbounded-host` resolved the last encoder blocker
  (opt-in operator scope attestation; default stays refuse) and the real CLI
  produced prompt-following images: 512²/4 steps and 1024²/40 steps, the
  latter in 1m43s cold / 1m27s warm, denoise 42.1s, decode 0.85s, peak
  device VRAM 15,439 MiB. Prompt fidelity confirmed by eye on "a red cube on
  a white table, studio photo" at both sizes. A measured defect was found and
  fixed: VAE **tiled** decode leaves faint vertical dashed streaks in flat
  regions (max 52/255, mean 1.24 vs the same latents decoded untiled), so the
  loader now releases the transformer after denoise (`pipeline.transformer =
  None` **plus `gc.collect()`** — diffusers' per-module hook registry keeps
  the module in a cycle; 13.89 GiB → 0.64 GiB measured) and decodes untiled by
  default. `--tiled-decode` stays as an explicit memory-first knob and its
  help now names the artifact. fp16 vs fp32 VAE decode differs by ≤1/255, so
  precision is not the cause. Measurements and honest limits (exploratory,
  not calibration; ~15.4/16 GiB is tight, one transient allocator retry;
  2K still needs M5) are in docs/generation-measurements-2026-10-24.md.
  Follow-up probes the same day: text rendering is correct (a legible
  hand-painted "OPEN" beach sign at 1024²/40/seed7); true CFG (negative
  prompt, scale 4.0) fits at 512² (15,889 MiB peak) and at 1024² with one
  transient allocator retry, but looks **worse** than scale 1.0 at equal
  steps (saturated/plastic vs natural shading), so the default scale 1.0 is
  the recommended quality path. Recorded in the same doc.
  Known stale surface, deliberately not changed here: the W2 web status
  endpoint and static page still say `adapter_not_implemented` / "generation
  is not implemented". The web process genuinely does not import or load the
  adapter, so it still cannot generate, but the *reason* text is now wrong.
  Fixing it spans api `_gates`/`_blockers`, the closed gate/blocker validator
  in `web_static/app.js` (it rejects any payload with a different gate set, so
  a server-only edit breaks the page), the handler guard/copy and both web
  test files. Recorded in docs/web-interface.md.
  Reproducibility checks (512²/20/seed42): two consecutive runs are
  byte-identical (sha256 69726d86…), and a third run with a freshly emptied
  embedding cache (cold encode, 1m05s) is byte-identical too — the disk
  embedding cache returns the conditioning it stored without perturbing
  sampling.

- 2026-10-24 **2048² works on the 16 GB V100 (experimental)**: the first 2K
  attempt OOM'd during denoise (15,983 MiB device peak) on a 386 MiB SwiGLU
  intermediate, not on weights. Three targeted changes, all exact/measured:
  (1) phase-exclusive residency — release the VAE before denoise
  (13.89→13.25 GiB), not just the transformer before decode; (2) above 4096
  latent tokens, evaluate each block's token-wise image MLP in 2048-token
  chunks (all 32 blocks wrapped; 2K denoise peak 14.79 GiB allocated /
  15,835 MiB device) — 1024² is exactly 4096 tokens and stays unchanged;
  (3) tiled decode with 1024 px tiles / 768 px stride and an fp32 VAE.
  The library default tiling (256/192) left a visible dashed grid at 1:1 and
  the fp16 tiled path was **non-finite** at 2048²; with the new geometry the
  tile boundary is below the image's own row/column noise (0.026 vs median
  0.223 at 767→768), and fp16 vs fp32 differs by ≤2/255. 2K/20 is sharp and
  prompt-faithful in 3m39s (denoise 149.5s ≈7.5s/step, decode 7.6s); 2K/8
  was visibly under-converged, so 20 steps is the documented minimum.
  `--tiled-decode` is required at 2K and its help now says so; 1024² keeps
  the cleaner untiled default. Full record with commands and honest limits in
  docs/generation-measurements-2026-10-24.md. MLP chunking and tiling
  geometry have ML-free tests; guarded suite: 2490 passed, 2 skipped.

- 2026-10-24 **fixed quality suite (`fixed-quality-v1`)**: added
  `tools/quality_suite.py` — six fixed prompts/seeds covering object, text,
  composition, counting, style and lighting, run through the real CLI with
  per-case hashes, phase timings, sampled device/host peaks and a captioned
  contact sheet. Measured 1024²/40 on the V100: six of six exited 0,
  107–109 s wall each, denoise 42.8–43.0 s, decode 1.26–1.29 s, peaks
  15,373–15,379 MiB, host MemAvailable ≥ 52.7 GiB. Review: object, text,
  composition, counting and style pass; lighting is partial (strong backlight
  and caustic, but the requested long shadows are not visible). The object
  image is byte-identical to the pre-change run
  (`sha256 8d274d4f…`), so phase-exclusive residency/chunking did not perturb
  1024². Record: docs/quality-suite-2026-10-24.md; review status is agent
  visual inspection with human sign-off pending.
