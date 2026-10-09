# Handoff — M2.2b stage 1 foundation complete (2026-10-09)

Agent: Qwen (sole product-code writer, DeepSeek Harness), task
`/tmp/v100-m22b-foundation-task.md`, working base accepted `3ce32fd`.
Shared vault remained read-only all session (no memctl writes; local notes
only). `sandbox.yaml` user edit (host port 3082→3081) untouched and
uncommitted; `models/assets/` (incl. the recoverable partial), manifest,
runtime lock, fetcher and services untouched. No network, no GPU, no real
model load, no dependency installs, no permission requests.

## Commits (three scoped implementation commits + admin/docs)

- `88d8acb` docs: administrative resume note (pre-existing unstaged).
- `6094887` **item 1** — `EncoderImplementationIdentity` (frozen, validated:
  encoder artifact-set sha256 = asset component only; worker protocol +
  implementation revisions; prompt-extraction revision; exact 40-hex
  diffusers `c6df88a5…`/transformers `856157a2…` commits via `PINNED_*`
  constants — installed-package verification is a later loader step; torch
  build/version; safetensors and torchvision versions, absence stated as an
  honest marker, never null/fabricated). Folded into `EncoderRequest` as
  the optional `implementation_identity`; schema bump
  `encoder_request.v1→v2`, `embed_cache_key.v2→v3`; deep-freeze preserved;
  generic fake requests may carry none; production workers must require
  one. Identity **plumbing** only — no production-encoder claim.
- `471b05d` **item 2** — bounded worker responses:
  `EncoderWorker.max_response_bytes` (validated positive int; documented
  default 256 MiB) enforced by pre-read `stat` **and** a `cap+1`-bounded
  read before any JSON/base64; growth after stat refused, not decoded;
  reaping/events/stderr intact; no new transport, no pickle. Fake worker
  gained the deterministic `_response_bytes_file` recorder for exact-boundary
  proofs.
- `9dd0201` **item 3** — `qwen_engine/mem_preflight.py`: stdlib Linux
  cgroup-v2-scoped preflight (host `MemAvailable`, `memory.max`/
  `memory.current`, tighter-applicable-budget rule, finite cap never
  substituted away, unknown state refuses, explicit dtype expansion +
  explicit headroom, typed `MemoryPreflightError` before any ML import /
  weight read / loader call). Documented as a **conservative prerequisite,
  not measured model-fit evidence**.

## Test evidence (all via `run_cpu_tests.sh`, guard active)

- Python 3.11.15 (`/work/.venv`), no pinned source env:
  **653 passed + 1 skipped** (the documented external-tree skip, never a
  pass), rc 0. Per-item checkpoints: 601+1 → 613+1 → 653+1.
- Python 3.13.5 (`/tmp/qwen-review.bIFHcI/venv`) with
  `PYTHONPATH=/work/src` and pinned trees
  (`QWEN_{DIFFUSERS,TRANSFORMERS}_SRC=/tmp/v100_m0_ml/src/*`):
  **654 passed**, rc 0 (includes the external pinned-interface
  checker test). Checker standalone exit 0; trees re-verified
  `git rev-parse HEAD` = `c6df88a5…`/`856157a2…`. Main suite is
  torch/transformers/diffusers-free by guard + import tests.
- Focused files: `tests/test_encoder.py` 153, `tests/test_mem_preflight.py`
  40; base encoder suite (141) intact — ordinary fake-worker behaviour
  unchanged.
- Real-sandbox read-only preflight diagnostic (stdlib only, no asset/proc
  model reads): cap 17,179,869,184 B, current 10,792,022,016 B, host
  available 32,903,032,832 B → **refuse** (budget 6,387,847,168 B vs
  required 37,216,162,624 B for the 17,534,339,488 B checkpoint at fp32
  expansion + 2 GiB headroom).

## Status vs pending

Foundation **implemented and offline-tested**: worker identity plumbing,
response-size bounding, memory preflight. **Pending, unchanged:** loader
(installed-package checks against the pins; production identity
construction from the real environment), safetensors payload, real numeric
parity, torchvision dependency, adapter generation, measured RAM/VRAM fit,
V100/GPU. Full-reference manifest stays safetensors; GGUF/single-file
probes belong to a separate reviewed profile. Next: root review of the
three commits, then stage 2 worker (`/tmp/v100-m22b-worker-task.md`).
