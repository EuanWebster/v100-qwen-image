# Encoder CPU performance — where the 186.6 s reference-image encode goes

Investigated 2026-10-25 from the real V100 run log `output/gpu/batch2/edit-1ref.log`
(`text_encoder_request: 186.601s`, prompt + ONE 1024² reference, bf16, cache
miss) versus ~19.5 s text-only. Driver: `output/gpu/encoder_profile_q2.py`
(variants re-run the REAL encoder child through `ProductionEncoder` with
private caches; every variant's payload gets a canonical sha256 so speed wins
can be checked for BIT-IDENTICAL output). Results: `output/gpu/q2-profile/results.jsonl`.

## Facts established by code reading (verified)

1. **The child is deliberately single-threaded.** `encoder_worker.install_offline_guard()`
   pins `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` and `_load_classes()` calls
   `torch.set_num_threads(1)` + `set_num_interop_threads(1)` before any ML
   import. This dominates everything else: the whole ~7-8 B Qwen3-VL forward
   (vision tower + language model) runs on ONE core. Text-only pays it too
   (19.5 s), and the image path multiplies it.
2. **Attention is forced to `attn_implementation="eager"`** in
   `_load_local_components`. Eager materializes the full attention matrix;
   the ViT sees thousands of native-resolution patches for one 1024² image,
   so this is quadratic work that CPU `sdpa` normally avoids.
3. **bf16 is the right dtype on this box (no lever here).** `lscpu` shows
   `avx512_bf16` + `avx512_vnni` (no AMX): bf16 matmuls execute natively via
   oneDNN, they are NOT emulated. fp16 CPU matmuls are the poorly-supported
   path, so `ENCODER_DTYPE=bf16` is already the fast choice. Fewer threads is
   where the loss is, not the dtype.
4. **No vision work is duplicated for this run's shape.** The negative
   branch is encoded ONLY when `cfg_active` (`encode_with_supplied_pipeline`),
   and the edit run had CFG off → exactly one `encode_prompt` call, one
   vision-tower pass. With CFG ON the pinned upstream contract passes the
   images to BOTH branches (each call re-runs processor + vision tower);
   deduplicating that would mean reimplementing upstream `encode_prompt`
   internals, which the design rejected (§1.6). CFG-on edit requests pay
   the vision cost twice; recorded, not changed.
5. **Environment inheritance is fine.** The child is spawned with plain
   `Popen` (no `env=`), so the parent's environment reaches it; the guard's
   `os.environ.update` overwrote the four thread pins (that was the blocker,
   now a knob).

## Diagnostic knobs added (this change)

`encoder_worker` now reads two environment knobs, validated in the child
BEFORE torch imports (OpenMP pinning is frozen at first BLAS init):

- `QWEN_ENCODER_THREADS` (default `1`, accepted 1..64) → applied to the four
  thread env vars in `install_offline_guard()` and to `torch.set_num_threads`.
- `QWEN_ENCODER_ATTN` (default `eager`, choices `eager|sdpa`) → passed to
  `from_pretrained`.

**Defaults reproduce the historical child byte-for-byte** (pinned by the
existing loader tests + new knob tests). Both knobs are documented as
diagnostic: any thread count above 1 changes matmul reduction partitioning
and any attention kernel change can reorder accumulation, so non-defaults
MAY change embedding bits. Production keeps the defaults; root may raise
them ONLY if the variant digest below matches.

## Measurements

See `output/gpu/q2-profile/results.jsonl` (filled below once runs land).

## What is cheap to fix, what is not

- **Threads (cheap, measured here):** the child CAN be raised via
  `QWEN_ENCODER_THREADS`. Production may adopt a non-1 value only if the
  variant digest row matches the baseline: thread count changes oneDNN
  reduction partitioning, so bit-equality must be demonstrated per pinned
  torch build. Interop threads stay 1 (sequential eager graph).
- **Eager -> sdpa (cheap to try, digest-unsafe until proven):** CPU sdpa
  avoids materializing the full ViT attention matrix; different kernel
  path, so equality must come from the `t16-sdpa` row.
- **bf16 (no action):** already optimal on this CPU (`avx512_bf16`
  native, no AMX); fp16 CPU matmuls are the slow path. Nothing to buy
  without changing numerics.
- **CFG-on double vision pass (not cheap):** by pinned-upstream design
  each `encode_prompt` branch re-runs processor + vision tower; dedup
  lives inside upstream and is out of scope (§1.6). CFG off (the edit
  default) already pays vision exactly once.
- **Read the `hooks` row first:** it decides whether vision tower or
  language prefill dominates; at 1 thread both are bandwidth-starved,
  so threading is expected to be the multi-fold lever, attention the
  second. Measure, don't assume.
