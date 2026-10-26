# Encoder CPU performance — where the 186.6 s reference-image encode goes

Investigated 2026-10-25 from the real V100 run log `output/gpu/batch2/edit-1ref.log`
(`text_encoder_request: 186.601s`, prompt + ONE 1024² reference, bf16, cache
miss) versus ~19.5 s text-only. Driver: `output/gpu/encoder_profile_q2.py`
(variants re-run the REAL encoder child through `ProductionEncoder` with
private caches; every variant's payload gets a canonical sha256 so speed wins
can be checked for BIT-IDENTICAL output). Results: `output/gpu/q2-profile/results.jsonl`.

## Facts established by code reading (verified)

1. **The child was deliberately single-threaded (the flip below lifts
   this).** `encoder_worker.install_offline_guard()` pinned
   `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` and `_load_classes()` called
   `torch.set_num_threads(1)` + `set_num_interop_threads(1)` before any ML
   import. This dominated everything else: the whole ~7-8 B Qwen3-VL
   forward (vision tower + language model) ran on ONE core. Text-only
   paid it too (19.5 s), and the image path multiplied it.
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

## Diagnostic knobs added

`encoder_worker` reads two environment knobs, validated in the child
BEFORE torch imports (OpenMP pinning is frozen at first BLAS init):

- `QWEN_ENCODER_THREADS` (accepted 1..64; **default is now
  `min(16, os.cpu_count())`** after the flip below — historically `1`) →
  applied to the four thread env vars in `install_offline_guard()` and to
  `torch.set_num_threads`.
- `QWEN_ENCODER_ATTN` (default `eager`, choices `eager|sdpa`) → passed to
  `from_pretrained`.

`QWEN_ENCODER_THREADS=1 QWEN_ENCODER_ATTN=eager` reproduces the
historical child exactly (pinned by loader/knob tests).

## Values ARE deterministic; the payload digest was a cache-key red herring

Run-level `blob_sha256` differed across ALL runs (`digests.jsonl`), but
that hash covers the safetensors response, whose header embeds
`request_identity -> worker-source.v1` — the hash of EVERY .py file under
`src/qwen_engine/` (`encoder_identity.worker_source_digest`). Concurrent
coders edit those files continuously, so the identity (and cache key)
churns between runs while tensor values stay put. Hashing the tensor
bytes directly (table: `output/gpu/q2-profile/value-hashes.txt`; raw
dumps `e-t1a.bin`/`e-t1b.bin`, `cmp`-identical) partitions cleanly:

- 9/9 eager runs at t1 AND t16 -> one identical `positive.embeds` sha
  (`5cf882a930…`); `image_pad_mask` identical everywhere.
- 3/3 sdpa runs -> one identical sha (`68aae15790…`), different from eager.

So a fresh encode IS bit-reproducible for a fixed (threads, attn) config,
**`QWEN_ENCODER_THREADS=16` is bit-identical to threads=1 on this box**
(bf16 AVX512 oneDNN matmuls partition-invariant here), and **sdpa is a
material output change, not last-bit noise**: global cosine 0.936,
per-token cosine mean 0.944 / min 0.21, mean abs delta 1.77 on values of
mean magnitude 7.1. Per the standing rule sdpa is therefore REPORTED,
NOT APPLIED. (Two operational notes fall out: sdpa also silently changes
the cache-key identity path, and the source-wide digest means ANY
`src/qwen_engine/*.py` edit — including web modules — invalidates every
embedding cache entry process-wide.)

## Measurements

Same request as the real run (same reference image bytes, exact edit-1ref
prompt, bf16, CFG off, private cache = genuine miss). Child-process wall
(load + encode, `results.jsonl`):

| variant | wall s | speedup vs pinned |
|---------|--------|-------------------|
| t1-eager (reproduction of production) | 185.485 | 1.00x (real run: 186.601) |
| t16-eager | 65.979 | 2.81x |
| t16-sdpa | 44.071 | 4.21x |

In-process breakdown at t16-eager (`hooks` row, 16 GB model load included
in a separate number: load 11.7 s; encode 47.9 s):

| stage | seconds | share of encode |
|-------|---------|-----------------|
| reference prep (decode/resize) | 1.80 | 3.8% |
| vision tower forwards (1 call) | 24.36 | 50.9% |
| language model forwards (1 call) | 22.40 | 46.8% |
| other in encode | 1.14 | 2.4% |

So the image path is NOT one dominant monster: vision tower and language
prefill are nearly 50/50, both CPU-bandwidth-bound at one thread; a single
image adds ~3x the text-only encode work, split evenly. Attention impl was
confirmed `eager` on both outer and text config; `torch.get_num_threads`
confirmed the thread knob lands (16).

See the red-herring section above for what the digests actually show
(value-level identity proven by direct tensor hashing, not payload sha).

## What is cheap to fix, what is not

- **Threads (adopt: bit-identical 2.8x):** `QWEN_ENCODER_THREADS=16`
  gave 185.5 s -> 66-73 s with `positive.embeds` byte-identical to t1
  (section above). Interop threads stay 1 (sequential eager graph).
- **Eager -> sdpa (REJECTED — do not adopt):** 44-53 s (4.2x) but the
  output change is material (global cosine 0.936, min per-token 0.21),
  far beyond kernel noise; it silently changes generated images.
  Reported only.
- **bf16 (no action):** already optimal on this CPU (`avx512_bf16`
  native, no AMX); fp16 CPU matmuls are the slow path. Nothing to buy
  without changing numerics.
- **CFG-on double vision pass (not cheap):** by pinned-upstream design
  each `encode_prompt` branch re-runs processor + vision tower; dedup
  lives inside upstream and is out of scope (§1.6). CFG off (the edit
  default) already pays vision exactly once.
- **Where the time goes (measured, t16-eager hooks):** vision tower
  24.4 s (51%) vs language prefill 22.4 s (47%) vs reference prep 1.8 s.
  Both big halves are the same 7-8 B-scale bf16 matmul stream starved at
  one core; that is why threading is the multi-fold lever and attention
  kernel choice the second one.
- **ADOPTED (worker default flipped, this change):** threads default is
  now `min(16, os.cpu_count())`; `QWEN_ENCODER_THREADS` override and
  1..64 validation kept; attention default unchanged (eager). Evidence
  below; tests pin the new default.

## Confirmation matrix before the default flip (all fresh children)

Per-tensor value hashes (payload parsing immune to cache-key churn):

| scenario | t1 | t16 | values equal | wall t1 -> t16 (s) |
|----------|----|-----|--------------|--------------------|
| text-only, CFG on (negative prompt, scale 4.0; pos+neg branches) | yes | yes | 4/4 tensors identical | see phase C (n=2) |
| edit 1-ref | yes | yes | 2/2 identical | 185.5 -> 66-73 |
| edit 2-ref | yes | yes | 2/2 identical | 378.2 -> 107.1 (3.5x) |

## Phase C: text-only timing, n=2, load/encode split (child stage report)

The single-run 20.9 -> 35.7 s "text+CFG slower at t16" observation did NOT
reproduce. Fresh children, warm page cache; `prompt_encoding` = child-side
encode seconds (`PerformanceRecorder` stage report, the same numbers the
app records), wall = parent-observed load+encode:

| scenario | threads | encode s (n=2) | wall s (n=2) | load s |
|----------|---------|------------------|--------------|--------|
| text (19 tok) | 1 | 3.82 / 3.72 | 16.6 / 16.4 | 2.5 |
| text (19 tok) | 16 | 2.98 / 4.90 | 15.7 / 17.8 | 2.5 |
| text+CFG (both branches) | 1 | 6.36 / 6.37 | 19.0 / 19.1 | 2.4 |
| text+CFG (both branches) | 16 | 6.83 / 6.36 | 19.5 / 19.3 | 2.5 |

Text-only encodes are too small (19-38 tokens) for threading to matter:
encode deltas sit inside run-to-run spread (t16 hit both a fastest and a
slowest single sample); wall differs by <= ~0.4 s (~2%). Reference-image
encodes keep their 2.8-3.5x. All eight cells reproduced the same value
hashes per scenario across thread counts (yet again bit-identical).
**Decision: keep the single default `min(16, cpu_count)`** — a
images-only thread rule would reorder guard-before-request reading for a
<= 2% effect that is not even consistently a loss.

## Child peak RSS vs the memory gate

Child peak RSS (`VmHWM`, production spawn path, 2-ref): **18396 MiB at
t1 vs 18361 MiB at t16 — thread count does not move the peak** (model
weights dominate; per-thread oneDNN arenas are noise here). Text+CFG
pair: 15295 vs 15300 MiB; post-flip 1-ref default run: 17445 MiB.

Memory gate for this checkpoint: bf16->bf16
expansion 16.33 GiB + explicit headroom 4 GiB (ENCODER_HEADROOM_BYTES)
= required 20.33 GiB vs `MemAvailable`; measured peaks sit below the
required bound at both thread counts, so the preflight budget sees no
change. Phase C adds eight more text+CFG peaks, all 15292-15302 MiB.
