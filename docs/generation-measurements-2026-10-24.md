# V100 generation measurements — 2026-10-24

First real end-to-end text-to-image runs of this engine on the installed
Tesla V100 (host, not the project container), with the default loader
(`adapters/qwen21_diffusers.py`) and the pinned local asset set.

Execution environment: CPython 3.13.14, torch 2.14.1+cu126, diffusers
0.41.0.dev0, transformers 5.17.0 (git `856157a2`), driver 580.178.04,
CUDA 12.6, `Tesla V100-SXM2-16GB` (`GPU-4a20902a-…`), sm_70. The encoder
runs in a separate CPU child; the resident profile is FP16, native SDPA,
transformer + VAE on the card, no text encoder in-process.

## Scope and honesty

These are **exploratory single-run measurements** on one workstation, not
calibration records. They do not satisfy the `benchmark_result.v1` /
`calibration_observation.v1` schemas (no device-wide peak collector, no
owned-process-tree host peak, no exact runtime descriptor match), so no
`calibration.json` entry was fabricated. Prompt-following was checked by eye
on two prompts; this is not a quality acceptance suite.

## End-to-end results

| Run | Resolution | Steps | Cold/warm | Wall | Denoise | Decode | Peak device VRAM |
|---|---|---|---|---|---|---|---|
| A | 512² | 4 | cold | ~40 s | 1.0 s | 1.5 s (tiled) | 14,806 MiB |
| B | 1024² | 40 | cold | 1m43s | 41.7 s | 1.1 s (tiled) | 15,435 MiB |
| C | 1024² | 40 | warm encode | 1m27s | 42.1 s | 0.85 s (untiled) | 15,439 MiB |
| D | 1024² | 40 | warm encode | 1m42s | 41.8 s | 0.85 s (untiled) | 15,439 MiB |

Phase breakdown for run C: load 26.6 s (two 31 GB manifest re-hashes plus
component load), encode 0.10 s (embedding-cache hit; cold encode was 15.7 s
with the 17.5 GB encoder already in page cache), denoise 42.1 s
(≈1.05 s/step at 1024²), decode 0.85 s, save 0.33 s.

Host `MemAvailable` stayed at or above 61.7 GB during run B; the encoder
child's 17.5 GB load is the largest host-memory event. GPU was returned to
0 MiB after every run.

## Decode: tiling costs visible quality

Resident (untiled) VAE decode of the **same latents** as the tiled decode
differs by up to 52/255 per channel (mean 1.24), and the difference is not
noise: tiled decode leaves faint vertical dashed streaks in flat regions
(clearly visible on white). Untiled decode is clean.

Consequences:

* The loader drops the transformer before decode. `pipeline.transformer = None`
  alone frees nothing — diffusers attaches a per-module hook registry that
  refers back to the module, so the transformer survives as an unreachable
  cycle (measured: 13.9 GiB still allocated). `gc.collect()` after dropping
  the reference breaks it (13.89 GiB → 0.64 GiB) and the freed blocks are
  reused by the decode allocation; no global allocator flush is performed.
* Untiled 1024² decode then fits (≈8.7 GiB peak with the VAE alone) and is the
  default path. `--tiled-decode` remains an explicit memory-first knob, now
  documented as artifact-prone.
* VAE precision is not the cause and not worth changing: fp16 vs fp32 decode
  of identical latents differs by at most 1/255 (mean 0.04).

## Headroom honesty

At 1024² untiled, device-wide peak is ~15.44 of 16.0 GiB. The allocator
logged a transient failed 312 MiB request in runs C and D and recovered by
retrying, so the configuration is **tight but working**, not comfortable.
A 2K run would exceed this envelope with the resident profile; it needs the
M5 memory work (streamed/offloaded components or a smaller decode footprint)
before it can be attempted.

## Text rendering and guidance

* Prompt-following and text rendering check (1024², 40 steps, seed 7,
  `true_cfg_scale` 1.0): "a hand-painted wooden sign reading "OPEN" leaning
  against a driftwood post on a misty beach at dawn" produced a legible
  "OPEN" sign with plausible weathering. Text was not garbled, mirrored or
  misspelled.
* True CFG costs roughly a doubled transformer batch. Measured peaks:
  15,889 MiB at 512²/20 steps (negative prompt, scale 4.0) and a transient
  allocator retry at 1024²/20 steps, which still completed.
* **Higher guidance looked worse at equal steps.** The same prompt and seed
  at 512²/20: scale 4.0 with a negative prompt gave saturated, plastic-looking
  red with blown highlights; scale 1.0 gave natural shading and a plausible
  material. The 1024²/40-step scale-1.0 image was the best of the set. So the
  default (`1.0`, guidance inactive without a negative prompt) is the
  recommended quality path on this profile; raising the scale is a user choice
  for a specific negative constraint, not a quality upgrade.


## Reproduction

```
.venv/bin/python -m qwen_engine --run \
  --manifest models/manifest.json --models-dir models/assets \
  --prompt "a red cube on a white table, studio photo, sharp focus, soft daylight" \
  --width 1024 --height 1024 --steps 40 --seed 42 \
  --output-dir <existing dir> --cache-dir <cache dir> \
  --log-dir <log dir> --attest-unbounded-host
```

The output directory must already exist: the save path refuses to create it.
