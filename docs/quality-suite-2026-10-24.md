# Fixed quality suite — 2026-10-24

First fixed, repeatable image-quality run of the engine on the V100. It
complements the exploratory performance numbers in
`docs/generation-measurements-2026-10-24.md`; it is **not** a calibration
record and not a human acceptance decision.

Runner: `tools/quality_suite.py` (host-side orchestration only; no ML import).
Suite identity: `fixed-quality-v1` — six fixed prompts, one fixed seed each,
1024², 40 steps, `true_cfg_scale` 1.0 (no negative prompt), FP16 resident
transformer, untiled decode after the transformer is released.

## Fixed cases

| Name | Category | Seed | Prompt |
|---|---|---|---|
| object | object | 42 | a red cube on a white table, studio photo, sharp focus, soft daylight |
| text-open | text | 7 | a hand-painted wooden sign reading "OPEN" leaning against a driftwood post on a misty beach at dawn, photorealistic |
| composition | composition | 11 | a lone lighthouse on a rocky cliff at the edge of a calm sea, wide establishing shot, rule of thirds, overcast sky |
| counting-four-mugs | counting | 23 | exactly four ceramic mugs in a row on a wooden shelf, product photo, soft window light |
| style-ukiyoe-fox | style | 31 | a fox in a snowy forest, Japanese ukiyo-e woodblock print, flat colors, visible paper grain |
| lighting-glass | lighting | 47 | a glass of water on a windowsill, strong low afternoon sunlight, long shadows, dust motes in the air, macro photo |

Changing a prompt or seed changes the suite identity; report the prompt text
with any result.

## Results

Environment: same build as the measurements doc (CPython 3.13.14, torch
2.14.1+cu126, diffusers 0.41.0.dev0, driver 580.178.04, sm_70). Load is per
process (the CLI is single-shot); encode was cold for every prompt except
`object` (cache hit), which is why the wall times are uniform.

| Name | Wall | load / encode / denoise / decode / save | Peak device VRAM | Min host MemAvailable | SHA-256 (prefix) |
|---|---|---|---|---|---|
| object | 107 s | 27.8 / 16.9 / 42.8 / 1.26 / 0.33 s | 15,373 MiB | 53.5 GiB | `8d274d4fc7f60ebf…` |
| text-open | 108 s | 28.2 / 16.7 / 42.9 / 1.28 / 0.30 s | 15,379 MiB | 53.6 GiB | `7b5f2ebe6de1005b…` |
| composition | 109 s | 28.5 / 17.8 / 42.9 / 1.28 / 0.29 s | 15,379 MiB | 53.4 GiB | `dfe68d5dcc4f29b0…` |
| counting-four-mugs | 109 s | 28.5 / 16.9 / 42.9 / 1.29 / 0.34 s | 15,375 MiB | 52.9 GiB | `ffb983043a8d2ff2…` |
| style-ukiyoe-fox | 109 s | 28.4 / 16.7 / 42.9 / 1.28 / 0.31 s | 15,377 MiB | 52.7 GiB | `6c4de1ba1a9e36b5…` |
| lighting-glass | 108 s | 27.9 / 17.4 / 43.0 / 1.28 / 0.36 s | 15,379 MiB | 52.8 GiB | `8af3d0ed9c0caa4d…` |

No run failed; every case exited 0 and returned one 1024×1024 PNG. Full
hashes and the exact commands are in the run's `report.json`. All six peaks
are within 6 MiB of each other, so resolution/steps — not prompt content —
set the memory envelope.

## Review

Status: **agent visual review, human sign-off pending.** The contact sheet is
`contact-sheet.png` in the run directory. Nothing here is an automated
quality score (no CLIP/FID/OCR metric was used); this is a checklist review of
the failures the suite was designed to expose.

| Case | Result | Observation |
|---|---|---|
| object | **pass** | Sharp cube edges, visible paint texture and a plausible contact shadow/reflection. It is a close-up rather than a table-wide studio shot; the prompt did not specify framing. |
| text-open | **pass** | "OPEN" is legible, correctly spelled and plausibly weathered on a wooden sign. The sign is mounted on the post rather than leaning against it — a composition miss, not a text failure. |
| composition | **pass** | Lighthouse sits on the left third with the horizon near the upper third and an overcast tonal range. The framing reads as a medium-wide shot more than a "wide establishing" one. |
| counting-four-mugs | **pass** | Exactly four distinct mugs, in a row, with no duplicates or extra handles. This is the check most likely to fail on weaker models and it did not. |
| style-ukiyoe-fox | **pass** | Flat ukiyo-e palette, visible paper grain, calligraphic block and red seal; the fox remains anatomically readable. The strongest result in the set. |
| lighting-glass | **partial** | Strong directional backlight, caustic on the sill and window grime are convincing, but the requested long shadows are not visible from this angle and the glass base/water line is slightly thick. Light direction itself is unambiguous. |

Summary: five passes and one partial, no failure of the core prompt. The suite
is deliberately small; one seed per prompt means seed variance is not
measured.

## Determinism regression

The `object` result is **byte-identical** to the pre-change 1024²/40/seed-42
run made before phase-exclusive VAE residency and MLP chunking landed:

```
sha256  8d274d4fc7f60ebf475a234ad65d4a22df7690f200655cd218655a6b5f6898c6
```

Those changes therefore did not perturb the 1024² sampling path (chunking
starts above 4096 tokens; 1024² is exactly 4096).

## Honest limits

* The peak is sampled from `nvidia-smi` every 250 ms **device-wide**; a
  shorter spike can be missed. It is evidence, not a calibration record.
* Each case ran in a fresh process, so load and cold encode are included;
  the denoise timings are the comparable part.
* Timings are single-run on one workstation, not `benchmark_result.v1`
  medians, and the suite does not write `calibration.json`.
* Review is the agent's visual inspection of the contact sheet and full-size
  PNGs. Human acceptance is still pending.
* Artifacts (PNGs, logs, `report.json`, contact sheet) live in
  `/tmp/qwen-quality-2026-10-24`; only the fixed runner and this record are
  committed.

## Re-run

```
.venv/bin/python tools/quality_suite.py \
  --output-dir /tmp/qwen-quality-2026-10-24 \
  --steps 40 --attest-unbounded-host
```

For the experimental 2K path add `--width 2048 --height 2048 --tiled-decode`
(about 22 minutes for six cases; 2K needs at least 20 steps, see the
measurements doc).
