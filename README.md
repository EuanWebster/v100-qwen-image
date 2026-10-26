# v100-qwen-image

A hand-rolled [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
text-to-image engine built to run on a single **Tesla V100 16GB** (Volta,
sm_70). It provides a headless CLI and a local web workspace, sharing one
engine. Image quality comes first; speed is improved where it does not cost
quality.

The design follows an earlier LTX-2.5 ROCm build: a CPU-only engine core with
heavy ML imports deferred to adapters, a separate encoder subprocess, an
embedding cache, pinned model assets and measured VRAM calibration.

## Status

Working today on a real V100:

- **Text-to-image** at FP16 (default) or **Q8_0 GGUF** (experimental).
- **All seven upstream aspect ratios** at 1K and 2K (`--preset`), e.g. 1024²,
  2048², 2752×1536 landscape, 1536×2752 portrait.
- **Web workspace** with prompt-first layout, live progress, cancellation,
  session history, config import/export, three themes and in-UI help for every
  option.
- **Turbo recipe** (`--recipe turbo`, or Quality/Turbo in the web workspace):
  the official 8-step Turbo transformer with the base encoder and VAE. Sampling
  is ~5x faster; total run time barely moves because loading dominates.
- **Transparent backgrounds** (`--transparent`, or the web checkbox): a
  see-through background saved with a real alpha channel in PNG or WebP.
- **Reference-image editing** (up to 10 ordered images, web or
  `--reference`): **experimental** — it works end to end, but an edit whose
  output is exactly 1024×1024 comes out over-processed and over-sharpened.
  That is the only size that shows it: 1024×960, 1024×1088, 1088×1024, 960²,
  768² and 512² were measured clean, and 1152² and larger run out of GPU memory
  in FP16. Cause still under investigation; pick any size other than exactly
  1024×1024 until the fix lands.
- **Optional prompt rewriting** through a local OpenAI-compatible service,
  off by default and never applied without an explicit **Apply**.
- Output as JPG (default), PNG or lossless WebP, named
  `qi21-DDMMYYYY-<seed>` and never overwriting existing files.

Day-to-day usage, plain-language trade-offs and what is measured versus still
unknown: **[`docs/user-guide.md`](docs/user-guide.md)**.

Measured on the development machine (single runs, not a benchmark suite):

| Job | Denoise | Wall time |
|---|---|---|
| 1024², 40 steps, FP16 (Quality) | 42.0 s | ~107–109 s, measured before the hashing/encoder speed-ups |
| 1024², 8 steps, FP16 (Turbo) | 8.5 s | 68.9 s cold, 26.8 s with the prompt cached |
| 1024², 40 steps, FP16, 1 reference image | 95.4 s | 130.2 s |
| 2048², 20 steps, FP16, tiled decode | 149.5 s | ~3 m 39 s (15,835 MiB peak) |

Model loading and the CPU text encoder, not sampling, dominate a cold run —
which is why the denoise column and the wall-time column tell different
stories, and why Turbo barely shortens a cold job. Model files are now hashed
once per job and the encoder uses all cores, so cold figures measured before
that are pessimistic; [`docs/user-guide.md`](docs/user-guide.md) has the
per-stage numbers.

Not done yet: memory preflight before generation (submitting a job is **not**
a guarantee it fits), 2K VAE-decode headroom, and the cause of the edit
artifact at exactly 1024×1024 (measured, not explained — workaround: any other
output size, see [`docs/user-guide.md`](docs/user-guide.md)). See
[`docs/next-features-plan-2026-10-25.md`](docs/next-features-plan-2026-10-25.md)
and [`PLAN.md`](PLAN.md).

## Requirements

- Linux x86_64, Python 3.11 or 3.13, [uv](https://docs.astral.sh/uv/).
- NVIDIA Tesla V100 16GB, driver supporting CUDA 12.6.
- About 33 GB of disk for the base model (+ the Q8 GGUF if wanted), and enough
  host RAM for the ~17.5 GB text encoder, which runs on CPU in a child process.

torch is pinned to `2.14.1+cu126`, the last series shipping sm_70 kernels;
diffusers and transformers are pinned to exact git commits. Rationale and the
full pin table are in [`docs/dependencies.md`](docs/dependencies.md).

## Setup

```bash
# 1. Environment (hash-checked lockfile; the index flag is required)
uv venv --python 3.13 .venv
uv pip sync --python .venv/bin/python --index-strategy unsafe-best-match requirements.lock

# 2. Model weights: 24 files pinned by revision and SHA256 in models/manifest.json.
#    Generation never downloads; fetching is a separate, explicit step.
.venv/bin/python tools/fetch_models.py --manifest models/manifest.json \
    --models-dir models/assets --fetch
```

The optional Q8 transformer comes from `unsloth/Qwen-Image-2.1-GGUF`
(`q8/qwen-image-2.1-Q8_0.gguf`), pinned in `models/q8_manifest.json`. Place it
under `models/q8/`.

## Usage

### Web workspace

```bash
./start-web.sh
```

Serves on port 8765 with a single shared job queue. It binds `127.0.0.1` plus
a LAN and a Tailscale address; override these with `QWEN_LAN_HOST`,
`QWEN_TAILSCALE_HOST` and `QWEN_WEB_PORT`. Set `QWEN_ATTEST_UNBOUNDED_HOST=0`
when running under a cgroup memory limit. See
[`docs/lan-tailscale.md`](docs/lan-tailscale.md).

### CLI

```bash
# Show the resolved plan without loading anything
python cli_gen.py --dry-run --prompt "a lighthouse at dusk" --preset square

# Generate (an actual run needs the manifest, models, output and cache paths)
python cli_gen.py --run --prompt "a lighthouse at dusk" --preset landscape2k \
    --steps 40 --seed 42 --output-format png \
    --manifest models/manifest.json --models-dir models/assets \
    --cache-dir cache --output-dir output/runs
```

Useful options: `--quant {fp16,q8}`, `--negative` with `--true-cfg-scale`,
`--width/--height` (overrides the preset), `--[no-]tiled-decode` (2K presets
turn it on; the web API refuses an untiled 2K job, the CLI does not),
`--config FILE` (JSON `GenerationConfig`), `--transparent`, `--recipe
{base,turbo}`, `--reference PATH` (repeatable). Run `python cli_gen.py --help`
for the full list.

### Turbo recipe

`--recipe turbo` (or the web workspace's **Turbo** choice) runs the official
8-step Turbo image model instead of the reference model. It shares the
**base** text encoder, VAE and processor — only the transformer differs — so
the honest label is "Turbo transformer + base encoder/VAE".

Measured on a V100 at 1K/FP16: denoise 8.5 s vs 42.0 s for the reference
model, but a whole job 68.9 s vs 73.9 s (different cache states — see the
table above), because model loading and the text encoder dominate a run.
Turbo's saved sigmas make the step count inert, so the CLI and UI both fix
steps at 8 and **refuse** anything else rather than accepting a number that
would be ignored. Turbo is evaluated at CFG 1.0 with no negative prompt and
has no reviewed Q8 artifact: a true-CFG request (negative prompt *and* scale
above 1) and `--quant q8` are refused with a typed field error. The web
workspace locks steps, guidance, the negative prompt and quantization while
Turbo is selected, says why, and restores your own values when you switch back
to **Quality**; an imported config without a `recipe` key imports as `base`.
See [`docs/turbo-plan.md`](docs/turbo-plan.md).

### Transparent backgrounds

`--transparent` (or the web workspace checkbox) asks the model for a
see-through background and keeps the alpha channel in the saved file. JPEG
cannot store alpha, so the format must be `png` or `webp`: the CLI selects
`png` when no format was given explicitly and **refuses** an explicit `jpg`
rather than silently changing it; the UI switches `jpg` to `png` and says so.
Your prompt text is never modified — a fixed instruction is added only to the
text the model reads (`GenerationConfig.effective_prompt`). The result is
model-dependent, not a guaranteed cutout. See
[`docs/transparency.md`](docs/transparency.md).

## Tests

```bash
bash run_cpu_tests.sh            # full suite; any pytest args pass through
```

The suite runs fully offline with no GPU or weights. A guard denies network
access, blocks torch/diffusers/transformers imports and confines all writes to
a throwaway temp directory.

## Layout

```
src/qwen_engine/      engine core, CLI, encoder worker, job controller
src/qwen_engine/web/  HTTP server for the web workspace (static UI in web_static/)
src/qwen_engine/adapters/  the only place heavy ML libraries are imported
models/               pinned manifests (weights themselves are gitignored)
tools/                model fetcher and standalone verification checks
scripts/              environment setup, preset sync, browser checks, benchmarks
tests/                CPU-only test suite
docs/                 design contracts, measurements and review notes
```

## License

[MIT](LICENSE). This covers the code in this repository only; Qwen-Image-2.1
model weights are subject to their own upstream license.
