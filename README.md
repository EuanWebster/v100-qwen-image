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
- Output as JPG (default), PNG or lossless WebP, named
  `qi21-DDMMYYYY-<seed>` and never overwriting existing files.

Measured on the development machine (single runs, not a benchmark suite):

| Job | Wall time |
|---|---|
| 1024², 40 steps, FP16 | ~107–109 s |
| 2048², 20 steps, FP16, tiled decode | ~3 m 39 s |

Not done yet: memory preflight before generation (submitting a job is **not**
a guarantee it fits), 2K VAE-decode headroom, single- and multi-reference
editing, native RGBA transparency, the official 8-step Turbo model and
optional prompt rewriting. See
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

# Generate
python cli_gen.py --run --prompt "a lighthouse at dusk" --preset landscape2k \
    --steps 40 --seed 42 --output-format png
```

Useful options: `--quant {fp16,q8}`, `--negative` with `--true-cfg-scale`,
`--width/--height` (overrides the preset), `--[no-]tiled-decode` (2K requires
tiling), `--config FILE` (JSON `GenerationConfig`). Run `python cli_gen.py
--help` for the full list.

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
