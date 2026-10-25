# VAE tile-geometry decode benchmark

Fixes the investigation behind the near-OOM 2K decode: the production FP32
tiled VAE path (1024 px tiles / 768 px stride) peaks at a measured **15.57
GiB with allocator retry** on the 16 GiB V100 (Q8, 2048², seed 42; denoise
alone is 10.4 GiB). This tool compares the reviewed geometry **1024/768**
against **768/576** and **512/384** over the *identical saved real latent
tensor*, so memory/time/seam differences are attributable to geometry alone.

Two pieces:

* `src/qwen_engine/decode_bench.py` — ML-free core: geometry/label
  validation, memory-observation records, image byte-diff math, snapshot
  metadata binding and the strict-JSON benchmark report. It deliberately
  does **not** reuse `CalibrationObservation` (that schema has no tile
  geometry field); the benchmark has its own `qwen.decode-bench` schema.
* `scripts/benchmark_vae_decode.py` — standalone CLI root runs on the V100.

## Usage (real GPU; root executes)

```bash
# 1. Plan only (no GPU touched; default without --run):
python scripts/benchmark_vae_decode.py --output-dir out/bench \
  --manifest-path <manifest> --models-dir <models> --cache-dir <cache> \
  --seed 42 --steps 4 --width 2048 --height 2048 --quant q8 \
  --q8-manifest-path <sidecar> --attest-unbounded-host

# 2. Capture real latents once + profile all geometries:
python scripts/benchmark_vae_decode.py ... --run

# 3. Replay the saved snapshot later without re-denoising:
python scripts/benchmark_vae_decode.py ... --run --latents out/bench/latents-seed42.pt
```

`--run` is mandatory for anything that touches CUDA; without it the script
prints the resolved plan as JSON and exits 0.

## How a profile run works

1. **Capture** (once): the normal `generate()` core lifecycle runs with a
   composition wrapper around the reviewed `Qwen21DiffusersAdapter.decode`.
   The wrapper snapshots the real latent tensor to CPU (`torch.save`) with
   metadata binding seed/config/artifact-set/runtime digests/capture
   geometry/latent SHA-256, then performs the **normal decode** so the run
   keeps the current-quality reference PNG. The adapter then closes fully —
   the denoiser retains no VRAM during any profile.
2. **Replay**: per geometry, a fresh FP32 `AutoencoderKLQwenImage21` is
   loaded from the verified local bundle (`local_files_only=True`), tiled
   with that geometry, and fed the identical snapshot through the exact
   production normalization (`pipe._unpack_latents`, `latents_mean`/
   `latents_std`, same as `supplied_qwen21.decode`). The snapshot is loaded
   with `weights_only=True` and its metadata re-validated against the live
   run's digests; an arbitrary external pickle is refused.
3. **Measurement**: each profile resets allocator peak stats, runs a fresh
   NVML device-wide sampler thread (declared 50 ms interval — a coverage
   statement, not a safety budget), synchronizes explicitly, and reports
   sampled NVML peak vs allocated vs reserved as distinct numbers. The
   sampler thread and NVML session close even when the decode raises.
4. **Comparison**: original PNGs per geometry are saved unmodified; RGB
   byte deltas (max/mean per channel, changed-pixel fraction) and alpha
   stats against the 1024/768 reference go into `decode-bench.json`.

## Honest limitations

* NVML peaks are interval samples; sub-interval spikes can be missed.
* Allocator retries are the actual `num_alloc_retries` counter delta over
  each profile. Cache and peak counters are reset between profiles; CUDA
  context memory and other GPU processes remain visible to NVML.
* Numeric deltas alone do not prove seam-freeness; quality acceptance
  requires human review of the saved PNGs. No claim that a smaller geometry
  preserves quality is made without those images.
* SIGINT/SIGTERM cancel cooperatively; partial artifacts are kept, no
  report is written, owned threads/NVML sessions still close.

Root review fixed runtime API mismatches, metadata/raw tensor binding, RGBA
preservation and pixel-difference counting. The snapshot digest is separate
from the raw tensor digest; replay binds semantic inputs rather than output
directory names. Snapshots are bounded at16MiB, reports at4MiB, and raw pixel
buffers at the supported4096² RGBA size. Existing artifacts are not replaced.

For quality evaluation use40steps explicitly: the default4steps is only a
quick diagnostic and is not a base-model quality recipe. Example:

```bash
mkdir -p output/decode-profiles/example
.venv/bin/python scripts/benchmark_vae_decode.py --run \
  --manifest-path models/manifest.json --models-dir models/assets \
  --cache-dir cache --quant q8 --q8-manifest-path models/q8_manifest.json \
  --output-dir output/decode-profiles/example --steps 40 --seed 42 \
  --prompt "a detailed mountain lake landscape at sunrise" \
  --attest-unbounded-host
```

These are diagnostic benchmark profiles, not new production configuration
choices or memory-policy calibration evidence. Production geometry must only
change after inspecting the saved images and recording the memory trade-off.
Profiles run after capture in the same process; library caches may already be
warm even though the Torch allocator is cleared. Confirm a proposed new default
with a complete fresh-process generation before claiming production headroom.
