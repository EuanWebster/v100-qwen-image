# Saved-image quality comparison (Phase 8)

Reproducible procedure and offline tool for comparing the images a tiled-decode
change produces. It complements `docs/decode-benchmark.md` (memory/time per tile
geometry) with the *image* half: numeric deltas plus a manual review protocol.

Nothing in this document is a measured result. It defines **how** to produce and
compare the evidence; every number in a completed report must come from a real
run recorded beside its images.

Two pieces:

* `scripts/compare_benchmark_images.py` — CPU-only CLI. Reads already-saved
  images, refuses anything non-comparable, writes one strict-JSON report.
  No generation, no downloads, no torch/diffusers import; Pillow is imported
  lazily and only when actually decoding an image.
* `docs/image-quality-benchmark.md` — this procedure (fixed settings, review
  checklist, evidence retention).

Existing evidence already in the repository: repeated FP16 runs of the same
prompt/seed/size produced **pixel-identical PNGs**, including the historical
pre-change 1024²/40/seed-42 anchor (`docs/quality-suite-2026-10-24.md`,
"Determinism regression"). That establishes run-to-run determinism for those
cases — it does **not** establish that a candidate tile geometry preserves
quality.

## The tool

```bash
python scripts/compare_benchmark_images.py \
  --reference out/phase8/1024-base/object.png \
  --candidate out/phase8/1024-cand768/object.png \
  --candidate out/phase8/1024-base-rerun/object.png \
  --expect-size 1024x1024 \
  --output out/phase8/compare-object-1024.json
```

* `--reference` — one image, normally the production-geometry output.
* `--candidate` — repeatable; each is compared against the reference.
* `--output` — a **new** path. Published with the reviewed exclusive-write
  helper (`O_EXCL` + fsync), so an existing report is never replaced and a
  failed write removes its own partial file; `--output` may not alias any input
  image. The write is exclusive, not atomic: a reader that opens the report
  while it is being written can see a partial file.
* `--expect-size WxH` — optional hard dimension gate (e.g. `1024x1024`).

Run it with the project interpreter and no GPU:
`PYTHONPATH=src .venv/bin/python scripts/compare_benchmark_images.py ...`

### What it refuses (no silent reinterpretation)

| Input | Behaviour |
|---|---|
| non-regular/oversized file (> 64 MiB raw pixel-buffer cap) | refused before allocation |
| > 4096 px on either side, or a decompression bomb | refused (Pillow bomb check active) |
| truncated or corrupt file, non-image bytes | refused as unreadable |
| palette / grayscale / CMYK / LA | refused, never converted |
| candidate size ≠ reference size (or ≠ `--expect-size`) | refused; no scaling or cropping |
| candidate mode ≠ reference mode (e.g. RGB PNG vs RGBA PNG) | refused: the modes differ, so comparing them would need an implicit channel conversion (adding, dropping or flattening alpha) — irrelevant of container |
| same mode, different container (PNG vs JPEG) | allowed, and the entry carries an `encoding_note` saying the deltas **may** include codec differences and are not attributable to geometry alone |

`alpha_stats`/`compare_rgb_bytes` come from `src/qwen_engine/decode_bench.py`;
alpha is measured only for RGBA, so an RGB pair gets no alpha block rather
than a fabricated 255.

### Report fields (per candidate, index-labelled)

`label` (`candidate-1`, `candidate-2`, …), `width`/`height`/`mode`/`format`,
`encoded_sha256` (container bytes, so PNG text metadata changes it),
`pixel_sha256` (decoded pixels only), `exact_pixels_equal`,
`max_channel_delta`, `mean_channel_delta` (mean absolute delta over every
channel sample, 0..255), `per_channel_max_delta`, `changed_pixel_fraction`,
`alpha` (RGBA only), `encoding_note`. The `reference` block carries the same
identity facts. There are no prompts, seeds, timestamps or absolute paths in
the report: bind a report to a run by its recorded hashes, not by filename.

`exact_pixels_equal` follows from the pixel digest; it is a byte fact, not a
quality verdict. Equal pixels across two decode configurations mean the VAE
output did not change — still review them, but numeric sampling noise is not a
possible explanation there.

## Quality procedure (mandatory shape of the run)

Fixed settings for every image in one evaluation:

| Setting | Value |
|---|---|
| quant | FP16 (base model, not Q8) for the 1K reference and its candidates; Q8 is a separate evaluation, and quantization differences are not decode-geometry deltas |
| steps | 40 |
| seed | 42 |
| `true_cfg_scale` | 1.0 (no negative prompt) |
| output format | PNG (lossless). JPEG is lossy; WebP may be lossy or lossless depending on the encoder, though this engine writes lossless WebP (`image_output.py`, `lossless=True`). Either way, prefer PNG so no codec difference enters the deltas |
| decode | **1K reference: untiled.** At 2K the working reference is the **production tiled geometry, FP32, tile 1024 / stride 768** — not untiled, and not a forced untiled 2K run. Whatever the candidate is, the reference must be the configuration actually in production at that resolution |

1. **Fix the prompt set.** Use one prompt per genre, e.g. the six fixed
   prompts of `fixed-quality-v1` (object, text, composition, counting, style,
   lighting) from `docs/quality-suite-2026-10-24.md`, all at seed 42. Changing a
   prompt or seed creates a different comparison set; record the exact text.
2. **Sizes.** Sweep 1024², 1536² and 2048² for each prompt, plus the nominal
   preset aspect ratios from `src/qwen_engine/presets.py` (`--preset`, 1K and
   2K tiers) to catch geometry-dependent seams on non-square images. 1536 is
   the intermediate case: above the untiled 1K regime and below the 2K memory
   ceiling, so its reference must be a tiled production geometry too.
   Whether FP16 fits the 1536²/2048² shapes is **not assumed** — if a shape
   does not fit FP16 on this card, record it as unsupported/unmeasured rather
   than substituting untiled or a different quant.
3. **Geometries.** Reference = the production geometry at that resolution:
   untiled for 1024², FP32 tiled **tile 1024 / stride 768** for the high-res
   (1536²/2048²) working reference. Candidate = tile 768 / stride 576. Do not
   compare against a geometry that was not the actual saved reference, and do
   not force an untiled 2K reference.
4. **Compare.** One report per image size with `--expect-size`, candidates as
   index labels. Save the command used alongside the report.
5. **Review at 1:1, manually.** For every candidate, at 100 % zoom, inspect the
   full image and walk **both** boundary sets as bands, not single lines:
   * the **candidate's own seams** at multiples of its **576 px stride**;
   * the **legacy 768 px bands**, where the production 1024/768 reference
     switches tiles — the candidate's output differs there too, and an
     improvement or regression can sit exactly on the reference's boundary.
   Each boundary is an overlap **region** (tile minus stride wide: 192 px for
   the candidate, 256 px for the legacy geometry), not a one-pixel line, so
   inspect the whole band across the full width/height. Look for visible
   seams/banding, changed high-frequency detail, text legibility, small-count
   correctness, and alpha edges if RGBA. A numeric report never substitutes for
   this step.

### Decision rule

* Start with **one genre only** — the genre where the candidate first looked
  promising — at one size. Widen to the full prompt/size/preset sweep only if
  that single genre passes manual review.
* Candidate 768/576 is **not an accepted default**. It becomes the default only
  after: every size/preset shows no visible seam at 1:1 review, deltas are
  explainable by expected numerical noise (not structural), alpha edges are
  unchanged, and root has recorded the memory/time trade-off from
  `scripts/benchmark_vae_decode.py`. Until then it is an experimental profile.
* A small `changed_pixel_fraction` with a localized large `max_channel_delta`
  deserves the closest look: that pattern is what a seam produces.

### Evidence retention

Images are the evidence; reports are an index to them.

* Keep the representative PNGs and their JSON reports in the run's
  git-ignored output directory (`output/` is ignored; nothing under it is
  committed). Record the run command, the exact prompt texts, sizes, presets,
  geometry and the image SHA-256s in the run's own notes.
* Representative = one image per genre per geometry, including the worst
  reviewed seam candidate and any RGBA/alpha case. Do not delete them while a
  geometry decision is open.
* Reports are small and may be attached to a review note, but they contain no
  paths; the mapping report → image lives in the run notes.

## Honest limitations

* Numeric deltas do not prove perceptual equivalence or absence of seams.
* The alpha channel is only checked when both images are RGBA; an opaque RGB
  comparison says nothing about alpha edges in the engine's real output.
* 1:1 review is a human step with no automation here, and this repository
  records no result for it yet.
* `mean_channel_delta` normalises over all channel samples, so a tiny localized
  change reads as a near-zero mean. Use `changed_pixel_fraction` and
  `max_channel_delta` together.
* The tool compares saved files only: it cannot detect a problem that the
  saving pipeline (PNG encode, colour handling) already masked.
