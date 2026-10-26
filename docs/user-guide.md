# User guide — Qwen Image Studio on the V100

How to run the app day to day, what the newer options actually do, and what is
measured versus still unknown. Design contracts and raw measurements live in
the per-feature docs linked at the end; this page is the operator's view.

Everything below was verified on the development V100 (16 GB, sm_70) unless a
line says otherwise. Numbers are single runs, not a benchmark suite.

## Quick start

### Web app

```bash
./start-web.sh                      # http://127.0.0.1:8765
```

`start-web.sh` enables real generation and a single shared job queue, and binds
`127.0.0.1` plus a LAN and a Tailscale address (see
[`lan-tailscale.md`](lan-tailscale.md)). Useful overrides:

| Variable | Effect |
|---|---|
| `QWEN_WEB_PORT` | port (default 8765) |
| `QWEN_LAN_HOST`, `QWEN_TAILSCALE_HOST` | extra bind addresses |
| `QWEN_ATTEST_UNBOUNDED_HOST=0` | do not attest "no cgroup RAM cap" (set this in a memory-capped container) |

`qwen-web --reuse-verified-assets[=SECONDS]` (off by default, and not set by
`start-web.sh`) relaxes the asset check each job starts with: instead of
hashing every model artifact per job, the server may reuse a recorded
**successful** verification while every artifact's stat fingerprint is
unchanged and the record is younger than `SECONDS` (bare flag = 3600). That
is a deliberately weaker guarantee than the default — size-or-mtime tricks
are not what it guards — so enable it only for model trees you control.
Failed verifications are never reused, the server's startup log states when
reuse is ON and its max age, and bad ages (zero, negative, non-finite)
refuse startup.

The page is prompt-first: type a prompt, pick a preset, press **Generate**.
Progress, cancellation, history, config import/export, three themes
and in-UI help for every option are all in the page. Nothing is written to
your browser except the theme choice.

History survives server restarts. Each finished job writes one small
`job.json` inside its own `output/web/<job-id>/` folder recording the request,
the terminal state, the error code (fixed literals only; never exception text)
and the image's filename, size and sha256. At startup the server rebuilds
history from those files, loading at most the newest record-cap entries and
listing nothing that fails validation — a corrupt or unknown record is skipped
with one log line and a job that was still running when the server stopped is
never resurrected. Restored entries count against the same cap as live ones, so
a rebuilt history that already fills it must be trimmed with **Forget entry**
before new jobs are accepted. The image endpoint re-hashes the bytes it returns,
so an image that was changed, resized or deleted on disk is never served; its
history entry stays listed with the thumbnail marked unavailable, so the job's
settings remain readable and it can still be forgotten.
**Forget entry** deletes only that `job.json`: the image file stays where it is,
and the entry stays gone after a restart. Per-phase timings are measurements of
a running job, so a restored entry reports no elapsed time — everything else it
shows (settings, state, image) is what the stopped server recorded.

### CLI

```bash
# Plan only: resolve paths and print the deterministic plan JSON, no ML import
python cli_gen.py --dry-run --prompt "a lighthouse at dusk" --preset square

# Generate (an actual run needs manifest, models, output and cache paths)
python cli_gen.py --run \
    --prompt "a lighthouse at dusk" --preset landscape2k \
    --steps 40 --seed 42 --output-format png \
    --manifest models/manifest.json --models-dir models/assets \
    --cache-dir cache --output-dir output/runs
```

`--dry-run` needs no weights and is the fastest way to check that a set of
flags is accepted. `--run` requires `--manifest`, `--models-dir`,
`--cache-dir` and `--output-dir`; each missing one is a typed config error, not
a traceback. Images are named `qi21-DDMMYYYY-<seed>` and never overwrite an
existing file — a repeat of the same seed in the same directory appends `-2`,
`-3`, and so on.

Exit codes: `0` ok, `1` unexpected, `2` config, `3` memory refusal, `4` asset,
`5` hardware. Full flag list: `python cli_gen.py --help`.

The generation worker child (used by the web/job runner) adds exit `6` =
`out_of_gpu_memory`: a CUDA out-of-memory crash is reported with that code —
the same classification the web API surfaces — instead of a generic failure.

## Quality vs Turbo

**Quality** (the default, `--recipe base`) is the reference model at the step
count you choose — 40 is the normal pick. **Turbo** (`--recipe turbo`, or the
**Turbo** radio button) is the official Turbo transformer with the *same* text
encoder, VAE and processor as Quality. The honest description of a Turbo run
is "Turbo transformer + base encoder/VAE", not "the Turbo model".

Measured at 1024², FP16, seed 42:

| Run | Denoise | Whole job |
|---|---|---|
| Quality, 40 steps, prompt already encoded | 42.0 s | 73.9 s |
| Turbo, 8 steps, prompt not yet encoded (17.2 s of encoder) | 8.5 s | 68.9 s |

Drawing is roughly five times faster, but loading the weights and reading the
prompt dominate a run, so the total barely moves on a cold start. That Turbo
figure also predates the hash-once change below, which cut Turbo's model
initialisation from 41.3 s to 15.6 s; substituting those stage times into the
same run gives ~43 s rather than 68.9 s (inferred, not re-measured end to end).
A run whose prompt is already cached pays almost none of the encoder cost:
26.8 s for Turbo including a 15.6 s model initialisation. (The Quality run in
the same batch missed the cache and took 83.6 s with 27.4 s of encoder time,
so that pair is not a like-for-like recipe comparison — only the denoise
figures are.)

Turbo rules, all enforced rather than silently ignored:

- **Steps are fixed at 8.** Turbo's saved sigmas make the step count inert, so
  any other value is refused: `steps must be exactly 8, got 40`. The web UI
  locks the field and says why.
- **CFG 1.0, no negative prompt.** A negative prompt *and* a scale above 1 is
  refused. A negative prompt alone is accepted but has no effect at CFG 1.
- **FP16 only.** Turbo has no reviewed Q8 artifact; `--quant q8` is refused.
- **Needs its own weights.** Turbo installs beside the base assets
  (`models/turbo/manifest.json` + `models/turbo/assets`, 13.25 GiB → use
  `tools/fetch_models.py` as in [`turbo-plan.md`](turbo-plan.md)). A missing
  Turbo tree is a refusal, never a fallback to the base transformer.

The web UI locks steps, guidance, the negative prompt and quantization while
Turbo is selected, restores your own values when you switch back to Quality,
and imports a config without a `recipe` key as `base`.

## Transparent backgrounds

Tick **Transparent background** (or pass `--transparent`) and the engine adds a
fixed instruction to the text the model reads
(`GenerationConfig.effective_prompt`). Your prompt is never rewritten.

- **PNG or WebP only.** JPEG cannot store alpha: with `--transparent` and no
  explicit format the CLI picks PNG; an explicit `--output-format jpg` is
  refused; the UI switches jpg to png and says so.
- **Decoded in FP32.** The transparent path decodes the VAE in FP32 because
  the FP16 decode failed on all three test prompts — each died at
  `vae_decoding` with the engine's non-finite attestation, so no image was
  written at all. The same apple prompt in FP32 produced a valid RGBA file.
- **Not a guaranteed cutout.** The prompt instruction raises the chance of a
  see-through background; the result is model-dependent. Measured on the one
  saved 1024² run ("a glossy red apple with a green leaf, product photo, soft
  studio lighting", 40 steps, seed 42): all four corners are fully transparent
  and all 256 alpha levels appear, but only 0.6% of pixels are fully clear
  against 48% fully opaque and 51% partially transparent — a soft, gradient
  alpha rather than a hard-edged cutout. Two other prompts produced no viable
  image in FP16 (see above) and were not retried in FP32, so treat
  transparency as a useful head start, not a finished mask.

See [`transparency.md`](transparency.md) for the contract.

## Reference-image editing — **experimental**

**Status: works end to end; the over-processed 1024×1024 artifact is fixed.**
One reference produces a genuine edit (a red cube became a blue cube with the
table, lighting and camera preserved — `output/gpu/batch5`, `batch9`), and
1024×1024 seed-42 edits are now clean in both base FP16 and Turbo
(`output/gpu/batch21`). Editing stays **experimental**: quality is
model-dependent and the memory limits below are real.

**Cause (measured, fixed in `ab2f3ab`).** It was never the output size. Every
test reference had been generated by this engine at 1024×1024 with seed 42, so
an edit at 1024×1024 with seed 42 started from the **identical initial noise
tensor that produced the reference** — the reference content entered twice, as
conditioning and as starting noise, and the result looked over-processed.
Confirming runs, V100, same cube reference and prompt: seeds 7 and 123 at
1024×1024 were clean (`batch20`); every other output size changes the noise
shape, which is why they were clean too; the same latents decoded three
different ways all showed the artifact, exonerating the decode (`batch19`);
per-query chunked attention reproduced it byte for byte, exonerating SDPA
(`batch16`). Source: `output/gpu/FINDINGS-edit-grid.md`, ROOT CAUSE section.

**Fix.** An edit's sampler noise is now derived from the seed **and** the
ordered reference ids (`qwen-engine.edit-noise.v1`, `sha256`-based) instead of
being the seed itself, so an edit can never reuse the noise that generated one
of its own references. The derived value is recorded as `noise_seed` in the run
metadata; the visible seed and the filename stay the user's seed, and
text-to-image is unchanged. Because the derivation is deterministic and ordered,
**the same seed with the same references (same order) reproduces an edit
exactly**; changing a reference or its position changes the noise, so an edit is
only reproducible against the identical reference set.

**What is left of the old size table is only memory.** Every size tested came
out clean once the noise collision was gone; these rows are cost, not quality:

| Output size | Result | Configuration | Evidence |
|---|---|---|---|
| 1024×1024 | clean after the fix | base FP16, 40 steps | `batch21/base-1024` |
| 1024×1024 | clean after the fix | Turbo, 8 steps | `batch21/turbo-1024` |
| 1024×960, 1024×1088, 960×960, 768×768, 512×512 | clean | base FP16 / Turbo, pre-fix runs | `batch12`, `batch14`, `batch15` |
| 1088×1088 edit | untiled FP32 decode ran out of memory; fits with tiled decode | Q8 | root, then `batch17` (tiled, succeeded) |
| 1152×1152, 1280×1280, 1536×1024 | out of GPU memory | FP16, Turbo, KV off | `batch15` |

**Tiled decode for bigger edits.** An edit decodes its reference in FP32, which
costs more than a text-to-image decode of the same size. Measured: a 1088×1088
Q8 edit ran out of memory in an **untiled** FP32 decode and completed with
`--tiled-decode` (`batch17`). So on the CLI, edits above roughly 1088×1024 need
`--tiled-decode`. In the web workspace, presets above 1K switch **VAE tiling**
on automatically, and the server refuses any config above 4096 latent tokens
(1088×1024 is 4352) that leaves tiling off (`422 tiled_decode_required`) — so
custom sizes of that size need the box ticked. The CLI has no such refusal: an
untiled run is accepted and fails later, at decode.

A plain 3×3 Laplacian variance of the full frame tracked the defect while it
existed — artifacted 1024×1024 outputs scored 483–780 against 37–56 for the
clean ones — and the fixed 1024×1024 runs score 40.3 (base) and 33.6 (Turbo),
inside the clean band (`batch21`). It is a one-scene hint, not a quality metric:
judge edits by eye.

How to use it:

1. **Web:** tick **Edit with reference images** (below the prompt) to show the
   *Reference images* section, then press **Add reference** and pick up to 10
   PNG/JPEG/WebP files. They are numbered in the order shown (reorder with the
   per-card **Up**/**Down** buttons, remove with **Remove**) and you refer to
   them as "image 1", "image 2" … in the prompt. The prompt is sent after the
   images. Output size comes from the preset, not from the images; for an edit
   above 4096 latent tokens (e.g. 1088×1088) also tick **VAE tiling** — the
   server refuses an untiled config that large. Unticking the box keeps your
   references in the list but stops using them; tick it again to bring them back.
2. **CLI:** `--reference PATH` (repeatable, ordered, at most 10) plus
   `--references-dir DIR` (default `<cache_dir>/references`). The bytes are
   copied verbatim into a content-addressed store; the run carries only ids.

```bash
python cli_gen.py --run \
    --prompt "Change the red cube in image 1 to a blue cube. Keep the table, lighting and camera exactly the same." \
    --reference path/to/cube.png --preset square --steps 40 \
    --manifest models/manifest.json --models-dir models/assets \
    --cache-dir cache --output-dir output/edits
```

`--preset square` (1024×1024) is the measured-clean configuration after the
noise-seed fix (`batch21`). For a bigger edit add `--tiled-decode`: a 1088×1088
edit decodes in FP32 and does not fit untiled (`batch17`). Verified with
`--dry-run`, which resolves `--preset square` to `width: 1024, height: 1024,
vae_tiled_decode: false` and `--width 1088 --height 1088 --tiled-decode` to
`vae_tiled_decode: true`.

Limits and automatic behaviour (each measured, none silent):

| Behaviour | Detail |
|---|---|
| Noise seed | An edit's sampler noise is derived from `(seed, ordered reference ids)` — `qwen-engine.edit-noise.v1`. Same seed + same references in the same order reproduces the edit; changing either changes the noise. The visible seed/filename stay yours; the derived value is recorded as `noise_seed` in the run metadata. Text-to-image is unaffected. |
| KV cache | FP16 + references keeps the KV cache **off** automatically (KV on OOMs at the first denoise step). Q8 keeps it on. Recorded as `kv_cache_policy` in the run metadata. |
| Decode precision | References always decode in FP32, including opaque output (a Q8 2-reference run sampled fine and died at `vae_decoding` in FP16). |
| Decode tiling | Above ~4096 latent tokens an edit needs tiled decode (1088×1088 untiled OOMed in FP32; the same run finished with `--tiled-decode`, `batch17`). The web server refuses an untiled config that large (`422 tiled_decode_required`); the CLI refuses it too, before any model load, as a config error naming `--tiled-decode`. |
| File caps | 32 MiB per file, sides ≤ 4096 px, ≤ 4096² pixels, at most 10 references. Larger images are accepted, then resized to about one megapixel for the model. |
| Transparency in a reference | The alpha channel is kept in the stored bytes, but the vision encoder sees the image composited over white. |
| First encode cost | The text+vision encoder runs on CPU: an uncached 1024² reference costs ~66–73 s to encode; two references ~107–116 s (measured 116.0 s on one 2-reference job). Repeats hit the embedding cache. |
| Turbo + references | Allowed and **run**: `output/gpu/batch9/turbo-edit` (1024², 1 reference, denoise 19.2 s, whole job 33.0 s), `output/gpu/batch10/turbo-fisher-hat`, and post-fix `output/gpu/batch21/turbo-1024` (1024², denoise 19.1 s, whole job 80.5 s cold). Quality is still model-dependent — the adapter records `turbo_reference_run_unmeasured` — but the run is no longer intrinsically defective. |

Measured timings, 1024², 40 steps, seed 42 (denoise / whole job). The noise-seed
fix changes only which noise the sampler starts from, so these cost figures
still stand; post-fix runs measure the same order (base 96.2 s denoise / 109.2 s
job, Turbo 19.1 s / 80.5 s, `batch21`):

| Job | Denoise | Whole job |
|---|---|---|
| FP16, 1 reference | 95.4 s | 130.2 s |
| Q8, 1 reference | 58.2 s | 120.6 s |
| Turbo, 1 reference | 19.2 s | 33.0 s |
| FP16, 1 reference, long prompt (cache miss) | 96.1 s | 187.6 s |
| FP16, 2 references (KV off) | 160.4 s | 290.8 s |
| Q8, 2 references | 71.5 s | 139.7 s |

Peak reserved VRAM: 14.8 GiB for FP16 with one reference, 15.3 GiB with two,
and 11.9 GiB for Q8 with one (Q8 keeps the KV cache and still fits). FP16
edits use essentially the whole card; Q8 does not.

Design and memory matrix: [`image-edit-design.md`](image-edit-design.md).

## Prompt rewriting — off by default

When the server is started **without** a rewriter (`--rewrite-backend off`, the
default), the **Rewrite prompt** button is hidden and `POST /api/v1/rewrite`
answers 503. Nothing about rewriting affects generation when it is off.

To enable it, start the server against an OpenAI-compatible local service
(`local-llm` is the only backend) and hand it the two system-prompt files the
weights were trained on:

```bash
python -m qwen_engine.web --enable-generation --rewrite-backend local-llm \
    --rewrite-endpoint http://127.0.0.1:8079/v1/chat/completions \
    --rewrite-model <served-model-name> \
    --rewrite-prompt-t2i path/to/system_prompt_t2i.txt \
    --rewrite-prompt-edit path/to/system_prompt_edit.txt \
    --manifest models/manifest.json --models-dir models/assets \
    --cache-dir cache --output-dir output/web
```

The endpoint is the **full URL the client POSTs to**, not a bare base URL: it
must be `http` (no TLS policy exists for a loopback service), name an explicit
port, be an IP literal or `localhost`, and be a loopback address (your prompt
draft is user text and never leaves this machine). Anything else refuses
startup. The prompt files are digest-pinned (md5 `a5e1efc4…` for t2i,
`493264f3…` for edit); a mismatch likewise refuses startup with one line and
exit 2 rather than sending a prompt the weights were not trained on.
`--rewrite-timeout` defaults to 180 s.

In the UI, **Rewrite prompt** produces a proposal block headed *"Suggested
prompt — not applied"* with your draft, the suggestion, and **Apply** / **Edit**
/ **Keep original** (plus **Apply ratio** when the proposal names a preset that
matches exactly). It never changes your draft unless you press Apply. Failed
rewrites are typed: 503 disabled/busy/cancelled, 504 timeout, 502
backend-unavailable or an invalid answer, 422 prompt too long (> 8000
characters), unsupported task, or a reference count/set the task cannot use,
and 404 for a reference id the server no longer holds.

With reference images attached the rewrite is an **edit** rewrite, and it sends
the references with the draft: each stored image is read on the server,
alpha-flattened onto white, downscaled so no side exceeds 768 px, and sent as a
PNG data URI with your draft to the same local LLM — in the order the images
are listed, because the edit prompt refers to them as *image 1*, *image 2* and
so on. Only those downscaled copies leave the server (as inline data, never as
a URL or a path); the stored originals are unchanged. Without references the
rewrite is text-only, and the button is unavailable in edit mode until one is
attached. Changing, reordering or removing a reference makes any visible
suggestion stale, exactly as editing the draft does.

**Quality is only spot-checked** (7 requests, one prompt each, 19–191 s
latency, from before the sampling-parameter fix that made the profile's full
upstream sampling settings reach the backend). Treat rewrites as suggestions
to review, not as an oracle.

Details: [`prompt-rewriter-plan.md`](prompt-rewriter-plan.md) and
[`rewriter-quality-check.md`](rewriter-quality-check.md).

## Performance changes worth knowing

- **Model files are hashed once per job.** Each run used to hash the full model
  trees twice — once in the CLI asset check and again seconds later in the
  adapter (~19 s per pass for the 33 GB base tree, ~9 s for the 14 GB Turbo
  tree). The first pass now primes a 30 s in-process memo that the adapter
  reuses for the identical profile, manifest and root; refusals are never
  memoised and every job still hashes its files at least once. Measured after
  the change: model initialisation 11.1 s (base) and 15.6 s (Turbo) on a warm
  page cache, versus 29.6 s / 41.3 s before.
- **The CPU encoder uses `min(16, cpu_count)` threads** by default (it was 1).
  On this box that took a 1024² reference from 185.5 s to 66–73 s with
  bit-identical embeddings — proven by hashing the tensors, not the payload.
  `QWEN_ENCODER_THREADS` (1–64) overrides it; `QWEN_ENCODER_ATTN=eager|sdpa`
  exists, and `sdpa` is **not** recommended because it changes the generated
  image materially (global cosine 0.936).

See [`encoder-cpu-performance.md`](encoder-cpu-performance.md).

## Known limits

- **No memory preflight.** Submitting a job is not a guarantee it fits. Peak
  reserved VRAM on measured 1K jobs is 14.6–15.3 GiB of 16 GiB (FP16 text-to-image
  and edits); a 2K job is larger and uses tiled decode.
- **2K decode headroom is unmeasured.** An untiled 2048² decode does not fit,
  so 2K **presets** turn VAE tiling on and 1K presets turn it off (both the CLI
  `--preset` and the web UI). The web API refuses a config above 4096 latent
  tokens that leaves tiling off (`422 tiled_decode_required`); the CLI does not
  refuse — `--no-tiled-decode` on a 2K run is accepted and fails later, at
  decode. Tiled decode also leaves faint vertical dashed streaks in flat
  areas, which is why 1K jobs keep it off. A successful 2048² run was measured
  at ~3 m 39 s (20 steps, 15,835 MiB peak).
- **Q8 is experimental.** It fits where FP16 does not (it keeps the KV cache on
  for reference edits, and is roughly 1.6× faster at 1 reference: 58.2 s vs
  95.4 s denoise), but small image differences are expected. The published Q8
  profile refuses an F16 GGUF despite one being advertised alongside it.
- **Layer offload modes are unavailable.** Only `resident` works; `streamed`
  and `offload` are listed in the UI as not available.
- **Reference editing quality is model-dependent.** The over-processed
  1024×1024 artifact turned out to be seed/noise reuse — the test references had
  been generated at 1024×1024 seed 42, and editing them at that size and seed
  started from the identical initial noise, so the reference entered twice. Edits
  now derive their noise from `(seed, ordered reference ids)`, and 1024×1024
  seed-42 edits are clean in base FP16 and Turbo (`batch21`, fix `ab2f3ab`). See
  above for what still limits edits: memory, FP32 reference decode and tiling.
- **Single-user by design.** One shared job queue, no authentication; it is
  meant for localhost, a trusted LAN and a Tailscale network.

## Where the detail lives

| Topic | Doc |
|---|---|
| Turbo assets, sigma schedule, refusals | [`turbo-plan.md`](turbo-plan.md) |
| Transparency contract | [`transparency.md`](transparency.md) |
| Reference editing design and memory matrix | [`image-edit-design.md`](image-edit-design.md) |
| Rewriter contract, routes and errors | [`prompt-rewriter-plan.md`](prompt-rewriter-plan.md) |
| Encoder threading and attention | [`encoder-cpu-performance.md`](encoder-cpu-performance.md) |
| Web app behaviour and limits | [`web-interface.md`](web-interface.md) |
| Measured runs and performance reports | [`performance-results.md`](performance-results.md) |
| Dependency pins and driver notes | [`dependencies.md`](dependencies.md) |
