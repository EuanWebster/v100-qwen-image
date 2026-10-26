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

**Status: works end to end, with a known visual defect that has a measured
cause.** One reference produces a genuine edit (a red cube became a blue cube
with the table, lighting and camera preserved — see `output/gpu/batch5` and
`batch9`), but at 1024×1024 the output looks over-processed and over-sharpened.

**Cause** (root, V100, `output/gpu/FINDINGS-edit-grid.md`, snapshot `df5d2a3`):
the defect appears exactly when the output latent grid equals a *reference's*
latent grid. References are always resized to a 1024² area
(`REFERENCE_TARGET_AREA`, `src/qwen_engine/encoder_production.py:166`), so a
square 1024×1024 output collides with any square reference. The rule is about
the grid, not the number 1024: an output that happens to match *any*
reference's resized grid collides too. Measured via
`_pinned_reference_helpers().calculate_dimensions` at
`REFERENCE_TARGET_AREA`: a square reference becomes 1024×1024, 4:3 becomes
1184×896, 3:2 becomes 1248×832, and 1024×960 becomes 1056×992 — so a 1024×960
output cannot collide with a square reference. Same settings
(FP16, seed 42, 40 steps, same cube reference and prompt) throughout:

| Output size | Result | Evidence |
|---|---|---|
| 1024×1024 | artifact | `batch5`, `batch6`, `batch9` (Turbo), `batch10`, `batch11` (2 refs) |
| 512×512 | clean | `output/gpu/batch12/edit-fp16-512` |
| 768×768 | clean | `output/gpu/batch12/edit-fp16-768` |
| 1024×960 | clean | `output/gpu/batch12/edit-fp16-1024x960` |
| 1024×960, 2 references | clean, correct composition | `output/gpu/batch12/edit2-fp16-1024x960` |

Two references of equal size are fine — only *target == reference* collides.
Likely mechanism: the RoPE centres each image block's grid on zero, so a
same-size target shares every spatial position with the reference.

**Workaround until the fix lands** (in progress — re-area the references rather
than change your output): for edits, avoid 1024×1024. Use a non-square size
(`--width 1024 --height 960`) or a different square (512, 768); all four were
measured clean. Note that the `square` preset *is* 1024×1024, so pass
`--width/--height` (web: the size fields) rather than the preset.

A plain 3×3 Laplacian variance does **not** detect this defect — it tracks
resolution, not the artifact: 1024² edit outputs score 458.6 (`batch5`) and
427.6 (`batch9`), the clean 1024×960 output scores 235.8, but the equally clean
512² output scores 440.5 and the equally artifacted `batch10` scores 254.5. Size
is the discriminator, so judge edits by eye, not by a sharpness number. Do not
treat editing as a production feature yet.

How to use it:

1. **Web:** under *Reference images*, press **Add reference** and pick up to 10
   PNG/JPEG/WebP files. They are numbered in the order shown (reorder with the
   per-card **Up**/**Down** buttons, remove with **Remove**) and you refer to
   them as "image 1", "image 2" … in the prompt. The prompt is sent after the
   images. Output size comes from the preset, not from the images — so for
   edits, override **Width**/**Height** (e.g. 1024×960) instead of taking the
   1024×1024 default, per the workaround above.
2. **CLI:** `--reference PATH` (repeatable, ordered, at most 10) plus
   `--references-dir DIR` (default `<cache_dir>/references`). The bytes are
   copied verbatim into a content-addressed store; the run carries only ids.

```bash
python cli_gen.py --run \
    --prompt "Change the red cube in image 1 to a blue cube. Keep the table, lighting and camera exactly the same." \
    --reference path/to/cube.png --preset square --width 1024 --height 960 --steps 40 \
    --manifest models/manifest.json --models-dir models/assets \
    --cache-dir cache --output-dir output/edits
```

The `--width 1024 --height 960` pair is the workaround above, not a preference:
`--preset square` alone is 1024×1024 and hits the grid artifact. Verified with
`--dry-run`, which resolves to `width: 1024, height: 960`.

Limits and automatic behaviour (each measured, none silent):

| Behaviour | Detail |
|---|---|
| KV cache | FP16 + references keeps the KV cache **off** automatically (KV on OOMs at the first denoise step). Q8 keeps it on. Recorded as `kv_cache_policy` in the run metadata. |
| Decode precision | References always decode in FP32, including opaque output (a Q8 2-reference run sampled fine and died at `vae_decoding` in FP16). |
| File caps | 32 MiB per file, sides ≤ 4096 px, ≤ 4096² pixels, at most 10 references. Larger images are accepted, then resized to about one megapixel for the model. |
| Transparency in a reference | The alpha channel is kept in the stored bytes, but the vision encoder sees the image composited over white. |
| First encode cost | The text+vision encoder runs on CPU: an uncached 1024² reference costs ~66–73 s to encode; two references ~107–116 s (measured 116.0 s on one 2-reference job). Repeats hit the embedding cache. |
| Turbo + references | Allowed and **run**: `output/gpu/batch9/turbo-edit` (1024², 1 reference, denoise 19.2 s, whole job 33.0 s) and `output/gpu/batch10/turbo-fisher-hat`. Both are unmeasured-quality runs — the adapter still records `turbo_reference_run_unmeasured` — and both are 1024×1024, so both show the grid artifact described above rather than saying anything about Turbo's edit quality. |

Measured timings, 1024², 40 steps, seed 42 (denoise / whole job). Every row is
a 1024×1024 run, so these measure the artifacted configuration — treat them as
cost, not as a quality comparison:

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
characters) or unsupported task.

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
- **Reference editing quality is the open problem** — a same-size target and
  reference collide, so 1024×1024 edits show the artifact; 512², 768² and
  1024×960 are clean, and the fix is in progress — see above.
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
