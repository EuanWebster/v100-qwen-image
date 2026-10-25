# Single- and multi-reference image editing — design

Status: design only, no `src/` changes. Written 2026-10-25 on branch
`feature/v100-features` for root to split into implementation tasks (§7).
Paths are relative to this worktree unless absolute. `pipeline.py` means the pinned
`/home/euan/Sandbox/v100-qwen-image/.venv/lib/python3.13/site-packages/diffusers/`
`pipelines/qwenimage21/pipeline_qwenimage21.py` (861 lines).

Marking: *(verified)* = read in the pinned source/tree this session. *(derived)* =
arithmetic from verified facts. *(unverified)* / *(measure)* = not established here;
nobody may treat them as known.

## 0. Starting facts (each verified this session)

- Runtime pins: diffusers `0.41.0.dev0`, transformers `5.17.0`, pillow `12.3.0`
  (`src/qwen_engine/_runtime_pins.py:12-13,57`).
- The pinned pipeline already supports condition images end-to-end (§1) *(verified)*.
  `qwenimage/pipeline_qwenimage_edit_plus.py` exists only for comparison; its rules
  differ (§1.1) and we do NOT copy them.
- Our engine calls the pipeline with `image=None` (`adapters/supplied_qwen21.py:240-244`)
  and the production encoder refuses images (`encoder_production.py:300-302`,
  "image conditioning is out of scope") *(verified)*.
- The text encoder (Qwen3-VL, BF16, ~17.5 GB) runs only in the CPU child
  (`encoder_worker.py`), parent-side entry `ProductionEncoder.encode`
  (`encoder_client.py:133-163`); GPU parent builds the pipeline with
  `text_encoder=None` (`adapters/qwen21_diffusers.py:343-345`) *(verified)*.
- Conditioning is cached by `EmbeddingCache` under `cache_key` v3, whose key document
  already includes an ordered `images` tuple of sha256 hashes
  (`encoder.py:113-116,451-452,482,630-638`) *(verified)*.
- Checkpoint transformer config (`models/assets/transformer/config.json`):
  `causal_condition=True`, 32 layers, 32 heads x 128 (width 4096), `mlp_ratio=3`,
  `in/out_channels=64`, `context_in_dim=4096` *(verified)*.
- Measured anchors (docs/performance-results.md): 1K FP16 40-step sampling 42.223 s
  (:53); post-job reserved 8.49 GiB (:41); 2K tiled decode peaks 15.55 GiB at tile
  1024/stride 768 (:76) and 9.69 GiB at 768/576 (:94) on the 16 GiB V100 *(verified)*.
- Wire caps today: web POST body 64 KiB JSON-only (`web/server.py:77,228,242-244`);
  encoder request file 256 KiB (`encoder_worker.py:37`); encoder response/cache entry
  256 MiB (`encoder.py:147,161`); job config document 256 KiB (`jobs.py:30,188`).
- Plans this implements: `docs/next-features-plan-2026-10-25.md` §3 (single ref, :85-99)
  and §4 (multi-ref, "target up to ten, subject to measured V100 limits", :101-111).

## 1. How the pinned pipeline consumes references (verified reading)

### 1.1 Preprocessing and resize rules

`__call__` (pipeline.py:511) with `image=[...]`:

1. Normalization (:607-625): each entry must be PIL or ndarray; a nested list is
   refused — `image` is one flat, **ordered** set shared by the whole batch.
2. Output size defaults come from the **last** image's aspect ratio, target area
   `output_resolution**2` (:630-634; default `output_resolution=1024`, :533). We always
   pass explicit height/width, so this branch never fires for us (§1.4).
3. `multiple_of = 32` clamp on height/width (:637-639); dims must be multiples of 32.
4. Per reference (:657-672): convert mode to RGBA if not already (:661-662), compute
   `calculate_dimensions(1024*1024, w/h)` (:666-667; function :149-156 — area
   1 048 576 px, both sides rounded to multiples of 32), then
   `image_processor.resize(...)` → `input_images` (PIL, for the VL encoder, :669) and
   `image_processor.preprocess(...).unsqueeze(2)` → `vae_images` (CHW(T=1) tensor in
   [0,1], normalized [-1,1], :670-672). One resize feeds both paths.
   `VaeImageProcessor` is built with `do_convert_rgb=False` (pipeline.py:208-210;
   default at `diffusers/image_processor.py:116`), so the 4-channel RGBA tensor
   survives `preprocess` (`image_processor.py:607,687-693`) *(verified)*.
5. Unlike `pipeline_qwenimage_edit_plus.py` (VL at 384², VAE at 1024²,
   :66-67,699-705), 2.1 uses **one** 1024²-area size for both; there is no extra
   per-reference shrink as the count grows. Max pixels per reference ≈ 1.05 MP after
   resize *(verified)*; uploads above that are accepted but downsampled (§2.3).

### 1.2 What the VL encoder receives

`_get_qwen_prompt_embeds` (pipeline.py:240) builds a raw template (never
`apply_chat_template`): the user turn becomes, in order, for N references a run of
placeholders `<image1>Picture 1: <vision…>`, then ` <image2>Picture 2: <vision…>`, …,
then the user prompt text (:222-226 template, :257-264 multi-image expansion) —
**all images precede the instruction text** *(verified)*.
Before tokenization, RGBA is composited over **white** for the vision tower only
(:271-278); the VAE path keeps the alpha channel (:661-672 + §1.1 step 4).
Processor is left-padded (:283-290); the final RMSNorm is neutralized by a forward
hook so `hidden_states[-1]` is pre-norm (:296-307). The returned third tensor
`image_pad_mask` is True exactly at the vision-pad token slots (:320-323).
Vision tokens per reference after Qwen3-VL patch/merge: expected ≈ 1024 per
1024²-area image *(unverified — architecture inference; confirm via `prompt_tokens`
in calibration facts, `supplied_qwen21.py:176-199`)*.

### 1.3 Reference VAE latents and the transformer sequence

`prepare_latents` (pipeline.py:452) encodes each `vae_images` entry with
`_encode_vae_image` (:427-449): `vae.encode` with `sample_mode="argmax"` — **no RNG,
deterministic** — then normalizes by `latents_mean/latents_std`, packs to
`[B, tokens, 64]` (16× spatial compression, one token per 16×16 px tile, :204-206),
and concatenates all references **in list order** (:463-489). In the loop, each step
feeds `hidden_states = cat([input_images_latents, latents], dim=1)` — references
**prepend** the noise (:778-780) — and slices `noise_pred[:, -target:]` (:796).
`img_shapes` lists one entry per reference plus the target (:721-730);
`img_mask` = `image_pad_mask` (vision slots in the prompt sequence) plus one slot per
2×2 group of target latent tokens (:752-755). How the transformer internally aligns
mask slots with the unpatched latent tokens is transformer-internal *(unverified;
not needed: we reproduce upstream inputs verbatim, §1.6)*.
`use_kv_cache=True` (default, :534) is active because `causal_condition=True`
(:762-764): step 0 runs `kv_cache_mode="extract"` over the step-independent prefix
(text + condition-image KV), later steps decode-only (:766-790).
Contrast (do not copy): edit_plus appends latents **after** noise (:811,
`pipeline_qwenimage_edit_plus.py`) and has no KV cache.
Decode is unchanged: only target latents are unpacked/decoded (:831-848); reference
latents never reach the VAE decoder.

### 1.4 Output size

Because our adapter always passes explicit `height`/`width` from `GenerationConfig`
(config.py:232-233; multiples of 16, range 64-4096, :33), the `image[-1]` aspect
heuristic (:630-634) never runs. Reference **order** therefore changes conditioning
numbers/latents order, never canvas size. UI MAY offer a client-side "match last
reference" action that fills width/height with the §1.1 `calculate_dimensions` result
for the chosen preset area; the server recomputes nothing (§5).

### 1.5 Device placement in our split

| Step | Where | Notes |
|---|---|---|
| upload/validation/thumbnail | web process (CPU) | new, §2.4 |
| PIL resize → VL pixels + white composite | encoder child (CPU) | §1.7 |
| Qwen3-VL forward (embeds, masks) | encoder child (CPU) | as today |
| reference VAE encode | GPU parent | during **load**, VAE alone before the transformer loads (§1.6 amendment 2) |
| noise + denoise loop | GPU | upstream `__call__` loop, §1.6 |
| decode | GPU (reloaded VAE) | unchanged |

### 1.6 Consuming our cached embeds with images: per-call `encode_prompt` override

`__call__` has **no** `image_pad_mask` parameter — it never forwards one (:687-693) —
and `encode_prompt` raises `ValueError("Pass image_pad_mask alongside prompt_embeds…")`
when both `prompt_embeds` and `image` are given (:370-374). But `encode_prompt` itself
**accepts** `image_pad_mask` (:347, :368-376). **Decision (amended at root review):**
instead of any adapter-owned loop, the adapter installs a small per-call **instance
override** of `pipe.encode_prompt` that injects the cached positive/negative
`image_pad_mask` (matched by the identity of the `prompt_embeds` tensor passed in) and
delegates to the original bound method, removed in `finally` — the existing VAE-hook
pattern. `__call__(image=[PIL...], prompt_embeds=..., negative_prompt_embeds=...,
height=…, width=…)` then runs upstream's own reference VAE encode (`prepare_latents`),
KV cache and denoise loop unchanged, so no loop-drift guard is needed. Rejected
alternative: an adapter-owned copy of the denoise loop reproducing the pinned inputs
input-for-input with a frozen-hash drift test (rejected: the override keeps upstream's
own loop as the single source of truth). ~~Consequence: the VAE must be resident
during `__call__`'s `prepare_latents`~~ **(amendment 2, measured on the V100):** a
mid-`__call__` VAE restore beside the resident transformer OOMs (15.05 GiB
allocated, +384 MiB requested, 1-reference 1K edit). The loader therefore encodes
every reference at LOAD time through the pipeline's own
`calculate_dimensions`/`image_processor`/`_encode_vae_image` while the VAE is the
only resident component (`reference_vae_encode` stage), drops the VAE, then loads
the transformer; a per-call `_encode_vae_image` override returns the stored
latents in order inside `prepare_latents` (removed in `finally`, refusal on
signature drift or count/geometry mismatch).

### 1.7 Passing reference images to the encoder child (IPC)

The request channel is a strict-JSON file capped at 256 KiB (`encoder_worker.py:37`) —
image bytes cannot ride on it. Design: bytes stay on disk; the request gains file
**paths** plus the content-hash binding that already exists:

- Bump `REQUEST_SCHEMA` (encoder.py:113) `…request.v2` → `.v3`; add top-level key
  `image_files`: absolute paths, same order as `EncoderRequest.images`.
  `image_files` is wire-only: NOT added to `EncoderRequest` and NOT in `key_document`
  (encoder.py:468-483) — identity stays content-hash-based.
- Parent (`ProductionEncoder.encode`, encoder_client.py:133-163) writes the document
  as today; `to_request_dict` (:486-490) gains the key.
- Worker `from_request_dict` accepts and returns `image_files` alongside
  `(request, request_id)`; `_encode_request` (encoder_worker.py:229-256) then, before
  any model load: read each file through a bounded FD (pattern: `read_bounded_json`
  :144-172, regular-file/no-follow), cap 32 MiB/file, re-hash bytes and require
  `sha256 == request.images[i]` (typed `EncoderRequestError` on mismatch → replacing
  a stored blob mid-run fails closed); decode with PIL; enforce decoded dimensions
  ≤ 4096×4096 before `load` (decompression-bomb gate); convert + resize by importing
  `calculate_dimensions` and the pipeline's `VaeImageProcessor` configuration from the
  pinned diffusers tree — one shared helper (§7 T1) so parent and child cannot drift;
  call `encode_prompt(prompt=…, image=resized_pils, device=cpu,
  num_images_per_prompt=1)` for positive and (when `cfg_active`) negative — replacing
  the hardcoded `image=None` (`encoder_production.py:521`).
- Mask policy: drop the all-False requirement (`encoder_production.py:419-432,1154`);
  require the mask be bool `[1,S]`, binary, and non-empty when `images` is non-empty;
  allow True slots only then (T2I keeps refusing True).
- Projection child (`--project-identity`) never opens image files; it echoes the
  request document (paths included) as today (:360-375).

## 2. Config, CLI and API changes

### 2.1 Engine config (`config.py`)

- `GenerationConfig.references: tuple[str, ...] = ()` — ordered content ids, each
  `[0-9a-f]{64}`, no count limit at config level beyond ≤ 10 (`REF_MAX`, plan §4 cap).
  Duplicates are legal (same image at two `<imageN>` slots).
- `GenerationConfig.references_dir: str | None = None` — server/operator directory;
  add to the directory-path checks (config.py:41-46) and to `RUNTIME_PATH_FIELDS`
  (web/api.py:55-57). Non-empty `references` without `references_dir` refuses in
  `_validate_policy` (adapters/qwen21_diffusers.py:193-229).
- `to_dict`/`from_dict` round-trip like existing fields (config.py:292-330); job
  envelope stays far below 256 KiB (10 × 64-char ids).

### 2.2 CLI (`cli.py`)

- `--reference PATH`, `action="append"`, repeatable: verify existence, ≤ 32 MiB,
  magic PNG/JPEG/WebP, decoded dims ≤ 4096; **copy** bytes into the reference store
  (`--references-dir`, default `<cache_dir>/references`) under its content id
  (no-clobber); config carries ids + dir only. `--references-dir` joins the
  path-flag loop (cli.py:123-125). `--dry-run` echoes ids, sizes, order — never
  resolves blobs.

### 2.3 Reference store (new `reference_store.py`, stdlib + PIL only)

Owns blob layout `references_dir/<64-hex>.img` + sidecar `<64-hex>.json` (width,
height, mode, format, bytes) + `<64-hex>.thumb.png` (≤ 256 px, LANCZOS, alpha kept).
API: `put_bytes(raw) -> id` (magic sniff, PIL decode, ≤ 32 MiB, ≤ 4096×4096 and
≤ 16.8 MP, no-clobber dedupe); `resolve(ids) -> tuple[Path, ...]` (strict-id regex —
never join untrusted text — regular file, re-hash equals id); `list_ids()`. No delete
endpoint; growth cap: refuse `put` beyond 512 blobs / 2 GiB with a typed refusal
(operator GC is a root task). Uploads keep original bytes verbatim: identity must be
over bytes the user chose. Re-encoding on upload was rejected (alpha loss, hash churn).

### 2.4 Web API (`web/api.py`, `web/server.py`, `web/generation.py`)

- Envelope: add `"references"` to `EDITABLE_FIELDS` (api.py:48-53) — an ordered array
  of 64-hex ids, ≤ 10; the browser never sends or receives paths
  (server-path leak guards api.py:264-290 keep holding; ids are content hashes).
- New endpoints (server.py `resolve_route` :248-296 + `header_policy` :178-244):
  - `POST /api/v1/references` — `Content-Type: application/octet-stream` (special-case
    the JSON-only rule at server.py:228 for this route), existing `X-Qwen-UI: 1` and
    single-`Content-Length` rules, route-specific cap 32 MiB (64 KiB
    `MAX_BODY_BYTES`, server.py:77, stays for JSON routes). Body → `put_bytes`;
    201 `{schema:"qwen_engine.web_reference.v1", id, width, height, has_alpha, bytes}`;
    typed 415 (magic/decode), 413 (size), 409 (store full).
  - `GET /api/v1/references/<id>` and `…/<id>/thumb` — serve stored bytes with sniffed
    content-type (reuse `image_content_type`, web/generation.py:186-194); 404 unknown.
  - No update/delete: replacing = new upload; the draft re-points at the new id.
- `WebGeneration.submit` (web/generation.py:150-170): after `build_config`, call
  `resolve_store_ids(runtime.references_dir, config.references)` — missing/tampered
  blob → 422 `reference_unavailable` with the offending ordinal only. Blobs are
  content-addressed and never rewritten, so upload→run TOCTOU is limited to deletion
  (typed failure). `public_job`/plan echo ids (already safe via `editable_config_dict`,
  api.py:323-331). Export/import flow through the same envelope unchanged.
- `tiled_decode_required` gate (web/generation.py:160-161) and `latent_tokens` stay
  target-image semantics; sequence-size admission is separate (§4.2).

### 2.5 Alpha handling

Upload side: alpha survives identically on both model paths — vision tower sees
white-composite (pipeline.py:271-278), VAE sees true 4-channel input
(`AutoencoderKLQwenImage21(in_channels=4)`,
`diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py:1136`) *(verified)*.
Output side already handles RGBA (decode emits 4 channels, supplied_qwen21.py:273-275;
`image_output.py:77-89` flattens only for JPEG).

## 3. Cache identity

`cache_key` v3 already hashes: prompt, negative, dtype, component identities, the
**ordered** `images` tuple, and `implementation_identity` (encoder.py:451-452,482,
630-638) *(verified)*. Consequences of the §1.7/§2.1 binding (ids = content hashes):

- Replace a reference (different bytes) → different id → different key. Reorder →
  different tuple order → different key. Duplicate ids encode as themselves — two
  identical references legitimately share with one only if the whole document matches.
- Preprocessing revision: bump `PROMPT_EXTRACTION_REVISION`
  ("qwen-image21.upstream-pre-norm.t2i.v1", encoder_identity.py:53) →
  `…ti2i.v2`, and it enters the key through `implementation_identity`
  (encoder_identity.py:311-321). Any future change to resize/composite rules MUST bump
  it. `worker_source_digest` (encoder_identity.py:301-306) additionally changes when
  encoder helper sources change — a second automatic invalidation *(verified mechanism;
  confirm the digest root covers `encoder_production.py` in T2 tests)*.
- Relax seed handling: `complete_encoder_seed` currently refuses `seed.images`
  (encoder_identity.py:374) — accept images, require the completed document's images
  tuple to equal the seed's; `_projection_fields` (:330-348) gains `images=` from
  `project_encoder_seed(config)` (:358-368). `kwargs` stays exactly `{"cfg_active"}`
  (:344, encoder_production.py:296-302) — generation size/steps/seed still never enter
  conditioning (T2I cache entries stay byte-identical across the upgrade because the
  revision bump only affects NEW encodes; old entries revalidate unchanged — verify in
  T2 that T2I hits survive a server restart post-upgrade).
- `validate_production_request` (:275-302): replace the images refusal with: count
  ≤ 10, ordered 64-hex, and non-empty images requires the request to carry
  `image_files` (worker-side check, §1.7).
- Tensor payload needs no format change: names/descriptors already carry
  `positive.image_pad_mask` (encoder_production.py:147,555-568); only the
  mask-value policy changes (§1.7). Response size stays within 256 MiB caps —
  embeds grow ~8 MiB bf16 per ≈1k vision tokens (derived; verify in §4).

## 4. GPU memory on the V100 — what exists during denoise, what to measure

Extra tensors vs T2I (1K target, references resized to 1024² area, derived from §1):

1. `input_images_latents` `[1, N·4096, 64]` fp16 — 0.5 MiB/reference *(derived)*.
   Negligible storage; its **sequence** effect dominates.
2. Per-reference VAE-encode activations at ≤1024² untiled — **unknown; measure**
   (first VAE-encode-ever on this card; decode-only history,
   docs/performance-results.md:74-94). Runs in its own phase before transformer load (§1.6).
3. `prompt_embeds`/masks grow with vision tokens (≈1k/ref, unverified §1.2) ≈ 8
   MiB bf16/ref *(derived)*.
4. Denoise sequence per step: `S_prompt + N·4096 + S_target`. KV cache
   (`use_kv_cache=True`, causal_condition=True) stores prefix K/V: 2(K,V) × 32 layers
   × 4096 width × 2 B = **512 KiB/token** → one 1K reference ≈ **2.0 GiB**, four ≈
   8.1 GiB *(derived formula; whether ref-latent KV is retained is UNVERIFIED —
   transformer-internal; MUST measure, and measure `use_kv_cache=False` as the
   fallback knob already plumbed through `supplied_qwen21.py:79`)*.
5. Attention compute/working set grows superlinearly with the joint sequence; SDPA fp16
   (V100: no FlashAttention2) working set scales with `seq²` head-slices *(inferred;
   measure step time)*.

Known correctness gap: MLP chunking decides on `config.latent_tokens > 4096`
(adapters/qwen21_diffusers.py:71-81,385-388). With references the joint sequence
exceeds 4096 at 1K/1-ref already → decision MUST use the joint length
`S_prompt + N·4096 + S_target` (S_prompt known only after encode — re-evaluate the
chunk decision after `encode`, before first step; T1).

**Measurement matrix (root, GPU; nothing here is guessable):** 1K preset (1024²),
seed 42, 40 steps, FP16 resident, one reference per run at counts **0, 1, 2, 4**, both
`use_kv_cache` on/off, sampling NVML peak + step time + allocator retries; VAE-encode
phase peak per count; vision-token count from `calibration_facts.prompt_tokens`;
repeat 4-ref at 2K preset with tiled decode. Decide defaults from data: max
references at 1K (target 10 per plan §4 only if it fits with headroom), whether 2K
editing ships at all, and the KV-cache default.

### 4.1 Running the measurement script (`scripts/benchmark_reference_memory.py`)

Implemented (CPU tests: `tests/test_benchmark_reference_memory.py`); the
recorder side it depends on is already in place — `calibration_facts` carries
`references` and the measured `joint_tokens` (`prompt_tokens + packed
reference tokens + latent_tokens`), and the standard adapter mirrors them
into performance-report metadata. `performance_summary.summarise_reports`
is not reused (it summarises T2I stage reports); the matrix report
`matrix.json` plays that role for these cells.

Plan-only default (no writes, no heavy imports; refuses an existing output
dir and a seedless config); GPU execution only with `--run`, fresh process
per cell, always through the lineage wrapper so an interrupt reaps the tree:

```
python -B scripts/process_lineage_owner.py -- \
  python -B scripts/benchmark_reference_memory.py --run \
    --config presets/1k-edit.json \
    --reference img1.png --reference img2.png --reference img3.png --reference img4.png \
    --counts 0,1,2,4 --kv-cache on,off --output-dir out/ref-matrix-$(date +%s)
```

Per cell (`counts` × KV policy; counts beyond the supplied files cycle them —
duplicates are legal): the references are ingested into a private
content store under the output dir (`reference_store`, ids recorded); the CLI
generation runs under `decode_bench.PeakMemoryProbe` (whole-run NVML +
allocator peaks + `num_alloc_retries` delta), a `PerformanceRecorder` report
(`cell-N.perf.json`), per-step progress timestamps (`step_ms`), and a fresh
allocator/NVML window around the adapter's `reference_vae_encode` stage
(`vae_encode` block; peak reset on entry only — the CalibrationCollector's
phase technique, never `empty_cache` mid-run; NVML from a second short-lived
session, degrading to allocator-only with an explicit note). The KV knob is
applied to the attached bridge adapter before its denoise and cross-checked
against the adapter's measured facts — a mismatch fails the cell. OOM is a
RECORDED cell outcome (`status: "failed", error_type: "OutOfMemoryError"`),
never a matrix crash; a child that dies without a record is `crashed`.
Artifacts: `plan.json`, per-cell `spec/cell-N.json/.perf.json/.log`, the
shared `store/`, and `matrix.json`. All reports are bounded (256 KiB),
prompt-free, exclusive-create; NVML peaks are interval samples
(`samples`/`interval_ms` state the coverage). No numbers here exist until
root runs it — the matrix decides §4's open defaults.

## 5. UI (web_static)

Create panel (`index.html:44-67`, app.js `buildEnvelope` :363-435):

- New fieldset "Reference images" between prompt panel and presets: file input
  (accept image/png,image/jpeg,image/webp), uploads to `POST /api/v1/references` as
  raw bytes (never base64 in JSON — 64 KiB JSON cap, server.py:77).
- Ordered list of numbered thumbnails (1..N via `GET …/thumb`), each with Remove and
  Move up/Move down buttons (no drag — keyboard-accessible); "Add reference"
  affordance; hard-stop at 10.
- Draft snapshot gains `references: [ids]` in the exact order shown; plan/export/import
  round-trip it; a stale draft referencing a forgotten blob shows the 422 reason per
  ordinal.
- Help text (static + `renderValidation` sync like the option-help pattern): "The
  prompt comes AFTER the images; number references as image 1, image 2 … in your
  words. Output size comes from the preset, not from the images. Alpha PNG/WebP keep
  transparency; the model composites alpha over white when reading them."
- Client validation before upload (magic sniff via `createImageBitmap` is NOT byte
  sniffing — send bytes and surface the typed server error instead): size > 32 MiB,
  undecodable, > 4096 px side, > 10 references → one-line `aria-live` message with the
  exact reason code text from the API; never a silent drop.
- Optional convenience: "match image N aspect" button setting width/height per §1.1
  rule client-side (server treats dims as user-set regardless).

## 6. Test plan

CPU (guarded offline runner; fakes; no torch import — keep the loop function
dependency-injected so fakes suffice):

- T1: store put/resolve/no-clobber/caps; tampered blob fails hash; id-regex refuses
  traversal; config validation (id shape, ≤10, dir pairing); preprocess helper golden
  vectors (synthetic PIL: side rounding to 32 at 1024² area, RGBA→white for VL copy,
  4-channel VAE tensor, L/P/CMYK→RGBA, byte determinism between two implementations of
  §1.1); request builder binds hashes 1:1 and refuses mismatch.
- T2: request v3 document round-trip incl. `image_files` (path list never enters
  `key_document`); worker refuses non-empty images without paths, count > 10, size and
  dims caps, wrong hash (typed); image_pad_mask policy (True refused for T2I, required
  and preserved through payload round-trip for TI2I); revision bump flips
  `implementation_identity`/key; seed↔completed images equality; T2I cache-key
  regression (a v2-era key for the same text-only document still hits).
- T3: envelope accepts `references`, refuses unknown fields; upload endpoint policy
  (octet-stream only on that route, size caps per route, magic refusals), GET/thumb
  serving, submit maps ids→blobs and 422s missing blobs per ordinal, public echo
  leaks no path; jobs: 10-id config within the 256 KiB cap.
- T4: guarded static suite extension: envelope carries ordered ids; thumbnail list
  renumbers after remove/reorder; >10 blocked with message.
- Drift guard (T1): test hashes the pinned pipeline.py denoise-loop region constant.
- Differential oracle (GPU, root): references=[] through the copied loop equals the
  pipeline `__call__` latents at fixed seed bit-for-bit (same device, same kv flag).

GPU acceptance (root only, per plan §3-§4 exits): 1K single-ref fresh→cache-hit byte
parity; replace-image invalidates (miss + visibly different result); reorder changes
conditioning but not canvas size; alpha-PNG edit preserves subject identity; §4
measurement matrix executed with report committed under `output/`; cancellation during
the reference-encode phase leaves no partial VRAM; 2K editing enabled only if the
matrix says so.

## 7. Implementation split (disjoint ownership)

Interfaces frozen here; T2-T4 start only after T1's helpers land.

- **T1 engine** — `config.py`, `cli.py`, `core.py` (plan echoes refs), NEW
  `reference_store.py`, NEW `reference_preprocess.py` (shared §1.1 rules: exports
  `resize_for_model(pil) -> (pil_for_vl, chw_tensor_for_vae, dims)` — pure,
  stdlib+PIL, imports `calculate_dimensions` from pinned diffusers),
  `adapters/base.py` (phase list), `adapters/qwen21_diffusers.py` (policy, phases,
  joint-length chunk decision), `adapters/supplied_qwen21.py` (per-call
  `encode_prompt` instance override injecting the cached `image_pad_mask` + phase
  plan + calibration facts — no copied loop, §1.6 amendment), `image_output.py`
  untouched; tests. Delivers the §4 measurement script too. Phase plan co-designed
  with root (reference VAE encode at load, VAE alone; §1.6 amendment 2).
- **T2 encoder child** — `encoder.py` (request v3, `image_files` wire key),
  `encoder_client.py` (pass paths), `encoder_production.py` (policy/mask rules,
  `image=` call), `encoder_worker.py` (bounded read + hash bind + decode caps),
  `encoder_identity.py` (seed relaxation, revision bump); tests. Needs T1's
  `reference_preprocess` import (landed first).
- **T3 web/jobs** — `web/api.py` (EDITABLE_FIELDS, endpoints, envelope),
  `web/server.py` (routes + header policy), `web/generation.py` (submit resolution,
  runtime `references_dir`); `jobs.py` unchanged; tests. Consumes T1 store API.
- **T4 UI** — `web_static/app.js`, `index.html`, `app.css`, static tests. Consumes
  §2.4 endpoint shapes (frozen above; can start mock-first in parallel with T3).

Order: T1 → (T2 ∥ T3) → T4 can run parallel from T1 onward (contract is §2.4/§5).
Integration owner: whoever lands last runs the full guarded suite once; GPU items
(§4, §6 oracle, acceptance) stay with root.

## 8. Open questions for root

1. KV-cache default under references — decide from the §4 matrix (2 GiB/ref derived
   retention may force `False` at ≥ 2 refs).
2. Whether `use_kv_cache` becomes user-facing (web field) or stays
   operator-only for the v1 cut (recommended: operator-only).
3. Whether the reference-encode phase joins the performance recorder phase names
   (yes recommended) and the calibration schema bump that requires (collector refuses
   on unknown fields — check `calibration.py` on landing).
4. Reference GC policy (store caps chosen here; deletion endpoint deliberately absent).
5. If matrix shows 10 refs infeasible at 1K: lower the advertised cap in UI copy and
   `REF_MAX` together — never silently clamp.
