# Next requested features — review before building

Planning-only checkpoint, 2026-10-25. No feature implementation or model download
is started by this document. V10016GB remains mandatory; maximum quality is the
priority. Preserve the working FP16/base path, Q8 option, existing AMD/services,
saved images and current user changes. Q4 stays removed.

## Feature inventory

| Requested capability | Current status | Remaining deliverable |
|---|---|---|
| Image editing, one reference | Not implemented | Upload/select one image, enter an instruction and generate an edit; preserve source image and requested content. |
| Image editing, multiple references | Not implemented | Numbered references with add/remove/reorder controls and prompts that can identify each reference. Target up to ten, subject to measured V100 limits. |
| Transparent image generation | Underlying four-channel decode/save path exists; no explicit workflow | A transparent-background choice, preserved original/effective prompts, real alpha preview and PNG/WebP output. |
| Official eight-step generation | Not implemented | Base-quality and official Turbo choices with the correct checkpoint and saved eight-step schedule. |
| Prompt rewriting | Not implemented | Explicit Rewrite action, original/proposed text, edit/apply/keep-original controls and optional ratio suggestions. Extend to edit instructions after reference support. |
| All supported aspect ratios | Delivered in UI/CLI | All seven ratios already have1K/2K presets; finish per-shape memory/quality validation, especially2K. |
| Less repeated model loading | Requested performance concern; not implemented | Evaluate reusable model/worker state and an explicit unload action without exhausting GPU or system RAM. |

Already delivered: JPG/PNG/WebP selection, date/seed filenames, config import,
themes, accessible option explanations and the fixed prompt/preview layout.

Upstream capability references: [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1)
and [official Turbo checkpoint](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo).
These describe model capabilities; they do not establish fit on this V100.

## Proposed build order

### 0. Finish memory admission and improve2K decoding

Reason: both Q8 and Q4 trials reached about15.57GiB during2K decoding and logged
an allocation retry. A single saved square image does not establish reliable
headroom for other shapes, editing, alpha or new recipes.

- Wire fresh host/GPU resource checks and artifact/runtime-specific calibration
  into actual dispatch/phase boundaries. Account for memory already owned by
  the worker; comparing whole-process peaks directly with remaining free VRAM
  mid-run would double-count allocations.
- Establish conditioning facts before expensive allocation. Resolve admission
  for new prompt lengths through exact evidence or explicitly reviewed
  conservative envelopes; do not silently substitute a measured24-token prompt
  for an arbitrary new request.
- Benchmark smaller/alternative tile geometries against the present quality
  reference. Change defaults only after inspecting seams, fine detail and alpha
  edges, with measured headroom. Retain the existing working geometry meanwhile.
- Measure all seven shapes at1K, then2K where feasible, for FP16/Q8 and their
  actual decode settings. Report unsupported or unmeasured combinations clearly.
- Keep a deliberate calibration path separate from normal job admission so
  unknown profiles can be measured without inventing evidence or silent bypasses.

Exit: reliable1K baseline; documented2K envelope; early useful failures;
cancellation/recovery still works. Existing calibration results are historical
evidence and must match the implemented source/profile before reuse.

### 1. Transparent generation

- Add an explicit alpha-output setting to versioned config, CLI, API and UI.
- Preserve the original prompt and show/record any applied RGBA prompt wrapper.
- Automatically select an alpha-capable format when the user enables
  transparency and explain the change; PNG or lossless WebP. Preserve JPG as
  the default for ordinary images.
- Preserve alpha through tiled/untiled decode, saved images, downloads and
  thumbnails. Checkerboard is a preview background only.
- Check useful transparent/semitransparent pixels, fine edges over light/dark
  backgrounds and repeated exports. A nearly opaque RGBA image is insufficient.

Exit: real1K transparent output with unchanged opaque mode;2K enabled only after
separate alpha memory/edge validation.

### 2. Official eight-step Turbo

- Review the pinned pipeline's saved-schedule support in an isolated runtime.
- Obtain and pin the correct Turbo assets explicitly, with distinct manifest,
  recipe and calibration identities. Do not mix base-transformer weights with
  a Turbo label or simply change the base step count to eight.
- Add a clear Quality/Base versus Turbo choice. Keep base40 as the quality
  default; progress/job metadata must reflect actual recipe steps.
- Compare portraits, products, landscapes, illustration and text at fixed
  seeds. Measure loading, encoding, sampling, decoding and total latency
  separately; fewer steps do not make loading faster.

Exit: finite V100-compatible execution, inspected1K output, measured trade-offs;
then test2K and transparent combinations separately. No fixed speedup promise.

### 3. Single-reference image editing

- Add a bounded image upload/input snapshot and thumbnail with a clear edit
  instruction. Keep upload ownership and source images separate from outputs.
- Extend the encoder worker's image processing/embeddings/masks, reference VAE
  latents and the upstream image-conditioned pipeline path. Current production
  code rejects reference images and calls the pipeline with image=None.
- Record image content/preprocessing identity in conditioning caches and job
  metadata. Replacing an image must invalidate the cached conditioning.
- Support native alpha inputs/outputs where tested. Test cancel/failure cleanup
  and a fresh versus cached edit with identical inputs.

Exit: meaningful1K edit that preserves requested identity/content, with correct
reference ownership and memory checks. Expand resolution/recipe combinations
only after measurements.

### 4. Multiple-reference editing

- Extend the working single-reference path to an ordered list, with numbered
  previews, removal/reordering and unambiguous prompt references.
- Bind ordered image identities and preprocessing to the cache/request snapshot.
- Measure2references, then4, then higher counts toward the upstream ten-image
  target. Validate combined image sizes, conditioning length and VRAM together.
- Show the measured local limit. Do not advertise ten full-resolution images
  merely because the upstream model accepts ten inputs.

Exit: successful multi-subject editing/composition and correct reorder/replace
behaviour at the supported local limits.

### 5. Prompt rewriting

- Evaluate the official generation/edit rewriters in a separate local runtime;
  this is a separate model requirement, not an existing engine feature.
- Preserve services and release rewriter resources before image generation.
  Measure feasible CPU/quantized execution on this machine before choosing it.
- Rewrite is an explicit action. Show original/proposed text with Edit, Apply
  and Keep Original; accept proposed dimensions/reference selection explicitly.
- Preserve exact requested text, subjects/counts and edit intent. Validate
  structured output and discard stale replies after draft edits. Failures and
  timeouts leave the user's draft intact; no hidden external fallback.

Exit: useful rewriting with measured latency, preserved intent and no silent
changes. T2I rewriter feasibility may be researched earlier; edit rewriting
depends on reference-image support.

## Performance companion: model reuse

Study after the memory foundation, without blocking every feature on it.
The current web architecture owns a generation process per job, and resident
means resident during sampling, not retained across jobs. Keeping models warm
therefore needs explicit worker lifetime and memory ownership changes.

Compare a reusable CPU model cache and a reusable owned worker against current
reloads. Measure system-RAM impact as well as VRAM; the encoder and other local
services must retain headroom. GPU model coexistence is a separate fit test.
Provide an unload/release-memory action and clean worker restart on cancellation,
failure or changed model/profile. Do not promise that transformer and VAE can
stay on the card together at2K.

Streamed offload remains a lower-priority capacity option: first establish a
need after decoder work. Its transfer cost must be measured on the actual
Gen3x4 connection. Define the ambiguous generic offload mode before implementing
it; do not expose placeholder modes as working choices.

## Delivery checks for each increment

Ship engine/CLI support and then the corresponding UI as one usable increment.
Include reproducible seed/recipe/reference/effective-prompt metadata, real
progress, cancellation and current config import/export compatibility. Run
focused CPU checks and real-Chrome UI checks, then meaningful V100 acceptance
for the newly enabled profile. Preserve current services and commit each
milestone with an updated handoff. Persistent History is a useful separate
companion improvement; it is not a prerequisite for all requested modes.

Development assistance: the requested DeepSeekV4.1FlashFree through OpenCode
previously returned a free-quota429. Recheck availability when building; do not
silently switch to a paid provider. No agents or builds were launched for this
planning checkpoint.
