# Upstream feature review for the expanded plan

Reviewed 2026-10-24 for Euan's request to add editing, transparent generation,
8-step generation, all aspect presets and prompt rewriting. This is a planning
update; no model downloads, dependency changes, app deployment or new GPU tests.

The work items and acceptance criteria live in [PLAN.md](../PLAN.md), especially
M5, M6.2, M8 and W4. Existing V100 and maximum-quality requirements still apply.

## Primary sources

- [Qwen-Image-2.1 README, revision ab96caa944157ca28a4f36b695c3047549f48f98](https://github.com/QwenLM/Qwen-Image-2.1/blob/ab96caa944157ca28a4f36b695c3047549f48f98/README.md):
  single-image editing, ordered multi-reference composition (up to ten), native
  RGBA and seven recommended 2K ratio/size pairs. The exact dimension table is
  recorded once in PLAN M8.1. Current convenient landscape presets differ from
  those canonical sizes. The pinned raw README was read as well as the live page.
- [Official Turbo model card](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo):
  a separate checkpoint with a saved eight-step schedule. Step-count changes
  alone do not override that saved schedule. Its revision and compatible runtime
  must be pinned when implementation begins; base/40 remains our quality default.
- [Official T2I prompt enhancer](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I):
  a Qwen3.5-VL 9B derivative returning a rewritten prompt and aspect suggestion.
- [Official editing prompt enhancer](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I):
  a separate 9B checkpoint handling ordered images and either an explicit aspect
  suggestion or a reference-image aspect selection. These are proposals for the
  user to review in our planned UI, not permission to overwrite a draft silently.

The repository commit was checked through GitHub's public API. The Hugging Face
cards are dated review inputs, not immutable deployment pins. Future work must
record actual weight/config/tokenizer/system-prompt/runtime hashes in manifests.

## Current local implementation findings

- `encoder_production.validate_production_request()` refuses images; the bridge
  supplies `image=None`. Editing needs vision preprocessing/encoding, condition
  latents, masks and cache semantics, not just a new upload widget.
- `SuppliedQwen21Adapter.decode()` already validates a four-channel VAE output;
  `save()` accepts RGB and RGBA PNGs. Native transparency has useful foundations
  but has not been exposed and evaluated as an explicit end-to-end workflow.
- Current ratio shortcuts are square/landscape/portrait at 1K and 2K. The new
  catalog must keep saved custom dimensions reproducible while adding all seven
  nominal ratios and upstream's rounded dimensions. Some native dimensions cannot
  simply be halved and still satisfy our 32-pixel alignment rule.
- Current full FP16 base model works. Q8, official Turbo, image conditioning and
  the official prompt enhancers are not implemented as supported app modes.
  A square 2K run is measured; the added shape/mode combinations are not.
- Collector records exist, but the calibrated memory policy is not in the live
  CLI/web execution path. New conditioning and recipes need new descriptors and
  measurements; neither reduced weight size nor fewer steps proves memory fit.

## Two local Qwen planning workers

At the user's request, exactly two independent requests were sent concurrently
to the existing OpenAI-compatible service at `127.0.0.1:8079`, model `Qwen3.8`.
One drafted editing/RGBA/rewriting milestones; the other reviewed Turbo, aspects,
sequencing and V100 constraints. No additional workers or service changes were
used. Workers wrote drafts only; Codex retained responsibility for source checks
and the single integrated repository edit.

Integrated suggestions include ordered reference identity, staged validation of
reference counts, explicit rewrite review/apply, recipe identity, separate shape
calibration and immutable job inputs. Rejected/corrected draft claims include:

- Invented GPU headroom cutoffs, encoder memory estimates and CPU rewrite SLAs.
- Treating AMP as proof of a compatible native INT8 implementation, or assuming
  that shared architecture implies shared base/Turbo weights.
- Calling Turbo peak memory identical to base without measurement.
- Putting sampler-only settings into every embedding-cache key.
- Recreating already-present cooperative cancellation or claiming RGBA save must
  be built from scratch.

Raw local draft artifacts are retained in the ignored
`output/planning-review-2026-10-24/` directory for audit. They contain unverified
suggestions and are not implementation authority. No draft text was accepted as
an instruction to execute commands or change existing services.

## Proposed order and feasibility

First prove Q8 on V100 and integrate measured memory preflight. Ratio UI coverage
and the existing RGBA path can be audited early. Official Turbo needs independent
weights/schedule/runtime validation; editing needs a broader encoder/conditioning
change, first one input then graduated multiple inputs. Rewriter feasibility can
be studied separately, with measured CPU/quantized/serial placement and an
explicit review/apply UI. No new automatic use of AMD services is assumed.

The proposed tests measure image intent, identity/alpha quality, actual sampler
steps, peak resources and full wall time. They preserve the current FP16/base
reference and establish per-recipe regression anchors. None of these future
features is marked complete or given a promised speed/quality/memory result here.
