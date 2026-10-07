# Upstream interface survey — Qwen-Image-2.1 in the pinned stack (task M0.2)

Survey of the exact upstream interfaces our future `Qwen21DiffusersAdapter` codes
against. Every claim carries a **line anchor into the verified local pinned
checkouts** (and the same commit's immutable GitHub permalink, provided the
default-branch file moves — the commit blob is the authority, not the link).

Pinned source identity (re-verified 2026-10-07 by `git rev-parse HEAD` in the
audit checkouts, matching `requirements.in` and the installed
`direct_url.json` from the M0.1 fix pass):

| Package | Commit | Local tree |
|---|---|---|
| diffusers | `c6df88a511a98740646ee55577b590c9852650ce` | `/tmp/v100_m0_ml/src/diffusers` |
| transformers | `856157a2f3e9594954310df18fdccc31ffddebe9` (v5.17.0 dereferenced) | `/tmp/v100_m0_ml/src/transformers` |

Permalinks (immutable, commit-pinned):

* diffusers: `https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/<path>#L<a>-L<b>`
* transformers: `https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/<path>#L<a>-L<b>`

Shortened below as `D:/…#L…` and `T:/…#L…`; expand with the prefixes above.

## Evidence classes — read this first

Every statement is tagged:

* **[S]** source evidence — read directly from the pinned checkout at the cited
  lines. True for those commits, full stop.
* **[I]** import smoke — the symbol imports cleanly in the temporary ML venvs
  (3.11.15 / 3.13.5, M0.1 fix pass, `QwenImage21Pipeline` import verified both
  interpreters). **No model weights were loaded and no pipeline inference was
  performed.** Honest limit: this is NOT an assertion that zero tensor code
  executed anywhere — import-time module code was never instrumented;
  "no model loading and no inference run" is what [I] claims and nothing more.
* **[U]** unverified — needs GPU execution, real checkpoints, or numerical
  fidelity work (M3/M4). Nothing in this document is [U] *evidence*; [U] marks
  the boundary of what [S]/[I] can prove: checkpoint fidelity, image quality,
  numerical agreement, GPU execution, VRAM behaviour, attention-backend
  performance on sm_70, and correctness of any GGUF quantization — all land
  with the gated milestones, not with this survey.

The taxonomy above is the tagging rule only; when each class was gathered and
what it does not prove is recorded once, in the §9 evidence ledger.

Key files:

* `D:src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py` (861 lines) [S]
* `D:src/diffusers/models/transformers/transformer_qwenimage21.py` (1018 lines) [S]
* `D:src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py` (1561 lines) [S]
* `D:src/diffusers/quantizers/gguf/utils.py` (624 lines) [S]
* `D:src/diffusers/loaders/single_file_model.py` / `single_file_utils.py` [S]
* `T:src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`,
  `T:src/transformers/models/qwen3_vl/processing_qwen3_vl.py`,
  `T:src/transformers/utils/output_capturing.py` [S]

Pipeline class `QwenImage21Pipeline`, VAE class **`AutoencoderKLQwenImage21`**
(the model-index uses this specific class — not the generic
`AutoencoderKLQwenImage`) [S: `D:src/diffusers/models/__init__.py#L54`,
`pipeline_qwenimage21.py#L171-L172`]. Registry import names `QwenImage21Pipeline`
[`D:src/diffusers/pipelines/__init__.py#L476`] [I].

---

## 1. Pipeline constructor

Immutable permalink: [pipeline_qwenimage21.py#L186-L232](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L186-L232).

`QwenImage21Pipeline.__init__(scheduler, vae, text_encoder, processor,
transformer, sample_sigmas=None)` — `pipeline_qwenimage21.py#L186-L232` [S]:

* Registers `scheduler` (FlowMatchEulerDiscreteScheduler), `vae`,
  `text_encoder` (`Qwen3VLForConditionalGeneration`), `processor`
  (`Qwen3VLProcessor`), `transformer` (`QwenImage21Transformer2DModel`)
  (#L197-L203) and `register_to_config(sample_sigmas=…)` (#L204).
* `vae_scale_factor = 16` hardcoded; `latent_channels = vae.config.z_dim` with
  fallback 64 (#L205-L208). Transformer default `in_channels=64`
  [`transformer_qwenimage21.py#L766`] — 16× spatial down, one token per 16×16
  pixel tile [S]. (Token math our config layer already uses: 1024²→4096 tokens.)
* `sys_prompt = "Comprehend and analyze the provided prompt."` (#L212).
* **Raw prompt templates — deliberate, do not "modernize" them** (#L213-L226):
  each template is a raw ChatML-style envelope — the checkpoint's message
  start/end control tokens around the `system` / `user` / `assistant` role
  segments — written as literal strings, interpolated with `str.format` and
  handed straight to `processor(text=…)`. The code comments state that
  `apply_chat_template` tokenizes *differently* and the checkpoint expects
  the raw form (#L213-L216). `prompt_template_ti2i` is the same envelope
  plus, before the `{}` prompt slot, a numbered per-image marker with its
  three vision-placeholder control tokens (T2I never uses them, #L222-L226).
  Verbatim from the pinned source (#L212-L232) — this mechanically copied
  block is the only sanctioned copy of those literal token bytes in this
  repo; they are never retyped by hand:

  ```python
  self.sys_prompt = "Comprehend and analyze the provided prompt."
  # The prompt is built as a raw template string and passed straight to
  # `self.processor(text=..., images=...)`, rather than going through `apply_chat_template`:
  # the two tokenize differently and the checkpoint expects this one. The "Picture 1: ..."
  # vision prefix only appears in the image-conditioned template.
  self.prompt_template_t2i = (
      f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
      f"<|im_start|>user\n{{}}<|im_end|>\n"
      f"<|im_start|>assistant\n"
  )
  self.prompt_template_ti2i = (
      f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
      f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
      f"<|im_start|>assistant\n"
  )
  # Number of leading system-role tokens to drop from the hidden states. Derived from the
  # tokenized system message rather than hardcoded, so it tracks the processor's template.
  sys_message = [{"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]}]
  sys_tokens = self.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
  self._drop_idx = len(sys_tokens[0])
  self._img_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]
  ```

* `_drop_idx` (number of leading system-role tokens stripped from the hidden
  states) is **derived at construction** by tokenizing the system message
  through `apply_chat_template` — the *only* place the chat template is used
  (#L227-L231); `_img_token_id` (#L232) is the token id obtained by encoding
  the tokenizer's vision-placeholder string (visible in the verbatim block
  above), marking VLM image slots. All of this is structural source evidence
  for the adapter contract, not runtime-behaviour proof.

`model_cpu_offload_seq = "text_encoder->transformer->vae"` (#L183) — relevant
to our M5 offload design. Constructor signature, both template-attribute
names and the `_drop_idx` derivation are asserted by
`tools/check_pinned_interfaces.py` (AST structure only — never a runtime
behaviour proof) [S].

---

## 2. Prompt encoding — template, drop index, masks, pre-RMSNorm hook

Immutable permalinks: [pipeline_qwenimage21.py#L240-L393](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L240-L393), [output_capturing.py#L234-L305](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/utils/output_capturing.py#L234-L305).

`_get_qwen_prompt_embeds` (#L240-L337) + `encode_prompt` (#L339-L393):

* **Blank prompts** become `" "` before formatting — Qwen has no BOS, an
  empty string leaves the encoder nothing to read (#L249). Never send a raw
  `""` from the adapter (the same trap as the CFG empty-string rule, §4).
* `processor(text=prompts, padding=True, padding_side="left",
  return_tensors="pt")` (#L284-L291) — left padding as trained [S]. In
  transformers 5.17 the processor returns `mm_token_type_ids` by default
  (`processing_qwen3_vl.py#L34`, `#L176`) and the pipeline forwards it when
  present (`pipeline_qwenimage21.py#L293-L301`) [S].
* **Pre-RMSNorm hook (critical, version-sensitive).** The code comments
  (#L303-L309) explain: `hidden_states[-1]` must be the last decoder-layer
  output *before* the text encoder's final RMSNorm — what the transformer was
  trained on. In transformers ≥5.0 output capture **ties** `hidden_states[-1]`
  to the *normalized* `last_hidden_state` by default
  (`output_capturing.py#L234-L305`, `@capture_outputs` on
  `Qwen3VLTextModel.forward` at `modeling_qwen3_vl.py#L774`; the pinned tree
  has no `tie_last_hidden_states` anywhere — grep-verified [S]). The pipeline
  neutralizes this with a forward hook on `text_encoder.model.language_model
  .norm` that **returns the module's input**, removed in a `finally`
  (#L310-L315). Consequences for us: (a) any future adapter that calls the
  text encoder directly MUST reproduce this hook or the config flag once
  available; (b) when the pinning of transformers moves to the release that
  lands huggingface/transformers#48087, the checker flips to drift on purpose
  (`tests/test_pinned_interfaces.py` covers it) — do not silently keep the
  hook plus the flag without re-surveying.
* **Extraction pipeline** (#L316-L337): unpad by attention mask
  (`_extract_masked_hidden`, #L234-L239) → strip `_drop_idx` rows (#L319) →
  build `image_pad_mask` by comparing unpadded ids to `_img_token_id`, also
  `_drop_idx`-stripped (#L321-L325) → re-pad embeds/mask/image-pad-mask to the
  batch max (#L327-L335). Returned triple is
  `(prompt_embeds, encoder_attention_mask, image_pad_mask)` (#L337) — the
  mask marks *VLM slots*; each slot later expands to 4 latent tokens (§5).
* `encode_prompt(prompt, image=None, device=None, num_images_per_prompt=1,
  prompt_embeds=None, prompt_embeds_mask=None, image_pad_mask=None)`:
  accepts **precomputed embeds + mask** (our encoder-subprocess path). With
  external embeds and `image=None`, a missing `image_pad_mask` is rebuilt as
  **all-False** ("embeddings without a mask can only be text") (#L369-L376);
  passing embeds that cover condition images without the mask is a hard
  `ValueError` (#L371-L375). An all-valid mask is collapsed to `None` —
  "a mask that carries no information costs the attention backends that
  reject one outright" (#L389-L391). Single-prompt `str` input is listified;
  `batch_size` comes from the prompt list or `prompt_embeds.shape[0]`
  (#L363-L365).
* **Hazard (upstream, UNVERIFIED for multi-output — documented, NOT patched):**
  `num_images_per_prompt > 1` repeats `prompt_embeds` (#L379-L380) and
  `prompt_embeds_mask` (#L384-L386) but **does not repeat `image_pad_mask`**
  (its rows stay at the original batch size; grep-verified [S]). The
  transformer only reads layout from row 0 ("samples share a layout",
  `transformer_qwenimage21.py#L910-L912`), so a homogeneous batch happens to
  survive — but that replication semantics is *untested by us and unstated by
  upstream*. **v1 pins `num_images_per_prompt=1`:** T2I-only is the user
  requirement (requirement 3); *one output image per `__call__`* is **our v1
  scope choice**, not a user-imposed one (it also sidesteps this hazard); the
  checker asserts the non-repetition
  so the note cannot rot silently. No upstream patch is made or proposed
  here.
* External-embeds + condition images cannot be expressed through `__call__`
  at all (it never passes `image_pad_mask` into `encode_prompt`, and the
  ValueError above fires when `image` is set): T2I-only v1 is unaffected [S].

---

## 3. `__call__` — signature, defaults, timesteps

Immutable permalink: [pipeline_qwenimage21.py#L511-L535](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L511-L535).

`__call__` (#L511-L535) v1-relevant defaults [S]: `num_inference_steps=40`
(#L519; the docstring is explicit that it is *ignored* when a sigma schedule
is used), `sigmas=None` (#L520), `true_cfg_scale=1.0` (#L516),
`num_images_per_prompt=1` (#L521), `output_resolution=1024` (#L533),
`use_kv_cache=True` (#L534), `output_type="pil"` (#L528),
`negative_prompt=None` / `negative_prompt_embeds=None` (#L515/#L526).
`height`/`width` fall back to `output_resolution` and are floor-rounded to
multiples of 32 (#L635-L642) — matching our config's multiple-of-32 rule.

**Sigmas precedence** (#L732-L735) [S]: call `sigmas` > pipeline-config
`sample_sigmas` (ctor `register_to_config`) >
`np.linspace(1.0, 1/num_inference_steps, num_inference_steps)`. Whatever the
source, a resolution-dependent shift `mu` is computed from the latent sequence
length via `calculate_shift` (#L736-L742, helper #L61-L71) and the scheduler
applies its sigma processing in `retrieve_timesteps(..., sigmas=…, mu=…)`
(#L743-L745, helper #L75-L133); the returned length *becomes*
`num_inference_steps`. Note for the manifest (M2): the pinned HF model index
contains no `sample_sigmas` (lead survey), so unless we set them, the base
`linspace(steps)` path applies — a config-level knob, not a hidden default.
Per-run instance state set in `__call__`: `_attention_kwargs` (#L644),
`_current_timestep` (#L645, #L775, reset #L836), `_interrupt` (#L646),
`_num_timesteps` (#L747) plus scheduler internals
(`set_begin_index` #L767, `step` #L816). All of it is *shared, mutable,
per-pipeline-object* — see §8 concurrency.

## 4. CFG inactive semantics — the `None` contract

Immutable permalink: [pipeline_qwenimage21.py#L677-L686](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L677-L686).

Activation is exactly (#L677-L686) [S]:

```
has_neg_prompt = negative_prompt is not None or negative_prompt_embeds is not None
do_true_cfg    = true_cfg_scale > 1 and has_neg_prompt
```

* A **non-None** negative of *any* kind arms guidance — including the empty
  string `""`, which `encode_prompt` would encode as the blank `" "` prompt
  (#L249), silently turning guidance on around an empty condition. Adapter
  rule (already in `config.cfg_active` semantics, enforced at the boundary):
  when `cfg_active` is false, pass `negative_prompt=None` **and**
  `negative_prompt_embeds=None` — never `""`, never empty embed tensors.
* scale>1 without a negative → warning, guidance stays off (#L679-L683);
  negative with scale<=1 → warning, off (#L684-L686).
* When active, the negative branch runs a second `encode_prompt` and a second
  transformer pass under its own cache context (#L697-L705, #L799-L813);
  combination `neg + scale*(cond-neg)` at #L813. **Guidance is two separate
  forwards with separate cache branches (`cond_cache`/`neg_cache`) — not one
  doubled batch**: per-forward activation stays single-batch; memory planning
  must not assume `2×batch` peak.

---

## 5. VAE / latent scales — no scalar factor anywhere

Immutable permalinks: [pipeline_qwenimage21.py#L416-L451](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L416-L451), [autoencoder_kl_qwenimage21.py#L977-L1004](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py#L977-L1004).

* VAE class `AutoencoderKLQwenImage21` (`autoencoder_kl_qwenimage21.py#L977`,
  ctor #L993-L1004: `z_dim=64` default, `dim_mult=[1,2,4,8,8]` → 16× spatial,
  `temperal_downsample=[False,True,True,True]`) [S]. The pipeline *reads*
  `vae.config.z_dim` (#L208); class defaults are not the contract — the
  checkpoint config is (verified against real files at M2.1 [U]).
* **Noise latents carry no scale factor.** `prepare_latents` draws plain
  `randn` and packs by spatial flatten (#L452-L492, `_pack_latents`
  #L416-L419 — "2.1 consumes latents unpatched"); there is no `/scale` like
  SD/Flux. Do **not** copy LTX/SD latent-scaling habits here.
* Condition-image latents normalize **per channel** with
  `vae.config.latents_mean` / `latents_std`: `(z-mean)/std` in
  `_encode_vae_image` (#L429-L451) and inverse `z*std+mean` at decode
  (#L842-L852) before `vae.decode` (#L853). The class ships concrete default
  lists (#L1004+); the checkpoint config remains the authority.
* Latent grid: `h/w -> 2*(h//32)`; token count `=(h/16)*(w/16)`
  (#L452-L456, #L721-L730 `img_shapes` `(1, h//16, w//16)`) — matches the
  4096@1024² config math.

## 6. Attention: default segmented processor, Flex opt-in, real backend selection

Immutable permalink: [transformer_qwenimage21.py#L462-L561](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/models/transformers/transformer_qwenimage21.py#L462-L561).

* Two processors: `QwenImage21AttnProcessor` (#L462-L549) and
  `QwenImage21FlexAttnProcessor` (#L369-L459). **Default is the segmented,
  non-Flex one** (`_default_processor_cls = QwenImage21AttnProcessor`,
  #L551-L561; Flex is opt-in via `set_processor`) [S]. The card's
  "FlexAttention block-causal" phrasing is *not* the default path.
* Segmented semantics: prefill = one `dispatch_attention_fn` call per prefix
  segment (text runs get a causal triangle over their own keys; image blocks
  attend prefix+own block) + one call for the target image (#L501-L540);
  decode (no `segments`) = single full attention over
  `[cached prefix, target]` (#L489-L499).
* **Actual backend selection (honest version).** The two call families differ
  (#L471, #L492-L500, #L522-L541): **decode** (no `segments`) passes
  `backend=self._attention_backend` — the *processor-specific* attribute,
  whose class default is `None` (the Flex processor's decode honours
  `set_attention_backend()`, #L378-L380; nothing sets it in our path).
  **Prefill** calls pass `backend=None` literally, so `dispatch_attention_fn`
  resolves the *active global registry backend* — default `native`
  (torch `F.scaled_dot_product_attention`) unless `DIFFUSERS_ATTN_BACKEND`
  or the `attention_backend` context manager set one
  (`attention_dispatch.py#L400-L430`, `#L267`,
  `utils/constants.py#L56`) [S]. **Do not read this as "Flash with automatic
  fallback on Volta" — and do not claim anything hardware-observed either:
  no V100 (no GPU at all) is installed here and no attention backend has
  ever been exercised on this machine.** With the default configuration the
  registry selects the native SDPA implementation (torch
  `F.scaled_dot_product_attention`); **which backend and which SDPA kernel
  (math / mem-efficient / flash-class) would actually run per call on sm_70
  is torch's runtime decision — unverified here [U]**. A backend explicitly
  selected via `DIFFUSERS_ATTN_BACKEND` / `set_attention_backend()` is used
  as configured — nothing surveyed here shows any user-selected backend
  automatically falling back. Do not claim a guaranteed kernel on sm_70
  without measurement; do not replace the
  processors with generic `AttnProcessor` (mask semantics would be wrong).
  The model deliberately keeps literal `backend=None` on prefill calls — the
  checker pins ≥2 such real call nodes so a future "fix" re-routing them is
  treated as a contract change.
* Flex path: prefill as one `flex_attention` call via a `BlockMask`
  (#L408-L441), **requires torch.compile** — the processor warns loudly
  because uncompiled flex materializes dense fp32 scores and OOMs at high
  resolution (#L410-L422). Decode honors `_attention_backend` set via
  `set_attention_backend()` (#L378-L380, #L446-L454). Compile-on-Volta is a
  later, measured decision [U].
* Mask/slot semantics: `img_mask` rows mark VLM image slots; the model reads
  **row 0 only** (samples share a layout) and expands each True slot ×4
  (`_IMG_TOKENS_PER_SLOT = 4`, #L910-L912); `build_token_metadata`
  (#L808-L845) assigns block ids from `img_shapes` token counts and *raises*
  if `img_shapes` and the mask disagree (#L829-L835) — replication mistakes
  fail loudly there, which is what the §2 hazard note stays honest about.

## 7. Single-file checkpoints (Qwen21 mapping) and GGUF fallback vs remote kernels

Immutable permalinks: [single_file_model.py#L199-L202](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/loaders/single_file_model.py#L199-L202), [gguf/utils.py#L34-L51](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/quantizers/gguf/utils.py#L34-L51).

* **Single-file is transformer-scoped only.**
  `SINGLE_FILE_LOADABLE_CLASSES["QwenImage21Transformer2DModel"] =
  {checkpoint_mapping_fn:
  convert_qwen_image21_transformer_checkpoint_to_diffusers,
  default_subfolder: "transformer"}`
  (`single_file_model.py#L199-L202`) — there is **no pipeline-level
  single-file entry**: a whole pipeline always needs the other components as
  separate local diffusers-format parts (§8) [S].
* The converter (`single_file_utils.py#L4327-L4343`) strips the
  `model.diffusion_model.` prefix and splits the ComfyUI-fused SwiGLU
  `img_mlp.gate_up.weight` into `gate_layer`/`proj` (gate rows first) [S].
* **GGUF runtime** lives in diffusers (`quantizers/gguf/`); the `gguf` PyPI
  package only supplies `GGUFReader`/`GGML_QUANT_SIZES`;
  `GGUFQuantizer.validate_environment` enforces `gguf>=0.10.0`
  (`gguf_quantizer.py#L58-L61`) [S].
* Optional CUDA kernels are **off by default and remote**:
  `can_use_cuda_kernels = os.getenv("DIFFUSERS_GGUF_CUDA_KERNELS","false")…
  and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7`
  (`gguf/utils.py#L34-L37`) — the gate is `>= 7`, **not `> 7`** (lead
  correction confirmed; **Volta sm_70 is not excluded by this source gate**).
  Enabling it (plus `kernels`) loads the *untrusted publisher* `Isotr0py/ggml`
  and requires `DIFFUSERS_TRUST_REMOTE_KERNELS=true` (#L39-L51). We ship no
  `kernels` dependency and never set these flags: baseline is the **native
  dequant policy** (`GGUFLinear.forward_native` dequantizes to `compute_dtype`
  then ordinary `F.linear`, #L593-L617); even the CUDA route has fused
  MMQ/MMVQ disabled (#L94-L107). This is a *policy/dependency choice*, not a
  Volta incompatibility; kernel performance on the V100 remains untested [U].
* GGUF loading enters through the quantizer with `from_single_file` on the
  transformer class with a `GGUFQuantizationConfig`
  (`quantization_config.py#L412`; same local-only discipline as
  §8); exact end-to-end GGUF load with a real file is an M4 task [U].

## 8. Local-only loading, KV lifecycle, and concurrency

Immutable permalink: [pipeline_qwenimage21.py#L762-L799](https://github.com/huggingface/diffusers/blob/c6df88a511a98740646ee55577b590c9852650ce/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L762-L799).

**Local-only loading (no-download contract, R3).** Every component loads from
explicit local paths with `local_files_only=True` —
`QwenImage21Pipeline.from_pretrained(<local pipeline dir>, local_files_only=True, …)`
(argument documented `pipelines/pipeline_utils.py#L701`, popped/consumed
#L785, propagated to component/model loading #L876/#L1001) and the
transformer-only single-file/GGUF path via
`QwenImage21Transformer2DModel.from_single_file(<file>, config=<local config
dir>, local_files_only=True, …)` (`single_file_model.py#L278+`). Adapter rule:
paths come **only** from `manifest.resolve_model_paths` (our resolver), never
from a repo id or ambient HF cache; prefer also exporting `HF_HUB_OFFLINE=1`
in the run environment as belt-and-braces. Nothing here was exercised against
real files (none exist) [U → M2/M4].

**KV lifecycle (per call).** `cache_enabled = use_kv_cache and
transformer.config.causal_condition` (#L762); fresh `cond_cache` and
(step-0 `"extract"` prefill, later `"cached"` decode; #L763-L764, #L776) —
**new `QwenImage21KVCache` containers every `__call__`**, never reused across
generations; the transformer validates mode/cache pairing
(`transformer_qwenimage21.py#L895-L904`). `cache_context("cond"/"uncond")`
(#L783/#L799) labels the denoise call for generic cache policies
(`cache_utils.py#L182-L197`) — the prefix KV itself is the explicit
`kv_cache=` argument. `use_kv_cache` changes fp16 output identity (docstring
#L588-L599) → it belongs in our reproducibility identity (M4), like the seed.

**Shared-pipeline concurrency: serialize.** One pipeline object holds the
mutable scheduler, `_current_timestep`/`_attention_kwargs`/`_interrupt`
(§3) and the per-call caches (§8): concurrent `__call__`s on one instance
**interleave state and must not happen**. v1 = one worker process, one
pipeline, calls serialized by construction; if M5 ever shares an instance
across threads, an explicit lock around `__call__` is mandatory [S: the
mutable-state sites].

## 9. Evidence ledger for this survey

| Class | What | When |
|---|---|---|
| [S] | Everything cited by line above; both pinned checkouts re-verified via `git rev-parse HEAD` | 2026-10-07 |
| [I] | `QwenImage21Pipeline` imports on 3.11.15 + 3.13.5 from the locked envs; signatures inspected in-process; **no model weights loaded and no inference run** (not a claim that zero tensor code ever executed — see [I] limits above) | M0.1 fix pass |
| [U] | checkpoint fidelity/quality, numerical agreement, GPU execution, VRAM, SDPA kernel identity/perf on sm_70, GGUF file loads, `from_pretrained`/`from_single_file` end-to-end | M3/M4 gates |

Contract automation: `tools/check_pinned_interfaces.py` (stdlib `ast` only —
no ML imports, no network, R7-safe) asserts **53 structural contract checks**
on the pinned trees — 38 AST-code checks and 15 text markers — covering the
facts this document depends on. Verified 53/53 (exit 0) against the pinned
trees on 2026-10-07; deliberate drift copies (steps default 40→4, CFG-gate
rewrite incl. *extra boolean operands* like `... or True` / `... and False`,
**a second assignment/overwrite of `has_neg_prompt` or `do_true_cfg` in
`__call__`**, `>=7`→`>7`, Flex-default processor swap, `image_pad_mask`
repetition "fix", `tie_last_hidden_states` arrival, norm-hook moved out of
the prompt-encoding method, **encoder forward moved out of the try whose
finally removes that registered hook handle**) each fail loudly with exit 1;
a source file that cannot be parsed
(SyntaxError or bare ValueError) is a contract FAIL (exit 1), never a crash;
missing/unreadable/non-UTF-8 inputs exit 2 = *not run*.
`tests/test_pinned_interfaces.py` adds 17 always-run synthetic regressions
(fresh state per invocation, comment/docstring-only evidence rejected,
commented-out CFG/hook code, missing methods, signature/kind drift, CFG
overwrite, forward-outside-the-hook-protected-try, removing a different
handle name, parse
failures, input errors) and executes the real-tree check only when
`QWEN_DIFFUSERS_SRC`/`QWEN_TRANSFORMERS_SRC` name real trees — otherwise it
skips **explicitly**; a skip is never a validation claim.

**Checker limitations (read before citing it as proof).** (1) AST *structure*
only: every PASS — including the "code" kind — says the surveyed shape exists
in the file; the checker never executes upstream code, so no check proves
runtime behaviour, numerical fidelity, or absence of network calls; that proof
belongs to adapter-level tests (M4+). Two specific scope statements: the
CFG-gate checks are an **AST shape tripwire** — the unique-assignment +
exact-operand shape is verified statically, and this is NOT runtime proof that
the interpreter reaches or uses that assignment; the pre-RMSNorm-hook check
confirms the surveyed forward-inside-try / finally-removes-that-handle shape
but does NOT prove execution or full exception safety (e.g. a raise between
registration and try entry is outside its scope). (2) Inputs are
caller-provided trees:
the checker does NOT verify them against the pinned commits above — revision
verification (`git -C <tree> rev-parse HEAD` vs the recorded hashes) is a
separate required step, done manually at the dates in the ledger. (3) The
guarded CPU suite never imports torch/diffusers/transformers or loads
weights; model loading and inference have NOT been performed anywhere in this
project to date — that statement is about our actions, not a proof that no
tensor code executed as an import side effect. (4) No GPU is installed and no
attention backend/kernel has ever been exercised here; nothing in this survey
or the checker constitutes a hardware-observed claim (§6).

*v1 scope reminder (confirmed requirements + lead review): T2I-only is the
user requirement; one image per `__call__` (`num_images_per_prompt=1`) is our
v1 scope choice; raw upstream templates, drop-index, mask and pre-RMSNorm-hook
semantics preserved verbatim; no upstream patch is made or endorsed.*
