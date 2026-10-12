# Next CPU milestone: encoder helper and safe tensor codec

Senior-lead design refinement, Codex/root, 2026-10-12. Not implemented yet.
Local Qwen through dsh remains the sole product writer; root and one read-only
peer review. Preserve user sandbox.yaml, services, assets and dependency locks.

The complete older contract is in 2026-10-09-next-worker-contract.md. The next
implementation task is ONLY lazy supplied-model upstream reuse plus safe tensor
codec: no loader, worker CLI, request factory, package validator or adapter yet.
Current detailed scratch briefs: /tmp/v100-worker-tensors-task.md,
/tmp/v100-m22b-worker-task.md, /tmp/v100-encoder-helper-reviewed-scope.md.

Reviewed refinements:

- Keep generic encoder.py request/cache transport stdlib-only and unchanged.
  Separate production validation rejects non-scalar/blank positive text, images,
  missing implementation identity, noncanonical fp16/bf16/fp32 and anything but
  exactly {cfg_active: bool} BEFORE heavy imports. Active CFG requires nonblank
  negative; inactive never encodes negative but retains its text in identity.
- Supplied processor/model only: pinned QwenImage21Pipeline with other components
  None, encode_prompt explicitly CPU/one image/inference_mode. Never copy prompt
  control tokens or substitute tokenizer-only/chat-template encoding. No loading
  or moving models. Upstream does not cast output dtype: actual embeddings must
  match requested dtype, never silently coerce a mismatch to make tests pass.
- Small typed branch record: floating finite embeddings [1,S,H], nullable int64
  attention mask [1,S] (binary, at least one valid token), mandatory bool image
  mask [1,S] all false. Positive/negative may have different S, but H/dtype agree.
- One exact safetensors blob, versioned metadata, full request identity, exact
  branch/tensor/descriptor keys. Generic cache remains opaque. Before library
  deserialization cap bytes/header/count/nesting; validate integer shapes excluding
  bool, products, offset spans versus actual bytes. Then validate actual tensor
  dtype/shape/finiteness/masks without casts/reshapes/drops. Preserve null masks and
  bitexact finite values. Invalid request is EncoderRequestError, malformed payload
  is EncoderProtocolError; wrap library exceptions, not KeyboardInterrupt/SystemExit.
- Separate tiny synthetic checks on BOTH existing pinned ML interpreters, socket
  denial BEFORE imports, -B, offline flags and fresh temporary HOME/XDG/HF caches,
  writes confined to owned temporary fixtures. Run sequentially with child-only
  OMP/MKL thread limits of one. No real processor/tokenizer/checkpoint load. Cover
  internal unequal batch, blank normalization, optional multimodal IDs, norm-hook
  cleanup on success/failure, nullable masks, CFG branches and all malformed codec
  cases. Main guarded suite remains ML-free.

Root verified 2026-10-12 16:38 via stdlib distribution metadata only:
ML311 Python3.11.15 / ML313 Python3.13.5, both torch2.14.1+cu126,
diffusers0.41.0.dev0 @c6df88a511a98740646ee55577b590c9852650ce,
transformers5.17.0 @856157a2f3e9594954310df18fdccc31ffddebe9, safetensors0.8.0.
torchvision absent in both. This is metadata evidence, not runtime/GPU fidelity.
Existing reviewvenv311 works too; earlier unavailable-Python3.11 claim was wrong.

Standing gates remain: V100 not installed (last user confirmation), real processor
load blocked by missing/unlocked torchvision, encoder17.5GB checkpoint exceeds
16GiB container budget even before overhead, complete host-memory evidence unknown.
No real load, image generation, numerical checkpoint parity or GPU/fit claim.
Commit narrow milestones, verify, handoff, stop for root review before wiring.
