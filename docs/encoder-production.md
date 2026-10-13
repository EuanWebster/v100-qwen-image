# Supplied-component CPU encoder and tensor codec

Accepted CPU-only helper milestone, 2026-10-13. Implementation:
`src/qwen_engine/encoder_production.py`. This is not yet a model loader,
production worker CLI, adapter, or generation endpoint. The web workspace
continues to offer planning/configuration export, not image generation.

## Boundary

Importing the module is stdlib-only. `validate_production_request` rejects
invalid requests before any heavy import: a scalar nonblank positive prompt,
no images, complete implementation identity, exact `fp16`/`bf16`/`fp32`,
and kwargs exactly `{"cfg_active": bool}` are required. Active CFG requires
nonblank negative text. Inactive CFG does not encode the negative text but
retains it in the cache identity.

`encode_with_supplied_model(model, processor, request)` requires an already
supplied torch module whose parameters and buffers are on CPU, and a callable
processor. It constructs the upstream `QwenImage21Pipeline` with scheduler,
VAE and transformer set to None, then calls the upstream `encode_prompt`
on CPU under inference mode, with no images and one image per prompt.
It never loads weights, moves components, casts output, or duplicates the
upstream prompt template/extraction/norm-hook implementation. The lower-level
`encode_with_supplied_pipeline` accepts an already-constructed pipeline.

Returned tensors must be dense, strided, contiguous CPU tensors:

- Finite embeddings `[1, S>=1, H>=1]`, exactly the requested floating dtype.
- Nullable int64 attention mask `[1, S]`, binary with a valid token.
- Mandatory bool image-pad mask `[1, S]`, all false for text-to-image.

CFG branches may have different sequence lengths but must agree on hidden
width and dtype. Null attention masks remain null; no substitution occurs.

## Payload

An `EncodedPayload` contains exactly one `tensors` blob, encoded using
`safetensors.torch.save`. Schema `qwen_engine.encoder_tensor_payload.v1` and
codec `qwen_engine.encoder_safetensors.v1` bind the exact full request document,
cache key, dtype, CFG decision and tensor descriptors in both the outer
metadata and safetensors metadata. Decode validates the request's own cache
key/policy even when `expect_request` is omitted. Passing `expect_request`
also binds the payload to that specific caller's request.

Before library deserialization, the codec bounds the raw blob to 256 MiB,
header to 1 MiB, JSON depth to 8 and nodes to 8192; refuses duplicate keys,
unsupported dtypes, noninteger/bool dimensions, excessive ranks/products,
out-of-range or overlapping offsets and unclaimed data; checks header,
descriptor and identity agreement. The production schema permits at most
six tensors. After deserialization it verifies actual tensors, finite values
and mask semantics without casts or reshaping. Finite tensor values round-trip
bit-exactly, including signed zero.

The raw blob cap is not a wire-size allowance: generic response/cache caps
apply independently, including base64 and metadata overhead. A future worker
must budget its complete response accordingly.

Invalid requests raise `EncoderRequestError`; malformed tensors/payloads raise
`EncoderProtocolError`; unavailable dependencies or invalid supplied setup
raise `EncoderSetupError`; upstream encoding failures raise `EncoderWorkerError`.
KeyboardInterrupt/SystemExit are never swallowed.

## Reproduce the review

Environments, runtime and cache persist under gitignored `.review-envs` on
the project mount. Reusable tests/scripts are committed, not stored in `/tmp`.
Only disposable guarded fixtures use temporary directories.

```bash
bash scripts/setup_review_envs.sh
bash run_cpu_tests.sh
PYTHON=/work/.review-envs/test311/bin/python bash run_cpu_tests.sh
.review-envs/py311/bin/python -B tools/synthetic_ml_encoder_check.py
.review-envs/py313/bin/python -B tools/synthetic_ml_encoder_check.py
```

Verified 2026-10-13 by Codex: full guarded suites **983 passed, 2 skipped**
on each Python version (3.11.15 and 3.13.5). These suites refuse ML imports
and use tensor/library test doubles. Separate real-ML checks execute the
locked upstream prompt method and safetensors codec with tiny synthetic
model/processor components: **36/36 checks on each interpreter**. They check
fp16/bf16/fp32, CFG, nullable masks, exact bytes, unequal internal batching,
blank normalization, optional multimodal IDs, norm-hook cleanup on success
and failure, dtype refusal, malformed framing/identity and invalid values.
Socket and write denials are proven before imports; the single urllib3 IPv6
capability probe is denied before OS socket creation and explicitly accounted
for. Other network attempts and writes outside owned fixtures fail the check.

## Remaining gates

The helper trusts supplied components/environment. The follow-on
`encoder_identity.py` provides package validation, manifest/bundle request
projection and full executing-worker identity comparison (see
`docs/encoder-identity.md`). The follow-on asset verifier and local-only worker
now integrate these CPU boundaries (`docs/encoder-assets.md` and
`docs/encoder-worker.md`). Production activation, full-project review and
adapter wiring remain separate. Locks are unchanged;
real processor loading still needs the previously identified torchvision
dependency fix. No real checkpoint, numerical checkpoint parity, GPU kernel,
VRAM/RAM fit, performance or image-generation claim is made. The V100 remains
unavailable according to the latest user confirmation.
