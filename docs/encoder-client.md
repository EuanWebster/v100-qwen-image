# Selected-child production encoder client

CPU-only boundary milestone, 2026-10-13. `encoder_client.ProductionEncoder`
projects runtime identity in a fresh **selected interpreter before every cache
lookup**. Parent metadata is never substituted for child facts. This closes
the production cache-hit gap left by the generic opaque runner; generation
and web activation remain disabled.

```python
from qwen_engine.encoder_client import ProductionEncoder
from qwen_engine.config import GenerationConfig

client = ProductionEncoder(
    manifest_path="/chosen/manifest.json",
    models_dir="/chosen/models",
    scratch_dir="/chosen/existing-scratch",
    python="/chosen/locked-environment/bin/python",
    headroom_bytes=EXPLICIT_MEASURED_RUNTIME_HEADROOM,
    timeout_s=EXPLICIT_LOADING_TIMEOUT,
    cache=None,  # or a deliberately configured EmbeddingCache
)
result = client.encode(GenerationConfig(prompt="..."), encoder_dtype="fp16")
# Tensor consumer: payload_to_branches(result.payload,
#                                    expect_request=result.request).
```

This is an API illustration, **not a working production recipe**: actual
vision/memory/hardware gates remain unresolved. Headroom is not guessed.
Runtime resource paths come from client construction, not GenerationConfig's
filesystem fields. Paths/interpreter are bound against later CWD changes.
Choosing the model root is lexical; projection does not access its directory.

## Sequence and contracts

1. Read the chosen manifest through the bounded snapshot reader and build
   `project_encoder_seed`: prompt/negative/dtype/CFG and declared model/
   processor/tokenizer/template identities, **without runtime identity**.
   A seed is not production and must never be cached/encoded as one.
2. Run the fixed worker module with `--project-identity`. This profile requires
   request/response/manifest paths and refuses models/headroom loading flags.
   Fresh-child offline guard, strict256KiB request/1MiB manifest inputs,
   reviewed package policy and source fingerprinting remain enforced.
   `complete_encoder_seed` checks manifest coherence before runtime observation.
   No ML imports, model-root access, asset-byte verification or memory-fit claim.
3. Return a metadata-only existing result-v1 payload: no blobs; metadata exactly
   `{schema: qwen_engine.encoder_projection.v1, request: completed_request_v2}`.
   Full wire response capped512KiB; atomic no-clobber publisher reused.
4. Parent validates request ID, production policy, complete seed equivalence
   and declared encoder-asset fingerprint against its chosen manifest. Both
   seed and completed loading request must fit the actual256KiB wire budget.
   Projection refusal/malformed responses touch **no cache**, including no
   key lock, and never launch loading.
5. Under the existing root-shared key lock, a hit undergoes stdlib-only
   serialized-payload identity/codec/header/descriptor checks. A miss invokes
   the separate existing loading profile with all actual memory/runtime/asset
   gates, then performs those checks **before cache publication**.
   Generic byte corruption keeps existing quarantine behavior; invalid
   semantic structures are typed refusals, not silent overwrites/fallback.

`ProductionEncoding` returns the opaque payload with the exact fresh request.
`validate_serialized_payload` shares the pre-deserialization path with
`payload_to_branches`, but **does not validate tensor finiteness/mask values**.
Consumers must decode with that expected request and finish actual tensor
validation before use. Parent/cache code stays ML-free. Uncached calls also
project freshly. No command/revision/root/policy bypass is exposed; generic
fake/opaque `EncoderWorker` behavior and schemas are unchanged.

Fingerprinting assumes settled environments; metadata/source snapshots do not
attest hostile installed-byte rewriting or dependency changes mid-job. Own
`qwen-engine` distribution metadata is excluded: editable builds legitimately
expose dist-info plus checkout egg-info, while executing source is independently
hashed. Other normalized-name duplicates still refuse.

## Reproduce and evidence

```bash
bash run_cpu_tests.sh
PYTHON=/work/.review-envs/test311/bin/python bash run_cpu_tests.sh
.review-envs/py311/bin/python -B tools/synthetic_encoder_client_check.py \
  --child-python /work/.review-envs/py313/bin/python
.review-envs/py313/bin/python -B tools/synthetic_encoder_client_check.py \
  --child-python /work/.review-envs/py311/bin/python
.review-envs/py311/bin/python -B tools/synthetic_worker_check.py
.review-envs/py313/bin/python -B tools/synthetic_worker_check.py
.review-envs/py311/bin/python -B tools/synthetic_ml_encoder_check.py
.review-envs/py313/bin/python -B tools/synthetic_ml_encoder_check.py
```

49new guarded regressions; both full suites **1642passed, 2skipped**,
Python3.11.15/3.13.5. Focused client33, identity410, worker69 and codec94+1skip.
Actual cross-version selected-child checks **8/8 each direction**: real child
Python/all63locked-package/source observations; fresh projection on cache hit;
CFG/dtype/wire/header structure; no parent observation or ML/model-root access;
real policy refusal before a previously valid cache hit; reaping/scratch cleanup.
Private bootstrap monkeypatches ONLY in the disposable test harness provide
synthetic vision approval and tiny hand-built opaque payloads, **not real
encoder/model/memory-fit acceptance**. Helpers in tools are not product
entrypoints/installed modules; no public runtime bypass was added. Test source/
setup stays in git/.review-envs; generated disposable fixtures alone are temporary.

Also rerun actual pinned identity6/6, supplied-model/helper36/36 and miniature
fixed-class worker12/12 per interpreter (see their docs for limits). No real
checkpoint/processor download/load, V100/CUDA/VRAM/parity or generation claim.
Full-project review continues through remaining modules.
