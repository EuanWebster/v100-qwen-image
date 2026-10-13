# CPU vision and local processor acceptance

Verified2026-10-13 without a GPU. Missing-vision dependency resolved;
full encoder loading, CUDA and generation remain unproved.

## Dependency decision

Added `torchvision==0.29.1+cu126` to the requirements, hashed runtime lock
and immutable package policy. The official
[cu126 index](https://download.pytorch.org/whl/cu126/torchvision/) publishes
CPython3.11/3.13 manylinux2_28 x86_64 wheels. Both metadata documents require
Python>=3.10 except3.14.1, Torch>=2.14.0, numpy and pillow; our exact locked
Torch2.14.1+cu126 satisfies them. Native CPU checks followed; no CUDA proof.

Published wheel SHA256s, present in the runtime lock:

- cp311: `c9e3cb4f609341d418d40a9bbae4280a029f9839d32f123f1c9bcc99aa003843`
- cp313: `7c4b8fac65ba8b7be1388330cdab2ab04c792e1482f287991a95a14d0ac76fdb`

Two constrained uv compiles produce identical entries/hashes (only command
comments differ). All original63 versions/sources/hashes retained, vision
alone added (64). Existing annotations retained; new block from the resolve.
Persistent ML environments synchronized with the setup script. The
pytest-only uv.lock and torch-free environments gain no heavy dependency.

The separate unreviewed vision constant is removed: exact version and
no unexpected direct-source installation now use the unified policy.
Missing/plain-PyPI/wrong-CUDA/source observations refuse. Fresh selected-child
projection still precedes cache hits/misses. Tools no longer approve fake
vision; client refusal fixtures corrupt one observation under real policy.
Miniature worker fixtures retain private memory/loader seams only.

## Actual processor evidence

Committed witness `tools/local_processor_check.py`, on3.11.15 and3.13.5,
denies socket construction and writes outside an owned fixture before imports:

- Nine real processor files (15,884,996B) bounded-read, size/SHA256 verified
  against the manifest, copied to a disposable fixture. No encoder/DiT/VAE/
  scheduler assets accessed.
- Exact imports, native CPU NMS and tensor resize passed.
- Local-only fixed-class Qwen3VLProcessor loaded real Qwen2Tokenizer, image
  and video processors with remote code disabled.
- Real tokenization + upstream raw template/system-token drop/pre-norm
  extraction/T2I masks/codec passed fp16/bf16/fp32, Unicode and whitespace,
  with a **synthetic tiny model**, not actual checkpoint tensors.

The snapshot helper has18 pure tests for scope/bounds/hash, closed flat
inventory, links/special files, mutation and nonempty destinations. This is
a cooperative settled-filesystem diagnostic, not a hostile race sandbox.
Only snapshots are temporary; scripts/tests/envs/caches stay persistent.

```bash
bash scripts/setup_review_envs.sh --check
.review-envs/py311/bin/python -B tools/encoder_identity_check.py
.review-envs/py313/bin/python -B tools/encoder_identity_check.py
.review-envs/py311/bin/python -B tools/local_processor_check.py
.review-envs/py313/bin/python -B tools/local_processor_check.py
```

Full guarded suites1852passed+2skips eachPython. Actual ML checks each:
identity6, synthetic helper36, miniature worker12; cross-version client8 each.
Actual memory refusal still precedes model-root access and ML imports.

Fresh zero-headroom fp16 preflight refuses:17,534,339,488B weights exceed
visible17,179,869,184B (16GiB) cap; remaining availability13,406,183,424B
at that instant. Scope unproven; hostMemAvailable29,993,926,656B cannot
override the cap. Load/runtime peaks need more capacity. No real encoder
weights, model fidelity or GPU tested. Remaining gates:
[runtime-readiness.md](runtime-readiness.md).
