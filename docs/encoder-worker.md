# Local-only CPU encoder worker

CPU worker/loader milestone, 2026-10-13. `qwen-encoder-worker` or
`python -B -m qwen_engine.encoder_worker` executes a fresh child. It is separate
from the image-generation adapter/web service and does not enable generation.
Import/`--help` remain ML-free. All five flags are mandatory for loading:

```text
--request /absolute/request.json
--response /chosen/existing-parent/new-response.json
--manifest /absolute/manifest.json
--models-dir /absolute/model-root
--headroom-bytes EXPLICIT_NONNEGATIVE_RUNTIME_BYTE_COUNT
```

The request is the existing encoder request v2 document with complete
implementation identity. Its request ID is echoed exactly (128-character cap).
No test loader, memory bypass, source-root override or dependency auto-install
is exposed through the CLI. Use the existing `EncoderWorker` subprocess runner
with `{python}`, `{request}` and `{response}` placeholders; run request identity
creation in the selected worker interpreter, not a different parent runtime.
`ProductionEncoder` now supplies fresh-child projection before cache lookup
(`docs/encoder-client.md`). Metadata-only `--project-identity` requires only
request/response/manifest and **refuses** loading flags; it reads no model
directory/asset bytes, imports no ML and makes no memory-fit claim. Loading
still requires explicit models/headroom and all existing gates below.
Never call `execute_worker` in the web/server parent: its socket guard is
irreversible and process-wide.

## Order and boundaries

1. Set offline/telemetry flags, disable tokenizer parallelism and bytecode,
   constrain child CPU/BLAS environment thread settings to one, and install
   a Python socket audit guard before lazy ML imports. No driver or parent
   process configuration is changed. This is not an OS sandbox against hostile
   native code; fixed reviewed library code plus explicit local-only loading
   remain necessary.
2. Read request (256 KiB) and manifest (1 MiB) using no-follow/nonblocking
   opened regular-file descriptors. Strict UTF-8 JSON, duplicate/nonfinite
   refusal, pre-parse depth bound (40), post-parse node bound (20000), size and
   before/after opened/published identity checks apply.
3. Validate the production request and full pinned manifest layout. Require
   actual memory approval before package inspection, model-root access, shard
   hashing or ML imports; compare the full request against current executing
   source/packages/manifest. The narrow verifier rechecks actual memory again
   immediately before fresh verification of its 16 component files.
4. Read the freshly verified encoder index (4 MiB cap), recheck its size/SHA,
   validate positive integral total size, and require exact allowlisted local
   shard names, with all four shards referenced. No traversal, URLs, arbitrary
   filenames or non-safetensors fallback. Recheck the closed component tree
   before importing loader classes.
5. Set torch intra/inter-op threads to one immediately after importing torch;
   load fixed `Qwen3VLProcessor` and `Qwen3VLForConditionalGeneration` classes
   from explicit absolute component directories: `local_files_only=True`,
   `trust_remote_code=False`, `token=False`, `force_download=False`, explicit
   torch dtype and CPU device map, safetensors/weights-only, eager attention
   and no hub kernels. Disable the nested text config's autoregressive cache;
   evaluate under inference mode. Never load DiT/VAE/full pipeline, use a model
   repo ID, move to CUDA or fall back to ambient model caches.
6. Use the accepted supplied-model helper and upstream prompt method/codec.
   Budget the **complete** result v1 JSON/base64 wire size before base64
   expansion (256 MiB default cap), preserving payload metadata/blob bytes.
7. Publish only a new response in the explicitly selected existing parent:
   exclusive same-directory temp, complete writes/fsync, parent identity check,
   atomic hard-link publication that refuses existing destinations, directory
   fsync and temp cleanup. File/directory/symlink/FIFO destinations and races
   never become an overwrite. Linux `/proc/self/fd`, hard-link and directory
   fsync support are required. If a failure occurs after publication (e.g.
   directory fsync), a complete final file can remain; it is not blindly deleted
   or overwritten on retry. Abrupt process death can leave an owned temp;
   the generic parent owns/cleans its private scratch directory.

Exit codes: 0 success; 2 invalid request/arguments; 3 memory refusal;
4 manifest/asset/index refusal; 1 setup/protocol/unexpected failure. Errors go
to stderr, without tracebacks for normal operation; no fake successful error
response is written. The generic parent retains stderr and reaps/cleans the
child using its existing cancellation/timeout lifecycle.

## Pinned loader gotchas verified against actual code

Transformers 5.17.0 ignores `low_cpu_mem_usage` and uses meta initialization
by default. We pass the legacy flag for compatibility, but it is **not**
memory-fit evidence. The real miniature loader test also found that `use_cache`
as a `from_pretrained` kwarg reaches Qwen's model constructor and fails;
setting `model.config.get_text_config().use_cache = False` after loading is
the accepted path. These findings are not inferred from mock loaders.

## Reproduce and limits

```bash
bash run_cpu_tests.sh tests/test_encoder_worker.py
bash run_cpu_tests.sh
PYTHON=/work/.review-envs/test311/bin/python bash run_cpu_tests.sh
.review-envs/py311/bin/python -B tools/synthetic_worker_check.py
.review-envs/py313/bin/python -B tools/synthetic_worker_check.py
```

Verified: 58 new guarded tests; full suites **1160 passed, 2 skipped** each
on Python 3.11.15/3.13.5. Real default memory refusal was exercised through
the generic parent, with captured typed stderr, child reaping and scratch
cleanup. Console `--help` and locked offline editable setup passed in both
lightweight environments; ML dependency locks are unchanged.

Separate pinned-ML checks: **12/12 each**. Six synthetic loader paths execute
real upstream encoding, masks/CFG/dtypes, codec and worker wire publication;
three more execute the **actual fixed-class Transformers loader** with a
freshly generated miniature four-shard checkpoint (under 20000 parameters,
100000 tensor bytes), verify CPU placement/cache/eval/hook cleanup and compare
outputs bit-exactly to the supplied-model helper. The processor remains a tiny
injected tokenizer/processor, not the downloaded processor. Memory and vision
policy are patched at private seams only in the disposable test process.
Socket/write confinement is proven before imports and the one denied urllib3
IPv6 probe is accounted for. Real production policy is restored and its vision
refusal verified at the end.

There is no downloaded reference-checkpoint load, real processor fidelity,
RAM/VRAM fit, GPU kernel or generation acceptance. Actual memory discovery and
missing/unlocked torchvision continue to refuse production. Full-project
review is ongoing. The transitive-runtime/cache identity gap has been corrected:
all 63 locked runtime versions are now validated, and Python patch/optional
distribution/source metadata enter identity (see `docs/encoder-identity.md`).
Parent/child runtime handoff is implemented and tested, including cache hits.
Remaining module review and real runtime acceptance remain before production
activation. Web generation stays disabled.
