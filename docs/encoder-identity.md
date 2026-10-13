# Production encoder request and runtime identity

CPU-only identity milestone, 2026-10-13. Module:
`src/qwen_engine/encoder_identity.py`; no ML imports, model reads, downloads,
or runtime writes. This prepares the worker boundary; it does not load models
or enable generation.

## Requests and assets

`project_encoder_request(config, manifest, encoder_dtype=...)` derives a
request from the reviewed full-reference manifest. Encoder dtype is required
and independent of DiT quantization. It preserves prompt text exactly and uses
`GenerationConfig.cfg_active`: guidance only when scale exceeds one and
negative text is nonblank. Negative text remains in cache identity even when
CFG is inactive. Resolution, seed, steps, scale value, offload, image count and
runtime memory headroom are not encoding kwargs; kwargs are exactly the CFG
boolean. No image-conditioning input is exposed.

Processor identity fingerprints all nine processor files; tokenizer identity
fingerprints its six vocabulary/tokenizer/config files; chat-template identity
uses the manifest's template hash. Each also carries the model revision. The
implementation identity fingerprints all processor+encoder assets, so every
encoder shard/config/index participates. Unrelated VAE/DiT identity changes do
not fragment the encoder cache.

This projection trusts **manifest claims**, not asset bytes. The companion
`request_from_verified_bundle` checks a supplied bundle's identity, digests,
roles and paths against the manifest without rehashing model files. The caller
must obtain that bundle from the existing resolver; direct dataclass
construction is not proof of verification. A future encoder-only verifier
must independently verify the narrow component tree after memory preflight,
without requiring unrelated DiT/VAE assets.

## Executing code and packages

`observe_installed_packages()` uses stdlib distribution metadata in the current
interpreter. It records absence explicitly, refuses corrupt metadata and does
not confuse a metadata read failure with a missing distribution.
`validate_installed_packages` requires all **63 exact reviewed runtime versions**,
including tokenizers, Jinja, NumPy and CUDA distributions, from the wheel-shipped
`_runtime_pins.py` policy. A guarded test compares its complete name/version and
VCS source set with `requirements.lock`; production does not need a checkout or
a caller-selected lock path. The two VCS packages must also match
their exact git URL/full commit; subdirectory/mixed-source metadata is refused.
Supported Python minor versions are 3.11 and 3.13. Metadata is evidence of
declared source/version, not cryptographic attestation of installed bytes or
proof that imports, CUDA, or model loading will succeed.

`runtime_metadata_digest()` includes the exact Python patch version, all
required-package observations (including explicit absence), and optional installed
distributions, with version and direct-source provenance. Canonical full
`direct_url.json` hashes catch source metadata changes beyond URL/commit. Duplicate
normalized distribution names, invalid metadata and oversized inventories are
refused. Optional presence/version/source changes conservatively invalidate
identity; fingerprinting an optional package is **not approval of its code**.
Our own `qwen-engine` distribution metadata is excluded because its sources are
hashed independently; editable checkout paths do not fragment these keys.
Both runtime and source digests enter the worker implementation revision.
Absolute interpreter paths and distribution enumeration order are excluded.
Same-version installed-byte tampering and environment-variable changes are not
attested by distribution metadata.

There is **no reviewed torchvision pin in the current lock**. Production
identity creation therefore fails closed whether torchvision is absent or
some arbitrary version has been installed. Tests patch a synthetic policy in
memory only to exercise successful construction; no product override or
automatic installation is provided. Real loading requires a separately
reviewed requirements/lock and validator policy update.

`worker_source_digest()` automatically hashes all own `qwen_engine/*.py`
sources recursively, using schema `qwen_engine.worker-source.v1` and sorted
relative names, byte lengths and content hashes. Absolute paths and mtimes
never enter the digest. Rename/content changes invalidate identity; relocation
and timestamp-only changes do not. This conservatively includes web modules,
but not data assets, bytecode caches, docs, tests or external ML sources.
Opened regular-file and byte/count limits, no-follow directory/file opens,
and before/after descriptor identity checks refuse unsafe or changing reads.
Source-less or zip-only deployments are outside this worker profile.
No caller-selected source root/manual revision is accepted by the public API.

`validate_worker_request_identity(request, manifest)` rebuilds the expected
complete request identity using the **executing worker's** packages and own
source, and refuses any mismatch. This is not a parent-runtime observation
substituted for a custom child interpreter: it must be called in the child.
Workers should be fresh processes with settled code/assets; these fingerprints
are not protection against hostile code rewriting an already-running process.

## Verification and pending wiring

```bash
bash run_cpu_tests.sh tests/test_encoder_identity.py
bash run_cpu_tests.sh
PYTHON=/work/.review-envs/test311/bin/python bash run_cpu_tests.sh
.review-envs/py311/bin/python -B tools/encoder_identity_check.py
.review-envs/py313/bin/python -B tools/encoder_identity_check.py
```

The reusable identity check reports success only when actual metadata pins
match, source fingerprints are stable, the pinned manifest projects without
model reads, ML modules stay unimported, and the current production refusal
is observed exactly. Its zero exit does **not** mean production loading works.

Verified 2026-10-13: 80 new guarded unit tests; five real metadata/source checks
on each Python 3.11.15 and 3.13.5 review interpreter. These tests exercise source
relocation/content/name drift, source caps/FIFO/symlink/replacement refusal,
package/provenance drift, worker identity forgery, bundle coherence, cache-key
invariance for generation-only settings and unrelated VAE changes, and cache
invalidation for encoder/tokenizer/template changes. Both full guarded suites
passed: **1063 passed, 2 skipped**, Python 3.11.15 and 3.13.5.

The follow-on `encoder_assets.py` supplies narrow closed-subtree and fresh
byte verification after the actual memory prerequisite (see
`docs/encoder-assets.md`). The local-only loader/worker now integrates these
boundaries (`docs/encoder-worker.md`), with required runtime headroom and memory
checks before ML imports or shard hashing/loading. Full-project review,
including parent/child runtime handoff and the other modules, remains before activation.
Real processor, checkpoint parity, host/GPU fit, CUDA execution
and generation remain unvalidated. The V100 is still unavailable according
to the last user confirmation.

Review correction, 2026-10-13: the original four-package policy could miss
transitive drift. Complete lock validation and runtime identity now close that
gap at request projection/worker validation; generic opaque cache hits still
require a freshly projected executing-runtime request at the production boundary.
Identity tests now total **400**, including all 64 required names/vision-gate
drift combinations, lock-policy consistency, optional inventory/source drift,
Python patch drift, canonical digest anchors and stale worker refusal. Both
guarded full suites: **1480 passed, 2 skipped**, Python 3.11.15/3.13.5.
Real identity checks: **6/6 each**, with all 63 locked packages matching and the
unreviewed torchvision gate still refusing production. Locks were not modified.
