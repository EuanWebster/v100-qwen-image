# Memory-gated encoder-only asset verification

CPU boundary milestone, 2026-10-13: `src/qwen_engine/encoder_assets.py`.
No ML imports, downloads, writes, loader calls or whole-bundle resolver calls.

`verify_encoder_assets(manifest, models_dir, target_dtype=..., headroom_bytes=...)`
validates the complete reviewed full-reference manifest layout and exact
repo/revision, selects its nine processor plus seven text-encoder artifacts,
and computes checkpoint bytes from the four encoder safetensors shards.
For the committed manifest this is **17,534,339,488 bytes**. Storage dtype is
the reviewed profile's bf16; target dtype must be canonical fp16/bf16/fp32.

Required explicit runtime headroom and actual memory discovery are checked
**before** model-root canonicalization, directory access, shard reads/hashes
or ML imports. The existing fail-closed memory policy is unchanged. There is
no public memory snapshot/provider, implicit headroom, or trusted-root bypass.
Passing the prerequisite would be conservative evidence only, not measured
peak-residency fit. Tests inject their tiny synthetic budget at a private seam;
production callers cannot select that seam through this API.

After the prerequisite passes, the verifier:

- Opens the canonical model root and both component directories using
  no-follow directory descriptors.
- Enumerates only `processor/` and `text_encoder/`, requiring their exact flat
  file sets: no extras, subdirectories, symlinks, FIFOs or special entries.
- Opens each file relative to its component descriptor with no-follow and
  nonblocking flags; checks the opened regular file's size; streams SHA256
  with reads bounded by the claimed size plus one.
- Checks opened/published file identity and timestamps against before/after
  state, then re-enumerates both component trees and checks directory identity.
  Edits to an earlier verified file while a later one is being hashed are
  refused, as are file/directory/root replacements and late stray entries.
- Closes every descriptor on success, refusal or interruption, and returns an
  immutable result with absolute component paths, artifact fingerprint,
  checkpoint bytes and memory report.

Missing VAE/DiT/scheduler/root pipeline files, a partial DiT download, or other
debris **outside** these two directories is deliberately outside the verifier's
scope. The loader must use the explicit component directories, not call a
full-pipeline/root loader or fall back to an ambient Hugging Face cache.

`preflight_encoder_tree(models_dir)` exposes the structural portion only,
without reading contents. Its success is not SHA verification. Directly
constructing a `VerifiedEncoderAssets` dataclass is also not an unforgeable
certificate: obtain it from the verifier. This API verifies bytes at the time
of the call; it cannot prevent subsequent mutations when a model library
reopens paths. Production requires settled code/assets and fresh verification.
Neither success nor the fingerprint establishes numerical fidelity/model fit.

## Reusable checks

```bash
bash run_cpu_tests.sh tests/test_encoder_assets.py
bash run_cpu_tests.sh
PYTHON=/work/.review-envs/test311/bin/python bash run_cpu_tests.sh
.review-envs/py311/bin/python -B tools/encoder_assets_check.py
.review-envs/py313/bin/python -B tools/encoder_assets_check.py
```

Verified 2026-10-13: 39 new guarded tiny-fixture tests, including byte mismatch,
post-preflight FIFO/symlink swaps, same-size edits/replacements/growth while
hashing, late component changes, earlier-file changes, relative-root handling,
descriptor cleanup and prerequisite ordering. Real-interpreter checks verify
the actual memory refusal before any model-directory access/shard read/ML
import: three checks each on Python 3.11.15 and 3.13.5. Zero exit from that tool
means the expected **refusal** was verified, not that the real assets loaded.
Both full guarded regressions passed: **1102 passed, 2 skipped** on Python
3.11.15 and 3.13.5. The real pinned upstream/codec checks also remained
**36/36 on each**; actual identity checks remained **5/5 on each**.

## Remaining worker scope

`encoder_worker.py` now wires the local-only loader/worker CLI, bounded reads,
offline socket guard, exact runtime identity comparison, index shard-reference
allowlisting and atomic no-clobber publication. See `docs/encoder-worker.md`
for its actual miniature-loader evidence and limitations. Production activation
and the complete project review remain separate from this CPU milestone.
Current torchvision and real memory gates remain; no actual checkpoint,
CUDA/V100, numerical parity, performance, fit or generation is validated.
