# Component interface probes without the card

Verified2026-10-13 on pinned Python3.11.15 and3.13.5. Committed reusable
`tools/component_cpu_check.py` runs with socket/out-of-fixture write guards
installed before ML imports. No real checkpoint tensors loaded, no GPU,
no generation adapter or user-facing image/job enabled.

Actual base assets are now complete: the explicit approved fetch continuation
verified all24 files (33,131,601,593B). The settled closed tree has29 entries
(24files+5directories), no partials. This is disk integrity, not model fit.

The component witness verifies bounded small configs against the manifest;
reads only bounded metadata headers (no mmap or tensor-body read) from the
DiT/VAE checkpoints; creates the **full architectures on META**; confirms
every state tensor is META and exact names/shapes match297DiT/238VAE header
tensors. Header checks bound bytes/count/rank, require nonzero shapes and
consistent contiguous extents, refuse links/special files and observed
mutation. They are metadata checks, not fresh whole-file SHA attestation or
a general safetensors deserializer; full-file integrity came from the fetch.
38pure regressions require neither real files nor ML.

Real local processor files are snapshotted/verified as in vision-review.
The actual scheduler config is used unchanged with a **tiny random** DiT
(one block,H16,two heads), synthetic finite embeddings and64²/two-step CPU
sampling. Eight cases (FP32/FP16 × CFGoff/on × KVoff/on) run pinned upstream
sampler/attention/KV paths, verify two real callbacks, latent shape/device/
dtype/finiteness. No equivalence/quality/speed claim between cases.

A tiny random FP32 VAE (reduced channels, same spatial compression and
real config's latent normalization constants) additionally exercises upstream
decode/postprocess to a finite CPU tensor `[1,4,64,64]`. This is RGBA tensor
interface evidence, **not an accepted image**, and nothing is saved/served.
Real base recipe remains40steps; these two-step random tests are not a new
sampling recipe/default. ModelMixin emits its generic dtype-cast warning
(empty keep-FP32 list) for tiny-model `.to(dtype=...)`; this is not silently
suppressed. Real loading must use the reviewed loader dtype policy and
validate actual BF16-storage→FP16 execution on the installed V100.

```bash
.review-envs/py311/bin/python -B tools/component_cpu_check.py
.review-envs/py313/bin/python -B tools/component_cpu_check.py
```

Remaining: real encoder fidelity/RAMcapacity, adapter lifecycle and
conditioning integration, process-owned job execution, real resource probe/
calibration, offload/quant equivalence and human quality acceptance. The
CPU probes reduce interface uncertainty; they do not close those gates.

Final post-review acceptance:2089passed+2skips each guarded Python3.11/3.13;
controller83, pure memory policy113, processor snapshot18, header38 tests.
Exact real metadata6 and memory refusal3 each; real offline processor6 each;
synthetic helper36/miniature worker12/cross-version client8 each. Real served
Chrome105checks each backend, generation disabled. Earlier logged totals
remain historical. Review fixes additionally own builtin request primitives,
refuse invalid UTF-8 before job creation and separate warm-cache calibration
from cold encoder allocation. Source is committed, reusable tests/envs remain
persistent; only the user-owned sandbox.yaml change is intentionally dirty.
