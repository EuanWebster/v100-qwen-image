# Supplied Qwen21 phase bridge

`adapters/supplied_qwen21.py` implements the managed five-phase protocol over
explicitly supplied resident components. It is a trusted Python integration
API, **not a loader or a resource-approval mechanism**. No CLI, environment,
worker or web option selects a synthetic provider or a fake model. The default
`qwen21_diffusers` factory remains absent and browser generation stays disabled.

The caller must first approve complete fresh resources, verify assets, load
the exact pinned classes locally, establish placement/dtype and transfer
exclusive lifecycle ownership. Matching a `LocalReferenceBundle`, its paths
and fingerprints proves structural coherence, not checkpoint provenance;
directly constructed bundles are not verification tokens. The adapter loads
no weights, fetches nothing, moves no model and never grants memory fit.

Supported attachment is unquantized resident native SDPA, no retained text
encoder, dispatch/offload hooks, meta parameters or training-mode models.
The explicit execution dtype is fp16, or fp32 for CPU component witnesses;
the latter is not an approved GPU or memory profile. Encoder fp16/bf16/fp32
is independent. AUTO resolves to native SDPA, without claiming its hardware
kernel choice/fitness. Tiled decode is explicit, not an automatic OOM fallback.

The encoder provider follows `ProductionEncoder.encode`. The adapter compares
its completed request with the current manifest/config projection seed, keeps
the child's runtime identity, then strictly decodes its bounded full-identity
payload. Mutable tensor branches are revalidated just before sampling. Hidden
width, masks, CFG and all-False T2I image slots are checked. Only the explicit
conditioning-to-execution bridge converts embeddings; overflow after conversion
refuses sampling. Nullable masks stay nullable and integer-valued.

Sampling uses the pinned upstream scheduler/DiT, one image, the requested
steps/dimensions/CFG and an owned seeded generator. `seed=None` records the
actual random seed. Step callbacks validate finite latents and acknowledge
cancellation by raising, never by returning a partial successful image. Encoder
cancellation is acknowledged only when this job's event was set. Decode uses
the pinned latent unpacking, mean/std normalization and VAE postprocessor;
intermediate/output shapes, device, dtype and finite values are checked.

PNG encoding uses a capped in-memory stream (256MiB); atomic no-clobber
publication reuses the reviewed anchored file publisher. It requires an
existing explicit output directory, invents no default write target, and never
overwrites existing files. This cap is not a memory budget: tensors, Pillow
buffers and encoded-byte copies require later whole-footprint calibration.
Core/controller perform their own fresh saved-artifact verification. `close`
is idempotent and drops owned references, without deleting saved images or
flushing global allocators. This single-use API is serial, not thread-safe.

## CPU acceptance, 2026-10-13

41 ML-free regressions cover metadata/layout/policy rejection before heavy import,
exact component classes, model placement/dtype/hooks, native attention/tiling,
phase/config ownership, cancellation acknowledgement, cleanup and PNG cap.
Run using the persistent test environments through `run_cpu_tests.sh`.

`tools/supplied_adapter_cpu_check.py` runs each pinned ML interpreter with
socket denial and fixture-only writes installed before ML import. Only bounded,
hash-checked configs and the nine real processor files are read, never real
checkpoint tensor bodies. Tiny random DiT/VAE and synthetic full-identity
embeddings exercise CFG/KV on/off, encoder fp16/bf16/fp32 to fp32 execution,
FP16 CPU sampler/VAE/PNG witnesses and an actual tiled decode above the default
256px threshold also run; neither establishes V100 numerics or tile quality.
PNG save and real Pillow reload, seed handling, stale identity and conversion
overflow refusal, nonfinite VAE refusal, no-clobber PNG collision handling,
mutable-mask rejection and cancellation in an actual sampler
callback. The split path's pixels are compared exactly with the unsplit pinned
upstream PIL path using the same tiny components/conditioning/generator.

These are interface/integration witnesses, not full-checkpoint fidelity,
accepted model images, V100 numerics, VRAM fit, throughput or quality. A gated
default loader, real calibration producer, hardware acceptance and web-job
wiring remain separate work. The full CPU encoder still cannot load under the
observed 16GiB cap/incomplete cgroup visibility.
