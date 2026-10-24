# Runtime readiness and next implementation gates

Verified inside this sandbox on2026-10-13; physical host/card state was not
measured. This checklist keeps software acceptance separate from model fit,
hardware bring-up and usable image generation. It does not change any guard.

## Current foundation

The production-source review and focused tooling corrections are complete;
the main guarded suites, pinned synthetic checks and real-browser workspace
checks are accepted in `code-review.md`. Environments, tools and tests live
on the persistent project mount. The web workspace remains planning/export
only; a trusted supplied-component adapter is CPU-witnessed, but the default
loader, GPU numerics/calibration and web jobs are not yet accepted.
Real processor loading is accepted separately below;
no actual encoder-checkpoint fidelity is claimed. An internal stdlib-only
bounded controller foundation is now tested ([job-controller.md](job-controller.md));
The fixed offline process executor/worker boundary is now tested
([generation-process.md](generation-process.md)), including scoped cancellation
and direct-child reaping. Neither is wired to web generation.
Pure exact-match memory-policy groundwork is also tested
([memory-policy.md](memory-policy.md)); no real probe/calibration or guard
integration is claimed.

## Runtime blockers independent of the card

1. **Vision dependency resolved**: all64 reviewed packages match, including
   torchvision0.29.1+cu126. Native CPU ops and real nine-file local
   Qwen3VLProcessor loading passed offline on3.11/3.13. Processor/upstream/
   codec checks used a tiny synthetic model, not actual encoder weights.
   Artifact/lock/policy decision and checks: [vision-review.md](vision-review.md).
   No CUDA or real encoder fidelity follows from processor acceptance.
2. **CPU memory**: the committed four encoder shards total17,534,339,488B
   (~16.33GiB). A fresh actual preflight observed a visible16GiB cgroup cap;
   the weights alone exceed it even before runtime/load/conversion headroom.
   Host MemAvailable was about28.9GiB at that instant, but does not override
   the container cap or prove a complete applicable budget. Cgroup root
   visibility remains unknown, and hidden ancestors can only tighten it;
   actual loading correctly refuses before model access.
   Next: establish trusted complete host/container budget evidence and
   explicitly provide enough bounded capacity for measured peaks while
   preserving existing services, OR review a separate genuinely smaller
   encoder profile. The current fixed-class full-reference profile cannot
   be relabelled quantized or given an invented memory pass. Merely installing
   the V100, rewriting orchestration in C++, or changing fp16/bf16 storage
   does not remove this raw-weight lower bound. FP32 CPU expansion is larger.

   **2026-10-24 host update**: on this workstation the encoder child runs
   outside any container (cgroup namespace is the initial one, ~122GiB RAM,
   ~72GiB available at test time), so the raw-weight lower bound is no longer
   the constraint — only the *unprovable* visible-root scope is. The opt-in
   `--attest-unbounded-host` flag records the operator's own attestation that
   the process tree has no cgroup memory limit and then applies a freshly read
   `MemAvailable`; the default is still refuse, the attestation supplies no
   number, a tight host still refuses, and the run log records its use.
   Design and limits: [memory-policy.md](memory-policy.md).
3. **Assets complete, model fidelity pending**: the explicitly approved base
   fetch continuation now verifies all24files (33,131,601,593B); settled
   closed-tree preflight passes29entries with no partials. Full DiT/VAE META
   architecture shapes match actual checkpoint headers; tiny random CPU
   sampler/VAE probes pass ([component-cpu-review.md](component-cpu-review.md)).
   These are disk/interface checks, not real-weight fidelity or fit. After
   memory readiness, freshly verify the exact
   supplied settled encoder tree, run the real processor/checkpoint and
   compare extraction/templates/masks and tensors against the pinned
   upstream path; measure CPU dtype accuracy, timing and peak memory. Tiny
   synthetic and miniature generated checkpoints are not substitutes.

## GPU bring-up after physical installation

Last physical-install confirmation was the user's "not installed" report;
current presence is unknown. Recheck actual host/card access before doing
driver/container work. Preserve the user-owned sandbox configuration and
existing AMD/services. Driver choice and hardware/tool support must be
reverified against current primary documentation at bring-up, not assumed
from the old plan's dated version example.

Acceptance requires real CUDA kernel execution with the supported runtime
and V100, not just enumeration/architecture strings; measured available
VRAM, PCIe generation/width and transfer throughput; and BF16-storage→FP16
execution numerics/finiteness. No GPU or quantization fitness claim exists
until these measurements and model-specific tests pass.
An explicit probe implementation now has CPU-only refusal/ABI witnesses
([hardware-probe.md](hardware-probe.md)); its real success path is pending
physical installation. It is not called by planning, CLI generation or web.

## Generation implementation sequence

1. The managed lifecycle and trusted preloaded component bridge are implemented
   ([supplied-adapter.md](supplied-adapter.md)). Full-identity conditioning,
   upstream sampling/VAE/save and refusal paths have tiny CPU witnesses only.
   Next build the gated default local loader; real full-component and accepted
   reference images remain separate gates, not implied by synthetic PNGs.
2. Complete phase/backend/conditioning/runtime/device-aware memory policy
   integration and calibrated measurements. Pure typed exact-match policy
   groundwork exists; it requires whole-device/process-tree peaks and fresh
   complete resource evidence, and does not collect those facts. Existing calibration matching is a
   staleness prefilter only, never authority to grant fit. Record unavailable
   timings as unmeasured, not zero or invented throughput. Benchmark
   resident/streamed and higher-quality quant candidates against an actual
   reference; do not default to an unvalidated low-bit or distilled recipe.
3. Complete the owned bounded generation-job executor/controller: validated immutable
   requests, explicit lifecycle/cancellation/errors, scoped artifacts and
   worker cleanup. Internal controller and fixed process protocol now have
   actual owned-subprocess cleanup witnesses, but default model generation
   remains unavailable. Wire
   the web job API after the real adapter/gates can support it; no fake jobs,
   thumbnails, success, progress percentages or availability claims.
4. Validate the full base recipe at1024² with measured resources and human
   evaluation across portraits/landscapes/illustration/product/text. Turbo
   stays an optional separately reviewed recipe/profile; 2K remains an
   experimental measured opt-in. Keep CLI and browser contracts consistent.

Remote web access is its own bind/auth-policy milestone; current literal
loopback service is not claimed reachable from a host/Tailscale peer.
Hardware, dependency, memory, assets and implementation gaps are independent:
the V100 is **not** the only remaining blocker, and the project is not done.
