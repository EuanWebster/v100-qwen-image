# Generation runtime identity

`runtime_identity.observe_generation_runtime()` observes the current process
for exact calibration matching. Import is stdlib-only; pinned package metadata
is checked before the lazy Torch observation. It loads no weights, writes
nothing, changes no setting and grants no hardware/resource approval.

The versioned digest includes the reviewed full package inventory/Python patch
identity, the executing package's own source digest, Linux x86_64 kernel/libc/
compiler identity, host boot fingerprint, CPU inventory/microcode/flags,
logical CPU count and process affinity. CPU clocks are deliberately omitted;
boot changes conservatively invalidate reuse. Complete host/container memory
scope is a separate resource gate, not inferred from this host fingerprint.

Torch observations include compiled-build identity, CPU capability, intra/inter
op threads, inherited and operation-specific precision controls, fp16/bf16
reduction/split-K/accumulation flags, math/flash/efficient/cuDNN SDPA switches,
determinism/cuDNN/MKLDNN controls, BLAS/linalg preferences and allocator backend.
Relevant CUDA/BLAS/Torch/thread/tokenizer/CPU-library/diffusers environment
prefixes and explicit Python/allocator/dynamic-library settings are bounded
and hashed. Unrelated environment and credentials are never exported; no raw
environment values or boot identifier appear in the returned record.

CUDA BLAS workspace-size getters touch the CUDA runtime, despite looking like
read-only settings. They are **not called** before Torch's CUDA context is
initialized. CPU snapshots explicitly carry `cuda_context_initialized=False`
and unknown workspace/device binding, not fabricated zero. Initialized CUDA
snapshots include the current logical index and UUID in their settings digest
and a separate hashed binding; `matches_cuda_device` requires both to match
the intended device. Observe inside the intended CUDA device context and
compare against the independently probed index/UUID, not an implicit index0.
A future gated CUDA caller
must collect the CUDA-side settings after independently accepted kernel/device
bring-up and refuse CPU-only identity as complete GPU execution evidence.
Initialized CUDA workspace observation is not accepted yet on real hardware.

Host/source/package/environment/settings are observed twice; drift refuses
the snapshot. This is a cooperative consistency check, not an atomic lock or
protection against hostile concurrent edits. Re-observe at each gated phase.
Unexpected observation failures produce fixed typed errors without exposing
raw settings. KeyboardInterrupt/SystemExit propagate. Metadata is not binary
attestation, a reservation, a peak measurement or a calibration approval.
The caller must use a fresh owned worker with a settled source tree, not a
hot-edited long-lived interpreter whose imported code may predate disk files.
Pipeline-specific policy/device/artifact/conditioning/cache identity remains
separate. Compiled/custom kernels or new runtime settings need new review;
this is not a universal fingerprint of arbitrary Python execution.

## CPU acceptance, 2026-10-13

Pure guarded tests cover digest sensitivity and typed completeness, environment
allowlist/bounds/Unicode, host inventory/affinity/platform, special/link files,
mid-observation drift, thread/precision/allocator/backend settings, unknown
CUDA workspace coherence and interrupts. Fabricated fixtures prove contracts,
not physical host/GPU measurements.

`tools/runtime_identity_cpu_check.py` runs with each pinned ML interpreter,
socket denial and fixture-only writes installed before ML import. It verifies
actual snapshot stability, actual Torch thread/environment drift and restoration,
no CUDA initialization and no workspace query on CPU. No real model assets
are read. No collector persistence, default-loader or web integration is enabled.
