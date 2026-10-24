# Run preflight contract (`qwen_engine.run_preflight`)

**Status:** scoped milestone. The facade is *not* live integration: root wires
it into the CLI/adapter. It performs no hardware probing and grants nothing on
its own.

## What it is

A thin, ML-free runtime facade over the already-strict memory policy:

1. `load_evidence(path)` re-reads a calibration log through the bounded
   (≤ 16 MiB, regular-file, no-mutation-during-read) bench reader and pushes
   every record through the exactness bridge
   (`calibration.calibration_evidence`). Legacy `calibration_observation.v1`
   records or any record missing an exactness field are a **typed refusal
   naming the missing fields** — never a silently created/partial
   measurement.
2. `assess_run_phase(descriptor, resource_reader, settings, *, now=None)`
   invokes `resource_reader()` **and re-reads the calibration file on every
   call** (no cached permissions across phases), then delegates verbatim to
   `memory.assess_memory` (exact descriptor + exact device identity matching;
   nothing weakened, no interpolation, no expert bypass).
3. `require_run_phase(...)` is the raising form: any refusal raises
   `MemoryPolicyError` (subclass of `MemoryGuardError`, CLI exit-3 family)
   listing the policy reasons.

`PreflightSettings` is frozen and fully explicit: `calibration_path`,
`host_headroom_bytes`, `gpu_headroom_bytes`, `max_resource_age_s`,
`max_calibration_age_s`. There are **no defaults** — a guessed safety budget
is refused at construction (bools, floats-as-ints, negatives, NaN/inf, empty
paths all raise `MemoryPolicyError`).

## Fail-closed rules

* Malformed descriptor/settings/`now`, non-callable reader, or a reader that
  returns anything but `ResourceEvidence` → `MemoryPolicyError` before any
  measurement.
* Reader exceptions (any `Exception`) → `MemoryPolicyError`
  ("resource measurement failed"); they never pass and never escape raw.
* Unreadable/missing/directory/oversized/non-UTF-8/schema-invalid calibration
  files → `MemoryPolicyError`. Unknown JSON keys are schema violations (an
  `expert_override` key cannot smuggle a pass).
* Unknown facts (unverified CUDA kernel, unknown cgroup scope, unknown GPU
  counters, unknown conditioning lengths, missing/stale exact calibration)
  surface as refusal reasons on the decision, and as raises from
  `require_run_phase`.
* The facade writes nothing to disk; it only reads.

## Root integration obligations (NOT provided by the facade)

The facade is a trusted-boundary arbitrator, not a measurement source. Root
must supply, per call:

* **`resource_reader`** — production glue that builds a *fresh*
  `ResourceEvidence` each invocation: current `captured_at` (epoch seconds),
  a real `MemorySnapshot`, the actual `DeviceIdentity` (uuid/architecture/
  driver/cuda build), an honest `cuda_kernel_verified` flag, and real
  device-wide total/available bytes. Exceptions from probing are fine — the
  facade converts them to typed refusals.
* **Valid fresh host scope** — `MemorySnapshot` must carry the *actual*
  cgroup state. If the complete constraint state cannot be proven (the real
  autodiscovery path in `mem_preflight` refuses to claim it), inject
  `cgroup_state="unknown"`: the policy then refuses with
  `complete_host_budget_unknown`. Never inject optimistic finite/unlimited
  values as a guess; injections are *trusted complete evidence*.
* **The exact `PolicyDescriptor`** — root must measure and pass the real
  runtime/artifact identity: pinned `model_revision`, `artifact_set_sha256`,
  `runtime_identity_sha256`, backend/quant/residency/dtypes/flags, and the
  **actual conditioning token lengths** from the tokenizer. The facade never
  infers prompt tokens from character counts, never substitutes another
  phase/config, and unknown lengths refuse with
  `conditioning_lengths_unknown`.
* **Explicit settings** — operator-chosen headroom bytes and freshness
  windows; the calibration path must point at records produced by the
  opt-in calibration collector on the same device profile.

## Limitations

* Not yet wired into CLI/adapter; no phase loop calls it in production.
* Cross-process calibration append races remain a documented `bench`
  limitation (atomic writes keep the file intact).
* A pass is a conservative prerequisite (fresh exact measured peaks +
  explicit headroom within proven budgets), never proof a future allocation
  succeeds.
