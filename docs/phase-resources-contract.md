# Phase GPU Budget Contract (`qwen_engine.phase_resources`)

Status: **standalone ML-free accounting helper**. Nothing is wired into
`run_preflight`/`memory` automatically; integration is Root's decision.
No hardware probes, no `torch` import, no production behaviour change.

Root review additions: applying a budget requires an explicit device UUID
matching `ResourceEvidence.device.uuid`; unknown or another card refuses.
Application rechecks both physical and owned-reserved timestamps, even for a
previously constructed budget. Caller measurements must describe the same
device and a coherent serial allocator state; a fresh timestamp alone cannot
prove ownership or guarantee that later allocations will succeed.

## Why this exists

`memory.assess_memory` compares a measured **WHOLE-process/device GPU peak**
(`CalibrationEvidence.peak_gpu_used_bytes`, which includes our own
allocations) against `ResourceEvidence.gpu_available_bytes`. If the caller
populates that field with only the driver's *physical free* bytes while our
engine already owns Torch CUDA caching-allocator `reserved` bytes, our own
memory is **double-counted**: it sits inside the measured peak *and* is
subtracted from the budget. The policy then refuses work that fits.

## The rule

```
eligible_gpu_budget_bytes = physical_free_bytes + owned_torch_reserved_bytes
```

* `physical_total_bytes` / `physical_free_bytes`: driver-reported whole-device
  counters covering **all** processes.
* `owned_torch_reserved_bytes`: caching-allocator `reserved` bytes belonging
  to **our process only** — the sole add-back term.
* `eligible_gpu_budget_bytes`: the value placed into
  `ResourceEvidence.gpu_available_bytes` as the **WHOLE-PHASE budget** for the
  existing policy.

### What the number is — and is not

`gpu_available_bytes` after `apply_phase_gpu_budget` is the **eligible
whole-phase budget** for the existing policy. It is **NOT a claim of
physical free memory** and must never be reported or logged as such.

It remains **conservative**:

* CUDA context bytes and any other *untracked* owned allocations are **not**
  added back — they appear in neither term, so the eligible budget is a
  lower bound of what the phase could truly reuse.
* Foreign processes' memory is **excluded**: only *known owned* Torch
  reserved bytes are credited; everything else in `total - free` stays
  out of the budget.
* No safety margins, fudge factors, interpolation or averages are applied
  anywhere.

### No host-side add-back

`MemorySnapshot.host_available_bytes` derives from `MemAvailable`, which
**already reflects reclaimable file cache**. Adding host RSS back would
double-count. `apply_phase_gpu_budget` therefore preserves the host
`MemorySnapshot` object unchanged (identity-preserved), along with device
identity, `cuda_kernel_verified`, `captured_at` and `gpu_total_bytes`;
exactly one field (`gpu_available_bytes`) changes.

## Strict validation (refuse, never clip)

All byte counters must be **exact nonnegative `int`s** — `bool` is refused
despite being an `int` subclass; floats and negatives are refused. Exact
zeros are meaningful measurements (a zero budget), never "unknown"; unknown
is represented by refusal upstream, never by a guessed zero here.

Inconsistent snapshots are **rejected**, never reconciled or clipped:

* `physical_free_bytes <= physical_total_bytes`;
* `owned_torch_reserved_bytes <= physical_total_bytes - physical_free_bytes`
  (our reserved bytes must fit inside the observed used bytes — a violation
  means the components did not come from one coherent device state);
* `eligible_gpu_budget_bytes == physical_free_bytes + owned_torch_reserved_bytes`
  exactly (checked again in `PhaseGpuBudget.__post_init__`, so a handcrafted
  result with a smuggled margin cannot exist).

## Freshness

Every component observation carries an explicit `captured_at`:

* `compute_phase_gpu_budget` freshness-checks **both** the physical snapshot
  and the owned-reserved reading against an **explicitly supplied**
  `now`/`max_age_s`; stale or future-dated components refuse.
* `apply_phase_gpu_budget` requires the budget's physical `captured_at` and
  the evidence's `captured_at` to agree within an **explicitly supplied**
  `max_update_window_s` (budget capture time may not precede the evidence
  capture time), and both to be fresh against `now`.
* There is **no global caching and no default freshness window** — every
  window is a required keyword argument.
* No optimistic unknown ownership: missing GPU counters, non-matching
  `gpu_total_bytes`, or a `physical_free_bytes` that differs from the
  evidence's existing raw `gpu_available_bytes` all refuse. The free-match
  check is the genuine no-double-count guard: the add-back can apply to a
  given raw-free snapshot exactly once.

## API

* `PhaseGpuBudget` — frozen dataclass: physical total/free, owned Torch
  reserved, eligible budget, per-component `captured_at` timestamps, source
  notes, notes tuple, schema `qwen_engine.phase_gpu_budget.v1`.
* `compute_phase_gpu_budget(*, physical_total_bytes, physical_free_bytes,
  physical_captured_at, physical_source, owned_torch_reserved_bytes,
  owned_reserved_captured_at, owned_reserved_source=..., now, max_age_s)`
  → validated `PhaseGpuBudget`.
* `apply_phase_gpu_budget(resources, budget, *, now, max_update_window_s)`
  → new `ResourceEvidence` via `dataclasses.replace` **only** when raw
  free, total and measurement time all match within the explicit window.
* `PhaseBudgetError` — subclass of `MemoryGuardError` (memory-refusal,
  exit-3 family). A refusal is never a permission to load.

## Limitations / non-claims

* This helper performs **no measurement**; it only re-arranges caller-
  supplied numbers under strict consistency rules. Truthfulness of the
  inputs is the caller's responsibility, as elsewhere in this codebase.
* `owned_torch_reserved_bytes` covers only the Torch caching allocator of
  one process; child-process or non-Torch owned allocations are not credited
  (conservative under-credit, never over-credit).
* Passing an eligible budget through the existing policy is still a
  conservative prerequisite, **not** proof that a future allocation
  succeeds.
* Production wiring, GPU probes and calibration remain entirely outside
  this module's scope.
