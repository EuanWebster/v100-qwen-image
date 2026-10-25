"""ML-free conservative whole-phase GPU budget accounting.

Problem this helper solves: ``memory.assess_memory`` compares a measured
WHOLE-process/device GPU peak (``CalibrationEvidence.peak_gpu_used_bytes``)
against ``ResourceEvidence.gpu_available_bytes``. If the caller passes only
the *physical free* bytes reported by the driver while our own engine
already owns Torch CUDA ``reserved`` bytes, our own already-owned memory is
double-counted: it is inside the measured peak AND subtracted from the
budget. This module computes the correct **eligible whole-phase budget**::

    eligible_gpu_budget_bytes = physical_free_bytes + known_owned_reserved_bytes

which stays **conservative**: CUDA context bytes and any other untracked
owned allocations are NOT added back (they sit in neither term), and foreign
processes' memory stays excluded because only *known owned* Torch reserved
bytes are credited.

Explicit non-goals / honesty:

* No hardware probes, no ``torch`` import, no production wiring. Root owns
  integration into ``run_preflight``/``memory``; nothing here is wired.
* ``eligible_gpu_budget_bytes`` replaces ``ResourceEvidence.gpu_available_bytes``
  as the WHOLE-PHASE budget consumed by the existing policy — it is NOT a
  claim of physical free memory and must never be reported as such.
* No host-side add-back: ``MemorySnapshot.host_available_bytes`` derives from
  ``MemAvailable``, which already reflects reclaimable file cache, so host
  evidence is left untouched (see contract doc).
* No safety margins, no caching, no default freshness window: every
  ``captured_at`` is validated against an explicitly supplied ``now`` and
  ``max_age_s``, and every update requires an explicitly supplied exact
  window. Unknown ownership is never optimistic — it refuses.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any

from qwen_engine.errors import MemoryGuardError
from qwen_engine.memory import ResourceEvidence


class PhaseBudgetError(MemoryGuardError):
    """Malformed phase-budget inputs or a conservative refusal to compute/
    apply an eligible budget. Typed refusal in the memory-refusal family;
    never a permission to load or a relaxed bound."""


#: Schema tag for the accounting result document (events/JSON later).
PHASE_GPU_BUDGET_SCHEMA = "qwen_engine.phase_gpu_budget.v1"

#: Torch CUDA caching-allocator ``reserved`` bytes are the only ownership
#: counter this helper credits back; the name is recorded verbatim in notes.
OWNED_RESERVED_SOURCE = "torch.cuda.caching_allocator_reserved_bytes"

_LABEL_RE = re.compile(r"[!-~]{1,128}")


def _nonneg_int(value: Any) -> bool:
    # Strict: exact int, bool excluded (bool is an int subclass), nonnegative.
    return type(value) is int and value >= 0


def _timestamp(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except (OverflowError, TypeError):
        return False


def _label(value: Any) -> bool:
    return type(value) is str and _LABEL_RE.fullmatch(value) is not None


def _check_freshness(captured_at: float, *, now: float, max_age_s: float, name: str) -> None:
    """Reject stale or future component observations. No default window:
    ``max_age_s`` must be supplied by the caller; ``captured_at > now`` is a
    future-dated observation and refuses rather than being treated as fresh."""
    if not 0 <= now - captured_at <= max_age_s:
        raise PhaseBudgetError(
            f"{name}: captured_at is stale or future-dated relative to the "
            f"explicitly supplied now/max_age_s window; refusing to treat it "
            f"as fresh (no default freshness window exists)")


@dataclass(frozen=True)
class PhaseGpuBudget:
    """Conservative eligible whole-phase GPU budget from one coherent
    physical/ownership snapshot set.

    Counters (all exact nonnegative ints):

    * ``physical_total_bytes`` / ``physical_free_bytes``: driver-reported
      whole-device total/free, including ALL processes.
    * ``owned_torch_reserved_bytes``: bytes the caching allocator has
      reserved for OUR process only — the sole add-back term.
    * ``eligible_gpu_budget_bytes``: ``physical_free + owned_torch_reserved``,
      the value intended for ``ResourceEvidence.gpu_available_bytes`` as the
      WHOLE-PHASE budget for the existing policy. NOT physical free.

    Conservatism: CUDA context and any untracked owned bytes are not added
    back, so the eligible budget is a lower bound on what our phase could
    truly reclaim-and-use; foreign-process bytes are never credited.
    """

    physical_total_bytes: int
    physical_free_bytes: int
    owned_torch_reserved_bytes: int
    eligible_gpu_budget_bytes: int
    #: Timestamps of every component observation actually used (freshness
    #: was validated for each at construction; nothing is cached or reused).
    physical_captured_at: float
    owned_reserved_captured_at: float
    #: Verbatim provenance labels for each component observation.
    physical_source: str
    owned_reserved_source: str
    device_uuid: str | None = None
    notes: tuple[str, ...] = ()
    schema: str = PHASE_GPU_BUDGET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PHASE_GPU_BUDGET_SCHEMA:
            raise PhaseBudgetError(f"unknown phase GPU budget schema {self.schema!r}")
        if not all(_nonneg_int(v) for v in (
                self.physical_total_bytes, self.physical_free_bytes,
                self.owned_torch_reserved_bytes, self.eligible_gpu_budget_bytes)):
            raise PhaseBudgetError(
                "all GPU byte counters must be exact nonnegative ints (bools "
                "and floats refused); zero is a meaningful exact zero, not unknown")
        if not (_timestamp(self.physical_captured_at)
                and _timestamp(self.owned_reserved_captured_at)):
            raise PhaseBudgetError(
                "every component observation needs an explicit nonnegative "
                "finite captured_at timestamp; unknown capture time is never assumed fresh")
        if not (_label(self.physical_source) and _label(self.owned_reserved_source)):
            raise PhaseBudgetError("component source notes must be nonempty printable ASCII labels")
        if self.device_uuid is not None and not _label(self.device_uuid):
            raise PhaseBudgetError("device_uuid must be an explicit device identifier or unknown")
        if not (isinstance(self.notes, (tuple, list))
                and all(type(n) is str and n for n in self.notes)):
            raise PhaseBudgetError("notes must be a tuple/list of nonempty strings")
        object.__setattr__(self, "notes", tuple(self.notes))
        # Consistency: reject inconsistent snapshots instead of clipping.
        if self.physical_free_bytes > self.physical_total_bytes:
            raise PhaseBudgetError(
                "inconsistent snapshot: physical_free_bytes exceeds "
                "physical_total_bytes; refusing to clip or guess")
        if self.owned_torch_reserved_bytes > self.physical_total_bytes - self.physical_free_bytes:
            raise PhaseBudgetError(
                "inconsistent snapshot: owned_torch_reserved_bytes exceeds "
                "physical_total_bytes - physical_free_bytes (our reserved "
                "bytes must be part of the used bytes); refusing to clip")
        if self.eligible_gpu_budget_bytes != self.physical_free_bytes + self.owned_torch_reserved_bytes:
            raise PhaseBudgetError(
                "eligible_gpu_budget_bytes must equal physical_free_bytes + "
                "owned_torch_reserved_bytes exactly; no margins, no clipping")


def compute_phase_gpu_budget(
    *,
    physical_total_bytes: int,
    physical_free_bytes: int,
    physical_captured_at: float,
    physical_source: str,
    owned_torch_reserved_bytes: int,
    owned_reserved_captured_at: float,
    owned_reserved_source: str = OWNED_RESERVED_SOURCE,
    device_uuid: str | None = None,
    now: float,
    max_age_s: float,
) -> PhaseGpuBudget:
    """Validate one coherent physical/ownership snapshot set and return the
    conservative eligible whole-phase budget.

    Strict rules (all refusals, never clipping or guessing):

    * byte counters: exact nonnegative ``int`` (``bool`` refused);
    * ``physical_free_bytes <= physical_total_bytes``;
    * ``owned_torch_reserved_bytes <= physical_total_bytes - physical_free_bytes``
      (our reserved bytes must fit inside the observed used bytes — a
      violation means the components come from different/incoherent
      snapshots, so it is refused, not reconciled);
    * freshness: BOTH component observations are checked against the
      explicitly supplied ``now``/``max_age_s``; there is no default window
      and no caching — stale or future-dated components refuse.
    """
    if not all(_nonneg_int(v) for v in (
            physical_total_bytes, physical_free_bytes, owned_torch_reserved_bytes)):
        raise PhaseBudgetError(
            "physical_total_bytes/physical_free_bytes/owned_torch_reserved_bytes "
            "must be exact nonnegative ints (bools/floats/negatives refused); "
            "unknown ownership must be represented by refusal upstream, "
            "never by a guessed zero here")
    if not (_timestamp(physical_captured_at) and _timestamp(owned_reserved_captured_at)):
        raise PhaseBudgetError("captured_at values must be nonnegative finite numbers")
    if not (_label(physical_source) and _label(owned_reserved_source)):
        raise PhaseBudgetError("source notes must be nonempty printable ASCII labels")
    if not _timestamp(now) or not _timestamp(max_age_s):
        raise PhaseBudgetError(
            "now and max_age_s must be explicit nonnegative finite numbers; "
            "there is no default freshness window")
    _check_freshness(physical_captured_at, now=now, max_age_s=max_age_s, name="physical snapshot")
    _check_freshness(owned_reserved_captured_at, now=now, max_age_s=max_age_s, name="owned reserved reading")
    if physical_free_bytes > physical_total_bytes:
        raise PhaseBudgetError(
            "inconsistent snapshot: physical_free_bytes exceeds physical_total_bytes")
    if owned_torch_reserved_bytes > physical_total_bytes - physical_free_bytes:
        raise PhaseBudgetError(
            "inconsistent snapshot: owned_torch_reserved_bytes exceeds the "
            "observed used bytes (total - free); components cannot be from "
            "one coherent device state")
    return PhaseGpuBudget(
        physical_total_bytes=physical_total_bytes,
        physical_free_bytes=physical_free_bytes,
        owned_torch_reserved_bytes=owned_torch_reserved_bytes,
        eligible_gpu_budget_bytes=physical_free_bytes + owned_torch_reserved_bytes,
        physical_captured_at=physical_captured_at,
        owned_reserved_captured_at=owned_reserved_captured_at,
        physical_source=physical_source,
        owned_reserved_source=owned_reserved_source,
        device_uuid=device_uuid,
        notes=(
            "eligible_gpu_budget_bytes = physical_free_bytes + "
            "owned_torch_reserved_bytes; it is the conservative WHOLE-PHASE "
            "budget for the existing policy, NOT claimed physical free",
            "CUDA context and untracked owned bytes are NOT added back "
            "(conservative lower bound); foreign-process bytes are excluded",
            f"physical_source={physical_source}; owned_reserved_source={owned_reserved_source}",
        ),
    )


def apply_phase_gpu_budget(
    resources: ResourceEvidence,
    budget: PhaseGpuBudget,
    *,
    now: float,
    max_update_window_s: float,
) -> ResourceEvidence:
    """Return a NEW ``ResourceEvidence`` whose ``gpu_available_bytes`` is the
    eligible WHOLE-PHASE budget, via ``dataclasses.replace`` only.

    The update is applied ONLY when the budget provably describes the SAME
    device state the evidence already recorded:

    * ``budget.physical_total_bytes == resources.gpu_total_bytes``;
    * ``budget.physical_free_bytes`` == the evidence's existing raw free bytes
      (the value being replaced — this is the genuine no-double-count check:
      the add-back happens exactly once);
    * ``resources.captured_at`` and ``budget.physical_captured_at`` are the
      SAME observation time within the explicitly supplied
      ``max_update_window_s`` (both are also freshness-checked against
      ``now``; no default window, no caching).

    Device identity, CUDA verification and the host ``MemorySnapshot`` are
    preserved unchanged — no host RSS add-back anywhere (``MemAvailable``
    already reflects reclaimable file cache). Any mismatch, missing counter
    or stale/future component refuses with ``PhaseBudgetError``; nothing is
    clipped, merged or guessed.
    """
    if type(resources) is not ResourceEvidence:
        raise PhaseBudgetError(
            "resources must be an exact ResourceEvidence instance")
    if type(budget) is not PhaseGpuBudget:
        raise PhaseBudgetError(
            "budget must be an exact PhaseGpuBudget from compute_phase_gpu_budget")
    if not _timestamp(now) or not _timestamp(max_update_window_s):
        raise PhaseBudgetError(
            "now and max_update_window_s must be explicit nonnegative finite "
            "numbers; there is no default update window")
    if resources.gpu_total_bytes is None or resources.gpu_available_bytes is None:
        raise PhaseBudgetError(
            "existing evidence lacks GPU counters: the raw free/total being "
            "replaced is unknown, so an eligible budget cannot be verified "
            "against it (unknown is never assumed to match)")
    _check_freshness(resources.captured_at, now=now, max_age_s=max_update_window_s,
                     name="resource evidence")
    _check_freshness(budget.physical_captured_at, now=now, max_age_s=max_update_window_s,
                     name="budget physical snapshot")
    _check_freshness(budget.owned_reserved_captured_at, now=now, max_age_s=max_update_window_s,
                     name="budget owned reserved reading")
    if (resources.device is None or budget.device_uuid is None
            or budget.device_uuid != resources.device.uuid):
        raise PhaseBudgetError("budget device UUID is unknown or differs from resource evidence")
    if not 0 <= budget.physical_captured_at - resources.captured_at <= max_update_window_s:
        raise PhaseBudgetError(
            "budget physical snapshot and resource evidence measurement times "
            "differ beyond the explicit window: they may not describe the "
            "same device state; refusing instead of reconciling")
    if budget.physical_total_bytes != resources.gpu_total_bytes:
        raise PhaseBudgetError(
            "budget physical_total_bytes does not match the evidence's "
            "gpu_total_bytes: different device or snapshot; refusing")
    if budget.physical_free_bytes != resources.gpu_available_bytes:
        raise PhaseBudgetError(
            "budget physical_free_bytes does not match the evidence's existing "
            "raw gpu_available_bytes: the owned-reserved add-back would apply "
            "to a different snapshot (double-count risk); refusing")
    return replace(
        resources,
        gpu_available_bytes=budget.eligible_gpu_budget_bytes,
    )
