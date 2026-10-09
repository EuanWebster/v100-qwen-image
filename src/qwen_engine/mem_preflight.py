"""Host-memory preflight for the forthcoming CPU encoder child (M2.2b).

Stdlib-only. **This module imports no ML package and reads no model file**:
it answers, before any ML import, weight read or loader call, the single
question "can the numbers presented possibly fit the applicable memory
budget?" — and refuses otherwise.

What it reads (documented Linux cgroup v2 target only — deliberately NOT a
general resource framework):

* **Host available memory**: ``MemAvailable`` from ``/proc/meminfo``.
* **This process's cgroup v2 memory cap and current usage**: the ``0::``
  line of ``/proc/self/cgroup`` locates the cgroup; the deepest existing
  ``memory.max`` on the walk to the cgroup root is the cap (an explicit
  ``max`` means *unlimited*), with ``memory.current`` from the same
  directory as the usage counter.

Budget rule: the **tighter applicable budget** — with a finite cgroup cap
the cgroup availability (``max - current``) always bounds the decision even
when host RAM looks plentiful; the cap is never substituted away by a
larger host number. An **unknown or unreadable constrained state must not
assert safety**: anything that cannot rule out a tighter constraint (no
``0::`` line, unreadable files, unparseable or negative counters, a cap
without its usage counter, missing/unparseable ``MemAvailable`` with no
finite cap) yields a refusal, never a pass.

Required bytes are explicit and conservative: the encoder checkpoint byte
total, an explicit dtype expansion (e.g. fp32 parameters from a two-byte
checkpoint dtype doubles the weight bytes; an apparent *shrink* is clamped
to the checkpoint total — loading does not get cheaper than the bytes on
disk), plus explicit runtime headroom (there is **no implicit headroom
default**; the caller states it).

Honest limits — read before trusting a ``proceed``: this is a
**conservative prerequisite, not measured model-fit evidence**. Passing it
alone never proves a load fits: allocator behaviour, conversion transients,
page cache pressure, fragmentation and actual peak residency are not
modelled here; only real measurement under the real loader can demonstrate
fit. The motivating measurement (2026-10-09): the pinned text-encoder
checkpoint totals 17,534,339,488 B while this sandbox's cgroup
``memory.max`` is 17,179,869,184 B (16 GiB) — the weights alone exceed the
cap before expansion or runtime allocations, and this preflight must say so
before anything expensive or heavy is attempted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from qwen_engine.errors import MemoryGuardError

#: Schema tag of the report document (for events/JSON later).
MEM_PREFLIGHT_SCHEMA = "qwen_engine.mem_preflight.v1"

#: Bytes per scalar for the dtype names this preflight models (lowercase).
#: Anything else is refused as an unknown dtype rather than guessed.
DTYPE_BYTES = {"f32": 4, "fp32": 4, "float32": 4,
               "bf16": 2, "bfloat16": 2,
               "f16": 2, "fp16": 2, "float16": 2}

#: cgroup constraint states.
CGROUP_FINITE = "finite"
CGROUP_UNLIMITED = "unlimited"
CGROUP_UNKNOWN = "unknown"


class MemoryPreflightError(MemoryGuardError):
    """The requested load is not provably within the applicable memory
    budget (or the constrained state is unknown). Typed refusal in the
    memory-refusal (exit 3) family; raised BEFORE any ML import, weight
    read or loader call."""


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True)
class MemorySnapshot:
    """One read of the machine's memory state. ``None`` means **unknown** —
    never assumed plentiful. ``cgroup_state`` is one of
    ``"finite"``/``"unlimited"``/``"unknown"``; a finite cap without its
    usage counter is unsafe, so ``cgroup_current_bytes`` is always present
    (non-``None``) when the state is finite."""

    host_available_bytes: int | None
    cgroup_state: str
    cgroup_limit_bytes: int | None = None
    cgroup_current_bytes: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cgroup_state not in (CGROUP_FINITE, CGROUP_UNLIMITED,
                                     CGROUP_UNKNOWN):
            raise MemoryPreflightError(
                f"cgroup_state: must be one of finite/unlimited/unknown, "
                f"got {self.cgroup_state!r}")
        if self.cgroup_state == CGROUP_FINITE and not _positive_int(
                self.cgroup_limit_bytes):
            raise MemoryPreflightError(
                "a finite cgroup cap requires a positive integer "
                f"cgroup_limit_bytes, got {self.cgroup_limit_bytes!r}")
        if self.cgroup_state == CGROUP_UNKNOWN and (
                self.cgroup_limit_bytes is not None
                or self.cgroup_current_bytes is not None):
            raise MemoryPreflightError(
                "an unknown cgroup state carries no counters")

    def budget(self) -> tuple[int | None, tuple[str, ...]]:
        """The tighter applicable budget, or ``None`` when no budget can be
        proven; plus human-readable reasons (refusals first)."""
        reasons: list[str] = list(self.notes)
        if self.cgroup_state == CGROUP_UNKNOWN:
            return None, tuple(reasons) + (
                "cgroup memory constraint state unknown: a finite cap cannot "
                "be ruled out, so safety cannot be asserted",)
        if self.cgroup_state == CGROUP_FINITE:
            assert self.cgroup_limit_bytes is not None  # __post_init__
            current = self.cgroup_current_bytes or 0
            cgroup_available = max(0, self.cgroup_limit_bytes - current)
            if self.host_available_bytes is None:
                reasons.append("host MemAvailable unknown, but the finite "
                               "cgroup budget alone bounds the process")
                return cgroup_available, tuple(reasons)
            budget = min(self.host_available_bytes, cgroup_available)
            if budget == cgroup_available \
                    and self.host_available_bytes > cgroup_available:
                reasons.append(
                    f"finite cgroup availability {cgroup_available} B is "
                    f"tighter than host MemAvailable "
                    f"{self.host_available_bytes} B: the cap is never "
                    f"substituted away by the host number")
            return budget, tuple(reasons)
        # unlimited: only the host number can decide
        if self.host_available_bytes is None:
            return None, tuple(reasons) + (
                "cgroup imposes no finite cap and host MemAvailable is "
                "unknown: no budget can be proven",)
        return self.host_available_bytes, tuple(reasons)


# ---------------------------------------------------------------------------
# Reading real state (paths injectable for tests)
# ---------------------------------------------------------------------------


def read_host_available(meminfo_path: str | Path = "/proc/meminfo"
                        ) -> tuple[int | None, tuple[str, ...]]:
    try:
        text = Path(meminfo_path).read_text(encoding="utf-8",
                                            errors="replace")
    except OSError as exc:
        return None, (f"meminfo unreadable: {exc}",)
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024, ()
            return None, (f"MemAvailable unparseable: {line!r}",)
    return None, ("meminfo has no MemAvailable line",)


def read_cgroup_budget(proc_cgroup_path: str | Path = "/proc/self/cgroup",
                       cgroup_fs_root: str | Path = "/sys/fs/cgroup",
                       ) -> tuple[str, int | None, int | None,
                                  tuple[str, ...]]:
    """Locate this process's cgroup v2 memory cap and usage counter.

    Deliberately conservative: anything short of an explicitly parsed
    ``memory.max`` (integer → finite, ``max`` → unlimited) with a parsed
    ``memory.current`` (when finite) is ``unknown``. v1 hierarchies are not
    this milestone's target: without a ``0::`` line the constrained state
    cannot be ruled out, so it is ``unknown``, not ``unlimited``.
    """
    try:
        lines = Path(proc_cgroup_path).read_text(encoding="utf-8",
                                                 errors="replace").splitlines()
    except OSError as exc:
        return CGROUP_UNKNOWN, None, None, (f"proc cgroup unreadable: {exc}",)
    cpath = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            cpath = parts[2]
    if cpath is None:
        return CGROUP_UNKNOWN, None, None, (
            "no cgroup v2 (0::) line in the proc cgroup file: a memory cap "
            "cannot be ruled out",)
    root = Path(cgroup_fs_root)
    comps = [c for c in cpath.split("/") if c]
    while True:
        directory = root.joinpath(*comps)
        max_file = directory / "memory.max"
        try:
            raw = max_file.read_text(encoding="utf-8",
                                     errors="replace").strip()
        except FileNotFoundError:
            if not comps:
                return CGROUP_UNKNOWN, None, None, (
                    f"no memory.max from {cpath!r} up to the cgroup root: "
                    f"the memory controller state is unknown",)
            comps = comps[:-1]  # walk toward the root
            continue
        except OSError as exc:
            return CGROUP_UNKNOWN, None, None, (f"memory.max unreadable: {exc}",)
        if raw == "max":
            return CGROUP_UNLIMITED, None, None, ()
        try:
            limit = int(raw)
        except ValueError:
            return CGROUP_UNKNOWN, None, None, (
                f"memory.max not an integer or 'max': {raw!r}",)
        if limit < 0:
            return CGROUP_UNKNOWN, None, None, (
                f"memory.max negative: {raw!r}",)
        try:
            cur_raw = (directory / "memory.current").read_text(
                encoding="utf-8", errors="replace").strip()
            current = int(cur_raw)
        except FileNotFoundError:
            return CGROUP_UNKNOWN, None, None, (
                "memory.max present but memory.current missing: usage "
                "unknown against a finite cap",)
        except (ValueError, OSError) as exc:
            return CGROUP_UNKNOWN, None, None, (
                f"memory.current unreadable/unparseable against a finite "
                f"cap: {exc or cur_raw!r}",)
        if current < 0:
            return CGROUP_UNKNOWN, None, None, (
                f"memory.current negative: {cur_raw!r}",)
        return CGROUP_FINITE, limit, current, ()


def read_memory_snapshot(*, meminfo_path: str | Path = "/proc/meminfo",
                         proc_cgroup_path: str | Path = "/proc/self/cgroup",
                         cgroup_fs_root: str | Path = "/sys/fs/cgroup",
                         ) -> MemorySnapshot:
    host, host_notes = read_host_available(meminfo_path)
    state, limit, current, notes = read_cgroup_budget(proc_cgroup_path,
                                                      cgroup_fs_root)
    if state == CGROUP_UNKNOWN:
        limit = current = None  # an unknown state carries no counters
    return MemorySnapshot(host_available_bytes=host, cgroup_state=state,
                          cgroup_limit_bytes=limit, cgroup_current_bytes=current,
                          notes=tuple(host_notes) + tuple(notes))


# ---------------------------------------------------------------------------
# Required bytes + verdict
# ---------------------------------------------------------------------------


def _dtype_bytes(name: Any, at: str) -> int:
    if not isinstance(name, str) or name.strip().lower() not in DTYPE_BYTES:
        raise MemoryPreflightError(
            f"{at}: must be a dtype this preflight models "
            f"({', '.join(sorted(set(DTYPE_BYTES)))}), got {name!r}")
    return DTYPE_BYTES[name.strip().lower()]


def encoder_memory_required(*, checkpoint_bytes: int, checkpoint_dtype: str,
                            target_dtype: str, headroom_bytes: int,
                            ) -> tuple[int, int]:
    """``(weights_bytes_in_ram, required_total)`` — an explicit, conservative
    bound. Expansion is ``checkpoint_bytes * target_width / storage_width``
    rounded **up**; an apparent shrink is clamped to the checkpoint total
    (the bytes on disk must be read and mapped whatever the target dtype);
    the explicit runtime headroom is added on top. There is no implicit
    headroom default: the caller states it."""
    if not _positive_int(checkpoint_bytes):
        raise MemoryPreflightError(
            f"checkpoint_bytes: must be a positive integer, got "
            f"{checkpoint_bytes!r}")
    if not _nonneg_int(headroom_bytes):
        raise MemoryPreflightError(
            f"headroom_bytes: must be an explicit non-negative integer "
            f"(no implicit default), got {headroom_bytes!r}")
    storage = _dtype_bytes(checkpoint_dtype, "checkpoint_dtype")
    target = _dtype_bytes(target_dtype, "target_dtype")
    expanded = math.ceil(checkpoint_bytes * target / storage)
    weights = max(checkpoint_bytes, expanded)
    return weights, weights + headroom_bytes


@dataclass(frozen=True)
class MemoryPreflightReport:
    """The preflight outcome as a JSON-safe record. ``decision`` is
    ``"proceed"`` (a conservative GO — NOT measured fit: passing alone never
    proves the load fits) or ``"refuse"`` (the numbers presented do not fit
    the applicable budget, or the constrained state is unknown)."""

    schema: ClassVar[str] = MEM_PREFLIGHT_SCHEMA

    checkpoint_bytes: int
    checkpoint_dtype: str
    target_dtype: str
    dtype_expanded_bytes: int
    headroom_bytes: int
    required_bytes: int
    host_available_bytes: int | None
    cgroup_state: str
    cgroup_limit_bytes: int | None
    cgroup_current_bytes: int | None
    cgroup_available_bytes: int | None
    budget_bytes: int | None
    decision: str
    reasons: tuple[str, ...]
    #: Always True: passing is a conservative prerequisite, never measured
    #: model-fit evidence (see module docstring).
    conservative_precondition: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "checkpoint_bytes": self.checkpoint_bytes,
                "checkpoint_dtype": self.checkpoint_dtype,
                "target_dtype": self.target_dtype,
                "dtype_expanded_bytes": self.dtype_expanded_bytes,
                "headroom_bytes": self.headroom_bytes,
                "required_bytes": self.required_bytes,
                "host_available_bytes": self.host_available_bytes,
                "cgroup_state": self.cgroup_state,
                "cgroup_limit_bytes": self.cgroup_limit_bytes,
                "cgroup_current_bytes": self.cgroup_current_bytes,
                "cgroup_available_bytes": self.cgroup_available_bytes,
                "budget_bytes": self.budget_bytes,
                "decision": self.decision,
                "reasons": list(self.reasons),
                "conservative_precondition": self.conservative_precondition}


def encoder_memory_preflight(checkpoint_bytes: int, *,
                             checkpoint_dtype: str, target_dtype: str,
                             headroom_bytes: int,
                             snapshot: MemorySnapshot | None = None,
                             meminfo_path: str | Path = "/proc/meminfo",
                             proc_cgroup_path: str | Path = "/proc/self/cgroup",
                             cgroup_fs_root: str | Path = "/sys/fs/cgroup",
                             ) -> MemoryPreflightReport:
    """Evaluate the preflight. Tests inject ``snapshot`` or the paths; the
    real call reads only the documented proc/cgroup files. Never reads a
    model file, never imports an ML package, never loads anything."""
    weights, required = encoder_memory_required(
        checkpoint_bytes=checkpoint_bytes, checkpoint_dtype=checkpoint_dtype,
        target_dtype=target_dtype, headroom_bytes=headroom_bytes)
    snap = snapshot if snapshot is not None else read_memory_snapshot(
        meminfo_path=meminfo_path, proc_cgroup_path=proc_cgroup_path,
        cgroup_fs_root=cgroup_fs_root)
    budget, reasons = snap.budget()
    cgroup_available = None
    if snap.cgroup_state == CGROUP_FINITE:
        cgroup_available = max(0, snap.cgroup_limit_bytes
                               - (snap.cgroup_current_bytes or 0))
    if budget is None:
        decision = "refuse"
        reasons = tuple(reasons) + ("no applicable budget can be proven",)
    elif required > budget:
        decision = "refuse"
        reasons = tuple(reasons) + (
            f"required {required} B exceeds the applicable budget "
            f"{budget} B",)
    else:
        decision = "proceed"
        reasons = tuple(reasons) + (
            "conservative prerequisite satisfied only: this does NOT prove "
            "the load fits (peak residency, allocator and conversion "
            "transients are not modelled)",)
    return MemoryPreflightReport(
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_dtype=checkpoint_dtype, target_dtype=target_dtype,
        dtype_expanded_bytes=weights, headroom_bytes=headroom_bytes,
        required_bytes=required,
        host_available_bytes=snap.host_available_bytes,
        cgroup_state=snap.cgroup_state,
        cgroup_limit_bytes=snap.cgroup_limit_bytes,
        cgroup_current_bytes=snap.cgroup_current_bytes,
        cgroup_available_bytes=cgroup_available,
        budget_bytes=budget, decision=decision, reasons=reasons)


def require_encoder_memory_fit(checkpoint_bytes: int, **kwargs: Any
                               ) -> MemoryPreflightReport:
    """:func:`encoder_memory_preflight` that RAISES
    :class:`MemoryPreflightError` instead of returning a refusal — the
    loader-side call site fails typed, before any ML import, weight read or
    loader call, on the first sign the numbers do not fit."""
    report = encoder_memory_preflight(checkpoint_bytes, **kwargs)
    if report.decision != "proceed":
        raise MemoryPreflightError(
            "encoder memory preflight refused: "
            + "; ".join(report.reasons)
            + f"; report={report.to_dict()}")
    return report
