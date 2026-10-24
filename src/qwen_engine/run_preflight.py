"""Runtime preflight facade over the strict memory policy (ML-free glue).

This module wires three EXISTING layers together for a production adapter:

* ``bench.load_calibration`` — bounded (<=16 MiB) strict JSON read of a
  calibration log,
* ``calibration.calibration_evidence`` — the exactness bridge that refuses
  legacy/incomplete observations instead of partially trusting them,
* ``memory.assess_memory`` — the exact-match conservative arithmetic, used
  verbatim and never weakened here.

Design rules (see ``docs/run-preflight-contract.md``):

* **No guessed safety budgets.** ``PreflightSettings`` is frozen and every
  headroom/age value must be an explicit finite non-negative number; there
  are no defaults anywhere in this facade.
* **Nothing is cached.** Each ``assess_run_phase`` call re-reads the
  calibration file AND invokes ``resource_reader`` afresh, so a stale
  permission can never survive between phases.
* **Fail closed.** Malformed settings/evidence, unreadable or oversized
  files, exceptions from the injected resource reader, and every unknown
  fact surface as typed ``MemoryPolicyError`` (a ``MemoryGuardError``) —
  never as a pass.
* **No hardware probing.** This facade imports no CUDA/torch/NVML and
  probes nothing. ``resource_reader`` is the deliberate trusted measurement
  boundary: the production glue supplies real fresh evidence; this module
  only validates and arbitrates it.
* **No expert bypass, no phase/config substitution.** The caller's exact
  ``PolicyDescriptor`` (including measured conditioning token lengths) is
  what gets assessed; nothing here infers tokens from characters or swaps
  phases.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from qwen_engine import bench
from qwen_engine.bench import MAX_BENCH_FILE_BYTES, load_calibration
from qwen_engine.calibration import calibration_evidence
from qwen_engine.errors import MemoryGuardError
from qwen_engine.memory import (
    CalibrationEvidence,
    MemoryDecision,
    PolicyDescriptor,
    ResourceEvidence,
    assess_memory,
    MemoryPolicyError,
)

__all__ = [
    "PreflightSettings",
    "load_evidence",
    "assess_run_phase",
    "require_run_phase",
]

#: The facade reads calibration through the same bounded path as bench
#: (regular file, <= 16 MiB, no growth during read); oversized is a refusal.
CALIBRATION_READ_CAP_BYTES = MAX_BENCH_FILE_BYTES

#: Hard ceiling on records one calibration file may contribute; beyond it the
#: file is refused outright rather than silently truncated to
#: ``memory.MAX_OBSERVATIONS`` (which would drop evidence silently).
MAX_EVIDENCE_RECORDS = 65536


def _nonneg_number(value) -> bool:
    """Explicit finite non-negative int/float (bools and NaN/inf refused)."""
    return (type(value) in (int, float) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


@dataclass(frozen=True)
class PreflightSettings:
    """Explicit operator-supplied preflight parameters. Frozen, and every
    field is mandatory: there is no default headroom, age limit or path to
    fall back on, because a guessed safety budget is exactly what the strict
    policy refuses to encode.

    * ``calibration_path`` — file the exact evidence is (re)read from.
    * ``host_headroom_bytes`` / ``gpu_headroom_bytes`` — non-negative
      integers added to the worst matching measured peak.
    * ``max_resource_age_s`` / ``max_calibration_age_s`` — finite
      non-negative freshness windows handed to ``assess_memory``.
    """

    calibration_path: str | Path
    host_headroom_bytes: int
    gpu_headroom_bytes: int
    max_resource_age_s: float
    max_calibration_age_s: float

    def __post_init__(self):
        if (not isinstance(self.calibration_path, (str, Path))
                or not str(self.calibration_path).strip()):
            raise MemoryPolicyError(
                "calibration_path must be a non-empty str/Path; the facade "
                "never guesses a calibration location")
        if not _nonneg_number(self.max_resource_age_s) or not _nonneg_number(self.max_calibration_age_s):
            raise MemoryPolicyError(
                "max_resource_age_s/max_calibration_age_s must be explicit "
                "finite non-negative numbers, not defaults")
        # assess_memory demands strict ints for headroom; refuse bools/floats
        # here with a facade-level message instead of a generic downstream one.
        if (type(self.host_headroom_bytes) is not int
                or type(self.gpu_headroom_bytes) is not int
                or self.host_headroom_bytes < 0 or self.gpu_headroom_bytes < 0):
            raise MemoryPolicyError(
                "headroom budgets must be explicit non-negative integer "
                "byte counts; no safety budget is ever guessed")


def load_evidence(path: str | Path) -> tuple[CalibrationEvidence, ...]:
    """Re-read a calibration file into typed policy evidence.

    Every record passes through the exactness bridge: a legacy or incomplete
    observation is a typed refusal listing its missing fields — it is never
    silently converted into a (partial) measurement. Unreadable, oversized
    (>16 MiB), non-UTF-8 or schema-invalid files refuse as
    ``MemoryPolicyError``. Returns a tuple (safe to hand to
    ``assess_memory`` verbatim).
    """
    if not isinstance(path, (str, Path)):
        raise MemoryPolicyError(
            f"calibration path must be str/Path, got {type(path).__name__}")
    target = Path(path)
    try:
        raw = bench.read_bounded_file(target, CALIBRATION_READ_CAP_BYTES)
    except (OSError, ValueError, RuntimeError) as exc:
        raise MemoryPolicyError(
            f"calibration file not readable within the {CALIBRATION_READ_CAP_BYTES}"
            f" byte bounded-read cap: {target}: {exc}") from exc
    try:
        # Parse the same bounded bytes, avoiding a second mutable file read.
        from qwen_engine._jsonsafe import strict_json_loads
        log = bench.CalibrationLog.from_dict(strict_json_loads(raw.decode("utf-8")))
    except (bench.BenchSchemaError, ValueError, RecursionError) as exc:
        raise MemoryPolicyError(f"calibration file refused: {exc}") from exc
    if len(log.observations) > MAX_EVIDENCE_RECORDS:
        raise MemoryPolicyError(
            f"calibration file carries {len(log.observations)} observations, "
            f"above the {MAX_EVIDENCE_RECORDS}-record facade ceiling; refusing "
            "rather than silently dropping evidence")
    try:
        # Exactness bridge: raises CalibrationError (a MemoryGuardError via
        # EngineError) naming every missing field on incomplete records.
        return tuple(calibration_evidence(o) for o in log.observations)
    except MemoryGuardError:
        raise
    except Exception as exc:  # defensive: unknown facts never become evidence
        raise MemoryPolicyError(
            f"calibration observations cannot form exact evidence: {exc}") from exc


def assess_run_phase(descriptor: PolicyDescriptor,
                     resource_reader: Callable[[], ResourceEvidence],
                     settings: PreflightSettings,
                     *, now: float | None = None) -> MemoryDecision:
    """One fresh preflight decision for one generation phase.

    Reads the calibration file and invokes ``resource_reader()`` on EVERY
    call — no permission is ever cached across calls — then delegates to
    ``memory.assess_memory`` with the exact descriptor, the freshly read
    device identity and the explicit settings. Exceptions from the resource
    reader are wrapped as typed ``MemoryPolicyError`` (an unreadable fact
    is a refusal, never a pass). Returns the decision without raising on a
    policy refusal; use :func:`require_run_phase` for the raising form.
    """
    if type(descriptor) is not PolicyDescriptor:
        raise MemoryPolicyError(
            "an exact PolicyDescriptor is required; the facade never derives "
            "one from prompts, characters or config guesses")
    if type(settings) is not PreflightSettings:
        raise MemoryPolicyError("explicit PreflightSettings are required")
    if not callable(resource_reader):
        raise MemoryPolicyError(
            "a resource_reader callable is the trusted measurement boundary; "
            "this facade never probes hardware itself")
    if now is not None and not _nonneg_number(now):
        raise MemoryPolicyError(
            "now override must be a finite non-negative number or None")
    try:
        resources = resource_reader()
    except MemoryGuardError:
        raise
    except Exception as exc:
        raise MemoryPolicyError(
            f"resource measurement failed: {type(exc).__name__}: {exc}") from exc
    if type(resources) is not ResourceEvidence:
        raise MemoryPolicyError(
            "resource_reader must return an exact ResourceEvidence; "
            f"got {type(resources).__name__}")
    evidence = load_evidence(settings.calibration_path)
    return assess_memory(
        descriptor, resources, evidence,
        host_headroom_bytes=settings.host_headroom_bytes,
        gpu_headroom_bytes=settings.gpu_headroom_bytes,
        max_resource_age_s=settings.max_resource_age_s,
        max_calibration_age_s=settings.max_calibration_age_s,
        now=now if now is not None else time.time())


def require_run_phase(descriptor: PolicyDescriptor,
                      resource_reader: Callable[[], ResourceEvidence],
                      settings: PreflightSettings,
                      *, now: float | None = None) -> MemoryDecision:
    """Like :func:`assess_run_phase` but raises ``MemoryGuardError``
    (``MemoryPolicyError``) on any refusal, listing the policy reasons.
    The raised type is the CLI exit-3 memory-refusal family."""
    decision = assess_run_phase(descriptor, resource_reader, settings, now=now)
    if not decision.allowed:
        raise MemoryPolicyError(
            "run preflight refused: " + ", ".join(decision.reasons))
    return decision
