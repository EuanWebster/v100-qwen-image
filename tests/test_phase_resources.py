"""CPU-only guarded tests for conservative phase-budget accounting.

No torch, no hardware probes, no GPU. Genuine no-double-count, foreign
memory exclusion, unknown/inconsistent/refused cases, freshness bounds,
host/device preservation and exact-zero semantics.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_engine.errors import MemoryGuardError
from qwen_engine.mem_preflight import MemorySnapshot
from qwen_engine.memory import DeviceIdentity, ResourceEvidence
from qwen_engine.phase_resources import (
    OWNED_RESERVED_SOURCE,
    PHASE_GPU_BUDGET_SCHEMA,
    PhaseBudgetError,
    PhaseGpuBudget,
    apply_phase_gpu_budget,
    compute_phase_gpu_budget,
)

NOW = 1_000_000.0
AGE = 30.0
TOTAL = 16 * 1024**3
FREE = 4 * 1024**3
OWNED = 2 * 1024**3


def device():
    return DeviceIdentity(uuid="GPU-abc123", architecture="sm_70",
                          driver_version="560.35.03", cuda_build="12.6")


def snapshot():
    return MemorySnapshot(host_available_bytes=8 * 1024**3, cgroup_state="unlimited")


def evidence(captured_at=NOW, total=TOTAL, free=FREE):
    return ResourceEvidence(
        captured_at=captured_at,
        host=snapshot(),
        device=device(),
        cuda_kernel_verified=True,
        gpu_total_bytes=total,
        gpu_available_bytes=free,
    )


def budget(free=FREE, owned=OWNED, captured_at=NOW, total=TOTAL, now=NOW, max_age_s=AGE):
    return compute_phase_gpu_budget(
        physical_total_bytes=total,
        physical_free_bytes=free,
        physical_captured_at=captured_at,
        physical_source="nvidia-smi:memory.total,memory.free",
        owned_torch_reserved_bytes=owned,
        owned_reserved_captured_at=captured_at,
        now=now,
        max_age_s=max_age_s,
        device_uuid=device().uuid,
    )


# ---------- genuine no-double-count ----------

def test_eligible_budget_adds_owned_reserved_exactly_once():
    b = budget()
    assert b.eligible_gpu_budget_bytes == FREE + OWNED
    updated = apply_phase_gpu_budget(evidence(), b, now=NOW, max_update_window_s=AGE)
    assert updated.gpu_available_bytes == FREE + OWNED
    # Applying again against the already-updated evidence must refuse: the
    # raw free no longer matches, so the add-back can never apply twice.
    with pytest.raises(PhaseBudgetError, match="raw gpu_available_bytes"):
        apply_phase_gpu_budget(updated, b, now=NOW, max_update_window_s=AGE)


def test_owned_zero_is_identity_update():
    b = budget(owned=0)
    assert b.eligible_gpu_budget_bytes == FREE
    updated = apply_phase_gpu_budget(evidence(), b, now=NOW, max_update_window_s=AGE)
    assert updated.gpu_available_bytes == FREE


# ---------- foreign memory excluded / conservatism ----------

def test_foreign_memory_stays_excluded():
    # 16 GiB total, 4 GiB free: 12 GiB used, of which only 2 GiB is OURS.
    b = budget(free=FREE, owned=OWNED)
    assert b.eligible_gpu_budget_bytes == FREE + OWNED
    # The 10 GiB of foreign/context used bytes are never credited back:
    # eligible budget stays strictly below whole-device total.
    assert b.eligible_gpu_budget_bytes < TOTAL
    assert TOTAL - b.eligible_gpu_budget_bytes == (TOTAL - FREE) - OWNED  # foreign 10 GiB
    assert any("foreign-process bytes are excluded" in n for n in b.notes)
    assert any("NOT added back" in n for n in b.notes)


# ---------- strict typing ----------

@pytest.mark.parametrize("value", [True, False, 1.0, 1.5, "1024", -1])
def test_non_int_or_negative_total_refused(value):
    with pytest.raises(PhaseBudgetError):
        compute_phase_gpu_budget(
            physical_total_bytes=value,
            physical_free_bytes=FREE,
            physical_captured_at=NOW,
            physical_source="probe",
            owned_torch_reserved_bytes=OWNED,
            owned_reserved_captured_at=NOW,
            now=NOW,
            max_age_s=AGE,
        )


def test_bool_owned_reserved_refused():
    with pytest.raises(PhaseBudgetError):
        budget(owned=True)


def test_float_free_refused():
    with pytest.raises(PhaseBudgetError):
        budget(free=float(FREE))


# ---------- inconsistent snapshots refused, never clipped ----------

def test_free_exceeds_total_refused():
    with pytest.raises(PhaseBudgetError, match="exceeds"):
        budget(free=TOTAL + 1)


def test_owned_exceeds_used_bytes_refused():
    # used = TOTAL - FREE = 12 GiB; claiming 13 GiB owned is incoherent.
    with pytest.raises(PhaseBudgetError, match="used bytes"):
        budget(owned=TOTAL - FREE + 1)


def test_owned_equal_used_boundary_allowed():
    b = budget(owned=TOTAL - FREE, free=FREE)
    assert b.eligible_gpu_budget_bytes == TOTAL  # all used bytes are ours


def test_handcrafted_inconsistent_dataclass_refused():
    with pytest.raises(PhaseBudgetError):
        PhaseGpuBudget(
            physical_total_bytes=100, physical_free_bytes=101,
            owned_torch_reserved_bytes=0, eligible_gpu_budget_bytes=101,
            physical_captured_at=NOW, owned_reserved_captured_at=NOW,
            physical_source="s", owned_reserved_source=OWNED_RESERVED_SOURCE)
    with pytest.raises(PhaseBudgetError):
        PhaseGpuBudget(
            physical_total_bytes=100, physical_free_bytes=50,
            owned_torch_reserved_bytes=60, eligible_gpu_budget_bytes=110,
            physical_captured_at=NOW, owned_reserved_captured_at=NOW,
            physical_source="s", owned_reserved_source=OWNED_RESERVED_SOURCE)
    with pytest.raises(PhaseBudgetError):
        PhaseGpuBudget(  # eligible != free + owned (a smuggled margin)
            physical_total_bytes=100, physical_free_bytes=50,
            owned_torch_reserved_bytes=10, eligible_gpu_budget_bytes=55,
            physical_captured_at=NOW, owned_reserved_captured_at=NOW,
            physical_source="s", owned_reserved_source=OWNED_RESERVED_SOURCE)


# ---------- freshness: stale / future / exact age bound ----------

def test_stale_physical_refused():
    with pytest.raises(PhaseBudgetError, match="stale or future"):
        budget(captured_at=NOW - AGE - 0.001)


def test_future_physical_refused():
    with pytest.raises(PhaseBudgetError, match="stale or future"):
        budget(captured_at=NOW + 0.001)


def test_stale_owned_reading_refused():
    with pytest.raises(PhaseBudgetError, match="owned reserved"):
        compute_phase_gpu_budget(
            physical_total_bytes=TOTAL, physical_free_bytes=FREE,
            physical_captured_at=NOW, physical_source="probe",
            owned_torch_reserved_bytes=OWNED,
            owned_reserved_captured_at=NOW - AGE - 1,
            now=NOW, max_age_s=AGE)


def test_exact_age_boundary_accepted():
    b = budget(captured_at=NOW - AGE)
    assert b.eligible_gpu_budget_bytes == FREE + OWNED


def test_no_default_freshness_window():
    import inspect
    sig = inspect.signature(compute_phase_gpu_budget)
    for name in ("now", "max_age_s"):
        assert sig.parameters[name].default is inspect.Parameter.empty
    sig2 = inspect.signature(apply_phase_gpu_budget)
    for name in ("now", "max_update_window_s"):
        assert sig2.parameters[name].default is inspect.Parameter.empty


# ---------- apply: mismatched device state refused ----------

def test_apply_refuses_total_mismatch():
    b = budget(total=TOTAL)
    with pytest.raises(PhaseBudgetError, match="gpu_total_bytes"):
        apply_phase_gpu_budget(evidence(total=TOTAL // 2), b,
                               now=NOW, max_update_window_s=AGE)


def test_apply_refuses_missing_counters():
    b = budget()
    with pytest.raises(PhaseBudgetError, match="lacks GPU counters"):
        apply_phase_gpu_budget(evidence(free=None), b, now=NOW, max_update_window_s=AGE)


def test_apply_refuses_measurement_time_gap_beyond_window():
    b = budget(captured_at=NOW)
    old = evidence(captured_at=NOW - AGE - 0.5)
    with pytest.raises(PhaseBudgetError):
        apply_phase_gpu_budget(old, b, now=NOW, max_update_window_s=AGE)


def test_apply_refuses_stale_evidence_even_if_times_agree():
    # Budget itself is fresh, but its physical capture time equals a stale
    # evidence capture time: the evidence freshness check must refuse.
    b = budget(captured_at=NOW - 100, max_age_s=200)
    old = evidence(captured_at=NOW - 100)
    with pytest.raises(PhaseBudgetError, match="stale or future"):
        apply_phase_gpu_budget(old, b, now=NOW, max_update_window_s=AGE)


def test_apply_refuses_wrong_types():
    with pytest.raises(PhaseBudgetError):
        apply_phase_gpu_budget(object(), budget(), now=NOW, max_update_window_s=AGE)
    with pytest.raises(PhaseBudgetError):
        apply_phase_gpu_budget(evidence(), object(), now=NOW, max_update_window_s=AGE)


# ---------- host/device facts preserved; host untouched ----------

def test_apply_preserves_device_cuda_and_host_unchanged():
    dev_host = MemorySnapshot(host_available_bytes=1234, cgroup_state="finite",
                              cgroup_limit_bytes=4096, cgroup_current_bytes=1024,
                              notes=("cap",))
    res = ResourceEvidence(captured_at=NOW, host=dev_host, device=device(),
                           cuda_kernel_verified=True,
                           gpu_total_bytes=TOTAL, gpu_available_bytes=FREE)
    updated = apply_phase_gpu_budget(res, budget(), now=NOW, max_update_window_s=AGE)
    assert updated.host is res.host  # host evidence byte-for-byte untouched
    assert updated.device is res.device
    assert updated.cuda_kernel_verified is res.cuda_kernel_verified
    assert updated.captured_at == res.captured_at
    assert updated.gpu_total_bytes == res.gpu_total_bytes
    assert updated.gpu_available_bytes == FREE + OWNED
    # Only one field changed.
    changed = {f.name for f in dataclasses.fields(updated)
               if getattr(updated, f.name) != getattr(res, f.name)}
    assert changed == {"gpu_available_bytes"}


# ---------- exact zeros semantics ----------

def test_exact_zero_free_and_total_semantics():
    # total=0 is refused by ResourceEvidence itself, but a zero FREE with
    # zero OWNED is an exact, meaningful zero budget (not unknown).
    b = compute_phase_gpu_budget(
        physical_total_bytes=1024, physical_free_bytes=0,
        physical_captured_at=NOW, physical_source="probe",
        owned_torch_reserved_bytes=0, owned_reserved_captured_at=NOW,
        now=NOW, max_age_s=AGE, device_uuid=device().uuid)
    assert b.eligible_gpu_budget_bytes == 0
    res = ResourceEvidence(captured_at=NOW, host=snapshot(), device=device(),
                           cuda_kernel_verified=True,
                           gpu_total_bytes=1024, gpu_available_bytes=0)
    updated = apply_phase_gpu_budget(res, b, now=NOW, max_update_window_s=AGE)
    assert updated.gpu_available_bytes == 0


def test_zero_owned_with_zero_free_is_zero_budget_not_unknown():
    b = compute_phase_gpu_budget(
        physical_total_bytes=1024, physical_free_bytes=0,
        physical_captured_at=NOW, physical_source="probe",
        owned_torch_reserved_bytes=0, owned_reserved_captured_at=NOW,
        now=NOW, max_age_s=AGE)
    assert b.eligible_gpu_budget_bytes == 0 and b.owned_torch_reserved_bytes == 0


# ---------- no guessed safety margins ----------

def test_no_margins_and_source_notes_present():
    b = budget()
    assert b.eligible_gpu_budget_bytes == b.physical_free_bytes + b.owned_torch_reserved_bytes
    assert b.owned_reserved_source == OWNED_RESERVED_SOURCE
    assert b.schema == PHASE_GPU_BUDGET_SCHEMA
    assert any("WHOLE-PHASE" in n and "NOT claimed physical free" in n for n in b.notes)


# ---------- no torch anywhere ----------

def test_module_never_imports_torch():
    code = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "import qwen_engine.phase_resources as m\n"
        "assert 'torch' not in sys.modules, 'phase_resources pulled in torch'\n"
        "assert 'diffusers' not in sys.modules\n"
        "print('clean')\n"
    ) % str(Path(__file__).resolve().parents[1] / "src")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


def test_error_is_memory_guard_family():
    assert issubclass(PhaseBudgetError, MemoryGuardError)


@pytest.mark.parametrize('captured', [NOW - AGE - 1, NOW + 1])
def test_apply_rechecks_owned_counter_freshness(captured):
    b = dataclasses.replace(budget(), owned_reserved_captured_at=captured)
    with pytest.raises(PhaseBudgetError, match='owned reserved'):
        apply_phase_gpu_budget(evidence(), b, now=NOW, max_update_window_s=AGE)


@pytest.mark.parametrize('uuid', [None, 'GPU-other-card'])
def test_apply_refuses_unknown_or_different_device(uuid):
    b = dataclasses.replace(budget(), device_uuid=uuid)
    with pytest.raises(PhaseBudgetError, match='UUID'):
        apply_phase_gpu_budget(evidence(), b, now=NOW, max_update_window_s=AGE)


def test_extreme_timestamp_is_typed_refusal():
    with pytest.raises(PhaseBudgetError):
        budget(now=10**400)
