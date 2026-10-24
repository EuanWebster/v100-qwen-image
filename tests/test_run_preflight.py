"""Clean the awkward conditional in the unverified-CUDA test."""
from __future__ import annotations

import json
import math
import re
from dataclasses import FrozenInstanceError, replace

import pytest

from qwen_engine import run_preflight as rp
from qwen_engine.bench import (
    CALIBRATION_FILE_SCHEMA,
    CalibrationLog,
    CalibrationObservation,
    save_calibration,
)
from qwen_engine.errors import MemoryGuardError
from qwen_engine.memory import (
    DeviceIdentity,
    MemoryPolicyError,
    PolicyDescriptor,
    ResourceEvidence,
)
from qwen_engine.mem_preflight import MemorySnapshot

NOW = 1000.0
DESCRIPTOR = PolicyDescriptor(
    phase="load", width=1024, height=1024, steps=40,
    prompt_tokens=25, negative_prompt_tokens=10, cfg_active=True,
    backend="sdpa", quant_format="fp16", residency="resident",
    vae_tiled_decode=False, encoder_dtype="fp16", execution_dtype="fp16",
    use_kv_cache=True, model_revision="a" * 40,
    artifact_set_sha256="b" * 64, runtime_identity_sha256="c" * 64)
DEVICE = DeviceIdentity("GPU-fixture-only", "sm_70", "fixture-driver", "12.6")


def resources(at=NOW, *, host_available=2000, cgroup_limit=1600, cgroup_current=100,
              device=DEVICE, cuda_verified=True, gpu_total=1000, gpu_available=900):
    return ResourceEvidence(at, MemorySnapshot(host_available, "finite",
                                               cgroup_limit, cgroup_current),
                            device, cuda_verified, gpu_total, gpu_available)


def observation(**overrides):
    """One COMPLETE exact observation matching DESCRIPTOR/DEVICE (peaks:
    gpu_used=250, host_tree=400), or legacy v1 when ``legacy=True``."""
    legacy = overrides.pop("legacy", False)
    fields = dict(
        observed_at=990.0, phase="load", model_revision="a" * 40,
        backend="sdpa", quant="fp16", residency="resident",
        resolution=[1024, 1024], tokens=25, prompt_tokens=25,
        steps=40, negative_prompt_tokens=10, cfg_active=True,
        encoder_dtype="fp16", execution_dtype="fp16", use_kv_cache=True,
        embedding_cache_hit=False, artifact_set_sha256="b" * 64,
        runtime_identity_sha256="c" * 64, device_uuid="GPU-fixture-only",
        device_architecture="sm_70", driver_version="fixture-driver",
        cuda_build="12.6", peak_gpu_reserved_bytes=200,
        peak_gpu_allocated_bytes=100, peak_gpu_used_bytes=250,
        peak_host_rss_bytes=300, peak_host_tree_bytes=400)
    if legacy:
        for key in ("steps", "negative_prompt_tokens", "cfg_active", "encoder_dtype",
                    "execution_dtype", "use_kv_cache", "embedding_cache_hit",
                    "artifact_set_sha256", "runtime_identity_sha256", "device_uuid",
                    "device_architecture", "driver_version", "cuda_build",
                    "peak_gpu_used_bytes", "peak_host_tree_bytes"):
            fields.pop(key)
    fields.update(overrides)
    return CalibrationObservation(**fields)


def write_calibration(path, observations):
    save_calibration(CalibrationLog(observations=tuple(observations)), path)
    return path


@pytest.fixture
def calib(tmp_path):
    return write_calibration(tmp_path / "calibration.json", [observation()])


def settings(calib, **overrides):
    fields = dict(calibration_path=calib, host_headroom_bytes=100,
                  gpu_headroom_bytes=100, max_resource_age_s=50,
                  max_calibration_age_s=60)
    fields.update(overrides)
    return rp.PreflightSettings(**fields)


def assess(calib, reader=None, *, now=NOW, **setting_overrides):
    return rp.assess_run_phase(DESCRIPTOR, reader or (lambda: resources()),
                               settings(calib, **setting_overrides), now=now)


# --- settings: strict, explicit, frozen --------------------------------------

def test_settings_frozen_and_reject_guessed_or_malformed_values():
    s = settings("cal.json")
    with pytest.raises(FrozenInstanceError):
        s.host_headroom_bytes = 999
    bad = [
        dict(host_headroom_bytes=-1), dict(gpu_headroom_bytes=-5),
        dict(host_headroom_bytes=100.0), dict(gpu_headroom_bytes=True),
        dict(max_resource_age_s=float("nan")), dict(max_calibration_age_s=math.inf),
        dict(max_resource_age_s=-1), dict(host_headroom_bytes=None),
        dict(calibration_path=""), dict(calibration_path=None),
    ]
    for override in bad:
        with pytest.raises(MemoryPolicyError):
            settings("cal.json", **override)


def test_settings_have_no_defaults_for_safety_budgets():
    import dataclasses
    for field in dataclasses.fields(rp.PreflightSettings):
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING


# --- exact evidence round trip -----------------------------------------------

def test_complete_exact_evidence_passes_with_exact_arithmetic(calib):
    decision = assess(calib)
    assert decision.allowed and not decision.reasons
    assert decision.matching_samples == 1
    assert decision.required_host_bytes == 500  # 400 tree peak + 100 headroom
    assert decision.required_gpu_bytes == 350   # 250 device-wide peak + 100
    assert decision.host_budget_bytes == 1500 and decision.gpu_budget_bytes == 900
    # require_run_phase returns the same allowed decision (fresh re-reads,
    # so equality — not object identity — is the contract).
    required = rp.require_run_phase(DESCRIPTOR, lambda: resources(),
                                    settings(calib), now=NOW)
    assert required == decision


def test_legacy_incomplete_record_refuses_typed_and_names_missing_fields(tmp_path):
    path = write_calibration(tmp_path / "c.json", [observation(legacy=True)])
    with pytest.raises(MemoryGuardError) as exc:
        rp.load_evidence(path)
    assert "not exact evidence" in str(exc.value)
    assert "device_uuid" in str(exc.value) and "peak_gpu_used_bytes" in str(exc.value)
    # The facade never lets a partial record become a measurement.
    with pytest.raises(MemoryGuardError):
        assess(path)


# --- policy refusals surface typed through require_run_phase ------------------

def test_require_run_phase_raises_memory_guard_error_listing_reasons(tmp_path):
    path = write_calibration(tmp_path / "c.json",
                             [observation(device_uuid="GPU-someone-elses")])
    with pytest.raises(MemoryGuardError, match="exact_calibration_missing"):
        rp.require_run_phase(DESCRIPTOR, lambda: resources(), settings(path), now=NOW)


def test_stale_resource_and_stale_calibration_refuse_freshly_each_time(calib):
    # Same file, same reader: only the clock moved. Nothing is cached, so the
    # exact permission from NOW cannot survive into a later phase.
    assert assess(calib).allowed
    stale_resources = assess(calib, now=NOW + 500)
    assert not stale_resources.allowed
    assert "resource_snapshot_stale_or_future" in stale_resources.reasons
    stale_calib = assess(calib, now=NOW + 200)
    assert not stale_calib.allowed
    assert "matching_calibration_stale_or_future" in stale_calib.reasons
    with pytest.raises(MemoryGuardError, match="matching_calibration_stale_or_future"):
        rp.require_run_phase(DESCRIPTOR, lambda: resources(),
                             settings(calib), now=NOW + 200)


@pytest.mark.parametrize("field,value,reason", [
    ("host_headroom_bytes", 1200, "host_budget_exceeded"),
    ("gpu_headroom_bytes", 700, "gpu_budget_exceeded"),
])
def test_insufficient_budget_refuses(calib, field, value, reason):
    decision = assess(calib, **{field: value})
    assert not decision.allowed and reason in decision.reasons


def test_unknown_conditioning_length_refuses_never_inferred(calib):
    query = replace(DESCRIPTOR, prompt_tokens=None)
    decision = rp.assess_run_phase(query, lambda: resources(), settings(calib), now=NOW)
    assert not decision.allowed
    assert "conditioning_lengths_unknown" in decision.reasons
    assert "exact_calibration_missing" in decision.reasons


def test_unverified_cuda_and_unknown_host_scope_refuse(calib):
    decision = assess(calib, reader=lambda: resources(cuda_verified=False))
    assert not decision.allowed
    assert "cuda_execution_unverified" in decision.reasons
    # A caller that cannot prove a complete host scope must inject unknown,
    # not optimistic: cgroup state "unknown" has no proven budget.
    unknown_scope = ResourceEvidence(
        NOW, MemorySnapshot(2000, "unknown"), DEVICE, True, 1000, 900)
    decision = assess(calib, reader=lambda: unknown_scope)
    assert not decision.allowed
    assert "complete_host_budget_unknown" in decision.reasons


# --- freshness: fresh reads on EVERY call -------------------------------------

def test_resources_and_calibration_are_reread_on_every_call(calib):
    calls = {"n": 0}

    def reader():
        calls["n"] += 1
        return resources()

    assert assess(calib, reader).allowed
    assert calls["n"] == 1
    assert assess(calib, reader).allowed
    assert calls["n"] == 2
    # Replace the file on disk: the next call sees the new truth immediately.
    write_calibration(calib, [observation(device_uuid="GPU-swapped")])
    second = assess(calib, reader)
    assert calls["n"] == 3
    assert not second.allowed and second.matching_samples == 0


# --- calibration file boundary: corrupted / unreadable / oversized ------------

def test_corrupted_and_unreadable_calibration_refuse_typed(tmp_path):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json at all", encoding="utf-8")
    with pytest.raises(MemoryPolicyError, match="refused"):
        rp.load_evidence(corrupt)
    with pytest.raises(MemoryPolicyError):
        rp.load_evidence(tmp_path / "does-not-exist.json")
    directory = tmp_path / "a-directory"
    directory.mkdir()
    with pytest.raises(MemoryPolicyError):
        rp.load_evidence(directory)
    # A schema-valid file whose record is internally malformed is refused.
    doc = json.loads(CalibrationLog(observations=(observation(),)).to_json())
    doc["observations"][0]["steps"] = 0
    (tmp_path / "bad-field.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(MemoryPolicyError):
        rp.load_evidence(tmp_path / "bad-field.json")
    # Unknown keys are schema violations, never silently ignored.
    doc = json.loads(CalibrationLog(observations=(observation(),)).to_json())
    doc["observations"][0]["expert_override"] = True
    (tmp_path / "extra-key.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(MemoryPolicyError):
        rp.load_evidence(tmp_path / "extra-key.json")


def test_oversized_calibration_refuses_without_reading_it_all(calib):
    calib.write_bytes(calib.read_bytes() + b" " * (rp.CALIBRATION_READ_CAP_BYTES + 1))
    with pytest.raises(MemoryPolicyError, match="bounded-read cap"):
        rp.load_evidence(calib)


def test_facade_writes_nothing_anywhere(tmp_path, calib):
    before = sorted(p.name for p in tmp_path.iterdir())
    assert assess(calib).allowed
    with pytest.raises(MemoryPolicyError):
        rp.load_evidence(tmp_path / "missing.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# --- resource_reader: the trusted boundary still fails closed -----------------

def test_source_errors_and_bad_reader_outputs_never_pass(calib):
    def boom():
        raise RuntimeError("nvml exploded")

    with pytest.raises(MemoryPolicyError, match="resource measurement failed"):
        assess(calib, boom)

    def wrong_type():
        return {"gpu_available_bytes": 10**12}  # a dict is not evidence

    with pytest.raises(MemoryPolicyError, match="ResourceEvidence"):
        assess(calib, wrong_type)

    with pytest.raises(MemoryPolicyError, match="measurement boundary"):
        rp.assess_run_phase(DESCRIPTOR, "not-callable", settings(calib), now=NOW)

    # A reader that lies with an invalid evidence object is refused by the
    # evidence type itself, not laundered by the facade.
    with pytest.raises(MemoryGuardError):
        assess(calib, lambda: ResourceEvidence(
            NOW, MemorySnapshot(2000, "finite", 1600, 100), DEVICE, True,
            1000, 1500))


def test_malformed_facade_inputs_refuse_before_any_measurement(calib):
    reader = lambda: resources()
    with pytest.raises(MemoryPolicyError, match="PolicyDescriptor"):
        rp.assess_run_phase({"phase": "load"}, reader, settings(calib))
    with pytest.raises(MemoryPolicyError, match="PreflightSettings"):
        rp.assess_run_phase(DESCRIPTOR, reader, {"calibration_path": str(calib)})
    with pytest.raises(MemoryPolicyError, match="now override"):
        assess(calib, reader, now=float("nan"))
    with pytest.raises(MemoryPolicyError):
        rp.load_evidence(12345)


def test_facade_is_ml_free_and_policy_errors_are_exit3_family():
    import pathlib
    import sys

    source = pathlib.Path(rp.__file__).read_text(encoding="utf-8")
    for forbidden in ("import torch", "import pynvml", "import ctypes",
                      "import subprocess"):
        assert forbidden not in source
    assert not {"torch", "diffusers", "transformers", "pynvml"} & set(sys.modules)
    assert issubclass(MemoryPolicyError, MemoryGuardError)
    assert rp.CALIBRATION_READ_CAP_BYTES == 16 * 1024 * 1024
    # Sanity: the fixture file really is the calibration schema we claim.
    assert CALIBRATION_FILE_SCHEMA == "qwen_engine.calibration.v1"
    assert re.fullmatch(r"[0-9a-f]{40}", "a" * 40)
