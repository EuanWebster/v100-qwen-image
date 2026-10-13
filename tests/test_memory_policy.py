"""Pure evidence/identity/arithmetic tests, not actual measured model fit."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from qwen_engine import memory as m
from qwen_engine.bench import CalibrationObservation
from qwen_engine.mem_preflight import MemorySnapshot


@pytest.fixture
def evidence():
    descriptor = m.PolicyDescriptor(phase="load", width=1024, height=1024, steps=40,
        prompt_tokens=25, negative_prompt_tokens=10, cfg_active=True, backend="sdpa",
        quant_format="fp16", residency="resident", vae_tiled_decode=False,
        encoder_dtype="fp16", execution_dtype="fp16", use_kv_cache=True,
        model_revision="a" * 40, artifact_set_sha256="b" * 64, runtime_identity_sha256="c" * 64)
    device = m.DeviceIdentity("GPU-fixture-only", "sm_70", "fixture-driver", "12.6")
    resources = m.ResourceEvidence(1000, MemorySnapshot(2000, "finite", 1600, 100),
                                   device, True, 1000, 900)
    sample = m.CalibrationEvidence(descriptor, device, 990, 200, 300, 100, 250, 400)
    return descriptor, resources, sample


def assess(evidence, **kwargs):
    descriptor, resources, sample = evidence
    return m.assess_memory(descriptor, resources, (sample,), host_headroom_bytes=100,
                           gpu_headroom_bytes=100, max_resource_age_s=5, max_calibration_age_s=60,
                           now=1000, **kwargs)


def test_exact_complete_trusted_fixture_arithmetic(evidence):
    result = assess(evidence)
    assert result.allowed and not result.reasons and result.matching_samples == 1
    assert result.host_budget_bytes == 1500 and result.gpu_budget_bytes == 900
    assert result.required_host_bytes == 500 and result.required_gpu_bytes == 350
    assert result.descriptor.latent_tokens == 4096
    with pytest.raises(FrozenInstanceError):
        result.allowed = False


@pytest.mark.parametrize("field,value", [
    ("phase", "decode"), ("width", 2048), ("height", 2048), ("steps", 41),
    ("prompt_tokens", 26), ("negative_prompt_tokens", 11), ("backend", "efficient"),
    ("quant_format", "gguf-q8_0"), ("residency", "streamed"), ("vae_tiled_decode", True),
    ("encoder_dtype", "bf16"), ("use_kv_cache", False), ("model_revision", "d" * 40),
    ("artifact_set_sha256", "d" * 64), ("runtime_identity_sha256", "d" * 64), ("embedding_cache_hit", True),
])
def test_every_execution_conditioning_identity_drift_refuses(evidence, field, value):
    descriptor, resources, sample = evidence
    result = assess((replace(descriptor, **{field: value}), resources, sample))
    assert not result.allowed and result.reasons == ("exact_calibration_missing",)


def test_cfg_activity_drift_refuses_and_never_assumes_doubled_batch(evidence):
    descriptor, resources, sample = evidence
    changed = replace(descriptor, cfg_active=False, negative_prompt_tokens=None)
    result = assess((changed, resources, sample))
    assert not result.allowed and "exact_calibration_missing" in result.reasons


def test_step_calibration_is_exact_and_never_interpolated(evidence):
    descriptor, resources, sample = evidence
    query = replace(descriptor, phase="denoise-step", step_index=2)
    sample = replace(sample, descriptor=query)
    assert assess((query, resources, sample)).allowed
    assert not assess((replace(query, step_index=3), resources, sample)).allowed


@pytest.mark.parametrize("field,value", [("uuid", "GPU-other"), ("architecture", "sm_80"),
                                         ("driver_version", "different"), ("cuda_build", "12.8")])
def test_device_driver_build_drift_refuses(evidence, field, value):
    descriptor, resources, sample = evidence
    resources = replace(resources, device=replace(resources.device, **{field: value}))
    result = assess((descriptor, resources, sample))
    assert not result.allowed and "exact_calibration_missing" in result.reasons


def test_no_gpu_and_no_observations_never_invent_readiness(evidence):
    descriptor, resources, sample = evidence
    resources = replace(resources, device=None, cuda_kernel_verified=False,
                        gpu_total_bytes=None, gpu_available_bytes=None)
    result = assess((descriptor, resources, sample))
    assert not result.allowed and "cuda_execution_unverified" in result.reasons and "gpu_budget_unknown" in result.reasons
    with pytest.raises(m.MemoryPolicyError, match="refused"):
        m.require_memory(descriptor, resources, (), host_headroom_bytes=0, gpu_headroom_bytes=0,
                          max_resource_age_s=5, max_calibration_age_s=60, now=1000)


def test_build_flags_without_kernel_execution_refuse(evidence):
    descriptor, resources, sample = evidence
    result = assess((descriptor, replace(resources, cuda_kernel_verified=False), sample))
    assert not result.allowed and result.reasons == ("cuda_execution_unverified",)


@pytest.mark.parametrize("host", [MemorySnapshot(10 ** 12, "unknown", advisory_visible_available_bytes=10 ** 12),
                                  MemorySnapshot(None, "unlimited"), MemorySnapshot(2000, "finite", 0, 0)])
def test_unknown_scope_or_zero_container_cap_not_overridden(evidence, host):
    descriptor, resources, sample = evidence
    result = assess((descriptor, replace(resources, host=host), sample))
    assert not result.allowed
    assert "complete_host_budget_unknown" in result.reasons or "host_budget_exceeded" in result.reasons


@pytest.mark.parametrize("field", ["prompt_tokens", "negative_prompt_tokens"])
def test_unmeasured_conditioning_is_not_zero_or_guessed(evidence, field):
    descriptor, resources, sample = evidence
    descriptor = replace(descriptor, **{field: None})
    sample = replace(sample, descriptor=descriptor)
    result = assess((descriptor, resources, sample))
    assert not result.allowed and result.reasons == ("conditioning_lengths_unknown",)


@pytest.mark.parametrize("field", ["peak_host_rss_bytes", "peak_host_tree_bytes", "peak_gpu_reserved_bytes", "peak_gpu_used_bytes"])
def test_incomplete_matching_peak_blocks_even_with_good_sample(evidence, field):
    descriptor, resources, sample = evidence
    incomplete = replace(sample, **{field: None})
    result = m.assess_memory(descriptor, resources, (sample, incomplete), host_headroom_bytes=0,
        gpu_headroom_bytes=0, max_resource_age_s=5, max_calibration_age_s=60, now=1000)
    assert not result.allowed and result.reasons == ("matching_peak_unmeasured",)
    assert result.required_gpu_bytes is None and result.required_host_bytes is None


@pytest.mark.parametrize("gpu", [None, 0, 300])
def test_missing_or_insufficient_gpu_available_refuses(evidence, gpu):
    descriptor, resources, sample = evidence
    result = assess((descriptor, replace(resources, gpu_available_bytes=gpu), sample))
    assert not result.allowed and ("gpu_budget_unknown" in result.reasons or "gpu_budget_exceeded" in result.reasons)


def test_worst_sample_not_mean_or_best_controls_arithmetic(evidence):
    descriptor, resources, sample = evidence
    high = replace(sample, peak_gpu_used_bytes=850, peak_host_rss_bytes=1450, peak_host_tree_bytes=1450)
    result = m.assess_memory(descriptor, resources, (sample, high), host_headroom_bytes=100,
        gpu_headroom_bytes=100, max_resource_age_s=5, max_calibration_age_s=60, now=1000)
    assert not result.allowed and result.required_gpu_bytes == 950 and result.required_host_bytes == 1550
    assert set(result.reasons) == {"host_budget_exceeded", "gpu_budget_exceeded"}


@pytest.mark.parametrize("kind", ["resource-stale", "resource-future", "sample-stale", "sample-future"])
def test_stale_or_future_evidence_refuses(evidence, kind):
    descriptor, resources, sample = evidence
    if kind.startswith("resource"):
        resources = replace(resources, captured_at=990 if kind.endswith("stale") else 1001)
    else:
        sample = replace(sample, observed_at=900 if kind.endswith("stale") else 1001)
    result = assess((descriptor, resources, sample))
    assert not result.allowed and any("stale_or_future" in reason for reason in result.reasons)


@pytest.mark.parametrize("clock", [None, True, float("nan"), float("inf"), 10 ** 1000, -1])
def test_unknown_clock_cannot_allow(evidence, monkeypatch, clock):
    descriptor, resources, sample = evidence
    monkeypatch.setattr(m.time, "time", lambda: clock)
    result = m.assess_memory(descriptor, resources, (sample,), host_headroom_bytes=0,
        gpu_headroom_bytes=0, max_resource_age_s=5, max_calibration_age_s=60)
    assert not result.allowed and "current_time_unknown" in result.reasons


def test_clock_fault_refuses_without_masking_or_fake_time(evidence, monkeypatch):
    monkeypatch.setattr(m.time, "time", lambda: (_ for _ in ()).throw(OSError("clock failed")))
    descriptor, resources, sample = evidence
    result = m.assess_memory(descriptor, resources, (sample,), host_headroom_bytes=0,
        gpu_headroom_bytes=0, max_resource_age_s=5, max_calibration_age_s=60)
    assert not result.allowed and "current_time_unknown" in result.reasons


@pytest.mark.parametrize("field,value", [("phase", []), ("phase", "denoise"), ("schema", "wrong"),
    ("width", True), ("width", 100), ("height", 8192), ("steps", 0), ("steps", 1001),
    ("backend", "auto"), ("backend", []), ("quant_format", ""), ("residency", "auto"),
    ("cfg_active", 1), ("vae_tiled_decode", 1), ("encoder_dtype", "auto"),
    ("execution_dtype", "bf16"), ("use_kv_cache", None), ("model_revision", "main"),
    ("artifact_set_sha256", "z" * 64), ("runtime_identity_sha256", None),
    ("prompt_tokens", 0), ("prompt_tokens", True), ("negative_prompt_tokens", 1_000_001),
    ("step_index", 0), ("embedding_cache_hit", 1)])
def test_invalid_or_unresolved_descriptor_typed_refusal(evidence, field, value):
    with pytest.raises(m.MemoryPolicyError):
        replace(evidence[0], **{field: value})


@pytest.mark.parametrize("field,value", [("captured_at", True), ("captured_at", 10 ** 1000),
    ("host", {}), ("device", {}), ("cuda_kernel_verified", 1), ("gpu_total_bytes", 0),
    ("gpu_total_bytes", -1), ("gpu_available_bytes", 1001), ("gpu_available_bytes", True)])
def test_invalid_resource_evidence_typed_refusal(evidence, field, value):
    with pytest.raises(m.MemoryPolicyError):
        replace(evidence[1], **{field: value})


@pytest.mark.parametrize("field,value", [("descriptor", {}), ("device", {}), ("observed_at", 0),
    ("peak_host_rss_bytes", 0), ("peak_host_rss_bytes", True), ("peak_gpu_reserved_bytes", -1),
    ("peak_gpu_allocated_bytes", 201), ("peak_gpu_used_bytes", 199), ("peak_host_tree_bytes", 299)])
def test_invalid_calibration_evidence_typed_refusal(evidence, field, value):
    with pytest.raises(m.MemoryPolicyError):
        replace(evidence[2], **{field: value})


@pytest.mark.parametrize("field,value", [("host_headroom_bytes", True), ("gpu_headroom_bytes", -1),
    ("max_resource_age_s", 0), ("max_resource_age_s", 10 ** 1000), ("max_calibration_age_s", float("inf"))])
def test_invalid_policy_options_typed_refusal(evidence, field, value):
    descriptor, resources, sample = evidence
    kwargs = dict(host_headroom_bytes=0, gpu_headroom_bytes=0, max_resource_age_s=5, max_calibration_age_s=60)
    kwargs[field] = value
    with pytest.raises(m.MemoryPolicyError):
        m.assess_memory(descriptor, resources, (sample,), now=1000, **kwargs)


def test_legacy_observation_is_not_upgraded_to_complete_evidence(evidence):
    descriptor, resources, sample = evidence
    old = CalibrationObservation(observed_at=990, phase="load", model_revision="a" * 40,
        backend="sdpa", quant="fp16", residency="resident", tokens=4096, resolution=(1024, 1024),
        prompt_tokens=25, peak_gpu_reserved_bytes=200, peak_host_rss_bytes=300)
    with pytest.raises(m.MemoryPolicyError, match="complete typed"):
        m.assess_memory(descriptor, resources, (old,), host_headroom_bytes=0, gpu_headroom_bytes=0,
                         max_resource_age_s=5, max_calibration_age_s=60, now=1000)


def test_empty_2k_calibration_refuses_and_has_no_expert_bypass(evidence):
    descriptor, resources, sample = evidence
    descriptor = replace(descriptor, width=2048, height=2048)
    result = m.assess_memory(descriptor, resources, (), host_headroom_bytes=0, gpu_headroom_bytes=0,
        max_resource_age_s=5, max_calibration_age_s=60, now=1000)
    assert not result.allowed and descriptor.latent_tokens == 16384


def test_oversized_or_mutable_inventory_refused(evidence):
    descriptor, resources, sample = evidence
    for records in ([sample], (sample,) * (m.MAX_OBSERVATIONS + 1)):
        with pytest.raises(m.MemoryPolicyError, match="bounded"):
            m.assess_memory(descriptor, resources, records, host_headroom_bytes=0, gpu_headroom_bytes=0,
                max_resource_age_s=5, max_calibration_age_s=60, now=1000)


def test_parent_rss_or_torch_reserved_alone_never_authorize_fit(evidence):
    descriptor, resources, sample = evidence
    incomplete = replace(sample, peak_gpu_used_bytes=None, peak_host_tree_bytes=None)
    result = assess((descriptor, resources, incomplete))
    assert not result.allowed and "matching_peak_unmeasured" in result.reasons


def test_absent_device_cannot_claim_counters_or_verified_execution(evidence):
    resources = evidence[1]
    with pytest.raises(m.MemoryPolicyError, match="absent device"):
        replace(resources, device=None)


@pytest.mark.parametrize("index", [None, True, -1, 40])
def test_invalid_denoise_step_index_refuses(evidence, index):
    with pytest.raises(m.MemoryPolicyError, match="step index"):
        replace(evidence[0], phase="denoise-step", step_index=index)


def test_wrong_profile_refuses_even_when_calibration_matches(evidence):
    descriptor, resources, sample = evidence
    other = replace(resources.device, architecture="sm_80")
    result = assess((descriptor, replace(resources, device=other), replace(sample, device=other)))
    assert not result.allowed and result.reasons == ("device_or_cuda_profile_unreviewed",)


@pytest.mark.parametrize("field,value", [("uuid", ""), ("architecture", []), ("driver_version", " 123"),
    ("cuda_build", "12.6\n"), ("uuid", "蓝"), ("uuid", "x" * 129)])
def test_device_identity_labels_bounded_and_typed(evidence, field, value):
    with pytest.raises(m.MemoryPolicyError, match="device identity"):
        replace(evidence[1].device, **{field: value})
