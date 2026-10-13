"""Pure exact-match generation memory policy; no hardware probes or loaders.

Caller-supplied resource/calibration records are trusted measurement evidence,
not cryptographic proof or automatic approval of hardware/code. This layer
checks completeness/freshness/identity and conservative arithmetic only.
Legacy bench v1 observations lack device/runtime identity and are ineligible.
There is no expert bypass, interpolation, automatic mode switch or CLI wiring.
Real host-budget discovery remains fail-closed in mem_preflight.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from qwen_engine.config import DIM_MAX, DIM_MIN, DIM_MULTIPLE
from qwen_engine.errors import MemoryGuardError
from qwen_engine.mem_preflight import MemorySnapshot

POLICY_SCHEMA = "qwen_engine.memory_policy.v1"
MAX_OBSERVATIONS = 4096
_PHASES = frozenset({"load", "encode", "denoise-step", "decode", "save"})


class MemoryPolicyError(MemoryGuardError):
    """Malformed evidence or a policy refusal; never a permission to load."""


def _integer(value, *, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


def _number(value, *, positive=True):
    try:
        return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)
    except (OverflowError, ValueError):
        return False


def _digest(value, length):
    return type(value) is str and len(value) == length and re.fullmatch("[0-9a-f]+", value) is not None


def _label(value):
    return (type(value) is str and 1 <= len(value) <= 128
            and value.isascii() and value == value.strip()
            and all(33 <= ord(c) <= 126 for c in value))


@dataclass(frozen=True)
class PolicyDescriptor:
    phase: str
    width: int
    height: int
    steps: int
    prompt_tokens: int | None
    negative_prompt_tokens: int | None
    cfg_active: bool
    backend: str
    quant_format: str
    residency: str
    vae_tiled_decode: bool
    encoder_dtype: str
    execution_dtype: str
    use_kv_cache: bool
    model_revision: str
    artifact_set_sha256: str
    runtime_identity_sha256: str
    step_index: int | None = None
    embedding_cache_hit: bool = False
    schema: str = POLICY_SCHEMA

    def __post_init__(self):
        if (any(type(v) is not str for v in (self.schema, self.phase, self.backend, self.residency,
                                           self.encoder_dtype, self.execution_dtype))
                or self.schema != POLICY_SCHEMA or self.phase not in _PHASES
                or any(not _integer(v, positive=True) or not DIM_MIN <= v <= DIM_MAX or v % DIM_MULTIPLE
                       for v in (self.width, self.height))
                or not _integer(self.steps, positive=True) or self.steps > 1000
                or any(type(v) is not bool for v in (self.cfg_active, self.vae_tiled_decode,
                                                    self.use_kv_cache, self.embedding_cache_hit))
                or self.backend not in ("sdpa", "efficient", "eager")
                or not _label(self.quant_format) or self.residency not in ("resident", "streamed", "offload")
                or self.encoder_dtype not in ("fp16", "bf16", "fp32") or self.execution_dtype != "fp16"
                or not _digest(self.model_revision, 40) or not _digest(self.artifact_set_sha256, 64)
                or not _digest(self.runtime_identity_sha256, 64)):
            raise MemoryPolicyError("invalid or unresolved generation policy descriptor")
        if any(v is not None and (not _integer(v, positive=True) or v > 1_000_000)
               for v in (self.prompt_tokens, self.negative_prompt_tokens)):
            raise MemoryPolicyError("prompt token lengths must be measured positive integers or unknown")
        if not self.cfg_active and self.negative_prompt_tokens is not None:
            raise MemoryPolicyError("inactive CFG must not claim a negative encoding length")
        if self.phase == "denoise-step":
            if not _integer(self.step_index) or self.step_index >= self.steps:
                raise MemoryPolicyError("denoise-step needs an exact in-range step index")
        elif self.step_index is not None:
            raise MemoryPolicyError("step index applies only to denoise-step")

    @property
    def latent_tokens(self):
        return self.width // 16 * (self.height // 16)


@dataclass(frozen=True)
class DeviceIdentity:
    uuid: str
    architecture: str
    driver_version: str
    cuda_build: str

    def __post_init__(self):
        if not all(_label(v) for v in (self.uuid, self.architecture, self.driver_version, self.cuda_build)):
            raise MemoryPolicyError("invalid device identity")


@dataclass(frozen=True)
class ResourceEvidence:
    captured_at: float
    host: MemorySnapshot
    device: DeviceIdentity | None
    cuda_kernel_verified: bool
    gpu_total_bytes: int | None
    gpu_available_bytes: int | None

    def __post_init__(self):
        if (not _number(self.captured_at) or type(self.host) is not MemorySnapshot
                or self.device is not None and type(self.device) is not DeviceIdentity
                or type(self.cuda_kernel_verified) is not bool
                or any(v is not None and not _integer(v) for v in (self.gpu_total_bytes, self.gpu_available_bytes))
                or self.gpu_total_bytes == 0
                or self.gpu_total_bytes is not None and self.gpu_available_bytes is not None
                and self.gpu_available_bytes > self.gpu_total_bytes):
            raise MemoryPolicyError("invalid current resource evidence")
        if self.device is None and (self.cuda_kernel_verified or self.gpu_total_bytes is not None or self.gpu_available_bytes is not None):
            raise MemoryPolicyError("absent device cannot carry verified CUDA or memory counters")


@dataclass(frozen=True)
class CalibrationEvidence:
    descriptor: PolicyDescriptor
    device: DeviceIdentity
    observed_at: float
    peak_gpu_reserved_bytes: int | None
    peak_host_rss_bytes: int | None
    peak_gpu_allocated_bytes: int | None = None
    # Device-wide peak includes context and non-Torch CUDA allocations. Torch
    # reserved bytes alone are not a complete GPU-footprint measurement.
    peak_gpu_used_bytes: int | None = None
    # Includes the owned job's children (not just the parent process RSS).
    # Summed RSS may double-count shared pages; that is conservative here.
    peak_host_tree_bytes: int | None = None

    def __post_init__(self):
        if (type(self.descriptor) is not PolicyDescriptor or type(self.device) is not DeviceIdentity
                or not _number(self.observed_at)
                or any(v is not None and not _integer(v) for v in (
                    self.peak_gpu_reserved_bytes, self.peak_host_rss_bytes,
                    self.peak_gpu_allocated_bytes, self.peak_gpu_used_bytes, self.peak_host_tree_bytes))
                or self.peak_host_rss_bytes == 0
                or self.peak_gpu_reserved_bytes is not None and self.peak_gpu_allocated_bytes is not None
                and self.peak_gpu_allocated_bytes > self.peak_gpu_reserved_bytes
                or self.peak_gpu_reserved_bytes is not None and self.peak_gpu_used_bytes is not None
                and self.peak_gpu_reserved_bytes > self.peak_gpu_used_bytes
                or self.peak_host_rss_bytes is not None and self.peak_host_tree_bytes is not None
                and self.peak_host_rss_bytes > self.peak_host_tree_bytes):
            raise MemoryPolicyError("invalid calibrated memory evidence")


@dataclass(frozen=True)
class MemoryDecision:
    descriptor: PolicyDescriptor
    allowed: bool
    reasons: tuple[str, ...]
    matching_samples: int
    required_host_bytes: int | None
    required_gpu_bytes: int | None
    host_budget_bytes: int | None
    gpu_budget_bytes: int | None


def assess_memory(descriptor: PolicyDescriptor, resources: ResourceEvidence,
                  observations: tuple[CalibrationEvidence, ...], *, host_headroom_bytes: int,
                  gpu_headroom_bytes: int, max_resource_age_s: float,
                  max_calibration_age_s: float, now: float | None = None) -> MemoryDecision:
    """Exact measured peaks + explicit headroom against fresh applicable budget.

    Full device and owned-process-tree peaks are compared to availability, not subtracted
    from guesses about residency/baselines. This can refuse unnecessarily but
    never relaxes a bound through interpolation or averages. Worst complete
    matching sample wins. A matching incomplete sample blocks too. This is
    a conservative prerequisite, NOT proof a future allocation succeeds.
    """
    if (type(descriptor) is not PolicyDescriptor or type(resources) is not ResourceEvidence
            or type(observations) is not tuple or len(observations) > MAX_OBSERVATIONS
            or any(type(o) is not CalibrationEvidence for o in observations)
            or not _integer(host_headroom_bytes) or not _integer(gpu_headroom_bytes)
            or not _number(max_resource_age_s) or not _number(max_calibration_age_s)):
        raise MemoryPolicyError("complete typed bounded evidence and explicit headroom/age limits required")
    if now is None:
        try:
            now = time.time()
        except Exception:
            now = None
    reasons = []
    host_budget, _ = resources.host.budget()
    gpu_budget = resources.gpu_available_bytes
    if not _number(now):
        reasons.append("current_time_unknown")
    elif not 0 <= now - resources.captured_at <= max_resource_age_s:
        reasons.append("resource_snapshot_stale_or_future")
    if resources.device is None or not resources.cuda_kernel_verified:
        reasons.append("cuda_execution_unverified")
    if resources.device is not None and (resources.device.architecture != "sm_70" or resources.device.cuda_build != "12.6"):
        reasons.append("device_or_cuda_profile_unreviewed")
    if host_budget is None:
        reasons.append("complete_host_budget_unknown")
    if resources.gpu_total_bytes is None or gpu_budget is None:
        reasons.append("gpu_budget_unknown")
    if descriptor.prompt_tokens is None or descriptor.cfg_active and descriptor.negative_prompt_tokens is None:
        reasons.append("conditioning_lengths_unknown")
    matching = tuple(o for o in observations if o.descriptor == descriptor and o.device == resources.device)
    if not matching:
        reasons.append("exact_calibration_missing")
    elif _number(now) and any(not 0 <= now - o.observed_at <= max_calibration_age_s for o in matching):
        reasons.append("matching_calibration_stale_or_future")
    host_required = gpu_required = None
    if matching:
        if any(o.peak_host_rss_bytes is None or o.peak_host_tree_bytes is None
               or o.peak_gpu_reserved_bytes is None or o.peak_gpu_used_bytes is None for o in matching):
            reasons.append("matching_peak_unmeasured")
        else:
            host_required = max(o.peak_host_tree_bytes for o in matching) + host_headroom_bytes
            gpu_required = max(o.peak_gpu_used_bytes for o in matching) + gpu_headroom_bytes
            if host_budget is not None and host_required > host_budget:
                reasons.append("host_budget_exceeded")
            if gpu_budget is not None and gpu_required > gpu_budget:
                reasons.append("gpu_budget_exceeded")
    return MemoryDecision(descriptor, not reasons, tuple(reasons), len(matching),
                          host_required, gpu_required, host_budget, gpu_budget)


def require_memory(*args, **kwargs) -> MemoryDecision:
    decision = assess_memory(*args, **kwargs)
    if not decision.allowed:
        raise MemoryPolicyError("generation memory policy refused: " + ", ".join(decision.reasons))
    return decision
