"""Fresh generation-runtime fingerprint, not resource or hardware approval.

Lazy pinned Torch observations plus Linux host/CPU/affinity, relevant bounded
environment and package/own-source identity. Raw environment/boot identifiers
are never exported. No CUDA initialization, weight reads, writes or mutations.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
from dataclasses import dataclass, fields

from qwen_engine.encoder_identity import (
    observe_installed_packages, runtime_metadata_digest, validate_installed_packages,
    worker_source_digest,
)
from qwen_engine.errors import EngineError

RUNTIME_SCHEMA = "qwen_engine.generation-runtime.v1"
MAX_HOST_BYTES = 4 * 1024 * 1024
MAX_ENV_BYTES = 64 * 1024
MAX_ENV_VALUE_BYTES = 4096
MAX_CPUS = 16384
ENV_PREFIXES = ("CUDA_", "CUBLAS_", "CUDNN_", "NVIDIA_", "TORCH_", "PYTORCH_",
                "OMP_", "MKL_", "OPENBLAS_", "KMP_", "TOKENIZERS_", "DIFFUSERS_",
                "DNNL_", "ONEDNN_", "BLIS_", "VECLIB_")
ENV_NAMES = frozenset({"MALLOC_ARENA_MAX", "GLIBC_TUNABLES", "LD_PRELOAD", "LD_LIBRARY_PATH",
                       "PYTHONHASHSEED", "PYTHONMALLOC", "PYTHON_GIL", "ATEN_CPU_CAPABILITY"})
CPU_KEYS = frozenset({"processor", "vendor_id", "cpu family", "model", "model name",
                      "stepping", "microcode", "physical id", "core id", "cpu cores",
                      "siblings", "flags", "cpuid level", "cache_alignment", "address sizes"})
TORCH_KEYS = frozenset({"build_sha256", "cpu_capability", "threads", "interop_threads", "matmul_precision",
    "matmul_fp32", "cudnn_conv_fp32", "fp16_reduction", "bf16_reduction", "flash_sdp",
    "efficient_sdp", "math_sdp", "cudnn_sdp", "deterministic", "deterministic_warn_only",
    "mkldnn", "cudnn", "cudnn_benchmark", "cudnn_deterministic", "allocator_backend",
    "global_fp32", "cudnn_fp32", "mkldnn_matmul_fp32", "mkldnn_conv_fp32", "fp16_accumulation",
    "fp16_split_k", "bf16_split_k", "prefer_grouped_gemm", "math_sdp_reduction",
    "preferred_blas", "preferred_linalg", "cublas_workspace", "cublaslt_workspace", "cuda_initialized"})


class RuntimeIdentityError(EngineError):
    """Runtime identity unavailable, changing or malformed; never a fit pass."""


def _digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


def _hex(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _read_host_file(path, cap):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise RuntimeIdentityError("host_identity_file_not_regular")
            raw = bytearray()
            while len(raw) <= cap:
                block = os.read(fd, min(65536, cap + 1 - len(raw)))
                if not block:
                    break
                raw.extend(block)
            if not raw or len(raw) > cap:
                raise RuntimeIdentityError("host_identity_file_empty_or_oversized")
            return raw.decode("ascii")
        finally:
            os.close(fd)
    except (OSError, UnicodeError):
        raise RuntimeIdentityError("host_identity_file_unavailable") from None


def _cpu_inventory(text):
    rows = []
    for block in text.strip().split("\n\n"):
        row = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            name, value = name.strip(), " ".join(value.split())
            if name not in CPU_KEYS:
                continue
            if name in row or not value or len(value) > 8192:
                raise RuntimeIdentityError("invalid_cpu_inventory")
            row[name] = " ".join(sorted(value.split())) if name == "flags" else value
        if not {"processor", "vendor_id", "model name", "flags", "microcode"} <= set(row):
            raise RuntimeIdentityError("incomplete_cpu_inventory")
        rows.append(row)
    if (not 1 <= len(rows) <= MAX_CPUS
            or any(not r["processor"].isascii() or not r["processor"].isdigit()
                   or len(r["processor"]) > 5 or int(r["processor"]) >= MAX_CPUS for r in rows)
            or len({int(r["processor"]) for r in rows}) != len(rows)):
        raise RuntimeIdentityError("invalid_cpu_inventory_count")
    return sorted(rows, key=lambda row: int(row["processor"]))


def _host_document():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeIdentityError("unreviewed_host_identity_platform")
    boot = _read_host_file("/proc/sys/kernel/random/boot_id", 64).strip()
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", boot) is None:
        raise RuntimeIdentityError("invalid_host_boot_identity")
    affinity = sorted(os.sched_getaffinity(0))
    count = os.cpu_count()
    if (type(count) is not int or not 1 <= count <= MAX_CPUS or not 1 <= len(affinity) <= MAX_CPUS
            or any(type(v) is not int or not 0 <= v < MAX_CPUS for v in affinity)):
        raise RuntimeIdentityError("invalid_host_cpu_availability")
    labels = [platform.release(), platform.version(), platform.machine(),
              *platform.libc_ver(), platform.python_implementation(), platform.python_compiler()]
    if any(type(v) is not str or not v or len(v) > 4096 for v in labels):
        raise RuntimeIdentityError("incomplete_host_platform_identity")
    return dict(platform=labels, boot_sha256=_digest({"boot": boot}), affinity=affinity,
                logical_cpus=count, cpu=_cpu_inventory(_read_host_file("/proc/cpuinfo", MAX_HOST_BYTES)))


def _environment_digest():
    snapshot = dict(os.environ)
    selected = {k: v for k, v in snapshot.items() if k.startswith(ENV_PREFIXES) or k in ENV_NAMES}
    total = 0
    for key, value in selected.items():
        try:
            raw = value.encode("utf-8", errors="strict")
            total += len(key.encode("utf-8", errors="strict")) + len(raw)
        except UnicodeError:
            raise RuntimeIdentityError("invalid_execution_environment_encoding") from None
        if len(key) > 128 or len(raw) > MAX_ENV_VALUE_BYTES or "\0" in value or total > MAX_ENV_BYTES:
            raise RuntimeIdentityError("execution_environment_byte_cap")
    return _digest(selected)


def _torch_document():
    # Private import seam supports CPU unit spies only; no public backend,
    # interpreter, source-root, environment or revision override is exposed.
    import torch
    cuda, cudnn = torch.backends.cuda, torch.backends.cudnn
    initialized = torch.cuda.is_initialized()
    doc = dict(build_sha256=_digest({"build": torch.__config__.show()}),
        cpu_capability=torch.backends.cpu.get_cpu_capability(),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        matmul_precision=torch.get_float32_matmul_precision(), matmul_fp32=cuda.matmul.fp32_precision,
        cudnn_conv_fp32=cudnn.conv.fp32_precision,
        fp16_reduction=cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduction=cuda.matmul.allow_bf16_reduced_precision_reduction,
        flash_sdp=cuda.flash_sdp_enabled(), efficient_sdp=cuda.mem_efficient_sdp_enabled(),
        math_sdp=cuda.math_sdp_enabled(), cudnn_sdp=cuda.cudnn_sdp_enabled(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        mkldnn=torch.backends.mkldnn.enabled, cudnn=cudnn.enabled,
        cudnn_benchmark=cudnn.benchmark, cudnn_deterministic=cudnn.deterministic,
        global_fp32=torch.backends.fp32_precision, cudnn_fp32=cudnn.fp32_precision,
        mkldnn_matmul_fp32=torch.backends.mkldnn.matmul.fp32_precision,
        mkldnn_conv_fp32=torch.backends.mkldnn.conv.fp32_precision,
        fp16_accumulation=cuda.matmul.allow_fp16_accumulation,
        fp16_split_k=cuda.matmul.allow_fp16_reduced_precision_reduction_split_k,
        bf16_split_k=cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
        prefer_grouped_gemm=cuda.matmul.prefer_cublaslt_grouped_gemm,
        math_sdp_reduction=cuda.fp16_bf16_reduction_math_sdp_allowed(),
        preferred_blas=str(cuda.preferred_blas_library()), preferred_linalg=str(cuda.preferred_linalg_library()),
        # These seemingly read-only getters touch the CUDA runtime. Never
        # call them in a CPU observation; unknown must not become guessed 0.
        cuda_initialized=initialized,
        cublas_workspace=cuda.cublas_workspace_size() if initialized else None,
        cublaslt_workspace=cuda.cublaslt_workspace_size() if initialized else None,
        allocator_backend=torch.cuda.get_allocator_backend())
    _validate_torch(doc)
    return doc


def _validate_torch(doc):
    string_keys = {"cpu_capability", "matmul_precision", "matmul_fp32", "cudnn_conv_fp32", "allocator_backend",
                   "global_fp32", "cudnn_fp32", "mkldnn_matmul_fp32", "mkldnn_conv_fp32",
                   "preferred_blas", "preferred_linalg"}
    integer_keys = {"threads", "interop_threads"}
    workspace_keys = {"cublas_workspace", "cublaslt_workspace"}
    if type(doc) is not dict or set(doc) != TORCH_KEYS or not _hex(doc["build_sha256"]):
        raise RuntimeIdentityError("incomplete_torch_execution_identity")
    for key in string_keys:
        if type(doc[key]) is not str or not doc[key] or not doc[key].isascii() or len(doc[key]) > 128:
            raise RuntimeIdentityError("invalid_torch_execution_label")
    if any(type(doc[k]) is not int or not 1 <= doc[k] <= MAX_CPUS for k in integer_keys):
        raise RuntimeIdentityError("invalid_torch_thread_identity")
    if (type(doc["cuda_initialized"]) is not bool
            or any((type(doc[k]) is not int or not 0 <= doc[k] <= 2**40) if doc["cuda_initialized"]
                   else doc[k] is not None for k in workspace_keys)):
        raise RuntimeIdentityError("invalid_torch_workspace_identity")
    if any(type(doc[k]) is not bool for k in TORCH_KEYS - string_keys - integer_keys - workspace_keys - {"build_sha256"}):
        raise RuntimeIdentityError("invalid_torch_execution_flag")


@dataclass(frozen=True)
class GenerationRuntimeIdentity:
    package_metadata_sha256: str
    worker_source_sha256: str
    host_execution_sha256: str
    torch_execution_sha256: str
    environment_sha256: str
    cuda_context_initialized: bool
    schema: str = RUNTIME_SCHEMA

    def __post_init__(self):
        if (type(self.schema) is not str or self.schema != RUNTIME_SCHEMA
                or type(self.cuda_context_initialized) is not bool
                or any(not _hex(getattr(self, f.name)) for f in fields(self)
                       if f.name not in ("schema", "cuda_context_initialized"))):
            raise RuntimeIdentityError("invalid_generation_runtime_identity")

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @property
    def sha256(self):
        return _digest(self.to_dict())


def observe_generation_runtime() -> GenerationRuntimeIdentity:
    """Observe THIS process; changes during observation refuse the snapshot.

    Not package-byte attestation, resource reservation, CUDA execution proof,
    a calibrated peak or authorization to load. Re-observe at gated phase entry.
    The boot identity deliberately invalidates reuse after host reboot.
    """
    try:
        packages = observe_installed_packages()
        validate_installed_packages(packages)
        source = worker_source_digest()
        host, environment = _digest(_host_document()), _environment_digest()
        settings = _torch_document()
        if (_digest(_host_document()) != host or _environment_digest() != environment
                or _torch_document() != settings or worker_source_digest() != source
                or runtime_metadata_digest(observe_installed_packages()) != runtime_metadata_digest(packages)):
            raise RuntimeIdentityError("generation_runtime_changed_during_observation")
        return GenerationRuntimeIdentity(runtime_metadata_digest(packages), source, host,
                                         _digest(settings), environment, settings["cuda_initialized"])
    except RuntimeIdentityError:
        raise
    except Exception:
        raise RuntimeIdentityError("generation_runtime_identity_unavailable") from None
