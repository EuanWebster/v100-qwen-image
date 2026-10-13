"""Pure identity/refusal tests, never CUDA, weights or measured calibration."""
from dataclasses import replace
import os
from pathlib import Path

import pytest

from qwen_engine import runtime_identity as module

CPU = """processor : 0
vendor_id : FixtureCPU
model name : fixture CPU
microcode : 0x10
flags : sse avx
cpu MHz : 1234
"""


def settings():
    values = {k: False for k in module.TORCH_KEYS}
    values.update(build_sha256="a" * 64, cpu_capability="AVX2", threads=2, interop_threads=2,
                  matmul_precision="highest", matmul_fp32="none", cudnn_conv_fp32="tf32",
                  allocator_backend="native")
    values.update(global_fp32="none", cudnn_fp32="none", mkldnn_matmul_fp32="none",
                  mkldnn_conv_fp32="none", preferred_blas="BlasBackend.Cublas",
                  preferred_linalg="LinalgBackend.Default", cublas_workspace=None, cublaslt_workspace=None)
    return values


@pytest.fixture
def fixture(monkeypatch):
    facts = object()
    monkeypatch.setattr(module, "observe_installed_packages", lambda: facts)
    monkeypatch.setattr(module, "validate_installed_packages", lambda value: None)
    monkeypatch.setattr(module, "runtime_metadata_digest", lambda value: "1" * 64)
    monkeypatch.setattr(module, "worker_source_digest", lambda: "2" * 64)
    monkeypatch.setattr(module, "_host_document", lambda: {"trusted_fixture": "host"})
    monkeypatch.setattr(module, "_environment_digest", lambda: "3" * 64)
    monkeypatch.setattr(module, "_torch_document", settings)


def test_observe_canonical_immutable_no_raw_environment(fixture):
    value = module.observe_generation_runtime()
    assert value == module.observe_generation_runtime()
    assert value.sha256 == module.observe_generation_runtime().sha256
    assert set(value.to_dict()) == {"schema", "package_metadata_sha256", "worker_source_sha256",
                                   "host_execution_sha256", "torch_execution_sha256", "environment_sha256",
                                   "cuda_context_initialized"}
    assert value.cuda_context_initialized is False
    copied = value.to_dict()
    copied["host_execution_sha256"] = "f" * 64
    assert value.host_execution_sha256 != copied["host_execution_sha256"]
    assert value.sha256 != replace(value, host_execution_sha256="f" * 64).sha256


@pytest.mark.parametrize("field", ["package_metadata_sha256", "worker_source_sha256", "host_execution_sha256",
    "torch_execution_sha256", "environment_sha256", "schema"])
def test_identity_fields_validated_and_drift_changes_digest(fixture, field):
    value = module.observe_generation_runtime()
    for bad in (None, True, "", "x" * 64, "0" * 63, "F" * 64):
        with pytest.raises(module.RuntimeIdentityError):
            replace(value, **{field: bad})
    if field != "schema":
        assert value.sha256 != replace(value, **{field: "f" * 64}).sha256


@pytest.mark.parametrize("name", ["worker_source_digest", "_host_document", "_environment_digest", "_torch_document",
                                 "runtime_metadata_digest"])
def test_mid_observation_change_refused(fixture, monkeypatch, name):
    def changing():
        yield "a" * 64 if name != "_host_document" and name != "_torch_document" else {"a": 1}
        yield "b" * 64 if name != "_host_document" and name != "_torch_document" else {"a": 2}
    iterator = changing()
    monkeypatch.setattr(module, name, lambda *args: next(iterator))
    with pytest.raises(module.RuntimeIdentityError, match="changed"):
        module.observe_generation_runtime()


def test_package_refusal_precedes_host_source_and_heavy_import(fixture, monkeypatch):
    def refuse(value):
        raise ValueError("private fixture details")
    monkeypatch.setattr(module, "validate_installed_packages", refuse)
    monkeypatch.setattr(module, "_torch_document", lambda: pytest.fail("Torch observation reached"))
    monkeypatch.setattr(module, "_host_document", lambda: pytest.fail("host observation reached"))
    with pytest.raises(module.RuntimeIdentityError, match="unavailable") as raised:
        module.observe_generation_runtime()
    assert "private" not in str(raised.value)


def test_interrupt_preserved(fixture, monkeypatch):
    def interrupt():
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, "_torch_document", interrupt)
    with pytest.raises(KeyboardInterrupt):
        module.observe_generation_runtime()


def test_environment_allowlist_digest_only(monkeypatch):
    monkeypatch.setattr(module.os, "environ", {"API_KEY": "must-never-export", "OMP_NUM_THREADS": "2"})
    original = module._environment_digest()
    module.os.environ["API_KEY"] = "different secret"
    assert module._environment_digest() == original
    module.os.environ["OMP_NUM_THREADS"] = "4"
    assert module._environment_digest() != original
    assert "secret" not in module._environment_digest()
    module.os.environ.pop("OMP_NUM_THREADS")
    assert module._environment_digest() != original


@pytest.mark.parametrize("key", ["CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "CUDNN_FIXTURE",
    "NVIDIA_TF32_OVERRIDE", "TORCH_LOGS", "PYTORCH_ALLOC_CONF", "OMP_PROC_BIND", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "KMP_AFFINITY", "TOKENIZERS_PARALLELISM", "MALLOC_ARENA_MAX",
    "GLIBC_TUNABLES", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHASHSEED", "PYTHONMALLOC", "PYTHON_GIL"])
def test_each_execution_environment_setting_changes_identity(monkeypatch, key):
    monkeypatch.setattr(module.os, "environ", {})
    before = module._environment_digest()
    module.os.environ[key] = "fixture"
    assert module._environment_digest() != before


@pytest.mark.parametrize("environment", [
    {"OMP_NUM_THREADS": "x" * 4097}, {"TORCH_" + "x" * 129: "x"},
    {"CUDA_VISIBLE_DEVICES": "a\0b"}, {"OMP_NUM_THREADS": "\ud800"},
    {"MKL_" + str(i): "x" * 4096 for i in range(17)},
])
def test_environment_bounds_and_unicode_refuse(monkeypatch, environment):
    monkeypatch.setattr(module.os, "environ", environment)
    with pytest.raises(module.RuntimeIdentityError):
        module._environment_digest()


def test_cpu_inventory_normalizes_order_and_omits_volatile_clocks():
    first = module._cpu_inventory(CPU)
    assert first == module._cpu_inventory(CPU.replace("1234", "9999").replace("sse avx", "avx sse"))
    assert first != module._cpu_inventory(CPU.replace("0x10", "0x11"))
    second = CPU.replace("processor : 0", "processor : 1")
    assert module._cpu_inventory(CPU + "\n" + second) == module._cpu_inventory(second + "\n" + CPU)


@pytest.mark.parametrize("text", ["", "processor : 0", CPU + "microcode : 0x10\n", CPU + "\n" + CPU,
    CPU.replace("processor : 0", "processor : -1"), CPU.replace("vendor_id : FixtureCPU\n", ""),
    CPU.replace("processor : 0", "processor : 999999"),
    CPU.replace("processor : 0", "processor : 16384"),
    CPU + "\n" + CPU.replace("processor : 0", "processor : 00"),
    CPU.replace("model name : fixture CPU", "model name : " + "x" * 8193)])
def test_cpu_inventory_invalid_refused(text):
    with pytest.raises(module.RuntimeIdentityError):
        module._cpu_inventory(text)


def test_host_file_bounded_regular_no_symlinks(tmp_path):
    path = tmp_path / "host"
    path.write_bytes(b"abc")
    assert module._read_host_file(path, 3) == "abc"
    with pytest.raises(module.RuntimeIdentityError, match="oversized"):
        module._read_host_file(path, 2)
    link = tmp_path / "link"
    link.symlink_to(path)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for target in (link, fifo, tmp_path / "missing", tmp_path):
        with pytest.raises(module.RuntimeIdentityError):
            module._read_host_file(target, 3)
    for raw in (b"", b"\xff"):
        path.write_bytes(raw)
        with pytest.raises(module.RuntimeIdentityError):
            module._read_host_file(path, 3)


@pytest.mark.parametrize("key", sorted(module.TORCH_KEYS))
def test_each_torch_setting_changes_digest_and_requires_completeness(key):
    doc = settings()
    module._validate_torch(doc)
    original = module._digest(doc)
    doc.pop(key)
    with pytest.raises(module.RuntimeIdentityError):
        module._validate_torch(doc)
    doc = settings()
    value = doc[key]
    doc[key] = not value if type(value) is bool else (value + 1 if type(value) is int else "changed")
    assert module._digest(doc) != original
    doc[key] = "wrong" if value is None else None
    with pytest.raises(module.RuntimeIdentityError):
        module._validate_torch(doc)


def test_torch_typed_flag_thread_string_bounds():
    for key, value in (("threads", True), ("interop_threads", 0), ("threads", module.MAX_CPUS + 1),
                       ("math_sdp", 1), ("matmul_precision", ""), ("cpu_capability", "x" * 129)):
        doc = settings()
        doc[key] = value
        with pytest.raises(module.RuntimeIdentityError):
            module._validate_torch(doc)


def test_cuda_workspace_unknown_is_not_zero_or_gpu_evidence():
    doc = settings()
    module._validate_torch(doc)
    doc["cublas_workspace"] = 0
    with pytest.raises(module.RuntimeIdentityError):
        module._validate_torch(doc)
    doc["cuda_initialized"] = True
    with pytest.raises(module.RuntimeIdentityError):
        module._validate_torch(doc)
    doc["cublaslt_workspace"] = 0
    module._validate_torch(doc)
    doc["cublas_workspace"] = -1
    with pytest.raises(module.RuntimeIdentityError):
        module._validate_torch(doc)


def test_context_flag_is_true_boolean_and_identity_drift(fixture):
    identity = module.observe_generation_runtime()
    for value in (0, 1, None, "false"):
        with pytest.raises(module.RuntimeIdentityError):
            replace(identity, cuda_context_initialized=value)
    assert replace(identity, cuda_context_initialized=True).sha256 != identity.sha256


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(module.platform, "release", lambda: "fixture-release")
    monkeypatch.setattr(module.platform, "version", lambda: "fixture-version")
    monkeypatch.setattr(module.platform, "libc_ver", lambda: ("glibc", "2.fixture"))
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: {0, 2})
    monkeypatch.setattr(module.os, "cpu_count", lambda: 4)
    def read(path, cap):
        return CPU if path == "/proc/cpuinfo" else "01234567-0123-0123-0123-012345678901\n"
    monkeypatch.setattr(module, "_read_host_file", read)


def test_host_identity_hashes_boot_cpu_affinity_and_platform(host, monkeypatch):
    original = module._host_document()
    assert "01234567" not in str(original)
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: {0})
    assert module._digest(original) != module._digest(module._host_document())
    monkeypatch.setattr(module.platform, "release", lambda: "new-release")
    assert original["platform"] != module._host_document()["platform"]


@pytest.mark.parametrize("case", ["platform", "machine", "boot", "count0", "countbool", "affinity0",
                                  "affinitynegative", "affinitybool", "label"])
def test_host_unavailable_or_invalid_refuses(host, monkeypatch, case):
    if case in ("platform", "machine"):
        monkeypatch.setattr(module.platform, "system" if case == "platform" else "machine", lambda: "unknown")
    elif case == "boot":
        monkeypatch.setattr(module, "_read_host_file", lambda *args: "not a boot UUID")
    elif case.startswith("count"):
        monkeypatch.setattr(module.os, "cpu_count", lambda: True if case == "countbool" else 0)
    elif case.startswith("affinity"):
        monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: {
            "affinity0": set(), "affinitynegative": {-1}, "affinitybool": {True}}[case])
    else:
        monkeypatch.setattr(module.platform, "release", lambda: "")
    with pytest.raises(module.RuntimeIdentityError):
        module._host_document()
