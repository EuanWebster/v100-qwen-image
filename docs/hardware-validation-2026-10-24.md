# Post-reboot V100 validation — 2026-10-24

Euan rebooted after the signed R580 installation recorded in
`driver-installation-2026-10-24.md`. The driver transition succeeded and small
real CUDA computations passed on the host.

## Verified on this boot

| Check | Result |
| --- | --- |
| Device | Tesla V100-SXM2-16GB, compute capability 7.0 |
| PCI address | `0000:0a:00.0` |
| UUID | `GPU-4a20902a-772a-15a9-6d1e-8b6bc2820751` |
| Bound driver | `nvidia`, version `580.178.04` |
| Nouveau | Not loaded; no Nouveau messages in current boot log |
| Secure Boot | Still enabled; signed NVIDIA modules loaded successfully |
| Reported memory | 16,384 MiB total; 0 MiB used before/after the test |
| PCIe link | Gen3 x4, confirmed by NVML and sysfs |
| ECC | Enabled; corrected and uncorrected volatile totals both zero before/after |
| Temperature | 47°C before, 48°C after; brief observations, not a thermal load test |
| Power after test | Approximately 24.4 W |
| Compute processes after test | None reported by `nvidia-smi` |

Current boot log search found no `Xid`, `AER:`, `PCIe Bus Error`, `nouveau`,
or `module verification failed` entries. Normal proprietary/out-of-tree module
taint messages were present; these are not signature-rejection messages.

## Actual CUDA execution

No host PyTorch, CUDA toolkit or compiler was present. The standalone
`tools/cuda_driver_smoke_check.py` uses Python's standard library and the
installed `libcuda.so.1`, selects the device by explicit PCI address, checks
its V100 identity/capability, JIT-compiles sm_70 PTX, and launches two kernels.

Executed successfully:

```sh
CUDA_CACHE_DISABLE=1 timeout 30s python3 tools/cuda_driver_smoke_check.py \
  --pci-bus-id 0000:0a:00.0
```

- FP32 vector addition: **4096/4096 elements matched** the CPU expectation.
- FP16 vector addition: **4096/4096 elements matched** the CPU expectation.
- Inputs include positive/negative values and exactly representable fractions.
- Output buffers are poisoned before launch. Kernel completion is synchronized
  before device-to-host copy and exact output comparison.
- The private CUDA context was destroyed after the checks, releasing its
  allocations/modules. No retained compute process or GPU memory was observed.

This establishes basic CUDA execution, JIT support, FP16/FP32 arithmetic and
small host/device transfers. It does **not** establish PyTorch/cu126 behavior,
tensor-core matrix multiplication, full VRAM integrity, sustained-load cooling,
transfer throughput, model quality, or Qwen memory-fit acceptance.

## Remaining project work

The project container is still in `Created` state and was not started. Its
existing project venv cannot execute on this host because its Python 3.13
interpreter path is unavailable here. Verify container GPU access and run the
existing pinned PyTorch/NVML fresh-worker probe next; those remain distinct
from this standalone host check. Encoder RAM/scope constraints and disabled
generation defaults remain unchanged.

The display manager and `radiance`, `searxng`, and `openwebui` containers stayed
active during the checks. No additional packages, GPU settings or host services
were changed, and the existing `sandbox.yaml` edit was preserved.

API references used for the standalone diagnostic:

- [CUDA driver kernel launch API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
- [CUDA driver memory and PCI device APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
- [PTX instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
