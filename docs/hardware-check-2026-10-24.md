# V100 installation check — 2026-10-24

Read-only host diagnostics following Euan's report that the V100 was installed.

## Result

The card is detected on PCIe, but CUDA computation cannot yet be tested:
the device is bound to Nouveau and the NVIDIA compute driver is absent.
Enumeration and a VRAM report are not compute, memory-integrity or thermal
acceptance tests. No CUDA kernel or model inference was run.

## Direct observations

| Check | Observed result |
| --- | --- |
| PCI address | `0000:0a:00.0` |
| PCI identity | `10de:1db1`, subsystem `10de:1212` |
| Reported model | `NVIDIA GV100GL [Tesla V100 SXM2 16GB]` |
| Bound kernel driver | `nouveau` |
| Nouveau VRAM report | `16384 MiB` |
| Current PCIe link | `8.0 GT/s`, width `4`: Gen3 x4 |
| Device maximum link | `8.0 GT/s`, width `16`: Gen3 x16 |
| Kernel bandwidth report | `31.504 Gb/s` available through upstream `0000:09:00.0` |
| Host | Ubuntu 26.04.1 LTS, kernel `7.0.0-38-generic` |
| Secure Boot | Enabled |
| `nvidia-smi` | Not found in the command path |
| NVIDIA driver packages | Neither `nvidia-driver-580` nor `nvidia-driver-580-server` installed |
| Available package candidate | Both list `580.178.04-0ubuntu0.26.04.1` in the local APT cache |
| Project container | `proj-v100-qwen-image` exists in `Created` state; not running |
| Project venv on host | `.venv/bin/python` points to `/usr/bin/python3.13`, which is unavailable here; invocation fails |

The SXM2 model string is the reported PCI identity. The physical module/carrier
construction was not inspected. The negotiated Gen3 x4 link confirms the earlier
bandwidth expectation for the M.2 connection, but no transfer benchmark was run.

## Boot-log findings

Nouveau initializes GV100 and reports the VRAM above. Its log also contains
`pmu: firmware unavailable`, VBIOS/display-table messages, and graphics traps
(`RT_HEIGHT_OVERRUN`) associated with `gst-plugin-scan`. These do not establish
a defective card; assess the hardware with the supported NVIDIA driver before
drawing a hardware-health conclusion.

The current boot's kernel-log search for `AER:`, `PCIe Bus Error`, `DPC:`,
`NVRM` and `Xid` returned no entries. Absence of these entries is not a stress
test, and NVIDIA-specific errors cannot be expected without its driver loaded.

## Next validation

1. Install a suitable proprietary NVIDIA driver using a Secure-Boot-compatible
   signing path. Package candidates above are availability observations, not
   evidence that installation/module loading has succeeded on this kernel.
2. After the required reboot/module transition, check `nvidia-smi`, driver
   binding, temperature, device memory and ECC status.
3. Verify container GPU access and the pinned CUDA 12.6 environment. The host
   cannot currently execute the sandbox-created project venv.
4. Run the existing fresh-worker FP16 matrix multiplication/NVML diagnostic
   described in `docs/hardware-probe.md`, then separately measure PCIe transfer
   throughput and sustained-load health.

No driver installation, driver unload, container start, host configuration
change, service interruption, or reboot was performed. Existing `sandbox.yaml`
changes were preserved.
