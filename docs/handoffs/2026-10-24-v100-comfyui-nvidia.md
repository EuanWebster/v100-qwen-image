# Handoff — V100 and independent NVIDIA ComfyUI

Updated 2026-10-24 by Codex. Euan requested a written handoff after the setup
below completed. Start with `AGENTS.md` and `memctl brief v100-qwen-image`.

## Current outcome

The V100 is operational on the host. An independent NVIDIA ComfyUI installation
is running and has generated a real SDXL image. The existing AMD ComfyUI was
preserved. The hand-rolled Qwen engine and Unsloth NVIDIA setup are separate,
unfinished work; ComfyUI success does not establish their readiness.

## Launch and manage NVIDIA ComfyUI

- URL: **http://127.0.0.1:8189**.
- Application menu: **ComfyUI NVIDIA V100**.
- Command: `comfyui-nvidia-open` (starts service, waits for readiness, opens browser).
- User service: `comfyui-nvidia.service`; verified active while writing this handoff.
- The service is **not enabled at login**; the launcher starts it on demand.
- Installation: `/home/euan/ComfyUI-NVIDIA`.
- Own Python environment, custom nodes, user settings, input, output and temp trees.
- Models are shared from `/home/euan/ComfyUI/models` through the new installation's
  `extra_model_paths.yaml`; no model-weight copies or downloads were needed.

```sh
systemctl --user start comfyui-nvidia.service
systemctl --user stop comfyui-nvidia.service
systemctl --user restart comfyui-nvidia.service
journalctl --user -u comfyui-nvidia.service -f
```

`comfyui-nvidia` runs the server in the foreground. Stop the service first when
using that command, since both bind port 8189.

## Installed versions and policy

- ComfyUI 0.42.0, source commit `7a28332047b6af638dc6c7f72ef3481a718769fd`,
  matching the AMD checkout's committed version, with independent Git objects.
- Python 3.13.14 in the new `.venv`.
- `torch==2.14.1+cu126`, `torchvision==0.29.1+cu126`; actual `sm_70` CUDA execution
  verified in this environment. Keep CUDA 12.6 pins for Volta.
- Frontend 1.57.1; installed dependency consistency checked for 105 packages.
- Independent ComfyUI-GGUF at `edd981b10e107d3b8f58e16c498f2d08f631bc47`.
  Other extra AMD custom nodes were not copied into this installation.
- Launcher selects V100 UUID `GPU-4a20902a-772a-15a9-6d1e-8b6bc2820751`,
  PyTorch attention, FP16 UNet and FP32 VAE.
- Upstream disables newer Comfy Kitchen CUDA/Triton optimization paths here;
  eager/PyTorch execution works. A generic startup message suggesting cu130 is
  not a reason to replace this Volta-compatible cu126 environment.

All deployed launcher/service/menu/model-path files, the CUDA constraints,
installed version snapshot and API smoke graph are recorded in
[`deployment/comfyui-nvidia/`](../../deployment/comfyui-nvidia/README.md),
commit **`9a028d7`**. The same README records exact deployed paths and commands.

## Validation completed

- Small host CUDA driver FP16 and FP32 addition tests, 4096 elements each,
  passed; context/allocation cleanup confirmed.
- New ComfyUI venv passed a real CUDA FP16 512×512 matrix multiplication.
- Frontend returned HTTP 200; `/system_stats` identified the V100 and cu126
  PyTorch; checkpoint menus and `UnetLoaderGGUF` were available.
- Actual SDXL image generation passed using existing Juggernaut-XL v9:
  512×512, seed 42, 12 Euler steps, CFG 6.5. ComfyUI reported 5.41 seconds for
  that individual validation run; this is not a general performance estimate.
- Output was visually inspected and depicts the requested lighthouse/coast:
  `/home/euan/ComfyUI-NVIDIA/output/validation/v100-sdxl-smoke_00001_.png`.
- Own test model cache was unloaded via `/free`; the server remains running.

Qwen-Image-2.1, higher resolutions and other custom-node workflows have not
been validated in the new ComfyUI environment.

## Hardware and AMD preservation

The card reports **Tesla V100-SXM2-16GB**, PCI `0000:0a:00.0`, compute
capability 7.0, 16,384 MiB VRAM and negotiated **PCIe Gen3 x4** through the
M.2 connection. Proprietary **580.178.04** driver is loaded, Nouveau is absent,
and Secure Boot remains enabled using Canonical-signed modules. Zero volatile
ECC errors were observed. Brief temperatures ranged roughly 35–48°C during
these checks; sustained-load/thermal and bandwidth acceptance remain pending.

Hardware records:

- [`docs/hardware-check-2026-10-24.md`](../hardware-check-2026-10-24.md), `de08e11`.
- [`docs/driver-installation-2026-10-24.md`](../driver-installation-2026-10-24.md), `c23ccf9`.
- [`docs/hardware-validation-2026-10-24.md`](../hardware-validation-2026-10-24.md), `597b7ca`.

Existing AMD ComfyUI remains `/home/euan/ComfyUI`, with
`torch==2.14.0+rocm7.2` and `torchvision==0.29.0+rocm7.2`.
Its existing three modified 3D source files and deleted output placeholder
were preserved; its binary Git diff hash matched before and after setup:
`ba949ba60ab1720ecb1e24dbdde6ca7ddb2ef38d74f8f9976c5c2c372283e12e`.
No AMD launcher, service or dependencies were changed.

The project worktree's sole pre-existing dirty file is `sandbox.yaml`, changing
host port 3082 to 3081. Preserve it and exclude it from unrelated commits.

## Remaining work and user decisions

1. **Hand-rolled project:** The project container was last observed stopped in
   `Created` state, with no NVIDIA Toolkit/CDI/GPU passthrough configured.
   Its locked CUDA packages already exist under `.review-envs` but their
   interpreter paths belong to the sandbox. The main `.venv` is intentionally
   lightweight and Torch-free. Do not install ML packages there.
2. **Memory:** The project container's 16 GiB RAM cap is below its 17.53 GB
   full text-encoder weights alone. Establish a complete viable budget before
   attempting that load. Host ComfyUI runs outside this cap.
3. **Engine acceptance:** Application implementation through `2aa9116` and
   prior offline tests are documented in the Oct-14 shared-memory handoff and
   existing review docs. The project default loader/live generation remain
   disabled pending actual model numerics, VRAM/peak evidence and calibration.
   The standalone driver/ComfyUI checks do not replace its pinned fresh-worker
   hardware/NVML and model acceptance checks.
4. **Unsloth:** Existing Studio remains an AMD ROCm installation; NVIDIA
   runtime/backend support has not been set up. Verify the exact Volta-compatible
   inference/training path before changing anything. Preserve its AMD setup.
5. **Cooling:** User says cooling/power are handled. Their fan connects to
   **FAN1, explicitly not a motherboard header**. Exact connector/controller
   remains unidentified. GPU temperature is readable; NVIDIA fan speed is N/A.
   No fan settings were changed. Do not map motherboard fan1/PWM1 to this fan.
6. **TUI:** User canceled a requested temperature TUI because they already
   have one. No TUI files were created.

Requirements remain: V100 mandatory, maximum quality first, optimize speed
where possible, text-to-image initially, all discussed image genres potentially
needed, 1024 default/2K optional, and preserve existing services where possible.

No background implementation agent or unsupervised diagnostic is running.
The NVIDIA ComfyUI user service is the intentional running application.
