# ComfyUI NVIDIA V100 installation

Installed and verified 2026-10-24 after Euan authorized an independent NVIDIA
ComfyUI alongside the existing AMD installation.

## Use

Open **ComfyUI NVIDIA V100** in the application menu, or run:

```sh
comfyui-nvidia-open
```

The launcher starts the user service, waits for readiness, and opens
<http://127.0.0.1:8189>. The service is running now. It is not enabled at login;
the application launcher starts it when requested.

```sh
systemctl --user start comfyui-nvidia.service
systemctl --user stop comfyui-nvidia.service
systemctl --user restart comfyui-nvidia.service
journalctl --user -u comfyui-nvidia.service -f
```

`comfyui-nvidia` runs the server in the foreground instead; stop the service
first if using it, because both use port 8189.

## Isolation and installed versions

| Item | NVIDIA installation |
| --- | --- |
| Source | `/home/euan/ComfyUI-NVIDIA` |
| ComfyUI | 0.42.0, commit `7a28332047b6af638dc6c7f72ef3481a718769fd` |
| Python | Separate `.venv`, CPython 3.13.14 |
| PyTorch | `2.14.1+cu126`, CUDA 12.6, includes `sm_70` |
| torchvision | `0.29.1+cu126` |
| Frontend | `1.57.1` |
| Extra node | Independent ComfyUI-GGUF, commit `edd981b10e107d3b8f58e16c498f2d08f631bc47` |
| Settings, input, output | NVIDIA installation's own `user`, `input`, `output` |
| Temporary files | NVIDIA installation's own `temp` tree |
| Shared files | Existing weights under `/home/euan/ComfyUI/models` |
| Device | V100 UUID `GPU-4a20902a-772a-15a9-6d1e-8b6bc2820751` |

The two source repositories use independent Git objects; source/custom-node
updates in NVIDIA do not update AMD. Only GGUF was added to the new installation;
AMD's additional custom nodes can be installed independently when needed.
Shared-model configuration does not add AMD's custom-node or Python paths.

The existing AMD `.venv` still has `torch==2.14.0+rocm7.2` and
`torchvision==0.29.0+rocm7.2`. Its pre-existing source modifications and deleted
output placeholder were preserved. The binary diff SHA256 was identical before
and after: `ba949ba60ab1720ecb1e24dbdde6ca7ddb2ef38d74f8f9976c5c2c372283e12e`.

## Configuration records

Files in this directory are the installed launcher/service/menu/model-path
configuration. Launchers live in `~/.local/bin`, the service in
`~/.config/systemd/user`, and the menu entry in `~/.local/share/applications`.

`constraints-nvidia.txt` is also installed in the new ComfyUI directory.
`requirements-installed.txt` records the initial 105-package version snapshot
(not artifact hashes), also installed as `requirements-installed-nvidia.txt`.
Keep the CUDA 12.6 Torch constraints when installing/updating dependencies in
this environment. The stock startup message about upgrading to cu130 applies
to newer NVIDIA cards; this V100 installation intentionally uses cu126.

The launch profile selects the V100 UUID, PyTorch attention, FP16 UNet and
FP32 VAE. Comfy Kitchen's newer CUDA/Triton optimizations are disabled by
upstream capability selection; the normal eager/PyTorch path works here.

## Actual verification

- Dependency consistency passed (`uv pip check`).
- Fresh CUDA FP16 512×512 matrix multiplication passed in the NVIDIA venv.
- `/system_stats` identified the V100 and `2.14.1+cu126`; frontend HTTP returned 200.
- Shared checkpoint choices and the `UnetLoaderGGUF` node were visible through the API.
- `smoke-sdxl.json` executed using the existing Juggernaut-XL v9 checkpoint:
  512×512, seed 42, 12 Euler steps, CFG 6.5. API status was success; ComfyUI
  reported 5.41 seconds for this particular validation run.
- Output visually inspected: a lighthouse/coast image at
  `/home/euan/ComfyUI-NVIDIA/output/validation/v100-sdxl-smoke_00001_.png`.
- Test model cache was unloaded through this instance's `/free` endpoint.

This confirms the SDXL pipeline on the V100; Qwen-Image-2.1, other custom-node
workflows and higher resolutions have not been validated in this installation.
