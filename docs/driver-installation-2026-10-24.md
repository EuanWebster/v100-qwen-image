# NVIDIA driver installation — 2026-10-24

Euan authorized disabling Nouveau and installing the correct NVIDIA driver
after the detection checks in `hardware-check-2026-10-24.md`.

## Completed

Installed Ubuntu's proprietary R580 server/compute driver, version
`580.178.04-0ubuntu0.26.04.1`, with prebuilt signed modules for the running
`7.0.0-38-generic` kernel. Secure Boot remains enabled. The module metadata
identifies the license as `NVIDIA` and signer as
`Canonical Ltd. Kernel Module Signing`; the UVM module has the same version
and signer. No custom signing key or MOK enrollment was needed for installation.
Actual module loading remains a post-reboot check.

The compute-only package set avoids installing an NVIDIA desktop/display
stack over the existing AMD setup. The installed branch metapackage tracks
R580 kernel-module updates; security updates within the branch are not held.

After refreshing APT metadata and inspecting a simulated installation, ran:

```sh
sudo env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l \
  apt-get install -y --no-install-recommends --no-remove \
  linux-modules-nvidia-580-server-generic \
  nvidia-headless-no-dkms-580-server \
  nvidia-utils-580-server
```

APT installed 12 new packages, upgraded none and removed none. This includes
`linux-modules-nvidia-580-server-7.0.0-38-generic` version `7.0.0-38.38+2`,
the signed module objects, compute libraries and `nvidia-smi` utility.
Package hooks regenerated the current initramfs and GRUB configuration.

## Nouveau disabled for subsequent boot

The driver package owns `/lib/modprobe.d/nvidia-graphics-drivers.conf`
(also accessible through `/usr/lib/modprobe.d` on this merged-/usr system).
It includes:

```text
blacklist nouveau
blacklist lbm-nouveau
blacklist nova
blacklist nova_core
alias nouveau off
alias lbm-nouveau off
alias nova off
alias nova_core off
```

Verified the effective blacklist through `modprobe -c` and extracted this
configuration from `/boot/initrd.img-7.0.0-38-generic` using `lsinitrd`.
No duplicate local blacklist file or AMD module configuration change was needed.

## Verified limits and remaining activation

- `apt-get check` and `dpkg --audit` passed.
- `modprobe -n -v nvidia` resolves the signed proprietary module at
  `/lib/modules/7.0.0-38-generic/kernel/nvidia-580srv/nvidia.ko`.
- The display manager remained active. The `radiance` and `openwebui`
  containers remained running; no existing service was deliberately restarted.
- Nouveau is still bound to `0000:0a:00.0` in the current boot. It was not
  forcibly unloaded from the active graphical session, and no reboot occurred.
- **Reboot is required to activate the driver switch.** After reboot verify
  `lspci -nnk -s 0a:00.0`, `nvidia-smi`, and absence of loaded Nouveau; inspect
  temperature/ECC and run the fresh-worker CUDA/NVML diagnostic next.
- No CUDA success, model inference, load/thermal acceptance, PCIe throughput,
  container passthrough or Secure Boot module-loading success is claimed yet.
- Existing project `sandbox.yaml` changes remain untouched.

## Sources used to select the installation

- [NVIDIA legacy GPU support: R580 is the last branch for Volta](https://nvidia.custhelp.com/app/answers/detail/a_id/3142/kw/ubuntu).
- [Ubuntu driver installation: compute/server variants and precompiled signed modules](https://ubuntu.com/server/docs/nvidia-drivers-installation/).
- Local APT dependency metadata, module metadata and installed package records
  were checked on 2026-10-24 for this host and kernel.
