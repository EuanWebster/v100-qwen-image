# Dependencies (task M0.1)

Runtime dependency identity for the Qwen-Image-2.1 stack. Everything below was
resolved/installed/verified **inside a task-specific temporary ML audit venv**
(`/tmp/v100_m0_ml/`, not persisted — recreate it with the commands in §4).
The repository environments stay torch-free: `/work/.venv` and the fresh
`run_cpu_tests.sh` environments contain no torch/diffusers/transformers, and
the production package still imports nothing heavy (§6). No driver, GPU, or
hardware change was made; nothing here proves GPU execution (§5).

## 1. Pins

| Package | Pinned identity | Where it installs from |
|---|---|---|
| torch | `2.14.1+cu126` (distribution name/version exactly that; build against CUDA **12.6**) | official cu126 wheel index `https://download.pytorch.org/whl/cu126` — the `+cu126` local label exists only there, so plain-PyPI or cu13 torch can never satisfy this pin |
| diffusers | commit `c6df88a511a98740646ee55577b590c9852650ce` (tree `d77a3e3882af758d606ca22b53f63be8764e643a`, commit date 2026-10-06T19:04:08Z, "[docs] Jobs guide (#14926)"), reported version `0.41.0.dev0` | git URL in `requirements.in` (no release wheel exists for this commit) |
| transformers | `5.17.0` PyPI wheel. The GitHub tag `v5.17.0` is an **annotated tag object** `595ff117c8412ec01262084058276e6b27a857d9`, which dereferences to commit `856157a2f3e9594954310df18fdccc31ffddebe9` — that commit (never the tag object) is the source identity; `git describe` on it prints `v5.17.0` | PyPI |
| accelerate | `1.15.0` | PyPI (diffusers `torch`/`gguf` extras dep: device_map/offload) |
| safetensors | `0.8.0` | PyPI (diffusers hard dep) |
| pillow | `12.3.0` | PyPI (diffusers hard dep) |
| gguf | `0.19.0` (floor `>=0.10.0`, which `GGUFQuantizer.validate_environment` enforces) | PyPI — native GGUF quant types live in diffusers (`quantizers/gguf/utils.py`); `gguf` provides `GGUFReader`/`GGML_QUANT_SIZES` |
| numpy | `2.4.6` — deliberate uniformity pin: numpy 2.5.x is `requires-python >= 3.12` (verified from PyPI metadata), so an unpinned resolve would differ between the two supported interpreters | PyPI |

Transitive pins (nvidia-*-cu12 set, triton 3.8.0, tokenizers 0.23.2,
huggingface-hub 1.33.0, cuda-bindings 12.9.7, …) are in
[`requirements.lock`](../requirements.lock) with wheel hashes. `torchvision`
is **not** required for the text-to-image import path: `transformers`'
`Qwen3VLProcessor` and the pipeline module import cleanly without it (verified
2026-10-07); if a future load path needs it, it joins via an explicit
requirements change, not silently.

No `requirements.in`/lock change touches the **developer-only CPU
environment**: `pyproject.toml` + `uv.lock` remain the pinned pytest-only env
(`pytest==9.1.1`); the ML stack is installed *only* from `requirements.lock`
by explicit operator command.

## 2. Verified installed facts (2026-10-07, no GPU present)

Two fresh venvs were built with `uv pip sync` from the committed lock (hashes
enforced) and checked:

| env | python | torch | `torch.version.cuda` | distribution identity |
|---|---|---|---|---|
| `/tmp/v100_m0_ml/venv-sync311` | 3.11.15 | 2.14.1+cu126 | 12.6 | `torch 2.14.1+cu126` (importlib.metadata) |
| `/tmp/v100_m0_ml/venv-sync313` | 3.13.5 | 2.14.1+cu126 | 12.6 | same |

`torch.cuda.get_arch_list()` — the installed 2.14.1 function gates on
`_is_compiled()` (**not** `is_available()`, matching the noted upstream change;
so it answers purely from build flags with no card installed). Verified
installed source (`torch/cuda/__init__.py`):

```python
def get_arch_list() -> list[str]:
    if not _is_compiled():
        return []
    arch_flags = torch._C._cuda_getArchFlags()
    if arch_flags is None:
        return []
    return arch_flags.split()
```

Actual recorded result **on both interpreters** (no CUDA device, no driver):

```
['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
```

**`sm_70` (Tesla V100) is in the list** — this is a wheel build-flag fact, not
GPU evidence. Pipeline import smoke also passed in both synced envs:
`QwenImage21Pipeline` ctor binds `Qwen3VLForConditionalGeneration` /
`Qwen3VLProcessor` from transformers 5.17.0 without `torchvision`.

## 3. Index / interpreter assumptions

* Target: Linux **x86_64**, CPython **3.11 and 3.13** (cp311/cp313
  manylinux_2_28_x86_64 wheels). No other platform/marker is locked; installing
  elsewhere will fail hash resolution rather than guess.
* The resolve was produced with `--python-platform x86_64-manylinux_2_28`.
  Some transitive entries (certifi/urllib3/…) resolve to the older copies the
  pytorch extra index serves — that is exactly what the documented command
  reproduces; do not "modernize" without re-running §4.
* The two per-interpreter compiles agree on every `package==version` (numpy
  forced equal via the `numpy==2.4.6` pin); the only diff is one markupsafe
  wheel hash, and **both** hashes are listed under markupsafe so one lock
  installs on either interpreter.

## 4. Reproduce / verify

```bash
# regenerate (both must yield identical package==version sets; the merge note
# in requirements.lock covers the markupsafe hash):
uv pip compile requirements.in --python-version 3.11 \
    --python-platform x86_64-manylinux_2_28 --generate-hashes -o /tmp/lock311.txt
uv pip compile requirements.in --python-version 3.13 \
    --python-platform x86_64-manylinux_2_28 --generate-hashes -o /tmp/lock313.txt
diff <(grep -E '^[A-Za-z0-9_.-]+(==|@)' /tmp/lock311.txt | sort) \
     <(grep -E '^[A-Za-z0-9_.-]+(==|@)' /tmp/lock313.txt | sort)

# audit venv (temporary; NEVER /work/.venv):
uv venv --python 3.13 /tmp/v100_m0_ml/venv-sync313
uv pip sync --python /tmp/v100_m0_ml/venv-sync313/bin/python requirements.lock

# sm_70 build-flag check (needs no card):
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

## 5. Real-CUDA gate (PENDING)

Still unproven, in order (M3): driver/detection (`nvidia-smi`), **real CUDA
kernel execution** from the locked torch (C2), then measured throughput.
`get_arch_list` containing `sm_70` only proves the kernels are *shipped* in
the wheel. Until M3.3, every statement in this file is a wheel-metadata /
import-level fact.

## 6. Heavy-import boundary

Unchanged and re-checked: `qwen_engine` imports stdlib only; `cli_gen.py
--help` / `--dry-run` pass with torch/diffusers/transformers absent (guarded
suite, `run_cpu_tests.sh`). The ML stack exists only in the temporary audit
venvs; nothing in this milestone installed a heavy import into `/work/.venv`
or the fresh test environments.
