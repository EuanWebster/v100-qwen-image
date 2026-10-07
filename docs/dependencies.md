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
| diffusers | commit `c6df88a511a98740646ee55577b590c9852650ce` (tree `d77a3e3882af758d606ca22b53f63be8764e643a`, commit date 2026-10-06T19:04:08Z, "[docs] Jobs guide (#14926)"), reported version `0.41.0.dev0` | git URL in `requirements.in` (no release wheel exists for this commit); installed `direct_url.json` records the exact commit |
| transformers | **git commit** `856157a2f3e9594954310df18fdccc31ffddebe9` per PLAN M0.1 ("exact git commits"). The GitHub tag `v5.17.0` is an **annotated tag object** `595ff117c8412ec01262084058276e6b27a857d9`; `git ls-remote …refs/tags/v5.17.0^{}` dereferences it to the commit pinned above — the commit (never the tag object) is the source identity. The commit's `__init__.__version__` is `5.17.0`, identical to the PyPI release wheel | git URL in `requirements.in`; installed `direct_url.json` records the exact commit |
| accelerate | `1.15.0` | PyPI (diffusers `torch`/`gguf` extras dep: device_map/offload) |
| safetensors | `0.8.0` | PyPI (diffusers hard dep) |
| pillow | `12.3.0` | PyPI (diffusers hard dep) |
| gguf | `0.19.0` (floor `>=0.10.0`, which `GGUFQuantizer.validate_environment` enforces) | PyPI — native GGUF quant types live in diffusers (`quantizers/gguf/utils.py`); `gguf` provides `GGUFReader`/`GGML_QUANT_SIZES` |
| numpy | `2.4.6` — deliberate uniformity pin: numpy 2.5.x is `requires-python >= 3.12` (verified from PyPI metadata), so an unpinned resolve would differ between the two supported interpreters | PyPI |
| networkx | `3.6.1` — same uniformity reason (networkx 3.7 is `requires-python >= 3.12`, verified from PyPI metadata; 3.6.1 is the newest release supporting both interpreters) | PyPI |

Transitive pins (nvidia-*-cu12 set, triton 3.8.0, tokenizers 0.23.2,
huggingface-hub 1.33.0, cuda-bindings 12.9.9, …) are in
[`requirements.lock`](../requirements.lock) with wheel hashes. uv emits **no**
`--hash` line for the two git requirements — a repository build cannot be
content-hashed at compile time — so there the immutable commit SHA *is* the
integrity anchor; every other package's wheel hash is enforced at install.
That mix is precisely why the lock installs with `uv pip sync` only:
`pip install --require-hashes -r requirements.lock` cannot handle VCS
requirements at all and is **not** a supported path here. `torchvision`
is **not** required for the text-to-image import path: `transformers`'
`Qwen3VLProcessor` and the pipeline module import cleanly without it (verified
2026-10-07); if a future load path needs it, it joins via an explicit
requirements change, not silently.

No `requirements.in`/lock change touches the **developer-only CPU
environment**: `pyproject.toml` + `uv.lock` remain the pinned pytest-only env
(`pytest==9.1.1`); the ML stack is installed *only* from `requirements.lock`
by explicit operator command.

## 2. Verified installed facts (2026-10-07 M0.1 fix pass, no GPU present)

Two fresh venvs were built with the §4 `uv pip sync` command from the committed
lock (wheel hashes enforced; VCS identities checked via the installed
`direct_url.json`) and checked:

| env | python | torch | `torch.version.cuda` | VCS identities (`direct_url.json`) |
|---|---|---|---|---|
| `/tmp/v100_m0_ml/venv-ml311` | 3.11.15 | 2.14.1+cu126 | 12.6 | diffusers `0.41.0.dev0` @ `c6df88a5…650ce`; transformers `5.17.0` @ `856157a2…ebe9` |
| `/tmp/v100_m0_ml/venv-ml313` | 3.13.5 | 2.14.1+cu126 | 12.6 | same |

`uv pip check --python <env python>` reports "All installed packages are
compatible" (63 packages) in both envs.

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
`QwenImage21Pipeline` (`diffusers.pipelines.qwenimage21.pipeline_qwenimage21`)
binds `Qwen3VLForConditionalGeneration` / `Qwen3VLProcessor` from the
git-installed transformers 5.17.0 without `torchvision`; its `__call__`
signature exposes `prompt/image/negative_prompt/true_cfg_scale/height/width/
num_inference_steps/…` as surveyed.

## 3. Index / interpreter assumptions

* Target: Linux **x86_64**, CPython **3.11 and 3.13** (cp311/cp313
  manylinux_2_28_x86_64 wheels). No other platform/marker is locked; installing
  elsewhere will fail hash resolution rather than guess.
* The resolve was produced with `--python-platform x86_64-manylinux_2_28` and
  `--index-strategy unsafe-best-match` (a CLI-only flag — uv's requirements
  parser rejects it inside a file, and the same flag is required at install
  time, which §4 shows). uv's *default* first-index strategy lets the torch
  extra index mask PyPI and selects the stale mirrors it hosts (certifi
  2022.12.7, requests 2.28.1, urllib3 1.26.13, charset-normalizer 2.1.1,
  idna 3.4, packaging 24.1) — that is a dependency-confusion-shaped
  artifact, not a desirable resolve. With best-match the lock's non-torch
  entries are **exactly** the pure-PyPI resolve (verified 2026-10-07: a
  torch-free compile against PyPI alone produced an identical package set),
  while `torch==2.14.1+cu126` can only ever come from the official cu126
  index because the `+cu126` local label is published nowhere else. torch's
  own CUDA subtree (nvidia-*-cu12, cuda-toolkit 12.6.3, cuda-bindings
  12.9.9) stays in the 12.x series because torch's declared constraints
  (`cuda-bindings<13,>=12.9.4`, `cuda-toolkit==12.6.3`) win over PyPI's
  newer cu13-era copies.
* The two per-interpreter compiles now produce **byte-identical** output
  apart from the autogenerated command line — no markupsafe hash merge is
  needed any more (the earlier hand-merged pair predates the index fix).

## 4. Reproduce / verify

```bash
# regenerate (the two runs must be byte-identical apart from the autogenerated
# command line; --index-strategy is REQUIRED — see §3):
uv pip compile requirements.in --python-version 3.11 \
    --python-platform x86_64-manylinux_2_28 --index-strategy unsafe-best-match \
    --generate-hashes -o /tmp/lock311.txt
uv pip compile requirements.in --python-version 3.13 \
    --python-platform x86_64-manylinux_2_28 --index-strategy unsafe-best-match \
    --generate-hashes -o /tmp/lock313.txt
diff /tmp/lock311.txt /tmp/lock313.txt   # only the command-line comment may differ

# audit venv (temporary; NEVER /work/.venv). uv pip sync enforces the recorded
# wheel hashes; the two git entries are anchored by their commit SHA instead
# (uv cannot hash a repo build; pip --require-hashes cannot process VCS
# requirements at all — do not use pip here):
uv venv --python 3.13 /tmp/v100_m0_ml/venv-ml313
uv pip sync --python /tmp/v100_m0_ml/venv-ml313/bin/python \
    --index-strategy unsafe-best-match requirements.lock
uv pip check --python /tmp/v100_m0_ml/venv-ml313/bin/python

# sm_70 build-flag check — invoke the SYNCED venv's python explicitly
# (a bare `python` is the wrong environment):
/tmp/v100_m0_ml/venv-ml313/bin/python -c \
    "import torch; print(torch.__version__, torch.cuda.get_arch_list())"

# VCS identity of what actually got installed (commit must match §1):
/tmp/v100_m0_ml/venv-ml313/bin/python -c \
    "import importlib.metadata as md, json; [print(n, md.version(n), json.loads(md.distribution(n).read_text('direct_url.json'))['url'], json.loads(md.distribution(n).read_text('direct_url.json'))['vcs_info']['commit_id']) for n in ('diffusers','transformers')]"
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
