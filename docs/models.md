# Model assets (M2.1 manifest preparation + explicit fetch tool, code/tests only)

Status: the **manifest is committed; the assets are not**; the explicit
operator **fetch tool exists (code + offline tests only) and has never been
run against the network**. No model *weight*
has ever been downloaded into this environment, nothing is installed under
`models/assets/`, no checkpoint has been loaded, no inference has run, and no
GPU is installed. What *has* happened: root fetched **bounded bytes of the 16
small non-weight upstream source files** at the pinned revision and hashed
them directly for the identity audit (§3) — fetched audit bytes, not
installed assets. Everything size- or hash-shaped below is **expected
identity from pinned upstream metadata**, never a measurement of installed
local storage.

## 1. What is pinned

[`models/manifest.json`](../models/manifest.json) (schema v1) pins all **24**
runtime files of the full-precision reference build:

* repo `Qwen/Qwen-Image-2.1`
* immutable revision (commit SHA) `d26bb61231c349cf6b7896fa83353113880e1ba3`
  — browse it at
  <https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3>;
  per-file immutable links have the form
  <https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/`filename`>
* per artifact: `role`, `repo`, `revision`, `filename`, `sha256`, `bytes`
* direct named links at that same immutable revision:
  [**README.md** — the model card](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/README.md)
  and
  [**LICENSE**](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE)

The model card (repo-root `README.md`) and the repository's `LICENSE` are
documentation/license resources at the pinned revision: they are **excluded
from the runtime file set** — the resolver's contract is the 24 files the
pipeline actually reads — which is a manifest-scope decision, **not**
permission to ignore licensing: the LICENSE governs any download and use of
these assets. The upstream `README.md`, `LICENSE`, `.gitattributes` and
preview image are likewise absent from the manifest by design.

There is no LoRA, no GGUF entry and no implicit accelerator/quant selection
anywhere in this manifest: it describes the full reference weights only.
Quantized capacity/quality variants (a GGUF-style ladder) remain a **future,
unchosen** option; nothing here benchmarks or endorses them, and no quality
claim is made for the full reference either (see §6).

## 2. Roles and file groups (expected sizes — metadata, not measurements)

| Role | Files | Group | Expected bytes | What it is |
|---|---|---|---|---|
| `pipeline` | 1 | `model_index.json` | 447 | Diffusers pipeline index: names `QwenImage21Pipeline`, `FlowMatchEulerDiscreteScheduler`, `AutoencoderKLQwenImage21`, `Qwen3VLForConditionalGeneration`, `Qwen3VLProcessor`, `QwenImage21Transformer2DModel` (no `sample_sigmas` in the index) |
| `processor` | 9 | `processor/*` | 15,884,996 | Qwen3-VL processor/tokenizer: vocab, merges, tokenizer JSON, chat template, preprocessor/video-preprocessor/tokenizer configs, added/special tokens. Config-style text/JSON; the chat template's raw content is deliberately **not** reproduced anywhere local (local generation stop-token gotcha) — only its name and hash are recorded |
| `scheduler` | 1 | `scheduler/scheduler_config.json` | 485 | `FlowMatchEulerDiscreteScheduler` config |
| `text_encoder` | 7 | `text_encoder/*` | 17,534,409,013 | Qwen3-VL 8B encoder: config + generation config + index + **4 safetensors shards** (~17.53 GB ≈ 16.33 GiB weights) |
| `dit` | 4 | `transformer/*` | 14,230,315,061 | `QwenImage21Transformer2DModel`: config (`in_channels` 64, `num_layers` 32, `num_attention_heads` 32, `attention_head_dim` 128, `causal_condition` true) + index + **2 safetensors shards** (~14.23 GB ≈ 13.25 GiB weights) |
| `vae` | 2 | `vae/*` | 1,350,991,591 | `AutoencoderKLQwenImage21`: config (`z_dim` 64, `latents_mean`/`latents_std` 64 entries each) + **1 safetensors** (~1.35 GB) |
| **Total** | **24** | | **33,131,601,593** (≈ 33.13 GB / 30.86 GiB) | of which 7 `.safetensors` weight files: 33,115,613,408 bytes |

These totals are **expected upstream metadata byte totals** (root verified
them on 2026-10-07; see §3). They are *not* measured local disk totals — no
asset is installed locally yet.

## 3. Provenance of the hashes (read this before trusting §2)

Two evidence classes, deliberately kept apart:

* **16 regular Git files** — every manifest file *except*
  `processor/tokenizer.json` and the seven `.safetensors` weight files:
  root fetched bounded bytes at the pinned revision and computed the SHA256
  **from the file bytes themselves**.
* **8 LFS-tracked files** — `processor/tokenizer.json` plus the 7 weight
  safetensors (4 text-encoder shards, 2 transformer shards, 1 VAE file): the
  bytes were **not** fetched; the sha256/size are the **pinned upstream
  Git-LFS pointer identities** at the pinned revision. A Git *blob* SHA1 for
  an LFS file is *not* the file's SHA256 — the manifest records the LFS
  **file SHA256** identity, which is what the resolver will check against
  real bytes.

So: every hash in the manifest is a claim about the file at the pinned
revision, and the two classes differ in *local* evidence. For the 16 small
regular files the claim is **locally verified**: root fetched their bounded
bytes at the pinned revision and hashed exactly those bytes here — that was
a bounded audit read, not an install: nothing landed under
`models/assets/`, no asset was put in place for the pipeline, and no weight
was ever fetched. For the 8 LFS identities (tokenizer.json + 7 weight
safetensors) no local byte verification has happened; there are no
installed-asset or weight bytes on this disk at all, and the first real
download plus the resolver's sha256 check (future task) is what *confirms*
those claims here. No statement in this file says "the installed assets hash
to X" — no assets are installed — and no statement extends to environment
files, tensor code or anything else beyond what §3 explicitly scopes.

## 4. Storage location and the no-download rule

Assets, when an operator deliberately fetches them, live under
**`models/assets/`** (gitignored; the *manifest itself* is committed under
`models/`). The fetch preserves the manifest `filename` layout verbatim under
that root (`models/assets/transformer/config.json`, …). No asset directory or
file exists yet: neither the manifest task nor the fetch-tool task created
one (the tool creates paths only when an operator actually runs `--fetch`,
which has not happened).

Config path defaults for `manifest_path`/`models_dir` stay **unset** — a run
without an explicit manifest refuses (exit 4) rather than guessing, and the
resolver **never** downloads, mirrors or "auto-fixes": local loading verifies
or fails.

**Fetch tool: ready (code + offline tests) — assets still NOT downloaded.**
`tools/fetch_models.py` now exists as the separate, explicit operator step
(stdlib-only; no engine import side effects; the resolver/CLI/generation
paths remain no-download and network-free). Its contract: the default and
`--help` paths are **read-only and offline** (a deterministic JSON plan);
the only network/write path is an unmistakable `--fetch` with explicit
`--manifest` and `--models-dir`. Downloads are pinned to
`https://huggingface.co/<repo>/resolve/<pinned-commit>/<filename>` (host is
a constant; non-HTTPS redirects refused; every message renders only scheme,
host/port and path — URL userinfo, query (incl. signed-URL signatures),
server reason text and untrusted header values are never echoed), stream in
bounded 1 MiB chunks with the
manifest byte count enforced as an upper bound while reading, and publish
**only** on exact size + SHA256 match via an atomic no-clobber link of the
`<file>.partial` in the same asset directory. Interrupted downloads resume
in that partial **only** against a 206 response whose numeric
`Content-Range` proves the server resumed at the requested offset with a
total exactly equal to the manifest size and a coherent span
(`start <= end < total`; a wildcard `*` total is refused) — Range ignored or
misstated → refused, partial untouched. A partial already at exactly the
manifest byte length is rehashed and published without any request (never an
impossible Range from EOF), or fails integrity without publishing.
Partials are opened `O_NOFOLLOW`, new ones created `O_EXCL` (a concurrently
appeared or changed partial is refused, never truncated over), and every
`os.write` is repeated until the chunk is fully written. Final verification
always rehashes
the whole assembled file from disk, so a corrupt partial can never publish.
An existing file that verifies is skipped; an existing file whose identity
does **not** match is refused, never deleted or overwritten; symlinked
paths on the write path are refused; an explicit `--models-dir` root
that is (or sits under) a symlink is refused before any mkdir, network
access or write, and the fetch root may contain **no `..` component** —
a lexical `..` collapse after a symlinked step (e.g. `in/link/../assets`
with `in/link` a symlink) resolves on the filesystem elsewhere than the
`abspath` the checker would inspect, so `--fetch` refuses it up front;
the read-only offline plan is unaffected; a manifest `repo` must be
exactly one `namespace/name` whose `[A-Za-z0-9._-]` segments are never
exactly `.` or `..` (whole dot-segments would collapse inside the URL
path); a bounded disk estimate (manifest
bytes + 1 MiB headroom per file) is compared with fresh free space before
any transfer — a **disk** estimate only, no RAM/GPU fit claim (fit is M5
calibration territory). Tool-specific exit codes are in its `--help`.

What has **not** happened: this assignment shipped code and offline
fake-stream tests only — **no real 33.1 GB transfer has been executed**, no
network request was made for it, and running it against the live Hub is a
separate lead/operator decision after root review. Operator-facing
examples:

```bash
# Read-only, offline plan against the committed manifest (no writes, no
# socket, exit 0; deterministic JSON):
python tools/fetch_models.py --manifest models/manifest.json --models-dir models/assets

# EXPLICIT operator download + verify (this assignment has NOT run it):
python tools/fetch_models.py --manifest models/manifest.json --models-dir models/assets --fetch
```

Engine examples — **dry-run / config-only examples, not a runnable complete
adapter and not a proven-fit preset**:

```bash
# From the repo root. Deterministic JSON plan; verifies nothing exists yet,
# reports every artifact missing, exits 0 (dry-run never runs the pipeline):
python cli_gen.py --dry-run --manifest models/manifest.json --models-dir models/assets

# Actual run: with assets absent this exits 4 and names all 24 required
# files with repo/revision — no traceback, no download, no adapter import:
python cli_gen.py --run --manifest models/manifest.json --models-dir models/assets
```

The `--run` form additionally needs an `output_dir` (else exit 2 after asset
verification) and ultimately the M4 adapter (does not exist yet), a calibrated
memory policy (M5) and a GPU (M3) — all **pending gates**, not failures of
this manifest.

## 5. Reference dtype vs. execution dtype (unverified territory)

* The weight shards are the **reference full-precision build**; the text
  encoder's `config.json` declares **`dtype: bfloat16`** at the top level
  (root inspected the pinned file directly at 2026-10-07 19:36 UTC: the key
  is `dtype` — also in the nested text/vision sub-configs — and there is no
  `torch_dtype` key), and the model card describes BF16 storage. A declared
  dtype is **config metadata**, not a measured fact about tensor contents in
  shards that were never downloaded.
* Our target path converts/loads for **CPU probes (measured dtype TBD)** and
  **fp16 execution on sm_70 (V100)**. Whether BF16→FP16 conversion is
  lossless-enough for the encoder, what dtype actually lands in memory, and
  what runs where: **unverified**, tasks M2.2/M3/M4 territory.
* The small-file config facts in §2 come from root's bounded reads of those
  *config files only* — they describe declared structure, not runtime
  behaviour. No load, forward pass or numerical comparison has been
  performed anywhere in this project to date.

## 6. What is *not* claimed

No statement in this file or the manifest asserts: downloaded *weight* byte
counts, a locally verified hash of an installed asset or weight (the §3
exception stands: the 16 small fetched audit files' SHA256s were computed
from bytes fetched directly at the pinned revision), a successful model
load, numerical fidelity, image quality, or 16 GB fit. The old PLAN
model-facts table's third-party size claims (DiT ~14.2 GB, encoder ~17.5 GB,
VAE ~0.7–1.4 GB BF16) are now backed by the **expected** upstream sizes in
§2 — an evidence upgrade from "card claim" to "pinned metadata", still *not*
a local measurement of installed assets.

## 7. Pending gates (honest state of play)

| Gate | State |
|---|---|
| Manifest committed, schema v1, 24 entries, roles, deterministic order | **done (this task)** |
| Offline tests: identity digest, roles/shards, dry-run all-missing, run→exit 4 | **done (this task)** — the *fresh-child* no-network / no-heavy-import evidence is a property of the **guarded runner** (`bash run_cpu_tests.sh`, both Pythons, audit hook riding into children); plain `pytest` proves only the exit-code/output contract, and acceptance is recorded on the guarded runs |
| Operator fetch **tool** (`tools/fetch_models.py`): code + offline fake-stream tests | **done (root-review fix pass applied), awaiting root re-review** — plan/help read-only and socket-free; `--fetch` only with explicit paths; resume/publish/escape rules above; **no real download executed** |
| Operator download of 33.1 GB + local sha256/size confirmation | pending (the tool exists but has never touched the network; a lead/operator decision after review; disk budget needed — the tool prints a bounded disk estimate, nothing about RAM/GPU fit) |
| M2.2 CPU load-path probes / encoder fidelity | pending |
| M4 adapter (`qwen21_diffusers`) | pending — `--run` reaches it only after assets verify |
| M3 GPU bring-up | gated on physical V100 installation (confirmed not installed) |
| M5 memory-policy calibration | pending; nothing here proves 16 GB fit |
