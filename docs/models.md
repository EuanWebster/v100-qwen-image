# Model assets (M2.1 manifest + explicit fetch tool; M2.2a local reference-bundle contract)

Status: the **manifest is committed**; root's **explicitly approved pinned
fetch into `models/assets/` ended at 2026-10-07 21:53 UTC with a timeout on
the first transformer shard**. An independent offline resolver/plan over
that root found **19 of the 24 final files present and size+SHA256-
verified, 5 missing, 0 bad**; a recoverable
`transformer/diffusion_pytorch_model-00001-of-00002.safetensors.partial`
(6,302,990,336 bytes) is in place, and the remaining expected transfer
after that partial is **9,278,315,946 bytes**. The processor and
text-encoder groups are complete. No full-bundle load and no GPU test has
been run; default sandbox DNS is denied and no further permission requests
are being made. No checkpoint has been loaded and no inference has run; no
GPU is installed. Earlier, root fetched **bounded bytes of the 16 small
non-weight upstream source files** at the pinned revision and hashed them
directly for the identity audit (§3) — audit bytes, distinct from the
installed assets above. Everything size- or hash-shaped below is **expected
identity from pinned upstream metadata**, not a local measurement, except
where §3/§7 explicitly record the 19 verified local files. Separately, the
**local reference-bundle contract (M2.2a, `qwen_engine/local_bundle.py`) is
implemented and offline-tested**: a pure *projection* of resolver-verified
paths plus a trusted factory that runs the resolver itself (§8); it asserts
nothing about installed assets, loads or fit.

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
them on 2026-10-07; see §3). They are *not* measured local disk totals:
root's fetch was interrupted at the first transformer shard, so 19 of the
24 files are installed and locally verified (§3/§7) and 5 are not installed
yet.

## 3. Provenance of the hashes (read this before trusting §2)

Two evidence classes, deliberately kept apart:

* **16 regular Git files** — every manifest file *except*
  `processor/tokenizer.json` and the seven `.safetensors` weight files:
  root fetched bounded bytes at the pinned revision and computed the SHA256
  **from the file bytes themselves**.
* **8 LFS-tracked files** — `processor/tokenizer.json` plus the 7 weight
  safetensors (4 text-encoder shards, 2 transformer shards, 1 VAE file): at
  manifest-commit time the bytes were **not** fetched; the sha256/size are
  the **pinned upstream Git-LFS pointer identities** at the pinned
  revision. A Git *blob* SHA1 for an LFS file is *not* the file's SHA256 —
  the manifest records the LFS **file SHA256** identity, which is what the
  resolver checks against real bytes (for the files that are now installed,
  it has — see below and §7).

So: every hash in the manifest is a claim about the file at the pinned
revision, and the evidence classes differ in *local* evidence. For the 16
small regular files the claim was **locally verified at audit time**: root
fetched their bounded bytes at the pinned revision and hashed exactly those
bytes here — a bounded audit read, not an install. For the 8 LFS identities
(tokenizer.json + 7 weight safetensors) the manifest recorded pinned
upstream pointer identities with no local byte verification at manifest
commit time. **Since then, root's approved fetch installed part of the set:
an independent offline resolver run (size + SHA256 over every present file)
verified 19 of the 24 final files on this disk — 0 mismatches — including
the tokenizer and the text-encoder shards; the 5 files that remain missing
include the transformer shard whose fetch timed out** (whose recoverable
`.partial` is *not* a verified file). No statement in this file claims a
*full-bundle* local verification — 5 files are not installed — and no
statement extends to environment files, tensor code or anything else beyond
what §3 explicitly scopes.

## 4. Storage location and the no-download rule

Assets, when an operator deliberately fetches them, live under
**`models/assets/`** (gitignored; the *manifest itself* is committed under
`models/`). The fetch preserves the manifest `filename` layout verbatim under
that root (`models/assets/transformer/config.json`, …). Neither the
manifest task nor the fetch-tool task created the asset directory — the
tool creates paths only when an operator actually runs `--fetch`. Root
later ran exactly that explicit step: `models/assets/` now holds the 19
size+SHA256-verified final files plus one recoverable interrupted
`.partial` (status paragraph and §7); the manifest itself stays committed
under `models/`.

Config path defaults for `manifest_path`/`models_dir` stay **unset** — a run
without an explicit manifest refuses (exit 4) rather than guessing, and the
resolver **never** downloads, mirrors or "auto-fixes": local loading verifies
or fails.

**Fetch tool: ready (code + offline tests; subsequently run by root —
partially complete).** `tools/fetch_models.py` exists as the separate, explicit operator step
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
A staging partial must be the **sole name of its inode**: a multiply
hard-linked partial — empty, partial-length or byte-complete alike — is
refused (exit 6) before any read, transport or write, because resume
appends through the shared inode and would silently rewrite every other
name linked to it, verified finals included (root reproduction: a verified
`victim.bin` hard-linked to `new.bin.partial` let a resume publish rc0 and
silently rewrite `victim.bin`); the link count is re-checked on the opened
FD before any truncate/seek/write, so an ordinary concurrent `link()`
cannot slip past the preflight, and every name and byte is preserved on
refusal. Atomic publish is the one deliberate exception: it creates its own
temporary second hardlink only **after** the full SHA256 rehash passed, and
unlinks the staging name immediately. If that unlink fails, the remaining
`*.partial` is recoverable debris sharing the verified final's inode: a
later fetch refuses to resume into a multiply-linked partial, so the debris
is inert, and removing the stale name loses nothing while the final name
stands.
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
path); a static **manifest-safety preflight** (no disk state, before any
mkdir/network/write) refuses a manifest whose staged partial path
`<final>.partial` for one artifact is also another artifact's final
filename — schema-v1 accepts a manifest listing both `foo` and
`foo.partial`, yet fetching `foo` would mistake the verified final
`foo.partial` for its own partial, append to it and delete it on a hash
mismatch; a filename that is simultaneously another's directory (`a` and
`a/b`) is refused the same way (usability preflight), as is a staged path
that lands on another artifact's **directory** (`a/x` stages at `a/x.partial`
while `a/x.partial/y` needs that exact path as a directory — neither manifest
order can ever download both); the read-only plan
reports the same typed usage error without any write or socket (the
committed 24-entry manifest contains no such names — the refusal protects
the reusable tool contract, not just this manifest); a bounded disk estimate (manifest
bytes + 1 MiB headroom per file) is compared with fresh free space before
any transfer — a **disk** estimate only, no RAM/GPU fit claim (fit is M5
calibration territory). Tool-specific exit codes are in its `--help`.

What has happened since the tool shipped: root ran the explicit approved
`--fetch` against the live Hub; it ended at 2026-10-07 21:53 UTC with a
timeout on the first transformer shard — 19 final files published and
verified, 5 missing, one recoverable partial (status paragraph, §3, §7).
Resuming that transfer remains a lead/operator decision; the tool itself
was exercised by root, not by this codebase's tests (which remain offline
fake-stream only). Operator-facing
examples:

```bash
# Read-only, offline plan against the committed manifest (no writes, no
# socket, exit 0; deterministic JSON):
python tools/fetch_models.py --manifest models/manifest.json --models-dir models/assets

# EXPLICIT operator download + verify (root ran this once — see the status
# paragraph and §7; the codebase's own tests never touch the network):
python tools/fetch_models.py --manifest models/manifest.json --models-dir models/assets --fetch
```

Engine examples — **dry-run / config-only examples, not a runnable complete
adapter and not a proven-fit preset**:

```bash
# From the repo root. Deterministic JSON plan; verifies local files against
# the manifest WITHOUT downloading, reports each artifact's state (today:
# 19 present-verified, 5 missing), exits 0 (dry-run never runs the
# pipeline):
python cli_gen.py --dry-run --manifest models/manifest.json --models-dir models/assets

# Actual run: with any required asset missing/unverified this exits 4 and
# names the unresolved files with repo/revision (today: the 5 outstanding
# ones) — no traceback, no download, no adapter import:
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
  dtype is **config metadata**, not a measured fact about tensor contents
  in shards (the transformer weight shards are not even fully downloaded
  yet).
* Our target path converts/loads for **CPU probes (measured dtype TBD)** and
  **fp16 execution on sm_70 (V100)**. Whether BF16→FP16 conversion is
  lossless-enough for the encoder, what dtype actually lands in memory, and
  what runs where: **unverified**, tasks M2.2/M3/M4 territory.
* The small-file config facts in §2 come from root's bounded reads of those
  *config files only* — they describe declared structure, not runtime
  behaviour. No **weight** load, forward pass or numerical comparison has
  been performed anywhere in this project to date; the only local
  library-level loads so far are the tokenizer-only success and the
  socket-denied processor-only failure recorded in §7 (neither touches
  weights).

## 6. What is *not* claimed

No statement in this file or the manifest asserts: a **full-bundle** local
hash verification (5 files are not installed), a *full* transferred byte
count (the fetch timed out mid-first-transformer-shard; the remaining
expected transfer after the recoverable partial is 9,278,315,946 B), a
successful model *weight* load, numerical fidelity, image quality, or
16 GB fit. Locally verified to date: the 16 small fetched audit files
(§3, at audit time) and the 19 installed final files size+SHA256-checked by
the independent offline resolver run (§3/§7). The old PLAN model-facts
table's third-party size claims (DiT ~14.2 GB, encoder ~17.5 GB, VAE
~0.7–1.4 GB BF16) are backed by the **expected** upstream sizes in §2 — an
evidence upgrade from "card claim" to "pinned metadata", and now partly to
local measurement (the 19 verified files), still not a full-set local
measurement.

## 7. Pending gates (honest state of play)

| Gate | State |
|---|---|
| Manifest committed, schema v1, 24 entries, roles, deterministic order | **done (this task)** |
| Offline tests: identity digest, roles/shards, dry-run all-missing, run→exit 4 | **done (this task)** — the *fresh-child* no-network / no-heavy-import evidence is a property of the **guarded runner** (`bash run_cpu_tests.sh`, both Pythons, audit hook riding into children); plain `pytest` proves only the exit-code/output contract, and acceptance is recorded on the guarded runs |
| Operator fetch **tool** (`tools/fetch_models.py`): code + offline fake-stream tests | **done (root-review fix passes applied)** — plan/help read-only and socket-free; `--fetch` only with explicit paths; resume/publish/escape rules above; subsequently exercised by root's approved run (next row) |
| Operator download of 33.1 GB + local sha256/size confirmation | **partially complete (2026-10-07 21:53 UTC)** — root's explicitly approved pinned fetch **ended with a timeout on the first transformer shard**; independent offline resolver/plan: **19/24 final files present, size+SHA256-verified; 5 missing; 0 bad**; recoverable `transformer/diffusion_pytorch_model-00001-of-00002.safetensors.partial` = 6,302,990,336 B; remaining expected transfer after it = 9,278,315,946 B; processor and text-encoder groups complete. Resuming via the same `--fetch` is a lead/operator decision (default sandbox DNS is denied; disk budget needed — the tool prints a bounded disk estimate, nothing about RAM/GPU fit) |
| Local reference-bundle contract (`qwen_engine/local_bundle.py`, M2.2a) | **implemented and offline-tested** (root-review fix passes applied) — trusted factory `resolve_local_bundle` runs `resolve_manifest` itself before projecting (explicit full-rehash cost; layout + the exact pinned profile identity `Qwen/Qwen-Image-2.1 @ d26bb61…e1ba3` validated first); lower-level `project_resolution_report` **trusts its caller's report** and is never authenticated evidence; role-grouped adapter `assets` + explicit component dirs, **absolute CWD-stable paths**; coherence-checked success claims; closed-tree `preflight_closed_tree` for future `from_pretrained` probes (refuses any entry outside the 24-file allowlist — lookalikes, extra code, `*.partial` debris, symlinks); construction-time scalar invariants + defensive freezing; version-tagged artifact-set fingerprints whose **encoder digest is only the asset component** of the future worker cache key (worker protocol/impl revision + diffusers/transformers pins required; worker milestone); **offline-tested only** with tiny fake files: real-asset resolve, CPU component load and everything after remain pending |
| M2.2 CPU load-path probes / encoder fidelity | pending — dependency work is a later scoped milestone; separately recorded facts so far: a **socket-denied processor-only load** reached the pinned transformers code but failed for missing `torchvision`; a **tokenizer-only local load succeeded** (Qwen2Tokenizer, 151,669 entries, 9 tokens for root's sample); no full-bundle load, no fidelity check, no GPU test |
| M4 adapter (`qwen21_diffusers`) | pending — `--run` reaches it only after assets verify |
| M3 GPU bring-up | gated on physical V100 installation (confirmed not installed) |
| M5 memory-policy calibration | pending; nothing here proves 16 GB fit |

## 8. Local reference-bundle contract (M2.2a — implemented, offline-tested only)

`qwen_engine/local_bundle.py` produces the `GenerationAdapter.load(config,
assets)` shape — verified paths grouped by role (manifest order, immutable)
and explicit component directories under the **explicit** model root, ready
for future `local_files_only=True` probes (pipeline dir = root; the `dit`
role's files live under `transformer/`) — through two honestly separated
APIs. **`resolve_local_bundle(manifest, models_dir)` is the trusted
factory**: it validates the layout — and the profile's **exact pinned
identity** `FULL_REFERENCE_REPO` @ `FULL_REFERENCE_REVISION`
(`Qwen/Qwen-Image-2.1` @ `d26bb61…e1ba3`) — *before* running the trusted
`resolve_manifest` itself over the real bytes; the rehash is the point, and
so is its explicit cost (for the real 33 GB set this is a full byte
re-verification, not a metadata call). **`project_resolution_report(manifest,
report, models_dir)` is the lower-level projection and TRUSTS ITS
CALLER**: `ResolutionReport` is a public dataclass, so even a perfectly
shape-consistent `ok=True` report can be hand-forged; the projection's
positional/field equality, coherence checks (`errors`, per-status `error`,
`resolved` mapping) and cheap regular-file/inside-root probes police the
*shape* of the success claim and are **not authenticated evidence** of the
bytes. Neither API closes a hostile concurrent-filesystem TOCTOU — actual
component load must reverify immediately before the library reopens files.
Internal consistency is not identity: a manifest at any other repo or
revision — however consistent — is refused as a different model identity;
**a new model revision requires a new reviewed profile**, never a relaxed
check here. The committed full-reference set is an explicit immutable table
(24 files; pipeline 1, processor 9, scheduler 1, text_encoder 7, dit 4, vae
2); no GGUF/LoRA/single-file in this profile — future variants are separate
profiles, not fallbacks. Everything the bundle publishes (root, `assets`,
`component_dirs`) is an **absolute** path under the canonical resolved root
— a worker started from a different working directory loads the same files
even when `models_dir` was given relative — while the report is still
matched against the caller's lexical `root/filename`; paths come only from
`root/filename`, never repo ids, never an ambient HF cache. Incomplete,
reordered, padded or field-forged reports, escaped symlinks and
file-vs-directory layouts are refused with typed errors (exit-4 channel)
before any ML import. **Closed-tree preflight** (`preflight_closed_tree`,
also a method on the bundle): because a future `from_pretrained(component_dir)`
may open files the manifest never listed, the settled tree under an explicit
root can be enumerated in full (contents never read, symlinks never
followed) and any entry outside the 24-file allowlist and its required
directories — recognized-looking extra config/weight/**code** files, empty
extra directories, leftover `*.partial` fetch debris, symlinks even to
inside-root targets — is refused; deliberately stricter than the resolver's
symlink tolerance, and explicitly **not** to be run on a root whose fetch is
in progress. Public direct constructors police their own **scalar
invariants** (profile name, absolute and mutually consistent
`root`/`root_resolved`, exact pinned repo/revision, 64-hex digests) with
typed errors and still perform **no** resolver or disk verification — data
hygiene, honestly labelled; `artifact_set_fingerprint` is a *generic*
identity function (it digests foreign identities so drift stays
demonstrable) but rejects non-artifact and duplicate-filename inputs that
would otherwise collapse into a plausible wrong digest. Version-tagged
**artifact-set fingerprints** (SHA256 over sorted
repo/revision/filename/sha256/bytes rows) give a whole-bundle digest and an
encoder+processor digest: **the encoder digest is only the ASSET component
of the future encoder-worker cache key** — the production worker key must
also carry the worker protocol/implementation revision and the exact
diffusers/transformers pins (wiring `EncoderRequest.model_sha256` or any
worker cache key to the asset digest alone is a contract violation; the
full key belongs to the worker milestone). Digests are identity metadata:
they prove nothing about tensor load, generation or quality. Tests are
offline with tiny fake contents (never `models/assets/`, never the 33 GB):
through the real resolver, plus the trusted factory's refusal when file
*content* changes under a forged consistent report.
