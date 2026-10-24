# Q8 GGUF sidecar artifact contract (`qwen_engine.q8_artifact`)

Status: **implemented (module + CPU test suite); not yet wired into any runtime
path.** Integration (CLI/web wiring, loader handoff) is owned by root during
review/integration. This document is the review contract for
`src/qwen_engine/q8_artifact.py`.

## 1. What this is

One **single-file GGUF (Q8_0)** checkpoint offered as a *sidecar* to the closed
24-file full-reference bundle (`qwen_engine.local_bundle`). The base bundle is
**unchanged and stays closed**: a `.gguf` file is not, and may never become, an
entry of `models/` or the closed `assets/` tree. Sidecar metadata (a small JSON
document) and sidecar data (the GGUF file) live **outside** that tree, e.g.:

```
data/q8/q8_artifact.json      # sidecar manifest (<= 1 MiB)
data/q8/qwen-image-q8_0.gguf  # the pinned GGUF file
```

`resolve_q8_artifact(manifest_path, models_dir=None, *,
expected_source_revision=FULL_REFERENCE_REVISION)` parses the sidecar, verifies
the file, and returns a frozen `Q8Artifact`. Stdlib only: no downloads, no ML
imports, no writes, no network.

Candidate upstream producer: `unsloth/Qwen-Image-2.1-GGUF`. The module
hardcodes **no** repo/revision/hash/size for it — those are per-deployment
reviewed values carried by the sidecar and reviewed by the operator.

## 2. Sidecar JSON schema — `qwen_engine.q8_artifact.v1`

Exact key set; **unknown keys, missing keys, duplicate JSON keys, nonfinite
constants, and depth/pathological input are all refused** (strict parse via
`_jsonsafe.strict_json_loads`, bounded 1 MiB read via `_local_io`).

| field | rule |
| --- | --- |
| `schema` | exactly `"qwen_engine.q8_artifact.v1"` |
| `repo` | `org/name`: exactly one `/`; each component 1–64 chars `[A-Za-z0-9._-]`, starts alphanumeric |
| `revision` | 40 lowercase hex (pinned commit sha — never a branch/tag) |
| `filename` | safe relative POSIX path ending `.gguf`; rejects absolute, `..`, `.`, empty components, backslashes, drive letters (`C:/...`), NUL |
| `sha256` | 64 lowercase hex |
| `bytes` | positive `int`; `bool` explicitly refused |
| `source_model_revision` | 40 lowercase hex; must equal `expected_source_revision` |
| `quant_format` | exactly `"gguf-q8_0"` |

Validation runs on **direct dataclass construction as well as JSON parsing**
(`Q8ArtifactSpec.__post_init__`), mirroring `manifest.ModelArtifact`.

## 3. Verification performed on the file

Under the data root (explicit `models_dir`, else the sidecar's parent):

1. **Containment** — the target must canonicalize inside the root; a symlink
   resolving outside it is `Q8FilePathError`. In-root symlinks are allowed
   (same policy as the base resolver).
2. **Regular file** — directories/FIFOs/devices refused from `stat` before
   open (never blocks on a FIFO).
3. **Exact declared size** (`bytes`).
4. **Streamed SHA256** over the whole file in 1 MiB chunks through
   no-follow, `dir_fd`-anchored descriptors — the file is never read into RAM
   wholesale.
5. **GGUF header** — magic `GGUF` and version 2 or 3 (little-endian uint32s),
   captured from the first 8 bytes *while hashing*.
6. **Stability** — file identity/size/mode/timestamps, the directory entry,
   and the root itself are re-checked after hashing; any change mid-verify is
   a typed failure (mirrors `manifest._verify_one`).

Order of reporting: containment → regular → size → hash → magic/version.

## 4. Trust model (what the checks do NOT mean)

* `sha256` match = **operator-reviewed identity metadata**: the bytes on disk
  are the bytes the operator reviewed. It is **not** a numerical or provenance
  proof — not proof the quantization is faithful, the upstream repo is honest,
  or outputs will be good.
* `identity_sha256` = SHA256 over the version-tagged
  (`qwen-image.q8-artifact.v1`) canonical JSON (sorted keys) of the eight
  manifest fields — **no ambient paths**. Same reviewed file → same
  fingerprint on any machine; any field change drifts it. Suitable for
  review/caching keys only.
* **Reader, architecture and Q8 tensor-level validation remain the loader's
  responsibility.** This module inspects only the 8-byte header; it does not
  parse GGUF metadata, tensor infos, or `general.architecture`.
* `source_model_revision` must equal the engine's reviewed base revision
  (default `local_bundle.FULL_REFERENCE_REVISION`). A GGUF quantized from any
  other base is a different, unreviewed model identity: refused with
  `Q8SidecarFormatError`; a new reviewed profile — never a relaxed check — is
  the way to accept one.

## 5. Errors (all `ManifestError` → CLI exit 4)

| error | trigger |
| --- | --- |
| `Q8SidecarFormatError` (⊂ `ManifestFormatError`) | unreadable/oversize/non-UTF8/non-JSON sidecar; duplicate keys; unknown/missing fields; any field-rule violation; `source_model_revision` mismatch; invalid `expected_source_revision` |
| `Q8FileMissingError` | file absent |
| `Q8FilePathError` | canonicalizes outside data root |
| `Q8FileSizeMismatchError` | size ≠ `bytes`, or file changed during hashing |
| `Q8FileHashMismatchError` | streamed SHA256 ≠ `sha256` |
| `Q8FileUnreadableError` | permissions, non-regular file, bad GGUF magic/version, unresolvable root/path |

File errors carry the sidecar identity (`repo`/`revision`/`filename`,
`role="q8_gguf"`) in their message and attributes, like `manifest.ArtifactError`.

## 6. API summary

* `Q8ArtifactSpec` — frozen, validated sidecar identity (8 fields), `.identity`, `.to_dict()`.
* `Q8Artifact` — frozen: `spec`, `path` (canonical absolute `Path`),
  `identity_sha256`; convenience properties for all manifest fields;
  `to_dict()` excludes `path` (the fingerprint covers exactly the dict's
  manifest fields + fingerprint).
* `resolve_q8_artifact(...) -> Q8Artifact` — the public entry point.
* `load_q8_sidecar`, `q8_artifact_spec_from_dict`, `verify_q8_file`,
  `q8_identity_sha256` — lower-level pieces, individually typed.
* Constants: `Q8_ARTIFACT_SCHEMA_VERSION`, `Q8_QUANT_FORMAT`,
  `Q8_IDENTITY_FINGERPRINT_VERSION`, `MAX_Q8_SIDECAR_BYTES`,
  `GGUF_MAGIC`, `GGUF_SUPPORTED_VERSIONS`.

## 7. Testing

`tests/test_q8_artifact.py` (CPU guard, stdlib only, 56 cases): happy path
(v2/v3 headers, nested filenames, in-root symlinks), strict-JSON refusals
(duplicate keys, `NaN`, unknown/missing fields, bool-as-int, bad repo/hash/
revision/filename forms incl. traversal/absolute/backslash/drive), source-
revision pin + override, path escape via symlink, directory swap, size/hash/
magic/version corruption, ordering (size gate before header), fingerprint
path-independence and drift, typed-error hierarchy, and no-side-effect
(byte/mtime invariance) checks.
