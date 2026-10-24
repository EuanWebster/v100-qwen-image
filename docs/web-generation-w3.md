# W3 — local web generation

Implemented and tested on 2026-10-24. The web workspace now submits real jobs
through `core.generate()` and the default Qwen adapter. It uses the CLI's
manifest verification and adapter/encoder checks. ComfyUI is not involved.

## Run

For this host with LAN and Tailscale access, use the executable
`./start-web.sh`. Addresses and overrides: [LAN/Tailscale startup](lan-tailscale.md).
The manual command below retains the default localhost-only bind.

From the repository with its ML virtual environment active:

```bash
python -m qwen_engine.web --enable-generation \
  --manifest models/manifest.json --models-dir models/assets \
  --output-dir output/web --cache-dir cache \
  --attest-unbounded-host
```

Open **http://127.0.0.1:8765**. Omitting `--enable-generation` retains the
planning/export workspace without a job worker or output-directory creation.
Generation requires all four explicit paths and valid manifest metadata at
startup; the output root is created if needed. Weights are verified per job.
There are no downloads. Existing AMD/NVIDIA ComfyUI environments are untouched.

`--attest-unbounded-host` has exactly the CLI meaning: the operator asserts
that this process tree has no cgroup RAM cap. It does not remove a cap. Omit
it where that assertion is false; the encoder then requires its normal
provable memory-scope evidence.

## Behavior

- Submit with Generate or Ctrl/Cmd+Enter. Jobs run serially, with up to four
  waiting jobs and 32 total history records. Each job owns a separate directory
  below the configured output root. Browser JSON cannot supply runtime paths.
- Status shows queued/running/cancelling/cancelled/succeeded/failed and the
  actual engine phase. Asset verification precedes the load phase. No invented
  percentage or step count is shown while work runs.
- Timings come from the engine's phase logger and a monotonic execution clock.
  Elapsed includes asset verification, model loading, generation and adapter
  cleanup; it excludes queue wait and the controller's final artifact check.
- Completed jobs appear in Preview and History, with Download, Open and
  Generate again. Generate again uses the selected job's saved configuration,
  including its exact seed, and preserves the current form draft. A null API
  seed is resolved once before enqueueing. Public seeds remain decimal strings.
- Page reload recovers the server's job records. **History is in memory for
  this server session**, not a persistent index of all CLI or prior-server files.
  Forget entry removes terminal metadata only; it never deletes images. The
  bounded history must be explicitly freed before more records can be added.
- Cancellation is cooperative. A running job stays cancelling until execution
  releases its resources; a verified saved image wins a late cancellation.
  Ctrl+C/SIGTERM closes HTTP handling and requests job cancellation, waiting
  at most 60 seconds for the generation worker and reporting a timeout honestly.
- A failed job exposes a fixed error code. Detailed diagnostics stay in the
  server terminal. Lost POST responses are not retried automatically: inspect
  History before retrying because the job may already have been accepted.

## Submission availability and memory

`generation_available` means the server accepts generation submissions; it
does **not** claim hardware verification or memory fitness. Status/Plan retain
unknown hardware/calibration and unverified assets until actual job execution.
The adapter is truthfully reported available. Static submission blockers are
`generation_disabled`, `runtime_unconfigured`, and the applicable manifest error.

This milestone preserves the currently working CLI execution policy. It does
not integrate `assess_memory()` or exact calibration preflight, add a GPU safety
reserve, or establish that every editable resolution/CFG setting fits. Measured
2K runs are particularly tight. The supported execution policy remains FP16,
resident, auto/SDPA. Above 4096 latent tokens tiled decode is required at submit.
Adapter/encoder checks and real execution failures remain authoritative.

## HTTP contract

The default loopback bind, exact Host/Origin, same-origin fetch, body/timeout limits,
`X-Qwen-UI: 1`, JSON-only writes, defensive headers and static allowlists remain.
No arbitrary filesystem route is added.

| Method | Route | Result |
|---|---|---|
| POST | `/api/v1/jobs` | Shared plan-request envelope → 202 job snapshot |
| GET | `/api/v1/jobs` | Bounded job-ID list |
| GET | `/api/v1/jobs/<32 lowercase hex ID>` | Job/config/phase/timings snapshot |
| POST | `/api/v1/jobs/<ID>/cancel` | `{}` → cancellation snapshot |
| POST | `/api/v1/jobs/<ID>/forget` | `{}` → forget terminal metadata |
| GET | `/api/v1/jobs/<ID>/image` | Verified successful PNG only |

Image serving opens through controller-owned directory descriptors, rejects
symlinks/replaced directories, and rechecks the saved digest on the returned
bytes. The browser checks job schema/state/config/seed/timing types and requires
the image URL to equal the exact local route derived from the validated job ID.
Status and Plan's closed gate/blocker contract was updated together with JS
and its tests; old “adapter not implemented” claims are removed.

## Evidence

- Guarded full CPU suite: **2550 passed, 2 skipped** (22.55 s), including
  shutdown/resource-release and startup-refusal checks.
- Existing HTTP + real Chrome planning/export regression: **105/105** executed
  checks passed.
- `node scripts/web_generation_check.mjs`: explicit CPU fixture over the actual
  HTTP server and real Chrome. Covers submit, exact uint64 seed, image loading,
  history/reload, UI cancellation, forget, and no overflow at 1440/375/320 px.
  Synthetic fixture execution lives only under `tests/fixtures/`.
- `node scripts/web_generation_check.mjs --real`: actual V100/FP16 Qwen weights,
  1024², 20 steps, seed 42, red-cube prompt. Engine wall **86.707 s**; load
  **28.457 s**, encode **17.098 s**, denoise **21.653 s**, decode **1.286 s**,
  save **0.308 s**. Preview and History loaded; page reload recovered the result;
  desktop/mobile checks passed; no uncaught browser exceptions. PNG visually
  inspected. This is one successful run, not a quality acceptance suite or a
  matched ComfyUI benchmark. No new calibration records were collected.

The real check's result, public job JSON, server log and screenshots were retained
locally under `output/w3-validation-2026-10-24/` (ignored runtime artifacts).
The temporary test server and Chrome were stopped; GPU returned to 0 MiB used.
