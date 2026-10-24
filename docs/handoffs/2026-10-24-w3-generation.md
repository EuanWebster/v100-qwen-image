# W3 handoff — 2026-10-24

User requested W3. Implemented opt-in local web generation through the working
CLI/default adapter/core path. Run command and full contract/evidence:
[web-generation-w3.md](../web-generation-w3.md).

## Delivered

- `web/generation.py` bridges validated browser config to existing serial
  `JobController`, manifest verification, default adapter and `core.generate`.
  Exact seed strings, per-job owned output directories, real phase/timing
  snapshots, private failure diagnostics and cooperative cancellation.
- `web/server.py` adds submit/list/status/cancel/forget/image routes through
  the existing Host/Origin/JSON/header/body policy. Generation is opt-in via
  `--enable-generation` and requires four explicit runtime paths. The CPU
  encoder host attestation is the same opt-in flag as the CLI. SIGTERM/Ctrl+C
  requests cancellation and waits up to 60 seconds for the worker.
- `jobs.py` can read a successful artifact through its owned directory FDs,
  checking file/link/directory identity and digest on returned bytes.
- Static workspace now enables Generate only after recognized enabled status,
  renders actual job phase/timings and PNGs, recovers session History on reload,
  supports cancellation, download/open, exact-config Generate again, and
  terminal metadata forgetting without file deletion. Lost submit responses
  never trigger automatic POST retries.
- Updated stale status/plan gates and coordinated JS closed validators/tests.
  Submission readiness is separate from unknown hardware/calibration evidence.
  Root handoff and PLAN current table/W3 milestone now reflect implemented W3.

## Verified

- Guarded CPU full suite **2550 passed, 2 skipped** in 22.55 seconds.
- Existing standalone HTTP + real Chrome planning/export regressions
  **105/105 executed checks passed**.
- New real-Chrome W3 CPU fixture checks passed, including cancellation/forget,
  exact uint64-max seed, preview/reload/history, and 1440/375/320-pixel layouts.
- New `scripts/web_generation_check.mjs --real` used actual V100/Qwen weights
  through browser form submission: 1024²/20 steps/seed42/red-cube prompt,
  **86.707 s** engine execution (load28.457, encode17.098, denoise21.653,
  decode1.286, save0.308 seconds). Actual PNG visually inspected, Preview and
  History loaded, reload recovered, layout and browser exception checks passed.
  Local retained evidence: `output/w3-validation-2026-10-24/` (ignored), with
  `result.png`, `job.json`, `server.log` and desktop/mobile screenshots.
- Temporary browser/server stopped cleanly; GPU returned to 0MiB used/52°C.
  No AMD/service/environment changes made. NVIDIA ComfyUI stays stopped.

## Limits / next work

History lasts for this server process, max32 records/four pending; files stay
on disk but there is no restart/CLI artifact index. Cancellation is cooperative
and phase progress has no invented step percentage. Adapter execution policy
remains FP16 resident auto/SDPA; >4096 latent tokens require tiled decode.

Exact calibration memory preflight is **still not wired into CLI/web jobs**.
This W3 follows the current working CLI policy, with real adapter/encoder gates;
submission availability does not prove memory fitness. Prior measured 2K has
~0.30GiB raw headroom; the conservative 1GiB GPU reserve refuses the recorded
configs. Source changes may invalidate exact calibration runtime identities.
No new calibration or ComfyUI comparison was done. Human quality acceptance
of the earlier six-case suite remains pending.

Preserve user's pre-existing `sandbox.yaml` modification and untracked `out/`.
The earlier CLI cache diagnostic fix is `bbafcb6`; explicit `--cache-dir cache`
remains required. Shared memory must be refreshed before the next task.
