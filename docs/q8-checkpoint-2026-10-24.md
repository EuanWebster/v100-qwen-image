# Q8 implementation checkpoint — 2026-10-24

**STOPPED at the user's explicit request.** No further development until the
agreed pause ends or the user explicitly resumes. Current code is checkpointed;
remaining work is listed below, not silently treated as finished.

Final evidence: guarded full CPU suite **2644 passed, 2 skipped** (23.13 s).
Real V100 Q8 smoke **passed**, saving
`output/q8-validation-2026-10-24/smoke/fd5488e0d3be4b4e8cbd0c48eddbf805.png`
and five measured calibration observations. Highest observed device-wide GPU
use was **9050.5 MiB** for this 512²/four-step smoke only. The PNG was inspected:
it is visibly underconverged, as expected for a base-model four-step run. This
does not establish full-recipe quality, performance or 2K fit.

The app was restored via `./start-web.sh`: PID2453765, owned execution session30780,
on localhost/LAN/Tailscale port8765. FP16 remains default; Q8 is experimental and
still labeled unvalidated in the UI. No development workers or test runs remain.
The preflight facade is tested but **not integrated into production dispatch**.

This is the first implementation checkpoint, not Q8 quality acceptance or a
claim that all 2K/reference-image modes fit. FP16 remains the default.

## Implemented

- `q8_artifact.py`: strict ML-free sidecar parsing, confined regular-file access,
  streamed SHA/size/container checks, and an identity fingerprint. The existing
  closed full-reference bundle is unchanged. Two local Qwen workers implemented
  the artifact and preflight support modules/tests; root reviewed and integrated.
- `models/q8_manifest.json`: pins Unsloth's Q8_0 transformer at revision
  `2c31ccd392b367a6637841a143813320a02dff55`, size **7,640,860,384 bytes**, SHA256
  `c0ed4b2ffd56cbe9c3df1e4a4098045256484ebe94ba5a7e4338d35ec046baa5`.
  Metadata checked against the producer's Hugging Face API; local bytes verified.
  `source_model_revision` binds our expected base reference profile; it is a
  declaration for review, not proof of the producer's exact conversion history.
- `adapters/q8_loader.py`: uses the already pinned Diffusers single-file Qwen21
  converter and pure-Torch GGUF dequantization, with FP16 computation. It rejects
  optional remote GGUF kernels before ML imports. No FP8/native-INT8 claim.
- GGUF file inspection found 196 Q8_0, 65 F32 and four BF16 tensors. Diffusers
  splits 32 fused MLP matrices during conversion. Packed BF16 auxiliaries are
  explicitly converted to plain FP16 before placement, including the text norm;
  Q8 linear weights stay packed. The first GPU load attempts exposed this issue;
  a regression test now protects that boundary.
- Config/CLI/server-owned runtime paths accept `q8_manifest_path` /
  `--q8-manifest`; browser requests cannot select the path. The host launcher
  supplies the committed sidecar. `--quant q8` requires it; Q4 remains refused.
- The supplied bridge validates quantized parameter type/placement and keeps
  native SDPA, cancellation, phase-exclusive VAE residency and tiled decode.
- Calibration identities bind both the base companions and the exact Q8 artifact.
  FP16 records cannot silently serve as Q8 evidence.
- `run_preflight.py`: a tested exact-memory-policy facade with fresh injected
  resource reads and one bounded calibration snapshot. **Not yet wired into the
  live execution path**; production observation/phase-budget integration remains.

## Validation and limits

The initial guarded full suite passed **2641 tests, 2 skipped** before the final
auxiliary-restoration tests. The final suite/result and real GPU smoke are recorded
in the handoff. Focused checks cover the sidecar, facade, policy boundaries,
configuration, CLI, web path projection, and packed auxiliary restoration.

Real smoke artifacts/logs: `output/q8-validation-2026-10-24/smoke/`, with opt-in
measured records at `output/q8-validation-2026-10-24/calibration.json`. The smoke
is 512×512 / four steps / seed42. It verifies loading, computation and publication;
four-step base-model output is underconverged and is not a quality comparison.

Next: compare Q8 against FP16 at fixed prompts/seeds with the full sampling
recipe, measure 1K/2K timings and resource peaks, visually review the genre suite,
then finish conservative runtime preflight and browser acceptance. Do not promote
Q8 to the quality default based on a smoke test. Official Turbo, editing, alpha
workflow, full canonical ratio catalog and rewriting remain planned work.

## Resume commands

```bash
python -m qwen_engine --run \
  --manifest models/manifest.json --models-dir models/assets \
  --q8-manifest models/q8_manifest.json --quant q8 \
  --prompt "a red cube on a white table, studio photo, sharp focus, soft daylight" \
  --width 1024 --height 1024 --steps 40 --seed 42 \
  --output-dir output/q8-validation-2026-10-24/reference \
  --cache-dir cache --attest-unbounded-host \
  --calibration-record output/q8-validation-2026-10-24/calibration.json
```

Avoid competing GPU jobs: check the running web queue and free the V100 before
benchmarks. Preserve job metadata/images before any server restart. The prior
idle web session was archived to
`output/web/history-before-q8-validation-2026-10-24.json` before testing.

User requested a handoff and a pause after this checkpoint until the later
**2026-10-25 01:50 GMT (01:50 UTC)**, after the ambiguous UK clock-change reset.
The user timer `v100-qwen-resume-reminder.timer` is scheduled for that instant
and sends a desktop reminder only; it does not automatically resume Codex.
