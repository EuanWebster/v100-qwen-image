# Q4_K_M evaluation — 2026-10-25

Requested: check the practical difference from Q8/FP16 before deciding whether
Q4 should be offered. This is an isolated evaluation; **the production app still
refuses Q4**. No default, loader, driver or service configuration was changed.

## Artifact and method

Producer: [Unsloth Qwen-Image-2.1 Q4_K_M](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/blob/2c31ccd392b367a6637841a143813320a02dff55/qwen-image-2.1-Q4_K_M.gguf).
Same pinned repository revision as our Q8 artifact. Verified **4199565024 bytes**,
SHA256 `631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe`.
`models/q4_eval_manifest.json` records the identity; weights are ignored under
`models/q4/`. The source-model revision is a binding declaration, not proof of
the producer's conversion history.

Actual tensors: 163Q4_K, 28Q5_K, 1Q6_K, 4Q8_0, 65F32, 4BF16. Q4_K_M therefore
uses mixed precision. File size is about45% below Q8's7640860384 bytes; this
does not predict end-to-end VRAM or speed.

`scripts/prepare_q4_experiment.py <new-directory>` copies the source package and
adapts the Q8 attachment path in that copy. It retains the V100/native-SDPA,
verified base assets, BF16 auxiliary restoration, CPU encoder, cancellation,
phase-exclusive VAE and artifact confinement checks. It adds a distinct Q4
sidecar/combined artifact identity and **actual q4 calibration labels**, never
records Q4 measurements as Q8. Allowed quantized tensor types match the mixed
Q4_K/Q5_K/Q6_K/Q8_0 file; computation still uses FP16 pure Torch dequantization.
No optional downloaded kernels or native four-bit matrix kernels are claimed.

Actual GPU trials use
`output/q4-validation-2026-10-25/experiment-v3/src` via `PYTHONPATH`. Its
`provenance.json` and `experiment.diff` capture the source snapshot/adaptations.
Initial v1/v2 setup attempts were rejected before GPU trials; v3 corrects the
mixed tensor allowlist and preserves separate Q4/Q8 enum members.

Five sequential trials: cubeQ4, portraitFP16/Q4, posterFP16/Q4. All1024²,
40steps, seed42, no CFG, native SDPA, identical encoder/VAE. `runs.json` records
the prompts, exact commands and exit status; `calibration.json` records actual
phase/device/host peaks and runtime/artifact identities. For cube, the earlier
matched FP16/Q8 images are reused; the copied FP16 arithmetic path is unchanged.
Only denoising times are directly compared: encoder cache states and initial
loading/verification differ between trials.

## Initial cube result

Q4 completed and the image was inspected: coherent cube, visually close to the
earlier FP16/Q8 samples. Denoise50.13s, peak6548.5MiB during denoise, but
9320.5MiB during decode/save. Load33.39s, encode16.19s (cache miss), decode1.05s.
Q8 cube: denoise44.84s, denoise/overall peak9678.5MiB. FP16 cube: denoise42.14s,
overall peak15612.5MiB. Thus Q4 saves substantial transformer memory but the
unchanged VAE limits the whole-run reduction versus Q8 to about3.7% here.
Q4 denoising is about11.8% slower than Q8 in this point trial.

## Full-recipe results

All five1K runs exited0 and their outputs were inspected.

| Scene/mode | Denoise seconds | Denoise peakMiB | Whole-run peakMiB |
|---|---:|---:|---:|
| Cube FP16 (earlier baseline) |42.14|15612.5|15612.5|
| Cube Q8 (earlier baseline) |44.84|9678.5|9678.5|
| Cube Q4 |50.13|6548.5|9320.5|
| Portrait FP16 |42.62|15634.5|15634.5|
| Portrait Q4 |50.53|6568.5|9320.5|
| Poster FP16 |42.84|15634.5|15634.5|
| Poster Q4 |50.74|6568.5|9320.5|

Q4 maintained coherent images in these samples. Portrait eyes/skin/hair detail
changed slightly. Both posters correctly rendered the requested headline and
opening-hours text; font geometry, spacing and cup/saucer illustration changed.
No obvious collapse or gross quality loss was observed. This is three prompts
at one seed, not a blinded human evaluation or a guarantee for other subjects.

Pixel differences versus FP16 (not perceptual quality scores): cube RGBMAE2.89
and PSNR30.77dB; portrait4.57/31.73dB; poster8.76/16.84dB. The poster's layout
changes strongly penalize pixel alignment even though its lettering is correct.
Raw metrics and image digests: `output/q4-validation-2026-10-25/metrics-1k.json`.

Open `output/q4-validation-2026-10-25/comparison.html` for original linked PNGs
side by side. Real Chrome verified all six images load at1024²; full-page
screenshot `comparison-desktop.png` was inspected. No generated/resynthesized
comparison pixels were substituted for the actual outputs.

The short2K/four-step capacity probe also exited0, saving a valid2048² RGBA PNG.
It took34.32s to denoise and6.93s to decode. Denoise peak7742.5MiB, but
**decode/save peak15942.5MiB**, essentially the same as the earlierQ8
15946.5MiB. The allocator again logged a failed1214251008-byte allocation
then recovered. Q4 therefore did not fix this decoder bottleneck. Four steps
are underconverged: this probe is capacity evidence, not a2K quality comparison
or clearance of a40-step Q4 recipe. There are30 total phase observations in
the separate Q4 evaluation log. `metrics-2k-capacity.json` summarizes the probe.

All evaluation processes have exited; `./start-web.sh` restored the app on
localhost/LAN/Tailscale port8765. Prior idle queue metadata is in
`output/web/history-before-q4-test-2026-10-25.json`; all images remain in place.
The preparation regression passed under the guarded CPU runner. Only research
scripts/sidecar/docs were added; production Q4 remains refused.

## Practical interpretation

The smaller file and lower transformer residency are real. Current1K end-to-end
GPU peak barely improves over Q8 because decoding dominates; at2K the same VAE
is already the capacity limit. Q4 loading was33–35s versus the earlierQ8's47s,
but sampling was slower (50–51s versusQ8's44.8s orFP16's42–43s). Total-run
timings also include different cache/verification states and must not be sold
as a controlled throughput result. A Q4 integration would be an optional
capacity mode, with a separately reviewed loader/manifest/runtime identity;
keep the maximum-quality FP16 default and Q8 option available.
