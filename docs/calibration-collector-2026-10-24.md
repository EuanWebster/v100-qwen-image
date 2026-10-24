# Measured-memory calibration: first exact observations (2026-10-24)

This records the first calibration evidence produced by the opt-in collector,
not a fit claim. The collector runs inside the real generation process and
writes one `calibration_observation.v1` record per phase only after every
exactness field has been measured. Both runs below used the real pinned
assets and the real CLI on the V100; neither image was regenerated for the
record.

## What was measured

- **Device-wide GPU used memory** from NVML, every 50 ms, through a handle
  looked up by the CUDA device UUID (never by assuming NVML/CUDA index
  order). This includes the CUDA context, the Torch allocator's cached
  blocks and any other process on the device.
- **Owned process-tree RSS** (the CLI process plus the CPU encoder child)
  every 50 ms. This is a sampled current-RSS peak, not a kernel high-water
  mark; a spike shorter than the interval can be missed, and summed child
  RSS can double-count shared pages.
- **Torch per-phase peaks** (`max_memory_allocated` / `max_memory_reserved`),
  reset at each phase boundary.
- **Conditioning facts read after encoding**: padded positive/negative
  sequence lengths actually processed, CFG activity, encoder/execution
  dtypes, KV policy and the encoder's own last cache decision. An
  unmeasured cache decision refuses the record.
- **Identity**: CUDA/NVML-bound device UUID, architecture, driver, CUDA
  build; generation-runtime digest; complete asset-set digest; pinned model
  revision and the five phase names (`load`, `encode`, `denoise`, `decode`,
  `save`).

`denoise` here is the **whole-loop** peak. It is deliberately a distinct
policy phase from `denoise-step`, which requires an exact step index; the
two never match. A record missing any exactness field is refused by name at
the bridge, so legacy records remain loadable but ineligible.

## Environment

| | |
|---|---|
| GPU | Tesla V100-SXM2-16GB, `sm_70`, UUID `GPU-4a20902a-772a-15a9-6d1e-8b6bc2820751` |
| Driver / CUDA build | 580.178.04 / 12.6 |
| NVML raw total | 17,179,869,184 B (16.00 GiB) |
| CUDA usable total | 16,928,342,016 B (15.77 GiB; 240 MiB driver-reserved) |
| Model revision | `d26bb61231c349cf6b7896fa83353113880e1ba3` |
| Artifact-set digest | `729fd317ddffd127b68c2acdfd56c1a93612c5c274b9ba61d710b3deeb4eaf73` |
| Runtime digest | `35951fc2229d9e0297280f4bf26a895f7f9e9e2ea514eb218bc35bd20872f361` |
| Execution | resident fp16 transformer, sdpa, KV cache on; encoder bf16, execution fp16; CFG inactive (scale 1.0) |
| Conditioning | 24 padded prompt tokens, no negative branch, embedding-cache miss (fresh cache) |

NVML and CUDA disagree on "total" by design: NVML reports the raw device
total while CUDA reports usable memory after driver reservation. The
collector therefore binds the device by UUID and requires the two totals to
agree within 1 GiB rather than demanding equality.

## Commands

```bash
# 1024², 40 steps (fresh output and cache dirs)
.venv/bin/python -m qwen_engine --run \
  --manifest models/manifest.json --models-dir models/assets \
  --prompt "a red cube on a white table, studio photo, sharp focus, soft daylight" \
  --width 1024 --height 1024 --steps 40 --seed 42 \
  --output-dir /tmp/qwen-cal-1k --cache-dir /tmp/qwen-cal-cache \
  --log-dir /tmp/qwen-cal-logs --attest-unbounded-host \
  --calibration-record /tmp/qwen-calibration.json

# 2048², 20 steps, separate fresh cache so this is also a cold encode
.venv/bin/python -m qwen_engine --run \
  --manifest models/manifest.json --models-dir models/assets \
  --prompt "a red cube on a white table, studio photo, sharp focus, soft daylight" \
  --width 2048 --height 2048 --steps 20 --seed 42 \
  --output-dir /tmp/qwen-cal-2k --cache-dir /tmp/qwen-cal-2k-cache \
  --log-dir /tmp/qwen-cal-2k-logs --tiled-decode --attest-unbounded-host \
  --calibration-record /tmp/qwen-calibration.json
```

The second run appends to the same record file: it contains ten
observations, five per resolution. Record sha256:
`d20784bc0b1920042824f48f4dcd87af68be75b4015b7ac8f8b11a5a856654bb`
(/tmp artifacts are deliberately not committed).

## Results (GiB; 1 GiB = 1024³ B)

### 1024², 40 steps — wall 1m47s

| phase | allocated | reserved | device used | process RSS | owned tree |
|---|---:|---:|---:|---:|---:|
| load | 13.889 | 13.920 | 14.458 | 19.160 | 19.160 |
| encode | 13.889 | 13.920 | 14.458 | 1.505 | 16.317 |
| denoise | 13.889 | 14.641 | 15.247 | 1.559 | 1.559 |
| decode | 6.930 | 14.641 | 15.247 | 3.371 | 3.371 |
| save | 0.644 | 8.490 | 9.098 | 2.158 | 2.158 |

Output `543ad795dfd54728a50d4bbd98943db6.png`,
`sha256 8d274d4fc7f60ebf475a234ad65d4a22df7690f200655cd218655a6b5f6898c6`
— byte-identical to the pre-collector anchor for this prompt/seed. The
collector did not perturb generation.

### 2048², 20 steps — wall 3m39s

| phase | allocated | reserved | device used | process RSS | owned tree |
|---|---:|---:|---:|---:|---:|
| load | 13.889 | 13.920 | 14.458 | 18.506 | 18.506 |
| encode | 13.889 | 13.920 | 14.458 | 1.505 | 16.344 |
| denoise | 14.790 | 15.375 | 15.698 | 1.559 | 1.559 |
| decode | 13.353 | 15.092 | 15.698 | 2.659 | 2.659 |
| save | 1.283 | 14.965 | 15.573 | 1.699 | 1.699 |

Output `1ca2d9af3f734d4c9795c172d8a65af2.png`,
`sha256 7a5aed2fbbbd3c07a967d95154f7d476ebdc38da38b0e912e42bce2d7b31ef32`.

The 2K device peak is 15.698 GiB (16,074 MiB) of a 16.00 GiB raw device:
about 0.30 GiB total headroom with zero configured margin. The `save` row
is conservative because its window is seeded at phase start, while the
`decode` peak is still resident. These are measured peaks, not comfortable
recommendations.

## Policy consumption (verified)

Bridging all ten observations and querying the denoise phase with a fresh
resource snapshot (`gpu_available = 16 GiB`, host MemAvailable 63.4 GiB,
4 GiB host headroom):

| query | required GPU | required host | 0 GiB GPU headroom | 1 GiB GPU headroom |
|---|---:|---:|---|---|
| 1024²/40 denoise | 15.247 GiB | 5.559 GiB | allowed | refused (`gpu_budget_exceeded`) |
| 2048²/20 denoise | 15.698 GiB | 5.559 GiB | allowed | refused (`gpu_budget_exceeded`) |

Exact-matching was also checked: changing only the step count (40→30) or
only the runtime digest made the decision refuse with
`exact_calibration_missing`. A matching record is never interpolated across
resolutions/steps, and stale/future records remain blocked by the policy's
age rules.

## Defects found by running it for real

1. **Device-profile gate compared types, not values.** The initial check
   `(type(major), type(minor)) != (7, 0)` refused every GPU before sampling.
   Fixed to compare the values and regression-tested; the real run then
   reached the sampler.
2. **NVML/CUDA total equality was wrong.** CUDA reports 15.77 GiB usable
   while NVML reports the 16.00 GiB raw total. The collector now requires a
   UUID match plus totals within 1 GiB.
3. **Setup was not atomic.** A failure after opening NVML or starting the
   sampler could leave both behind. `start()` now closes them on any
   failure, and the CLI binds adapter facts before sampling begins.
4. **`abort()`/`finish()` could mask or leak shutdown errors.** `abort()`
   no longer masks the primary failure; a sampler that will not stop is a
   measurement failure in the normal path.

## Honest limits

- Interval sampling can miss a spike shorter than 50 ms. It is not a kernel
  high-water mark, and the record does not prove coverage of every
  microsecond.
- Summed child RSS can double-count shared pages.
- Device-wide used memory includes any other process on the GPU. The
  measurements above were taken with the GPU otherwise idle; a busy device
  would inflate them (conservative for fit, but not comparable).
- The runtime digest observes the CLI process; the separate CPU encoder
  child is covered by recorded dtypes, cache decision and the fact that it
  runs the same installed package set, but its imports are not separately
  attested.
- A calibration pass is a conservative prerequisite, not a guarantee that a
  future allocation succeeds, stays fragmentation-free, or that another
  service will not claim the remaining memory.
- 2K remains experimental and tight. A 1 GiB GPU headroom policy refuses it
  on this device; only a zero-margin policy passes.

## Reproducing the policy query

```python
import json, time
from qwen_engine.bench import CalibrationLog
from qwen_engine.calibration import calibration_evidence, policy_descriptor
from qwen_engine.mem_preflight import MemorySnapshot
from qwen_engine.memory import ResourceEvidence, assess_memory

log = CalibrationLog.from_dict(json.load(open("/tmp/qwen-calibration.json")))
evidence = tuple(calibration_evidence(o) for o in log.observations)
denoise = next(o for o in log.observations
               if o.phase == "denoise" and o.resolution == (2048, 2048))
decision = assess_memory(
    policy_descriptor(denoise),
    ResourceEvidence(time.time(), MemorySnapshot(63 * 1024**3, "unlimited"),
                     evidence[0].device, True, 16 * 1024**3, 16 * 1024**3),
    evidence, host_headroom_bytes=4 * 1024**3, gpu_headroom_bytes=1024**3,
    max_resource_age_s=30, max_calibration_age_s=3600)
assert not decision.allowed and "gpu_budget_exceeded" in decision.reasons
```
