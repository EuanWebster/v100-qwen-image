# Optional stage metrics and repeated runs

Metrics are disabled by default. They do not change config/API schemas or the
queue protocol. Use the normal CLI with `--performance-report NEW.json` and
`--run`. The parent directory must exist; an existing file is never replaced.
For the web server, add the operator-only `--performance` flag: each job writes
`performance.json` inside its own output directory. The browser cannot choose
diagnostic paths. The original running app is not automatically upgraded.

Reports separate model initialisation, transformer loading, the initial VAE load
and disposal, encoder request, CPU-child encoder loading / encoding / cleanup,
denoising, transformer cleanup, decode VAE loading, decode, output conversion,
saving, adapter cleanup and total engine job time. Encoder-child stages are
absent on an embedding-cache hit. A killed child may leave no sidecar, which is
an explicit measurement error rather than a successful missing measurement.

Each CUDA leaf stage synchronizes before and after its body and resets local
allocator peaks once at entry. There is no added per-step synchronization or
`empty_cache`. Allocated is live tensor memory; reserved includes cached blocks
already present at entry. CUDA counters are process-local, **not NVML device-wide
usage**. See the separate calibration/decode tools for sampled device-wide peaks.
CPU scopes can contain CUDA scopes and overlap; never sum them to invent total
time. `total_job` covers adapter creation through cleanup, after CLI asset
verification; shell elapsed time additionally includes Python startup and the
first asset verification. Current RSS is sampled at boundaries; reported host
peak is each PID's lifetime high-water mark, not a sampled per-stage/tree peak.
Encoder-child and generation-process RSS peaks must not be added together.

Measurements contain no prompt text, negative text, config paths, exception
messages or hostname. Resolution/seed/steps, requested precision and attention,
actual tile geometry, cache decision, token lengths, runtime/artifact digests and
device facts are recorded when observed. Actual attention kernel dispatch is
unknown (`attention_observed: null`) until separately profiled. Diagnostics do
not constitute GPU memory admission/calibration evidence. Run performance and
calibration collectors separately because both reset allocator peaks. A report
write failure does not hide a primary exception or invalidate a saved image.

## Repeatable command

Use an existing exported CLI configuration with fixed seed and absolute model,
cache and manifest paths. Keep prompts fixed in that input; do not publish private
prompt configs as benchmark diagnostics. From this worktree with the project
interpreter and `PYTHONPATH=src`:

```bash
python scripts/benchmark_generation.py --config /absolute/config.json \
  --output-dir output/bench-fresh --runs 3 --mode fresh-process
# The above plans only, touches no GPU and writes nothing. Explicit execution:
python scripts/benchmark_generation.py --config /absolute/config.json \
  --output-dir output/bench-fresh --runs 3 --mode fresh-process \
  --attest-unbounded-host --run
python scripts/benchmark_generation.py --config /absolute/config.json \
  --output-dir output/bench-reuse --runs 3 --mode process-reuse \
  --attest-unbounded-host --run
```

Only attest when appropriate for the actual host. The reused-process mode
recreates models for every run, matching current web behaviour. It is **not** a
CPU-weight cache or persistent-model worker. Inspect `adapter_cleanup` allocated
bytes over repeated jobs for growth; reserved bytes alone are not a tensor leak.
Fresh Python processes do not mean cold OS page cache. An embedding cache miss
does not mean cold disk/model weights. Keep these distinctions in comparisons.

The command writes original images, per-run reports and a summary. Alternatively:

```bash
python -m qwen_engine.performance_summary run1.json run2.json run3.json \
  --output new-summary.json
```

The summariser (initial implementation by local Qwen, reviewed and corrected by
Codex) rejects failed reports, measurement errors and incompatible configuration,
runtime/artifact identity or devices. It separates hit/miss/unknown cohorts and
reports count, median, min, max and population standard deviation. Null means
unmeasured, never zero. Prompts must be held constant by the benchmark operator;
their contents are deliberately absent from diagnostic reports. Do not report
one sample as a statistically established improvement.

## Quality and rollback

Use FP16/base40/seed42 with unchanged scheduler, prompt and dimensions. Save PNG
references for objects, faces, detailed landscapes, typography, flat gradients
and transparency edges where applicable. For tiling compare **identical latents**
with `scripts/benchmark_vae_decode.py` and inspect full-size tile boundaries.
Record max/mean channel deltas alongside manual review; non-identical bytes do
not alone prove degradation, and matching dimensions do not prove quality.

Neither production precision nor tile geometry has changed in this milestone.
Remove the optional flags to restore default measurement-free execution; revert
the instrumentation commit to remove the tool. No model files/dependencies are
changed. Baseline results and remaining validation are in `performance-audit.md`.
