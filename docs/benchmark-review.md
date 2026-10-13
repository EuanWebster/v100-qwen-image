# Benchmark/calibration review corrections

Codex, 2026-10-13. CPU-only schema/persistence/math/timing corrections;
no measured GPU calibration or memory-fit claim. Schemas are unchanged.

- Result/calibration files use the shared bounded regular snapshot reader:
  nonblocking/no-follow final descriptor after intentional canonicalization,
  stable symlinks accepted, mutation checks and16MiB cap. Strict JSON refuses
  duplicate keys, nonfinite constants/overflow floats; mixed-type mapping
  keys report BenchSchemaError instead of sorting crashes.
- Persistence uses the same16MiB cap **before any directory/temp creation**,
  atomic temp/fsync/replace, raw-FD cleanup if fdopen fails, and best-effort
  temp cleanup that never masks the primary error. A cleanup failure can
  leave a temp as evidence; no broader deletion/recovery is attempted.
- Finite nonnegative phase medians avoid overflow from adding large floats;
  byte medians use exact integer arithmetic (fractional-byte midpoint floors,
  preserving the previous intended integer result without float rounding).
  Per-step derivation handles huge positive step counts without float-casting
  the denominator. These are synthetic boundary regressions, not plausible
  measured timings or memory on this machine.
- Ordinary PhaseCollector clock faults do not abort a body or mask its error.
  Invalid/negative/nonfinite/overflowing durations are **unmeasured**, never
  zero or partial totals. Once any repeated occurrence is unmeasured, that
  whole phase remains omitted; unmeasured_phases exposes sorted names. If
  every phase is unmeasured, a valid nonempty BenchmarkResult cannot be
  fabricated. Interrupts still propagate normally.
- Calibration append resolves one absolute canonical I/O path before taking
  its matching lock. Later CWD changes cannot redirect it, and intentional
  file aliases stay symlinks while the selected target gets the append.
  Concurrent threads remain serialized; cross-process append loss is still
  an explicit existing limitation, not fixed/claimed here.

36new guarded regressions; benchmark132/132; full suites1697passed+2skip on
Python3.11.15/3.13.5. Run `bash run_cpu_tests.sh` and
`PYTHON=/work/.review-envs/test311/bin/python bash run_cpu_tests.sh`.

Existing calibration-v1 staleness matching remains a **prefilter**, not a
production fit decision. Conditioning, actual hardware/backend/runtime identity
and measured peaks must be accounted for before any future M5.3 activation;
these records alone cannot authorize a load. Unknown/null measurements remain
unknown, not zero. No services, assets, dependency locks or generation changed.
