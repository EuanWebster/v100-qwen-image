---
title: 'handoff: v100-qwen-image fix-json (parser-boundary follow-up)'
type: handoff
source: 'agent:Qwen (writer2035)'
created: '2026-10-09'
task: v100-qwen-image
---

completed: /tmp/v100-json-parse-review.py's six raw failures are typed
boundary rejections (code commit 08993bc on baseline 42b1eb3). The seven
`json.loads` parse boundaries — `cli._load_config`,
`GenerationConfig.from_json`, `load_manifest`, `BenchmarkResult.from_json`
(covering `load_result`), `load_calibration`, `EncodedPayload.from_entry_bytes`
header, `EncoderWorker._result_from_files` response — now catch plain
`ValueError` (JSONDecodeError already subclasses it; the integer-string digit
limit raises plain ValueError) and `RecursionError` (the C decoder's nesting
guard fires before the depth-200 value validation), wrapping into the
EXISTING typed errors only: ConfigError, ManifestFormatError,
BenchSchemaError, _EntryCorrupt, EncoderProtocolError. No limits disabled,
_jsonsafe untouched (depth-200 kept), no catch-all over pipeline/schema
logic, no new exception classes.

Behaviour: cache corrupt parse = miss + quarantine (verified: get→None, one
`.corrupt` file, entry name gone, key reusable); worker failure keeps the
captured stderr on the error, and run-level terminate/reap/scratch cleanup
runs unchanged around the same typed path; CLI giant-integer config = exit 2
with a `config error:` line, no traceback. Normal JSON roundtrips unchanged.

Post-fix review script (read-only, unmodified) — identical on 3.11.15 and
3.13.5: `{"benchmark": "BenchSchemaError", "cache_deep_corrupt": "accepted"
(miss+quarantine, verified separately — no exception escapes), "calibration":
"BenchSchemaError", "cli_giant_integer_exit": 2, "config_from_json":
"ConfigError", "manifest": "ManifestFormatError", "worker_deep_response":
"EncoderProtocolError"}`.

Tests: +19 parametrized regressions (config 4, cli 1, manifest 4, bench 6,
encoder 4) — real 5000-digit tests guarded by `sys.get_int_max_str_digits`,
real 2000-deep cache/worker cases (deterministically typed under either
decoder or schema validation), and direct injected ValueError/RecursionError
wrappers portable across 3.11/3.13. Focused five files: 486 passed both
interpreters. Full suites with QWEN_DIFFUSERS_SRC/QWEN_TRANSFORMERS_SRC=/
tmp/v100_m0_ml/src/*: **772 passed + 1 skipped, rc 0 on 3.11.15 AND on
3.13.5** (`/tmp/qwen-review.bIFHcI/venv/bin/python` + PYTHONPATH=/work/src)
— baseline 753+1skip, exactly the 19 new tests; the single skip is the
run_cpu_tests.sh-only guard.

docs milestone: root admin notes committed AS FOUND (review-queue 11:00
acceptance incl. cache ACCEPTED + the 3.13-availability correction;
next-worker-contract 11:04 design refinement); the historical "3.13 pytest
absent" claim in 2026-10-09-cache-publish-done.md corrected per that note.
sandbox.yaml left dirty/untouched; no network/GPU/installs/real weights.

next: root review of 08993bc; queue's later worker/encoder milestone per the
11:04 contract note (narrow component verifier is a later task).
