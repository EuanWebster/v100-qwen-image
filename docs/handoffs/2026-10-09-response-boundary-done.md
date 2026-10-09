# Response/post-exit log I/O boundary — done 2026-10-09

Agent: Qwen (DSH), sole product writer. Task `/tmp/v100-fix-response-task.md`
(root queue item 2): full-review-fixes items 8 (worker response cap only) and
10, plus the two post-exit output-boundary findings appended to the task file
(excerpt unbounded `fh.read()`; `sys.maxsize-1` one-shot buffered read).
Baseline `d509930`; product commit **6351c2f**; this docs milestone on top.
Vault untouched this run (access evidence differs by context — root is
re-checking; no policy change, no vault writes attempted).

Completed:

- Constructor: `max_response_bytes + 1 > sys.maxsize` (incl. `10**100`) →
  typed `EncoderSetupError`, same rule as `max_entry_bytes`; default stays
  the documented 256 MiB.
- Response path: ONE `O_RDONLY|O_NOFOLLOW|O_NONBLOCK` open, `fstat`
  regular-file validation on the OPENED descriptor before any read, cap
  pre-size, then chunked read (≤ 1 MiB per `os.read`, `cap+1` total).
  Symlink → ELOOP refusal without following; FIFO/dir/device → "not a
  regular file" refusal with no block; growth-after-sizing still refused
  pre-JSON; no-response vs unreadable typed errors, exit-first ordering,
  stderr/events/reaping/cleanup unchanged. Honest lstat fallback where
  `O_NOFOLLOW` is absent (Linux target unaffected).
- `_read_excerpt`: same opened-descriptor boundary; tail seek from the
  fstat size; BOUNDED chunked read of `_EXCERPT_READ_BYTES` (the old
  sizeless `fh.read()` allowed root's 1,048,580 B growth-at-read);
  nonregular/unreadable → empty fallback, ordinary logs keep diagnostics.

Evidence: focused `tests/test_encoder.py` 184 passed (174 before, +8
functions/+11 cases); guarded full suite 691 passed + 1 skipped, 7.7 s, no
hangs (FIFO tests SIGALRM-watchdog-guarded). Root probes:
`v100-response-cap-review.py` → `EncoderSetupError` at the constructor;
`v100-response-index-bound-review.py` → `EncoderProtocolError` (schema)
instead of raw `OverflowError`; `v100-log-bound-review.py` is now inert by
design (it patches `Path.open`, which the excerpt no longer uses — the
adapted regression patches `os.read` and asserts the total read bound).

Next (root): independent review/probes of 6351c2f; then queue item 3
(fetch task). Untouched: `sandbox.yaml` (user-owned dirty), `models/`,
assets, manifest, cache/response protocol design, production worker.
