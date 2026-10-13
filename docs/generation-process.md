# Process-owned generation execution (internal, not web activation)

GenerationProcessExecutor is a stdlib POSIX executor for JobController. It
always invokes the fixed generation_worker module in a new session using
an explicitly trusted selected interpreter. No command/backend/profile
override exists. Neither module is wired into the web server; the production
adapter factory still refuses its unimplemented milestone.

## Protocol and bounds

Config snapshots use UTF-8 JSON,256KiB plus1KiB fixed-envelope allowance;
request id/schema/config are explicit. The child validates, installs the
offline socket guard before any ML import, requires configured output and
manifest, freshly resolves assets, and uses the real CLI adapter factory and
engine lifecycle. No model download/fake success/hardware approval implied.

Five phase-entry events have exact schema/id/order/true-integer sequence,
no percentage/timing/prompt/path fields. Parent pipes are nonblocking and
bounded (max1MiB each,16KiB event); stderr is counted/discarded, never exposed
as arbitrary client diagnostics or written to an unbounded log. Result file
is bounded64KiB, no-follow/nonblocking regular single-link input with stable
FD/path signature checks; strict duplicate/nonfinite/depth16 JSON and exact
status/id/exit-code/output-path/error-code coherence. A valid success still
needs all phases and controller's fresh scoped artifact verification.

Child results publish via the existing reviewed anchored no-clobber response
publisher. Error categories are fixed codes; raw exceptions are not returned.
TERM signals an Event in the worker; the core/bound sampler/encoder must
honour it. Signal handlers are restored on exit. Real KeyboardInterrupt is
not silently converted to success. Private test bootstrap/redirects exist
only in tests and are never selected by product code or environment flags.

## Process ownership and limits

Timeout, close, cancellation, malformed/flooding output and callback errors
stop only the live owned process group with TERM→bounded grace→KILL. Direct
child is waited/reaped before acknowledgement; never signal a group after
reaping its leader (PID/group reuse risk). Worker-owned encoder descendants
inherit the session and must be reaped **by the worker** before normal exit.
Group signals alone are not a promise to reap arbitrary grandchildren or
hostile code escaping sessions. No global subreaper/foreign process cleanup.

If reaping cannot finish (e.g. kernel uninterruptible I/O), keep the Popen
and scratch ownership, refuse another spawn, and report unsuccessful close;
never claim cancellation with a still-live direct child. A later finite
close can retry. Default process timeout600s, grace15s; operator can choose
bounded values. Close has its own finite deadline, including retry reaping.

Coherent committed success with clean exit wins a late cancellation/close/
timeout. Forced termination without a coherent clean result cannot establish
success even if files exist; completed/partial output files are retained,
not automatically deleted. Scratch cleanup checks the owned directory identity
and never deletes a replacement peer. This is cooperative local ownership,
not a hostile filesystem or executable-attestation sandbox. Queue bounds
remain distinct from disk quotas.

## CPU acceptance

58 executor +19 worker guarded tests on each Python3.11/3.13: fixed command,
real process startup/reaping, success/phases/artifacts, cooperative and
TERM-ignoring cancellation, timeout, flood bounds, malformed/special results,
actual descendant cleanup, late publication, self/finite close, primary
callback errors, retained synthetic unreapable owner, PID-reuse refusal,
strict schemas/coherence/JSON, setup bounds, root replacement and signal
restoration. A real product child refuses missing assets with ML unavailable.
Synthetic successful backend appears only in the isolated private fixtures.
No actual image generation, CUDA, live API or hardware/memory-fit evidence.
