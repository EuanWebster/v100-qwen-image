# Verification tooling and persistent setup review

Codex direct review, 2026-10-13. This is acceptance of tooling corrections,
not of real model loading, numerical fidelity, memory fit or GPU execution.

## Interface checker

`tools/check_pinned_interfaces.py` now uses bounded2MiB regular-file
snapshots with no-follow/nonblocking opens, initial-size+1 read bounds and
opened/published identity checks. Stable explicit aliases remain supported.
Parser and subsequent AST-analysis recursion failures are typed contract
failures, not tracebacks. The input-error/contract-drift exit distinction
remains2/1, and absent external trees still skip, never pass.

Previously some lookalikes could satisfy code checks: nested/duplicate/
overwritten definitions, async methods replacing the synchronous ABI,
boolean/float defaults numerically equal to pinned integers, wrong norm or
lambda/index targets, overwritten hook handles, unrelated `backend=None`
calls and extra/non-name processor-list entries. These now refuse against
the explicitly surveyed source shape. No claim of general static binding,
data-flow/control-flow analysis, execution, numerical fidelity or absence
of network behavior; source markers remain separate textual tripwires.

33new regressions; focused checker52passed+1skipped. Actual cached upstream
trees' HEADs independently checked against pinned commits
`c6df88a511a98740646ee55577b590c9852650ce` and
`856157a2f3e9594954310df18fdccc31ffddebe9`; tracked source diffs were empty.
The checker53/53 passes (38AST/15markers) on those trees. Installed package
source also passed53/53, but package provenance is not a byte attestation.
The checker itself still does not authenticate an arbitrary caller's tree.

## Persistent setup

`bash scripts/setup_review_envs.sh --check` checks all fixed environment
targets, exact Python patch versions, isolated/canonical interpreter
prefixes and dependency consistency without sync/install/download/mkdir.
Actual `.review-envs/py311`, `py313`, `test311` passed; ML environments each
contain63compatible pinned distributions. Compatibility is not exact-pin
or production-vision approval: use the identity verifier separately.

Setup mode preflights the entire fixed target set before any synchronization:
root/cache/runtime/environment/bin aliases, non-directories, unexpected
versions, non-venv/base interpreters, redirected prefixes or malformed
existing environments refuse. `.venv` cannot become a synchronization
target through those paths. Created environments are rechecked before use.
17new tests use fake interpreters/uv on disposable fixture trees only;
no real installation or package changes were made by this review.
This is a cooperative setup check, not a sandbox for hostile executables
or arbitrary concurrent filesystem mutation.

## Assertion-based witnesses

All six encoder identity/assets/synthetic checks, including the private
selected-child bootstrap, refuse `-O`, `-OO` and `PYTHONOPTIMIZE` before
work. Otherwise stripping assert statements could print misleading PASS
rows despite failed expectations.18new guarded subprocess tests verify
refusal/no-work/no-PASS; normal pinned checks remain accepted: identity6,
memory-refusal3, supplied-model/helper36, miniature-worker12 per interpreter,
selected-child8 each cross-version direction. Private synthetic seams only;
actual production memory/vision refusals are restored and observed.

## Browser helper lifecycle

First CDP timeout now rejects every pending exchange immediately, retaining
the original deadline/cause. Explicit cleanup abort releases pending waits
even without a WebSocket close event. Browser/child timeout races clear
their losing timers, so completed cleanup does not keep Node alive for an
otherwise unused deadline. Nine pure lifecycle checks pass (three new);
raw byte-framing7/7 remains separate, and real Chrome is exercised again
by the served105-check and standalone61-check scripts. Cleanup still signals
only the group's own spawned child; sandbox defaults are unchanged.

Full regression found a stale test expecting the old global "browser checks
pending" footer, changed in17e5b7c after that round's last full-suite run.
The test now verifies the correct pure-check disclaimer and pointers to
separate real-browser checks, not the obsolete pending claim. This is a
test/diagnostic-contract correction, not suppression of browser failures.

Final whole-suite/browser results and remaining scope are recorded in
`code-review.md`; no service, ML lock, actual model or generation changes.
