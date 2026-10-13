# Owned engine lifecycle

The core accepts an optional cancellation Event and phase callback. Both
validate before adapter entry; unconfigured output still refuses first.
ManagedGenerationAdapter extends the unchanged legacy phase protocol with
set_cancel_event/close. Production implementations must honour cancellation
in encoder children and sampler callbacks; the core checks between phases
and again after a phase callback. No forced thread termination is implied.

Once adapter entry starts, ordinary failures, partial load, callback faults,
acknowledged cancellation and interrupts always attempt close. Intermediate
references are dropped first. Cleanup faults cannot mask primary failures
(including interrupts); after a verified saved artifact, an ordinary cleanup
fault is diagnostic and cannot hide the successful file. A fresh interrupt
inside cleanup is not swallowed. Close must be idempotent; device allocators
are not touched by the core. Legacy fixture adapters without close remain
valid. Production per-job process ownership is needed to release even a
stuck/failed backend or device context.

Phase reports reflect entry into actual engine phases, not progress percent,
fit or GPU proof. The save/verification boundary wins late cancellation;
before it, GenerationCancelled propagates and the internal controller
recognizes it only alongside an actual cancellation request. There is no
adapter loader/web activation or generation-worker CLI in this milestone.

22 guarded lifecycle regressions cover order, partial failures, callback
cancellation/faults, interrupts, cleanup faults, early refusal, verified
save and internal-controller cancellation. New Path-based fixtures also
exposed a pre-existing GenerationConfig bug: normalized Path was stored but
the local variable was not updated before .strip(). Fixed with14 regressions
for all six path fields, nonexistent lexical paths and wrong-kind errors.
