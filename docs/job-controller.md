# Internal job-controller foundation (no live generation API)

`qwen_engine.jobs` is stdlib-only and CPU-testable. It is not imported/wired
by the CLI/web service and cannot create a generation backend: an explicit
trusted executor is required. Generate remains disabled. Fixture executors
create labelled synthetic `.bin` files only inside pytest temporary roots.
No user-facing fake image/job/success/progress claim is introduced.

## Ownership and bounds

- One owned worker thread, serial execution, bounded pending queue (default4,
  max32), bounded retained records (default32,max128),256KiB request cap.
- Validated frozen request snapshots; explicit relative runtime paths bind
  at submission without collapsing lexical symlink/`..` semantics. Submitted
  output_dir must be unconfigured; the controller assigns an exclusive UUID
  directory beneath an existing explicitly chosen canonical artifact root.
- Root directory FD anchors mkdir/open; root and per-job directory identities
  detect observed replacement. Directory creation never overwrites a peer.
- Verified nonempty single-link regular result file, direct job directory
  only, no final symlink/FIFO,256MiB cap; bounded SHA256 read checks opened/
  published identity, size, timestamps and directory identity.
- No implicit disk cleanup. Forget removes only terminal metadata, preserving
  completed/partial files. Queue/history bounds are **not disk quotas**; a
  failed submission after mkdir may leave an empty owned directory. Artifacts
  are settled snapshots, not a hostile filesystem sandbox or image-format/
  quality validation. Future serving must freshly verify bytes and MIME.

States: queued→running→succeeded/failed; queued cancellation terminates before
executor entry; running cancellation becomes **cancelling**, not cancelled,
until the executor ends. Explicit phase reports must follow load/encode/
denoise/decode/save; no invented percentages, clocks or model-fit evidence.
Absent reports stay unknown. Failed executor diagnostics expose only fixed
codes, not arbitrary exception text. Unexpected BaseException still reaches
the thread exception handler; all unfinished records fail, admission closes.

## Cancellation and shutdown limitations

Cancellation is cooperative. Executors check the context and raise
JobCancelled only after releasing resources. A verified saved artifact wins
late cancellation and is reported succeeded with cancel_requested=true,
never silently hidden/deleted. Unrequested cancellation acknowledgments fail.
Close cancels pending/active work and joins for a validated finite timeout
(0–60s). False means the worker is **still live** and its root FD remains
owned until it exits; never fake shutdown or clear its tracking. Self-close
does not join itself. Normal exit closes the owned root FD.

Python threads cannot forcibly stop stuck executors. Production acceptance
therefore still requires a process-owned/reaped generation executor, adapter
resource release, bounded subprocess output/timeouts, fresh asset/runtime/
memory/device gates, image serialization/validation and real model tests.
This module alone provides none of those. It starts no background worker
until a caller explicitly constructs a controller. No production caller
does so at this milestone.

## CPU acceptance

81 guarded tests: serial execution, queue/history/request bounds, immutable
snapshot/scoped verified artifact, pending/running/racing cancellation,
honest noncooperative timed-out shutdown, late save, phases, sanitized
failure/recovery, BaseException fail-all, identifier/timeout validation,
startup FD cleanup, root/job/file replacement/growth, links/FIFOs, id/disk
collision, metadata-only eviction, self-close and owned thread cleanup.
No ML, network, weights, GPU or persistent artifacts required.
