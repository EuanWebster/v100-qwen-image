# Fetcher review — offline acceptance

Codex direct review, 2026-10-13. Scope: `tools/fetch_models.py`, the generic
manifest-backed operator fetcher. No model download/resume, real-asset
mutation, dependency changes, service activation or RAM/GPU fit claim.
The earlier separately approved partial live transfer remains historical
evidence in `models.md`; its current on-disk state was not rehashed here.

## Corrections

- Bind the explicit root absolutely before disk/transport callbacks can
  change CWD. Walk/create destination directories via anchored no-follow
  directory FDs, refusing appeared symlinks and observed parent replacement.
- Record existing partial identity, not just length. Reuse empty partials
  without unlink/truncate; refuse same-size replacement before append.
  Partial write opens are nonblocking/no-follow. A kernel-owned proc parent
  FD path exposes the real destination to the CPU open audit hook (CPython's
  ordinary open audit event omits `dir_fd`); the guard was not weakened.
- Rehash an opened regular, singly linked partial, at most expected size+1
  bytes, checking size and before/after/opened/named identity, including
  ctime to detect observed mutation with restored mtime.
- Publish only after exact full size/hash match and fresh snapshot checks.
  Linux linkat follows the kernel-owned `/proc/self/fd/<source>` alias of
  the opened inspected inode; replacing the staging name inside publication
  cannot substitute a different inode. Existing finals are never overwritten.
  Recheck both names before removing staging; ordinary cleanup refusal leaves
  verified-final/hardlinked debris and warns. Files are synced before publish;
  directory-sync failure is a typed refusal explicitly reporting the already
  published final and unconfirmed durability.
- Unrecoverable-byte cleanup removes only the still-matching owned staging
  name. Replacement or cleanup failure preserves it and says so; no false
  "removed" diagnostic. Descriptor-close faults cannot mask an active primary
  refusal. Ordinary stream-close faults warn with type only, preserving the
  body result/error and never echoing untrusted error text; even a broken
  warning sink cannot mask the primary failure.
- Content-Range is ASCII, full-match and capped512characters before numeric
  conversion; body cannot exceed its declared range. Reject invalid/non-byte
  or over1MiB transport chunks before any write. Unsupported encoding values
  produce typed, sanitized errors rather than leaking server text.

## Evidence and limits

30 new offline regressions; focused fetch suite129passed. Tests use tiny
local fixtures and fake streams only, including actual FIFO/symlink swaps,
same-size replacement, bounded growth, restored-mtime mutation, publication
replacement, cleanup/close/fsync faults, malformed headers/chunks, directory
creation/transport races and callback CWD drift. Full dual-version suite
results are recorded in `code-review.md` when the milestone is accepted.

Fetch is Linux-specific: no-follow/nonblocking directory-FD APIs, proc FD
aliases, same-filesystem hardlinks and file/directory fsync are required.
Plan/help remain offline and read-only. These snapshots target settled or
cooperatively managed filesystems, not hostile simultaneous inode mutation;
there is no atomic compare-and-unlink guarantee. No cross-process fetch lock
was added: simultaneous fetchers must still not share a staging tree. A final
can remain after a post-link refusal; inspect/rerun the normal verified-final
plan, never blindly remove or overwrite it. No proof of model loading,
processor fidelity, complete local assets, calibration or GPU operation.
