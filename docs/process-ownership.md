# Browser-test descendant ownership

Reviewed 2026-10-14 after resuming the interrupted work at `0e80749`.

The container's PID1 was directly verified as `sleep`. Earlier Chrome test
leaders exited, but some grandchildren were adopted by PID1 and remained
zombies. Successful CDP checks/direct-child exits did not prove full cleanup.
Existing zombies cannot be signalled away or waited by a nonparent; no PID1,
foreign process, container restart or shared-tool mutation was attempted.

`scripts/process_lineage_owner.py` is a private Linux wrapper for terminating
browser/test/tool commands. Before spawning its one command, it becomes a
subreaper **only in its own fresh process**. It adopts orphaned descendants,
waits them and finishes only when the kernel reports no remaining children.
Its flag is not inherited by Chrome. The Node harness owns/reaps the wrapper;
normal Chrome sandbox settings stay enabled.

On normal exit or cancellation, the wrapper escalates TERM→KILL on owned
descendants. Live unreaped command leaders reserve their session/group ID;
adopted children are individually signalled only after `waitid(WNOWAIT)`
confirms child ownership. No group signal after leader reaping, no stale
foreign PID signal, no global subreaper. Proc/permission faults retain pending
ownership rather than fabricating cleanup success.

Node sends TERM only to its wrapper PID, never KILL/unref/group-kill of that
owner. Pending cleanup retains references and the profile/fixture tree and
marks failure; callers use `process.exitCode`, not forced `process.exit`.
Protocol waits remain bounded. Kernel-uninterruptible children can keep the
owner alive beyond those waits; a hard externally forced owner kill/OOM cannot
guarantee descendant cleanup. Never use this wrapper to launch a service that
intends to leave background processes running.

The same private wrapper can contain an explicitly invoked short-lived
`memctl` command's own Git descendants, without editing the shared tool or
vault policy. Git zombies were also observed, but a historical causal
attribution to a particular invocation was not established.

## Acceptance

17 guarded Python ownership regressions on each supported interpreter cover
actual fork-tree adoption/reaping (normal orphaning and TERM-ignoring trees),
missing commands, fixed local prctl/ABI, inventory bounds, PID ownership,
permission/proc failures and leader-reap/group-reuse safety. Ten pure JS
lifecycle checks cover CDP deadlines, owner-only signalling, pending cleanup,
timer cleanup and sandbox defaults. No fake DOM or stub browser acceptance.

Real sandboxed Chrome: standalone61/61; served105/105 with Python3.11.15 and
105/105 with Python3.13.5. Controlled before/after process snapshots stayed
at185existing zombies, including32Chrome zombies: **no new zombies** across
these checks. Existing zombies were preserved, not claimed removed. These
counts are dated snapshots, not permanent host facts.

Whole guarded suites:2435passed+2intentional skips on each interpreter.
Reusable scripts/tests remain on the persistent project mount; only profiles
and disposable fixtures are temporary. No generation/GPU/model-fit claim.

Primary lifecycle references checked2026-10-14:

- [Linux child-subreaper scope and orphan adoption](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html).
- [Linux wait/waitid ownership and reaping](https://man7.org/linux/man-pages/man2/waitpid.2.html).
