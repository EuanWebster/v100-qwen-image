# Project: v100-qwen-image

Hand-rolled Qwen-Image-2.1 image generation pipeline for a Tesla V100 16GB (fp16/GGUF, engine+CLI, modelled on the LTX-2.5 ROCm build)

<!-- memctl:begin (managed block — regenerate with `memctl project-init`) -->
## Shared memory vault (managed)

This project participates in the shared LLM memory vault (`~/Documents/Obsidian Vault`).

**Task slug:** use this repository's name as the task slug (e.g. `memctl brief
<repo-name>`), so the project is always discoverable from its own name. A more
specific work item may append a suffix (`<repo>-<topic>`).

**Session start, before acting on any task:** run

    memctl brief <task>

It prints the task's STATE file (`memory/state/<task>.md` — current state of play)
plus its most recent handoff. Do not work from your own assumptions about a task's
state until you have run this.

**During work:**

- **Commit your work to this repository — mandatory.** The vault records *state*;
  this repo's git records *the work itself*. Make small, scoped commits at every
  milestone; never leave finished work uncommitted.
- At each milestone: `memctl state-write <task> "<state of play>"`
- **At 85% of the context window: handoff → compact → re-read → continue.** Write the
  state and the handoff FIRST, then compact (your harness's own compaction), then run
  `memctl brief <task>` to re-read what you just wrote, and carry on. Never compact
  first and hope you remember — the handoff is the part that survives.
- **Wrap up with a handoff — every session, without exception.** Task done, 85%
  context, or session ending:
  `memctl handoff <slug> --state <task> "<completed / in-flight / next action>"`.
  If the user says "wrap up", that means: final `state-write` (if the state
  changed) then `handoff`. Thin snapshots only; detail goes to events
  (`memctl event-add <slug> --type <event|milestone|incident|attempted>`)
- **Return-turn rule.** If you resume this task and its `memctl brief` shows a
  STATE/handoff that predates work done since (user walked away mid-session),
  write a handoff capturing that work before doing anything else — the resume
  turn is the last moment the in-flight state is complete.
- Facts: `memctl fact-add <entity> <predicate> "<text>"`

All writes identify you with `--agent <name>` and commit to the vault's git
history. Full protocol: `~/Documents/Obsidian Vault/AGENTS.md`.
<!-- memctl:end -->
