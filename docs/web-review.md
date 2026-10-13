# Web planning workspace — reviewed CPU milestone

Codex direct review, 2026-10-13. Read the complete API/server and static
JS/CSS/HTML source. This accepts the current planning/config-export
workspace only, not image generation, host/Tailscale exposure or model fit.

## Corrections

- Manifest metadata uses the same strict JSON decoder as the engine loader;
  duplicate keys and nonfinite literals cannot become "valid" UI metadata.
- WebRuntime owns a deep immutable JSON snapshot of startup metadata.
  Public projection accepts typed schema-v1/count fields and closed reason
  codes only; malformed internal metadata becomes invalid and its blocker
  agrees, rather than leaking unexpected strings/paths.
- Startup manifest read refuses initial oversize before any bytes, reads at
  most initial size+1, and checks opened/published size, inode, timestamps
  and type. Existing final-symlink refusal, nonblocking regular-file policy,
  one-shot1MiB cap and cleanup behavior remain. This is metadata evidence,
  not model verification or a hostile-concurrent-filesystem guarantee.
- Mixed programmatic request/config mapping keys produce typed API errors,
  not mixed-key sorting failures. Browser JSON already supplies string keys.
- Frontend manifest parsing recognizes exactly schema-v1 and nonempty valid
  artifact sets. A coherent plan for *different* settings is rejected before
  rendering: all11editable echo fields must equal the submitted configuration,
  regardless of JSON key order. Current/stale labels cannot grant validation
  to a different request; draft text remains untouched.
- Idle worker queue reads poll shutdown state, so failure to enqueue a
  sentinel does not leave an idle worker blocked forever. Auxiliary port
  checks bound digit conversion. Socket timeouts are documented honestly as
  **idle**, not total request deadlines; still-live daemon references remain
  retained after finite joins, never falsely reported stopped.

## Verification

25new guarded regressions; focused web/API/server/static142passed. Together
with bundle/fetch/logger fixes, full guarded suites1762passed+2skipped each
on Python3.11.15/3.13.5 (65new regressions since the prior1697 milestone).

Real Chrome, sandbox enabled, ephemeral loopback server and disposable
fixture profile/downloads only:

- Served integration105/105 with Python3.11.15 backend and105/105 with
  Python3.13.5 backend. Three new checks cover metadata schema/count refusal,
  different-request coherent-echo refusal and restored real-server planning.
- Separate file:// browser check61/61: exact uint64 seed handling, rejected
  original overlong/non-ASCII input, presets/manual dimensions, preview
  aspect, keyboard/hash navigation and focus, accessible invalid fields,
  generation refusal/no new submit fetch, disconnected state and responsive
  layout at1440/375/320px. No fake DOM/node-canvas used.
- Pure raw HTTP framing7/7 and Chrome lifecycle6/6 remain separate evidence,
  never substituted for real-browser checks. Lifecycle diagnostic text no
  longer makes a stale global claim that real-browser verification is pending.

Actual pinned environments also rechecked: identity6, actual-memory refusal3,
synthetic upstream helper/codec36 and miniature four-shard worker12 per
interpreter; selected-child/cache8 in each cross-version direction. These
tools use private synthetic vision/budget/model seams and generated tiny
fixtures; real torchvision/memory policy refusals are restored and observed.
They prove no real processor/checkpoint fidelity, model fit or V100 operation.

## Limits kept explicit

Only literal127.0.0.1 binds; no new public/Tailscale bind or auth policy.
No downloads, real weight reads/hashes, dependency/lock changes, persistent
services or generation endpoint were added. Own diagnostic browser/server
children were closed by the existing scoped lifecycle checks.

The HTTP server is for a cooperative local development environment, not
public deployment: four fixed workers/finite queue and idle socket timeouts
do not provide a total slow-trickle request deadline. Browser raw exports
deliberately preserve exact server bytes without parse/reserialize and trust
the server's validated export contract. Full project/tooling audit and real
runtime/hardware readiness remain separate work; this is not final project
acceptance.
