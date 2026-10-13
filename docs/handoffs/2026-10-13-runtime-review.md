# Runtime identity review correction

Codex, 2026-10-13. Full CPU review started; first confirmed defect corrected:
the old four-package identity policy allowed locked transitive version drift
without cache invalidation. No Qwen jobs/agents, dependency installs/lock changes,
real assets, services or generation activation.

Wheel-shipped immutable 63-package policy exactly matches requirements.lock
(guarded consistency test); validates required versions/registry provenance and
the two exact git URLs/commits. Runtime fingerprint includes Python patch,
all required observations and all optional distribution metadata, including
canonical full direct_url provenance. Duplicate normalized names/corrupt or
oversized inventories refused; own editable distribution metadata excluded
because its sources are already hashed independently. Runtime + source digests
enter existing implementation revision (no generic protocol/schema changes).

Focused identity400/400; both guarded full suites1480/2skip. Actual persistent
ML identity checks6/6 each: all63pins match, missing/unlocked vision still
explicitly refuses. Actual miniature worker regression12/12 each on
Python3.11.15/3.13.5. No production/GPU claim.

Next review: worker parent/child runtime handoff before any cache lookup;
generic encoder transport/cache; bounded filesystem reads; config/CLI/core/log;
bench/memory/local bundle/fetch/web/UI. Potential issues under investigation:
phase clock failures can mask body errors, generic manifest FIFO replacement
can block a blocking open, worker response-path wrapper labels body RuntimeError
as destination failure. These are not yet fixed or fully accepted findings.
Only user's sandbox.yaml was pre-existing dirty; preserve it.
