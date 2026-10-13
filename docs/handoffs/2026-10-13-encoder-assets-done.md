# Narrow memory-gated encoder verifier accepted

Codex, 2026-10-13. Direct implementation; no Qwen jobs launched.
Previous identity/request milestone committed 172cbc8; helper/codec 8eefa81.
`encoder_assets.py` now validates the full pinned manifest profile, derives
checkpoint bytes from the four encoder shards, requires explicit runtime
headroom, and calls actual fail-closed memory discovery BEFORE model path
access, shard reads/hashing or ML imports. No public budget/snapshot bypass.

After a proceed result it verifies ONLY processor+text_encoder (16 files):
closed flat component trees; no-follow directory-FD traversal; nonblocking
regular-file opens; fresh SHA/size checks; bounded reads; before/after opened
and published identity checks; final tree/directory/root checks. Cleanup on
success/refusal/interrupt is tested. Missing/unsettled unrelated DiT/VAE assets
do not block this scope. Point-in-time integrity is not protection against
later mutations/library reopening; settled assets/fresh verification required.

39 new tiny-fixture tests; full guarded regressions 1102 passed / 2 skipped on
each Python3.11.15/3.13.5; real pinned upstream/codec 36/36 each; metadata/source
identity 5/5 each; actual memory-gate checks 3/3 each. Real memory refusal occurs
before ANY model-directory access/weight read; committed encoder checkpoint
17,534,339,488B alone exceeds visible 16GiB cap. Hierarchy completeness also
remains unknown. No real model hash/load/GPU/parity/fit/generation claim.
Commands/contracts in docs/encoder-assets.md and persistent reusable tools.

Next CPU milestone: local-only loader/worker CLI with bounded no-follow JSON
request input, required headroom, exact executing-runtime identity, offline
flags/socket confinement BEFORE lazy ML imports, verified index shard-reference
allowlisting, explicit local fixed-class processor/CPU encoder loading, and
bounded atomic no-clobber response publication. Test only injected tiny loaders
and refusals; do not attempt real weights or wire generation through gates.
Missing/unlocked torchvision remains a separately reviewed dependency-policy
change, not auto-installed. The complete project review remains after remaining
CPU wiring. V100 last user-confirmed absent. Preserve user sandbox.yaml.
