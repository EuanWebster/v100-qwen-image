# Encoder request / executing-runtime identity accepted

Codex, 2026-10-13. Direct implementation, no Qwen jobs or sub-agents.
`encoder_identity.py` now projects production requests from the pinned manifest
or a caller's fresh verified bundle; derives CFG from config and requires
explicit encoder dtype; derives processor/tokenizer/template identities and
encoder artifact-set identity; observes current-interpreter package metadata;
checks exact reviewed versions and VCS URL/git/commit; hashes own Python source
under a versioned schema; reconstructs and compares complete worker requests.

Identity projection does not verify asset bytes. Bundle coherence is not an
unforgeable resolver proof. Installed metadata is not package-byte attestation
or import/kernel/load evidence. The source digest is relocation/mtime independent
and bounded; no-follow directory-FD traversal, regular-file checks and descriptor
identity checks refuse unsafe or changing reads. No public source-root/manual
revision bypass. Actual missing OR arbitrary installed torchvision remains a
hard refusal: the lock contains no reviewed vision pin. No dependency edits.

Evidence: 80 new guarded tests; full guarded suites 1063 passed / 2 skipped on
Python 3.11.15 and 3.13.5; five actual metadata/source/manifest checks on each
persistent ML interpreter with NO ML imports and expected production refusal.
Reusable commands and scope: docs/encoder-identity.md; tool:
tools/encoder_identity_check.py. Existing helper/codec accepted as 8eefa81.

Next: narrow processor+encoder closed-tree verification independent of
unrelated DiT/VAE state; then local-only loader/worker with required runtime
headroom and memory prerequisite before ML imports or shard hashing/loading.
No real model, GPU, checkpoint parity, fit or generation claim. User's
sandbox.yaml remains modified and untouched; V100 last confirmed unavailable.
