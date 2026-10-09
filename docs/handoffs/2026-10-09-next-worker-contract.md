# Next CPU encoder milestones — senior-lead design handoff

Agent: Codex/root, 2026-10-09 10:47 UTC. These are scoped implementation
requirements from the current user-led design, not claims of completion.
Local Qwen via DSH writes product code; root reviews/fixes iteratively.
At most three active participants: root, one Qwen writer, one reviewer.
No cloud image generation, real-weight load, dependency installation,
asset mutations or GPU actions in this sandbox. Preserve sandbox.yaml.

Current remaining review corrections: cache unsafe publication/read-identity
quarantine plus persistence error typing, then compact JSON decoder error
wrapping. Source of precise current state: review-queue.md and scoped task
briefs in /tmp. Root memctl locking is read-only; Qwen's normal own-context
memctl state/handoff succeeded 10:45 locally (remote push failed).

After accepted corrections and both full guarded Python suites:

1. Lazy upstream encode helper and safe tensor codec ONLY. Overall current
   specification /tmp/v100-m22b-worker-task.md; scoped implementation brief
   /tmp/v100-worker-tensors-task.md. Parent transport/cache stays stdlib.
   Reuse pinned QwenImage21Pipeline with only supplied encoder+processor,
   other components None, and its encode_prompt method. Do not copy prompt
   control-token strings or substitute apply_chat_template for raw prompt
   formatting. T2I scalar, one image, negative only when CFG active. Safe
   versioned safetensors, exact tensor keys/descriptors, full request+asset+
   implementation identity; floating finite embeddings [1,S,H], nullable
   int64 attention mask [1,S], mandatory bool all-False image mask [1,S].
   Bounds and typed refusals for malformed names/keys/shapes/dtypes/identity.
   Separate tiny synthetic socket-denied checks on both pinned ML envs,
   -B/temporary caches, no real weights. Main guarded suite remains ML-free.
   Internal unequal batch/blank normalization/mm IDs/norm hook success and
   failure/nullable mask checks are not public batch support or real parity.

2. Split follow-on identity/request/preflight and worker wiring into fresh
   commit-sized tasks after helper acceptance. Request factory derives
   processor/tokenizer/template identities from pinned manifest, carries
   GenerationConfig.cfg_active semantics and explicit encoder dtype.
   Worker implementation identity automatically hashes relevant OWN source
   files with a digest schema; no mtime/path/manual-version-only identity.
   Exact installed package versions and VCS direct_url commits checked
   stdlib-only; missing torchvision is a known blocker, not auto-installed.
   Production requires matching full implementation identity, pinned repo/
   revision and encoder artifact-set digest; refuse images/unknown kwargs.

3. Explicit CPU worker --request/--response/--manifest/--models-dir plus
   REQUIRED validated runtime headroom (not an encoding key/default). Bound
   request JSON with no-follow/nonblocking open, opened regular FD checks,
   typed UTF8/parse/schema/cap refusal. Memory prerequisite BEFORE ML imports
   or model shard hashing/loading. Default actual discovery now refuses
   unknown hierarchy completeness; never invent trusted-root bypass. Any
   future complete-budget provider needs separate reviewed evidence.
   Full-reference manifest validation, then ONLY processor+encoder16files
   closed-subtree preflight and fresh SHA/size checks under canonical model
   root; unrelated DiT partial/missing VAE does not block encoder-only scope.
   Offline flags/socket audit before lazy imports; explicit local directories,
   local_files_only=True, trust_remote_code=False, safetensors/weights_only,
   low_cpu_mem_usage, explicit CPU dtype, eval/inference_mode. Never load
   DiT/VAE/full pipeline or implicit HF model IDs/cache fallback. Atomic
   no-clobber response in selected parent, clean temporary output, generic
   protocol error/stderr integration and existing parent lifecycle intact.
   Synthetic/injected loader boundaries only here; no actual full encoder.

Verified prerequisites/gates dated 2026-10-09: V100 not installed;19/24
final assets verified earlier,5missing plus6,302,990,336B interrupted DiT
partial preserved; torchvision missing in both exact ML envs; encoder
17,534,339,488B checkpoint exceeds16GiB container cap before extra residency.
No numeric real-checkpoint parity, image quality, measured CPU/GPU fit or
CUDA execution proven. Remaining adapter/quant/offload/calibration/image
acceptance milestones stay gated on the actual card/runtime evidence.
