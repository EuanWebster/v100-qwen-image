# Turbo checkpoint assessment and resume handoff

Agent: Codex/root. Verified read-only 2026-10-10 17:17 UTC.

Resume state: Qwen JSON writer completed code08993bc/docs0a8c13e while root
was inactive. Existing exec2035 no longer exists. Its handoff reports typed
seven parser boundaries and +19 tests; root independent final review/rerun
is STILL PENDING. Cache7e0456d previously root+peer accepted. Tensor helper
has not been launched; /tmp/v100-worker-tensors-task.md and committed
next-worker-contract.md remain next after JSON acceptance. Preserve user's
only dirty product file sandbox.yaml. Root memctl brief still fails EROFS
on vault lock; direct STATE agrees JSON done, not independently accepted.

Current user request is research, not a switch/download/install directive:
assess https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo for V10016GB fit.
No product code/plan/profile/dependency/asset changes were made for this.

Official source observations (Hub metadata, NOT downloaded/loaded weights):
- Model card: same7B architecture, eight denoising steps, CFG1 default,
  prefix KV caching, checkpoint-saved sigma schedule. Explicit sigmas
  override schedule; num_inference_steps alone does not. Source:
  https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo
- Hub head observed d65dbc9a7e8f6b5479e33dee6030eaab2a906509.
- DiT32layers/32heads/128head_dim/context4096 matches base local config;
  index total raw tensor bytes14,230,249,472 (13.253GiB). Two weight files
  listed9.97GB+4.26GB. Source transformer/config.json and
  https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/transformer/diffusion_pytorch_model.safetensors.index.json
- Encoder still Qwen3VLForConditionalGeneration BF16/config4096hidden,
  one17.5GB model.safetensors file, rather than base four shards+index.
  https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/main/text_encoder
- VAE checkpoint676MB versus base pinned manifest1,350,989,512B.
  https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/main/vae
  Smaller stored file is not measured decode peak or proof smaller resident
  tensor set at a chosen dtype; no tensor-header inspection was possible.
- Existing pinned diffusers source already includes sample_sigmas parameter,
  config registration and call-sigmas>config-sigmas>steps precedence. Required
  PR14950 is merged; no dependency update automatically justified.

Assessment/inference: speed win, not a substantially smaller denoiser.
Raw DiT alone leaves about2.75GiB of nominal16GiB for all activations,
attention/cache/workspace/allocator/context if resident; no safe-fit promise.
Fewer sequential denoising steps do not proportionately reduce per-step peak.
40->8 means fivefold fewer steps, NOT measured5x end-to-end speed. Streaming
could transfer weights fewer times, useful on narrow PCIe. CFG1 already base
default; no additional CFG saving assumed. Prefix caching stores extra KV
tensors (verified pinned source clones), so not a free memory reduction.
V100 native Tensor Cores support FP16, not BF16; FP16 numerical safety must
be tested, not assumed from checkpoint storage. Source NVIDIA Ampere tuning
guide's Volta/Ampere datatype table:
https://docs.nvidia.com/cuda/archive/11.0/ampere-tuning-guide/
Existing CPU encoder/cache, offload and higher-quality quant evaluation remain
relevant. Recommend a separate optional Turbo candidate, retaining base as
quality reference until real card memory/numeric and balanced prompt review.
Turbo needs its OWN manifest/profile: do not substitute into the base24-file
pinned closed-tree contract (encoder file layout differs). No current card
install evidence supersedes user's last confirmation not installed.

Next: answer assessment; on implementation resume root reviews08993bc and
runs both guarded full suites, then launches scoped lazy helper/codec task.
Do not claim CPU build complete, Turbo fitted or GPU tested.

Administrative persistence blocker17:18: two bounded docs-only dsh headless
attempts failed without edits/commit; final error SERVER500 Internal Server
Error ('Server got itself in trouble'). No live writer remains. Root .git
is read-only, so this handoff remains untracked pending harness recovery and
a scoped docs-only commit. Do not stage sandbox.yaml or restart/change services
implicitly. No product modifications to lose; research findings preserved here.

Normal authorized memctl handoff DID succeed locally (vault commit1026fe1),
despite memctl brief lock failure; command contexts/approved-prefix permissions
differ. Automatic remote push failed for unavailable GitHub credentials; no
retry/credential change. Do not call the entire vault globally read-only.
