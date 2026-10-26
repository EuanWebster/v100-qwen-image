#!/usr/bin/env bash
# Host launcher: one V100 queue shared by localhost, LAN and Tailscale.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing project virtual environment: $PROJECT_DIR/.venv/bin/python" >&2
  exit 1
fi
qwen_args=(
  -m qwen_engine.web --enable-generation
  --host 127.0.0.1
  --host "${QWEN_LAN_HOST:-192.168.68.103}"
  --host "${QWEN_TAILSCALE_HOST:-100.98.64.62}"
  --port "${QWEN_WEB_PORT:-8765}"
  --manifest models/manifest.json --models-dir models/assets
  --q8-manifest models/q8_manifest.json
  --output-dir output/web --cache-dir cache
)
# This launcher targets the direct host environment used for CLI validation.
# Set to 0 when running under a cgroup RAM cap; normal encoder scope checks apply.
if [[ ${QWEN_ATTEST_UNBOUNDED_HOST:-1} == 1 ]]; then
  qwen_args+=(--attest-unbounded-host)
fi
# Skip re-hashing model files whose size and mtime are unchanged since they
# were last verified (saves ~20 s per job). Set to 0 to hash every job.
if [[ ${QWEN_REUSE_VERIFIED_ASSETS:-1} == 1 ]]; then
  qwen_args+=(--reuse-verified-assets)
fi
# Prompt rewriter on the shared local Qwen3.8 service. The upstream system
# prompts are licence-restricted, so they live in the ignored models/rewrite/
# and are digest-checked at startup. Set QWEN_REWRITE=0 to disable.
rewrite_dir=models/rewrite
if [[ ${QWEN_REWRITE:-1} == 1 && -f $rewrite_dir/system_prompt_t2i.txt \
      && -f $rewrite_dir/system_prompt_edit.txt ]]; then
  qwen_args+=(
    --rewrite-backend local-llm
    --rewrite-endpoint "${QWEN_REWRITE_ENDPOINT:-http://127.0.0.1:8079/v1/chat/completions}"
    --rewrite-model "${QWEN_REWRITE_MODEL:-Qwen3.8}"
    --rewrite-prompt-t2i "$rewrite_dir/system_prompt_t2i.txt"
    --rewrite-prompt-edit "$rewrite_dir/system_prompt_edit.txt"
  )
fi
exec .venv/bin/python "${qwen_args[@]}" "$@"
