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
# Prompt rewriting. Two rewriters exist and BOTH may be configured: the
# official enhancer below (the standard one, which briefly takes the GPU) and
# the shared local Qwen3.8 service (which never touches the GPU and is the only
# one that can rewrite an edit draft). The upstream system prompts are
# licence-restricted, so they live in the ignored models/rewrite/ and are
# digest-checked at startup. Which one is ACTIVE is a user setting, chosen on
# the web page's Settings tab and stored by the server; these flags only say
# what is possible and seed the first start.
rewrite_dir=models/rewrite
rewrite_prompt_t2i_passed=0
if [[ ${QWEN_REWRITE:-1} == 1 && -f $rewrite_dir/system_prompt_t2i.txt \
      && -f $rewrite_dir/system_prompt_edit.txt ]]; then
  qwen_args+=(
    --rewrite-backend local-llm
    --rewrite-endpoint "${QWEN_REWRITE_ENDPOINT:-http://127.0.0.1:8079/v1/chat/completions}"
    --rewrite-model "${QWEN_REWRITE_MODEL:-Qwen3.8}"
    --rewrite-prompt-t2i "$rewrite_dir/system_prompt_t2i.txt"
    --rewrite-prompt-edit "$rewrite_dir/system_prompt_edit.txt"
  )
  rewrite_prompt_t2i_passed=1
fi
# The official Qwen-Image-2.1-PE-T2I enhancer, run as a Q8_0 GGUF under
# llama-server for the duration of one rewrite. Both files must exist or the
# enhancer simply is not offered. It reads the same digest-pinned t2i system
# prompt as the local service, so that file is required too. Set QWEN_PE=0 to
# leave it out even when the files are present.
pe_binary="${QWEN_PE_LLAMA_SERVER:-$HOME/Sandbox/llama.cpp-build/llama.cpp/build/bin/llama-server}"
pe_model="${QWEN_PE_MODEL:-models/pe/PE-T2I-Q8_0/Qwen-Image-2.1-PE-T2I-Q8_0.gguf}"
if [[ ${QWEN_PE:-1} == 1 && -x "$pe_binary" && -r "$pe_model" \
      && -f $rewrite_dir/system_prompt_t2i.txt ]]; then
  qwen_args+=(--pe-llama-server "$pe_binary" --pe-t2i-model "$pe_model")
  if [[ $rewrite_prompt_t2i_passed == 0 ]]; then
    qwen_args+=(--rewrite-prompt-t2i "$rewrite_dir/system_prompt_t2i.txt")
  fi
  if [[ -n ${QWEN_PE_TIMEOUT:-} ]]; then
    qwen_args+=(--pe-timeout "$QWEN_PE_TIMEOUT")
  fi
fi
exec .venv/bin/python "${qwen_args[@]}" "$@"
