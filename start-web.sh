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
  --output-dir output/web --cache-dir cache
)
# This launcher targets the direct host environment used for CLI validation.
# Set to 0 when running under a cgroup RAM cap; normal encoder scope checks apply.
if [[ ${QWEN_ATTEST_UNBOUNDED_HOST:-1} == 1 ]]; then
  qwen_args+=(--attest-unbounded-host)
fi
exec .venv/bin/python "${qwen_args[@]}" "$@"
