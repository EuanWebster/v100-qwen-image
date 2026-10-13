#!/usr/bin/env bash
# Recreate the locked ML review environments on the persistent project mount.
# Run from any directory; reusable checks live in git, generated fixtures in /tmp.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
review_root="$project_root/.review-envs"
export UV_CACHE_DIR="$review_root/uv-cache"
export UV_PYTHON_INSTALL_DIR="$review_root/python"
mkdir -p "$review_root"

for python_version in 3.11.15 3.13.5; do
    case "$python_version" in
        3.11.*) review_env="$review_root/py311" ;;
        3.13.*) review_env="$review_root/py313" ;;
    esac
    if [ ! -x "$review_env/bin/python" ]; then
        uv venv --python "$python_version" "$review_env"
    fi
    actual_version="$("$review_env/bin/python" -B -c 'import platform; print(platform.python_version())')"
    if [ "$actual_version" != "$python_version" ]; then
        printf 'Expected Python %s at %s, found %s\n' \
            "$python_version" "$review_env" "$actual_version" >&2
        exit 1
    fi
    uv pip sync --python "$review_env/bin/python" \
        --index-strategy unsafe-best-match "$project_root/requirements.lock"
    uv pip check --python "$review_env/bin/python"
done

# A separate torch-free 3.11 environment for the guarded regression suite.
UV_PROJECT_ENVIRONMENT="$review_root/test311" uv sync --locked \
    --project "$project_root" --python "$review_root/py311/bin/python"

printf 'Review environments: %s/py311 and %s/py313\n' "$review_root" "$review_root"
printf 'Guarded Python 3.11 tests: PYTHON=%s/test311/bin/python bash run_cpu_tests.sh\n' "$review_root"
