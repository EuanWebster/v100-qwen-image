#!/usr/bin/env bash
# Recreate the locked ML review environments on the persistent project mount.
# Run from any directory; reusable checks live in git, generated fixtures in /tmp.
set -euo pipefail

check_only=0
case "${1:-}" in
    "") ;;
    --check) check_only=1 ;;
    *) printf 'Usage: %s [--check]\n' "$0" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then printf 'Only --check is supported\n' >&2; exit 2; fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
review_root="$project_root/.review-envs"

# Inspect the WHOLE fixed target set before mkdir/sync. An existing alias to
# .venv or another tree must never become an ML synchronization destination.
for review_path in "$review_root" "$review_root/uv-cache" "$review_root/python" \
                   "$review_root/py311" "$review_root/py313" "$review_root/test311" \
                   "$review_root/py311/bin" "$review_root/py313/bin" "$review_root/test311/bin"; do
    if [ -L "$review_path" ]; then
        printf 'Refusing symlinked review path: %s\n' "$review_path" >&2
        exit 1
    fi
    if [ -e "$review_path" ] && [ ! -d "$review_path" ]; then
        printf 'Review path is not a directory: %s\n' "$review_path" >&2
        exit 1
    fi
done

validate_environment() {
    local target_env="$1" expected_version="$2" observed metadata
    if [ ! -e "$target_env" ]; then
        if [ "$check_only" -eq 1 ] || [ "${3:-0}" -eq 1 ]; then
            printf 'Review environment missing: %s\n' "$target_env" >&2
            return 1
        fi
        return 0
    fi
    if [ ! -x "$target_env/bin/python" ]; then
        printf 'Existing review environment has no executable Python: %s\n' "$target_env" >&2
        return 1
    fi
    if ! observed="$("$target_env/bin/python" -I -B -c \
        'import os, platform, sys; print(platform.python_version()); print(os.path.realpath(sys.prefix)); print("venv" if sys.prefix != sys.base_prefix else "base")')"; then
        printf 'Cannot inspect review interpreter: %s\n' "$target_env" >&2
        return 1
    fi
    mapfile -t metadata <<< "$observed"
    if [ "${#metadata[@]}" -ne 3 ] || [ "${metadata[0]}" != "$expected_version" ] \
       || [ "${metadata[1]}" != "$target_env" ] || [ "${metadata[2]}" != venv ]; then
        printf 'Expected isolated Python %s rooted exactly at %s; refusing mismatched interpreter\n' \
            "$expected_version" "$target_env" >&2
        return 1
    fi
}

validate_environment "$review_root/py311" 3.11.15
validate_environment "$review_root/py313" 3.13.5
validate_environment "$review_root/test311" 3.11.15
command -v uv >/dev/null || { printf 'uv is required\n' >&2; exit 1; }

if [ "$check_only" -eq 1 ]; then
    for review_env in "$review_root/py311" "$review_root/py313" "$review_root/test311"; do
        uv pip check --python "$review_env/bin/python"
    done
    printf 'Review paths/interpreters and dependency consistency checked; no synchronization or downloads\n'
    printf 'This is not exact ML-pin approval, model fit or GPU acceptance; use the pinned identity checks separately\n'
    exit 0
fi

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
    validate_environment "$review_env" "$python_version" 1
    uv pip sync --python "$review_env/bin/python" \
        --index-strategy unsafe-best-match "$project_root/requirements.lock"
    uv pip check --python "$review_env/bin/python"
done

# A separate torch-free 3.11 environment for the guarded regression suite.
UV_PROJECT_ENVIRONMENT="$review_root/test311" uv sync --locked \
    --project "$project_root" --python "$review_root/py311/bin/python"
validate_environment "$review_root/test311" 3.11.15 1

printf 'Review environments: %s/py311 and %s/py313\n' "$review_root" "$review_root"
printf 'Guarded Python 3.11 tests: PYTHON=%s/test311/bin/python bash run_cpu_tests.sh\n' "$review_root"
