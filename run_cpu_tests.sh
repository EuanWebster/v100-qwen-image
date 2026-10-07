#!/usr/bin/env bash
# R7 fresh-environment CPU test runner (PLAN: no network, no GPU, no model
# weights; writes ONLY inside temporary directories; repository reads allowed).
#
# What it changes for the pytest process AND every subprocess it spawns
# (fake encoder workers included — the guard rides PYTHONPATH into child
# interpreters via sitecustomize + a sys.addaudithook audit hook):
#   * HOME / XDG_* / TMPDIR / pytest basetemp -> fresh dirs under one
#     mktemp run root; writes anywhere else fail BEFORE they happen;
#   * network denied (socket connect/sendto/getaddrinfo/gethostbyname);
#   * heavy ML imports denied before execution (torch, diffusers,
#     transformers) even if stubs are on the path;
#   * bytecode writing disabled (-B / PYTHONDONTWRITEBYTECODE) and the
#     pytest cache provider disabled — nothing is written into the repo;
#   * reads (repo, stdlib) are unrestricted.
#
# Exact fresh offline command (no network access needed at any point):
#     bash run_cpu_tests.sh                 # full suite
#     bash run_cpu_tests.sh -k manifest     # any pytest args pass through
# Python 3.11 execution is still PENDING (R7): this runs the interpreter
# found as $PYTHON, else .venv/bin/python, else python3 — the guard and the
# suite do not claim 3.11 has been exercised.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(mktemp -d "${TMPDIR:-/tmp}/qwen-cpu-run-XXXXXXXX")" || exit 1
mkdir -p "$RUN/home" "$RUN/tmp" "$RUN/xdg/cache" "$RUN/xdg/config" \
         "$RUN/xdg/data" "$RUN/pytest-tmp"

export HOME="$RUN/home"
export XDG_CACHE_HOME="$RUN/xdg/cache"
export XDG_CONFIG_HOME="$RUN/xdg/config"
export XDG_DATA_HOME="$RUN/xdg/data"
export TMPDIR="$RUN/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RUN/pycache"
export PYTHONPATH="$ROOT/tests/cpu_guard${PYTHONPATH:+:$PYTHONPATH}"
export QWEN_CPU_TEST_GUARD=1
export QWEN_CPU_TMP_ROOTS="$RUN"
export QWEN_CPU_DENY_IMPORT="torch,diffusers,transformers"

PY="${PYTHON:-}"
if [ -z "$PY" ] && [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; fi
if [ -z "$PY" ]; then PY="$(command -v python3 || true)"; fi
if [ -z "$PY" ]; then echo "run_cpu_tests: no python3 interpreter found" >&2; exit 127; fi

echo "cpu-run root (writes allowed here only; delete when done): $RUN" >&2
"$PY" -B -m pytest -p no:cacheprovider --basetemp="$RUN/pytest-tmp" \
       -q "$@"
rc=$?
exit $rc
