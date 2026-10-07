#!/usr/bin/env python3
"""Repo-root entry shim for the engine CLI (thin on purpose — plan R1).

All logic lives in ``qwen_engine.cli``; this file only makes the CLI runnable
as ``python cli_gen.py …`` straight from the repo, with or without the
package installed (falls back to ``src/`` on sys.path).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qwen_engine.cli import main  # noqa: E402  (path shim must run first)

if __name__ == "__main__":
    raise SystemExit(main())
