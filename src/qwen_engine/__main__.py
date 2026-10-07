"""``python -m qwen_engine`` entry point."""
from __future__ import annotations

import sys

from qwen_engine.cli import main

if __name__ == "__main__":
    sys.exit(main())
