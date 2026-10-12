"""``python -m qwen_engine.web`` entry point (qwen-web)."""
from __future__ import annotations

import sys

from qwen_engine.web.server import main

if __name__ == "__main__":
    sys.exit(main())
