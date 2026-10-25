"""Regenerate the packaged browser catalog from the shared Python source."""
from pathlib import Path
import re

from qwen_engine.presets import browser_catalog

path = Path(__file__).resolve().parents[1] / "src/qwen_engine/web_static/app.js"
source = path.read_text()
source, count = re.subn(r"  // BEGIN GENERATED PRESETS.*?  // END GENERATED PRESETS",
                       lambda _: browser_catalog(), source, flags=re.S)
if count != 1:
    raise SystemExit("expected exactly one generated preset block")
path.write_text(source)
