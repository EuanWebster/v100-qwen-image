"""Canonical nominal aspect catalog; browser data is generated from this table."""
import json

PRESETS = {}
for name, ratio, small, large in (
    ("square", "1:1", (1024, 1024), (2048, 2048)),
    ("fourthree", "4:3", (1152, 864), (2400, 1792)),
    ("threefour", "3:4", (864, 1152), (1792, 2400)),
    ("threetwo", "3:2", (1248, 832), (2528, 1696)),
    ("twothree", "2:3", (832, 1248), (1696, 2528)),
    ("landscape", "16:9", (1344, 768), (2752, 1536)),
    ("portrait", "9:16", (768, 1344), (1536, 2752)),
):
    for suffix, tier, (width, height) in (("", "1K", small), ("2k", "2K", large)):
        PRESETS[name + suffix] = dict(width=width, height=height, ratio=ratio, tier=tier)


def browser_catalog():
    """Exact generated block embedded in app.js; checked by CPU tests."""
    return "  // BEGIN GENERATED PRESETS (scripts/sync_presets.py)\n  var PRESETS = " + json.dumps(
        PRESETS, ensure_ascii=True, indent=2
    ).replace("\n", "\n  ") + ";\n  // END GENERATED PRESETS"


def preset_config(name):
    p = PRESETS[name]
    return {"width": p["width"], "height": p["height"],
            "vae_tiled_decode": (p["width"] // 16) * (p["height"] // 16) > 4096}
