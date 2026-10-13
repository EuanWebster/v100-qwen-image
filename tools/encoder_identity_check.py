#!/usr/bin/env python3
"""Check real installed identity without ML imports; current vision gate expected.

Run with each persistent pinned ML interpreter and -B. Exit zero means the
metadata/source checks and the EXPECTED production refusal were verified,
not that production loading is enabled. Unexpected failures exit nonzero.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from qwen_engine.encoder_identity import (
        PACKAGE_NAMES, observe_installed_packages, validate_installed_packages,
        worker_source_digest,
    )
    from qwen_engine.encoder import EncoderSetupError
    from qwen_engine.local_bundle import reference_bundle_fingerprints
    from qwen_engine.manifest import load_manifest

    assert not any(name.split(".")[0] in PACKAGE_NAMES for name in sys.modules)
    facts = observe_installed_packages()
    assert facts.executable == sys.executable
    assert facts.packages["torchvision"].version is None
    print(f"PASS package metadata observed in Python {'.'.join(map(str, facts.python_version))}")
    expected = ("installed encoder packages refused: torchvision: no reviewed pin "
                "in requirements.lock (observed absent); production loading is blocked")
    try:
        validate_installed_packages(facts)
    except EncoderSetupError as exc:
        assert str(exc) == expected, str(exc)
    else:
        raise AssertionError("current production torchvision gate was bypassed")
    print("PASS all reviewed package pins match; missing/unlocked torchvision explicitly refuses production")
    first = worker_source_digest()
    assert first == worker_source_digest() and len(first) == 64
    print(f"PASS own-source digest stable: {first}")
    manifest = load_manifest(ROOT / "models/manifest.json")
    assert len(reference_bundle_fingerprints(manifest).encoder) == 64
    print("PASS pinned full-reference manifest identity projected without reading model files")
    assert not any(name.split(".")[0] in PACKAGE_NAMES for name in sys.modules)
    print("PASS no ML imports during real installed-package/source/manifest checks")
    print("5 identity checks passed; production loading remains BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
