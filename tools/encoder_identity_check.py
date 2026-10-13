#!/usr/bin/env python3
"""Check real installed identity without ML imports or model access.

Run with each persistent pinned ML interpreter and -B. Exit zero means the
metadata/source checks passed, not that model loading, fit or CUDA is proved.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.flags.optimize:
    raise SystemExit("Verification assertions must be enabled; unset PYTHONOPTIMIZE and do not use -O/-OO")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from qwen_engine.encoder_identity import (
        ML_MODULE_NAMES, PACKAGE_VERSIONS, observe_installed_packages, validate_installed_packages,
        worker_source_digest, runtime_metadata_digest,
    )
    from qwen_engine.local_bundle import reference_bundle_fingerprints
    from qwen_engine.manifest import load_manifest

    assert not any(name.split(".")[0] in ML_MODULE_NAMES for name in sys.modules)
    facts = observe_installed_packages()
    assert facts.executable == sys.executable
    print(f"PASS package metadata observed in Python {'.'.join(map(str, facts.python_version))}")
    validate_installed_packages(facts)
    print(f"PASS all {len(PACKAGE_VERSIONS)} reviewed package pins match, including torchvision")
    assert runtime_metadata_digest(facts) == runtime_metadata_digest(observe_installed_packages())
    print("PASS complete distribution inventory and Python runtime digest stable")
    first = worker_source_digest()
    assert first == worker_source_digest() and len(first) == 64
    print(f"PASS own-source digest stable: {first}")
    manifest = load_manifest(ROOT / "models/manifest.json")
    assert len(reference_bundle_fingerprints(manifest).encoder) == 64
    print("PASS pinned full-reference manifest identity projected without reading model files")
    assert not any(name.split(".")[0] in ML_MODULE_NAMES for name in sys.modules)
    print("PASS no ML imports during real installed-package/source/manifest checks")
    print("6 identity checks passed; model memory/asset/fidelity and CUDA gates remain separate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
