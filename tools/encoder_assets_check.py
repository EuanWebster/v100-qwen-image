#!/usr/bin/env python3
"""Verify actual memory refusal before any model path access or ML import.

No model is loaded/hashed. Run with each persistent review interpreter and -B.
Success verifies the CURRENT refusal gate, not production readiness or fit.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from qwen_engine import encoder_assets as ea
    from qwen_engine.manifest import load_manifest
    from qwen_engine.mem_preflight import MemoryPreflightError

    manifest = load_manifest(ROOT / "models/manifest.json")
    checkpoint = sum(a.bytes for a in manifest.artifacts
                     if a.role == "text_encoder" and a.identity.endswith(".safetensors"))
    assert checkpoint == 17_534_339_488
    print(f"PASS exact encoder checkpoint bytes derived from manifest: {checkpoint}")
    accessed = False
    original = ea._canonical_root

    def prohibited(_):
        nonlocal accessed
        accessed = True
        raise AssertionError("memory gate failed to precede model root access")

    ea._canonical_root = prohibited
    try:
        ea.verify_encoder_assets(manifest, ROOT / "models", target_dtype="fp16", headroom_bytes=0)
    except MemoryPreflightError as exc:
        assert "encoder memory preflight refused" in str(exc), str(exc)
        print("PASS actual memory discovery refuses before model access (no budget bypass)")
    else:
        raise AssertionError("expected actual memory refusal was not observed")
    finally:
        ea._canonical_root = original
    assert not accessed
    assert not any(m.split(".")[0] in {"torch", "diffusers", "transformers", "safetensors"}
                   for m in sys.modules)
    print("PASS no model directory access, shard reads or ML imports")
    print("3 asset-boundary checks passed; actual model verification/loading remains BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
