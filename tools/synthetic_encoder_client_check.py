#!/usr/bin/env python3
"""Actual selected-child metadata/protocol/cache checks with synthetic payloads.

Run with each pinned ML interpreter, optionally choosing the OTHER pinned
interpreter with --child-python. No ML imports, models, GPU or downloads.
Corrupted observation/fake payload producer are private test-process seams only;
product commands/policy have no bypass flag. Fixtures alone are temporary.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if sys.flags.optimize:
    raise SystemExit("Verification assertions must be enabled; unset PYTHONOPTIMIZE and do not use -O/-OO")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-python", default=sys.executable)
    args = parser.parse_args()
    run = Path(tempfile.mkdtemp(prefix="qwen-client-synthetic-"))
    try:
        return checks(run, os.path.abspath(args.child_python))
    finally:
        shutil.rmtree(run)


def checks(run, child_python):
    os.environ.update(QWEN_CPU_TMP_ROOTS=str(run), QWEN_CPU_TEST_GUARD="1",
        QWEN_CPU_DENY_IMPORT="torch,diffusers,transformers,safetensors,torchvision",
        PYTHONDONTWRITEBYTECODE="1", QWEN_CLIENT_TEST_ROOT=str(run), QWEN_CLIENT_TEST_CORRUPT_VISION="0")
    for variable, sub in {"HOME": "home", "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
                          "XDG_DATA_HOME": "data", "HF_HOME": "hf", "TMPDIR": "tmp"}.items():
        (run / sub).mkdir()
        os.environ[variable] = str(run / sub)
    os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT / "tests/cpu_guard"), str(ROOT / "src")))
    tempfile.tempdir = str(run / "tmp")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("_client_guard", ROOT / "tests/cpu_guard/sitecustomize.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.install()
    for attempt in (lambda: socket.socket(), lambda: open(ROOT / "must-not-write-client-probe", "wb")):
        try:
            attempt()
        except guard.GuardViolation:
            pass
        else:
            raise AssertionError("client check guard failed")
    print("PASS network/repository-write guard before product imports")

    sys.path.insert(0, str(ROOT / "src"))
    from qwen_engine import encoder_identity as ei
    from qwen_engine.encoder_client import ProductionEncoder
    from qwen_engine.encoder import EmbeddingCache, EncoderWorkerError
    from qwen_engine.config import GenerationConfig

    def parent_observation_refused():
        raise AssertionError("production client substituted parent package facts")
    ei.observe_installed_packages = parent_observation_refused
    original_popen = subprocess.Popen
    def test_popen(command, *a, **kw):
        assert command[:4] == [child_python, "-B", "-m", "qwen_engine.encoder_worker"]
        return original_popen([*command[:2], str(ROOT / "tools/synthetic_encoder_client_child.py"),
                               *command[4:]], *a, **kw)
    subprocess.Popen = test_popen
    client = ProductionEncoder(manifest_path=ROOT / "models/manifest.json", models_dir=run / "never-created-models",
        scratch_dir=run, python=child_python, headroom_bytes=0, timeout_s=30, projection_timeout_s=30,
        cache=EmbeddingCache(run / "embeds"))
    print("PASS fixed product commands choose explicit child interpreter; parent observer forbidden")
    read_records = lambda name: [json.loads(line) for line in (run / name).read_text().splitlines()]
    config = GenerationConfig(prompt=" exact local prompt ", negative_prompt="inactive negative")
    first = client.encode(config, encoder_dtype="fp16")
    fact = read_records("projection-facts.jsonl")[-1]
    revision = first.request.implementation_identity.worker_implementation_revision
    assert revision == (f"{ei.RUNTIME_DIGEST_SCHEMA}:{fact['runtime_digest']}:"
                        f"{ei.SOURCE_DIGEST_SCHEMA}:{fact['source_digest']}")
    assert fact["executable"] == child_python and not fact["heavy_imports"]
    assert fact["python_version"][:2] in ([3, 11], [3, 13])
    print(f"PASS miss uses actual selected Python {'.'.join(map(str, fact['python_version']))} "
          "runtime/source identity; synthetic opaque payload only")
    second = client.encode(config, encoder_dtype="fp16")
    assert first == second
    assert len(read_records("projection-facts.jsonl")) == 2 and len(read_records("synthetic-loads.jsonl")) == 1
    print("PASS cache hit repeats fresh child projection without loading")
    for dtype in ("bf16", "fp32"):
        result = client.encode(config, encoder_dtype=dtype)
        assert result.request.dtype == dtype
        assert result.payload.metadata["dtype"] == dtype
    print("PASS separate bf16/fp32 identity + wire/header checks without ML imports")
    cfg = client.encode(GenerationConfig(prompt="positive", negative_prompt="negative", true_cfg_scale=2),
                        encoder_dtype="fp16")
    assert cfg.payload.metadata["cfg_active"]
    assert len(cfg.payload.metadata["tensors"]) == 4  # masks nullable, image masks mandatory
    print("PASS CFG projection + positive/negative synthetic payload structure")
    assert not client.models_dir.exists() and not list(run.glob("qwen-enc-*"))
    assert not set(ei.ML_MODULE_NAMES) & set(sys.modules)
    print("PASS no model-root access/heavy imports; scratch processes reaped/cleaned")
    os.environ["QWEN_CLIENT_TEST_CORRUPT_VISION"] = "1"
    def no_cache(*a, **kw):
        raise AssertionError("cache touched after actual child policy refusal")
    for name in ("lock_for", "get", "put"):
        setattr(client.cache, name, no_cache)
    try:
        client.encode(config, encoder_dtype="fp16")
    except EncoderWorkerError as exc:
        assert exc.exit_code == 1 and "torchvision: expected" in exc.stderr
    else:
        raise AssertionError("corrupted child observation bypassed the real vision policy")
    assert not list(run.glob("qwen-enc-*"))
    print("PASS real child policy refuses corrupted observation before cache, including a previously valid hit")
    print("8 selected-child client checks passed; synthetic payload/refusal fixture only, no model/processor/GPU/fit claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
