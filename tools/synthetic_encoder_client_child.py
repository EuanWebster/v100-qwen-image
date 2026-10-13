"""PRIVATE test bootstrap, never a product entrypoint or a runtime override.

The client check redirects fixed commands to this file only inside its test
process. Observe actual selected-child metadata; optionally corrupt one
observation for a refusal test, and emit tiny hand-built payloads instead of
any model loading. The real version policy is never relaxed.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path

if sys.flags.optimize:
    raise SystemExit("Verification assertions must be enabled; unset PYTHONOPTIMIZE and do not use -O/-OO")

from qwen_engine import encoder_identity as ei, encoder_worker as ew, encoder_production as ep
from qwen_engine.encoder import EncodedPayload, Blob, cache_key


def tiny_payload(request):
    header, descriptors, body = {}, [], b""
    for side in ("positive", "negative") if request.kwargs["cfg_active"] else ("positive",):
        code = ep.DTYPE_TO_ST_CODE[request.dtype]
        for suffix, dtype, shape, size in ((".embeds", code, [1, 2, 3], 6 * ep.ST_CODE_TO_TORCH_STR_ITEMSIZE[code][1]),
                                          (".image_pad_mask", "BOOL", [1, 2], 2)):
            name = side + suffix
            start = len(body)
            body += bytes(size)
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(body)]}
            descriptors.append({"name": name, "dtype": dtype, "shape": shape, "bytes": size})
    descriptors.sort(key=lambda d: d["name"])
    canonical = lambda doc: json.dumps(doc, sort_keys=True, separators=(",", ":"))
    header["__metadata__"] = {"codec": ep.SAFETENSORS_CODEC, "request_identity": canonical(request.key_document()),
        "cache_key": cache_key(request), "dtype": request.dtype,
        "cfg_active": "true" if request.kwargs["cfg_active"] else "false",
        "tensor_descriptors": canonical(descriptors)}
    raw_header = canonical(header).encode()
    return EncodedPayload(metadata={"schema": ep.TENSOR_PAYLOAD_SCHEMA, "codec": ep.SAFETENSORS_CODEC,
        "cache_key": cache_key(request), "dtype": request.dtype, "cfg_active": request.kwargs["cfg_active"],
        "request": request.key_document(), "tensors": descriptors},
        blobs=(Blob("tensors", struct.pack("<Q", len(raw_header)) + raw_header + body),))


def main():
    root = Path(os.environ["QWEN_CLIENT_TEST_ROOT"]).resolve(strict=True)
    assert os.environ.get("QWEN_CPU_TEST_GUARD") == "1"
    assert not set(ei.ML_MODULE_NAMES) & set(sys.modules)
    ew.install_offline_guard()
    facts = ei.observe_installed_packages()
    ei.validate_installed_packages(facts)
    if os.environ.get("QWEN_CLIENT_TEST_CORRUPT_VISION") == "1":
        facts = replace(facts, packages={**facts.packages,
            "torchvision": ei.PackageObservation("synthetic-wrong-vision")})
        ei.observe_installed_packages = lambda: facts
    if "--project-identity" in sys.argv:
        # Actual versions/VCS and Python observed in this interpreter.
        with (root / "projection-facts.jsonl").open("a") as fh:
            fh.write(json.dumps({"python_version": list(sys.version_info[:3]), "executable": sys.executable,
                "runtime_digest": ei.runtime_metadata_digest(facts), "source_digest": ei.worker_source_digest(),
                "heavy_imports": sorted(set(ei.ML_MODULE_NAMES) & set(sys.modules))}) + "\n")
    else:
        def emit_synthetic(request_path, response_path, manifest_path, models_dir, *, headroom_bytes):
            request, rid = ew._read_request(request_path)
            ep.validate_production_request(request)
            ei.validate_worker_request_identity(request, ew._read_manifest(manifest_path))
            with (root / "synthetic-loads.jsonl").open("a") as fh:
                fh.write(json.dumps({"request_id": rid, "dtype": request.dtype}) + "\n")
            with ew._response_destination(response_path) as (parent, name, directory):
                ew._publish_response(parent, name, directory, ew.result_bytes(tiny_payload(request), rid))
        ew.execute_worker = emit_synthetic  # PRIVATE test-only fake body; no fit/loader claim
    result = ew.main()
    assert not set(ei.ML_MODULE_NAMES) & set(sys.modules)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
