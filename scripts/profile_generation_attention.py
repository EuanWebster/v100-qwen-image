#!/usr/bin/env python3
"""Observe actual denoiser ATen attention dispatch; explicit GPU run only.

CPU activity profiling identifies the dispatched PyTorch SDPA operator. It
does not trace CUDA kernel names, and profiled times are not speed benchmarks.
"""
import argparse
import json
from pathlib import Path
import sys

from qwen_engine.config import GenerationConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attest-unbounded-host", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    config = GenerationConfig.from_json(args.config.read_text())
    if not args.run:
        print(json.dumps(dict(run=False, steps=config.steps, quant=config.quant.value,
                              width=config.width, height=config.height)))
        return 0
    from qwen_engine.adapters.qwen21_diffusers import Qwen21DiffusersAdapter
    from qwen_engine.cli import _verify_assets
    from qwen_engine.core import generate

    class ProfiledAdapter(Qwen21DiffusersAdapter):
        def denoise(self, config, conditioning):
            torch = self._torch
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                result = super().denoise(config, conditioning)
                torch.cuda.synchronize(0)
            self.attention_ops = {event.key: event.count for event in profile.key_averages()
                                  if "scaled_dot_product" in event.key}
            return result

    # Explicit diagnostics reserve a new file before any load. Original
    # generation output handling and lifecycle remain owned by core.generate.
    with args.output.open("x", encoding="utf-8") as handle:
        adapter = ProfiledAdapter(trust_unbounded_host=args.attest_unbounded_host)
        result = generate(config, adapter, _verify_assets(config).resolved)
        document = dict(schema="qwen_engine.attention_dispatch.v1", steps=config.steps,
                        width=config.width, height=config.height, quant=config.quant.value,
                        requested_backend=config.attention_backend.value,
                        operators=adapter.attention_ops,
                        scope="actual denoiser ATen operators; CUDA kernels not traced",
                        timing_warning="profiler overhead; do not use for timing comparisons")
        json.dump(document, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result.to_dict()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("attention profiling failed: " + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
