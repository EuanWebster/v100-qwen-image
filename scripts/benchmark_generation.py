#!/usr/bin/env python3
"""Explicit repeated generation; no GPU work without --run.

Use the normal CLI/config and existing single-use adapters. Process reuse
measures the current web-style lifecycle, NOT persistent model retention.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from qwen_engine.config import GenerationConfig, ConfigError
from qwen_engine.performance_summary import SummaryError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new benchmark directory (must not already exist)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--mode", choices=("fresh-process", "process-reuse"), default="fresh-process")
    parser.add_argument("--attest-unbounded-host", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.runs <= 64:
            raise ValueError("runs must be between 1 and 64")
        config = GenerationConfig.from_json(args.config.read_text())
        if config.seed is None:
            raise ValueError("a fixed seed is required for reproducible comparisons")
        if args.output_dir.exists():
            raise ValueError("benchmark output directory must be new")
        if not args.run:
            # Do not echo prompts or server/runtime paths in diagnostic plans.
            print(json.dumps(dict(runs=args.runs, mode=args.mode, width=config.width,
                                  height=config.height, steps=config.steps, seed=config.seed,
                                  quant=config.quant.value, run=False)))
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for index in range(args.runs):
            out = args.output_dir / f"run-{index+1}"
            out.mkdir()
            report = args.output_dir / f"run-{index+1}.json"
            if args.mode == "fresh-process":
                command = [sys.executable, "-B", "-m", "qwen_engine", "--run",
                           "--config", str(args.config.resolve()), "--output-dir", str(out.resolve()),
                           "--performance-report", str(report.resolve())]
                if args.attest_unbounded_host:
                    command.append("--attest-unbounded-host")
                with (args.output_dir / f"run-{index+1}.log").open("x") as log:
                    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                    try:
                        code = proc.wait()
                    finally:
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                if code:
                    raise RuntimeError("generation failed; partial evidence retained")
            else:
                from qwen_engine.cli import _verify_assets, _run_generate
                from qwen_engine.log import PhaseLogger
                run_config = replace(config, output_dir=str(out.resolve()))
                assets = _verify_assets(run_config)
                _run_generate(run_config, PhaseLogger(), assets=assets.resolved,
                              trust_unbounded_host=args.attest_unbounded_host,
                              performance_report=report)
        from qwen_engine.performance_summary import summarise_reports, write_summary
        write_summary(summarise_reports(sorted(args.output_dir.glob("run-*.json"))),
                      args.output_dir / "summary.json")
        print("benchmark complete; summaries separate embedding-cache states")
        return 0
    except (ConfigError, ValueError, OSError, RuntimeError, SummaryError) as exc:
        # Do not echo config/prompt contents from schema exceptions.
        print("benchmark refused or failed: " + type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
