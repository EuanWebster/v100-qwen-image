"""The v1 CLI front-end (plan R1/R2, task 1.3) — argparse + exit-code contract.

Guarantees:
* ``--help``, ``--dry-run`` and plain import never touch torch/diffusers/
  transformers: the real adapter (not written yet) is the only heavy-import
  path, reached only by ``--run`` after asset verification.
* ``--dry-run`` prints deterministic, key-sorted, timestamp-free JSON.
* Expected errors print a single ``error:``-prefixed line (or per-field
  lines) and exit with the contract code — never a traceback; unexpected
  errors are logged with context (traceback into the log sink) and exit 1.

Exit codes: 0 ok · 2 invalid/malformed config · 3 memory refusal (reserved,
M5.3) · 4 missing/unverified asset · 5 hardware absent (reserved, M3) ·
1 unexpected.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_engine import __version__
from qwen_engine.adapters.base import (
    ATTENTION_BACKEND_CHOICES,
    OFFLOAD_CHOICES,
    QUANT_CHOICES,
)
from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.core import generate, plan
from qwen_engine.errors import (
    EngineError,
    ExitCode,
    HardwareAbsentError,
    MemoryGuardError,
)
from qwen_engine.log import PhaseLogger, file_sink
from qwen_engine.manifest import ManifestError, load_manifest, manifest_status, resolve_manifest

_OVERRIDES = {
    "prompt": "prompt",
    "negative": "negative_prompt",
    "width": "width",
    "height": "height",
    "steps": "steps",
    "seed": "seed",
    "quant": "quant",
    "offload": "offload",
    "attention_backend": "attention_backend",
    "output_dir": "output_dir",
    "cache_dir": "cache_dir",
    "log_dir": "log_dir",
    "benchmarks_dir": "benchmarks_dir",
    "models_dir": "models_dir",
    "manifest": "manifest_path",
}


def build_parser() -> argparse.ArgumentParser:
    from qwen_engine.config import AttentionBackend, OffloadMode, Quant

    p = argparse.ArgumentParser(
        prog="qwen-engine",
        description=(
            "Qwen-Image-2.1 engine (V100 16GB). Verifies pinned model assets "
            "from models/manifest.json; never downloads."
        ),
    )
    p.add_argument("--version", action="version", version=f"qwen-engine {__version__}")
    p.add_argument("--config", type=Path, default=None,
                   help="JSON config file (fields = GenerationConfig)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the deterministic key-sorted plan JSON and exit 0")
    p.add_argument("--run", action="store_true",
                   help="actually generate (verifies manifest assets first; no downloads)")
    p.add_argument("--tiled-decode", action="store_true",
                   help="VAE tiling (decode-memory knob only, C6)")
    # Field overrides (take precedence over --config):
    p.add_argument("--prompt")
    p.add_argument("--negative", default=None)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--quant", choices=QUANT_CHOICES)
    p.add_argument("--offload", choices=OFFLOAD_CHOICES)
    p.add_argument("--attention-backend", dest="attention_backend",
                   choices=ATTENTION_BACKEND_CHOICES)
    for flag in ("output_dir", "cache_dir", "log_dir", "benchmarks_dir",
                 "models_dir"):
        p.add_argument(f"--{flag.replace('_', '-')}", dest=flag, default=None,
                       help=f"override config field {flag!r} (default: unset -> no writes)")
    p.add_argument("--manifest", type=Path, default=None,
                   help="model manifest path override (default: unset -> run refuses)")
    return p


def _load_config(args: argparse.Namespace) -> GenerationConfig:
    data: dict[str, Any] = {}
    if args.config is not None:
        try:
            text = args.config.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f'config: config file not readable: {args.config}: {exc.strerror or exc}'
            ) from exc
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config: config file is not valid JSON: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ConfigError("config: config file must contain a JSON object")
        data = dict(loaded)
    for flag, field in _OVERRIDES.items():
        value = getattr(args, flag, None)
        if value is not None:
            data[field] = str(value) if isinstance(value, Path) else value
    if args.tiled_decode:
        data["vae_tiled_decode"] = True
    return GenerationConfig.from_dict(data)


def _logger_for(config: GenerationConfig) -> PhaseLogger:
    if config.log_dir is not None:
        return PhaseLogger(sink=file_sink(Path(config.log_dir) / "engine.jsonl"))
    return PhaseLogger()  # nothing configured -> nothing written


def _verify_assets(config: GenerationConfig) -> None:
    """No-download contract (R3): refuse to run on any unresolved asset."""
    if config.manifest_path is None:
        raise ManifestError(
            "no model manifest configured (manifest_path is null): assets are "
            "unverifiable and the resolver never downloads — set manifest_path "
            "(e.g. models/manifest.json)"
        )
    manifest = load_manifest(config.manifest_path)
    root = (Path(config.models_dir) if config.models_dir is not None
            else Path(config.manifest_path).parent)
    report = resolve_manifest(manifest, root)
    if not report.ok:
        for err in report.errors:
            print(f"asset error: {err}", file=sys.stderr)
        raise ManifestError(
            f"{len(report.errors)} model artifact(s) unresolved; nothing "
            f"downloaded (fetching is a separate operator action)"
        )


def _make_adapter() -> Any:
    """Instantiate the real adapter. Absent at M1 — importing a heavy
    adapter is the *only* path to torch/diffusers/transformers."""
    import importlib

    try:
        module = importlib.import_module("qwen_engine.adapters.qwen21_diffusers")
    except ImportError as exc:
        raise EngineError(
            "no pipeline adapter available: Qwen21DiffusersAdapter is not "
            "implemented yet (M4+); heavy ML imports are reachable only "
            "through it — this is the M1 boundary, not a fault to retry"
        ) from exc
    return module.Qwen21DiffusersAdapter()


def _run_generate(config: GenerationConfig, logger: PhaseLogger,
                  adapter: Any = None) -> int:
    adapter = adapter if adapter is not None else _make_adapter()
    result = generate(config, adapter, logger)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return int(ExitCode.OK)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # usage errors: argparse prints + exit 2
    logger = PhaseLogger()
    try:
        config = _load_config(args)
        logger = _logger_for(config)
        if args.dry_run:
            status = manifest_status(config.manifest_path, config.models_dir)
            print(json.dumps(plan(config, manifest=status),
                             sort_keys=True, indent=2))
            return int(ExitCode.OK)
        if not args.run:
            parser.print_help()
            return int(ExitCode.OK)
        _verify_assets(config)  # -> ManifestError -> exit 4
        return _run_generate(config, logger)
    except ConfigError as exc:
        for err in exc.errors:
            print(f"config error: {err}", file=sys.stderr)
        return int(ExitCode.CONFIG)
    except MemoryGuardError as exc:
        print(f"memory refusal: {exc}", file=sys.stderr)
        return int(ExitCode.MEMORY_REFUSAL)
    except HardwareAbsentError as exc:
        print(f"hardware: {exc}", file=sys.stderr)
        return int(ExitCode.HARDWARE)
    except ManifestError as exc:
        print(f"asset error: {exc}", file=sys.stderr)
        return int(ExitCode.ASSET)
    except EngineError as exc:
        logger.event("engine_error", error=str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return int(ExitCode.UNEXPECTED)
    except Exception:
        tb = traceback.format_exc()
        logger.event("unexpected_error", error=tb.splitlines()[-1],
                     traceback=tb)
        print(f"error: unexpected {sys.exc_info()[0].__name__}: "
              f"{sys.exc_info()[1]}", file=sys.stderr)
        return int(ExitCode.UNEXPECTED)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
