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
from qwen_engine._jsonsafe import strict_json_loads
from qwen_engine._local_io import read_bounded_file
from qwen_engine.adapters.base import (
    ATTENTION_BACKEND_CHOICES,
    OFFLOAD_CHOICES,
    QUANT_CHOICES,
)
from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.core import generate, plan, require_run_config
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
    "true_cfg_scale": "true_cfg_scale",
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
MAX_CONFIG_BYTES = 1 << 20


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
                   help="VAE tiling: required at 2048x2048, where an untiled "
                        "decode does not fit; uses fp32 1024 px tiles with a "
                        "768 px stride. At 1024x1024 the default untiled decode "
                        "is cleaner (the transformer is released first)")
    p.add_argument("--attest-unbounded-host", action="store_true",
                   help="operator attestation that this process tree has no cgroup "
                        "memory limit, so the CPU encoder gate uses the freshly read "
                        "host MemAvailable alone (default: the gate refuses when the "
                        "cgroup scope cannot be proven)")
    p.add_argument("--calibration-record", type=Path, default=None, metavar="PATH",
                   help="opt-in measured-memory collector: sample device-wide NVML "
                        "and owned-process-tree peaks per phase and append exact "
                        "calibration observations to PATH (requires --run). A "
                        "mid-run measurement failure keeps the generated image but "
                        "exits 1 without writing a record")
    # Field overrides (take precedence over --config):
    p.add_argument("--prompt")
    p.add_argument("--negative", default=None)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--true-cfg-scale", dest="true_cfg_scale", type=float, default=None,
                   help="true CFG scale (>= 1.0; guidance active only with a "
                        "negative prompt AND scale > 1)")
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
            raw = read_bounded_file(args.config, MAX_CONFIG_BYTES)
        except OSError as exc:
            raise ConfigError(
                f'config: config file not readable: {args.config}: {exc.strerror or exc}'
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise ConfigError(f"config: config input invalid: {args.config}: {exc}") from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigError(
                f"config: config file is not valid UTF-8: {args.config}: {exc}"
            ) from exc
        try:
            loaded = strict_json_loads(text)
        except (ValueError, RecursionError) as exc:
            # JSONDecodeError subclasses ValueError; the integer-string digit
            # limit raises plain ValueError and pathologically deep nesting
            # RecursionError, both before any schema validation runs.
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


def _verify_assets(config: GenerationConfig):
    """No-download contract (R3): refuse to run on any unresolved asset.
    Returns the :class:`ResolutionReport` so the caller can hand the verified,
    role-grouped paths to the adapter."""
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
    return report


def _make_adapter(*, trust_unbounded_host: bool = False) -> Any:
    """Instantiate the real adapter; the only path to torch/diffusers/transformers.

    Only the *absence of the adapter module itself* is a packaging error; a
    :class:`ModuleNotFoundError` for anything else (e.g. torch missing *inside*
    the adapter) keeps its own identity — a packaging fault must never
    masquerade as a missing dependency, and vice versa.

    ``trust_unbounded_host`` is forwarded verbatim as the operator's own
    cgroup-scope attestation; it changes no other gate.
    """
    import importlib

    try:
        module = importlib.import_module("qwen_engine.adapters.qwen21_diffusers")
    except ModuleNotFoundError as exc:
        if exc.name != "qwen_engine.adapters.qwen21_diffusers":
            raise
        raise EngineError(
            "no pipeline adapter module available: "
            "qwen_engine.adapters.qwen21_diffusers is missing from this "
            "installation (packaging error, not a missing dependency)"
        ) from exc
    return module.Qwen21DiffusersAdapter(trust_unbounded_host=trust_unbounded_host)


def _run_generate(config: GenerationConfig, logger: PhaseLogger,
                  assets: Any = None, adapter: Any = None,
                  trust_unbounded_host: bool = False,
                  calibration_record: Path | None = None) -> int:
    # Refuse an unconfigured output BEFORE the adapter exists at all: an
    # adapter import/load failure is an exit-1 boundary and must not mask
    # the config-error-2 for a run with nowhere to save.
    require_run_config(config)
    adapter = adapter if adapter is not None else _make_adapter(
        trust_unbounded_host=trust_unbounded_host)
    collector = None
    if calibration_record is not None:
        # Measurement setup happens before the load phase so its samples
        # cover the whole run. A setup failure refuses the run: an explicit
        # measurement request is never silently ignored.
        from qwen_engine.calibration import CalibrationCollector

        collector = CalibrationCollector(config, record_path=calibration_record)
        # Bind before sampling: an adapter that cannot report measured facts
        # refuses the run before any device state is touched.
        collector.bind(adapter)
        collector.start()
    try:
        result = generate(
            config, adapter, assets or {}, logger,
            phase_callback=collector.on_phase if collector is not None else None)
    except BaseException:
        if collector is not None:
            collector.abort()
        raise
    # The verified save is already published: a measurement failure below
    # must keep the image and fail only the record (exit 1 via EngineError).
    print(json.dumps(result.to_dict(), sort_keys=True))
    if collector is not None:
        observations = collector.finish()
        _safe_log(logger, "calibration_record", path=str(collector.path),
                  observations=len(observations))
    return int(ExitCode.OK)


def _safe_log(logger: PhaseLogger, name: str, **fields: Any) -> None:
    """Log an event without ever changing the CLI's exit path: a broken log
    sink gets one short diagnostic line and is otherwise ignored."""
    try:
        logger.event(name, **fields)
    except Exception as log_exc:
        print(f"warning: log sink failed: {type(log_exc).__name__}: {log_exc}",
              file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # usage errors: argparse prints + exit 2
    if args.run and args.dry_run:
        print("error: --run and --dry-run are mutually exclusive", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return int(ExitCode.CONFIG)
    if args.calibration_record is not None and not args.run:
        print("error: --calibration-record requires --run", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return int(ExitCode.CONFIG)
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
        # Asset problems are checked first and keep exit 4 even when the
        # config is *also* incomplete. A missing output_dir is then a
        # config error (2), checked BEFORE the adapter is created/imported
        # — the absent M4 adapter must not turn it into exit 1.
        report = _verify_assets(config)  # -> ManifestError -> exit 4
        if args.attest_unbounded_host:
            print(
                "notice: --attest-unbounded-host: attesting that this process "
                "tree has no cgroup memory limit; the encoder gate will apply "
                "only a freshly read host MemAvailable (recorded in run evidence)",
                file=sys.stderr)
            _safe_log(logger, "host_scope_attested",
                      attestation="unbounded-host", source="operator")
        return _run_generate(config, logger, assets=report.resolved,
                             trust_unbounded_host=args.attest_unbounded_host,
                             calibration_record=args.calibration_record)
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
        _safe_log(logger, "engine_error", error=str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return int(ExitCode.UNEXPECTED)
    except Exception:
        tb = traceback.format_exc()
        _safe_log(logger, "unexpected_error", error=tb.splitlines()[-1],
                  traceback=tb)
        print(f"error: unexpected {sys.exc_info()[0].__name__}: "
              f"{sys.exc_info()[1]}", file=sys.stderr)
        return int(ExitCode.UNEXPECTED)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
