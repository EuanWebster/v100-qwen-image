"""Fixed offline generation child, protocol only; no default model downloads.

The real adapter factory still refuses until its separate milestone exists.
No fixture backend/command override, profile bypass or GPU-readiness flag.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import threading
from pathlib import Path

from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.core import generate, require_run_config
from qwen_engine.encoder_worker import (
    install_offline_guard, read_bounded_json, _response_destination, _publish_response,
)
from qwen_engine.encoder import EncoderRequestError
from qwen_engine.errors import EngineError, GenerationCancelled, MemoryGuardError, HardwareAbsentError
from qwen_engine.manifest import ManifestError, load_manifest, resolve_manifest

REQUEST_SCHEMA = "qwen_engine.generation_request.v1"
RESULT_SCHEMA = "qwen_engine.generation_result.v1"
EVENT_SCHEMA = "qwen_engine.generation_phase.v1"
# Controller config cap plus a bounded fixed-envelope allowance.
MAX_REQUEST_BYTES = 256 * 1024 + 1024
MAX_RESULT_BYTES = 64 * 1024
MAX_EVENT_BYTES = 16 * 1024
PHASES = ("load", "encode", "denoise", "decode", "save")
ERROR_CODES = {1: "execution_failed", 2: "config_refused", 3: "memory_refused", 4: "assets_refused", 5: "hardware_refused"}


def valid_request_id(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{32}", value) is not None


def parse_request(path):
    try:
        doc, _ = read_bounded_json(path, MAX_REQUEST_BYTES, "generation request")
    except EncoderRequestError as exc:
        raise ConfigError("invalid bounded generation request") from exc
    if (not isinstance(doc, dict) or set(doc) != {"schema", "request_id", "config"}
            or doc["schema"] != REQUEST_SCHEMA or not valid_request_id(doc["request_id"])):
        raise ConfigError("invalid generation request envelope")
    return doc["request_id"], GenerationConfig.from_dict(doc["config"])


def run_generation(config, cancel, report):
    """Same engine boundary as CLI, with no implicit local-path defaults."""
    require_run_config(config)
    if cancel.is_set():
        raise GenerationCancelled()
    if config.manifest_path is None:
        raise ManifestError("an explicit manifest is required")
    manifest = load_manifest(config.manifest_path)
    root = Path(config.models_dir) if config.models_dir is not None else Path(config.manifest_path).parent
    resolved = resolve_manifest(manifest, root)
    if cancel.is_set():
        raise GenerationCancelled()
    if not resolved.ok:
        raise ManifestError("generation assets are not freshly verified")
    from qwen_engine.cli import _make_adapter
    return generate(config, _make_adapter(), resolved.resolved, cancel_event=cancel, phase_callback=report)


def execute(request_path, response_path):
    install_offline_guard()
    rid, config = parse_request(request_path)
    cancel = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda *_: cancel.set())
    try:
        with _response_destination(response_path) as (parent, name, directory):
            sequence = 0
            def report(phase):
                nonlocal sequence
                if phase not in PHASES or PHASES.index(phase) != sequence:
                    raise EngineError("invalid generation phase order")
                doc = dict(schema=EVENT_SCHEMA, request_id=rid, event="phase_start", phase=phase, sequence=sequence)
                sys.stdout.write(json.dumps(doc, sort_keys=True) + "\n")
                sys.stdout.flush()
                sequence += 1
            output = None
            error = None
            try:
                result = run_generation(config, cancel, report)
                output = result.output_path
                status, code = "succeeded", 0
            except GenerationCancelled:
                if not cancel.is_set():
                    status, code, error = "failed", 1, "invalid_cancellation_acknowledgment"
                else:
                    status, code = "cancelled", 130
            except ConfigError:
                status, code = "failed", 2
            except ManifestError:
                status, code = "failed", 4
            except MemoryGuardError:
                status, code = "failed", 3
            except HardwareAbsentError:
                status, code = "failed", 5
            except Exception:
                status, code = "failed", 1
            error = error or (ERROR_CODES[code] if status == "failed" else None)
            document = dict(schema=RESULT_SCHEMA, request_id=rid, status=status,
                            exit_code=code, output_path=output, error_code=error)
            raw = json.dumps(document, sort_keys=True, allow_nan=False).encode("utf-8")
            if len(raw) > MAX_RESULT_BYTES:
                raise EngineError("generation result byte cap exceeded")
            _publish_response(parent, name, directory, raw)
            return code
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)
    try:
        return execute(args.request, args.response)
    except (ConfigError, ValueError):
        print("generation worker refused invalid input", file=sys.stderr)
        return 2
    except Exception:
        print("generation worker failed before publishing a valid result", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
