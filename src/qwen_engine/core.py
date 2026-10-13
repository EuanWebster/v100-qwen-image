"""Engine core: ``plan()`` and ``generate()`` over the adapter protocol (R1/R2).

Stdlib-only; importable with no torch/diffusers/transformers installed. The
dry-run ``plan`` document is deterministic, key-sortable and free of
timestamps; unmeasured things (hardware, calibration) are reported as
``unknown`` — never fabricated.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable
import threading

from qwen_engine.adapters.base import (
    CALIBRATION_UNKNOWN,
    HARDWARE_UNKNOWN,
    GenerationAdapter,
)
from qwen_engine.config import ConfigError, FieldError, GenerationConfig
from qwen_engine.errors import EngineError, GenerationCancelled
from qwen_engine.log import PhaseLogger

#: Schema tag of the dry-run document (versioned like the bench schemas, R6).
DRY_RUN_SCHEMA = "qwen_engine.dry_run.v1"

#: Assets the adapter may load: verified paths grouped by role.
Assets = Mapping[str, tuple[Path, ...]]


class OutputError(EngineError):
    """The save step did not produce the output it claimed (missing, not a
    regular file, or outside ``output_dir``). Exit 1 — the run failed."""


@dataclass(frozen=True)
class GenerationResult:
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return {"output_path": self.output_path}


def plan(config: GenerationConfig, *, manifest: dict[str, Any]) -> dict[str, Any]:
    """Deterministic dry-run plan for ``config`` (M1 acceptance criterion 4).

    Reports resolved config, latent token count, manifest resolution status,
    and *explicitly unknown* hardware / calibration — no timestamps, nothing
    measured that was not measured. Planning is adapter-independent at this
    milestone (no adapter ``plan`` call).
    """
    return {
        "schema": DRY_RUN_SCHEMA,
        "config": config.to_dict(),
        "latent_tokens": config.latent_tokens,
        "cfg_active": config.cfg_active,
        "manifest": manifest,
        "hardware": HARDWARE_UNKNOWN.to_dict(),
        "calibration": CALIBRATION_UNKNOWN.to_dict(),
    }


def _validate_output(output: Any, config: GenerationConfig) -> Path:
    """The adapter *claims* it saved ``output``; verify the claim. Never let a
    failed, missing or escaping save report success."""
    if config.output_dir is None:  # defensive: generate() checks first
        raise ConfigError(FieldError("output_dir", "output is not configured"))
    out_dir = Path(config.output_dir)
    path = Path(str(output))
    if not path.is_file():
        raise OutputError(
            f"adapter reported save at {path}, but no regular file exists there"
        )
    resolved = path.resolve()
    out_root = out_dir.resolve()
    if resolved != out_root and out_root not in resolved.parents:
        raise OutputError(
            f"saved file {resolved} is outside the configured output_dir {out_root}"
        )
    return resolved


def require_run_config(config: GenerationConfig) -> None:
    """Config that must be present before an adapter is even *created*.

    The CLI calls this after asset verification and before adapter import:
    an absent output setting is a config error (exit 2) and must never be
    masked by the M4 adapter boundary failing first (exit 1)."""
    if config.output_dir is None:
        raise ConfigError(FieldError(
            "output_dir",
            "an actual run requires output_dir (nowhere to save); set it in "
            "the config or pass --output-dir",
        ))


def generate(
    config: GenerationConfig,
    adapter: GenerationAdapter,
    assets: Assets,
    logger: PhaseLogger | None = None,
    *,
    cancel_event: threading.Event | None = None,
    phase_callback: Callable[[str], None] | None = None,
) -> GenerationResult:
    """Run the full pipeline through the adapter protocol, logging phases.

    ``assets`` are the resolver's verified artifact paths grouped by role and
    are handed to ``adapter.load`` verbatim. Nothing is written when the
    output is unconfigured: ``output_dir`` is refused **before** the adapter
    loads, so a missing output setting cannot cost a model load or touch the
    filesystem.  After a verified save, the final ``complete`` emission is
    nonfatal like the phase completion lines: a broken log sink must never
    fail (or traceback) a run whose image is already on disk — only primary
    pipeline failures propagate.
    """
    require_run_config(config)
    if cancel_event is not None and not isinstance(cancel_event, threading.Event):
        raise ConfigError(FieldError("cancel_event", "must be a threading.Event or None"))
    if phase_callback is not None and not callable(phase_callback):
        raise ConfigError(FieldError("phase_callback", "must be callable or None"))
    logger = logger or PhaseLogger()
    conditioning: Any = None
    latents: Any = None
    image: Any = None
    def phase(name):
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("generation cancelled before verified save")
        if phase_callback is not None:
            phase_callback(name)
        # A callback may request cancellation itself.
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("generation cancelled before verified save")
        return logger.phase(name)

    primary = None
    saved = False
    try:
        binder = getattr(adapter, "set_cancel_event", None)
        if binder is not None:
            binder(cancel_event)
        with phase("load"):
            adapter.load(config, assets)
        with phase("encode"):
            conditioning = adapter.encode(config)
        with phase("denoise"):
            latents = adapter.denoise(config, conditioning)
        with phase("decode"):
            image = adapter.decode(latents)
        with phase("save"):
            output = adapter.save(image, config)
            output_path = _validate_output(output, config)
            saved = True
    except BaseException as exc:
        primary = exc
        raise
    finally:
        # Drop intermediates before releasing model references. Production
        # adapters own device teardown; never touch global allocators here.
        conditioning = latents = image = None
        try:
            closer = getattr(adapter, "close", None)
            if closer is not None:
                closer()
        except Exception:
            if primary is None and not saved:
                raise
            # Preserve the primary failure (including interrupts), and a
            # verified saved result. Cleanup failure is diagnostic only there.
            try:
                logger.event("cleanup_failed")
            except Exception:
                pass
    try:
        logger.event("complete", latent_tokens=config.latent_tokens,
                     steps=config.steps)
    except Exception:
        pass  # The save is done and verified: a sink that dies on this
        # ordinary completion emission must not turn a successful run into
        # a failure (review fix 11 — the phase guards do not cover it).
        # The emitted field names are fixed literals that shadow no
        # reserved key, so the only exceptions reachable here come from
        # the logging plumbing itself; every primary pipeline, save and
        # config failure was raised *before* this point and propagates
        # unchanged.
    return GenerationResult(output_path=str(output_path))
