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

from qwen_engine.adapters.base import (
    CALIBRATION_UNKNOWN,
    HARDWARE_UNKNOWN,
    GenerationAdapter,
)
from qwen_engine.config import ConfigError, FieldError, GenerationConfig
from qwen_engine.errors import EngineError
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


def generate(
    config: GenerationConfig,
    adapter: GenerationAdapter,
    assets: Assets,
    logger: PhaseLogger | None = None,
) -> GenerationResult:
    """Run the full pipeline through the adapter protocol, logging phases.

    ``assets`` are the resolver's verified artifact paths grouped by role and
    are handed to ``adapter.load`` verbatim. Nothing is written when the
    output is unconfigured: ``output_dir`` is refused **before** the adapter
    loads, so a missing output setting cannot cost a model load or touch the
    filesystem.
    """
    if config.output_dir is None:
        raise ConfigError(FieldError(
            "output_dir",
            "an actual run requires output_dir (nowhere to save); set it in "
            "the config or pass --output-dir",
        ))
    logger = logger or PhaseLogger()
    conditioning: Any = None
    latents: Any = None
    image: Any = None
    with logger.phase("load"):
        adapter.load(config, assets)
    with logger.phase("encode"):
        conditioning = adapter.encode(config)
    with logger.phase("denoise"):
        latents = adapter.denoise(config, conditioning)
    with logger.phase("decode"):
        image = adapter.decode(latents)
    with logger.phase("save"):
        output = adapter.save(image, config)
        output_path = _validate_output(output, config)
    logger.event("complete", latent_tokens=config.latent_tokens,
                 steps=config.steps)
    return GenerationResult(output_path=str(output_path))
