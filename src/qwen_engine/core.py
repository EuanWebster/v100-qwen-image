"""Engine core: ``plan()`` and ``generate()`` over the adapter protocol (R1/R2).

Stdlib-only; importable with no torch/diffusers/transformers installed. The
dry-run ``plan`` document is deterministic, key-sortable and free of
timestamps; unmeasured things (hardware, calibration) are reported as
``unknown`` — never fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qwen_engine.adapters.base import (
    CALIBRATION_UNKNOWN,
    HARDWARE_UNKNOWN,
    GenerationAdapter,
)
from qwen_engine.config import GenerationConfig
from qwen_engine.log import PhaseLogger

#: Schema tag of the dry-run document (versioned like the bench schemas, R6).
DRY_RUN_SCHEMA = "qwen_engine.dry_run.v1"

#: Phase sequence generate() drives, in order.
PHASES = ("load", "encode", "denoise", "decode", "save")


@dataclass(frozen=True)
class GenerationResult:
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return {"output_path": self.output_path}


def plan(config: GenerationConfig, *, manifest: dict[str, Any]) -> dict[str, Any]:
    """Deterministic dry-run plan for ``config`` (M1 acceptance criterion 4).

    Reports resolved config, latent token count, manifest resolution status,
    and *explicitly unknown* hardware / calibration — no timestamps, nothing
    measured that was not measured.
    """
    return {
        "schema": DRY_RUN_SCHEMA,
        "config": config.to_dict(),
        "latent_tokens": config.latent_tokens,
        "manifest": manifest,
        "hardware": HARDWARE_UNKNOWN.to_dict(),
        "calibration": CALIBRATION_UNKNOWN.to_dict(),
    }


def generate(
    config: GenerationConfig,
    adapter: GenerationAdapter,
    logger: PhaseLogger | None = None,
) -> GenerationResult:
    """Run the full pipeline through the adapter protocol, logging phases."""
    logger = logger or PhaseLogger()
    conditioning: Any = None
    latents: Any = None
    image: Any = None
    with logger.phase("load"):
        adapter.load()
    with logger.phase("encode"):
        conditioning = adapter.encode(config)
    with logger.phase("denoise"):
        latents = adapter.denoise(config, conditioning)
    with logger.phase("decode"):
        image = adapter.decode(latents)
    with logger.phase("save"):
        output = adapter.save(image, config)
    logger.event("complete", latent_tokens=config.latent_tokens,
                 steps=config.steps)
    return GenerationResult(output_path=str(output))
