"""Pipeline adapter protocol + plan/dry-run types (plan R1, task 1.3).

The engine core speaks only this protocol. The only module allowed to import
torch / diffusers / transformers is ``qwen_engine.adapters.qwen21_diffusers``
— **which does not exist yet**; stubbing it at M1 is a boundary violation the
import test catches. Heavy imports stay deferred inside that future adapter.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from qwen_engine.config import (
    AttentionBackend,
    GenerationConfig,
    OffloadMode,
    Quant,
)

#: Enum choices surfaced by the CLI (kept here so cli stays enum-agnostic).
QUANT_CHOICES = [m.value for m in Quant]
OFFLOAD_CHOICES = [m.value for m in OffloadMode]
ATTENTION_BACKEND_CHOICES = [m.value for m in AttentionBackend]


@runtime_checkable
class GenerationAdapter(Protocol):
    """What the engine core calls, in order: load/encode/denoise/decode/save.

    (Dry-run planning is adapter-independent at this milestone — the core
    builds the plan document itself; there is deliberately no ``plan`` here.)
    """

    def load(self, config: GenerationConfig,
             assets: Mapping[str, tuple[Path, ...]]) -> None:
        """Load/attach the verified model components (the heavy step; deferred
        imports live here). ``assets`` are the resolver's verified paths
        grouped by role — the adapter must take its weights from there, never
        from a repo id or an ambient HF cache."""
        ...

    def encode(self, config: GenerationConfig) -> Any:
        """Encode prompts into conditioning (subprocess encoder in production)."""
        ...

    def denoise(self, config: GenerationConfig, conditioning: Any) -> Any:
        """Run the sampling loop; return latents."""
        ...

    def decode(self, latents: Any) -> Any:
        """VAE-decode latents to an image."""
        ...

    def save(self, image: Any, config: GenerationConfig) -> Any:
        """Persist the image under ``config.output_dir``; return its path."""
        ...


@dataclass(frozen=True)
class UnknownStatus:
    """A value that was *not measured* — reported honestly in dry-run output.

    Never fabricated: hardware and calibration report this until real probes
    (M3) and calibration records (M1.5/M5.3) exist.
    """

    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"status": "unknown", "detail": self.detail}


HARDWARE_UNKNOWN = UnknownStatus(
    "no hardware probe at M1; a confirmed V100 (CUDA sm_70, measured CUDA "
    "execution) is required and lands with M3"
)
CALIBRATION_UNKNOWN = UnknownStatus(
    "no benchmark_result.v1 / calibration records exist yet (M1.5/M5.3); the "
    "memory policy will refuse uncalibrated configs rather than guess"
)
