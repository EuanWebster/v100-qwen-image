"""Immutable generation config + field-level validation (plan R1/R2, task 1.1).

Stdlib-only. No I/O at import or construction time; every runtime path is a
config-relocatable string field (R7) — ``None`` means *not configured*, and
then nothing is written there (there are no implicit default write targets).

Latent token math follows the confirmed /16 spatial compression (C6):
1024² -> 4096 tokens, 2048² -> 16384 tokens. Width and height must be
multiples of 32.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

#: Linear spatial compression factor of the VAE latent grid (C6).
SPATIAL_COMPRESSION = 16

#: Width and height must be multiples of this (latent patching at /16 with 2x2
#: patches; validated per plan R2 criterion 2).
DIM_MULTIPLE = 32

DIM_MIN, DIM_MAX = 64, 4096
STEPS_MIN, STEPS_MAX = 1, 1000
SEED_MIN, SEED_MAX = 0, 2**64 - 1


class Quant(StrEnum):
    FP16 = "fp16"
    Q8 = "q8"
    Q4 = "q4"


class OffloadMode(StrEnum):
    RESIDENT = "resident"
    STREAMED = "streamed"
    OFFLOAD = "offload"


class AttentionBackend(StrEnum):
    AUTO = "auto"
    SDPA = "sdpa"
    EFFICIENT = "efficient"
    EAGER = "eager"


@dataclass(frozen=True)
class FieldError:
    """A single field-level validation failure; ``field`` names the config field."""

    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


class ConfigError(ValueError):
    """Raised for any invalid/malformed configuration. Exit code 2 in the CLI."""

    def __init__(self, errors: list[FieldError] | FieldError | str) -> None:
        if isinstance(errors, str):
            errors = [FieldError("config", errors)]
        if isinstance(errors, FieldError):
            errors = [errors]
        self.errors: tuple[FieldError, ...] = tuple(errors)
        super().__init__("invalid config: " + "; ".join(str(e) for e in self.errors))

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(e.field for e in self.errors)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_int(name: str, value: Any, lo: int, hi: int, errors: list[FieldError]) -> None:
    if not _is_int(value):
        errors.append(FieldError(name, f"must be an integer, got {value!r}"))
    elif not lo <= value <= hi:
        errors.append(FieldError(name, f"must be in range [{lo}, {hi}], got {value}"))


def _check_dim(name: str, value: Any, errors: list[FieldError]) -> None:
    if not _is_int(value):
        errors.append(FieldError(name, f"must be an integer, got {value!r}"))
        return
    if not DIM_MIN <= value <= DIM_MAX:
        errors.append(
            FieldError(name, f"must be in range [{DIM_MIN}, {DIM_MAX}], got {value}")
        )
    elif value % DIM_MULTIPLE != 0:
        errors.append(
            FieldError(
                name,
                f"must be a multiple of {DIM_MULTIPLE}, got {value}",
            )
        )


@dataclass(frozen=True)
class GenerationConfig:
    """Frozen generation configuration with lossless JSON round-trip."""

    prompt: str = "a test image"
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    seed: int | None = 0
    quant: Quant = Quant.FP16
    offload: OffloadMode = OffloadMode.RESIDENT
    attention_backend: AttentionBackend = AttentionBackend.AUTO
    vae_tiled_decode: bool = False
    # Relocatable runtime paths (R7). None = not configured -> nothing is
    # written there; tests redirect all of these to tmp_path.
    output_dir: str | None = None
    cache_dir: str | None = None
    log_dir: str | None = None
    benchmarks_dir: str | None = None
    models_dir: str | None = None
    manifest_path: str | None = None

    _ENUM_FIELDS: ClassVar[dict[str, type[StrEnum]]] = {
        "quant": Quant,
        "offload": OffloadMode,
        "attention_backend": AttentionBackend,
    }

    def __post_init__(self) -> None:
        errors: list[FieldError] = []
        # Coerce enum-valued strings and Path-like values so construction and
        # from_dict agree; coercion failures become field errors.
        for name, enum_cls in self._ENUM_FIELDS.items():
            value = getattr(self, name)
            if isinstance(value, enum_cls):
                continue
            if isinstance(value, str):
                try:
                    object.__setattr__(self, name, enum_cls(value))
                except ValueError:
                    errors.append(
                        FieldError(
                            name,
                            f"invalid value {value!r}, expected one of "
                            + ", ".join(repr(m.value) for m in enum_cls),
                        )
                    )
            else:
                errors.append(
                    FieldError(
                        name,
                        f"must be one of "
                        + ", ".join(repr(m.value) for m in enum_cls)
                        + f", got {value!r}",
                    )
                )
        for name in (
            "output_dir", "cache_dir", "log_dir", "benchmarks_dir",
            "models_dir", "manifest_path",
        ):
            value = getattr(self, name)
            if isinstance(value, Path):
                object.__setattr__(self, name, str(value))
            elif value is not None and not isinstance(value, str):
                errors.append(
                    FieldError(name, f"must be a path string or null, got {value!r}")
                )

        if not isinstance(self.prompt, str) or not self.prompt.strip():
            errors.append(FieldError("prompt", "must be a non-empty string"))
        if not isinstance(self.negative_prompt, str):
            errors.append(
                FieldError("negative_prompt", f"must be a string, got {self.negative_prompt!r}")
            )
        _check_dim("width", self.width, errors)
        _check_dim("height", self.height, errors)
        _check_int("steps", self.steps, STEPS_MIN, STEPS_MAX, errors)
        if self.seed is not None:
            if not _is_int(self.seed):
                errors.append(FieldError("seed", f"must be an integer or null, got {self.seed!r}"))
            elif not SEED_MIN <= self.seed <= SEED_MAX:
                errors.append(
                    FieldError("seed", f"must be in range [{SEED_MIN}, {SEED_MAX}], got {self.seed}")
                )
        if not isinstance(self.vae_tiled_decode, bool):
            errors.append(
                FieldError(
                    "vae_tiled_decode",
                    f"must be a boolean, got {self.vae_tiled_decode!r}",
                )
            )

        if errors:
            raise ConfigError(errors)

    # --- derived values -------------------------------------------------------

    @property
    def latent_tokens(self) -> int:
        """Exact latent token count: /16 spatial compression (C6).

        Validated configs guarantee multiples of 32, so the division is exact.
        """
        return (self.width // SPATIAL_COMPRESSION) * (self.height // SPATIAL_COMPRESSION)

    # --- JSON / dict round-trip -----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = value.value if isinstance(value, StrEnum) else value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenerationConfig":
        if not isinstance(data, Mapping):
            raise ConfigError(
                FieldError("config", f"must be a JSON object, got {type(data).__name__}")
            )
        known = {f.name for f in fields(cls)}
        errors = [
            FieldError(key, "unknown config field")
            for key in data
            if key not in known
        ]
        if errors:
            raise ConfigError(errors)
        try:
            return cls(**dict(data))
        except TypeError as exc:  # missing positional / structural problem
            raise ConfigError(FieldError("config", str(exc))) from exc

    def to_json(self) -> str:
        """Deterministic, key-sorted JSON (stable byte-for-byte across calls)."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "GenerationConfig":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(FieldError("config", f"invalid JSON: {exc}")) from exc
        return cls.from_dict(data)
