"""Immutable generation config + field-level validation (plan R1/R2, task 1.1).

Stdlib-only. No I/O at *import* time; construction performs only lightweight
``exists``/``isdir`` checks on configured paths (never creates anything).
Every runtime path is a config-relocatable string field (R7) — ``None`` means
*not configured*, and then nothing is written there (there are no implicit
default write targets).

Latent token math follows the confirmed /16 spatial compression (C6):
1024² -> 4096 tokens, 2048² -> 16384 tokens. Width and height must be
multiples of 32.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from qwen_engine._jsonsafe import strict_json_loads

#: Linear spatial compression factor of the VAE latent grid (C6).
SPATIAL_COMPRESSION = 16

#: Width and height must be multiples of this (latent patching at /16 with 2x2
#: patches; validated per plan R2 criterion 2).
DIM_MULTIPLE = 32

DIM_MIN, DIM_MAX = 64, 4096
STEPS_MIN, STEPS_MAX = 1, 1000
SEED_MIN, SEED_MAX = 0, 2**64 - 1
#: true-CFG scale: finite and >= 1.0; 1.0 means CFG is off regardless of the
#: negative prompt (official Qwen example runs 40 steps — the default).
CFG_SCALE_MIN = 1.0
DEFAULT_STEPS = 40

#: Fields that must name an existing *directory* when they exist on disk.
_DIR_FIELDS = ("output_dir", "cache_dir", "log_dir", "benchmarks_dir",
               "models_dir")
#: Fields that must name an existing *file* when they exist on disk.
_FILE_FIELDS = ("manifest_path", "q8_manifest_path")


class Quant(StrEnum):
    FP16 = "fp16"
    Q8 = "q8"


class OutputFormat(StrEnum):
    JPG = "jpg"
    PNG = "png"
    WEBP = "webp"


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


def _is_finite_number(value: Any) -> bool:
    """Finite real check that never raises: ``math.isfinite(10**1000)``
    overflows at float conversion — a config error, not an OverflowError
    escaping the constructor (and escaping the CLI's exit-2 path)."""
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        return False


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
    steps: int = DEFAULT_STEPS
    true_cfg_scale: float = CFG_SCALE_MIN
    seed: int | None = 0
    quant: Quant = Quant.FP16
    offload: OffloadMode = OffloadMode.RESIDENT
    attention_backend: AttentionBackend = AttentionBackend.AUTO
    vae_tiled_decode: bool = False
    output_format: OutputFormat = OutputFormat.JPG
    # Relocatable runtime paths (R7). None = not configured -> nothing is
    # written there; tests redirect all of these to tmp_path.
    output_dir: str | None = None
    cache_dir: str | None = None
    log_dir: str | None = None
    benchmarks_dir: str | None = None
    models_dir: str | None = None
    manifest_path: str | None = None
    q8_manifest_path: str | None = None

    _ENUM_FIELDS: ClassVar[dict[str, type[StrEnum]]] = {
        "quant": Quant,
        "offload": OffloadMode,
        "attention_backend": AttentionBackend,
        "output_format": OutputFormat,
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
        for name in (*_DIR_FIELDS, *_FILE_FIELDS):
            value = getattr(self, name)
            if isinstance(value, Path):
                value = str(value)
                object.__setattr__(self, name, value)
            elif value is not None and not isinstance(value, str):
                errors.append(
                    FieldError(name, f"must be a path string or null, got {value!r}")
                )
                continue
            if value is None:
                continue
            if not value.strip():
                errors.append(FieldError(name, f"must not be blank, got {value!r}"))
            elif "\0" in value:
                errors.append(FieldError(name, "must not contain a NUL byte"))
            elif os.path.exists(value):
                # Known file/directory mismatch — validated, never repaired:
                # validation never creates directories or files.
                if name in _DIR_FIELDS and not os.path.isdir(value):
                    errors.append(
                        FieldError(name, f"path exists but is not a directory: {value!r}")
                    )
                elif name in _FILE_FIELDS and not os.path.isfile(value):
                    errors.append(
                        FieldError(name, f"path exists but is not a file: {value!r}")
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
        if isinstance(self.true_cfg_scale, bool) or not isinstance(
                self.true_cfg_scale, (int, float)):
            errors.append(
                FieldError("true_cfg_scale",
                           f"must be a finite number >= {CFG_SCALE_MIN}, "
                           f"got {self.true_cfg_scale!r}")
            )
        elif not _is_finite_number(self.true_cfg_scale) \
                or self.true_cfg_scale < CFG_SCALE_MIN:
            errors.append(
                FieldError("true_cfg_scale",
                           f"must be a finite number >= {CFG_SCALE_MIN}, "
                           f"got {self.true_cfg_scale!r}")
            )
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

    @property
    def cfg_active(self) -> bool:
        """True-CFG semantics: guidance is active only when a negative prompt
        is present **and** ``true_cfg_scale > 1``.

        With the scale at exactly 1.0 the negative prompt is not consulted at
        all (batch duplication would be wasted work); with no negative prompt
        a scale above 1 is meaningless. Both conditions must hold — this is
        the single source of truth for the adapter and the memory policy
        (CFG duplication in the R4 descriptor keys off it).
        """
        return bool(self.negative_prompt.strip()) and self.true_cfg_scale > 1.0

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
            data = strict_json_loads(text)
        except (ValueError, RecursionError, TypeError) as exc:
            # JSONDecodeError subclasses ValueError; the integer-string digit
            # limit raises plain ValueError and pathologically deep nesting
            # RecursionError — both are typed parse failures, never crashes.
            raise ConfigError(FieldError("config", f"invalid JSON: {exc}")) from exc
        return cls.from_dict(data)
