"""qwen_engine — hand-rolled Qwen-Image-2.1 engine core (M1 skeleton).

Importing this package must never pull in torch / diffusers / transformers;
heavy ML imports live only inside adapters (and those are not written yet).
"""
from qwen_engine.config import (
    AttentionBackend,
    ConfigError,
    FieldError,
    GenerationConfig,
    OffloadMode,
    Quant,
)
from qwen_engine.log import PhaseLogger, file_sink

__version__ = "0.1.0"

__all__ = [
    "AttentionBackend",
    "ConfigError",
    "FieldError",
    "GenerationConfig",
    "OffloadMode",
    "PhaseLogger",
    "Quant",
    "file_sink",
    "__version__",
]
