"""Engine error types + the CLI exit-code contract (plan R2 / M1 criteria).

Exit codes: ``0`` success · ``2`` invalid/malformed config · ``3`` memory guard
refusal (reserved — M5.3) · ``4`` missing/unverified model asset · ``5``
required hardware absent or unverified (reserved — M3/M5.3) · ``1`` unexpected
(logged with context; traceback goes to the log sink, never stderr).
"""
from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    UNEXPECTED = 1
    CONFIG = 2
    MEMORY_REFUSAL = 3
    ASSET = 4
    HARDWARE = 5


class EngineError(Exception):
    """Base for expected engine failures (no traceback on the console)."""


class MemoryGuardError(EngineError):
    """Reserved (M5.3): the memory policy refused this config. Exit 3."""


class HardwareAbsentError(EngineError):
    """Reserved (M3/M5.3): required hardware absent or unverified. Exit 5."""


class GenerationCancelled(EngineError):
    """Cooperative cancellation acknowledged before a verified save."""
