"""Structured phase/event JSONL logging (plan R1, task 1.1).

Stdlib-only. A :class:`PhaseLogger` never writes anywhere by itself: it emits
JSONL lines to an *injected* sink, and the only sink that touches the
filesystem is :func:`file_sink`, which writes exclusively to the path it was
configured with.  Clocks are injectable so tests (and future benchmark
replay) can make timing deterministic.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Record keys that events may not shadow with extra fields.
RESERVED_KEYS = frozenset({"ts", "phase", "event", "elapsed_ms",
                           "schema_version", "run_id"})

#: Schema version of a log record (future readers dispatch on it).
LOG_SCHEMA_VERSION = 1

#: Phase name reported when an event arrives outside any active phase.
DEFAULT_PHASE = "engine"

Sink = Callable[[str], None]


def _null_sink(line: str) -> None:  # pragma: no cover - trivial
    """Default sink: discard. Nothing is written unless a sink is configured."""


def file_sink(path: str | Path) -> Sink:
    """A sink that appends JSONL lines to ``path`` and nowhere else.

    Parent directories are created on first write; each line is flushed so a
    crashed run keeps the events it logged.
    """
    target = Path(path)

    def sink(line: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    return sink


@dataclass
class PhaseLogger:
    """Emit ``{ts, phase, event, elapsed_ms, ...}`` JSONL records.

    ``sink`` receives one serialized line per event (default: discard —
    **no default writes**). ``clock`` (monotonic seconds) drives
    ``elapsed_ms``; ``wall_clock`` (epoch seconds) fills ``ts``. Both are
    injectable. ``human_sink`` additionally receives short phase lines.
    """

    sink: Sink = _null_sink
    human_sink: Sink | None = None
    clock: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], float] = time.time
    #: Record schema version — every record carries it.
    schema_version: int = LOG_SCHEMA_VERSION
    #: Injectable run identifier so logs from concurrent runs stay
    #: attributable. Default ``None``: single-run logs need no attribution;
    #: there are still **no default file writes** anywhere.
    run_id: str | None = None
    _stack: list[tuple[str, float | None]] = field(default_factory=list, repr=False)

    def _clock_value(self) -> float | None:
        try:
            value = self.clock()
            if type(value) not in (int, float) or not math.isfinite(value):
                return None
            return float(value)
        except Exception:
            return None  # timing is diagnostic, not a pipeline prerequisite

    def _elapsed(self, start: float | None) -> float | None:
        end = self._clock_value() if start is not None else None
        if start is None or end is None:
            return None
        elapsed = (end - start) * 1000.0
        return elapsed if math.isfinite(elapsed) else None

    # --- emitting ---------------------------------------------------------

    def event(self, name: str, *, phase: str | None = None, **fields: Any) -> None:
        reserved = RESERVED_KEYS & set(fields)
        if reserved:
            raise ValueError(
                "event fields shadow reserved record keys: "
                + ", ".join(sorted(reserved))
            )
        active = self._stack[-1] if self._stack else None
        current = phase if phase is not None else (active[0] if active else DEFAULT_PHASE)
        elapsed_ms: float | None = None
        if active is not None and current == active[0]:
            elapsed_ms = self._elapsed(active[1])
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "ts": self.wall_clock(),
            "phase": current,
            "event": name,
            "elapsed_ms": elapsed_ms,
        }
        record.update(fields)
        self._emit(record)

    def _emit(self, record: dict[str, Any]) -> None:
        self.sink(json.dumps(record, sort_keys=True))

    # --- phases -----------------------------------------------------------

    @contextmanager
    def phase(self, name: str):
        """Context manager timing one engine phase (load/encode/denoise/...).

        The stack entry is popped on **every** exit path, including a sink
        that fails while the ``phase_start`` record is emitted, and a failing
        ``phase_error`` emission never masks the primary exception.
        Completion logging is equally nonfatal: a sink that dies while the
        ``phase_end`` record or the ``human_sink`` line is emitted on an
        otherwise-successful body does **not** fail the phase — an image
        saved successfully must still exit 0, not report a log fault.  Only
        the body's own exceptions propagate. Ordinary clock failures likewise
        do not abort work or mask a body error; unavailable timing is null.
        """
        start = self._clock_value()
        self._stack.append((name, start))
        try:
            try:
                self._emit({"schema_version": self.schema_version,
                            "run_id": self.run_id,
                            "ts": self.wall_clock(), "phase": name,
                            "event": "phase_start", "elapsed_ms": 0.0})
            except Exception:
                pass  # a dead sink must not abort the phase itself
            try:
                yield self
            except BaseException as exc:
                try:
                    elapsed_ms = self._elapsed(start)
                    self._emit({"schema_version": self.schema_version,
                                "run_id": self.run_id,
                                "ts": self.wall_clock(), "phase": name,
                                "event": "phase_error", "elapsed_ms": elapsed_ms,
                                "error": type(exc).__name__})
                except Exception:
                    pass  # never mask the primary exception with a log fault
                raise
            else:
                elapsed_ms = self._elapsed(start)
                try:
                    self._emit({"schema_version": self.schema_version,
                                "run_id": self.run_id,
                                "ts": self.wall_clock(), "phase": name,
                                "event": "phase_end",
                                "elapsed_ms": elapsed_ms})
                except Exception:
                    pass  # a completed phase stays completed despite a log fault
                if self.human_sink is not None:
                    try:
                        self.human_sink(
                            f"[{name}] done in {elapsed_ms:.0f} ms" if elapsed_ms is not None
                            else f"[{name}] done (timing unavailable)")
                    except Exception:
                        pass  # likewise for the human-readable completion line
        finally:
            self._stack.pop()
