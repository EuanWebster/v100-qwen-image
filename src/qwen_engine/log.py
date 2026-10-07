"""Structured phase/event JSONL logging (plan R1, task 1.1).

Stdlib-only. A :class:`PhaseLogger` never writes anywhere by itself: it emits
JSONL lines to an *injected* sink, and the only sink that touches the
filesystem is :func:`file_sink`, which writes exclusively to the path it was
configured with.  Clocks are injectable so tests (and future benchmark
replay) can make timing deterministic.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Record keys that events may not shadow with extra fields.
RESERVED_KEYS = frozenset({"ts", "phase", "event", "elapsed_ms"})

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
    _stack: list[tuple[str, float]] = field(default_factory=list, repr=False)

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
            elapsed_ms = (self.clock() - active[1]) * 1000.0
        record: dict[str, Any] = {
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
        """Context manager timing one engine phase (load/encode/denoise/...)."""
        start = self.clock()
        self._stack.append((name, start))
        self._emit({"ts": self.wall_clock(), "phase": name,
                    "event": "phase_start", "elapsed_ms": 0.0})
        try:
            yield self
        except BaseException as exc:
            elapsed_ms = (self.clock() - start) * 1000.0
            self._emit({"ts": self.wall_clock(), "phase": name,
                        "event": "phase_error", "elapsed_ms": elapsed_ms,
                        "error": type(exc).__name__})
            raise
        else:
            elapsed_ms = (self.clock() - start) * 1000.0
            self._emit({"ts": self.wall_clock(), "phase": name,
                        "event": "phase_end", "elapsed_ms": elapsed_ms})
            if self.human_sink is not None:
                self.human_sink(f"[{name}] done in {elapsed_ms:.0f} ms")
        finally:
            self._stack.pop()
