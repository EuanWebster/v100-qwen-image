"""Strict JSON snapshot helpers shared by the encoder and bench records.

Frozen records must keep holding *exactly* the validated values they were
constructed with: no mutation through a reference the caller still owns, no
``NaN``/``Infinity`` sneaking in via ``json.dumps`` (which otherwise emits
invalid JSON silently), and no non-string object keys (``json.dumps`` turns
``{1: ...}`` into ``"1"`` — a different document than the cache key saw).

:func:`freeze_json` deep-copies a validated JSON value into structures the
record owns — mappings as :class:`types.MappingProxyType`, sequences as
``tuple``, scalars unchanged — so a frozen dataclass is immutable *all the
way down*. :func:`plain_json` converts such a snapshot back into plain
dicts/lists for canonical serialization; both representations describe the
same JSON document, which is what lets one request produce one key and one
wire document.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_json(value: Any, at: str, errors: list[str]) -> Any:
    """Deep-freeze a strict JSON value, appending every violation found to
    ``errors``. Valid values come back as immutable snapshots (mapping →
    ``MappingProxyType``, list/tuple → ``tuple``); invalid values append
    errors and are not usable."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"{at}: must be a finite JSON number, got {value!r}")
            return None
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, val in value.items():
            if not isinstance(key, str):
                errors.append(f"{at}: JSON object keys must be strings, "
                              f"got {key!r}")
                continue
            out[key] = freeze_json(val, f"{at}.{key}", errors)
        return MappingProxyType(out)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(v, f"{at}[{i}]", errors)
                     for i, v in enumerate(value))
    errors.append(f"{at}: unsupported JSON type {type(value).__name__}")
    return None


def plain_json(value: Any) -> Any:
    """Inverse view of :func:`freeze_json`: plain dicts/lists ready for
    ``json.dumps`` (which does not serialize ``MappingProxyType``)."""
    if isinstance(value, Mapping):
        return {k: plain_json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [plain_json(v) for v in value]
    return value


def json_violations(value: Any, at: str = "value") -> list[str]:
    """Validation-only view of :func:`freeze_json` (snapshot discarded)."""
    errors: list[str] = []
    freeze_json(value, at, errors)
    return errors


def reject_constant(name: str) -> Any:
    """``json.loads(..., parse_constant=reject_constant)`` turns the
    ``NaN``/``Infinity``/-``Infinity`` literals into a typed parse failure
    instead of non-finite floats."""
    raise ValueError(f"non-finite JSON constant {name!r}")
