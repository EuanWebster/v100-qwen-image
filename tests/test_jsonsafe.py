"""Shared strict-JSON snapshot helpers: cycles, depth bound, DAG sharing.

These are the freeze/validation primitives behind the encoder request and
the bench records, tested at the boundary where a cyclic or pathological
value must become a path-specific error — never a RecursionError.
"""
from __future__ import annotations

import pytest

from qwen_engine._jsonsafe import (
    MAX_JSON_DEPTH,
    freeze_json,
    json_violations,
    plain_json,
)


def _freeze(value):
    errors: list[str] = []
    out = freeze_json(value, "root", errors)
    return out, errors


def test_dict_cycle_reports_path_specific_error():
    d: dict = {"a": {}}
    d["a"]["back"] = d
    out, errors = _freeze(d)
    assert errors and out is not None  # error appended at the revisit point
    assert any("cyclic" in e and e.startswith("root.a.back") for e in errors)


def test_list_cycle_direct_self_reference():
    loop: list = []
    loop.append(loop)
    _, errors = _freeze({"l": loop})
    assert any("cyclic" in e and "root.l" in e for e in errors)


def test_indirect_cycle_list_through_dict():
    a: list = []
    b = {"a": a}
    a.append(b)
    _, errors = _freeze({"top": a})
    assert any("cyclic" in e and e.startswith("root.top[0].a") for e in errors)


def test_shared_acyclic_subtree_is_allowed_and_frozen():
    shared = {"x": [1, 2]}
    value = {"p": shared, "q": {"r": shared}}
    out, errors = _freeze(value)
    assert errors == []
    assert plain_json(out) == value
    with pytest.raises(TypeError):
        out["p"]["x"] = 0  # type: ignore[index]
    with pytest.raises(AttributeError):
        out["p"]["x"].append(3)  # type: ignore[union-attr]


def test_excessive_depth_is_a_typed_error_not_recursion_error():
    deep: list = []
    for _ in range(MAX_JSON_DEPTH + 50):
        deep = [deep]
    _, errors = _freeze(deep)
    assert any("deeper" in e for e in errors)


def test_json_violations_detects_cycles_without_recurring_forever():
    d: dict = {}
    d["self"] = d
    errors = json_violations(d)
    assert any("cyclic" in e for e in errors)


def test_plain_values_still_freeze_identically():
    value = {"s": "x", "n": 3, "f": 1.5, "t": True, "z": None,
             "l": [1, [2, {"k": "v"}]]}
    out, errors = _freeze(value)
    assert errors == []
    assert plain_json(out) == value
