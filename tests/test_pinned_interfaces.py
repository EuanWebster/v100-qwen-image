"""Optional pinned-upstream contract test (task M0.2).

The two upstream checkouts are *external, explicitly-provided inputs*. This
test therefore NEVER fabricates a validation claim:

* ``QWEN_DIFFUSERS_SRC`` / ``QWEN_TRANSFORMERS_SRC`` set → the trees are
  required to exist; the contract checker runs in-process (pure stdlib AST —
  no torch/diffusers/transformers import, no network, R7-safe) and any
  contract failure fails this test.
* Either variable unset → SKIPPED with a loud, explicit reason. A skip says
  "the pinned-interface contract was NOT checked on this run", never "it
  passed". Run ``tools/check_pinned_interfaces.py`` against the pinned trees
  whenever a checkout is available (``docs/interfaces.md`` lists the commits).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_pinned_interfaces.py"

_diffusers = os.environ.get("QWEN_DIFFUSERS_SRC")
_transformers = os.environ.get("QWEN_TRANSFORMERS_SRC")

if not (_diffusers and _transformers):
    pytest.skip(
        "pinned upstream source trees NOT provided (QWEN_DIFFUSERS_SRC / "
        "QWEN_TRANSFORMERS_SRC unset): the pinned-interface contract was not "
        "checked on this run — this skip is not a pass (see docs/interfaces.md)",
        allow_module_level=True,
    )


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_pinned_interfaces", CHECKER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pinned_interface_contracts_hold(capsys):
    mod = _load_checker()
    rc = mod.main(["--diffusers-src", _diffusers, "--transformers-src", _transformers])
    out = capsys.readouterr().out
    assert rc == 0, f"pinned-interface drift (exit {rc}):\n{out}"
    assert "NOT RUN" not in out, f"source trees missing:\n{out}"
