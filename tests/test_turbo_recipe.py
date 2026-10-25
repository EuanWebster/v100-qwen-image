"""Recipe-layer guards for docs/turbo-plan.md §3–§5 (CPU-only, no weight reads).

Everything here is either pure logic or reads only the small JSON configs a
bundle carries; safetensors are never opened. The constant-fidelity guard
(§4.4) needs the real Turbo asset directory: point QWEN_TURBO_ASSETS_DIR at
it (e.g. ``models/turbo/assets`` of a materialised checkout); without the
files the guard skips, but every file it reads is first tied to the sha256
recorded in this repo's committed models/turbo/manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from qwen_engine.adapters.qwen21_diffusers import (
    TURBO_SAMPLE_SIGMAS,
    TURBO_SCHEDULER_CONFIG,
    _assert_turbo_scheduler,
    _validate_policy,
    turbo_bundle_paths,
)
from qwen_engine.config import GenerationConfig
from qwen_engine.errors import EngineError

ROOT = Path(__file__).resolve().parents[1]


# --- path derivation (§3.3) -----------------------------------------------------


def test_turbo_bundle_paths_derive_from_the_base_manifest_file():
    manifest, assets = turbo_bundle_paths("/srv/models/manifest.json")
    assert manifest == Path("/srv/models/turbo/manifest.json")
    assert assets == Path("/srv/models/turbo/assets")
    manifest, assets = turbo_bundle_paths(Path("models/manifest.json"))
    assert manifest == Path("models/turbo/manifest.json")
    assert assets == Path("models/turbo/assets")


# --- the shipped-file scheduler assertion (§3.2) --------------------------------


def _write_scheduler_config(dir_path: Path, **over) -> Path:
    document = dict(TURBO_SCHEDULER_CONFIG)
    document["_class_name"] = "FlowMatchEulerDiscreteScheduler"
    document["_diffusers_version"] = "0.41.0.dev0"
    document.update(over)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "scheduler_config.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return dir_path


def test_assert_accepts_the_turbo_file_and_ignores_internal_keys(tmp_path):
    _assert_turbo_scheduler(_write_scheduler_config(tmp_path / "scheduler"))


def test_assert_refuses_the_base_two_field_drift(tmp_path):
    base = _write_scheduler_config(tmp_path / "scheduler",
                                   shift_terminal=0.02,
                                   use_dynamic_shifting=True)
    with pytest.raises(EngineError, match="changed: shift_terminal, use_dynamic_shifting"):
        _assert_turbo_scheduler(base)


def test_assert_refuses_missing_and_unexpected_fields(tmp_path):
    directory = tmp_path / "scheduler"
    directory.mkdir()
    document = dict(TURBO_SCHEDULER_CONFIG)
    del document["shift_terminal"]
    document["temperature"] = 0.7
    (directory / "scheduler_config.json").write_text(json.dumps(document))
    with pytest.raises(EngineError, match=r"missing: shift_terminal; unexpected: temperature"):
        _assert_turbo_scheduler(directory)


def test_assert_refuses_unreadable_or_non_object_configs(tmp_path):
    (tmp_path / "scheduler_config.json").write_text("{oops")
    with pytest.raises(EngineError, match="unreadable"):
        _assert_turbo_scheduler(tmp_path)
    (tmp_path / "scheduler_config.json").write_text("[1, 2]")
    with pytest.raises(EngineError, match="not a JSON object"):
        _assert_turbo_scheduler(tmp_path)


# --- policy mirror (§5.2): a valid turbo config passes the adapter gate ---------


def test_policy_accepts_a_valid_turbo_config(tmp_path):
    _validate_policy(GenerationConfig(
        recipe="turbo", steps=8, output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"), manifest_path=str(tmp_path / "m.json"),
        models_dir=str(tmp_path / "models")))


# --- constant fidelity against the sha256-pinned bundle files (§4.4) ------------

_ASSETS_CANDIDATES = [
    os.environ.get("QWEN_TURBO_ASSETS_DIR"),
    str(ROOT / "models" / "turbo" / "assets"),
]


def _pinned(relative: str) -> str | None:
    """The sha256 the committed turbo manifest claims for a relative file."""
    manifest = json.loads((ROOT / "models" / "turbo" / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        if artifact["filename"] == relative:
            return artifact["sha256"]
    return None


@pytest.mark.parametrize("relative", ["model_index.json",
                                      "scheduler/scheduler_config.json"])
def test_reviewed_constants_match_the_hash_pinned_bundle_files(relative):
    root = next((Path(candidate) for candidate in _ASSETS_CANDIDATES
                 if candidate and (Path(candidate) / relative).is_file()), None)
    if root is None:
        pytest.skip("Turbo asset directory not present (set QWEN_TURBO_ASSETS_DIR)")
    path = root / relative
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == _pinned(relative), (
        f"{relative} on disk does not match the committed turbo manifest")
    document = json.loads(content)
    if relative.endswith("model_index.json"):
        assert list(TURBO_SAMPLE_SIGMAS) == document["sample_sigmas"]
    else:
        observed = {key: value for key, value in document.items()
                    if not key.startswith("_")}
        assert observed == TURBO_SCHEDULER_CONFIG
