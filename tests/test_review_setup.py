"""Setup preflight tests: fake interpreters/uv, no installs or downloads.

All fixtures are disposable; reusable tests and actual review environments
stay on the project mount. Bash operates only on fixture paths here.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/setup_review_envs.sh"
PROFILES = {"py311": "3.11.15", "py313": "3.13.5", "test311": "3.11.15"}


def prepare(tmp_path, *, environments=True):
    project = tmp_path / "project with spaces"
    script = project / "scripts/setup_review_envs.sh"
    script.parent.mkdir(parents=True)
    script.write_text(SCRIPT.read_text())
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$UV_REVIEW_TEST_LOG"\n')
    uv.chmod(0o755)
    if environments:
        for name, version in PROFILES.items():
            interpreter(project, name, version)
    log = tmp_path / "uv.calls"
    env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
               UV_REVIEW_TEST_LOG=str(log))
    return project, script, log, env


def interpreter(project, name, version, *, prefix=None, kind="venv"):
    root = project / ".review-envs" / name
    path = root / "bin/python"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = (version, str(root) if prefix is None else str(prefix), kind)
    path.write_text("#!/bin/sh\nprintf '%s\\n' " + " ".join(map(shlex.quote, values)) + "\n")
    path.chmod(0o755)


def run(prepared, *args):
    project, script, log, env = prepared
    result = subprocess.run(["bash", str(script), *args], env=env,
                            capture_output=True, text=True, timeout=5)
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


def test_check_mode_never_synchronizes_or_creates_review_paths(tmp_path):
    prepared = prepare(tmp_path)
    project = prepared[0]
    result, calls = run(prepared, "--check")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3 and all(call.startswith("pip check --python ") for call in calls)
    assert "no synchronization or downloads" in result.stdout
    assert not (project / ".review-envs/uv-cache").exists()
    assert not (project / ".review-envs/python").exists()


def test_check_missing_environment_is_read_only_and_never_calls_uv(tmp_path):
    prepared = prepare(tmp_path, environments=False)
    result, calls = run(prepared, "--check")
    assert result.returncode == 1
    assert "environment missing" in result.stderr
    assert not calls
    assert not (prepared[0] / ".review-envs").exists()


@pytest.mark.parametrize("name", [".review-envs", ".review-envs/uv-cache", ".review-envs/python",
                                 ".review-envs/py311", ".review-envs/py313", ".review-envs/test311",
                                 ".review-envs/py313/bin"])
def test_setup_refuses_aliases_before_any_uv_or_target_mutation(tmp_path, name):
    prepared = prepare(tmp_path, environments=False)
    project = prepared[0]
    outside = tmp_path / "unrelated"
    outside.mkdir()
    sentinel = outside / "keep"
    sentinel.write_text("untouched")
    alias = project / name
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(outside, target_is_directory=True)
    result, calls = run(prepared)
    assert result.returncode == 1
    assert "symlinked review path" in result.stderr
    assert not calls
    assert sentinel.read_text() == "untouched"
    assert list(outside.iterdir()) == [sentinel]


@pytest.mark.parametrize("fault", ["version", "prefix", "base", "missing-python"])
def test_preflight_checks_all_existing_profiles_before_sync(tmp_path, fault):
    prepared = prepare(tmp_path)
    project = prepared[0]
    if fault == "version":
        interpreter(project, "py313", "3.13.6")
    elif fault == "prefix":
        interpreter(project, "py313", "3.13.5", prefix=project / ".venv")
    elif fault == "base":
        interpreter(project, "py313", "3.13.5", kind="base")
    else:
        (project / ".review-envs/py313/bin/python").unlink()
    result, calls = run(prepared)
    assert result.returncode == 1, result.stderr
    assert not calls  # py311 must not sync first just because it matched
    assert not (project / ".review-envs/uv-cache").exists()


def test_non_directory_target_refuses_before_sync(tmp_path):
    prepared = prepare(tmp_path, environments=False)
    target = prepared[0] / ".review-envs"
    target.write_text("not a directory")
    result, calls = run(prepared)
    assert result.returncode == 1
    assert "not a directory" in result.stderr
    assert target.read_text() == "not a directory"
    assert not calls


def test_explicit_setup_syncs_only_review_profiles_with_locked_inputs(tmp_path):
    prepared = prepare(tmp_path)
    project = prepared[0]
    result, calls = run(prepared)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 5
    assert sum(call.startswith("pip sync --python ") for call in calls) == 2
    assert sum(call.startswith("pip check --python ") for call in calls) == 2
    assert calls[-1].startswith("sync --locked --project ")
    for call in calls:
        assert str(project / ".venv") not in call
    assert all("--index-strategy unsafe-best-match" in call and
               str(project / "requirements.lock") in call
               for call in calls if call.startswith("pip sync"))


@pytest.mark.parametrize("args", [("--unknown",), ("--check", "extra")])
def test_invalid_setup_arguments_refuse_before_side_effects(tmp_path, args):
    prepared = prepare(tmp_path, environments=False)
    result, calls = run(prepared, *args)
    assert result.returncode == 2
    assert not calls
    assert not (prepared[0] / ".review-envs").exists()
