"""Task 1.3 acceptance: adapter protocol + core plan/generate boundary, CLI
contract (import/--help/--dry-run without heavy modules; exit-code contract
0/2/3(reserved)/4/5(reserved)/1; no tracebacks on expected errors)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_engine import cli, manifest as mf
from qwen_engine.adapters.base import GenerationAdapter
from qwen_engine.config import GenerationConfig
from qwen_engine.core import GenerationResult, generate, plan
from qwen_engine.errors import HardwareAbsentError, MemoryGuardError

REPO = "Qwen/Qwen-Image-2.1"
REV = "0" * 40
ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "diffusers", "transformers")


# --- helpers ---------------------------------------------------------------


class FakeAdapter:
    """In-process stand-in for the eventual Qwen21DiffusersAdapter."""

    name = "fake"

    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []

    def plan(self, config):
        self.calls.append("plan")
        return {"adapter": self.name}

    def load(self):
        self.calls.append("load")

    def encode(self, config):
        self.calls.append("encode")
        return {"embeds": "fake"}

    def denoise(self, config, conditioning):
        self.calls.append("denoise")
        return {"latents": "fake"}

    def decode(self, latents):
        self.calls.append("decode")
        return {"image": "fake"}

    def save(self, image, config):
        self.calls.append("save")
        return Path(config.output_dir or ".") / "out.png"


def write_manifest(tmp_path, role="dit", content=b"payload"):
    import hashlib
    data = {
        "schema_version": mf.MANIFEST_SCHEMA_VERSION,
        "artifacts": [{
            "role": role, "repo": REPO, "revision": REV,
            "filename": f"{role}/model.bin",
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
        }],
    }
    path = tmp_path / "models" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def config_file(tmp_path, **over):
    cfg = {"prompt": "a red cube", "width": 1024, "height": 1024,
           "seed": 7, "log_dir": str(tmp_path / "logs")}
    cfg.update(over)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def run_cli(tmp_path, args):
    return cli.main([str(a) for a in args])


# --- adapter protocol boundary ------------------------------------------------


def test_fake_adapter_satisfies_protocol():
    assert isinstance(FakeAdapter(), GenerationAdapter)


def test_generate_calls_adapter_in_phase_order(tmp_path, base_config):
    calls: list[str] = []
    result = generate(base_config, FakeAdapter(calls))
    assert calls == ["load", "encode", "denoise", "decode", "save"]
    assert isinstance(result, GenerationResult)
    assert result.output_path.endswith("out.png")


def test_plan_is_deterministic_and_free_of_heavy_state(base_config):
    status = mf.manifest_status(None, None)
    doc1 = plan(base_config, manifest=status)
    doc2 = plan(base_config, manifest=mf.manifest_status(None, None))
    text = json.dumps(doc1, sort_keys=True)
    assert text == json.dumps(doc2, sort_keys=True)
    assert doc1["latent_tokens"] == 4096
    assert doc1["hardware"]["status"] == "unknown"
    assert doc1["calibration"]["status"] == "unknown"
    assert doc1["manifest"]["status"] == "unconfigured"


def test_qwen21_diffusers_module_must_not_exist_yet():
    # Stubbing the real adapter at M1 is a boundary violation (PLAN R1).
    assert not (ROOT / "src" / "qwen_engine" / "adapters" /
                "qwen21_diffusers.py").exists()


# --- import / --help boundary (subprocess = true fresh interpreter) ------------


def _subp(*args, expect_code=0):
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin",
                               "PYTHONPATH": str(ROOT / "src"),
                               "HOME": "/tmp"})
    assert proc.returncode == expect_code, proc.stderr
    return proc


def test_import_engine_in_subprocess_without_heavy_modules():
    code = (
        "import sys, qwen_engine, qwen_engine.cli, qwen_engine.core, "
        "qwen_engine.manifest, qwen_engine.adapters.base;"
        f"bad=[m for m in sys.modules if m.split('.')[0] in {HEAVY}];"
        "print(bad); assert not bad"
    )
    _subp(sys.executable, "-c", code)


def test_cli_help_exits_zero_without_heavy_modules():
    proc = _subp(sys.executable, "-c",
                 "from qwen_engine.cli import main; main(['--help'])")
    assert "--dry-run" in proc.stdout


def test_cli_gen_shim_help_and_module_entry():
    proc = _subp(sys.executable, ROOT / "cli_gen.py", "--help")
    assert "--dry-run" in proc.stdout
    proc = _subp(sys.executable, "-m", "qwen_engine", "--help")
    assert "--dry-run" in proc.stdout


def test_cli_gen_shim_dry_run_runs_from_repo_root(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "cli_gen.py"), "--dry-run",
         "--config", str(config_file(tmp_path))],
        cwd=ROOT, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"})  # package from .venv
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["latent_tokens"] == 4096


# --- CLI: dry-run contract ------------------------------------------------------


def test_dry_run_is_key_sorted_deterministic_json(tmp_path, capsys):
    cfg = config_file(tmp_path)
    code1 = run_cli(tmp_path, ["--dry-run", "--config", cfg])
    out1 = capsys.readouterr().out
    code2 = run_cli(tmp_path, ["--dry-run", "--config", cfg])
    out2 = capsys.readouterr().out
    assert code1 == code2 == 0
    assert out1 == out2
    assert out1 == json.dumps(json.loads(out1), sort_keys=True, indent=2) + "\n"


def test_dry_run_reports_required_fields(tmp_path, capsys):
    write_manifest(tmp_path)  # present but not materialized
    assert run_cli(tmp_path, ["--dry-run", "--config", config_file(tmp_path)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["config"]["width"] == 1024 and doc["config"]["seed"] == 7
    assert doc["latent_tokens"] == 4096
    assert doc["manifest"]["status"] in ("unconfigured", "incomplete",
                                         "manifest_file_missing")
    assert doc["hardware"]["status"] == "unknown"
    assert doc["calibration"]["status"] == "unknown"


def test_dry_run_ok_manifest_and_no_timestamps(tmp_path, capsys):
    mpath = write_manifest(tmp_path)
    (tmp_path / "models" / "dit" / "model.bin").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "dit" / "model.bin").write_bytes(b"payload")
    cfg = config_file(tmp_path, manifest_path=str(mpath), models_dir=str(tmp_path / "models"))
    assert run_cli(tmp_path, ["--dry-run", "--config", cfg]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["manifest"]["status"] == "ok"
    lowered = json.dumps(doc).lower()
    for banned in ('"ts"', "timestamp", "time_", "date"):
        assert banned not in lowered


def test_dry_run_never_writes_outside_temp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "repoish"
    out_dir.mkdir()
    monkeypatch.chdir(out_dir)
    assert run_cli(tmp_path, ["--dry-run"]) == 0
    assert list(out_dir.iterdir()) == []


def test_dry_run_does_not_import_heavy_modules_in_process(tmp_path, capsys):
    assert run_cli(tmp_path, ["--dry-run"]) == 0
    capsys.readouterr()
    assert not set(HEAVY) & set(sys.modules)


# --- CLI: exit-code contract -----------------------------------------------------


def test_missing_config_file_exits_2(tmp_path, capsys):
    assert run_cli(tmp_path, ["--config", tmp_path / "nope.json"]) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err and "config" in err


def test_malformed_config_exits_2_naming_fields(tmp_path, capsys):
    bad = tmp_path / "config.json"
    bad.write_text('{"prompt": "p", "steps": 0, "quant": "int8"}')
    assert run_cli(tmp_path, ["--config", bad]) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "steps" in err and "quant" in err


def test_bad_cli_override_exits_2(tmp_path, capsys):
    assert run_cli(tmp_path, ["--width", "1000"]) == 2
    assert "width" in capsys.readouterr().err


def test_run_with_missing_assets_exits_4_and_names_artifact(tmp_path, capsys, monkeypatch):
    import socket
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("download attempted")))
    mpath = write_manifest(tmp_path)
    cfg = config_file(tmp_path, manifest_path=str(mpath), models_dir=str(tmp_path / "models"))
    assert run_cli(tmp_path, ["--config", cfg, "--run"]) == 4
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert REPO in err and REV in err and "dit/model.bin" in err


def test_run_without_manifest_exits_4(tmp_path, capsys):
    assert run_cli(tmp_path, ["--config", config_file(tmp_path), "--run"]) == 4
    err = capsys.readouterr().err
    assert "Traceback" not in err and "manifest" in err


def test_reserved_exit_codes_are_wired(tmp_path, capsys, monkeypatch):
    """3 (memory refusal) and 5 (hardware absent) are reserved: the mapping
    must exist and work before the real guard/hardware probe land."""
    mpath = write_manifest(tmp_path)
    (tmp_path / "models" / "dit" / "model.bin").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "dit" / "model.bin").write_bytes(b"payload")
    cfg = config_file(tmp_path, manifest_path=str(mpath), models_dir=str(tmp_path / "models"))
    monkeypatch.setattr(cli, "_run_generate",
                        lambda *a, **k: (_ for _ in ()).throw(MemoryGuardError("would not fit")))
    assert run_cli(tmp_path, ["--config", cfg, "--run"]) == 3
    assert "Traceback" not in capsys.readouterr().err
    monkeypatch.setattr(cli, "_run_generate",
                        lambda *a, **k: (_ for _ in ()).throw(HardwareAbsentError("no V100")))
    assert run_cli(tmp_path, ["--config", cfg, "--run"]) == 5


def test_unexpected_error_exits_1_without_traceback(tmp_path, capsys, monkeypatch):
    mpath = write_manifest(tmp_path)
    (tmp_path / "models" / "dit" / "model.bin").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "dit" / "model.bin").write_bytes(b"payload")
    cfg = config_file(tmp_path, manifest_path=str(mpath), models_dir=str(tmp_path / "models"))
    monkeypatch.setattr(cli, "_run_generate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom")))
    assert run_cli(tmp_path, ["--config", cfg, "--run"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err and "kaboom" in err


def test_success_path_returns_zero(tmp_path, capsys, monkeypatch):
    calls: list[str] = []
    mpath = write_manifest(tmp_path)
    (tmp_path / "models" / "dit" / "model.bin").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "dit" / "model.bin").write_bytes(b"payload")
    monkeypatch.setattr(cli, "_make_adapter", lambda: FakeAdapter(calls))
    cfg = config_file(tmp_path, output_dir=str(tmp_path / "output"),
                      manifest_path=str(mpath), models_dir=str(tmp_path / "models"))
    assert run_cli(tmp_path, ["--config", cfg, "--run"]) == 0
    assert calls == ["load", "encode", "denoise", "decode", "save"]


def test_cli_module_entry_point_declared():
    import tomllib
    with open(ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["scripts"]["qwen-engine"] == "qwen_engine.cli:main"
