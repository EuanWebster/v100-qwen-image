"""Task 1.3 acceptance: adapter protocol + core plan/generate boundary, CLI
contract (import/--help/--dry-run without heavy modules; exit-code contract
0/2/3(reserved)/4/5(reserved)/1; no tracebacks on expected errors)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_engine import cli, manifest as mf
from qwen_engine.adapters.base import GenerationAdapter
from qwen_engine.config import GenerationConfig
from qwen_engine.core import (
    OutputError,
    GenerationResult,
    generate,
    plan,
)
from qwen_engine.errors import EngineError, HardwareAbsentError, MemoryGuardError

REPO = "Qwen/Qwen-Image-2.1"
REV = "0" * 40
ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "diffusers", "transformers")


# --- helpers ---------------------------------------------------------------


class FakeAdapter:
    """In-process stand-in for the eventual Qwen21DiffusersAdapter.

    ``save`` writes a **real** regular file under ``config.output_dir`` so the
    core's post-save verification proves something; ``saved_to`` overrides the
    reported path (to test missing/escaped output claims), and ``fail_save``
    models a save that blows up.
    """

    name = "fake"

    def __init__(self, calls=None, saved_to=None, fail_save=False):
        self.calls = calls if calls is not None else []
        self.assets: dict = {}
        self.saved_to = saved_to
        self.fail_save = fail_save

    def load(self, config, assets):
        self.calls.append("load")
        self.assets = dict(assets)

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
        if self.fail_save:
            raise RuntimeError("save exploded")
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        real = out_dir / "out.png"
        real.write_bytes(b"\x89PNG fake")
        return Path(self.saved_to) if self.saved_to is not None else real


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
    result = generate(base_config, FakeAdapter(calls), {})
    assert calls == ["load", "encode", "denoise", "decode", "save"]
    assert isinstance(result, GenerationResult)
    assert result.output_path.endswith("out.png")


def test_generate_save_succeeds_when_completion_sink_fails(tmp_path, base_config):
    """Review fix: a sink that dies on phase_end (and a human sink that
    always fails) must not turn a successful save into a failed run — the
    image is on disk and the CLI still exits 0."""
    import json as _json

    from qwen_engine.log import PhaseLogger

    class CompletionFailingSink:
        def __init__(self):
            self.lines: list[str] = []

        def __call__(self, line: str) -> None:
            if _json.loads(line)["event"] == "phase_end":
                raise OSError("log sink exploded at completion")
            self.lines.append(line)

    def angry_human(line: str) -> None:
        raise OSError("human sink exploded")

    logger = PhaseLogger(sink=CompletionFailingSink(), human_sink=angry_human)
    calls: list[str] = []
    code = cli._run_generate(base_config, logger, assets={},
                             adapter=FakeAdapter(calls))
    assert code == 0
    assert calls == ["load", "encode", "denoise", "decode", "save"]
    assert (tmp_path / "output" / "out.png").is_file()


def test_generate_reports_body_exception_despite_completion_sink_faults(base_config):
    """The guard is completion-only: a failing save still fails the run even
    when every completion sink also explodes."""
    from qwen_engine.log import PhaseLogger

    class AllFailSink:
        def __call__(self, line: str) -> None:
            raise OSError("sink exploded")

    logger = PhaseLogger(sink=AllFailSink())
    with pytest.raises(RuntimeError):
        generate(base_config, FakeAdapter(fail_save=True), {}, logger)


# --- review fix 11: the FINAL complete event after a verified save ----------
# Phase-end guards do not cover generate's trailing logger.event("complete"):
# root's /tmp/v100-complete-log-review.py showed a sink failing ONLY on
# complete saved out.png and then raised OSError (CLI reported failure).


def test_generate_succeeds_when_sink_fails_only_on_complete(tmp_path, base_config):
    """Direct generate, mirroring root's probe: the image is saved and the
    result is returned even though the sink explodes on the complete record;
    the machine records emitted before it are intact and the stack is clean."""
    from qwen_engine.log import PhaseLogger

    class CompleteFailingSink:
        def __init__(self):
            self.lines: list[str] = []

        def __call__(self, line: str) -> None:
            if json.loads(line)["event"] == "complete":
                raise OSError("completion sink failed")
            self.lines.append(line)

    sink = CompleteFailingSink()
    logger = PhaseLogger(sink=sink)
    calls: list[str] = []
    result = generate(base_config, FakeAdapter(calls), {}, logger)  # no raise
    assert calls == ["load", "encode", "denoise", "decode", "save"]
    assert Path(result.output_path).is_file()
    assert (tmp_path / "output" / "out.png").is_file()
    events = [json.loads(line)["event"] for line in sink.lines]
    assert "phase_end" in events and "complete" not in events  # failed record
    assert logger._stack == []


def test_cli_run_exits_zero_when_sink_fails_only_on_complete(
        tmp_path, capsys, monkeypatch):
    """End-to-end CLI successful save with a complete-only failing sink:
    exit 0, the file on disk, the result JSON printed, no traceback — a log
    fault after a successful save must never report a failed run."""
    from qwen_engine.log import PhaseLogger

    class CompleteFailingSink:
        def __call__(self, line: str) -> None:
            if json.loads(line)["event"] == "complete":
                raise OSError("completion sink failed")

    monkeypatch.setattr(cli, "_logger_for",
                        lambda config: PhaseLogger(sink=CompleteFailingSink()))
    calls: list[str] = []
    adapter = FakeAdapter(calls)
    mpath = write_manifest(tmp_path)
    (tmp_path / "models" / "dit" / "model.bin").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "dit" / "model.bin").write_bytes(b"payload")
    monkeypatch.setattr(cli, "_make_adapter", lambda **k: adapter)
    cfg = config_file(tmp_path, output_dir=str(tmp_path / "output"),
                      manifest_path=str(mpath), models_dir=str(tmp_path / "models"))
    assert run_cli(tmp_path, ["--config", cfg, "--run"]) == 0
    assert calls == ["load", "encode", "denoise", "decode", "save"]
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert Path(out["output_path"]).is_file()
    assert "Traceback" not in captured.err


def test_generate_passes_config_and_grouped_assets_into_load(tmp_path, base_config):
    adapter = FakeAdapter()
    assets = {"dit": (tmp_path / "dit-1.bin", tmp_path / "dit-2.bin"),
              "vae": (tmp_path / "vae.bin",)}
    generate(base_config, adapter, assets)
    assert adapter.assets == assets  # verbatim, role-grouped tuples


def test_generate_refuses_unconfigured_output_before_loading_adapter(base_config):
    calls: list[str] = []
    cfg = GenerationConfig(prompt="p", output_dir=None)
    with pytest.raises(Exception) as exc:  # ConfigError -> CLI exit 2
        generate(cfg, FakeAdapter(calls), {})
    assert calls == []  # nothing loaded, nothing touched
    from qwen_engine.config import ConfigError
    assert isinstance(exc.value, ConfigError)


def test_save_that_fails_cannot_claim_success(tmp_path, base_config):
    with pytest.raises(RuntimeError):
        generate(base_config, FakeAdapter(fail_save=True), {})


def test_missing_saved_file_cannot_claim_success(tmp_path, base_config):
    adapter = FakeAdapter(saved_to=tmp_path / "output" / "never_written.png")
    with pytest.raises(OutputError) as exc:
        generate(base_config, adapter, {})
    assert "no regular file" in str(exc.value)


def test_escaped_saved_file_cannot_claim_success(tmp_path, base_config):
    sneaky = tmp_path / "elsewhere.png"
    sneaky.write_bytes(b"data")  # real file, but outside output_dir
    adapter = FakeAdapter(saved_to=sneaky)
    with pytest.raises(OutputError) as exc:
        generate(base_config, adapter, {})
    assert "outside" in str(exc.value)


def test_saved_symlink_escaping_output_dir_is_rejected(tmp_path, base_config):
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    sneaky = tmp_path / "elsewhere.png"
    sneaky.write_bytes(b"data")
    link = out_dir / "out.png"
    link.symlink_to(sneaky)

    class LinkSaver(FakeAdapter):
        def save(self, image, config):
            self.calls.append("save")
            return link

    with pytest.raises(OutputError):
        generate(base_config, LinkSaver(), {})


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
    assert doc1["cfg_active"] is False  # default scale 1.0 -> CFG off


def test_plan_has_no_dead_phases_constant():
    import qwen_engine.core as core
    assert not hasattr(core, "PHASES")


def test_cpu_surface_imports_no_heavy_modules():
    """Boundary is behavioural, not file-existence: importing the whole CPU
    surface (config/manifest/log/core/CLI/adapter protocol) must never drag
    in torch/diffusers/transformers, whatever else exists on disk."""
    import importlib
    for mod in ("qwen_engine.config", "qwen_engine.manifest", "qwen_engine.log",
                "qwen_engine.core", "qwen_engine.cli", "qwen_engine.errors",
                "qwen_engine.adapters.base"):
        importlib.import_module(mod)
    assert not set(HEAVY) & set(sys.modules)


def test_make_adapter_names_a_missing_adapter_module(monkeypatch):
    """A ModuleNotFoundError whose .name is exactly the adapter module becomes
    a clear packaging EngineError (the module now ships with the package)."""
    import importlib

    def fake_import(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(EngineError) as exc:
        cli._make_adapter()
    assert "missing from this installation" in str(exc.value)


def test_make_adapter_internal_dependency_error_keeps_identity(monkeypatch):
    """A ModuleNotFoundError from *inside* a (future) adapter — e.g. torch
    missing — must propagate unchanged, never be laundered into 'M1
    boundary'."""
    import importlib

    def fake_import(name):
        if name == "qwen_engine.adapters.qwen21_diffusers":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        raise AssertionError("unexpected import")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError) as exc:
        cli._make_adapter()
    assert exc.value.name == "torch"


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
    assert doc["config"]["steps"] == 40  # official Qwen example default
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


def test_true_cfg_scale_flag_reaches_config(tmp_path, capsys):
    cfg = config_file(tmp_path, negative_prompt="blurry")
    assert run_cli(tmp_path, ["--dry-run", "--config", cfg,
                              "--true-cfg-scale", "3.5"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["config"]["true_cfg_scale"] == 3.5
    assert doc["cfg_active"] is True


def test_true_cfg_scale_below_one_exits_2(tmp_path, capsys):
    assert run_cli(tmp_path, ["--dry-run", "--true-cfg-scale", "0.5"]) == 2
    assert "true_cfg_scale" in capsys.readouterr().err


def test_run_and_dry_run_are_mutually_exclusive(tmp_path, capsys):
    assert run_cli(tmp_path, ["--dry-run", "--run"]) == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err and "Traceback" not in err


def test_calibration_record_requires_run(tmp_path, capsys):
    target = tmp_path / "calibration.json"
    assert run_cli(tmp_path, ["--calibration-record", target]) == 2
    err = capsys.readouterr().err
    assert "requires --run" in err and "Traceback" not in err
    assert not target.exists()
    # Dry-run cannot smuggle in a measurement request either.
    assert run_cli(tmp_path, ["--dry-run", "--calibration-record", target]) == 2
    assert not target.exists()


def test_invalid_utf8_config_file_exits_2(tmp_path, capsys):
    bad = tmp_path / "config.json"
    bad.write_bytes(b'{"prompt": "caf\xe9 invalid \xff"}')
    assert run_cli(tmp_path, ["--config", bad, "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "UTF-8" in err and "Traceback" not in err


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


def test_success_path_returns_zero_and_passes_verified_assets(tmp_path, capsys, monkeypatch):
    calls: list[str] = []
    adapter = FakeAdapter(calls)
    mpath = write_manifest(tmp_path)
    (tmp_path / "models" / "dit" / "model.bin").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "dit" / "model.bin").write_bytes(b"payload")
    monkeypatch.setattr(cli, "_make_adapter", lambda **k: adapter)
    cfg = config_file(tmp_path, output_dir=str(tmp_path / "output"),
                      manifest_path=str(mpath), models_dir=str(tmp_path / "models"))
    assert run_cli(tmp_path, ["--config", cfg, "--run"]) == 0
    assert calls == ["load", "encode", "denoise", "decode", "save"]
    assert adapter.assets == {"dit": (tmp_path / "models" / "dit" / "model.bin",)}
    out = json.loads(capsys.readouterr().out)
    assert Path(out["output_path"]).is_file()


def test_actual_run_without_output_dir_exits_2_without_loading(tmp_path, capsys, monkeypatch):
    """Assets verified but output unconfigured: config error 2, the adapter
    never loads and nothing is written (no implicit output targets)."""
    calls: list[str] = []
    adapter = FakeAdapter(calls)
    mpath = write_manifest(tmp_path)
    (tmp_path / "models" / "dit" / "model.bin").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "dit" / "model.bin").write_bytes(b"payload")
    monkeypatch.setattr(cli, "_make_adapter", lambda **k: adapter)
    cfg = config_file(tmp_path, manifest_path=str(mpath),
                      models_dir=str(tmp_path / "models"))
    cfg = json.loads(Path(cfg).read_text())
    cfg.pop("output_dir", None)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    assert run_cli(tmp_path, ["--config", cfg_path, "--run"]) == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "output_dir" in err and "Traceback" not in err


def test_asset_errors_take_precedence_over_missing_output(tmp_path, capsys):
    """Missing assets keep exit 4 even when output_dir is *also* unset."""
    mpath = write_manifest(tmp_path)  # nothing materialized
    cfg = json.dumps({"prompt": "p", "manifest_path": str(mpath),
                      "models_dir": str(tmp_path / "models")})
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(cfg)
    assert run_cli(tmp_path, ["--config", cfg_path, "--run"]) == 4
    err = capsys.readouterr().err
    assert "asset error" in err


def test_unwritable_log_target_keeps_intended_exit_code(tmp_path, capsys, monkeypatch):
    """A log sink that cannot write must never change the exit code or print
    a traceback — only a short secondary diagnostic."""
    log_dir = tmp_path / "logdir"
    log_dir.mkdir()
    (log_dir / "engine.jsonl").mkdir()  # blocks the sink's append target
    mpath = write_manifest(tmp_path)
    (tmp_path / "models" / "dit" / "model.bin").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "dit" / "model.bin").write_bytes(b"payload")
    monkeypatch.setattr(cli, "_run_generate",
                        lambda *a, **k: (_ for _ in ()).throw(EngineError("planned failure")))
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "prompt": "p", "log_dir": str(log_dir), "output_dir": str(tmp_path / "out"),
        "manifest_path": str(mpath), "models_dir": str(tmp_path / "models")}))
    assert run_cli(tmp_path, ["--config", cfg_path, "--run"]) == 1
    err = capsys.readouterr().err
    assert "planned failure" in err and "Traceback" not in err
    assert "log sink failed" in err


def test_cli_module_entry_point_declared():
    import tomllib
    with open(ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["scripts"]["qwen-engine"] == "qwen_engine.cli:main"


# --- second senior-review pass: ordering + overflow at the CLI ---------------


def test_verified_assets_no_output_dir_never_creates_adapter_exits_2(
        tmp_path, capsys, monkeypatch):
    """THE real CLI ordering path.

    The loader adapter now exists; an actual run whose assets verify but whose
    output is unconfigured must still be a CONFIG error (exit 2) and must not
    create the adapter: the config check runs after asset verification (missing
    assets keep exit 4) and BEFORE any adapter creation/load."""
    import importlib.util

    assert importlib.util.find_spec(
        "qwen_engine.adapters.qwen21_diffusers") is not None, \
        "the packaged loader module must be importable"

    def exploding_make_adapter():
        raise AssertionError("adapter was created before the output_dir check")

    monkeypatch.setattr(cli, "_make_adapter", exploding_make_adapter)

    mpath = write_manifest(tmp_path)
    (tmp_path / "models" / "dit").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "dit" / "model.bin").write_bytes(b"payload")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "prompt": "a red cube", "manifest_path": str(mpath),
        "models_dir": str(tmp_path / "models"),
    }))  # deliberately no output_dir
    assert run_cli(tmp_path, ["--config", cfg_path, "--run"]) == 2
    err = capsys.readouterr().err
    assert "output_dir" in err and "Traceback" not in err
    assert "adapter" not in err.lower(), \
        "the config error must not be masked by the M4 adapter boundary"
    assert not (tmp_path / "output").exists()


def test_config_file_true_cfg_scale_overflow_exits_2(tmp_path, capsys):
    """A 10**1000 int literal in the config file: OverflowError from
    math.isfinite would exit 1 with a generic unexpected-error line; the
    field must be refused as config error 2."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"true_cfg_scale": ' + "1" + "0" * 1000 + "}")
    assert run_cli(tmp_path, ["--config", cfg, "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "true_cfg_scale" in err and "Traceback" not in err


def test_cli_flag_true_cfg_scale_infinite_exits_2(tmp_path, capsys):
    """argparse float('1e1000') is +inf — also a field-level config error."""
    assert run_cli(tmp_path, ["--dry-run", "--true-cfg-scale", "1e1000"]) == 2
    err = capsys.readouterr().err
    assert "true_cfg_scale" in err and "Traceback" not in err


# --- json parser safety limits reach the exit-code contract -----------------


def test_cli_config_giant_integer_seed_is_exit_two(tmp_path, capsys):
    if not getattr(sys, "get_int_max_str_digits", lambda: 0)():
        pytest.skip("integer-string digit limit disabled")
    cfg = tmp_path / "config.json"
    cfg.write_text('{"prompt": "cube", "seed": ' + "9" * 5000 + "}")
    # ConfigError -> ExitCode.CONFIG, a typed boundary rejection, never the
    # unexpected-error exit 1 with a traceback.
    assert run_cli(tmp_path, ["--config", cfg, "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "config error:" in err and "Traceback" not in err


@pytest.mark.parametrize("kind", ["fifo", "directory", "oversized"])
def test_cli_config_input_is_bounded_regular_and_never_blocks(tmp_path, capsys, kind):
    path = tmp_path / "config.json"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b" " * (cli.MAX_CONFIG_BYTES + 1))
    assert run_cli(tmp_path, ["--config", path, "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "bounded regular" in err and "Traceback" not in err


@pytest.mark.parametrize("raw", ['{"prompt":"first","prompt":"second"}',
                                 '{"steps":0,"steps":40}',
                                 '{"true_cfg_scale":NaN}', '{"true_cfg_scale":1e999}'])
def test_cli_duplicate_or_nonfinite_config_is_exit_two_even_with_override(tmp_path, capsys, raw):
    path = tmp_path / "config.json"
    path.write_text(raw)
    assert run_cli(tmp_path, ["--config", path, "--dry-run", "--prompt", "override", "--steps", "40"]) == 2
    err = capsys.readouterr().err
    assert "not valid JSON" in err and "Traceback" not in err


def test_cli_valid_input_symlink_and_override_semantics_preserved(tmp_path, capsys, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text('{"prompt":"file prompt","steps":20}')
    link = tmp_path / "link.json"
    link.symlink_to(path)
    monkeypatch.setattr(cli, "MAX_CONFIG_BYTES", path.stat().st_size)
    assert run_cli(tmp_path, ["--config", link, "--dry-run", "--prompt", "override", "--steps", "40"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["config"]["prompt"] == "override" and doc["config"]["steps"] == 40
    monkeypatch.setattr(cli, "MAX_CONFIG_BYTES", path.stat().st_size - 1)
    assert run_cli(tmp_path, ["--config", link, "--dry-run"]) == 2


def test_cli_config_replacement_during_read_is_typed_and_no_adapter(tmp_path, capsys, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text('{"prompt":"p"}')
    original = os.read
    changed = False
    def read(fd, size):
        nonlocal changed
        raw = original(fd, size)
        if raw and not changed:
            changed = True
            other = tmp_path / "other.json"
            other.write_bytes(path.read_bytes())
            other.replace(path)
        return raw
    monkeypatch.setattr(cli, "_make_adapter", lambda: pytest.fail("adapter after input refusal"))
    monkeypatch.setattr(os, "read", read)
    assert run_cli(tmp_path, ["--config", path, "--run"]) == 2
    assert "changed during read" in capsys.readouterr().err


# --- recipe flag + turbo asset resolution (docs/turbo-plan.md §4.3/§5.2) --------


def test_recipe_flag_echoes_into_the_plan(tmp_path, capsys):
    assert run_cli(tmp_path, ["--dry-run", "--recipe", "turbo", "--steps", "8"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["config"]["recipe"] == "turbo" and doc["config"]["steps"] == 8


def test_recipe_defaults_to_base_in_the_plan(tmp_path, capsys):
    assert run_cli(tmp_path, ["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["recipe"] == "base"


def test_cli_refuses_turbo_step_mismatch_before_any_asset_work(tmp_path, capsys):
    assert run_cli(tmp_path, ["--dry-run", "--recipe", "turbo", "--steps", "40"]) == 2
    assert "eight-step" in capsys.readouterr().err


def _asset_tree(root: Path, roles, repo: str,
                manifest: Path | None = None) -> Path:
    """A minimal resolvable manifest + one small file per role."""
    import hashlib

    artifacts = []
    for role in roles:
        content = f"{repo}:{role}".encode()
        path = root / role / "model.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts.append({"role": role, "repo": repo, "revision": REV,
                          "filename": f"{role}/model.bin",
                          "sha256": hashlib.sha256(content).hexdigest(),
                          "bytes": len(content)})
    manifest = manifest if manifest is not None else root / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema_version": mf.MANIFEST_SCHEMA_VERSION,
                                    "artifacts": artifacts}))
    return manifest


def test_verify_assets_merges_turbo_roles_without_touching_base_encoder(tmp_path):
    base_manifest = _asset_tree(tmp_path / "models",
                                ["text_encoder", "dit", "vae", "scheduler", "processor"], REPO)
    _asset_tree(tmp_path / "models" / "turbo" / "assets",
                ["dit", "scheduler", "pipeline"], "Qwen/Qwen-Image-2.1-Turbo",
                manifest=tmp_path / "models" / "turbo" / "manifest.json")
    turbo = GenerationConfig(manifest_path=str(base_manifest),
                             models_dir=str(tmp_path / "models"),
                             recipe="turbo", steps=8)
    report = cli._verify_assets(turbo)
    assert report.resolved["dit"] == (
        tmp_path / "models" / "turbo" / "assets" / "dit" / "model.bin",)
    assert report.resolved["text_encoder"] == (
        tmp_path / "models" / "text_encoder" / "model.bin",)
    # the same tree under the base recipe never sees the turbo directory
    base = cli._verify_assets(GenerationConfig(manifest_path=str(base_manifest),
                                               models_dir=str(tmp_path / "models")))
    assert base.resolved["dit"] == (tmp_path / "models" / "dit" / "model.bin",)


def test_verify_assets_refuses_turbo_without_its_manifest_or_artifacts(tmp_path):
    from qwen_engine.manifest import ManifestError

    base_manifest = _asset_tree(tmp_path / "models", ["dit"], REPO)
    _asset_tree(tmp_path / "models" / "turbo" / "assets", ["dit"],
                "Qwen/Qwen-Image-2.1-Turbo",
                manifest=tmp_path / "models" / "turbo" / "manifest.json")
    turbo = GenerationConfig(manifest_path=str(base_manifest),
                             models_dir=str(tmp_path / "models"),
                             recipe="turbo", steps=8)
    (tmp_path / "models" / "turbo" / "manifest.json").unlink()
    with pytest.raises(ManifestError, match="turbo manifest"):
        cli._verify_assets(turbo)


def test_verify_assets_refuses_hash_drifted_turbo_artifact(tmp_path):
    from qwen_engine.manifest import ManifestError

    base_manifest = _asset_tree(tmp_path / "models", ["dit"], REPO)
    _asset_tree(tmp_path / "models" / "turbo" / "assets", ["dit"],
                "Qwen/Qwen-Image-2.1-Turbo",
                manifest=tmp_path / "models" / "turbo" / "manifest.json")
    (tmp_path / "models" / "turbo" / "assets" / "dit" / "model.bin").write_bytes(b"swallowed")
    turbo = GenerationConfig(manifest_path=str(base_manifest),
                             models_dir=str(tmp_path / "models"),
                             recipe="turbo", steps=8)
    with pytest.raises(ManifestError, match="turbo artifact"):
        cli._verify_assets(turbo)
