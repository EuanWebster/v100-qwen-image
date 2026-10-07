"""M2.1 acceptance: the **committed** ``models/manifest.json`` (schema v1) and the
no-download CLI behaviour against it — offline, stdlib-only, no weights.

Deliberately self-contained: nothing here reads the /tmp root-reference
files (they are not part of the repo). The full pinned identity is anchored
by an **independent canonical-identity digest constant** — sha256 over

    json.dumps({"repo": ..., "revision": ...,
                "artifacts": {filename: [bytes, sha256], ...}},
               sort_keys=True, separators=(",", ":"))

which pins repo, revision, every filename, and every (size, SHA256) pair in
one value; the explicit filename/role fixtures below pin the set and the
role mapping that the digest payload deliberately excludes. All writes stay
inside pytest ``tmp_path`` (R7). No network, no heavy imports, no GPU, no
model files.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from qwen_engine import cli
from qwen_engine.manifest import (
    AmbiguousRoleError,
    load_manifest,
    resolve_model_path,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "manifest.json"

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"

#: sha256 of the canonical identity payload (see module docstring). Any drift
#: in repo, revision, a filename, a byte size or a sha256 changes this value.
IDENTITY_DIGEST = "14147c1698f2755e2637e56e10fc4bf3543d1bfebbdc29ec51457678e03e6ba3"

#: Committed fixture: the exact 24 runtime filenames (upstream layout).
EXPECTED_FILES = frozenset({
    "model_index.json",
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
})

#: Committed fixture: expected role counts (24 total).
EXPECTED_ROLE_COUNTS = {"pipeline": 1, "processor": 9, "scheduler": 1,
                        "text_encoder": 7, "dit": 4, "vae": 2}

#: Expected upstream metadata byte totals — identities, never local measurements.
EXPECTED_TOTAL_BYTES = 33_131_601_593          # all 24 files
EXPECTED_WEIGHT_BYTES = 33_115_613_408         # the 7 .safetensors weight files

HEAVY_MODULES = ("torch", "diffusers", "transformers")


def _load_committed():
    assert MANIFEST_PATH.is_file(), "models/manifest.json must be committed"
    return load_manifest(MANIFEST_PATH)


def _canonical_identity(manifest) -> str:
    repos = {a.repo for a in manifest.artifacts}
    revs = {a.revision for a in manifest.artifacts}
    assert repos == {REPO} and revs == {REVISION}
    payload = {
        "repo": REPO,
        "revision": REVISION,
        "artifacts": {a.filename: [a.bytes, a.sha256] for a in manifest.artifacts},
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _run_cli(args, cwd=None):
    """Fresh-interpreter CLI run (guard env, if any, rides os.environ)."""
    return subprocess.run(
        [sys.executable, "cli_gen.py", *map(str, args)],
        cwd=str(cwd or ROOT), env=os.environ.copy(),
        capture_output=True, text=True, timeout=120,
    )


# --- schema / identity / roles ------------------------------------------------


def test_committed_manifest_parses_as_schema_v1():
    m = _load_committed()
    assert m.schema_version == 1
    raw = json.loads(MANIFEST_PATH.read_text("utf-8"))
    assert set(raw) == {"schema_version", "artifacts"}
    for entry in raw["artifacts"]:
        assert set(entry) == {"role", "repo", "revision", "filename",
                              "sha256", "bytes"}


def test_full_pinned_identity_digest():
    """Independent digest: pins repo+revision+filename+size+sha256 for all
    24 files in one value, computed here from the COMMITTED manifest — not
    from any /tmp reference."""
    assert _canonical_identity(_load_committed()) == IDENTITY_DIGEST


def test_exact_filename_set_layout_and_order():
    m = _load_committed()
    names = [a.filename for a in m.artifacts]
    assert len(names) == 24 and set(names) == EXPECTED_FILES
    assert names == sorted(names), "artifacts must be in deterministic filename order"
    # Upstream layout preserved: every file is model_index.json or under its
    # upstream component directory.
    for name in names:
        assert name == "model_index.json" or "/" in name


def test_role_mapping_exact():
    m = _load_committed()
    def expected_role(fn: str) -> str:
        if fn == "model_index.json":
            return "pipeline"
        return {"processor": "processor", "scheduler": "scheduler",
                "text_encoder": "text_encoder", "transformer": "dit",
                "vae": "vae"}[fn.split("/", 1)[0]]
    assert {a.filename: a.role for a in m.artifacts} == {
        fn: expected_role(fn) for fn in EXPECTED_FILES}
    assert dict(Counter(a.role for a in m.artifacts)) == EXPECTED_ROLE_COUNTS


def test_expected_byte_totals_are_metadata_not_measurements():
    m = _load_committed()
    assert sum(a.bytes for a in m.artifacts) == EXPECTED_TOTAL_BYTES
    assert sum(a.bytes for a in m.artifacts
               if a.filename.endswith(".safetensors")) == EXPECTED_WEIGHT_BYTES
    # No asset exists locally: nothing here may be reported as a measured total.
    assert not (ROOT / "models" / "assets").exists() or \
        not any((ROOT / "models" / "assets").rglob("*"))


def test_excludes_non_runtime_extras():
    names = {a.filename for a in _load_committed().artifacts}
    forbidden = ("README.md", "LICENSE", ".gitattributes", "example.png",
                 ".model-card", "model-card")
    assert not [n for n in names
                if any(f.lower() in n.lower() for f in forbidden)]
    assert names == EXPECTED_FILES  # no extras at all


def test_multi_shard_roles_retained():
    m = _load_committed()
    by_role = {}
    for a in m.artifacts:
        by_role.setdefault(a.role, []).append(a.filename)
    te = sorted(n for n in by_role["text_encoder"] if n.endswith(".safetensors"))
    dit = sorted(n for n in by_role["dit"] if n.endswith(".safetensors"))
    assert len(te) == 4 and all("-of-00004.safetensors" in n for n in te)
    assert len(dit) == 2 and all("-of-00002.safetensors" in n for n in dit)
    # Structural ambiguity is refused BEFORE any disk access (never one shard
    # of a sharded role handed back silently):
    with pytest.raises(AmbiguousRoleError):
        resolve_model_path(m, "/nonexistent-root-m2", "dit")


# --- CLI: deterministic all-missing dry-run + exit-4 run -----------------------


def test_dry_run_all_missing_is_deterministic(tmp_path, capsys):
    """Committed manifest vs a fresh, never-created storage root: exit 0,
    24x missing, byte-identical JSON across runs (deterministic, timestamp-free)."""
    missing_root = tmp_path / "fresh-root-does-not-exist"
    outputs = []
    for _ in range(2):
        rc = cli.main(["--dry-run", "--manifest", str(MANIFEST_PATH),
                       "--models-dir", str(missing_root)])
        assert rc == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1]
    doc = json.loads(outputs[0])
    status = doc["manifest"]
    assert status["configured"] is True and status["schema_version"] == 1
    assert status["status"] == "incomplete"
    arts = status["artifacts"]
    assert len(arts) == 24
    assert all(a["status"] == "missing" for a in arts)
    assert {a["filename"] for a in arts} == EXPECTED_FILES
    assert all(a["repo"] == REPO and a["revision"] == REVISION for a in arts)
    assert not set(HEAVY_MODULES) & set(sys.modules)


def test_run_all_missing_exits_4_naming_every_file(tmp_path, monkeypatch, capsys):
    """Fresh-interpreter actual run against a fresh missing root: exit 4, all
    24 files + repo + revision named, no traceback, no adapter/ML imports, no
    network. (Sockets are additionally poisoned in-process for the in-process
    variant; the R7 guard denies them for the subprocess.)"""
    proc = _run_cli(["--run", "--manifest", MANIFEST_PATH,
                     "--models-dir", tmp_path / "fresh-root"])
    assert proc.returncode == 4, proc.stderr
    combined = proc.stdout + proc.stderr
    assert "Traceback" not in combined
    for name in sorted(EXPECTED_FILES):
        assert name in proc.stderr, name
    assert REPO in proc.stderr and REVISION in proc.stderr
    assert proc.stdout.strip() == ""
    # The absent-adapter boundary was never reached:
    assert "M4+" not in combined


def test_run_all_missing_in_process_no_heavy_imports_no_network(tmp_path, monkeypatch, capsys):
    block_network(monkeypatch)
    rc = cli.main(["--run", "--manifest", str(MANIFEST_PATH),
                   "--models-dir", str(tmp_path / "fresh-root")])
    assert rc == 4
    err = capsys.readouterr().err
    assert "Traceback" not in err and err.count("asset error:") >= 24
    assert "qwen_engine.adapters.qwen21_diffusers" not in sys.modules
    assert not set(HEAVY_MODULES) & set(sys.modules)


def test_run_missing_assets_keep_exit4_even_without_output_dir(tmp_path, capsys):
    """Asset verification precedes the output-dir config check: the operator
    learns what to fetch, not a config nag."""
    proc = _run_cli(["--run", "--manifest", MANIFEST_PATH,
                     "--models-dir", tmp_path / "fresh-root"])
    assert proc.returncode == 4
    assert "output" not in (proc.stdout + proc.stderr).lower()


def block_network(monkeypatch):
    import urllib.request  # import *before* patching so ssl's import isn't broken

    def explode(*a, **k):
        raise AssertionError("manifest CLI attempted network access")
    monkeypatch.setattr(socket, "socket", explode)
    monkeypatch.setattr(socket, "create_connection", explode)
    monkeypatch.setattr(urllib.request, "urlopen", explode)
