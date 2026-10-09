"""M2.2b stage 1 item 3 — host-memory preflight tests (no ML, no assets).

Injected proc/cgroup fixtures only (tiny files under tmp_path): the real
procfs paths and every model/asset file stay untouched. The headline case
is the measured sandbox scenario: a 17,534,339,488 B checkpoint against a
17,179,869,184 B (16 GiB) cgroup cap must be refused before anything else
happens.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_engine.mem_preflight import (
    CGROUP_FINITE,
    CGROUP_UNKNOWN,
    CGROUP_UNLIMITED,
    MEM_PREFLIGHT_SCHEMA,
    MemoryPreflightError,
    MemoryPreflightReport,
    MemorySnapshot,
    encoder_memory_preflight,
    encoder_memory_required,
    read_memory_snapshot,
    require_encoder_memory_fit,
)

HEAVY_MODULES = ("torch", "diffusers", "transformers")

# Measured sandbox facts (docs/handoffs/2026-10-09-resume.md).
ENCODER_CHECKPOINT_BYTES = 17_534_339_488
CGROUP_CAP_16GIB = 17_179_869_184
GIB = 1024 ** 3


def make_env(tmp_path, *, host_kb=40 * 1024 * 1024, cgroup_path="/",
             mem_max=None, mem_current=None, raw_meminfo=None,
             raw_cgroup=None):
    """Tiny proc/cgroup fixture tree; returns the kwargs for the preflight."""
    env = tmp_path / "proc"
    env.mkdir(exist_ok=True)
    meminfo = env / "meminfo"
    if raw_meminfo is not None:
        meminfo.write_text(raw_meminfo, encoding="utf-8")
    else:
        meminfo.write_text("MemTotal:  a kB\nMemAvailable:   "
                           f"{host_kb} kB\n", encoding="utf-8")
    cgroup = env / "cgroup"
    cgroup.write_text(raw_cgroup if raw_cgroup is not None
                      else f"0::{cgroup_path}\n", encoding="utf-8")
    fs = tmp_path / "sys"
    leaf = fs.joinpath(*[c for c in cgroup_path.split("/") if c])
    leaf.mkdir(parents=True, exist_ok=True)
    if mem_max is not None:
        (leaf / "memory.max").write_text(str(mem_max), encoding="utf-8")
    if mem_current is not None:
        (leaf / "memory.current").write_text(str(mem_current),
                                             encoding="utf-8")
    return {"meminfo_path": meminfo, "proc_cgroup_path": cgroup,
            "cgroup_fs_root": fs}


def preflight(env, checkpoint_bytes=ENCODER_CHECKPOINT_BYTES,
              checkpoint_dtype="bf16", target_dtype="fp32",
              headroom_bytes=2 * GIB):
    return encoder_memory_preflight(checkpoint_bytes,
                                    checkpoint_dtype=checkpoint_dtype,
                                    target_dtype=target_dtype,
                                    headroom_bytes=headroom_bytes, **env)


# -- the measured scenario --------------------------------------------------


def test_real_sandbox_scenario_refuses_fp32_expansion(tmp_path):
    env = make_env(tmp_path, host_kb=24 * 1024 * 1024,
                   mem_max=CGROUP_CAP_16GIB, mem_current=1 * GIB)
    report = preflight(env)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_FINITE
    assert report.cgroup_limit_bytes == CGROUP_CAP_16GIB
    assert report.budget_bytes == CGROUP_CAP_16GIB - 1 * GIB
    # fp32 from a two-byte checkpoint dtype doubles the weight bytes.
    assert report.dtype_expanded_bytes == 2 * ENCODER_CHECKPOINT_BYTES
    assert report.required_bytes == 2 * ENCODER_CHECKPOINT_BYTES + 2 * GIB
    assert "exceeds the applicable budget" in " ".join(report.reasons)


def test_real_sandbox_scenario_refuses_even_without_expansion(tmp_path):
    """bf16 target (factor 1): the checkpoint alone still exceeds the cap."""
    env = make_env(tmp_path, host_kb=24 * 1024 * 1024,
                   mem_max=CGROUP_CAP_16GIB, mem_current=0)
    report = preflight(env, target_dtype="bf16", headroom_bytes=1024)
    assert report.decision == "refuse"
    assert report.required_bytes == ENCODER_CHECKPOINT_BYTES + 1024
    assert report.budget_bytes == CGROUP_CAP_16GIB


def test_require_raises_before_any_work(tmp_path):
    env = make_env(tmp_path, mem_max=CGROUP_CAP_16GIB, mem_current=0)
    with pytest.raises(MemoryPreflightError, match="refused"):
        require_encoder_memory_fit(ENCODER_CHECKPOINT_BYTES,
                                   checkpoint_dtype="bf16",
                                   target_dtype="fp32",
                                   headroom_bytes=2 * GIB, **env)


# -- tighter-budget selection ------------------------------------------------


def test_finite_cgroup_never_substituted_by_larger_host(tmp_path):
    env = make_env(tmp_path, host_kb=100 * 1024 * 1024,   # 100 GiB host
                   mem_max=4 * GIB, mem_current=0)
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="fp32",
                       target_dtype="fp32", headroom_bytes=2 * GIB)
    assert report.decision == "proceed"
    assert report.budget_bytes == 4 * GIB          # cgroup, not host
    assert "never substituted away" in " ".join(report.reasons)


def test_host_can_be_the_tighter_budget_under_a_finite_cap(tmp_path):
    env = make_env(tmp_path, host_kb=2 * 1024 * 1024,     # 2 GiB host
                   mem_max=16 * GIB, mem_current=0)
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="fp32",
                       target_dtype="fp32", headroom_bytes=2 * GIB)
    assert report.decision == "refuse"
    assert report.budget_bytes == 2 * GIB


def test_host_tighter_under_unlimited_cgroup(tmp_path):
    env = make_env(tmp_path, host_kb=2 * 1024 * 1024, mem_max="max")
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="fp32",
                       target_dtype="fp32", headroom_bytes=3 * GIB)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNLIMITED
    assert report.budget_bytes == 2 * GIB


def test_unlimited_cgroup_healthy_host_proceeds(tmp_path):
    env = make_env(tmp_path, host_kb=64 * 1024 * 1024, mem_max="max")
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "proceed"
    assert report.budget_bytes == 64 * GIB
    assert "conservative prerequisite" in " ".join(report.reasons)


def test_exact_budget_boundary(tmp_path):
    env = make_env(tmp_path, host_kb=2 * 1024 * 1024, mem_max="max")
    fits = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                     target_dtype="fp32", headroom_bytes=0)
    assert fits.decision == "proceed"           # required == budget
    off = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                    target_dtype="fp32", headroom_bytes=1)
    assert off.decision == "refuse"             # one byte over


def test_current_at_or_over_cap_leaves_zero_room(tmp_path):
    env = make_env(tmp_path, mem_max=4 * GIB, mem_current=4 * GIB)
    report = preflight(env, checkpoint_bytes=1, checkpoint_dtype="fp32",
                       target_dtype="fp32", headroom_bytes=0)
    assert report.decision == "refuse"
    assert report.budget_bytes == 0


# -- unknown state never asserts safety --------------------------------------


def test_no_cgroup_v2_line_is_unknown_not_unlimited(tmp_path):
    env = make_env(tmp_path, host_kb=64 * 1024 * 1024,
                   raw_cgroup="1:name=systemd:/\n")
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN
    assert "cannot be ruled out" in " ".join(report.reasons)


def test_unreadable_cgroup_file_is_unknown(tmp_path):
    env = make_env(tmp_path)
    Path(env["proc_cgroup_path"]).unlink()
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN


def test_missing_memory_max_is_unknown(tmp_path):
    env = make_env(tmp_path, host_kb=64 * 1024 * 1024)  # no memory.max
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "refuse"


def test_nested_cgroup_walk_prefers_deepest_and_falls_back(tmp_path):
    env = make_env(tmp_path, cgroup_path="/kubepods/pod123")  # leaf empty
    fs = Path(env["cgroup_fs_root"])
    (fs / "memory.max").write_text("max", encoding="utf-8")
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.cgroup_state == CGROUP_UNLIMITED  # walked up to the root
    leaf = fs / "kubepods" / "pod123"
    (leaf / "memory.max").write_text(str(8 * GIB), encoding="utf-8")
    (leaf / "memory.current").write_text("0", encoding="utf-8")
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "proceed"
    assert report.cgroup_state == CGROUP_FINITE
    assert report.cgroup_limit_bytes == 8 * GIB  # deepest wins


@pytest.mark.parametrize("bad", ["banana", "-5", "", "MAX", "1.5"])
def test_invalid_memory_max_is_unknown(tmp_path, bad):
    env = make_env(tmp_path, host_kb=64 * 1024 * 1024, mem_max=bad,
                   mem_current=0)
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN


@pytest.mark.parametrize("bad", ["banana", "-1"])
def test_invalid_memory_current_against_finite_cap_is_unknown(tmp_path, bad):
    env = make_env(tmp_path, mem_max=8 * GIB, mem_current=bad)
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "refuse"


def test_finite_cap_without_current_is_unknown(tmp_path):
    env = make_env(tmp_path, host_kb=64 * 1024 * 1024, mem_max=8 * GIB)
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN


@pytest.mark.parametrize("raw", ["MemTotal: 1 kB\n", "MemAvailable: x kB\n"])
def test_unknown_host_with_unlimited_cgroup_refuses(tmp_path, raw):
    env = make_env(tmp_path, raw_meminfo=raw, mem_max="max")
    report = preflight(env, checkpoint_bytes=1, checkpoint_dtype="fp32",
                       target_dtype="fp32", headroom_bytes=0)
    assert report.decision == "refuse"


def test_unknown_host_under_finite_cap_still_bounded(tmp_path):
    env = make_env(tmp_path, raw_meminfo="MemTotal: 1 kB\n",
                   mem_max=8 * GIB, mem_current=0)
    fits = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="fp32",
                     target_dtype="fp32", headroom_bytes=GIB)
    assert fits.decision == "proceed"            # cap alone bounds it
    assert fits.budget_bytes == 8 * GIB
    assert "finite cgroup budget alone bounds the process" \
        in " ".join(fits.reasons)


# -- required-bytes arithmetic + input validation -----------------------------


def test_dtype_expansion_arithmetic():
    # fp32 target from a two-byte checkpoint dtype: exactly doubled.
    weights, required = encoder_memory_required(
        checkpoint_bytes=1001, checkpoint_dtype="bf16", target_dtype="fp32",
        headroom_bytes=7)
    assert weights == 2002 and required == 2009       # ceil, then headroom
    # apparent shrink clamps to the checkpoint total (disk bytes are real).
    weights, _ = encoder_memory_required(
        checkpoint_bytes=4096, checkpoint_dtype="fp32", target_dtype="fp16",
        headroom_bytes=0)
    assert weights == 4096
    # fp16 checkpoint to fp32 also doubles; same-width is exact.
    assert encoder_memory_required(
        checkpoint_bytes=10, checkpoint_dtype="fp16", target_dtype="fp32",
        headroom_bytes=0)[0] == 20
    assert encoder_memory_required(
        checkpoint_bytes=10, checkpoint_dtype="bf16", target_dtype="bfloat16",
        headroom_bytes=0)[0] == 10


@pytest.mark.parametrize("kwargs", [
    {"checkpoint_bytes": 0, "checkpoint_dtype": "fp32",
     "target_dtype": "fp32", "headroom_bytes": 0},
    {"checkpoint_bytes": -5, "checkpoint_dtype": "fp32",
     "target_dtype": "fp32", "headroom_bytes": 0},
    {"checkpoint_bytes": True, "checkpoint_dtype": "fp32",
     "target_dtype": "fp32", "headroom_bytes": 0},
    {"checkpoint_bytes": 1.5, "checkpoint_dtype": "fp32",
     "target_dtype": "fp32", "headroom_bytes": 0},
    {"checkpoint_bytes": 4096, "checkpoint_dtype": "int8",
     "target_dtype": "fp32", "headroom_bytes": 0},
    {"checkpoint_bytes": 4096, "checkpoint_dtype": "fp32",
     "target_dtype": "unknown-dtype", "headroom_bytes": 0},
    {"checkpoint_bytes": 4096, "checkpoint_dtype": "fp32",
     "target_dtype": "fp32", "headroom_bytes": -1},
    {"checkpoint_bytes": 4096, "checkpoint_dtype": "fp32",
     "target_dtype": "fp32", "headroom_bytes": True},
    {"checkpoint_bytes": 4096, "checkpoint_dtype": "fp32",
     "target_dtype": "fp32", "headroom_bytes": 1.0},
])
def test_invalid_required_inputs_are_typed(kwargs):
    with pytest.raises(MemoryPreflightError):
        encoder_memory_required(**kwargs)


# -- snapshot injection + report shape ----------------------------------------


def test_snapshot_injection_decides_without_touching_paths(tmp_path):
    bogus = {"meminfo_path": tmp_path / "nope",
             "proc_cgroup_path": tmp_path / "nope2",
             "cgroup_fs_root": tmp_path / "nope3"}
    snap = MemorySnapshot(host_available_bytes=10 * GIB,
                          cgroup_state=CGROUP_FINITE,
                          cgroup_limit_bytes=CGROUP_CAP_16GIB,
                          cgroup_current_bytes=0)
    report = encoder_memory_preflight(ENCODER_CHECKPOINT_BYTES,
                                      checkpoint_dtype="bf16",
                                      target_dtype="fp32",
                                      headroom_bytes=GIB,
                                      snapshot=snap, **bogus)
    assert report.decision == "refuse"
    good = MemorySnapshot(host_available_bytes=64 * GIB,
                          cgroup_state=CGROUP_UNLIMITED)
    assert encoder_memory_preflight(GIB, checkpoint_dtype="fp32",
                                    target_dtype="fp32", headroom_bytes=GIB,
                                    snapshot=good, **bogus).decision \
        == "proceed"


def test_unknown_snapshot_cannot_carry_counters():
    with pytest.raises(MemoryPreflightError):
        MemorySnapshot(host_available_bytes=1, cgroup_state=CGROUP_UNKNOWN,
                       cgroup_limit_bytes=1)
    with pytest.raises(MemoryPreflightError):
        MemorySnapshot(host_available_bytes=1, cgroup_state=CGROUP_FINITE)
    with pytest.raises(MemoryPreflightError):
        MemorySnapshot(host_available_bytes=1, cgroup_state="plenty")


def test_report_is_json_safe_and_tagged(tmp_path):
    env = make_env(tmp_path, mem_max=CGROUP_CAP_16GIB, mem_current=0)
    report = preflight(env)
    assert isinstance(report, MemoryPreflightReport)
    doc = json.loads(json.dumps(report.to_dict()))
    assert doc["schema"] == MEM_PREFLIGHT_SCHEMA
    assert doc["decision"] == "refuse"
    assert doc["conservative_precondition"] is True
    assert doc["checkpoint_bytes"] == ENCODER_CHECKPOINT_BYTES
    assert doc["cgroup_state"] == "finite"


def test_read_memory_snapshot_reads_the_fixture_tree(tmp_path):
    env = make_env(tmp_path, host_kb=8 * 1024 * 1024, mem_max=16 * GIB,
                   mem_current=GIB)
    snap = read_memory_snapshot(**env)
    assert snap.host_available_bytes == 8 * GIB
    assert snap.cgroup_state == CGROUP_FINITE
    assert snap.cgroup_limit_bytes == 16 * GIB
    assert snap.cgroup_current_bytes == GIB


def test_module_docstring_states_the_honest_limits():
    import qwen_engine.mem_preflight as mod
    doc = mod.__doc__
    assert "conservative prerequisite" in doc
    assert "not measured model-fit evidence" in doc
    assert "never substitute" in doc.lower() or "never substituted away" in doc


def test_import_pulls_in_no_heavy_modules():
    code = (
        "import sys, qwen_engine.mem_preflight as m;"
        "heavy = [n for n in ('torch','diffusers','transformers') if n in sys.modules];"
        "assert not heavy, heavy;"
        "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
