"""M2.2b stage 1 item 3 — host-memory preflight tests (no ML, no assets).

Injected proc/cgroup fixtures only (tiny files under tmp_path): the real
procfs paths and every model/asset file stay untouched. Two evidentiary
tracks, matching the module's scope gate:

* **Read-path tests fail closed**: real autodiscovery cannot prove the
  visible cgroup root is the true hierarchy root (a namespace '/' can hide
  a tighter ancestor; a missing memory.max at ANY level proves nothing
  about the controller), so every real read refuses with an unknown
  cgroup state — while still carrying truthful per-level notes and the
  advisory visible availability — an UPPER bound a hidden ancestor can
  only tighten (e.g. the measured 17,534,339,488 B checkpoint vs the
  17,179,869,184 B (16 GiB) visible cap).
* **Pass/arithmetic tests use explicitly trusted injected snapshots** —
  documented caller evidence for pure budget decisions.
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
    _tightest_visible_level,
    encoder_memory_preflight,
    encoder_memory_required,
    read_host_available,
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


def put_level(env, path, mem_max=None, mem_current=None):
    """Add memory.max/memory.current to the fixture directory for cgroup
    path `path` (relative to the supplied cgroup fs root)."""
    fs = Path(env["cgroup_fs_root"])
    directory = fs.joinpath(*[c for c in path.split("/") if c])
    directory.mkdir(parents=True, exist_ok=True)
    if mem_max is not None:
        (directory / "memory.max").write_text(str(mem_max), encoding="utf-8")
    if mem_current is not None:
        (directory / "memory.current").write_text(str(mem_current),
                                                  encoding="utf-8")
    return directory


def preflight(env, checkpoint_bytes=ENCODER_CHECKPOINT_BYTES,
              checkpoint_dtype="bf16", target_dtype="fp32",
              headroom_bytes=2 * GIB):
    return encoder_memory_preflight(checkpoint_bytes,
                                    checkpoint_dtype=checkpoint_dtype,
                                    target_dtype=target_dtype,
                                    headroom_bytes=headroom_bytes, **env)


# -- the measured scenario --------------------------------------------------


def test_real_sandbox_scenario_refuses_on_visible_availability(tmp_path):
    """Real read path: the visible cap is truthful refusal evidence, but
    the visible root's completeness is unproven, so the decision is a
    fail-closed refusal — never a mislabeled proven finite budget."""
    env = make_env(tmp_path, host_kb=24 * 1024 * 1024,
                   mem_max=CGROUP_CAP_16GIB, mem_current=1 * GIB)
    report = preflight(env)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN
    assert report.budget_bytes is None
    # fp32 from a two-byte checkpoint dtype doubles the weight bytes.
    assert report.dtype_expanded_bytes == 2 * ENCODER_CHECKPOINT_BYTES
    assert report.required_bytes == 2 * ENCODER_CHECKPOINT_BYTES + 2 * GIB
    reasons = " ".join(report.reasons)
    # required (allocation lower bound) > visible availability (upper
    # bound) survives the gate as guaranteed-refusal evidence.
    assert "exceeds the tightest visible availability" in reasons
    assert str(CGROUP_CAP_16GIB - 1 * GIB) in reasons


def test_trusted_snapshot_fp32_expansion_refuses_finite_cap():
    """Same arithmetic against TRUSTED injected complete evidence: the
    finite cap is then the authoritative tighter budget."""
    snap = MemorySnapshot(host_available_bytes=24 * GIB,
                          cgroup_state=CGROUP_FINITE,
                          cgroup_limit_bytes=CGROUP_CAP_16GIB,
                          cgroup_current_bytes=1 * GIB)
    report = encoder_memory_preflight(ENCODER_CHECKPOINT_BYTES,
                                      checkpoint_dtype="bf16",
                                      target_dtype="fp32",
                                      headroom_bytes=2 * GIB, snapshot=snap)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_FINITE
    assert report.cgroup_limit_bytes == CGROUP_CAP_16GIB
    assert report.budget_bytes == CGROUP_CAP_16GIB - 1 * GIB
    assert "exceeds the applicable budget" in " ".join(report.reasons)


def test_real_sandbox_no_expansion_still_refuses(tmp_path):
    """bf16 target (factor 1): the checkpoint alone exceeds the visible
    cap — refused either way the scope question resolves."""
    env = make_env(tmp_path, host_kb=24 * 1024 * 1024,
                   mem_max=CGROUP_CAP_16GIB, mem_current=0)
    report = preflight(env, target_dtype="bf16", headroom_bytes=1024)
    assert report.decision == "refuse"
    assert report.required_bytes == ENCODER_CHECKPOINT_BYTES + 1024
    assert ("exceeds the tightest visible availability"
            in " ".join(report.reasons))
    snap = MemorySnapshot(host_available_bytes=24 * GIB,
                          cgroup_state=CGROUP_FINITE,
                          cgroup_limit_bytes=CGROUP_CAP_16GIB,
                          cgroup_current_bytes=0)
    trusted = encoder_memory_preflight(ENCODER_CHECKPOINT_BYTES,
                                       checkpoint_dtype="bf16",
                                       target_dtype="bf16",
                                       headroom_bytes=1024, snapshot=snap)
    assert trusted.decision == "refuse"
    assert trusted.budget_bytes == CGROUP_CAP_16GIB


def test_real_sandbox_16gib_cap_as_finite_ancestor_of_unlimited_child(
        tmp_path):
    """Root reproduction: an unlimited child under the actual 16 GiB parent
    must REFUSE the 17,534,339,488 B encoder — a descendant cannot escape
    an ancestor cap (this previously returned proceed)."""
    env = make_env(tmp_path, host_kb=32 * 1024 * 1024, cgroup_path="/child")
    put_level(env, "/", mem_max=CGROUP_CAP_16GIB, mem_current=0)
    put_level(env, "/child", mem_max="max")
    report = preflight(env, target_dtype="bf16", headroom_bytes=0)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN     # scope gate also applies
    assert report.budget_bytes is None
    reasons = " ".join(report.reasons)
    assert "minimum limit-minus-current" in reasons   # ancestor walk truth
    assert str(Path(env["cgroup_fs_root"]) / "child") in reasons
    assert "completeness cannot be proven" in reasons
    assert "exceeds the tightest visible availability" in reasons


def test_require_raises_before_any_work(tmp_path):
    env = make_env(tmp_path, mem_max=CGROUP_CAP_16GIB, mem_current=0)
    with pytest.raises(MemoryPreflightError, match="refused"):
        require_encoder_memory_fit(ENCODER_CHECKPOINT_BYTES,
                                   checkpoint_dtype="bf16",
                                   target_dtype="fp32",
                                   headroom_bytes=2 * GIB, **env)


# -- whole-hierarchy budgets (root review fixes item 9) -----------------------


def test_looser_finite_child_tighter_parent_selection_is_advisory(tmp_path):
    """The walk selects the parent's tighter room truthfully (minimum
    limit-minus-current, source paths named) even though the default read
    refuses on unproven root scope."""
    env = make_env(tmp_path, cgroup_path="/parent/child")
    put_level(env, "/", mem_max="max")
    put_level(env, "/parent", mem_max=4 * GIB, mem_current=0)
    put_level(env, "/parent/child", mem_max=32 * GIB, mem_current=0)
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="fp32",
                       target_dtype="fp32", headroom_bytes=2 * GIB)
    assert report.decision == "refuse"        # unproven root scope
    assert report.cgroup_state == CGROUP_UNKNOWN
    reasons = " ".join(report.reasons)
    assert f"advisory visible availability {4 * GIB} B" in reasons
    assert "minimum limit-minus-current" in reasons
    assert str(Path(env["cgroup_fs_root"]) / "parent") in reasons
    assert str(Path(env["cgroup_fs_root"]) / "parent" / "child") in reasons


def test_tightest_visible_level_selects_minimum_available():
    """Pure selection demonstrates the minimum of EACH finite
    limit-minus-current (a parent's current includes sibling usage),
    independently of the read path's scope refusal."""
    levels = [(Path("/x/loose"), 32 * GIB, 0),
              (Path("/x/tight"), 16 * GIB, 15 * GIB),
              (Path("/x/mid"), 16 * GIB, 0)]
    assert _tightest_visible_level(levels) == (Path("/x/tight"), 16 * GIB,
                                              15 * GIB)
    # An exact tie goes to the first (deepest) level seen.
    assert _tightest_visible_level([(Path("/a"), 4 * GIB, 0),
                                    (Path("/b"), 8 * GIB, 4 * GIB)]) \
        == (Path("/a"), 4 * GIB, 0)


def test_parent_current_usage_including_siblings_bounds_child(tmp_path):
    """Parent memory.current covers the whole subtree: sibling usage
    tightens the child even when the child's own counters show full
    room — the refusal names the 1 GiB sibling-tightened bound."""
    env = make_env(tmp_path, cgroup_path="/parent/child")
    put_level(env, "/", mem_max="max")
    put_level(env, "/parent", mem_max=16 * GIB, mem_current=15 * GIB)
    put_level(env, "/parent/child", mem_max=16 * GIB, mem_current=0)
    report = preflight(env, checkpoint_bytes=2 * GIB,
                       checkpoint_dtype="fp32", target_dtype="fp32",
                       headroom_bytes=0)
    assert report.decision == "refuse"
    reasons = " ".join(report.reasons)
    assert f"advisory visible availability {1 * GIB} B" in reasons
    assert "exceeds the tightest visible availability" in reasons


def test_finite_child_unlimited_parent_reports_deepest_tightest_advisory(
        tmp_path):
    env = make_env(tmp_path, cgroup_path="/parent/child")
    put_level(env, "/", mem_max="max")
    put_level(env, "/parent", mem_max="max")
    put_level(env, "/parent/child", mem_max=8 * GIB, mem_current=0)
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="fp32",
                       target_dtype="fp32", headroom_bytes=2 * GIB)
    assert report.decision == "refuse"   # unproven root scope, not arithmetic
    reasons = " ".join(report.reasons)
    assert f"advisory visible availability {8 * GIB} B" in reasons


def test_all_unlimited_visible_root_still_refuses_unproven_completeness(
        tmp_path):
    """Root+peer scope review: 'max' at every visible level — including
    the root — is a scope observation, not proof of no hidden tighter
    ancestor; a disclaimer does not make allowing a real loader safe."""
    env = make_env(tmp_path, host_kb=64 * 1024 * 1024,
                   cgroup_path="/parent/child")
    put_level(env, "/", mem_max="max")
    put_level(env, "/parent", mem_max="max")
    put_level(env, "/parent/child", mem_max="max")
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN
    assert report.budget_bytes is None
    reasons = " ".join(report.reasons)
    assert "completeness cannot be proven" in reasons
    assert "cannot be ruled out" in reasons


@pytest.mark.parametrize("root_max", ["max", str(16 * GIB)])
def test_unproven_namespace_root_refuses_whatever_root_file_says(tmp_path,
                                                                 root_max):
    """Root+peer scope reproduction: a namespace '/' that is itself a
    non-root cgroup hides any tighter ancestor — present memory.max at
    the visible root (finite OR 'max') is not completeness evidence."""
    env = make_env(tmp_path, host_kb=32 * 1024 * 1024, cgroup_path="/")
    put_level(env, "/", mem_max=root_max,
              mem_current=0 if root_max != "max" else None)
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="bf16", headroom_bytes=0)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN
    assert "completeness cannot be proven" in " ".join(report.reasons)


def test_trusted_unlimited_snapshot_proceeds_with_scope_note():
    """An explicitly injected unlimited snapshot is TRUSTED caller
    evidence for pure budget arithmetic (documented) — the visible-scope
    disclaimer still rides along honestly."""
    snap = MemorySnapshot(host_available_bytes=64 * GIB,
                          cgroup_state=CGROUP_UNLIMITED)
    report = encoder_memory_preflight(GIB, checkpoint_dtype="bf16",
                                      target_dtype="fp32", headroom_bytes=GIB,
                                      snapshot=snap)
    assert report.decision == "proceed"
    assert report.budget_bytes == 64 * GIB
    reasons = " ".join(report.reasons)
    assert "VISIBLE cgroup hierarchy" in reasons
    assert "whole-host/root absence of constraints is not proven" in reasons


@pytest.mark.parametrize("child_max", ["max", 8 * GIB])
def test_missing_memory_max_at_visible_root_fails_closed(tmp_path, child_max):
    """Kernel nuance: memory.max exists only on non-root cgroups and a
    namespace '/' can hide ancestors — an absent root file is unknown,
    never equated with unlimited, even when a finite child was seen."""
    env = make_env(tmp_path, cgroup_path="/child")
    put_level(env, "/child", mem_max=child_max,
              mem_current=0 if child_max != "max" else None)
    # '/' deliberately has NO memory.max file.
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN
    reasons = " ".join(report.reasons)
    assert "cannot be proven" in reasons
    if child_max != "max":
        # The finite child constraint is still reported truthfully.
        assert "finite caps were already seen at" in reasons


@pytest.mark.parametrize("parent_max,parent_current", [
    ("banana", None),        # unparseable ancestor cap
    (8 * GIB, None),         # finite ancestor cap without its counter
    (8 * GIB, "-1"),         # negative ancestor usage
])
def test_unknown_ancestor_state_refuses_below_a_clean_child(
        tmp_path, parent_max, parent_current):
    env = make_env(tmp_path, cgroup_path="/parent/child")
    put_level(env, "/parent", mem_max=parent_max, mem_current=parent_current)
    put_level(env, "/parent/child", mem_max=8 * GIB, mem_current=0)
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="fp32",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN


def test_missing_memory_max_at_intermediate_level_fails_closed(tmp_path):
    """Root+peer scope review: ENOENT on an intermediate memory.max does
    NOT prove the memory controller is disabled — the directory may be
    missing or mismapped and no controller evidence was read — so a
    missing file at ANY visible level refuses: never silently skipped,
    never asserted disabled."""
    env = make_env(tmp_path, host_kb=32 * 1024 * 1024,
                   cgroup_path="/gp/mid/leaf")
    put_level(env, "/", mem_max="max")
    put_level(env, "/gp", mem_max="max")
    put_level(env, "/gp/mid/leaf", mem_max=2 * GIB, mem_current=0)
    # /gp/mid deliberately has NO memory.max file (the walk goes leaf →
    # root, so the leaf's finite cap was truthfully seen before the gap).
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="fp32",
                       target_dtype="fp32", headroom_bytes=2 * GIB)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN
    reasons = " ".join(report.reasons)
    assert "does not prove the memory controller is disabled" in reasons
    assert "finite caps were already seen at" in reasons   # grandparent kept


@pytest.mark.parametrize("raw", [
    "0::relative/path\n",              # relative: could escape the root
    "0::/a/../b\n",                    # '..' component
    "0::.\n",                          # '.' path
    "0::/a\n0::/b\n",                  # duplicate, conflicting records
    "0::/a\nfake:0:/a\n0::/a\n",       # duplicated 0:: record
])
def test_malformed_proc_cgroup_paths_refuse(tmp_path, raw):
    env = make_env(tmp_path, mem_max=8 * GIB, mem_current=0, raw_cgroup=raw)
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="fp32",
                       target_dtype="fp32", headroom_bytes=GIB)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNKNOWN


# -- tighter-budget selection (trusted injected snapshots) --------------------


def test_finite_cgroup_never_substituted_by_larger_host():
    snap = MemorySnapshot(host_available_bytes=100 * GIB,   # plentiful host
                          cgroup_state=CGROUP_FINITE,
                          cgroup_limit_bytes=4 * GIB,
                          cgroup_current_bytes=0)
    report = encoder_memory_preflight(GIB, checkpoint_dtype="fp32",
                                      target_dtype="fp32",
                                      headroom_bytes=2 * GIB, snapshot=snap)
    assert report.decision == "proceed"
    assert report.budget_bytes == 4 * GIB          # cgroup, not host
    assert "never substituted away" in " ".join(report.reasons)


def test_host_can_be_the_tighter_budget_under_a_finite_cap():
    snap = MemorySnapshot(host_available_bytes=2 * GIB,
                          cgroup_state=CGROUP_FINITE,
                          cgroup_limit_bytes=16 * GIB,
                          cgroup_current_bytes=0)
    report = encoder_memory_preflight(GIB, checkpoint_dtype="fp32",
                                      target_dtype="fp32",
                                      headroom_bytes=2 * GIB, snapshot=snap)
    assert report.decision == "refuse"
    assert report.budget_bytes == 2 * GIB


def test_host_tighter_under_unlimited_cgroup():
    snap = MemorySnapshot(host_available_bytes=2 * GIB,
                          cgroup_state=CGROUP_UNLIMITED)
    report = encoder_memory_preflight(GIB, checkpoint_dtype="fp32",
                                      target_dtype="fp32",
                                      headroom_bytes=3 * GIB, snapshot=snap)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_UNLIMITED
    assert report.budget_bytes == 2 * GIB


def test_unlimited_cgroup_healthy_host_proceeds():
    snap = MemorySnapshot(host_available_bytes=64 * GIB,
                          cgroup_state=CGROUP_UNLIMITED)
    report = encoder_memory_preflight(GIB, checkpoint_dtype="bf16",
                                      target_dtype="fp32",
                                      headroom_bytes=GIB, snapshot=snap)
    assert report.decision == "proceed"
    assert report.budget_bytes == 64 * GIB
    assert "conservative prerequisite" in " ".join(report.reasons)


def test_exact_budget_boundary():
    snap = MemorySnapshot(host_available_bytes=2 * GIB,
                          cgroup_state=CGROUP_UNLIMITED)
    fits = encoder_memory_preflight(GIB, checkpoint_dtype="bf16",
                                    target_dtype="fp32", headroom_bytes=0,
                                    snapshot=snap)
    assert fits.decision == "proceed"           # required == budget
    off = encoder_memory_preflight(GIB, checkpoint_dtype="bf16",
                                   target_dtype="fp32", headroom_bytes=1,
                                   snapshot=snap)
    assert off.decision == "refuse"             # one byte over


def test_current_at_or_over_cap_leaves_zero_room():
    snap = MemorySnapshot(host_available_bytes=8 * GIB,
                          cgroup_state=CGROUP_FINITE,
                          cgroup_limit_bytes=4 * GIB,
                          cgroup_current_bytes=4 * GIB)
    report = encoder_memory_preflight(1, checkpoint_dtype="fp32",
                                      target_dtype="fp32",
                                      headroom_bytes=0, snapshot=snap)
    assert report.decision == "refuse"
    assert report.budget_bytes == 0


def test_zero_cap_is_zero_budget_refusal_not_raw_failure():
    snap = MemorySnapshot(host_available_bytes=GIB,
                          cgroup_state=CGROUP_FINITE,
                          cgroup_limit_bytes=0, cgroup_current_bytes=0)
    assert snap.budget()[0] == 0
    report = encoder_memory_preflight(1, checkpoint_dtype="fp32",
                                      target_dtype="fp32", headroom_bytes=0,
                                      snapshot=snap)
    assert report.decision == "refuse"
    assert report.cgroup_state == CGROUP_FINITE
    assert report.cgroup_limit_bytes == 0
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
    assert report.cgroup_state == CGROUP_UNKNOWN


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


@pytest.mark.parametrize("raw", [
    "MemTotal: 1 kB\n",                    # no MemAvailable line
    "MemAvailable: x kB\n",                # unparseable
    "MemAvailable: 1048576\n",             # documented kB unit missing
    "MemAvailable: 1048576 MB\n",          # wrong unit is not kB
    "MemAvailable: +1024 kB\n",            # sign-prefixed: not plain digits
])
def test_unknown_host_always_refuses(tmp_path, raw):
    """Host MemAvailable unknown refuses in EVERY cgroup state — unknown
    host pressure cannot be ruled out, even under a finite cap (this
    previously proceeded under a finite cap)."""
    for cgroup_kwargs in ({"mem_max": "max"},
                          {"mem_max": 32 * GIB, "mem_current": 0}):
        env = make_env(tmp_path, raw_meminfo=raw, **cgroup_kwargs)
        report = preflight(env, checkpoint_bytes=GIB,
                           checkpoint_dtype="bf16", target_dtype="bf16",
                           headroom_bytes=0)
        assert report.decision == "refuse"
        assert report.budget_bytes is None
        assert "host pressure cannot be ruled out" in " ".join(
            report.reasons)


def test_giant_memavailable_digits_are_unknown_not_raw_error(tmp_path):
    """Root scope review: a unit-correct '<5000 digits> kB' line trips
    CPython's int-str conversion SAFETY limit — caught locally as
    unknown. The interpreter's limit is never raised for this and the
    value is never approximated into a budget."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: " + "9" * 5000 + " kB\n",
                       encoding="utf-8")
    value, notes = read_host_available(meminfo)
    assert value is None
    assert "integer-conversion limit" in " ".join(notes)
    env = make_env(tmp_path, raw_meminfo=meminfo.read_text(encoding="utf-8"),
                   mem_max="max")
    report = preflight(env, checkpoint_bytes=GIB, checkpoint_dtype="bf16",
                       target_dtype="bf16", headroom_bytes=0)
    assert report.decision == "refuse"
    assert report.budget_bytes is None
    assert "host pressure cannot be ruled out" in " ".join(report.reasons)


def test_snapshot_with_finite_cap_but_no_current_is_typed_refusal():
    """Root reproduction: a directly constructed snapshot with a finite
    cap and current=None must NOT silently substitute zero and proceed."""
    with pytest.raises(MemoryPreflightError, match="never assumed"):
        MemorySnapshot(host_available_bytes=32 * GIB,
                       cgroup_state=CGROUP_FINITE,
                       cgroup_limit_bytes=16 * GIB,
                       cgroup_current_bytes=None)


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


def test_huge_integers_use_exact_integer_ceiling_division():
    """No float conversion: 10**400-scale inputs neither raise OverflowError
    nor round through a float (which would break the conservative bound)."""
    huge = 10 ** 400 + 1
    weights, required = encoder_memory_required(
        checkpoint_bytes=huge, checkpoint_dtype="bf16", target_dtype="fp32",
        headroom_bytes=0)
    assert weights == 2 * huge and required == 2 * huge   # exact doubling
    # odd byte count with an expansion ratio: ceiling is exact.
    assert encoder_memory_required(
        checkpoint_bytes=2 * huge + 1, checkpoint_dtype="f16",
        target_dtype="fp32", headroom_bytes=3)[0] == 2 * (2 * huge + 1)
    # A huge request against a tiny finite budget is a clean integer refusal.
    snap = MemorySnapshot(host_available_bytes=GIB,
                          cgroup_state=CGROUP_FINITE,
                          cgroup_limit_bytes=2 * GIB,
                          cgroup_current_bytes=0)
    report = encoder_memory_preflight(huge, checkpoint_dtype="bf16",
                                      target_dtype="bf16", headroom_bytes=0,
                                      snapshot=snap)
    assert report.decision == "refuse"
    assert report.required_bytes == huge


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


def test_snapshot_injection_is_validated_as_strictly_as_reads():
    with pytest.raises(MemoryPreflightError):
        MemorySnapshot(host_available_bytes=1, cgroup_state=CGROUP_UNKNOWN,
                       cgroup_limit_bytes=1)
    with pytest.raises(MemoryPreflightError):
        MemorySnapshot(host_available_bytes=1, cgroup_state=CGROUP_FINITE)
    with pytest.raises(MemoryPreflightError):
        MemorySnapshot(host_available_bytes=1, cgroup_state="plenty")
    # Counters must be non-negative plain integers wherever required.
    for bad_limit in (True, 1.0, -1, "1"):
        with pytest.raises(MemoryPreflightError):
            MemorySnapshot(host_available_bytes=1,
                           cgroup_state=CGROUP_FINITE,
                           cgroup_limit_bytes=bad_limit,
                           cgroup_current_bytes=0)
    for bad_current in (True, 1.0, -1):
        with pytest.raises(MemoryPreflightError):
            MemorySnapshot(host_available_bytes=1,
                           cgroup_state=CGROUP_FINITE,
                           cgroup_limit_bytes=8 * GIB,
                           cgroup_current_bytes=bad_current)
    for bad_host in (True, 1.0, -1):
        with pytest.raises(MemoryPreflightError):
            MemorySnapshot(host_available_bytes=bad_host,
                           cgroup_state=CGROUP_UNLIMITED)
    with pytest.raises(MemoryPreflightError):
        MemorySnapshot(host_available_bytes=1, cgroup_state=CGROUP_UNLIMITED,
                       cgroup_current_bytes=0)
    # Unknown host (None) is legal to construct — it just always refuses.
    snap = MemorySnapshot(host_available_bytes=None,
                          cgroup_state=CGROUP_FINITE,
                          cgroup_limit_bytes=32 * GIB,
                          cgroup_current_bytes=0)
    assert snap.budget()[0] is None
    # The advisory visible bound is validated too (None or non-negative
    # int) — and never participates in budget(): it is diagnostics only.
    for bad_advisory in (True, 1.0, -1, "1"):
        with pytest.raises(MemoryPreflightError):
            MemorySnapshot(host_available_bytes=1,
                           cgroup_state=CGROUP_UNKNOWN,
                           advisory_visible_available_bytes=bad_advisory)
    snap = MemorySnapshot(host_available_bytes=GIB,
                          cgroup_state=CGROUP_UNKNOWN,
                          advisory_visible_available_bytes=4096)
    assert snap.budget()[0] is None


def test_report_is_json_safe_and_tagged(tmp_path):
    env = make_env(tmp_path, mem_max=CGROUP_CAP_16GIB, mem_current=0)
    report = preflight(env)
    assert isinstance(report, MemoryPreflightReport)
    doc = json.loads(json.dumps(report.to_dict()))
    assert doc["schema"] == MEM_PREFLIGHT_SCHEMA
    assert doc["decision"] == "refuse"
    assert doc["conservative_precondition"] is True
    assert doc["checkpoint_bytes"] == ENCODER_CHECKPOINT_BYTES
    # A real read never claims a proven complete cgroup state.
    assert doc["cgroup_state"] == "unknown"
    assert doc["cgroup_limit_bytes"] is None
    assert doc["budget_bytes"] is None


def test_read_memory_snapshot_fails_closed_with_advisory_availability(tmp_path):
    env = make_env(tmp_path, host_kb=8 * 1024 * 1024, mem_max=16 * GIB,
                   mem_current=GIB)
    snap = read_memory_snapshot(**env)
    assert snap.host_available_bytes == 8 * GIB
    assert snap.cgroup_state == CGROUP_UNKNOWN      # root scope unproven
    assert snap.cgroup_limit_bytes is None          # never authoritative
    assert snap.advisory_visible_available_bytes == 15 * GIB
    assert snap.budget()[0] is None


def test_read_memory_snapshot_reports_hierarchy_provenance(tmp_path):
    env = make_env(tmp_path, cgroup_path="/parent/child")
    put_level(env, "/", mem_max="max")
    put_level(env, "/parent", mem_max=8 * GIB, mem_current=GIB)
    put_level(env, "/parent/child", mem_max="max")
    snap = read_memory_snapshot(**env)
    assert snap.cgroup_state == CGROUP_UNKNOWN
    assert snap.advisory_visible_available_bytes == 7 * GIB
    notes = " ".join(snap.notes)
    assert "/parent/child" in notes        # child level named truthfully
    assert "/parent" in notes              # parent level named truthfully
    assert "chosen" in notes               # provenance of the tightest level
    assert "advisory" in notes             # explicitly never authoritative


def test_module_docstring_states_the_honest_limits():
    import qwen_engine.mem_preflight as mod
    doc = mod.__doc__
    assert "conservative prerequisite" in doc
    assert "not measured model-fit evidence" in doc
    assert "never substituted away" in doc
    assert "fail-closed" in doc
    assert "namespace" in doc
    assert "visible" in doc
    # The scope gate itself is documented, not just hinted at.
    assert "does not prove the memory controller is disabled" in doc
    assert "does not prove completeness" in doc
    assert "trusted caller evidence" in doc
    assert "advisory" in doc


def test_import_pulls_in_no_heavy_modules():
    code = (
        "import sys, qwen_engine.mem_preflight as m;"
        "heavy = [n for n in ('torch','diffusers','transformers') if n in sys.modules];"
        "assert not heavy, heavy;"
        "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
