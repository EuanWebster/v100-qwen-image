"""Tiny synthetic encoder assets; no real weights, ML imports or GPU access."""
from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest

from qwen_engine import encoder_assets as ea
from qwen_engine.local_bundle import LOCAL_REFERENCE_LAYOUT, reference_bundle_fingerprints
from qwen_engine.manifest import ModelArtifact, ModelManifest
from qwen_engine.mem_preflight import (
    CGROUP_UNLIMITED, MemorySnapshot, MemoryPreflightError, require_encoder_memory_fit,
)


@pytest.fixture
def tiny(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    artifacts = []
    for role, filenames in LOCAL_REFERENCE_LAYOUT.items():
        for filename in filenames:
            raw = f"synthetic-{filename}".encode()
            artifacts.append(ModelArtifact(role, "Qwen/Qwen-Image-2.1",
                "d26bb61231c349cf6b7896fa83353113880e1ba3", filename,
                hashlib.sha256(raw).hexdigest(), len(raw)))
            if role in ea.ENCODER_ROLES:
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
    return ModelManifest(1, tuple(artifacts)), root


@pytest.fixture
def passing_memory(monkeypatch):
    calls = []
    # Explicit test-only evidence injection. Product API has no snapshot option.
    def memory(checkpoint_bytes, **kwargs):
        calls.append((checkpoint_bytes, dict(kwargs)))
        return require_encoder_memory_fit(checkpoint_bytes, **kwargs, snapshot=MemorySnapshot(
            host_available_bytes=1024 ** 3, cgroup_state=CGROUP_UNLIMITED))
    monkeypatch.setattr(ea, "require_encoder_memory_fit", memory)
    return calls


def test_narrow_verification_publishes_only_encoder_components(tiny, passing_memory):
    manifest, root = tiny
    result = ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=42)
    assert result.root == root.resolve()
    assert set(result.assets) == set(result.component_dirs) == {"processor", "text_encoder"}
    assert [len(result.assets[r]) for r in ea.ENCODER_ROLES] == [9, 7]
    assert all(p.is_absolute() for paths in result.assets.values() for p in paths)
    assert result.encoder_artifact_set_sha256 == reference_bundle_fingerprints(manifest).encoder
    weight_bytes = sum(a.bytes for a in manifest.artifacts if a.identity.endswith(".safetensors") and a.role == "text_encoder")
    assert result.checkpoint_bytes == weight_bytes
    assert passing_memory == [(weight_bytes, dict(checkpoint_dtype="bf16", target_dtype="fp16", headroom_bytes=42))]
    assert result.memory_report.decision == "proceed"
    with pytest.raises(TypeError):
        result.assets["vae"] = ()


def test_unrelated_debris_and_missing_components_not_accessed(tiny, passing_memory, monkeypatch):
    manifest, root = tiny
    (root / "transformer").mkdir()
    (root / "transformer/a.partial").write_bytes(b"incomplete")
    os.mkfifo(root / "transformer/fifo")
    (root / "vae").symlink_to(root / "absent-vae", target_is_directory=True)
    os.mkfifo(root / "model_index.json")
    original = os.open
    def opened(path, *args, **kwargs):
        assert "transformer" not in str(path) and "vae" not in str(path)
        assert "model_index" not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(ea.os, "open", opened)
    result = ea.verify_encoder_assets(manifest, root, target_dtype="bf16", headroom_bytes=0)
    assert len(result.artifacts) == 16


@pytest.mark.parametrize("name", ["processor/extra.py", "text_encoder/weights.bin",
                                 "processor/unfinished.partial", "text_encoder/extra-dir"])
def test_stray_component_entries_refused_before_hash(tiny, passing_memory, monkeypatch, name):
    manifest, root = tiny
    path = root / name
    if name.endswith("extra-dir"):
        path.mkdir()
    else:
        path.write_bytes(b"stray")
    monkeypatch.setattr(ea, "_verify_one", lambda *args: pytest.fail("hash before preflight"))
    with pytest.raises(ea.EncoderAssetError, match="unexpected"):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=0)


@pytest.mark.parametrize("kind", ["missing", "link", "fifo", "directory"])
@pytest.mark.parametrize("filename", ["processor/tokenizer.json", "text_encoder/config.json"])
def test_bad_required_entries_refused_without_contents(tiny, passing_memory, monkeypatch, kind, filename):
    manifest, root = tiny
    path = root / filename
    path.unlink()
    if kind == "link":
        path.symlink_to(root / "processor/vocab.json")
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    monkeypatch.setattr(ea, "_verify_one", lambda *args: pytest.fail("hash before preflight"))
    with pytest.raises(ea.EncoderAssetError):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=0)


@pytest.mark.parametrize("role", ea.ENCODER_ROLES)
def test_component_directory_symlink_refused(tiny, role):
    _, root = tiny
    original = root / role
    moved = root / (role + "-moved")
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ea.EncoderAssetError):
        ea.preflight_encoder_tree(root)


def test_structural_preflight_reads_no_contents(tiny, monkeypatch):
    _, root = tiny
    monkeypatch.setattr(ea.os, "read", lambda *args: pytest.fail("content read"))
    entries = ea.preflight_encoder_tree(root)
    assert len(entries) == 18 and set(entries) == {*ea.ENCODER_ROLES, *ea.ENCODER_FILENAMES}


def test_memory_refusal_precedes_all_asset_io(tiny, monkeypatch):
    manifest, root = tiny
    events = []
    def memory(*args, **kwargs):
        events.append("memory")
        raise MemoryPreflightError("not enough memory")
    monkeypatch.setattr(ea, "require_encoder_memory_fit", memory)
    monkeypatch.setattr(ea, "_canonical_root", lambda *args: pytest.fail("asset root accessed"))
    monkeypatch.setattr(ea, "_verify_one", lambda *args: pytest.fail("hash started"))
    with pytest.raises(MemoryPreflightError, match="not enough memory"):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=42)
    assert events == ["memory"]


@pytest.mark.parametrize("headroom", [-1, True, None, "1024"])
def test_invalid_required_headroom_is_refused_before_tree(tiny, passing_memory, monkeypatch, headroom):
    manifest, root = tiny
    monkeypatch.setattr(ea, "_canonical_root", lambda *args: pytest.fail("asset root accessed"))
    with pytest.raises(MemoryPreflightError):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=headroom)


def test_no_public_memory_bypass_or_implicit_headroom(tiny):
    manifest, root = tiny
    with pytest.raises(TypeError):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16")
    with pytest.raises(TypeError):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=0, snapshot=None)


@pytest.mark.parametrize("change", ["size", "sha"])
def test_fresh_bytes_required(tiny, passing_memory, change):
    manifest, root = tiny
    path = root / "processor/tokenizer.json"
    content = path.read_bytes()
    path.write_bytes(content + b"x" if change == "size" else b"X" + content[1:])
    with pytest.raises(ea.EncoderAssetError, match="size|sha256"):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=0)


@pytest.mark.parametrize("swap", ["fifo", "link"])
def test_swap_after_preflight_does_not_block_or_follow(tiny, passing_memory, monkeypatch, swap):
    manifest, root = tiny
    original = ea._verify_one
    path = root / "processor/added_tokens.json"
    replaced = False
    def verify(artifact, fd):
        nonlocal replaced
        if not replaced:
            replaced = True
            path.unlink()
            if swap == "fifo":
                os.mkfifo(path)
            else:
                path.symlink_to(root / "processor/vocab.json")
        return original(artifact, fd)
    monkeypatch.setattr(ea, "_verify_one", verify)
    with pytest.raises(ea.EncoderAssetError):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=0)


def test_change_earlier_file_during_later_hash_refused(tiny, passing_memory, monkeypatch):
    manifest, root = tiny
    original = ea._verify_one
    count = 0
    def verify(artifact, fd):
        nonlocal count
        result = original(artifact, fd)
        count += 1
        if count == 2:
            path = root / "processor/added_tokens.json"
            path.write_bytes(b"X" + path.read_bytes()[1:])
        return result
    monkeypatch.setattr(ea, "_verify_one", verify)
    with pytest.raises(ea.EncoderAssetError, match="tree changed"):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=0)


@pytest.mark.parametrize("change", ["add", "directory-replace", "root-replace"])
def test_tree_changed_during_verification_refused(tiny, passing_memory, monkeypatch, change):
    manifest, root = tiny
    original = ea._verify_one
    mutated = False
    def verify(artifact, fd):
        nonlocal mutated
        result = original(artifact, fd)
        if not mutated:
            mutated = True
            if change == "add":
                (root / "processor/late.py").write_bytes(b"unexpected")
            elif change == "directory-replace":
                (root / "processor").rename(root / "old-processor")
                (root / "processor").mkdir()
            else:
                root.rename(root.parent / "old-models")
                root.mkdir()
        return result
    monkeypatch.setattr(ea, "_verify_one", verify)
    with pytest.raises(ea.EncoderAssetError, match="changed|refused|unexpected"):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=0)


def test_invalid_manifest_and_dtype_precede_memory(tiny, monkeypatch):
    manifest, root = tiny
    monkeypatch.setattr(ea, "require_encoder_memory_fit", lambda *args, **kwargs: pytest.fail("memory"))
    with pytest.raises(ea.EncoderAssetError):
        ea.verify_encoder_assets({}, root, target_dtype="fp16", headroom_bytes=0)
    with pytest.raises(ea.EncoderAssetError):
        ea.verify_encoder_assets(manifest, root, target_dtype="FP16", headroom_bytes=0)


def test_result_rejects_directly_forged_scalar_coherence(tiny, passing_memory):
    manifest, root = tiny
    result = ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=0)
    for changes in (dict(root=Path("relative")), dict(artifacts=()),
                    dict(memory_report=replace(result.memory_report, decision="refuse")),
                    dict(memory_report=replace(result.memory_report, checkpoint_bytes=999))):
        with pytest.raises(ea.EncoderAssetError):
            replace(result, **changes)


@pytest.mark.parametrize("kind", ["grow", "edit", "replace"])
def test_file_mutation_during_hash_refused(tiny, passing_memory, monkeypatch, kind):
    manifest, root = tiny
    target = root / "processor/added_tokens.json"
    original = os.read
    mutated = False
    def read(fd, length):
        nonlocal mutated
        block = original(fd, length)
        if block and not mutated:
            mutated = True
            raw = target.read_bytes()
            if kind == "grow":
                target.write_bytes(raw + b"extra")
            elif kind == "edit":
                target.write_bytes(b"X" + raw[1:])
            else:
                other = root / "replacement"
                other.write_bytes(raw)
                other.replace(target)
        return block
    monkeypatch.setattr(ea.os, "read", read)
    with pytest.raises(ea.EncoderAssetError, match="grew|changed"):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=0)


def test_relative_root_is_published_as_canonical_absolute(tiny, passing_memory, monkeypatch):
    manifest, root = tiny
    monkeypatch.chdir(root.parent)
    result = ea.verify_encoder_assets(manifest, Path("models"), target_dtype="fp16", headroom_bytes=0)
    assert result.root == root.resolve() and result.root.is_absolute()


def test_failed_preflight_does_not_leak_directory_descriptors(tiny):
    _, root = tiny
    (root / "text_encoder/unexpected.py").write_bytes(b"stray")
    before = len(list(Path("/proc/self/fd").iterdir()))
    for _ in range(10):
        with pytest.raises(ea.EncoderAssetError):
            ea.preflight_encoder_tree(root)
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_interrupt_closes_all_verifier_descriptors(tiny, passing_memory, monkeypatch):
    manifest, root = tiny
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(ea, "_verify_one", interrupt)
    before = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(KeyboardInterrupt):
        ea.verify_encoder_assets(manifest, root, target_dtype="fp16", headroom_bytes=0)
    assert len(list(Path("/proc/self/fd").iterdir())) == before
