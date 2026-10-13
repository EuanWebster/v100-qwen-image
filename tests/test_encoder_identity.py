"""ML-free production identity tests; all synthetic writes are guarded fixtures."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_engine import encoder_identity as ei
from qwen_engine.config import GenerationConfig
from qwen_engine.encoder import ComponentIdentity, EncoderRequestError, EncoderSetupError, cache_key
from qwen_engine.local_bundle import (
    LOCAL_REFERENCE_LAYOUT, artifact_set_fingerprint, resolve_local_bundle,
    reference_bundle_fingerprints,
)
from qwen_engine.manifest import ModelManifest, ModelArtifact, load_manifest

ROOT = Path(__file__).resolve().parents[1]
VISION_FIXTURE = "test-only-reviewed-vision"


def package_facts(vision=VISION_FIXTURE):
    facts = {name: ei.PackageObservation(version) for name, version in ei.PACKAGE_VERSIONS.items()}
    for name, (url, commit) in ei.VCS_PINS.items():
        facts[name] = ei.PackageObservation(ei.PACKAGE_VERSIONS[name], url, "git", commit)
    facts["torchvision"] = ei.PackageObservation(vision)
    return ei.InstalledEncoderPackages(sys.executable, (3, 11, 15), facts)


@pytest.fixture
def manifest():
    return load_manifest(ROOT / "models/manifest.json")


@pytest.fixture
def reviewed_fixture(monkeypatch):
    # Synthetic policy only: never installs torchvision or relaxes the product gate.
    monkeypatch.setattr(ei, "PINNED_TORCHVISION_VERSION", VISION_FIXTURE)
    monkeypatch.setattr(ei, "observe_installed_packages", package_facts)
    return package_facts()


def fake_distributions(monkeypatch, *, documents=None, missing=(), versions=None, extra=()):
    documents = {} if documents is None else documents
    versions = {} if versions is None else versions
    calls = []

    def lookup(name):
        calls.append(name)
        if name in missing:
            raise ei.metadata.PackageNotFoundError(name)
        url, commit = ei.VCS_PINS.get(name, (None, None))
        default = None if url is None else json.dumps({"url": url, "vcs_info": {
            "vcs": "git", "commit_id": commit, "requested_revision": commit}})
        return SimpleNamespace(metadata={"Name": name},
                               version=versions.get(name, ei.PACKAGE_VERSIONS.get(name, VISION_FIXTURE)),
                               read_text=lambda field: documents.get(name, default))

    monkeypatch.setattr(ei.metadata, "distribution", lookup)
    inventory = [lookup(name) for name in (*ei.PACKAGE_NAMES, *extra) if name not in missing]
    calls.clear()
    monkeypatch.setattr(ei.metadata, "distributions", lambda: iter(inventory))
    return calls


def test_observer_current_interpreter_without_ml_imports(monkeypatch):
    calls = fake_distributions(monkeypatch, missing=("torchvision",))
    before = {name for name in sys.modules if name.split(".")[0] in ei.PACKAGE_NAMES}
    facts = ei.observe_installed_packages()
    assert calls == list(ei.PACKAGE_NAMES)
    assert facts.executable == sys.executable
    assert facts.python_version == tuple(sys.version_info[:3])
    assert facts.packages["torchvision"].version is None
    assert facts.packages["diffusers"].commit == ei.PINNED_DIFFUSERS_COMMIT
    assert before == {name for name in sys.modules if name.split(".")[0] in ei.PACKAGE_NAMES}


@pytest.mark.parametrize("direct", [
    "{", "[]", '{"url":1}', '{"url":"x","url":"y"}',
    '{"url":"x","vcs_info":[]}', '{"url":"x","vcs_info":{"vcs":"git"}}',
    '{"url":"x","vcs_info":{"vcs":"git","commit_id":"a"},"dir_info":{}}',
    '{"url":"x","extra":NaN}', "[" * 2000 + "0" + "]" * 2000,
    "x" * (ei.MAX_DIRECT_URL_CHARS + 1),
])
def test_corrupt_metadata_refused_not_relabelled_as_absent(monkeypatch, direct):
    fake_distributions(monkeypatch, documents={"diffusers": direct})
    with pytest.raises(EncoderSetupError, match="diffusers metadata"):
        ei.observe_installed_packages()


def test_observer_missing_all_packages(monkeypatch):
    fake_distributions(monkeypatch, missing=ei.PACKAGE_NAMES)
    facts = ei.observe_installed_packages()
    assert all(p.version is None for p in facts.packages.values())
    with pytest.raises(EncoderSetupError, match="absent"):
        ei.validate_installed_packages(facts)


def test_read_failure_is_not_misreported_as_missing_distribution(monkeypatch):
    fake_distributions(monkeypatch)
    def read(_):
        raise ei.metadata.PackageNotFoundError("broken metadata file")
    monkeypatch.setattr(ei.metadata, "distribution", lambda _: SimpleNamespace(
        version="2.14.1+cu126", read_text=read))
    with pytest.raises(EncoderSetupError, match="cannot inspect installed torch metadata"):
        ei.observe_installed_packages()


def test_keyboard_interrupt_is_never_wrapped(monkeypatch):
    fake_distributions(monkeypatch)
    def interrupted(_):
        raise KeyboardInterrupt()
    monkeypatch.setattr(ei.metadata, "distribution", interrupted)
    with pytest.raises(KeyboardInterrupt):
        ei.observe_installed_packages()


@pytest.mark.parametrize("vision", [None, VISION_FIXTURE, "arbitrary-installed-version"])
def test_unlocked_torchvision_always_blocks_production(vision):
    assert ei.PINNED_TORCHVISION_VERSION is None
    with pytest.raises(EncoderSetupError, match="no reviewed pin"):
        ei.validate_installed_packages(package_facts(vision))


def test_reviewed_synthetic_package_policy_passes(reviewed_fixture):
    ei.validate_installed_packages(reviewed_fixture)


@pytest.mark.parametrize("name", ei.PACKAGE_NAMES)
@pytest.mark.parametrize("drift", ["missing", "version", "url", "commit", "vcs"])
def test_version_and_provenance_drift_refused(reviewed_fixture, name, drift):
    facts = dict(reviewed_fixture.packages)
    original = facts[name]
    if drift == "missing":
        facts[name] = ei.PackageObservation(None)
    else:
        facts[name] = replace(original, **{drift: "wrong"})
    with pytest.raises(EncoderSetupError, match=name):
        ei.validate_installed_packages(replace(reviewed_fixture, packages=facts))


@pytest.mark.parametrize("version", [(3, 12, 10), (3, 14, 1), (2, 7, 18)])
def test_unreviewed_python_refused(reviewed_fixture, version):
    with pytest.raises(EncoderSetupError, match="Python"):
        ei.validate_installed_packages(replace(reviewed_fixture, python_version=version))


def test_observations_and_snapshot_defensively_frozen():
    facts = package_facts()
    source = dict(facts.packages)
    snapshot = replace(facts, packages=source)
    source["torch"] = ei.PackageObservation(None)
    assert snapshot.packages["torch"].version == ei.PACKAGE_VERSIONS["torch"]
    with pytest.raises(TypeError):
        snapshot.packages["torch"] = ei.PackageObservation(None)
    for value in ("", " leading", 123, []):
        with pytest.raises(EncoderSetupError):
            ei.PackageObservation(value)
    with pytest.raises(EncoderSetupError):
        ei.PackageObservation(None, url="x")
    with pytest.raises(EncoderSetupError):
        replace(facts, packages={})


def test_runtime_policy_matches_complete_reviewed_lock():
    registry, vcs = {}, {}
    for line in (ROOT / "requirements.lock").read_text().splitlines():
        match = re.match(r"([a-z0-9-]+)==([^\s]+)", line)
        if match:
            assert match[1] not in registry
            registry[match[1]] = match[2]
        match = re.match(r"([a-z0-9-]+) @ git\+([^@]+)@([a-f0-9]{40})$", line)
        if match:
            vcs[match[1]] = (match[2], match[3])
    assert len(registry) + len(vcs) == 63
    assert vcs == dict(ei.VCS_PINS)
    assert registry == {n: v for n, v in ei.PACKAGE_VERSIONS.items() if n not in vcs}
    assert set(ei.PACKAGE_VERSIONS) == set(registry) | set(vcs)


def test_optional_inventory_observed_without_imports(monkeypatch):
    fake_distributions(monkeypatch, extra=("optional-kernel", "qwen-engine"),
                       versions={"optional-kernel": "1.0", "qwen-engine": "0.1.0"})
    facts = ei.observe_installed_packages()
    assert dict(facts.extra_packages) == {"optional-kernel": ei.PackageObservation("1.0")}
    with pytest.raises(TypeError):
        facts.extra_packages["optional-kernel"] = ei.PackageObservation("2.0")


@pytest.mark.parametrize("names", [("Torch", "torch"), ("some_lib", "some.lib"),
                                  ("bad/name",), (None,), ("",), ("x" * 129,)])
def test_ambiguous_or_invalid_distribution_inventory_refused(monkeypatch, names):
    monkeypatch.setattr(ei.metadata, "distributions", lambda: iter(
        SimpleNamespace(metadata={"Name": name}) for name in names))
    with pytest.raises(EncoderSetupError, match="inventory"):
        ei.observe_installed_packages()


def test_inventory_bound_and_interrupt(monkeypatch):
    fake_distributions(monkeypatch)
    monkeypatch.setattr(ei, "MAX_DISTRIBUTIONS", 1)
    with pytest.raises(EncoderSetupError, match="count cap"):
        ei.observe_installed_packages()
    def interrupted():
        raise KeyboardInterrupt()
    monkeypatch.setattr(ei.metadata, "distributions", interrupted)
    with pytest.raises(KeyboardInterrupt):
        ei.observe_installed_packages()


def test_direct_source_metadata_canonical_and_complete(monkeypatch):
    first = {"url": "file:///fixture", "dir_info": {"editable": True}}
    fake_distributions(monkeypatch, extra=("optional-kernel",),
                       documents={"optional-kernel": json.dumps(first)})
    original = ei.runtime_metadata_digest(ei.observe_installed_packages())
    fake_distributions(monkeypatch, extra=("optional-kernel",), documents={
        "optional-kernel": json.dumps(first, sort_keys=True, indent=4)})
    assert original == ei.runtime_metadata_digest(ei.observe_installed_packages())
    first["dir_info"]["editable"] = False
    fake_distributions(monkeypatch, extra=("optional-kernel",), documents={
        "optional-kernel": json.dumps(first)})
    assert original != ei.runtime_metadata_digest(ei.observe_installed_packages())


def test_runtime_digest_canonical_anchor_and_relocation(reviewed_fixture):
    facts = reviewed_fixture
    rows = [[n, v.version, v.url, v.vcs, v.commit, v.direct_url_sha256]
            for n, v in sorted(facts.packages.items())]
    doc = {"schema": "qwen_engine.runtime-metadata.v1", "python_version": [3, 11, 15],
           "packages": rows}
    expected = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert ei.runtime_metadata_digest(facts) == expected
    relocated = replace(facts, executable="/relocated/python",
                        packages=dict(reversed(list(facts.packages.items()))))
    assert ei.runtime_metadata_digest(relocated) == expected


@pytest.mark.parametrize("change", ["patch", "minor", "extra", "extra-version", "extra-source"])
def test_runtime_change_invalidates_cache_and_worker_request(manifest, reviewed_fixture, monkeypatch, change):
    facts = replace(reviewed_fixture, extra_packages={"optional-kernel": ei.PackageObservation("1.0")})
    monkeypatch.setattr(ei, "observe_installed_packages", lambda: facts)
    config = GenerationConfig()
    original = ei.project_encoder_request(config, manifest, encoder_dtype="fp16")
    if change in ("patch", "minor"):
        facts = replace(facts, python_version=(3, 11, 16) if change == "patch" else (3, 13, 5))
    elif change == "extra":
        facts = replace(facts, extra_packages={})
    else:
        facts = replace(facts, extra_packages={"optional-kernel": ei.PackageObservation(
            "2.0" if change == "extra-version" else "1.0",
            url="file:///different-source" if change == "extra-source" else None)})
    other = ei.project_encoder_request(config, manifest, encoder_dtype="fp16")
    assert cache_key(original) != cache_key(other)
    with pytest.raises(EncoderRequestError, match="disagrees"):
        ei.validate_worker_request_identity(original, manifest)


@pytest.mark.parametrize("name", ["tokenizers", "jinja2", "numpy", "nvidia-cublas-cu12"])
def test_transitive_drift_refuses_before_cache_identity(manifest, reviewed_fixture, monkeypatch, name):
    original = ei.project_encoder_request(GenerationConfig(), manifest, encoder_dtype="fp16")
    facts = dict(reviewed_fixture.packages)
    facts[name] = ei.PackageObservation("unreviewed")
    monkeypatch.setattr(ei, "observe_installed_packages", lambda: replace(reviewed_fixture, packages=facts))
    with pytest.raises(EncoderSetupError, match=name):
        ei.project_encoder_request(GenerationConfig(), manifest, encoder_dtype="fp16")
    with pytest.raises(EncoderSetupError, match=name):
        ei.validate_worker_request_identity(original, manifest)


@pytest.mark.parametrize("extras", [{"torch": ei.PackageObservation("1.0")},
                                   {"qwen-engine": ei.PackageObservation("1.0")},
                                   {"bad_name": ei.PackageObservation("1.0")},
                                   {"optional": ei.PackageObservation(None)},
                                   {"optional": "invalid"}])
def test_invalid_extra_snapshots_refused(extras):
    with pytest.raises(EncoderSetupError, match="extra"):
        replace(package_facts(), extra_packages=extras)


def source_tree(root):
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "encoder.py").write_text("VALUE = 1\n")
    (root / "sub").mkdir()
    (root / "sub/codec.py").write_text("VALUE = 2\n")
    return root


def test_source_identity_content_and_names_not_location_or_mtime(tmp_path):
    root = source_tree(tmp_path / "source")
    other = tmp_path / "relocated"
    shutil.copytree(root, other)
    digest = ei._source_digest(root)
    assert len(digest) == 64 and digest == ei._source_digest(other)
    os.utime(root / "encoder.py", ns=(1000000000, 2000000000))
    assert digest == ei._source_digest(root)
    (root / "ignored.bin").write_bytes(b"not source")
    (root / "__pycache__").mkdir()
    (root / "__pycache__/ignored.py").write_text("ignored cache")
    assert digest == ei._source_digest(root)
    (root / "encoder.py").write_text("VALUE = 3\n")
    assert digest != ei._source_digest(root)
    (root / "encoder.py").write_text("VALUE = 1\n")
    (root / "encoder.py").rename(root / "renamed.py")
    assert digest != ei._source_digest(root)


def test_source_digest_independent_canonical_anchor(tmp_path):
    root = source_tree(tmp_path / "source")
    rows = [[p.relative_to(root).as_posix(), p.stat().st_size,
             hashlib.sha256(p.read_bytes()).hexdigest()] for p in root.rglob("*.py")]
    document = {"schema": "qwen_engine.worker-source.v1", "files": sorted(rows)}
    expected = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert ei._source_digest(root) == expected


@pytest.mark.parametrize("kind", ["file-link", "directory-link", "fifo", "empty", "missing"])
def test_unsafe_source_tree_refused(tmp_path, kind):
    root = source_tree(tmp_path / "source")
    if kind == "file-link":
        (root / "alias.py").symlink_to(root / "encoder.py")
    elif kind == "directory-link":
        (root / "alias").symlink_to(root / "sub", target_is_directory=True)
    elif kind == "fifo":
        os.mkfifo(root / "blocked.py")
    elif kind == "empty":
        root = tmp_path / "empty"
        root.mkdir()
    else:
        root = tmp_path / "missing"
    with pytest.raises(EncoderSetupError):
        ei._source_digest(root)


@pytest.mark.parametrize("cap", ["MAX_SOURCE_FILES", "MAX_SOURCE_BYTES", "MAX_SOURCE_TOTAL_BYTES"])
def test_source_bounds(tmp_path, monkeypatch, cap):
    root = source_tree(tmp_path / "source")
    monkeypatch.setattr(ei, cap, 1)
    with pytest.raises(EncoderSetupError, match="cap|bounded"):
        ei._source_digest(root)


@pytest.mark.parametrize("change", ["edit", "replace"])
def test_source_change_during_read_is_refused(tmp_path, monkeypatch, change):
    root = source_tree(tmp_path / "source")
    original = os.read
    edited = False

    def read(fd, size):
        nonlocal edited
        block = original(fd, size)
        if block and not edited:
            edited = True
            if change == "edit":
                (root / "encoder.py").write_text("VALUE = 4\n")
            else:
                replacement = root / "replacement"
                replacement.write_text("VALUE = 1\n")
                replacement.replace(root / "encoder.py")
        return block

    monkeypatch.setattr(ei.os, "read", read)
    with pytest.raises(EncoderSetupError, match="changed"):
        ei._source_digest(root)


def test_parent_symlink_swap_cannot_redirect_source_open(tmp_path, monkeypatch):
    root = source_tree(tmp_path / "source")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "codec.py").write_text("outside content must not be read")
    original = os.open

    def swapped(path, flags, *args, **kwargs):
        if path == "sub":
            (root / "sub").rename(root / "old-sub")
            (root / "sub").symlink_to(outside, target_is_directory=True)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(ei.os, "open", swapped)
    with pytest.raises(EncoderSetupError, match="cannot fingerprint"):
        ei._source_digest(root)


def test_own_source_digest_and_invalid_request_prechecks(manifest, monkeypatch):
    assert len(ei.worker_source_digest()) == 64
    monkeypatch.setattr(ei, "observe_installed_packages", lambda: pytest.fail("package observation"))
    with pytest.raises(EncoderRequestError, match="dtype"):
        ei.project_encoder_request(GenerationConfig(), manifest, encoder_dtype="FP16")
    with pytest.raises(EncoderRequestError):
        ei.project_encoder_request({}, manifest, encoder_dtype="fp16")
    with pytest.raises(EncoderRequestError):
        ei.project_encoder_request(GenerationConfig(), {}, encoder_dtype="fp16")


@pytest.mark.parametrize("scale,negative,active", [(1, "bad", False), (2, "", False),
                                                 (2, "  ", False), (2, " bad ", True)])
def test_config_cfg_semantics_and_manifest_components(manifest, reviewed_fixture, scale, negative, active):
    config = GenerationConfig(prompt="  exact positive  ", negative_prompt=negative,
                              true_cfg_scale=scale, quant="q4")
    req = ei.project_encoder_request(config, manifest, encoder_dtype="bf16")
    assert req.prompt == config.prompt and req.negative_prompt == negative
    assert req.dtype == "bf16" and dict(req.kwargs) == {"cfg_active": active}
    by_name = {a.identity: a for a in manifest.artifacts}
    assert req.chat_template.sha256 == by_name["processor/chat_template.jinja"].sha256
    assert req.processor.sha256 == artifact_set_fingerprint(a for a in manifest.artifacts if a.role == "processor")
    assert req.tokenizer.sha256 == artifact_set_fingerprint(by_name[f] for f in ei.TOKENIZER_FILES)
    assert req.implementation_identity.encoder_artifact_set_sha256 == reference_bundle_fingerprints(manifest).encoder
    assert req.implementation_identity.worker_implementation_revision.endswith(ei.worker_source_digest())
    assert req.implementation_identity.torch_build == "cu126"
    ei.validate_worker_request_identity(req, manifest)


def test_generation_only_parameters_do_not_fragment_encoder_cache(manifest, reviewed_fixture):
    config = GenerationConfig(negative_prompt="negative", true_cfg_scale=2)
    first = ei.project_encoder_request(config, manifest, encoder_dtype="fp16")
    second = ei.project_encoder_request(replace(config, width=2048, steps=4, seed=999,
                                              true_cfg_scale=3, offload="offload", quant="q8"),
                                       manifest, encoder_dtype="fp16")
    assert cache_key(first) == cache_key(second)
    inactive = ei.project_encoder_request(replace(config, true_cfg_scale=1), manifest, encoder_dtype="fp16")
    assert cache_key(inactive) != cache_key(first)
    changed_text = ei.project_encoder_request(replace(config, true_cfg_scale=1, negative_prompt="different"),
                                             manifest, encoder_dtype="fp16")
    assert cache_key(changed_text) != cache_key(inactive)


@pytest.mark.parametrize("field", ["processor", "tokenizer", "chat_template", "model_repo",
                                  "model_revision", "model_sha256", "implementation_identity"])
def test_worker_refuses_forged_request_identity(manifest, reviewed_fixture, field):
    req = ei.project_encoder_request(GenerationConfig(), manifest, encoder_dtype="fp16")
    if field in ("processor", "tokenizer", "chat_template"):
        value = ComponentIdentity(sha256="f" * 64)
    elif field == "implementation_identity":
        value = replace(req.implementation_identity, worker_implementation_revision="manual-tag")
    elif field == "model_revision":
        value = "f" * 40
    elif field == "model_sha256":
        value = "f" * 64
    else:
        value = "wrong/repository"
    with pytest.raises(EncoderRequestError, match="disagrees"):
        ei.validate_worker_request_identity(replace(req, **{field: value}), manifest)


def test_worker_refuses_changed_own_source(manifest, reviewed_fixture, monkeypatch):
    req = ei.project_encoder_request(GenerationConfig(), manifest, encoder_dtype="fp16")
    monkeypatch.setattr(ei, "worker_source_digest", lambda: "f" * 64)
    with pytest.raises(EncoderRequestError, match="disagrees"):
        ei.validate_worker_request_identity(req, manifest)


@pytest.mark.parametrize("filename", ["processor/tokenizer.json", "processor/chat_template.jinja",
                                      "text_encoder/model-00001-of-00004.safetensors"])
def test_encoder_asset_identity_change_invalidates_request(manifest, reviewed_fixture, filename):
    req = ei.project_encoder_request(GenerationConfig(), manifest, encoder_dtype="fp16")
    changed = replace(manifest, artifacts=tuple(
        replace(a, sha256="f" * 64) if a.identity == filename else a for a in manifest.artifacts))
    other = ei.project_encoder_request(GenerationConfig(), changed, encoder_dtype="fp16")
    assert cache_key(req) != cache_key(other)
    with pytest.raises(EncoderRequestError, match="disagrees"):
        ei.validate_worker_request_identity(req, changed)


def test_unrelated_vae_identity_does_not_fragment_encoder_cache(manifest, reviewed_fixture):
    changed = replace(manifest, artifacts=tuple(
        replace(a, sha256="f" * 64) if a.role == "vae" else a for a in manifest.artifacts))
    original = ei.project_encoder_request(GenerationConfig(), manifest, encoder_dtype="fp16")
    other = ei.project_encoder_request(GenerationConfig(), changed, encoder_dtype="fp16")
    assert cache_key(original) == cache_key(other)


def test_request_projection_does_not_verify_claimed_asset_bytes(manifest, reviewed_fixture, monkeypatch):
    # Asset IO is forbidden here; own-source hashing remains real and allowed.
    import qwen_engine.manifest as mm
    monkeypatch.setattr(mm, "resolve_manifest", lambda *args: pytest.fail("asset verification"))
    req = ei.project_encoder_request(GenerationConfig(), manifest, encoder_dtype="fp16")
    assert req.implementation_identity is not None


def test_verified_bundle_coherence(tmp_path, reviewed_fixture):
    artifacts = []
    for role, filenames in LOCAL_REFERENCE_LAYOUT.items():
        for filename in filenames:
            path = tmp_path / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = filename.encode()
            path.write_bytes(raw)
            artifacts.append(ModelArtifact(role, "Qwen/Qwen-Image-2.1",
                "d26bb61231c349cf6b7896fa83353113880e1ba3", filename,
                hashlib.sha256(raw).hexdigest(), len(raw)))
    manifest = ModelManifest(1, tuple(artifacts))
    bundle = resolve_local_bundle(manifest, tmp_path)
    config = GenerationConfig()
    expected = ei.project_encoder_request(config, manifest, encoder_dtype="fp16")
    assert ei.request_from_verified_bundle(config, manifest, bundle, encoder_dtype="fp16") == expected
    with pytest.raises(EncoderRequestError, match="fingerprints"):
        ei.request_from_verified_bundle(config, manifest, replace(bundle,
            fingerprints=replace(bundle.fingerprints, encoder="f" * 64)), encoder_dtype="fp16")
    with pytest.raises(EncoderRequestError, match="paths/roles"):
        ei.request_from_verified_bundle(config, manifest, replace(bundle, assets={}), encoder_dtype="fp16")
    with pytest.raises(EncoderRequestError, match="paths/roles"):
        ei.request_from_verified_bundle(config, manifest, replace(bundle, component_dirs={}), encoder_dtype="fp16")
