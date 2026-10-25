"""M1.4 encoder worker + cache tests — fake worker only, no model (plan R5).

Covers: cache-key v2/v3 semantics (negative-key fix both variants,
production implementation identity), atomic writes, corrupt-entry miss +
quarantine, LRU cap, per-key concurrency, bounded worker responses, worker
timeout / cancel (terminate → kill) / error + stderr capture / protocol
violations, scratch cleanup on every path, and the no-heavy-imports boundary.
"""
from __future__ import annotations

import contextlib
import gc
import dataclasses
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
import types
import weakref
from pathlib import Path

import pytest

from qwen_engine.encoder import (
    CACHE_KEY_SCHEMA,
    DEFAULT_MAX_RESPONSE_BYTES,
    ENTRY_SCHEMA,
    PINNED_DIFFUSERS_COMMIT,
    PINNED_TRANSFORMERS_COMMIT,
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
    Blob,
    ComponentIdentity,
    EmbeddingCache,
    EncoderCancelledError,
    EncoderError,
    EncoderImplementationIdentity,
    EncoderProtocolError,
    EncoderRequest,
    EncoderRequestError,
    EncoderSetupError,
    EncoderTimeoutError,
    EncoderWorker,
    EncoderWorkerError,
    EncodedPayload,
    _EntryCorrupt,
    cache_key,
    embedding_cache_for,
    request_image_files,
)

HEAVY_MODULES = ("torch", "diffusers", "transformers")

FAKE_WORKER = Path(__file__).parent / "fake_encoder_worker.py"

REV_A = "a" * 40
SHA_A = "b" * 64
SHA_B = "c" * 64


def make_request(**overrides) -> EncoderRequest:
    base = dict(
        prompt="a red cube on a table",
        dtype="fp16",
        model_repo="Qwen/Qwen-Image-2.1",
        model_revision=REV_A,
        processor=ComponentIdentity(sha256=SHA_A),
        tokenizer=ComponentIdentity(revision=REV_A),
        chat_template=ComponentIdentity(sha256=SHA_B),
    )
    base.update(overrides)
    return EncoderRequest(**base)


def make_worker(tmp_path, *, timeout_s=15.0, cache=None, logger=None,
                grace=None, extra_kwargs=None,
                max_response_bytes=None) -> tuple[EncoderWorker, dict]:
    """Worker wired to the fake script; ``extra_kwargs`` drive the fake."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    kwargs = dict(extra_kwargs or {})
    # Fake behaviour knobs ride in the request kwargs (see fake script) and
    # are also part of the cache key — fine for these tests.
    worker = EncoderWorker(
        command=[sys.executable, str(FAKE_WORKER), "--request", "{request}",
                 "--response", "{response}"],
        timeout_s=timeout_s,
        cache=cache,
        scratch_dir=scratch,
        **({"cancel_grace_s": grace} if grace is not None else {}),
        **({"max_response_bytes": max_response_bytes}
           if max_response_bytes is not None else {}),
        logger=logger,
    )
    return worker, kwargs


def assert_scratch_clean(tmp_path) -> None:
    assert list((tmp_path / "scratch").iterdir()) == []


# ---------------------------------------------------------------------------
# Schema tags + request validation
# ---------------------------------------------------------------------------


def test_schema_tags_are_versioned():
    assert REQUEST_SCHEMA == "qwen_engine.encoder_request.v3"
    assert RESULT_SCHEMA == "qwen_engine.encoder_result.v1"
    assert CACHE_KEY_SCHEMA == "qwen_engine.embed_cache_key.v3"
    assert ENTRY_SCHEMA == "qwen_engine.embed_entry.v1"


def test_request_roundtrip_is_lossless():
    req = make_request(negative_prompt="ugly", kwargs={"a": 1, "b": {"c": [2, 3]}},
                       images=(SHA_A,))
    doc = req.to_request_dict("rid-1")
    assert doc["schema"] == REQUEST_SCHEMA and doc["request_id"] == "rid-1"
    parsed, rid = EncoderRequest.from_request_dict(doc)
    assert rid == "rid-1"
    assert parsed.key_document() == req.key_document()


def test_request_v3_image_files_are_wire_only():
    req = make_request(images=(SHA_A, SHA_B))
    doc = req.to_request_dict("rid", image_files=["/store/a.img", "/store/b.img"])
    assert doc["schema"] == REQUEST_SCHEMA
    assert doc["image_files"] == ["/store/a.img", "/store/b.img"]
    parsed, rid = EncoderRequest.from_request_dict(doc)
    assert rid == "rid"
    assert "image_files" not in parsed.key_document()
    assert parsed.key_document() == req.key_document()
    assert cache_key(parsed) == cache_key(req)  # identity stays hash-based
    assert request_image_files(doc) == ("/store/a.img", "/store/b.img")


def test_text_only_request_document_stays_byte_identical():
    req = make_request()
    plain = req.to_request_dict("rid")
    assert "image_files" not in plain
    assert req.to_request_dict("rid", image_files=None) == plain
    assert req.to_request_dict("rid", image_files=()) == plain
    assert request_image_files(plain) == ()


def test_t2i_cache_key_survives_the_v3_wire_upgrade():
    # Frozen golden value computed from the pre-upgrade (schema v2) encoder
    # module: the key document of text-only conditioning is byte-identical
    # across the v2 -> v3 transport bump, so existing cache entries keep
    # hitting. Any change to this value is a cache-wipe regression.
    req = make_request(negative_prompt="ugly, blurry", kwargs={"cfg_active": True})
    assert cache_key(req) == ("ee4c3686e973d8b7fa01fbff8472a414b821de5c90dca984"
                              "ebeb1202cb3eac68")


@pytest.mark.parametrize("files", [
    "/store/a.img",
    ["/store/a.img"],
    ["/store/a.img", "relative/b.img"],
    ["/store/a.img", "  "],
    ["/store/a.img", "/store/b\0.img"],
])
def test_malformed_image_files_are_typed(files):
    req = make_request(images=(SHA_A, SHA_B))
    with pytest.raises(EncoderRequestError, match="image_files"):
        req.to_request_dict("rid", image_files=files)


def test_image_files_without_images_is_refused():
    req = make_request()
    with pytest.raises(EncoderRequestError, match="image_files"):
        req.to_request_dict("rid", image_files=["/store/a.img"])
    doc = make_request().to_request_dict("rid")
    doc["image_files"] = ["/store/a.img"]
    with pytest.raises(EncoderRequestError, match="image_files"):
        EncoderRequest.from_request_dict(doc)


def test_from_request_dict_validates_image_files_shape():
    doc = make_request(images=(SHA_A,)).to_request_dict("rid")
    doc["image_files"] = ["relative/a.img"]
    with pytest.raises(EncoderRequestError, match="image_files"):
        EncoderRequest.from_request_dict(doc)
    doc = make_request(images=(SHA_A,)).to_request_dict("rid")
    doc["image_files"] = "/store/a.img"
    with pytest.raises(EncoderRequestError, match="image_files"):
        EncoderRequest.from_request_dict(doc)
    assert request_image_files(
        make_request(images=(SHA_A,)).to_request_dict(
            "rid", image_files=["/store/a.img"])) == ("/store/a.img",)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"prompt": "  "}, "prompt"),
        ({"prompt": None}, "prompt"),
        ({"negative_prompt": 5}, "negative_prompt"),
        ({"dtype": " "}, "dtype"),
        ({"dtype": " fp16 "}, "dtype"),
        ({"model_repo": ""}, "model_repo"),
        ({"model_revision": "main"}, "model"),
        ({"model_revision": None, "model_sha256": None}, "model"),
        ({"model_revision": REV_A, "model_sha256": "nothex"}, "model"),
        ({"kwargs": {"s": {1, 2}}}, "kwargs"),
        ({"images": ("not-hex",)}, "images[0]"),
        ({"images": "ab" * 32}, "images"),
    ],
)
def test_request_validation_names_the_field(overrides, field):
    with pytest.raises(EncoderRequestError) as exc:
        make_request(**overrides)
    assert field in str(exc.value)


def test_component_identity_requires_a_pin():
    with pytest.raises(EncoderRequestError):
        ComponentIdentity()  # floating: neither revision nor hash
    with pytest.raises(EncoderRequestError):
        ComponentIdentity(revision="branch-name")
    ComponentIdentity(revision=REV_A, sha256=SHA_A)  # both is fine


def test_request_from_dict_rejects_unknown_keys_and_bad_schema():
    doc = make_request().to_request_dict("rid")
    doc["surprise"] = True
    with pytest.raises(EncoderRequestError, match="surprise"):
        EncoderRequest.from_request_dict(doc)
    doc = make_request().to_request_dict("rid")
    doc["schema"] = "qwen_engine.encoder_request.v0"
    with pytest.raises(EncoderRequestError, match="schema"):
        EncoderRequest.from_request_dict(doc)
    doc = make_request().to_request_dict("")
    with pytest.raises(EncoderRequestError, match="request_id"):
        EncoderRequest.from_request_dict(doc)


# ---------------------------------------------------------------------------
# Production encoder implementation identity (cache key v3)
# ---------------------------------------------------------------------------

IDENTITY_VALUES = {
    "encoder_artifact_set_sha256": SHA_A,
    "worker_protocol_revision": "worker-proto.v1",
    "worker_implementation_revision": "worker-impl.3ce32fd",
    "prompt_extraction_revision": "prompt-extract.v1",
    "diffusers_commit": PINNED_DIFFUSERS_COMMIT,
    "transformers_commit": PINNED_TRANSFORMERS_COMMIT,
    "torch_build": "cpu",
    "torch_version": "2.9.0",
    "safetensors_version": "0.5.2",
    "torchvision_version": "absent",
}


def make_identity(**overrides) -> EncoderImplementationIdentity:
    return EncoderImplementationIdentity(**{**IDENTITY_VALUES, **overrides})


def test_pinned_commit_constants_match_the_reviewed_pins():
    assert PINNED_DIFFUSERS_COMMIT == \
        "c6df88a511a98740646ee55577b590c9852650ce"
    assert PINNED_TRANSFORMERS_COMMIT == \
        "856157a2f3e9594954310df18fdccc31ffddebe9"


def test_generic_request_may_have_no_implementation_identity():
    """Existing generic (fake) requests stay valid with identity=None; the
    key document states the absence explicitly, and a wire document without
    the block parses back to the same key."""
    req = make_request()
    assert req.implementation_identity is None
    assert req.key_document()["implementation_identity"] is None
    doc = req.to_request_dict("rid")
    parsed, _ = EncoderRequest.from_request_dict(doc)
    assert parsed.implementation_identity is None
    assert cache_key(parsed) == cache_key(req)
    # An absent key parses identically to an explicit null.
    del doc["implementation_identity"]
    assert cache_key(EncoderRequest.from_request_dict(doc)[0]) == cache_key(req)
    assert cache_key(req) != cache_key(make_request(
        implementation_identity=make_identity()))


def test_identity_request_roundtrip_is_exact():
    req = make_request(implementation_identity=make_identity(),
                       kwargs={"a": 1}, images=(SHA_A,))
    doc = req.to_request_dict("rid-1")
    assert doc["schema"] == REQUEST_SCHEMA
    assert doc["implementation_identity"] == req.key_document()[
        "implementation_identity"]
    parsed, rid = EncoderRequest.from_request_dict(doc)
    assert rid == "rid-1"
    assert parsed.implementation_identity == make_identity()
    assert parsed.key_document() == req.key_document()
    assert cache_key(parsed) == cache_key(req)


@pytest.mark.parametrize("field,alt", [
    ("encoder_artifact_set_sha256", SHA_B),
    ("worker_protocol_revision", "worker-proto.v2"),
    ("worker_implementation_revision", "worker-impl.deadbee"),
    ("prompt_extraction_revision", "prompt-extract.v2"),
    ("diffusers_commit", "f" * 40),
    ("transformers_commit", "e" * 40),
    ("torch_build", "cu126"),
    ("torch_version", "2.10.0"),
    ("safetensors_version", "0.6.0"),
    ("torchvision_version", "0.20.0"),
])
def test_every_identity_field_changes_the_key(field, alt):
    base = cache_key(make_request(implementation_identity=make_identity()))
    other = cache_key(make_request(
        implementation_identity=make_identity(**{field: alt})))
    assert base != other, f"cache key ignores implementation identity {field}"


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("encoder_artifact_set_sha256", "z" * 64),
        ("encoder_artifact_set_sha256", SHA_A.upper()),
        ("encoder_artifact_set_sha256", SHA_A[:63]),
        ("encoder_artifact_set_sha256", None),
        ("diffusers_commit", "c" * 41),
        ("diffusers_commit", "main"),
        ("diffusers_commit", 123),
        ("transformers_commit", None),
        ("worker_protocol_revision", " "),
        ("worker_implementation_revision", ""),
        ("worker_implementation_revision", 42),
        ("prompt_extraction_revision", None),
        ("prompt_extraction_revision", " spaced "),
        ("torch_build", ""),
        ("torch_version", None),
        ("safetensors_version", 0.5),
        ("torchvision_version", None),  # absence is stated with a marker, not null
    ],
)
def test_identity_rejects_malformed_values(field, bad):
    with pytest.raises(EncoderRequestError) as exc:
        make_identity(**{field: bad})
    assert field in str(exc.value)


def test_identity_parsing_is_strict():
    d = dict(IDENTITY_VALUES)
    with pytest.raises(EncoderRequestError, match="unknown"):
        EncoderImplementationIdentity.from_dict({**d, "surprise": 1})
    for drop in ("torch_version", "encoder_artifact_set_sha256"):
        with pytest.raises(EncoderRequestError, match=drop):
            EncoderImplementationIdentity.from_dict(
                {k: v for k, v in d.items() if k != drop})
    with pytest.raises(EncoderRequestError, match="must be an object"):
        EncoderImplementationIdentity.from_dict("not-a-mapping")
    with pytest.raises(EncoderRequestError, match="non-empty"):
        EncoderImplementationIdentity.from_dict({**d, "torch_version": None})
    EncoderImplementationIdentity.from_dict(d)  # the exact document parses


def test_request_from_dict_rejects_bad_identity():
    doc = make_request(implementation_identity=make_identity()
                       ).to_request_dict("rid")
    doc["implementation_identity"]["surprise"] = 1
    with pytest.raises(EncoderRequestError, match="surprise"):
        EncoderRequest.from_request_dict(doc)
    doc = make_request(implementation_identity=make_identity()
                       ).to_request_dict("rid")
    doc["implementation_identity"]["torch_version"] = ""
    with pytest.raises(EncoderRequestError, match="torch_version"):
        EncoderRequest.from_request_dict(doc)
    doc = make_request(implementation_identity=make_identity()
                       ).to_request_dict("rid")
    doc["implementation_identity"] = "identity-as-string"
    with pytest.raises(EncoderRequestError, match="implementation_identity"):
        EncoderRequest.from_request_dict(doc)


def test_identity_mutation_cannot_move_key_or_wire():
    """The record is frozen and to_dict/key_document are fresh documents:
    nothing a caller holds can move the key or the wire document later."""
    ident = make_identity()
    req = make_request(implementation_identity=ident)
    key_before = cache_key(req)
    wire_before = req.to_request_dict("rid")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ident.torch_version = "9.9.9"  # type: ignore[misc]
    snapshot = ident.to_dict()
    snapshot["torch_build"] = "poison"
    assert cache_key(req) == key_before
    assert req.to_request_dict("rid") == wire_before


def test_request_rejects_non_identity_implementation_values():
    for bad in ({"torch_version": "1"}, "identity", 5, [1]):
        with pytest.raises(EncoderRequestError, match="implementation_identity"):
            make_request(implementation_identity=bad)


# ---------------------------------------------------------------------------
# Cache key v2/v3
# ---------------------------------------------------------------------------


def test_key_is_deterministic_for_identical_requests():
    assert cache_key(make_request()) == cache_key(make_request())
    doc = json.dumps(make_request().key_document(), sort_keys=True)
    assert doc.count('"negative_prompt"') == 1  # negative is IN the key


def test_negative_key_fix_both_variants():
    """Variant 1: a different negative prompt must change the key.
    Variant 2: identical negatives (incl. both empty) must not."""
    plain = cache_key(make_request())
    with_neg = cache_key(make_request(negative_prompt="blurry, worst quality"))
    with_neg_again = cache_key(make_request(negative_prompt="blurry, worst quality"))
    assert plain != with_neg, "negative prompt missing from cache key"
    assert with_neg == with_neg_again
    assert plain != cache_key(make_request(negative_prompt=" "))


@pytest.mark.parametrize(
    "overrides",
    [
        {"dtype": "fp32"},
        {"kwargs": {"max_sequence_length": 512}},
        {"kwargs": {"nested": {"deep": 2}}},
        {"model_revision": "f" * 40},
        {"model_revision": None, "model_sha256": SHA_A},
        {"model_sha256": "d" * 64},
        {"processor": ComponentIdentity(sha256="e" * 64)},
        {"tokenizer": ComponentIdentity(revision="f" * 40)},
        {"chat_template": ComponentIdentity(sha256="a" * 64)},
        {"images": (SHA_A,)},
        {"images": (SHA_A, SHA_B)},
        {"images": (SHA_B, SHA_A)},
        {"prompt": "a blue cube"},
    ],
)
def test_every_key_v2_input_changes_the_key(overrides):
    base = cache_key(make_request(images=()))
    other = cache_key(make_request(**overrides))
    assert base != other, f"cache key ignores {overrides.keys()}"


def test_kwargs_key_is_order_independent():
    k1 = cache_key(make_request(kwargs={"a": 1, "b": 2}))
    k2 = cache_key(make_request(kwargs={"b": 2, "a": 1}))
    assert k1 == k2


# ---------------------------------------------------------------------------
# Payload + entry format (fake bytes roundtrip, tensor-ready, torch-free)
# ---------------------------------------------------------------------------


def test_payload_roundtrip_is_byte_exact():
    payload = EncodedPayload(
        metadata={"dtype": "fp16", "shape_note": "fake", "n": 2},
        blobs=(Blob("prompt_embeds", b"\x00\x01\xfe\xff embed"),
               Blob("mask", b"\x01" * 8)),
    )
    key = cache_key(make_request())
    data = payload.to_entry_bytes(key)
    head, _, body = data.partition(b"\n")
    header = json.loads(head)
    assert header["schema"] == ENTRY_SCHEMA and header["key"] == key
    # Blob payloads are *raw bytes in the file*: a future tensor payload is
    # stored by the worker side without the cache importing torch.
    assert body == b"\x00\x01\xfe\xff embed" + b"\x01" * 8
    back = EncodedPayload.from_entry_bytes(data, key)
    assert [(b.name, bytes(b.data)) for b in back.blobs] == [
        ("prompt_embeds", b"\x00\x01\xfe\xff embed"), ("mask", b"\x01" * 8)]
    assert back.metadata == payload.metadata


@pytest.mark.parametrize("mutate", [
    lambda d: d[:-1],                                        # truncated
    lambda d: d + b"extra",                                  # trailing bytes
    lambda d: d.replace(b'"schema"', b'"shema"', 1),         # header loses schema
    lambda d: d[:-1] + bytes([d[-1] ^ 0xFF]),                # blob sha mismatch
])
def test_corrupt_entry_bytes_are_detected(mutate):
    payload = EncodedPayload(blobs=(Blob("prompt_embeds", b"payload-bytes"),))
    key = cache_key(make_request())
    data = mutate(payload.to_entry_bytes(key))
    with pytest.raises(_EntryCorrupt):
        EncodedPayload.from_entry_bytes(data, key)


def test_entry_with_foreign_key_is_corrupt():
    payload = EncodedPayload(blobs=(Blob("b", b"x"),))
    data = payload.to_entry_bytes("a" * 64)
    with pytest.raises(_EntryCorrupt, match="does not match"):
        EncodedPayload.from_entry_bytes(data, "b" * 64)


def test_duplicate_blob_names_rejected():
    with pytest.raises(EncoderRequestError, match="duplicate blob names"):
        EncodedPayload(blobs=(Blob("x", b"1"), Blob("x", b"2")))


# ---------------------------------------------------------------------------
# EmbeddingCache: atomic writes, quarantine, LRU, locks
# ---------------------------------------------------------------------------


def test_cache_put_get_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache" / "embeds")
    key = cache_key(make_request())
    payload = EncodedPayload(metadata={"m": 1}, blobs=(Blob("e", b"bytes"),))
    path = cache.put(key, payload)
    assert path.name == f"{key}.embed"
    assert not list(path.parent.glob(".*.tmp"))  # atomic: no temp leftovers
    got = cache.get(key)
    assert got is not None
    assert [(b.name, bytes(b.data)) for b in got.blobs] == [("e", b"bytes")]
    assert got.metadata == {"m": 1}


def test_cache_get_misses_are_silent(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache" / "embeds")
    assert cache.get("a" * 64) is None          # nothing existed
    assert not (tmp_path / "cache" / "embeds").exists()  # no implicit writes


@pytest.mark.parametrize("corrupt", [
    lambda p: p.write_bytes(b"garbage\x00not-an-entry"),
    lambda p: p.write_bytes(p.read_bytes()[:-3]),
])
def test_corrupt_entry_is_miss_and_quarantined(tmp_path, corrupt):
    cache = EmbeddingCache(tmp_path / "cache" / "embeds")
    key = cache_key(make_request())
    cache.put(key, EncodedPayload(blobs=(Blob("e", b"abcdef"),)))
    corrupt(cache.entry_path(key))
    assert cache.get(key) is None
    quarantined = list(cache.quarantine_dir.iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].name.startswith(key) and ".corrupt" in quarantined[0].name
    assert not cache.entry_path(key).exists()
    # The key is reusable afterwards (miss path, not poisoned).
    cache.put(key, EncodedPayload(blobs=(Blob("e", b"new"),)))
    assert cache.get(key) is not None


def test_corrupt_entry_never_silently_overwritten_in_place(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache" / "embeds")
    key = cache_key(make_request())
    cache.entry_path(key).parent.mkdir(parents=True, exist_ok=True)
    cache.entry_path(key).write_bytes(b"\x00bad")
    assert cache.get(key) is None
    assert list(cache.quarantine_dir.iterdir()), "bad file must be moved aside"


def test_lru_cap_evicts_oldest_and_hits_refresh(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeds", max_entries=2)
    reqs = [make_request(prompt=f"p{i}") for i in ("a", "b", "c")]
    keys = [cache_key(r) for r in reqs]
    now = time.time()
    cache.put(keys[0], EncodedPayload(blobs=(Blob("e", b"1"),)))
    os.utime(cache.entry_path(keys[0]), (now - 200, now - 200))
    cache.put(keys[1], EncodedPayload(blobs=(Blob("e", b"2"),)))
    os.utime(cache.entry_path(keys[1]), (now - 100, now - 100))
    assert cache.get(keys[0]) is not None          # hit refreshes keys[0]
    cache.put(keys[2], EncodedPayload(blobs=(Blob("e", b"3"),)))
    assert sorted(cache.entries()) == sorted([keys[0], keys[2]])
    assert cache.get(keys[1]) is None, "LRU must evict the stalest entry"
    # The fresh entry itself is never the eviction victim.
    cache.put(cache_key(make_request(prompt="p")), EncodedPayload(blobs=(Blob("e", b"4"),)))
    assert len(cache.entries()) == 2


def test_lru_rejects_bad_caps(tmp_path):
    with pytest.raises(EncoderRequestError):
        EmbeddingCache(tmp_path, max_entries=0)
    with pytest.raises(EncoderRequestError):
        EmbeddingCache(tmp_path, max_entries=True)
    with pytest.raises(EncoderRequestError):
        EmbeddingCache("  ")
    with pytest.raises(EncoderRequestError):
        EmbeddingCache(tmp_path).entry_path("nope")


def test_per_key_locks_are_stable_and_distinct(tmp_path):
    cache = EmbeddingCache(tmp_path)
    k1, k2 = "a" * 64, "b" * 64
    assert cache.lock_for(k1) is cache.lock_for(k1)
    assert cache.lock_for(k1) is not cache.lock_for(k2)


def test_embedding_cache_for_config(tmp_path, base_config):
    assert embedding_cache_for(base_config).root == Path(base_config.cache_dir) / "embeds"
    from qwen_engine.config import GenerationConfig
    assert embedding_cache_for(GenerationConfig(cache_dir=None)) is None


# ---------------------------------------------------------------------------
# Worker runner
# ---------------------------------------------------------------------------


def test_worker_plain_encode_and_stderr_capture(tmp_path):
    events = []
    from qwen_engine.log import PhaseLogger
    worker, kwargs = make_worker(tmp_path, logger=PhaseLogger(sink=events.append),
                                 extra_kwargs={"_stderr": "hello from stderr"})
    req = make_request(kwargs=kwargs)
    payload = worker.run(req)
    assert payload.metadata["source"] == "fake-worker"
    assert payload.blobs[0].name == "prompt_embeds"
    assert payload.blobs[0].data.startswith(b"FAKEEMBED:")
    assert b"|".join([req.prompt.encode(), b""]) == payload.blobs[0].data[len(b"FAKEEMBED:"):]
    logged = [json.loads(line) for line in events]
    ok = [e for e in logged if e["event"] == "encoder_worker_ok"]
    assert ok and "hello from stderr" in ok[0]["stderr"], "stderr must reach the log"
    assert_scratch_clean(tmp_path)


def test_worker_python_placeholder_substitution(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    worker = EncoderWorker(
        command=["{python}", str(FAKE_WORKER), "--request", "{request}",
                 "--response", "{response}"],
        timeout_s=15.0, scratch_dir=scratch)
    payload = worker.run(make_request())
    assert payload.blobs[0].data.startswith(b"FAKEEMBED:")
    assert_scratch_clean(tmp_path)


def test_worker_rejects_bad_setup(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    base = {"timeout_s": 5.0, "scratch_dir": str(scratch)}
    with pytest.raises(EncoderSetupError, match="non-empty"):
        EncoderWorker(command=[], **base)
    with pytest.raises(EncoderSetupError, match="{response}"):
        EncoderWorker(command=["prog", "{request}"], **base)
    with pytest.raises(EncoderSetupError, match="timeout_s"):
        EncoderWorker(command=["prog", "{request}", "{response}"], timeout_s=0)
    with pytest.raises(EncoderSetupError, match="timeout_s"):
        EncoderWorker(command=["prog", "{request}", "{response}"], timeout_s=True)
    with pytest.raises(EncoderSetupError, match="cancel_grace_s"):
        EncoderWorker(command=["prog", "{request}", "{response}"], timeout_s=5,
                      cancel_grace_s=-1)
    worker = EncoderWorker(command=["definitely-not-a-real-binary",
                                    "{request}", "{response}"], **base)
    with pytest.raises(EncoderSetupError, match="launch"):
        worker.run(make_request())
    assert_scratch_clean(tmp_path)


def test_worker_timeout_terminates_child_and_cleans_up(tmp_path):
    pid_file = tmp_path / "worker.pid"
    worker, kwargs = make_worker(tmp_path, timeout_s=0.3,
                                 extra_kwargs={"_sleep": 30, "_pid_file": str(pid_file)})
    t0 = time.monotonic()
    with pytest.raises(EncoderTimeoutError, match="timed out"):
        worker.run(make_request(kwargs=kwargs))
    assert time.monotonic() - t0 < 5.0
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("worker child still alive after timeout")
    assert_scratch_clean(tmp_path)


def test_worker_timeout_kills_child_that_ignores_sigterm(tmp_path):
    pid_file = tmp_path / "worker.pid"
    worker, kwargs = make_worker(
        tmp_path, timeout_s=0.3, grace=0.2,
        extra_kwargs={"_sleep": 60, "_ignore_sigterm": True,
                      "_pid_file": str(pid_file)})
    with pytest.raises(EncoderTimeoutError):
        worker.run(make_request(kwargs=kwargs))
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("SIGTERM-ignoring child survived the kill path")
    assert_scratch_clean(tmp_path)


def test_worker_cancellation_via_event(tmp_path):
    pid_file = tmp_path / "worker.pid"
    worker, kwargs = make_worker(tmp_path, timeout_s=30.0,
                                 extra_kwargs={"_sleep": 30,
                                               "_pid_file": str(pid_file)})
    event = threading.Event()
    threading.Timer(0.2, event.set).start()
    t0 = time.monotonic()
    with pytest.raises(EncoderCancelledError, match="cancelled"):
        worker.run(make_request(kwargs=kwargs), cancel_event=event)
    assert time.monotonic() - t0 < 5.0
    assert_scratch_clean(tmp_path)


def test_worker_nonzero_exit_surfaces_stderr(tmp_path):
    worker, kwargs = make_worker(
        tmp_path, extra_kwargs={"_fail_exit": 3, "_stderr": "kaboom on stderr"})
    with pytest.raises(EncoderWorkerError) as exc:
        worker.run(make_request(kwargs=kwargs))
    assert exc.value.exit_code == 3
    assert "kaboom on stderr" in exc.value.stderr
    assert "kaboom on stderr" in str(exc.value)
    assert_scratch_clean(tmp_path)


def test_worker_error_result_is_typed_and_carries_stderr(tmp_path):
    worker, kwargs = make_worker(
        tmp_path,
        extra_kwargs={"_worker_error": {"type": "TokenizerError",
                                       "message": "chat template exploded"},
                      "_stderr": "template note"})
    with pytest.raises(EncoderWorkerError) as exc:
        worker.run(make_request(kwargs=kwargs))
    assert exc.value.worker_error["type"] == "TokenizerError"
    assert "chat template exploded" in str(exc.value)
    assert "template note" in exc.value.stderr
    assert_scratch_clean(tmp_path)


@pytest.mark.parametrize("mode", ["_garbage_response", "_no_response", "_bad_status"])
def test_worker_protocol_violations_are_typed(tmp_path, mode):
    worker, kwargs = make_worker(tmp_path, extra_kwargs={mode: True})
    with pytest.raises(EncoderProtocolError):
        worker.run(make_request(kwargs=kwargs))
    assert_scratch_clean(tmp_path)


# ---------------------------------------------------------------------------
# Bounded worker responses (max_response_bytes)
# ---------------------------------------------------------------------------


class _ExitedChild:
    """Stand-in for a finished Popen in direct `_result_from_files` probes."""
    returncode = 0


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "1024", None, 2.0,
                                 sys.maxsize, 10 ** 100])
def test_bad_max_response_caps_are_setup_errors(tmp_path, bad):
    with pytest.raises(EncoderSetupError, match="max_response_bytes"):
        EncoderWorker(
            command=[sys.executable, str(FAKE_WORKER), "--request", "{request}",
                     "--response", "{response}"],
            timeout_s=5.0, scratch_dir=tmp_path, max_response_bytes=bad)
    # The default is the documented practical bound.
    assert EncoderWorker(
        command=["x", "{request}", "{response}"], timeout_s=5.0,
        scratch_dir=tmp_path).max_response_bytes == DEFAULT_MAX_RESPONSE_BYTES


def test_representable_high_cap_validates_tiny_response_normally(tmp_path):
    """cap = sys.maxsize - 1 is representable (cap + 1 == sys.maxsize), so
    the constructor accepts it — but a one-shot buffered read of cap + 1
    would raise a raw OverflowError / speculative-allocate the cap. The
    chunked bounded read must instead read only the bytes that exist:
    normal tiny-file protocol validation (direct probe) and a normal
    end-to-end success (fake worker), never a raw OverflowError /
    MemoryError. No huge fixture is allocated anywhere."""
    worker = EncoderWorker(
        command=["x", "{request}", "{response}"], timeout_s=1,
        scratch_dir=tmp_path, max_response_bytes=sys.maxsize - 1)
    resp = tmp_path / "response.json"
    resp.write_bytes(b"{}")
    with pytest.raises(EncoderProtocolError) as exc:
        worker._result_from_files(_ExitedChild(), resp, "", "0" * 64, "rid")
    assert type(exc.value) is EncoderProtocolError
    assert "schema" in str(exc.value)

    real, kwargs = make_worker(tmp_path, max_response_bytes=sys.maxsize - 1)
    payload = real.run(make_request(kwargs=kwargs))
    assert payload.blobs[0].data.startswith(b"FAKEEMBED:")


def test_small_cap_refuses_before_json_decode(tmp_path):
    """End to end: the fake writes a perfectly valid response, but it is
    bigger than the tiny configured cap — a typed protocol error naming the
    cap and sizes, scratch cleaned, child reaped, stderr intact."""
    worker, kwargs = make_worker(tmp_path, max_response_bytes=256)
    with pytest.raises(EncoderProtocolError) as exc:
        worker.run(make_request(kwargs=kwargs))
    msg = str(exc.value)
    assert "max_response_bytes=256" in msg and "refusing to read" in msg
    assert exc.value.stderr == "" or isinstance(exc.value.stderr, str)
    assert_scratch_clean(tmp_path)


def test_exact_boundary_response_is_accepted(tmp_path):
    """cap == response size: valid (the bound is inclusive at the cap);
    cap == size - 1: refused. Same request + request_id → byte-identical
    fake responses, so the boundary is exact, not estimated."""
    rec = tmp_path / "response.bytes"
    worker, kwargs = make_worker(
        tmp_path, extra_kwargs={"_response_bytes_file": str(rec)})
    req = make_request(kwargs=kwargs)
    payload = worker.run(req, request_id="boundary-rid")
    assert payload.blobs[0].data.startswith(b"FAKEEMBED:")
    size = rec.stat().st_size
    assert size > 64

    exact, kwargs_exact = make_worker(
        tmp_path, max_response_bytes=size,
        extra_kwargs={"_response_bytes_file": str(tmp_path / "rec2")})
    again = exact.run(make_request(kwargs=kwargs_exact),
                      request_id="boundary-rid")
    assert again.to_entry_bytes("0" * 64) == payload.to_entry_bytes("0" * 64)

    over, kwargs_over = make_worker(
        tmp_path, max_response_bytes=size - 1)
    with pytest.raises(EncoderProtocolError, match="over max_response_bytes"):
        over.run(make_request(kwargs=kwargs_over), request_id="boundary-rid")
    assert_scratch_clean(tmp_path)


def test_oversized_response_refused_on_the_opened_descriptor(tmp_path):
    """A pre-existing oversized file is refused on the fstat of the OPENED
    descriptor before any byte is read: the message reports the sized
    size, and nothing is decoded."""
    resp = tmp_path / "response.json"
    resp.write_bytes(b'{"schema": ' + b"x" * 4096)
    size = resp.stat().st_size
    worker, _ = make_worker(tmp_path, max_response_bytes=1024)
    with pytest.raises(EncoderProtocolError,
                       match=rf"is {size} bytes, over max_response_bytes=1024"):
        worker._result_from_files(_ExitedChild(), resp, "", "0" * 64, "rid")


def test_growing_response_file_is_refused_by_the_bounded_read(tmp_path):
    """Deterministic growth simulation: a procfs file fstats as 0 bytes (so
    the opened-descriptor pre-size check passes) but reads back far more —
    exactly the sizing-vs-read race shape. The chunked bounded read (cap +
    1 bytes total) must refuse it as a grown file rather than decode an
    unbounded amount."""
    status = Path("/proc/self/status")
    if not status.exists() or os.stat(status).st_size != 0:
        pytest.skip("needs Linux procfs (st_size == 0, non-empty read)")
    worker, _ = make_worker(tmp_path, max_response_bytes=256)
    with pytest.raises(EncoderProtocolError, match="grew past"):
        worker._result_from_files(_ExitedChild(), status, "", "0" * 64, "rid")


class _BlockedByWatchdog(Exception):
    """Raised by the test's SIGALRM watchdog when a guarded call blocks."""


@contextlib.contextmanager
def _hang_guard(seconds=5.0):
    """Turn a would-hang read into a loud test failure instead of a hung
    suite: the SIGALRM watchdog interrupts the blocked call after
    ``seconds`` (external/process-style time bound, tiny fixtures only)."""
    if not hasattr(signal, "SIGALRM"):  # pragma: no cover - non-POSIX
        yield
        return

    def _fire(signum, frame):
        raise _BlockedByWatchdog(f"call blocked for more than {seconds}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_fifo_response_after_exit_is_refused_not_blocking(tmp_path):
    """Item-10 shape, direct probe: an exited child left a FIFO at the
    response path. Its st_size of 0 would pass a path-only cap check and a
    plain open() would hang forever (no worker timeout applies any more),
    so the no-follow/nonblocking OPENED descriptor plus fstat must refuse
    it as a typed protocol error immediately — the FIFO is left in place."""
    resp = tmp_path / "response.json"
    os.mkfifo(resp)
    worker, _ = make_worker(tmp_path)
    with _hang_guard():
        with pytest.raises(EncoderProtocolError, match="not a regular file"):
            worker._result_from_files(_ExitedChild(), resp, "", "0" * 64,
                                      "rid")
    assert os.path.exists(resp)


def test_directory_and_symlink_responses_are_refused(tmp_path):
    """Nonregular responses never reach the decoder: a directory is
    refused on the fstat (mode in the message), a symlink is refused
    without being followed (ELOOP) and its target is untouched."""
    worker, _ = make_worker(tmp_path)
    resp_dir = tmp_path / "dir.json"
    resp_dir.mkdir()
    with pytest.raises(EncoderProtocolError, match="not a regular file"):
        worker._result_from_files(_ExitedChild(), resp_dir, "", "0" * 64,
                                  "rid")
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(EncoderProtocolError, match="symlink"):
        worker._result_from_files(_ExitedChild(), link, "", "0" * 64, "rid")
    assert target.read_bytes() == b"{}"


def test_worker_left_fifo_at_response_path_typed_refusal_no_hang(tmp_path):
    """End-to-end item-10 reproduction: a real child mkfifo's the response
    path and exits 0. The parent must reap it, refuse the nonregular
    response with a typed protocol error (watchdog-guarded, so a
    regression fails instead of hanging the suite), and clean the scratch
    dir."""
    worker, kwargs = make_worker(
        tmp_path, timeout_s=15.0, extra_kwargs={"_fifo_response": True})
    with _hang_guard(seconds=10.0):
        with pytest.raises(EncoderProtocolError, match="not a regular file"):
            worker.run(make_request(kwargs=kwargs))
    assert_scratch_clean(tmp_path)


def test_valid_response_swapped_to_fifo_before_parent_reads(tmp_path):
    """Replacement race, made deterministic: the child writes a perfectly
    valid response, then unlinks it and leaves a FIFO on the path before
    exiting 0. A path-only pre-stat of the valid file can no longer be
    turned into a blocking FIFO open — the single opened descriptor is
    validated BEFORE reading, so this is the same typed refusal. The
    response bytes are never decoded."""
    worker, kwargs = make_worker(
        tmp_path, timeout_s=15.0,
        extra_kwargs={"_swap_response_to_fifo": True})
    with _hang_guard(seconds=10.0):
        with pytest.raises(EncoderProtocolError, match="not a regular file"):
            worker.run(make_request(kwargs=kwargs))
    assert_scratch_clean(tmp_path)


def test_fifo_at_stderr_log_path_falls_back_to_empty_excerpt(tmp_path):
    """End-to-end post-exit log-boundary reproduction: the child unlinks
    the scratch stderr.log and leaves a FIFO there. The excerpt read must
    neither block (watchdog-guarded) nor follow anything — it yields the
    empty fallback — and the missing response stays the typed error it
    was, with stderr empty rather than a hang or a new failure."""
    worker, kwargs = make_worker(
        tmp_path, timeout_s=15.0,
        extra_kwargs={"_fifo_stderr_log": True, "_no_response": True,
                      "_stderr": "written before the swap"})
    with _hang_guard(seconds=10.0):
        with pytest.raises(EncoderProtocolError, match="no response file") as exc:
            worker.run(make_request(kwargs=kwargs))
    assert exc.value.stderr == ""
    assert_scratch_clean(tmp_path)


def test_excerpt_nonregular_and_symlink_paths_fall_back_empty(tmp_path):
    """Direct _read_excerpt probe: a leftover FIFO, directory or symlink
    at the log path yields the empty fallback immediately (no block, no
    follow); an ordinary regular file keeps full diagnostics and a
    missing file stays empty."""
    import qwen_engine.encoder as encoder_mod

    fifo = tmp_path / "stderr.log"
    os.mkfifo(fifo)
    with _hang_guard():
        assert encoder_mod._read_excerpt(fifo) == ""
    log_dir = tmp_path / "dir.log"
    log_dir.mkdir()
    assert encoder_mod._read_excerpt(log_dir) == ""
    real = tmp_path / "real.log"
    real.write_bytes(b"boom\n")
    link = tmp_path / "link.log"
    link.symlink_to(real)
    assert encoder_mod._read_excerpt(link) == ""
    assert encoder_mod._read_excerpt(real) == "boom\n"
    assert encoder_mod._read_excerpt(tmp_path / "missing.log") == ""


def test_excerpt_read_is_bounded_when_the_file_grows_at_read(tmp_path,
                                                             monkeypatch):
    """Growth-at-read regression for the log boundary: every os.read is
    chunk-bounded and the TOTAL bytes read are bounded by
    _EXCERPT_READ_BYTES even though the file grows by a megabyte before
    each read — the documented tail bound is a READ bound, not a
    hope-the-file-still-has-that-size read()."""
    import qwen_engine.encoder as encoder_mod

    path = tmp_path / "stderr.log"
    path.write_bytes(b"a" * 4096)
    real_read = os.read
    calls = []

    def growing_read(fd, count):
        with open(path, "ab") as fh:
            fh.write(b"x" * (1 << 20))
        data = real_read(fd, count)
        calls.append((count, len(data)))
        return data

    monkeypatch.setattr(encoder_mod.os, "read", growing_read)
    text = encoder_mod._read_excerpt(path)
    assert calls, "the excerpt must actually read"
    assert all(count <= (1 << 20) for count, _ in calls), \
        "excerpt reads must be chunk bounded"
    assert sum(n for _, n in calls) <= encoder_mod._EXCERPT_READ_BYTES, \
        "total bytes read must stay at or under the documented bound"
    assert len(text) <= encoder_mod.STDERR_EXCERPT_CHARS + 20


# ---------------------------------------------------------------------------
# encode(): cache-first + concurrency
# ---------------------------------------------------------------------------


def test_encode_miss_then_hit_without_double_encode(tmp_path):
    counter = tmp_path / "encodes.log"
    cache = EmbeddingCache(tmp_path / "embeds")
    worker, kwargs = make_worker(tmp_path, cache=cache,
                                 extra_kwargs={"_counter_file": str(counter)})
    req = make_request(kwargs=kwargs)
    first = worker.encode(req)
    second = worker.encode(req)
    lines = counter.read_text().splitlines()
    assert len(lines) == 1, "second call must be a cache hit, not a re-encode"
    assert [(b.name, bytes(b.data)) for b in first.blobs] == \
           [(b.name, bytes(b.data)) for b in second.blobs]
    assert second.metadata == first.metadata
    assert cache.entry_path(cache_key(req)).is_file()
    assert_scratch_clean(tmp_path)


def test_concurrent_same_key_encodes_exactly_once(tmp_path):
    counter = tmp_path / "encodes.log"
    cache = EmbeddingCache(tmp_path / "embeds")
    worker, kwargs = make_worker(
        tmp_path, cache=cache,
        extra_kwargs={"_counter_file": str(counter), "_sleep": 0.4})
    req = make_request(kwargs=kwargs)
    results: list = []
    errors: list = []

    def go():
        try:
            results.append(worker.encode(req))
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors
    assert len(counter.read_text().splitlines()) == 1, "same-key race double-encoded"
    assert len(results) == 4
    assert all(bytes(r.blobs[0].data) == bytes(results[0].blobs[0].data)
               for r in results)
    assert_scratch_clean(tmp_path)


def test_failed_encode_is_not_cached(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeds")
    worker, kwargs = make_worker(tmp_path, cache=cache,
                                 extra_kwargs={"_fail_exit": 7})
    with pytest.raises(EncoderWorkerError):
        worker.encode(make_request(kwargs=kwargs))
    assert cache.entries() == []
    assert_scratch_clean(tmp_path)


# ---------------------------------------------------------------------------
# Senior-review hardening: immutability, typed validation, lifecycle, wire
# ---------------------------------------------------------------------------


def test_request_snapshot_survives_external_mutation():
    """Mutating the caller's (nested) kwargs after construction must not
    move the cache key, the wire document, or their agreement."""
    inner = {"deep": [1, {"deeper": 2}]}
    kwargs = {"a": inner, "b": [1, 2]}
    req = make_request(kwargs=kwargs)
    key_before = cache_key(req)
    wire_kwargs = req.to_request_dict("rid")["kwargs"]

    inner["deep"].append("poison")
    inner["added"] = True
    kwargs["c"] = [3]
    kwargs["a"] = "rebound"

    assert cache_key(req) == key_before, "cache key drifted after mutation"
    assert req.to_request_dict("rid")["kwargs"] == wire_kwargs

    # Key and wire document describe exactly the same request.
    doc = req.to_request_dict("rid-9")
    key_doc = req.key_document()
    assert doc["schema"] == REQUEST_SCHEMA
    assert {k: v for k, v in doc.items()
            if k not in ("schema", "request_id")} == \
           {k: v for k, v in key_doc.items() if k != "schema"}
    assert cache_key(req) == hashlib.sha256(json.dumps(
        key_doc, sort_keys=True, ensure_ascii=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()

    # The snapshot is frozen all the way down.
    with pytest.raises(TypeError):
        req.kwargs["x"] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        req.kwargs["b"].append(4)  # type: ignore[union-attr]


def test_payload_metadata_snapshot_survives_mutation():
    meta = {"nested": {"list": [1, {"x": 2}]}}
    key = cache_key(make_request())
    payload = EncodedPayload(metadata=meta, blobs=(Blob("e", b"1"),))
    before = payload.to_entry_bytes(key)
    meta["nested"]["list"].append("poison")
    meta["top"] = 1
    assert payload.to_entry_bytes(key) == before
    with pytest.raises(TypeError):
        payload.metadata["top"] = 2  # type: ignore[index]


def test_blob_bytearray_snapshot_is_immutable():
    ba = bytearray(b"embed-bytes")
    blob = Blob("e", ba)
    assert isinstance(blob.data, bytes)
    sha_before = blob.sha256
    entry = EncodedPayload(blobs=(blob,)).to_entry_bytes("a" * 64)
    ba[0] = ba[0] ^ 0xFF
    ba.append(9)
    assert blob.sha256 == sha_before
    assert EncodedPayload(blobs=(blob,)).to_entry_bytes("a" * 64) == entry


@pytest.mark.parametrize("overrides", [
    {"processor": "gpt2"}, {"tokenizer": {}}, {"chat_template": None},
])
def test_components_must_be_component_identity(overrides):
    with pytest.raises(EncoderRequestError):
        make_request(**overrides)


@pytest.mark.parametrize("kwargs", [
    {1: "int key"},
    {"n": float("nan")},
    {"i": float("inf")},
    {"c": 1 + 2j},
    {"o": object()},
])
def test_kwargs_reject_invalid_json_values(kwargs):
    with pytest.raises(EncoderRequestError, match="kwargs"):
        make_request(kwargs=kwargs)


@pytest.mark.parametrize("meta", [
    {"n": float("nan")},
    {5: "v"},
    {"s": {1, 2}},
])
def test_payload_metadata_rejects_invalid_json(meta):
    with pytest.raises(EncoderRequestError, match="metadata"):
        EncodedPayload(metadata=meta)


def test_payload_rejects_non_blob_entries():
    with pytest.raises(EncoderRequestError, match="blobs"):
        EncodedPayload(blobs=(Blob("a", b"0"), "not-a-blob"))
    with pytest.raises(EncoderRequestError, match="blobs"):
        EncodedPayload(blobs=("not-a-blob",))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_timeouts_are_rejected(bad):
    cmd = ["prog", "{request}", "{response}"]
    with pytest.raises(EncoderSetupError, match="timeout_s"):
        EncoderWorker(command=cmd, timeout_s=bad)
    with pytest.raises(EncoderSetupError, match="cancel_grace_s"):
        EncoderWorker(command=cmd, timeout_s=5.0, cancel_grace_s=bad)


def test_concurrent_puts_never_empty_the_cache(tmp_path):
    """max_entries=1, concurrent puts: every publish+evict is one
    transaction, so the LRU can never evict the whole cache away."""
    cache = EmbeddingCache(tmp_path / "embeds", max_entries=1)
    keys = [cache_key(make_request(prompt=f"p-{i}")) for i in range(4)]
    barrier = threading.Barrier(len(keys))

    def go(k: str) -> None:
        barrier.wait()
        cache.put(k, EncodedPayload(blobs=(Blob("e", k.encode()),)))

    threads = [threading.Thread(target=go, args=(k,)) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    entries = cache.entries()
    assert len(entries) == 1, "LRU race emptied the cache or exceeded the cap"
    assert entries[0] in keys


def test_instances_on_same_root_share_locks(tmp_path):
    root = tmp_path / "embeds"
    c1 = EmbeddingCache(root)
    c2 = EmbeddingCache(root / ".")  # same canonical path, different spelling
    c3 = EmbeddingCache(tmp_path / "other")
    assert c1.lock_for("a" * 64) is c2.lock_for("a" * 64), \
        "same canonical root must share per-key locks"
    assert c1.lock_for("a" * 64) is not c3.lock_for("a" * 64)


def test_two_cache_instances_same_root_encode_once(tmp_path):
    counter = tmp_path / "encodes.log"
    root = tmp_path / "embeds"
    kwargs = {"_counter_file": str(counter), "_sleep": 0.4}
    req = make_request(kwargs=kwargs)
    workers = []
    for i in (1, 2):
        scratch = tmp_path / f"scratch{i}"
        scratch.mkdir()
        workers.append(EncoderWorker(
            command=[sys.executable, str(FAKE_WORKER), "--request", "{request}",
                     "--response", "{response}"],
            timeout_s=20.0, cache=EmbeddingCache(root), scratch_dir=scratch))
    results: list = []
    errors: list = []
    barrier = threading.Barrier(2)

    def go(w):
        try:
            barrier.wait()
            results.append(w.encode(req))
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(w,)) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors
    assert len(counter.read_text().splitlines()) == 1, \
        "two cache instances on one root double-encoded"
    assert len(results) == 2
    for i in (1, 2):
        assert list((tmp_path / f"scratch{i}").iterdir()) == []


def test_keyboard_interrupt_during_wait_reaps_child(tmp_path, monkeypatch):
    """An interrupted poll loop must still terminate/reap the live child
    before the scratch dir disappears."""
    pid_file = tmp_path / "worker.pid"
    worker, kwargs = make_worker(tmp_path, timeout_s=30.0,
                                 extra_kwargs={"_sleep": 30,
                                               "_pid_file": str(pid_file)})
    calls: list = []
    real_sleep = time.sleep

    def boom(_seconds):
        calls.append(_seconds)
        # Let the child actually start (and record its pid) before the
        # simulated interrupt, so the reap is verified against a live child.
        end = time.monotonic() + 5.0
        while not pid_file.exists() and time.monotonic() < end:
            real_sleep(0.02)
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr("qwen_engine.encoder.time.sleep", boom)
        with pytest.raises(KeyboardInterrupt):
            worker.run(make_request(kwargs=kwargs))
    assert calls, "the wait was not actually interrupted"
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("worker child still alive after interrupted wait")
    assert_scratch_clean(tmp_path)


def test_setup_failures_are_typed_and_clean_up(tmp_path, monkeypatch):
    import qwen_engine.encoder as enc

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    worker = EncoderWorker(
        command=[sys.executable, str(FAKE_WORKER), "--request", "{request}",
                 "--response", "{response}"],
        timeout_s=10.0, scratch_dir=scratch)

    def boom_mkdtemp(*a, **k):
        raise OSError("no space left on device")

    with monkeypatch.context() as m:
        m.setattr(enc.tempfile, "mkdtemp", boom_mkdtemp)
        with pytest.raises(EncoderSetupError, match="scratch"):
            worker.run(make_request())

    def boom_write(self, data):
        raise OSError("read-only file system")

    with monkeypatch.context() as m:
        m.setattr(Path, "write_bytes", boom_write)
        with pytest.raises(EncoderSetupError, match="request file"):
            worker.run(make_request())

    real_mkdtemp = enc.tempfile.mkdtemp

    def squat_mkdtemp(*a, **k):  # a directory squatting on the stdout log
        d = real_mkdtemp(*a, **k)
        (Path(d) / "stdout.log").mkdir()
        return d

    with monkeypatch.context() as m:
        m.setattr(enc.tempfile, "mkdtemp", squat_mkdtemp)
        with pytest.raises(EncoderSetupError, match="stdout"):
            worker.run(make_request())
    assert list(scratch.iterdir()) == [], "setup failures must clean scratch"


def test_protocol_failure_event_carries_stderr(tmp_path):
    events = []
    from qwen_engine.log import PhaseLogger
    worker, kwargs = make_worker(
        tmp_path, logger=PhaseLogger(sink=events.append),
        extra_kwargs={"_no_response": True, "_stderr": "died quietly"})
    with pytest.raises(EncoderProtocolError) as exc:
        worker.run(make_request(kwargs=kwargs))
    assert "died quietly" in exc.value.stderr, "typed failure must keep stderr"
    failed = [json.loads(line) for line in events
              if json.loads(line)["event"] == "encoder_worker_failed"]
    assert failed, "a failure branch must emit a terminal event"
    assert "died quietly" in failed[0]["stderr"]
    assert failed[0]["error"] == "EncoderProtocolError"
    assert_scratch_clean(tmp_path)


def test_error_result_emits_terminal_event(tmp_path):
    events = []
    from qwen_engine.log import PhaseLogger
    worker, kwargs = make_worker(
        tmp_path, logger=PhaseLogger(sink=events.append),
        extra_kwargs={"_worker_error": {"type": "OOM", "message": "vram gone"},
                      "_stderr": "note from stderr"})
    with pytest.raises(EncoderWorkerError):
        worker.run(make_request(kwargs=kwargs))
    failed = [json.loads(line) for line in events
              if json.loads(line)["event"] == "encoder_worker_failed"]
    assert failed and "note from stderr" in failed[0]["stderr"]


def test_dead_log_sink_does_not_mask_failure(tmp_path):
    from qwen_engine.log import PhaseLogger

    def dead_sink(line):
        raise RuntimeError("the log sink is dead")

    worker, kwargs = make_worker(
        tmp_path, logger=PhaseLogger(sink=dead_sink),
        extra_kwargs={"_fail_exit": 4, "_stderr": "real problem"})
    with pytest.raises(EncoderWorkerError) as exc:
        worker.run(make_request(kwargs=kwargs))
    assert "real problem" in exc.value.stderr
    assert_scratch_clean(tmp_path)

    worker2, kwargs2 = make_worker(
        tmp_path, cache=EmbeddingCache(tmp_path / "embeds"),
        logger=PhaseLogger(sink=dead_sink))
    payload = worker2.encode(make_request(kwargs=kwargs2))
    assert payload.blobs[0].data.startswith(b"FAKEEMBED:"), \
        "dead cache hit/miss events must not break encode"


@pytest.mark.parametrize("patch", [
    {"surprise": 1},
    {"status": "error", "error": {"message": "boom", "surprise": 2}},
    {"blobs": [{"name": "b", "data_b64": "AA==", "surprise": True}]},
])
def test_unknown_result_fields_are_protocol_errors(tmp_path, patch):
    worker, kwargs = make_worker(tmp_path, extra_kwargs={"_result_patch": patch})
    with pytest.raises(EncoderProtocolError, match="unknown"):
        worker.run(make_request(kwargs=kwargs))
    assert_scratch_clean(tmp_path)


def test_nonfinite_metadata_in_response_is_protocol_error(tmp_path):
    worker, kwargs = make_worker(tmp_path, extra_kwargs={"_nan_metadata": True})
    with pytest.raises(EncoderProtocolError, match="finite"):
        worker.run(make_request(kwargs=kwargs))
    assert_scratch_clean(tmp_path)


def test_entry_nonfinite_or_unknown_key_is_miss_and_quarantined(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeds")
    key = cache_key(make_request())
    cache.entry_path(key).parent.mkdir(parents=True, exist_ok=True)
    nan_header = json.dumps({"schema": ENTRY_SCHEMA, "key": key,
                             "metadata": {"n": float("nan")}, "blobs": []},
                            sort_keys=True)
    cache.entry_path(key).write_bytes(nan_header.encode("utf-8") + b"\n")
    assert cache.get(key) is None
    assert len(list(cache.quarantine_dir.iterdir())) == 1
    extra_header = json.dumps({"schema": ENTRY_SCHEMA, "key": key,
                               "metadata": {}, "blobs": [], "surprise": True},
                              sort_keys=True)
    cache.entry_path(key).write_bytes(extra_header.encode("utf-8") + b"\n")
    assert cache.get(key) is None
    assert len(list(cache.quarantine_dir.iterdir())) == 2


# ---------------------------------------------------------------------------
# Boundary: stdlib only
# ---------------------------------------------------------------------------


def test_encoder_import_pulls_in_no_heavy_modules():
    import qwen_engine.encoder  # noqa: F401

    assert not set(HEAVY_MODULES) & set(sys.modules)


def test_encoder_import_in_subprocess_without_heavy_modules():
    code = (
        "import sys, qwen_engine.encoder;"
        "bad=[m for m in sys.modules if m.split('.')[0] in "
        f"{HEAVY_MODULES}];"
        "sys.exit(1 if bad else 0)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


# ---------------------------------------------------------------------------
# Second senior-review correction pass: cycles, overflow, quarantine race
# ---------------------------------------------------------------------------


def test_kwargs_cycle_is_typed_not_recursion_error():
    inner: dict = {}
    inner["self"] = inner
    with pytest.raises(EncoderRequestError, match=r"kwargs.*cyclic"):
        make_request(kwargs={"a": [inner]})


def test_payload_metadata_cycle_is_typed_not_recursion_error():
    loop: list = [1]
    loop.append(loop)
    with pytest.raises(EncoderRequestError, match=r"metadata.*cyclic"):
        EncodedPayload(metadata={"l": loop})


@pytest.mark.parametrize("bad", [10 ** 400, -(10 ** 400)])
def test_overflowing_timeouts_are_setup_errors_not_overflow_error(bad):
    """math.isfinite(10**400) raises OverflowError; the constructor must
    report a typed setup error instead."""
    cmd = ["prog", "{request}", "{response}"]
    with pytest.raises(EncoderSetupError, match="timeout_s"):
        EncoderWorker(command=cmd, timeout_s=bad)
    with pytest.raises(EncoderSetupError, match="cancel_grace_s"):
        EncoderWorker(command=cmd, timeout_s=5.0, cancel_grace_s=bad)


@pytest.mark.parametrize("bad_command", [5, "not-a-list", ("ok", 7)])
def test_command_invalid_types_are_typed_setup_errors(bad_command):
    with pytest.raises(EncoderSetupError, match="command"):
        EncoderWorker(command=bad_command, timeout_s=1.0)


@pytest.mark.parametrize("bad_root", [5, None, ["a"]])
def test_cache_root_invalid_types_are_typed(tmp_path, bad_root):
    with pytest.raises(EncoderRequestError, match="cache root"):
        EmbeddingCache(bad_root)


def test_worker_python_override_must_be_a_string(tmp_path):
    with pytest.raises(EncoderSetupError, match="python"):
        EncoderWorker(command=["prog", "{request}", "{response}"],
                      timeout_s=1.0, python=5)


def test_stale_corrupt_read_cannot_quarantine_a_freshly_published_entry(
        tmp_path, monkeypatch):
    """Deterministic form of the confirmed get() race: the corrupt read
    finishes its parse AFTER a valid entry was published. The quarantine
    must identity-check under the root mutex and leave the NEW entry in
    place. Events coordinate the interleaving; the reader holds NO lock
    while paused, so the publisher cannot deadlock on the new mutex."""
    import qwen_engine.encoder as encoder_mod

    cache = EmbeddingCache(tmp_path / "embeds")
    key = "a" * 64
    path = cache.entry_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"corrupt-header-line\nleftover")

    good = EncodedPayload(metadata={"m": 1}, blobs=(Blob("e", b"new-valid"),))
    parsed = threading.Event()
    published = threading.Event()
    outcomes: list = []

    real_payload_cls = EncodedPayload

    class _StaleParse:
        """Stands in for the parse of the STALE bytes only: it signals the
        test, waits until the replacement publish is done, then reports the
        (stale) bytes corrupt — which is when quarantine would normally
        fire."""

        @staticmethod
        def from_entry_bytes(data, k):
            assert data.startswith(b"corrupt-header-line"), \
                "shim only expected to see the stale read"
            parsed.set()
            assert published.wait(10), "publisher never ran"
            raise encoder_mod._EntryCorrupt("stale parse")

    monkeypatch.setattr(encoder_mod, "EncodedPayload", _StaleParse)

    def reader() -> None:
        outcomes.append(cache.get(key))

    t = threading.Thread(target=reader)
    t.start()
    assert parsed.wait(10), "reader never reached the parse"
    cache.put(key, good)  # must not block: the reader holds no lock here
    published.set()
    t.join(10)
    assert not t.is_alive()

    assert outcomes == [None], "stale read must report a miss"
    quarantine = cache.quarantine_dir
    moved = list(quarantine.iterdir()) if quarantine.is_dir() else []
    assert moved == [], f"the NEW entry was quarantined: {moved}"
    assert path.is_file(), "the newly published entry must survive"

    monkeypatch.undo()
    hit = cache.get(key)
    assert hit is not None
    assert hit.blobs[0].data == b"new-valid"


def test_corrupt_quarantine_still_works_after_identity_check(tmp_path):
    """Regression guard the other way: the SAME corrupt file (nothing
    replaced it) is still a miss AND still quarantined."""
    cache = EmbeddingCache(tmp_path / "embeds")
    key = "c" * 64
    cache.entry_path(key).parent.mkdir(parents=True, exist_ok=True)
    cache.entry_path(key).write_bytes(b"corrupt\nx")
    assert cache.get(key) is None
    assert len(list(cache.quarantine_dir.iterdir())) == 1


# ---------------------------------------------------------------------------
# Cache ownership: links / squats / foreign ownership are refused as misses
# ---------------------------------------------------------------------------


def _touch_fixed_mtime(path: Path) -> None:
    path.write_bytes(b"EXTERNAL-BYTES-do-not-touch")
    os.utime(path, (1_000_000_000, 1_000_000_000))


def test_external_symlink_entry_is_miss_and_target_untouched(tmp_path):
    """A symlink squatting on an entry name is refused WITHOUT ever being
    read or touched: the external target's bytes and mtime cannot move
    through a cache path; only the link itself is moved aside by name."""
    cache = EmbeddingCache(tmp_path / "embeds")
    external = tmp_path / "external.bin"
    _touch_fixed_mtime(external)
    before = external.stat()
    key = "d" * 64
    entry = cache.entry_path(key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.symlink_to(external)
    assert cache.get(key) is None, "symlink squat must be a miss"
    after = external.stat()
    assert external.read_bytes() == b"EXTERNAL-BYTES-do-not-touch"
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns,
                                                  before.st_size), \
        "external mtime/size must be untouched"
    assert not entry.exists() and not entry.is_symlink()
    moved = list(cache.quarantine_dir.iterdir())
    assert len(moved) == 1 and os.path.islink(moved[0])


def test_external_hardlink_entry_is_miss_and_target_untouched(tmp_path):
    """A second hard link on the entry name (nlink > 1) is refused as a
    miss: never read, never touched — only the extra link is moved aside,
    so the external file's bytes, mtime and content never change."""
    cache = EmbeddingCache(tmp_path / "embeds")
    external = tmp_path / "external.bin"
    _touch_fixed_mtime(external)
    before = external.stat()
    key = "e" * 64
    entry = cache.entry_path(key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    os.link(external, entry)
    assert cache.get(key) is None, "multiply linked squat must be a miss"
    after = external.stat()
    assert external.read_bytes() == b"EXTERNAL-BYTES-do-not-touch"
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_size == before.st_size
    assert not entry.exists(), "the extra link must leave the entry name"
    moved = list(cache.quarantine_dir.iterdir())
    assert len(moved) == 1
    # Move-by-name: the squat was renamed, never read, copied or touched —
    # the quarantined name still shares the external inode (the cache never
    # duplicates foreign content), and no write ever went through it.
    assert moved[0].stat().st_ino == after.st_ino
    assert moved[0].read_bytes() == b"EXTERNAL-BYTES-do-not-touch"


def test_external_quarantine_dir_symlink_is_refused_evidence_stays(tmp_path):
    """root/quarantine squatting as a symlink to an external directory is
    refused: nothing is mkdir'd through it, nothing is moved into it — the
    corrupt evidence stays at the entry name (still a miss, still
    listable), and the external directory stays empty."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    cache = EmbeddingCache(tmp_path / "embeds")
    cache.root.mkdir(parents=True)
    cache.quarantine_dir.symlink_to(outside)
    key = "f" * 64
    cache.entry_path(key).write_bytes(b"corrupt-header\njunk")
    assert cache.get(key) is None
    assert cache.entry_path(key).is_file(), \
        "a refused quarantine move must leave the evidence in place"
    assert cache.entry_path(key).read_bytes() == b"corrupt-header\njunk"
    assert list(outside.iterdir()) == [], "nothing may be moved outside"
    assert list(cache.entries()) == [key], \
        "the cache file listing stays accurate with the squat in place"


def test_fifo_entry_squat_is_refused_without_blocking(tmp_path):
    """A FIFO on the entry name is refused as a miss immediately: the
    cache never opens it for a blocking read (lstat branch first,
    O_NONBLOCK as the backstop on the descriptor)."""
    cache = EmbeddingCache(tmp_path / "embeds")
    key = "a1" * 32
    entry = cache.entry_path(key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(entry)
    started = time.monotonic()
    assert cache.get(key) is None
    assert time.monotonic() - started < 5.0, "get() must not block on a FIFO"
    assert not entry.exists()
    assert len(list(cache.quarantine_dir.iterdir())) == 1


def test_foreign_owned_entry_is_refused_as_miss(tmp_path, monkeypatch):
    """Ownership rule: a regular entry not owned by this user is a clean
    miss with the evidence and the name PRESERVED — no read, no touch, no
    rename — so generic OSError routing can never move a refused foreign
    file into quarantine."""
    import qwen_engine.encoder as encoder_mod

    cache = EmbeddingCache(tmp_path / "embeds")
    key = "b1" * 32
    cache.entry_path(key).parent.mkdir(parents=True, exist_ok=True)
    cache.entry_path(key).write_bytes(b"someone-elses-file")
    before = cache.entry_path(key).stat()
    real_geteuid = os.geteuid
    monkeypatch.setattr(encoder_mod.os, "geteuid",
                        lambda: real_geteuid() + 1000)
    assert cache.get(key) is None, "foreign-owned entry must be a miss"
    assert cache.entry_path(key).is_file(), \
        "a refused foreign-owned entry must be preserved in place"
    assert cache.entry_path(key).read_bytes() == b"someone-elses-file"
    after = cache.entry_path(key).stat()
    assert (after.st_ino, after.st_mtime_ns, after.st_nlink) == \
        (before.st_ino, before.st_mtime_ns, before.st_nlink), \
        "a refused foreign-owned entry must not be touched or renamed"
    assert not cache.quarantine_dir.exists() or \
        list(cache.quarantine_dir.iterdir()) == [], \
        "the refusal must not move anything into quarantine"


@pytest.mark.parametrize("nlink", [1, 2], ids=["single-link", "multi-link"])
def test_foreign_owned_regular_refused_at_any_link_count(tmp_path, monkeypatch,
                                                         nlink):
    """Ownership BEFORE link shape (root's extended
    /tmp/v100-cache-ownership-review.py follow-up): a foreign-owned
    REGULAR entry is a preserved miss at nlink 1 AND nlink 2 — the
    multi-link case must not be routed into the name-only quarantine
    move. Same bytes / mtime / nlink before and after; nothing moves."""
    import qwen_engine.encoder as encoder_mod

    cache = EmbeddingCache(tmp_path / "embeds")
    key = "b2" * 32
    cache.entry_path(key).parent.mkdir(parents=True, exist_ok=True)
    entry = cache.entry_path(key)
    if nlink == 2:
        twin = tmp_path / "foreign-twin"
        twin.write_bytes(b"someone-elses-multi-linked-file")
        os.link(twin, entry)
    else:
        entry.write_bytes(b"someone-elses-multi-linked-file")
    os.utime(entry, (1234, 5678))
    before = entry.stat()
    real_geteuid = os.geteuid
    monkeypatch.setattr(encoder_mod.os, "geteuid",
                        lambda: real_geteuid() + 1000)
    assert cache.get(key) is None, "foreign-owned entry must be a miss"
    after = entry.stat()
    assert entry.is_file() and entry.read_bytes() == \
        b"someone-elses-multi-linked-file"
    assert (after.st_ino, after.st_mtime_ns, after.st_nlink) == \
        (before.st_ino, before.st_mtime_ns, before.st_nlink), \
        "a foreign-owned regular entry must keep bytes/mtime/nlink at " \
        "ANY link count — the refusal never becomes a quarantine move"
    assert not cache.quarantine_dir.exists() or \
        list(cache.quarantine_dir.iterdir()) == [], \
        "the link-shape branch must never move a foreign-owned regular name"


def test_lru_linked_name_preserved_and_valid_entries_still_evicted(tmp_path):
    """Regression (root's /tmp/v100-cache-ownership-review.py): LRU counts,
    orders and evicts only regular, singly linked, current-user-owned
    entries by NO-FOLLOW stat. An entry-shaped hard link onto an external
    file is never unlinked (the outside inode's nlink and the linked name
    are preserved) and never uses up the valid-entry budget; ordinary LRU
    over valid entries — evicting the oldest, protecting the new — still
    works under the cap."""
    payload = EncodedPayload(blobs=(Blob("x", b"a"),))
    cache = EmbeddingCache(tmp_path / "embeds", max_entries=1)
    cache.root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(payload.to_entry_bytes("a" * 64))
    os.utime(outside, (1, 1))  # looks like the oldest entry by mtime
    link = cache.entry_path("a" * 64)
    os.link(outside, link)
    before = outside.stat()
    cache.put("b" * 64, payload)
    after = outside.stat()
    assert after.st_nlink == before.st_nlink == 2, \
        "LRU eviction must never decrement an external inode's nlink"
    assert link.exists(), "an unsafe entry-shaped name is not deleted as an eviction"
    assert cache.get("b" * 64) is not None, \
        "the unsafe name must not use up the valid-entry budget"
    # Ordinary LRU stays ordinary: the cap now holds one valid entry, so
    # the next valid publish evicts the older valid one — and only it.
    cache.put("c" * 64, payload)
    assert cache.get("c" * 64) is not None, "the new entry is protected"
    assert cache.get("b" * 64) is None, "the older valid entry is evicted"
    assert link.exists() and outside.stat().st_nlink == 2, \
        "the linked name and external nlink survive ordinary eviction too"


# ---------------------------------------------------------------------------
# Cache publication: put validates the EXISTING TARGET shape before writing
# ---------------------------------------------------------------------------


def _fake_foreign(monkeypatch, flag=None):
    """Make the module see the current user as one uid away (fake UID only —
    no privileged chown). With ``flag`` the switch is deferred until the
    flag dict sets ``flag['foreign']``."""
    import qwen_engine.encoder as encoder_mod

    real_geteuid = os.geteuid
    if flag is None:
        monkeypatch.setattr(encoder_mod.os, "geteuid",
                            lambda: real_geteuid() + 1000)
    else:
        monkeypatch.setattr(encoder_mod.os, "geteuid",
                            lambda: real_geteuid()
                            + (1000 if flag.get("foreign") else 0))


def _assert_target_preserved(entry, before, nlink=None):
    after = entry.lstat()
    assert (after.st_ino, after.st_mtime_ns, after.st_nlink) == \
        (before.st_ino, before.st_mtime_ns,
         before.st_nlink if nlink is None else nlink), \
        "a refused publish must preserve the target's inode / mtime / nlink"


def _assert_no_new_artifacts(cache, allowed_names):
    names = sorted(p.name for p in cache.root.iterdir())
    assert names == sorted(allowed_names), \
        "a refused publish must create no temp file and no quarantine dir"


@pytest.mark.parametrize("shape", ["foreign_single", "foreign_multi",
                                   "own_multi", "symlink", "fifo"])
def test_put_refuses_unsafe_target_shapes_preserving_everything(
        tmp_path, monkeypatch, shape):
    """Publication defect (root's /tmp/v100-cache-publish-review.py): an
    unconditional ``os.replace`` destroyed preserved foreign names and
    decremented external inodes' nlink on the ordinary get-miss-then-put
    path. Now ``put`` lstats the target under the root mutex BEFORE any
    temp file exists and refuses every unsafe shape (foreign-owned at any
    link count, own multiply linked, symlink, nonregular) with a typed
    EncoderError — name, bytes, inode, mtime and links preserved, no temp,
    no quarantine move, no unlink."""
    import qwen_engine.encoder as encoder_mod

    cache = EmbeddingCache(tmp_path / "embeds")
    cache.root.mkdir()
    key = "d4" * 32
    entry = cache.entry_path(key)
    if shape in ("foreign_single", "own_multi"):
        entry.write_bytes(b"target-bytes-do-not-destroy")
        if shape == "own_multi":
            os.link(entry, tmp_path / "twin")
    elif shape == "foreign_multi":
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"target-bytes-do-not-destroy")
        os.link(outside, entry)
    elif shape == "symlink":
        external = tmp_path / "external.bin"
        _touch_fixed_mtime(external)
        entry.symlink_to(external)
    else:
        os.mkfifo(entry)
    os.utime(entry, (1234, 5678), follow_symlinks=False)
    before = entry.lstat()
    if shape.startswith("foreign"):
        _fake_foreign(monkeypatch)
    with pytest.raises(EncoderError, match="refusing to publish"):
        cache.put(key, EncodedPayload(blobs=(Blob("x", b"new"),)))
    monkeypatch.undo()
    if shape == "fifo":
        assert stat.S_ISFIFO(entry.lstat().st_mode), \
            "the refused nonregular squat stays exactly where it is"
    elif shape == "symlink":
        assert entry.is_symlink(), "the refused symlink stays in place"
        external = tmp_path / "external.bin"
        assert external.read_bytes() == b"EXTERNAL-BYTES-do-not-touch", \
            "the symlink's external target must be untouched"
    else:
        assert entry.is_file() and entry.read_bytes() == \
            b"target-bytes-do-not-destroy"
        if "multi" in shape:
            twin = (tmp_path / "twin" if shape == "own_multi"
                    else tmp_path / "outside.bin")
            assert twin.stat().st_nlink == 2, \
                "a refused publish must not decrement an external link"
    _assert_target_preserved(entry, before)
    _assert_no_new_artifacts(cache, [entry.name])


def test_put_replaces_safe_owned_entry_and_publishes_when_absent(tmp_path):
    """Ordinary overwrite is UNCHANGED: an owned, regular, singly linked
    target is replaced atomically, and an absent target is a normal new
    publish."""
    cache = EmbeddingCache(tmp_path / "embeds")
    key = "d5" * 32
    cache.put(key, EncodedPayload(blobs=(Blob("x", b"first"),)))
    entry = cache.entry_path(key)
    assert cache.get(key).blobs[0].data == b"first"
    cache.put(key, EncodedPayload(blobs=(Blob("x", b"second"),)))
    hit = cache.get(key)
    assert hit is not None and hit.blobs[0].data == b"second", \
        "owned single-link overwrite stays ordinary"
    assert entry.stat().st_nlink == 1
    other = "d6" * 32
    cache.put(other, EncodedPayload(blobs=(Blob("x", b"new"),)))
    assert cache.get(other) is not None, "absent target: normal new publish"


def test_worker_get_miss_then_put_refuses_to_destroy_foreign_entry(
        tmp_path, monkeypatch):
    """The actual normal-workflow defect: ``get()`` correctly preserved a
    foreign regular name on miss, but the encode path then ran the worker
    and its ``put()`` replaced that same preserved name. Through the fake
    worker's get→miss→put path the publication now ends in a typed
    refusal with the foreign entry fully preserved and no temp or
    quarantine artifacts."""
    import qwen_engine.encoder as encoder_mod

    cache = EmbeddingCache(tmp_path / "embeds")
    request = make_request()
    key = cache_key(request)
    entry = cache.entry_path(key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_bytes(b"someone-elses-entry")
    os.utime(entry, (1234, 5678))
    before = entry.stat()

    class _FakeWorker(encoder_mod.EncoderWorker):
        def run(self, *args, **kwargs):  # fake transport: no child
            return EncodedPayload(blobs=(Blob("x", b"fresh"),))

    worker = _FakeWorker(command=["{python}", "{request}", "{response}"],
                         timeout_s=15.0, cache=cache)
    _fake_foreign(monkeypatch)
    with pytest.raises(EncoderError, match="refusing to publish"):
        worker.encode(request)
    monkeypatch.undo()
    assert entry.read_bytes() == b"someone-elses-entry"
    _assert_target_preserved(entry, before)
    _assert_no_new_artifacts(cache, [entry.name])


def test_corrupt_read_that_became_foreign_after_read_is_not_moved(
        tmp_path, monkeypatch):
    """Quarantine boundary: the verified-read identity tuple omits the uid,
    so the regular-file ownership refusal must run BEFORE the
    ``read_ident`` branch — a regular file that became foreign-owned after
    the corrupt read is preserved even though the identity still matches.
    (Fake UID change at parse time; no privileged chown.)"""
    import qwen_engine.encoder as encoder_mod

    cache = EmbeddingCache(tmp_path / "embeds")
    key = "d7" * 32
    entry = cache.entry_path(key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_bytes(b"corrupt-header\njunk")
    os.utime(entry, (1234, 5678))
    before = entry.stat()

    flag = {"foreign": False}
    _fake_foreign(monkeypatch, flag)
    real_from = encoder_mod.EncodedPayload.from_entry_bytes

    def parse_then_ownership_flips(data, k):
        try:
            return real_from(data, k)  # raises _EntryCorrupt on these bytes
        finally:
            flag["foreign"] = True  # after the verified read completes

    monkeypatch.setattr(encoder_mod.EncodedPayload, "from_entry_bytes",
                        staticmethod(parse_then_ownership_flips))
    assert cache.get(key) is None, "a corrupt read is still a miss"
    assert entry.is_file(), \
        "the verified-identity path must never move a now-foreign regular file"
    assert entry.read_bytes() == b"corrupt-header\njunk"
    _assert_target_preserved(entry, before, nlink=1)
    assert not cache.quarantine_dir.exists() or \
        list(cache.quarantine_dir.iterdir()) == [], \
        "the refused quarantine move must not have happened"


def test_put_with_file_as_root_is_typed_not_raw_file_exists_error(tmp_path):
    """Persistence boundary: root creation failures surface as typed
    EncoderError; the ordinary file squatting on the root path is NOT
    deleted or mkdir-repaired."""
    root = tmp_path / "embeds"
    root.write_bytes(b"i-am-a-file-not-a-dir")
    cache = EmbeddingCache(root)
    with pytest.raises(EncoderError, match="cache root creation failed"):
        cache.put("a" * 64, EncodedPayload(blobs=(Blob("x", b"1"),)))
    assert root.is_file() and root.read_bytes() == b"i-am-a-file-not-a-dir", \
        "the refusal must not delete or repair user data"


def test_failed_temp_cleanup_does_not_replace_publish_failure(tmp_path,
                                                              monkeypatch):
    """Persistence boundary: when the temp cleanup ALSO raises, the primary
    typed publish failure (and its OSError cause) is what surfaces."""
    import qwen_engine.encoder as encoder_mod

    cache = EmbeddingCache(tmp_path / "embeds")
    cache.root.mkdir()

    def replace_boom(*args, **kwargs):
        raise OSError("injected publish failure")

    def unlink_boom(self, missing_ok=False):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(encoder_mod.os, "replace", replace_boom)
    monkeypatch.setattr(Path, "unlink", unlink_boom)
    with pytest.raises(EncoderError, match="cache write failed") as excinfo:
        cache.put("a" * 64, EncodedPayload(blobs=(Blob("x", b"1"),)))
    assert isinstance(excinfo.value.__cause__, OSError)
    assert "injected publish failure" in str(excinfo.value.__cause__), \
        "the cleanup error must not mask the publish failure"


def test_failed_exclusive_temp_open_preserves_preexisting_temp(tmp_path,
                                                               monkeypatch):
    """Cleanup invariant: if the exclusive ``xb`` open fails because the
    selected temp NAME already exists, that pre-existing file is somebody
    else's — the typed publish failure must not unlink it. Cleanup touches
    only a temp this call actually created. (Deterministic name collision
    via a tiny fake uuid4; static fixture, no concurrent writer.)"""
    import qwen_engine.encoder as encoder_mod

    cache = EmbeddingCache(tmp_path / "embeds")
    cache.root.mkdir()
    key = "d8" * 32
    existing_temp = cache.root / f".{key}.{os.getpid()}.collision.tmp"
    existing_temp.write_bytes(b"keep-existing-temp")
    monkeypatch.setattr(
        encoder_mod.uuid, "uuid4",
        lambda: types.SimpleNamespace(hex="collision"))
    with pytest.raises(EncoderError, match="cache write failed"):
        cache.put(key, EncodedPayload(blobs=(Blob("x", b"new"),)))
    assert existing_temp.is_file() and \
        existing_temp.read_bytes() == b"keep-existing-temp", \
        "a failed O_EXCL open must not unlink the pre-existing temp name"
    assert not cache.entry_path(key).exists(), \
        "the failed publish published nothing"


# ---------------------------------------------------------------------------
# Cache entry size bounds: validation, exact cap, over-cap behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "1024", None,
                                 10 ** 100, sys.maxsize])
def test_max_entry_bytes_validation_is_typed(tmp_path, bad):
    """Positive non-bool int only, and cap + 1 must stay index-representable
    (sys.maxsize itself is refused; 10**100 never reaches a raw
    OverflowError at a read)."""
    with pytest.raises(EncoderRequestError, match="max_entry_bytes"):
        EmbeddingCache(tmp_path / "embeds", max_entry_bytes=bad)


def test_max_entry_bytes_maxsize_boundary_accepted(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeds",
                           max_entry_bytes=sys.maxsize - 1)
    assert cache.max_entry_bytes == sys.maxsize - 1


def _sized_payload():
    return EncodedPayload(metadata={"m": 1},
                          blobs=(Blob("prompt_embeds", b"BODY" * 50),))


def test_entry_exactly_at_cap_roundtrips(tmp_path):
    """The exact-cap boundary is a pass, not a refusal: declared size ==
    cap publishes, and the bounded read answers a hit."""
    key = cache_key(make_request())
    payload = _sized_payload()
    exact = len(payload.to_entry_bytes(key))
    cache = EmbeddingCache(tmp_path / "embeds", max_entry_bytes=exact)
    cache.put(key, payload)
    hit = cache.get(key)
    assert hit is not None
    assert bytes(hit.blobs[0].data) == bytes(payload.blobs[0].data)


def test_entry_over_cap_is_miss_quarantined_never_parsed(tmp_path,
                                                         monkeypatch):
    """One byte over the cap: pre-sized on the descriptor, so the oversized
    content is a corrupt miss — no oversized parser call, no unbounded
    (quarantine re-)read — and the SAME file is quarantined with its
    evidence bytes intact."""
    import qwen_engine.encoder as encoder_mod

    key = cache_key(make_request())
    payload = _sized_payload()
    entry_size = len(payload.to_entry_bytes(key))
    EmbeddingCache(tmp_path / "embeds").put(key, payload)  # full-size write

    parses: list[int] = []
    real_parse = EncodedPayload.from_entry_bytes

    def spy(data, k):
        parses.append(len(data))
        return real_parse(data, k)

    monkeypatch.setattr(EncodedPayload, "from_entry_bytes", spy)
    cache = EmbeddingCache(tmp_path / "embeds",
                           max_entry_bytes=entry_size - 1)
    assert cache.get(key) is None, "cap+1 bytes must be a miss"
    assert parses == [], "oversized entry bytes must never reach the parser"
    moved = list(cache.quarantine_dir.iterdir())
    assert len(moved) == 1
    assert len(moved[0].read_bytes()) == entry_size, \
        "quarantined evidence must be the whole entry, moved not truncated"


def test_put_over_cap_refuses_before_building_the_buffer(tmp_path,
                                                         monkeypatch):
    """A put whose entry would exceed the cap fails typed BEFORE the
    combined buffer exists: the declared-size check runs without ever
    concatenating the body (to_entry_bytes is never reached), and nothing
    is written — not even a temp file."""
    key = cache_key(make_request())
    payload = _sized_payload()
    entry_size = len(payload.to_entry_bytes(key))
    calls: list[int] = []
    real_build = EncodedPayload.to_entry_bytes

    def spy(self, k):
        calls.append(1)
        return real_build(self, k)

    monkeypatch.setattr(EncodedPayload, "to_entry_bytes", spy)
    cache = EmbeddingCache(tmp_path / "embeds",
                           max_entry_bytes=entry_size - 1)
    with pytest.raises(EncoderError, match="over max_entry_bytes"):
        cache.put(key, payload)
    assert calls == [], "an over-cap put must not construct the entry buffer"
    assert not cache.entry_path(key).exists()
    # The refusal happens before any filesystem mutation at all: no entry,
    # no temp file — the root may not even exist yet.
    assert not cache.root.exists() or [
        p for p in cache.root.iterdir() if p.name.endswith(".tmp")] == []


# ---------------------------------------------------------------------------
# Root canonicalization: resolve-once + CWD stability
# ---------------------------------------------------------------------------


def test_explicit_root_symlink_retarget_cannot_redirect_writes(tmp_path):
    """Regression (root's /tmp/v100-cache-root-snapshot-review.py): the
    root is resolved ONCE at construction, so retargeting an explicitly
    chosen root symlink afterwards cannot move cache writes into a new
    root; the ORIGINAL resolved root receives the entry, and instances on
    that actual root still share one mutation lock."""
    first, second, link = tmp_path / "first", tmp_path / "second", \
        tmp_path / "configured"
    first.mkdir()
    second.mkdir()
    link.symlink_to(first, target_is_directory=True)
    cache = EmbeddingCache(link)
    link.unlink()
    link.symlink_to(second, target_is_directory=True)
    key = "a" * 64
    cache.put(key, EncodedPayload(blobs=(Blob("x", b"a"),)))
    assert (first / (key + ".embed")).is_file(), \
        "the ORIGINAL resolved root must receive the entry"
    assert not (second / (key + ".embed")).exists(), \
        "the retargeted root must receive nothing"
    assert cache._state is EmbeddingCache(first)._state, \
        "instances on the actual root share the mutation lock"


def test_relative_cache_root_and_scratch_are_cwd_stable(tmp_path,
                                                        monkeypatch):
    """Relative configured paths are resolved at construction: a later
    chdir can neither redirect a cache write nor send the worker's scratch
    files to a same-named directory under the new CWD."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "rel-scratch").mkdir()
    monkeypatch.chdir(tmp_path)
    cache = EmbeddingCache("rel-cache")
    worker = EncoderWorker(
        command=[sys.executable, str(FAKE_WORKER), "--request", "{request}",
                 "--response", "{response}"],
        timeout_s=15.0, scratch_dir="rel-scratch")
    assert cache.root.is_absolute() and worker.scratch_dir.startswith(
        os.path.realpath(str(tmp_path)))
    assert cache.root == Path(os.path.realpath(str(tmp_path))) / "rel-cache"
    assert worker.scratch_dir == os.path.realpath(
        str(tmp_path / "rel-scratch"))
    monkeypatch.chdir(elsewhere)
    payload = worker.run(make_request())
    key = cache_key(make_request())
    cache.put(key, payload)
    assert (tmp_path / "rel-cache" / (key + ".embed")).is_file()
    assert not (elsewhere / "rel-cache").exists()
    assert not (elsewhere / "rel-scratch").exists()
    assert list(elsewhere.iterdir()) == []


# --- json parser safety limits at the entry/worker parse boundaries ---------


@pytest.mark.parametrize("value", ["9" * 5000, "[" * 2000 + "0" + "]" * 2000],
                         ids=["digit-limit", "deep-nesting"])
def test_parser_safety_limit_corrupt_entry_is_miss_and_quarantine(tmp_path,
                                                                  value):
    if value.startswith("9") and not getattr(
            sys, "get_int_max_str_digits", lambda: 0)():
        pytest.skip("integer-string digit limit disabled")
    cache = EmbeddingCache(tmp_path / "cache")
    cache.root.mkdir()
    key = "a" * 64
    header = ('{"schema": ' + json.dumps(ENTRY_SCHEMA) + ', "key": '
              + json.dumps(key) + ', "metadata": {"n": ' + value
              + '}, "blobs": []}\n')
    cache.entry_path(key).write_text(header)
    assert cache.get(key) is None  # typed miss, never a raw exception
    quarantined = list(cache.quarantine_dir.iterdir())
    assert len(quarantined) == 1 and ".corrupt" in quarantined[0].name
    assert not cache.entry_path(key).exists()


@pytest.mark.parametrize("boom", [
    ValueError("exceeds the integer-string digit limit"),
    RecursionError("decoder nesting limit"),
], ids=["injected-ValueError", "injected-RecursionError"])
def test_injected_parser_failures_stay_typed_in_cache_and_worker(
        tmp_path, monkeypatch, boom):
    def raiser(*args, **kwargs):
        raise boom

    monkeypatch.setattr("json.loads", raiser)
    with pytest.raises(_EntryCorrupt):
        EncodedPayload.from_entry_bytes(b'{"schema": 1}\n', "0" * 64)
    worker = EncoderWorker(command=["x", "{request}", "{response}"],
                           timeout_s=1, scratch_dir=tmp_path)
    resp = tmp_path / "response.json"
    resp.write_text('{"status": "ok"}')
    with pytest.raises(EncoderProtocolError) as exc:
        worker._result_from_files(_ExitedChild(), resp, "worker said hi",
                                  "0" * 64, "rid")
    assert "worker said hi" in exc.value.stderr  # diagnostics retained


@pytest.mark.parametrize("field", ["schema", "key", "metadata", "blob-bytes", "blob-sha"])
def test_duplicate_entry_json_is_corrupt_and_quarantined(tmp_path, field):
    cache = EmbeddingCache(tmp_path / "cache")
    key = cache_key(make_request())
    payload = EncodedPayload(metadata={"m": 1}, blobs=(Blob("e", b"x"),))
    head, _, body = payload.to_entry_bytes(key).partition(b"\n")
    if field in ("schema", "key"):
        head = ('{"' + field + '":"wrong",').encode() + head[1:]
    elif field == "metadata":
        head = head.replace(b'"m":1', b'"m":99,"m":1')
    elif field == "blob-bytes":
        head = head.replace(b'"bytes":1', b'"bytes":99,"bytes":1')
    else:
        head = head.replace(b'"sha256":', b'"sha256":"wrong","sha256":')
    raw = head + b"\n" + body
    with pytest.raises(_EntryCorrupt, match="duplicate"):
        EncodedPayload.from_entry_bytes(raw, key)
    cache.root.mkdir()
    cache.entry_path(key).write_bytes(raw)
    assert cache.get(key) is None
    assert not cache.entry_path(key).exists()
    assert next(cache.quarantine_dir.iterdir()).read_bytes() == raw


@pytest.mark.parametrize("field", ["schema", "request_id", "status", "metadata", "blob"])
def test_duplicate_response_json_is_typed_protocol_refusal(tmp_path, field):
    worker = EncoderWorker(command=["x", "{request}", "{response}"], timeout_s=1, scratch_dir=tmp_path)
    doc = {"schema": RESULT_SCHEMA, "request_id": "rid", "status": "ok",
           "metadata": {"m": 1}, "blobs": [{"name": "e", "data_b64": "eA=="}]}
    raw = json.dumps(doc, separators=(",", ":")).encode()
    if field in ("schema", "request_id", "status"):
        raw = ('{"' + field + '":"wrong",').encode() + raw[1:]
    elif field == "metadata":
        raw = raw.replace(b'"m":1', b'"m":99,"m":1')
    else:
        raw = raw.replace(b'"data_b64":', b'"data_b64":"AA==","data_b64":')
    path = tmp_path / "response.json"
    path.write_bytes(raw)
    with pytest.raises(EncoderProtocolError, match="duplicate") as caught:
        worker._result_from_files(_ExitedChild(), path, "diagnostic survived", "0" * 64, "rid")
    assert caught.value.stderr == "diagnostic survived"


def test_inactive_prompt_key_locks_do_not_accumulate(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache", max_entries=1)
    held = cache.lock_for("a" * 64)
    for i in range(10000):
        with cache.lock_for(f"{i:064x}"):
            pass
    gc.collect()
    assert list(cache._state.key_locks) == ["a" * 64]
    assert cache.lock_for("a" * 64) is held
    ref = weakref.ref(held)
    del held
    gc.collect()
    assert ref() is None and not cache._state.key_locks


def test_retained_key_lock_still_shared_after_cache_instance_dropped(tmp_path):
    root = tmp_path / "cache"
    cache = EmbeddingCache(root)
    held = cache.lock_for("a" * 64)
    del cache
    gc.collect()
    assert EmbeddingCache(root).lock_for("a" * 64) is held


def test_waiting_same_key_callers_keep_shared_lock_alive(tmp_path):
    cache, second = EmbeddingCache(tmp_path / "cache"), EmbeddingCache(tmp_path / "cache")
    held = cache.lock_for("a" * 64)
    held.acquire()
    entered, acquired = threading.Event(), threading.Event()
    def waiter():
        lock = second.lock_for("a" * 64)
        assert lock is held
        entered.set()
        with lock:
            acquired.set()
    thread = threading.Thread(target=waiter)
    thread.start()
    try:
        assert entered.wait(5)
        gc.collect()
        assert cache.lock_for("a" * 64) is held and not acquired.is_set()
    finally:
        held.release()
        thread.join(5)
    assert acquired.is_set() and not thread.is_alive()


@pytest.mark.parametrize("point", ["fsync", "replace"])
def test_cache_temp_cleanup_on_interrupt_preserves_old_entry(tmp_path, monkeypatch, point):
    cache = EmbeddingCache(tmp_path / "cache")
    key = cache_key(make_request())
    payload = EncodedPayload(blobs=(Blob("e", b"old"),))
    path = cache.put(key, payload)
    previous = path.read_bytes()
    error = KeyboardInterrupt()
    def interrupted(*a):
        raise error
    monkeypatch.setattr(os, point, interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        cache.put(key, EncodedPayload(blobs=(Blob("e", b"new"),)))
    assert caught.value is error
    assert path.read_bytes() == previous
    assert list(cache.root.iterdir()) == [path]


def test_large_payload_unique_names_checked_linearly():
    # Deterministic operation-count regression, not a timing-sensitive benchmark.
    class Name(str):
        comparisons = 0
        __hash__ = str.__hash__
        def __eq__(self, other):
            type(self).comparisons += 1
            return str.__eq__(self, other)
    blobs = tuple(Blob(Name(f"b{i}"), b"") for i in range(3000))
    payload = EncodedPayload(blobs=blobs)
    assert len(payload.blobs) == 3000 and Name.comparisons < 12000


@pytest.mark.parametrize("part", ["component", "implementation", "request", "model", "processor"])
def test_mixed_mapping_keys_are_typed_encoder_request_errors(part):
    data = make_request(implementation_identity=make_identity()).to_request_dict("rid")
    if part == "component":
        value = {"sha256": SHA_A, 1: "wrong", None: "wrong"}
        action = lambda: ComponentIdentity.from_dict(value)
    elif part == "implementation":
        value = dict(data["implementation_identity"])
        value[1], value[None] = "wrong", "wrong"
        action = lambda: EncoderImplementationIdentity.from_dict(value)
    else:
        value = data if part == "request" else data[part]
        value[1], value[None] = "wrong", "wrong"
        action = lambda: EncoderRequest.from_request_dict(data)
    with pytest.raises(EncoderRequestError, match="unknown"):
        action()


def test_worker_run_wires_image_files_into_the_child_document(tmp_path):
    # End-to-end v3 subprocess: EncoderWorker.run(..., image_files=...)
    # serializes the ordered paths into the wire document next to the
    # images hashes, and a generic worker consumes the document verbatim.
    req = make_request(images=(SHA_A,),
                       kwargs={"_echo_wire": True})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    worker = EncoderWorker(
        command=["{python}", str(FAKE_WORKER), "--request", "{request}",
                 "--response", "{response}"],
        timeout_s=15.0, scratch_dir=scratch)
    payload = worker.run(req, request_id="v3-img-1",
                         image_files=["/store/a.img"])
    assert payload.metadata["images"] == (SHA_A,)
    assert payload.metadata["image_files"] == ("/store/a.img",)
    assert_scratch_clean(tmp_path)
