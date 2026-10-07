"""M1.4 encoder worker + cache tests — fake worker only, no model (plan R5).

Covers: cache-key v2 semantics (negative-key fix both variants), atomic
writes, corrupt-entry miss + quarantine, LRU cap, per-key concurrency,
worker timeout / cancel (terminate → kill) / error + stderr capture /
protocol violations, scratch cleanup on every path, and the no-heavy-imports
boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from qwen_engine.encoder import (
    CACHE_KEY_SCHEMA,
    ENTRY_SCHEMA,
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
    Blob,
    ComponentIdentity,
    EmbeddingCache,
    EncoderCancelledError,
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
                grace=None, extra_kwargs=None) -> tuple[EncoderWorker, dict]:
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
        logger=logger,
    )
    return worker, kwargs


def assert_scratch_clean(tmp_path) -> None:
    assert list((tmp_path / "scratch").iterdir()) == []


# ---------------------------------------------------------------------------
# Schema tags + request validation
# ---------------------------------------------------------------------------


def test_schema_tags_are_versioned():
    assert REQUEST_SCHEMA == "qwen_engine.encoder_request.v1"
    assert RESULT_SCHEMA == "qwen_engine.encoder_result.v1"
    assert CACHE_KEY_SCHEMA == "qwen_engine.embed_cache_key.v2"
    assert ENTRY_SCHEMA == "qwen_engine.embed_entry.v1"


def test_request_roundtrip_is_lossless():
    req = make_request(negative_prompt="ugly", kwargs={"a": 1, "b": {"c": [2, 3]}},
                       images=(SHA_A,))
    doc = req.to_request_dict("rid-1")
    assert doc["schema"] == REQUEST_SCHEMA and doc["request_id"] == "rid-1"
    parsed, rid = EncoderRequest.from_request_dict(doc)
    assert rid == "rid-1"
    assert parsed.key_document() == req.key_document()


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
# Cache key v2
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
