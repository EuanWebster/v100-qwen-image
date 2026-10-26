"""Guarded R3 tests: the rewrite service (no sockets, no torch, no writes).

Every backend is an injected transport callable, so nothing here opens a
socket and no service is required. Concurrency tests use real threads, which
is the point: single flight, cancel and deadline are thread behaviour.
"""
from __future__ import annotations

import base64
import io
import json
import pathlib
import threading
import time

import pytest
from PIL import Image

from qwen_engine import reference_store
from qwen_engine import rewrite as rw
from qwen_engine import rewrite_client as rc
from qwen_engine.web import api
from qwen_engine.web import rewrite_service as rs

ENDPOINT = "http://127.0.0.1:8079/v1/chat/completions"
MODEL = "Qwen3.8"
PROMPTS = {"t2i": "system prompt for t2i", "edit": "system prompt for edit"}


def reply(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant",
                                                "content": content}}]}).encode()


def answer(prompt: str = "A red bicycle.", ratio: str | None = None) -> bytes:
    obj = {"rewritten_prompt": prompt}
    if ratio is not None:
        obj["wh_ratio"] = ratio
    return reply(json.dumps(obj))


def settings(**overrides) -> rs.RewriteSettings:
    base = dict(backend="local-llm", endpoint=ENDPOINT, model=MODEL,
                prompts=dict(PROMPTS), timeout=30.0)
    base.update(overrides)
    return rs.RewriteSettings(**base)


def service(transport, **overrides) -> rs.RewriteService:
    """`transport` is a callable, or a fixed response body for the common case."""
    body = transport if callable(transport) else (lambda _request, _raw=transport: _raw)
    return rs.RewriteService(settings(**overrides), transport=body)


def request(**overrides) -> dict:
    base = {"task": "t2i", "prompt": "a cat", "draft_revision": 7,
            "image_ids": []}
    base.update(overrides)
    return base


def code(exc_info) -> str:
    assert isinstance(exc_info.value, api.ApiError)
    return exc_info.value.code


def store_image(root, mode="RGB", size=(64, 48), colour=(200, 30, 30),
                fmt="PNG"):
    """Store a real image and return its id (the store is the real store)."""
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.new(mode, size, colour).save(buffer, fmt)
    return reference_store.put_bytes(root, buffer.getvalue())


def user_parts(log_entry: dict) -> list:
    return log_entry["messages"][-1]["content"]


def edit_service(tmp_path, transport, **overrides) -> tuple:
    """A service with a REAL store and the edit request that names one image.

    An edit rewrite legally requires at least one reference, so every edit-path
    test starts from a stored image rather than a request the server refuses.
    """
    root = tmp_path / "store"
    ref = store_image(root)
    body = transport if callable(transport) else (lambda _raw, _b=transport: _b)
    svc = rs.RewriteService(settings(references_dir=str(root), **overrides),
                            transport=body)
    return svc, request(task="edit", image_ids=[ref])


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------


def test_disabled_is_the_default_and_refuses_without_a_transport():
    service_off = rs.RewriteService(rs.RewriteSettings.disabled())
    assert service_off.available is False
    assert service_off.status_block()["code"] == "rewrite_disabled"
    assert service_off.status_block()["backend"] is None
    with pytest.raises(api.ApiError) as exc:
        service_off.submit(request())
    assert exc.value.status == 503 and code(exc) == "rewrite_disabled"


def test_disabled_settings_refuse_to_carry_a_backend_surface():
    """Setting an endpoint without enabling means the operator intended to."""
    for overrides in ({"endpoint": ENDPOINT}, {"model": MODEL},
                      {"prompts": dict(PROMPTS)}):
        with pytest.raises(rc.RewriteConfigError):
            rs.RewriteSettings(backend="off", **overrides)


def test_a_disabled_service_builds_no_client_at_all():
    service_off = rs.RewriteService(rs.RewriteSettings.disabled())
    assert service_off._clients == {}
    assert service_off.active_backend is None
    assert service_off.available_backends() == []


def test_status_block_carries_no_endpoint_or_path():
    block = service(answer()).status_block()
    # The closed shape the UI parser expects, plus the two keys the amendment
    # adds: which rewriters exist here, and the ROUTE their choice is written
    # to. An on-disk path never appears — the file lives in the cache dir and
    # is named on the operator's console instead.
    assert set(block) == {"state", "code", "backend", "reason", "backends",
                          "settings"}
    assert block["backend"] == "local-llm" and block["reason"] == ""
    assert [row["id"] for row in block["backends"]] == ["local-llm"]
    assert block["settings"] == {"href": api.REWRITE_SETTINGS_HREF,
                                 "saved": False}
    flat = json.dumps(block)
    assert "127.0.0.1" not in flat and "8079" not in flat and "http" not in flat


def test_disabled_reason_mentions_no_host_or_path():
    flat = json.dumps(rs.RewriteService(rs.RewriteSettings.disabled()).status_block())
    assert "127.0.0.1" not in flat and ":8079" not in flat


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overrides", [
    {"endpoint": "http://10.0.0.5:8079/v1"},      # not loopback
    {"endpoint": "https://127.0.0.1:8079/v1"},    # no TLS policy
    {"endpoint": "http://127.0.0.1/v1"},          # port must be explicit
    {"endpoint": "  http://127.0.0.1:8079/v1"},
    {"model": ""},
    {"model": None},
    {"timeout": 0},
    {"timeout": 10**6},
    {"timeout": "180"},
    {"prompts": {"t2i": "only one task"}},
    {"prompts": {"t2i": "  ", "edit": "edit prompt"}},
])
def test_bad_startup_settings_are_refused_immediately(overrides):
    with pytest.raises(rc.RewriteConfigError):
        settings(**overrides)


def test_unknown_backend_is_refused():
    with pytest.raises(rc.RewriteConfigError, match="unknown rewrite backend"):
        settings(backend="official-cpu")


def test_settings_are_immutable():
    frozen = settings()
    with pytest.raises(Exception):
        frozen.backend = "off"


# ---------------------------------------------------------------------------
# Happy path payload
# ---------------------------------------------------------------------------


def test_success_payload_has_exactly_the_contract_keys():
    response = service(answer("A red bicycle.", "16:9")).submit(request())
    assert response.status == 200
    payload = response.payload
    assert set(payload) == {
        "schema", "task", "draft_revision", "original_prompt",
        "rewritten_prompt", "parse_ok", "unparsed_answer", "ratio",
        "ratio_conflict", "backend", "elapsed_ms"}
    assert payload["schema"] == rs.WEB_REWRITE_SCHEMA
    assert payload["task"] == "t2i"
    assert payload["draft_revision"] == 7
    assert payload["original_prompt"] == "a cat"
    assert payload["rewritten_prompt"] == "A red bicycle."
    assert payload["parse_ok"] is True
    assert payload["unparsed_answer"] is None
    assert payload["ratio"] == {"kind": "wh_ratio", "value": "16:9"}
    assert payload["ratio_conflict"] is False
    assert payload["backend"] == "local-llm"
    assert isinstance(payload["elapsed_ms"], int) and payload["elapsed_ms"] >= 0


def test_the_v3_request_schema_tag_is_defined_here_for_the_route():
    assert rs.REWRITE_REQUEST_SCHEMA == "qwen_engine.web_rewrite_request.v1"
    assert rs.WEB_REWRITE_SCHEMA == "qwen_engine.web_rewrite.v1"


def test_prompt_survives_verbatim_including_unicode_and_newlines():
    draft = "  a cat\nwith ünïcode — and punctuation!  "
    payload = service(answer()).submit(request(prompt=draft)).payload
    assert payload["original_prompt"] == draft


def test_the_system_prompt_selected_matches_the_task(tmp_path):
    seen = {}

    def transport(body):
        seen["doc"] = json.loads(body)
        return answer()

    svc, edit = edit_service(tmp_path, transport)
    svc.submit(dict(edit, prompt="recolour it"))
    assert seen["doc"]["messages"][0]["content"] == PROMPTS["edit"]


def test_the_production_path_sends_the_whole_trained_profile():
    """The end-to-end body through RewriteService, field for field.

    This is the regression that mattered: the service used to transmit only
    temperature/top_p/max_tokens, so t2i ran at presence_penalty 0 instead of
    the trained 1.5. Asserting the exact document (not "contains") is what stops
    a future field from being dropped in silence.
    """
    seen = {}

    def transport(body):
        seen["doc"] = json.loads(body)
        return answer()

    service(transport).submit(request(task="t2i", prompt="a corgi in rain"))
    document = seen["doc"]
    assert document["temperature"] == 1.0
    assert document["top_p"] == 0.95
    assert document["top_k"] == 20
    assert document["min_p"] == 0.0
    assert document["presence_penalty"] == 1.5
    assert document["max_tokens"] == 16256
    assert document["chat_template_kwargs"] == {"enable_thinking": True}
    assert set(document) == {
        "model", "messages", "stream", "temperature", "top_p", "top_k",
        "min_p", "presence_penalty", "max_tokens", "chat_template_kwargs",
    }


def test_the_edit_path_sends_penalty_zero_and_its_own_budget(tmp_path):
    seen = {}

    def transport(body):
        seen["doc"] = json.loads(body)
        return answer()

    svc, edit = edit_service(tmp_path, transport)
    svc.submit(dict(edit, prompt="recolour it"))
    document = seen["doc"]
    assert document["presence_penalty"] == 0.0
    assert document["max_tokens"] == 24000
    assert document["top_k"] == 20
    assert document["min_p"] == 0.0


def test_sampling_comes_from_the_r1_profile_not_the_service(tmp_path):
    seen = {}

    def transport(body):
        seen["max_tokens"] = json.loads(body)["max_tokens"]
        return answer()

    svc, edit = edit_service(tmp_path, transport)
    svc.submit(dict(edit, prompt="x"))
    assert seen["max_tokens"] == rw.sampling("edit")["max_new_tokens"]


# ---------------------------------------------------------------------------
# Unparsed answers are a 200, never a suggestion
# ---------------------------------------------------------------------------


def test_unparsed_answer_is_200_with_no_prompt_to_apply():
    payload = service(reply("I could not produce JSON at all.")).submit(
        request()).payload
    assert payload["parse_ok"] is False
    assert payload["rewritten_prompt"] == ""
    assert payload["unparsed_answer"] == "I could not produce JSON at all."
    assert payload["ratio"] is None and payload["ratio_conflict"] is False


def test_a_ratio_conflict_offers_neither_ratio(tmp_path):
    body = reply(json.dumps({"rewritten_prompt": "Kept.",
                             "wh_ratio": "16:9", "ratio_follow": "<image1>"}))
    svc, edit = edit_service(tmp_path, body)
    payload = svc.submit(dict(edit, prompt="x", draft_revision=1)).payload
    assert payload["ratio"] is None and payload["ratio_conflict"] is True
    assert payload["rewritten_prompt"] == "Kept."


# ---------------------------------------------------------------------------
# Single flight
# ---------------------------------------------------------------------------


def test_a_second_caller_is_refused_busy_never_queued():
    release = threading.Event()
    entered = threading.Event()

    def transport(body):
        entered.set()
        release.wait(10)
        return answer()

    rewriting = service(transport)
    results = {}

    def first():
        results["first"] = rewriting.submit(request())

    thread = threading.Thread(target=first)
    thread.start()
    assert entered.wait(10)

    with pytest.raises(api.ApiError) as exc:
        rewriting.submit(request())
    assert exc.value.status == 503 and code(exc) == "rewrite_busy"

    release.set()
    thread.join(10)
    assert results["first"].status == 200


def test_the_slot_is_reusable_after_a_completed_call():
    rewriting = service(lambda body: answer())
    for _ in range(3):
        assert rewriting.submit(request()).status == 200


def test_the_slot_is_released_after_a_failure():
    def boom(body):
        raise rc.RewriteUnavailable("down")

    rewriting = service(boom)
    for _ in range(2):
        with pytest.raises(api.ApiError) as exc:
            rewriting.submit(request())
        assert code(exc) == "rewrite_unavailable"


def test_concurrent_callers_yield_exactly_one_success_and_one_busy():
    """Both threads race for the slot; exactly one wins. Deterministic overlap:
    the winner is held inside the backend until both verdicts are in."""
    entered = threading.Event()
    release = threading.Event()
    outcomes = []
    barrier = threading.Barrier(2)

    def transport(body):
        entered.set()
        release.wait(10)
        return answer()

    rewriting = service(transport)

    def attempt():
        barrier.wait(10)  # both threads release together, so they really race
        try:
            rewriting.submit(request())
            outcomes.append("ok")
        except api.ApiError as exc:
            outcomes.append(exc.code)

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert entered.wait(10)  # a winner is inside the backend, holding the slot
    # The loser must have decided while the winner is still blocked: wait for
    # exactly one appended verdict before allowing the winner to finish.
    deadline = time.monotonic() + 10
    while len(outcomes) < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(outcomes) == 1, f"loser did not decide promptly: {outcomes}"
    assert outcomes[0] == "rewrite_busy"  # the only verdict that can arrive now
    release.set()
    for thread in threads:
        thread.join(10)
    assert sorted(outcomes) == ["ok", "rewrite_busy"]


# ---------------------------------------------------------------------------
# Deadline
# ---------------------------------------------------------------------------


def test_deadline_expiry_is_504_and_leaves_the_slot_held_until_the_worker_ends():
    release = threading.Event()
    entered = threading.Event()

    def transport(body):
        entered.set()
        release.wait(10)
        return answer()

    rewriting = service(transport, timeout=0.2)
    try:
        rewriting.submit(request())
    except api.ApiError as exc:
        assert exc.status == 504 and exc.code == "rewrite_timeout"
    assert entered.is_set()

    # Still one worker running: a new attempt must be refused, not overlapped.
    with pytest.raises(api.ApiError) as exc:
        rewriting.submit(request())
    assert code(exc) == "rewrite_busy"

    release.set()
    assert rewriting.close(timeout_s=5) is True
    with pytest.raises(api.ApiError) as exc:  # closed, so disabled rather than busy
        rewriting.submit(request())
    assert code(exc) == "rewrite_disabled"


def test_a_slow_transport_past_the_client_budget_is_a_timeout():
    def transport(body):
        time.sleep(0.3)
        return answer()

    with pytest.raises(api.ApiError) as exc:
        service(transport, timeout=0.1).submit(request())
    assert exc.value.status == 504


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_releases_the_waiter_immediately_and_reports_cancelled():
    entered = threading.Event()
    release = threading.Event()

    def transport(body):
        entered.set()
        release.wait(10)
        return answer()

    rewriting = service(transport)
    caught = {}

    def attempt():
        try:
            rewriting.submit(request())
        except api.ApiError as exc:
            caught["code"] = exc.code

    thread = threading.Thread(target=attempt)
    thread.start()
    assert entered.wait(10)
    assert rewriting.cancel() is True
    thread.join(5)
    assert not thread.is_alive()  # released without waiting for the backend
    assert caught["code"] == "rewrite_cancelled"
    release.set()
    rewriting.close(timeout_s=5)


def test_cancel_with_nothing_in_flight_is_false():
    assert service(lambda body: answer()).cancel() is False


def test_a_cancelled_answer_is_never_applied():
    """The worker finishes after cancel: its payload must not be returned."""
    entered = threading.Event()
    release = threading.Event()

    def transport(body):
        entered.set()
        release.wait(10)
        return answer("SHOULD NOT APPEAR")

    rewriting = service(transport)
    caught = {}

    def attempt():
        try:
            rewriting.submit(request())
        except api.ApiError as exc:
            caught["exc"] = exc

    thread = threading.Thread(target=attempt)
    thread.start()
    entered.wait(10)
    rewriting.cancel()
    thread.join(5)
    release.set()
    assert "SHOULD" not in str(caught["exc"])
    assert rewriting.close(timeout_s=5) is True


def test_cancel_does_not_poison_the_next_rewrite():
    """Per-job cancel: a later call must succeed, not inherit the flag."""
    entered = threading.Event()
    release = threading.Event()
    call = {"n": 0}

    def transport(body):
        call["n"] += 1
        if call["n"] == 1:
            entered.set()
            release.wait(10)
        return answer("second")

    rewriting = service(transport)
    caught = {}

    def attempt():
        try:
            rewriting.submit(request())
        except api.ApiError as exc:
            caught["code"] = exc.code

    thread = threading.Thread(target=attempt)
    thread.start()
    entered.wait(10)
    rewriting.cancel()
    thread.join(5)
    assert caught["code"] == "rewrite_cancelled"

    # The cancelled worker still owns the slot until it returns; once it has,
    # the SAME service must serve the next call (the cancel flag was per-job).
    release.set()
    deadline = time.monotonic() + 5
    while rewriting._current is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert rewriting._current is None, "worker did not release the slot"
    assert rewriting.submit(request()).payload["rewritten_prompt"] == "second"
    assert rewriting.close(timeout_s=5) is True


# ---------------------------------------------------------------------------
# Typed backend failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc,status,expected", [
    (rc.RewriteTimeout("slow"), 504, "rewrite_timeout"),
    (rc.RewriteUnavailable("down"), 502, "rewrite_unavailable"),
    (rc.RewriteInvalidAnswer("not a completion"), 502, "rewrite_invalid_answer"),
    (OSError("socket exploded"), 502, "rewrite_unavailable"),
    (RuntimeError("surprise"), 502, "rewrite_unavailable"),
])
def test_backend_failures_map_to_typed_refusals(exc, status, expected):
    def transport(body):
        raise exc

    with pytest.raises(api.ApiError) as info:
        service(transport).submit(request())
    assert (info.value.status, info.value.code) == (status, expected)


def test_invalid_completion_documents_are_502_invalid_answer():
    for raw in (b"not json", b"{}", b'{"choices": []}',
                b'{"choices": [{"message": {"content": 7}}]}'):
        with pytest.raises(api.ApiError) as exc:
            service(lambda body, r=raw: r).submit(request())
        assert (exc.value.status, exc.value.code) == (502, "rewrite_invalid_answer")


def test_no_failure_message_contains_the_prompt():
    secret = "SECRET-DRAFT-CONTENT-1234"

    def transport(body):
        raise rc.RewriteUnavailable(f"failed while sending {body!r}")

    with pytest.raises(api.ApiError) as exc:
        service(transport).submit(request(prompt=secret))
    blob = json.dumps(exc.value.to_payload())
    assert secret not in blob
    assert len(exc.value.message) <= api.MAX_MESSAGE_CHARS + 1


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    {"task": "t2i", "prompt": "a cat", "draft_revision": 7},          # missing
    {"task": "t2i", "prompt": "a cat", "draft_revision": 7,
     "image_ids": [], "extra": 1},                                    # unknown
    {"task": "t2i", "prompt": 7, "draft_revision": 7, "image_ids": []},
    {"task": "t2i", "prompt": "a cat", "draft_revision": -1, "image_ids": []},
    {"task": "t2i", "prompt": "a cat", "draft_revision": True, "image_ids": []},
    {"task": "t2i", "prompt": "a cat", "draft_revision": 7.5, "image_ids": []},
    {"task": "t2i", "prompt": "a cat", "draft_revision": 7, "image_ids": "x"},
    {"task": "t2i", "prompt": "   ", "draft_revision": 7, "image_ids": []},
    {"task": "t2i", "prompt": "", "draft_revision": 7, "image_ids": []},
])
def test_malformed_bodies_are_400_invalid_envelope(bad):
    with pytest.raises(api.ApiError) as exc:
        service(lambda body: answer()).submit(bad)
    assert (exc.value.status, exc.value.code) == (400, "invalid_envelope")


def test_unknown_task_is_422_not_400():
    with pytest.raises(api.ApiError) as exc:
        service(lambda body: answer()).submit(request(task="video"))
    assert (exc.value.status, exc.value.code) == (422, "task_unsupported")


def test_over_long_prompt_is_422_and_never_reaches_the_backend():
    sent = {"n": 0}

    def transport(body):
        sent["n"] += 1
        return answer()

    with pytest.raises(api.ApiError) as exc:
        service(transport).submit(request(prompt="x" * (rw.PROMPT_MAX_CHARS + 1)))
    assert (exc.value.status, exc.value.code) == (422, "prompt_too_long")
    assert sent["n"] == 0


def test_prompt_at_exactly_the_cap_is_accepted():
    payload = service(answer()).submit(
        request(prompt="x" * rw.PROMPT_MAX_CHARS)).payload
    assert len(payload["original_prompt"]) == rw.PROMPT_MAX_CHARS


def test_an_edit_rewrite_without_references_is_refused_not_downgraded():
    with pytest.raises(api.ApiError) as exc:
        service(lambda body: answer()).submit(request(task="edit"))
    assert (exc.value.status, exc.value.code) == (422, "reference_required")


def test_malformed_image_ids_are_an_envelope_refusal():
    with pytest.raises(api.ApiError) as exc:
        service(lambda body: answer()).submit(request(image_ids=[1]))
    assert (exc.value.status, exc.value.code) == (400, "invalid_envelope")


def test_validation_happens_before_the_slot_is_claimed():
    sent = {"n": 0}

    def transport(body):
        sent["n"] += 1
        return answer()

    rewriting = service(transport)
    with pytest.raises(api.ApiError):
        rewriting.submit(request(task="video"))
    # A refused envelope must not leave the slot taken.
    assert rewriting.submit(request()).status == 200
    assert sent["n"] == 1


def test_the_route_cap_fits_the_worst_case_envelope():
    """A full-length 4-byte-per-character draft plus envelope must fit.

    The browser's ``JSON.stringify`` emits raw UTF-8 (never ``\\uXXXX``
    escapes), so the cap is derived against raw bytes. A body that DID escape
    every character would be ~3x larger and be refused at this same cap —
    refused, not mis-parsed, which is the safe failure.
    """
    worst = "𝄞" * rw.PROMPT_MAX_CHARS  # 4 bytes/char in UTF-8
    assert len(worst.encode("utf-8")) == rw.PROMPT_MAX_CHARS * 4
    envelope = json.dumps({"schema": rs.REWRITE_REQUEST_SCHEMA, "task": "t2i",
                           "prompt": worst, "draft_revision": 2**31 - 1,
                           "image_ids": []},
                          ensure_ascii=False).encode("utf-8")
    assert len(envelope) <= rs.MAX_REQUEST_BYTES
    # And the cap is not so loose that a body twice the legal size sneaks in.
    assert len(envelope) > rs.MAX_REQUEST_BYTES // 2


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def test_close_is_idempotent_and_refuses_new_work():
    rewriting = service(lambda body: answer())
    assert rewriting.close(timeout_s=5) is True
    assert rewriting.close(timeout_s=5) is True
    with pytest.raises(api.ApiError) as exc:
        rewriting.submit(request())
    assert code(exc) == "rewrite_disabled"


def test_close_waits_for_an_in_flight_worker():
    release = threading.Event()
    entered = threading.Event()

    def transport(body):
        entered.set()
        release.wait(10)
        return answer()

    rewriting = service(transport)
    threading.Thread(target=lambda: _swallow(rewriting.submit, request()),
                     daemon=True).start()
    entered.wait(10)
    assert rewriting.close(timeout_s=0.1) is False  # still busy, reports honestly
    release.set()
    assert rewriting.close(timeout_s=5) is True


def _swallow(fn, *args):
    try:
        return fn(*args)
    except api.ApiError:
        return None


# ---------------------------------------------------------------------------
# Logging surface
# ---------------------------------------------------------------------------


def test_the_log_event_never_carries_the_prompt():
    secret = "SECRET-DRAFT-CONTENT-1234"
    rewriting = service(lambda body: answer())
    rewriting.submit(request(prompt=secret))
    event = rewriting.event("t2i", prompt_chars=len(secret),
                            elapsed_ms=10, parse_ok=True, outcome="ok")
    assert set(event) == {"event", "task", "backend", "elapsed_ms",
                          "prompt_chars", "parse_ok", "outcome"}
    assert secret not in json.dumps(event)
    assert event["event"] == "rewrite"


def test_nothing_is_logged_while_a_rewrite_runs(capsys, caplog):
    """The draft must reach neither stdout/stderr nor any logging record.

    Behavioural, not a source scan: a future ``print`` or logger call that
    mentioned the prompt would fail here.
    """
    import logging

    secret = "SECRET-DRAFT-CONTENT-1234"
    with caplog.at_level(logging.DEBUG):
        service(lambda body: answer()).submit(request(prompt=secret))
    captured = capsys.readouterr()
    assert secret not in captured.out and secret not in captured.err
    for record in caplog.records:
        assert secret not in record.getMessage()


def test_status_block_is_built_by_the_shared_validator():
    """The service must not hand-roll its block: api.validate_rewrite_block is
    the one rule, so the server can never emit a shape its own UI rejects."""
    from qwen_engine.web import api as web_api
    for svc in (service(answer()), rs.RewriteService(rs.RewriteSettings.disabled())):
        block = svc.status_block()
        assert web_api.validate_rewrite_block(block) == block
    # The disabled block is the canonical constant, not a paraphrase of it.
    assert rs.RewriteService(rs.RewriteSettings.disabled()).status_block() == \
        dict(web_api.REWRITE_DISABLED_BLOCK)


def test_status_block_refuses_a_backend_name_that_would_leak_a_location():
    """A backend string is echoed to the browser, so a location in it is a
    configuration error caught at startup, not a leaked address."""
    from qwen_engine.web import api as web_api
    configured = {"state": "available", "code": "rewrite_available",
                  "backend": "http://127.0.0.1:8079/v1", "reason": ""}
    with pytest.raises(ValueError):
        web_api.validate_rewrite_block(configured)


# ---------------------------------------------------------------------------
# Edit rewriting with reference images (d3): the images the backend receives
#
# The store is the REAL store (tmp dirs, real Pillow); the backend is a fake
# transport that records the request it was handed, so every assertion below is
# about the bytes that would have left the server.
# ---------------------------------------------------------------------------


def test_edit_rewrite_sends_the_references_first_and_in_the_given_order(tmp_path):
    first = store_image(tmp_path / "store", colour=(255, 0, 0))
    second = store_image(tmp_path / "store", colour=(0, 0, 255))
    log: list = []
    svc = rs.RewriteService(
        rs.RewriteSettings(backend="local-llm", endpoint=ENDPOINT, model=MODEL,
                           prompts=dict(PROMPTS), timeout=30.0,
                           references_dir=str(tmp_path / "store")),
        transport=lambda raw: log.append(json.loads(raw)) or answer())
    payload = svc.submit(request(task="edit", prompt="make it night",
                                 image_ids=[second, first])).payload
    assert payload["task"] == "edit"
    parts = user_parts(log[0])
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    assert parts[2]["text"] == "make it night"
    # The ORDER is the argument order, not the store's or a sorted one: image 1
    # is <image1> in the edit system prompt.
    assert parts[0]["image_url"]["url"] != parts[1]["image_url"]["url"]
    assert parts[0]["image_url"]["url"] == rs._reference_image_uri(
        str(tmp_path / "store"), second, 1)


def test_reference_images_are_png_data_uris_never_paths_or_urls(tmp_path):
    root = tmp_path / "store"
    ref = store_image(root, fmt="JPEG", mode="RGB")
    log: list = []
    svc = rs.RewriteService(
        rs.RewriteSettings(backend="local-llm", endpoint=ENDPOINT, model=MODEL,
                           prompts=dict(PROMPTS), timeout=30.0,
                           references_dir=str(root)),
        transport=lambda raw: log.append(json.loads(raw)) or answer())
    svc.submit(request(task="edit", image_ids=[ref]))
    url = user_parts(log[0])[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # The source was a JPEG; the wire copy is normalised to PNG data.
    assert base64.b64decode(url.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"
    # Nothing locatable travels: no store path, no filesystem path, no scheme.
    wire = json.dumps(log[0])
    assert str(root) not in wire and ref not in wire
    assert "file:" not in wire and "http://127.0.0.1:8079/v1" not in url


def wire_size(uri: str) -> tuple[int, int]:
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).size


def test_references_are_downscaled_but_never_enlarged(tmp_path):
    root = tmp_path / "store"
    big = store_image(root, size=(1600, 800), fmt="JPEG")
    small = store_image(root, size=(40, 20))
    # The stored ORIGINAL is untouched: only the wire copy is shrunk.
    assert Image.open(io.BytesIO(reference_store.read_blob(root, big))).size == (1600, 800)
    assert wire_size(rs._reference_image_uri(str(root), big, 1)) == (768, 384)
    # And a small reference is sent at its own size, not blown up.
    assert wire_size(rs._reference_image_uri(str(root), small, 1)) == (40, 20)


def test_the_downscale_bound_preserves_aspect_and_stays_within_the_cap(tmp_path):
    root = tmp_path / "store"
    # A non-power-of-two portrait: the longest side lands exactly on the cap,
    # the short side is rounded, and the aspect ratio survives.
    ref = store_image(root, size=(500, 999))
    width, height = wire_size(rs._reference_image_uri(str(root), ref, 1))
    assert max(width, height) == rs.REWRITE_IMAGE_MAX_DIM
    assert width < height
    assert width / height == pytest.approx(500 / 999, abs=0.005)


def test_no_image_is_sent_before_the_store_is_configured_for_a_t2i_rewrite(tmp_path):
    """A t2i rewrite stays text-only and must not touch the store at all."""
    root = tmp_path / "store"
    store_image(root)
    log: list = []
    svc = rs.RewriteService(
        rs.RewriteSettings(backend="local-llm", endpoint=ENDPOINT, model=MODEL,
                           prompts=dict(PROMPTS), timeout=30.0,
                           references_dir=str(root)),
        transport=lambda raw: log.append(json.loads(raw)) or answer())
    svc.submit(request())
    assert user_parts(log[0]) == "a cat"


def test_alpha_is_flattened_onto_white_not_dropped(tmp_path):
    """A half-transparent red pixel over white must arrive PINK-ish, not the
    arbitrary RGB that Pillow leaves under an alpha pixel, and never black."""
    root = tmp_path / "store"
    ref = store_image(root, mode="RGBA", size=(8, 8), colour=(255, 0, 0, 0))
    uri = rs._reference_image_uri(str(root), ref, 1)
    image = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert image.mode == "RGB"
    assert image.getpixel((4, 4)) == (255, 255, 255)
    # Half alpha of pure red composites to a pink, not to (255,0,0) or (0,0,0).
    assert rs._flatten_alpha(Image.new("RGBA", (2, 2), (255, 0, 0, 128))).getpixel((0, 0)) == (255, 127, 127)


def test_an_unknown_reference_id_is_404_without_a_path(tmp_path):
    root = tmp_path / "store"
    store_image(root)  # the store exists and holds something else
    svc = rs.RewriteService(
        rs.RewriteSettings(backend="local-llm", endpoint=ENDPOINT, model=MODEL,
                           prompts=dict(PROMPTS), timeout=30.0,
                           references_dir=str(root)),
        transport=lambda raw: answer())
    with pytest.raises(api.ApiError) as exc:
        svc.submit(request(task="edit", image_ids=["b" * 64]))
    assert (exc.value.status, exc.value.code) == (404, "reference_unknown")
    assert str(root) not in exc.value.message and "b" * 64 not in exc.value.message
    assert "reference 1" in exc.value.message


def test_an_unconfigured_store_refuses_an_edit_rewrite():
    with pytest.raises(api.ApiError) as exc:
        service(answer()).submit(request(task="edit", image_ids=["a" * 64]))
    assert (exc.value.status, exc.value.code) == (422, "reference_unavailable")


def test_a_blocked_decode_holds_no_lock_and_never_deadlocks(tmp_path):
    """The decode runs on the worker thread, outside the Condition.

    Proven by parking a decode and showing that, meanwhile, a status read and a
    competing submit both answer at once — the second refused by the SLOT rule
    (503 busy), which is only reachable while the lock is free. If the decode
    held the lock, both would hang and the whole test would time out.
    """
    root = tmp_path / "store"
    ref = store_image(root)
    release = threading.Event()
    entered = threading.Event()
    from qwen_engine.web import rewrite_service as module

    original = module._reference_image_uri

    def slow(root_arg, reference_id, ordinal):
        entered.set()
        assert release.wait(10.0)
        return original(root_arg, reference_id, ordinal)

    module._reference_image_uri = slow
    try:
        svc = rs.RewriteService(
            rs.RewriteSettings(backend="local-llm", endpoint=ENDPOINT, model=MODEL,
                               prompts=dict(PROMPTS), timeout=30.0,
                               references_dir=str(root)),
            transport=lambda raw: answer())
        done: list = []

        def first():
            try:
                done.append(svc.submit(request(task="edit", image_ids=[ref]))
                            .payload["task"])
            except BaseException as exc:  # pragma: no cover - failure detail
                done.append(exc)

        thread = threading.Thread(target=first)
        thread.start()
        assert entered.wait(10.0)
        assert svc.status_block()["code"] == "rewrite_available"
        with pytest.raises(api.ApiError) as exc:
            svc.submit(request(task="edit", image_ids=[ref]))
        assert (exc.value.status, exc.value.code) == (503, "rewrite_busy")
        release.set()
        thread.join(10.0)
        assert not thread.is_alive()
        assert done == ["edit"]
    finally:
        module._reference_image_uri = original
        release.set()
