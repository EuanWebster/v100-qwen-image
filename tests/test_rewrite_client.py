"""Rewrite transport (R2): endpoint policy, request bytes, response handling.

Every test injects a fake transport, so no socket is constructed (the CPU
guard denies construction outright) and no service is required. The HTTP
transport's own read/cap logic is exercised through ``_read_capped`` with an
``io.BytesIO``, which is the same code path the response object feeds.
"""
from __future__ import annotations

import io
import json

import pytest

from qwen_engine import rewrite as rw
from qwen_engine import rewrite_client as rc
from qwen_engine.errors import EngineError

ENDPOINT = "http://127.0.0.1:8079/v1/chat/completions"
MODEL = "Qwen3.8"


def client(transport=None, endpoint=ENDPOINT, model=MODEL, timeout=180.0):
    return rc.RewriteClient(endpoint=endpoint, model=model, timeout=timeout,
                            transport=transport)


def completion(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant",
                                                "content": content},
                                    "finish_reason": "stop"}]}).encode("utf-8")


def messages(text: str = "a cat on a stool"):
    return rw.build_messages("system prompt", text)


# ---------------------------------------------------------------------------
# Endpoint policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint,host,port,path", [
    ("http://127.0.0.1:8079/v1/chat/completions", "127.0.0.1", 8079, "/v1/chat/completions"),
    ("http://127.1.2.3:9000/v1", "127.1.2.3", 9000, "/v1"),
    ("http://localhost:8079/v1/chat/completions", "127.0.0.1", 8079, "/v1/chat/completions"),
    ("http://[::1]:8079/v1", "::1", 8079, "/v1"),
])
def test_loopback_endpoints_are_accepted(endpoint, host, port, path):
    parsed = rc.parse_endpoint(endpoint)
    assert (parsed.host, parsed.port, parsed.path) == (host, port, path)


@pytest.mark.parametrize("endpoint", [
    "http://10.0.0.5:8079/v1",              # private, but not loopback
    "http://192.168.1.9:8079/v1",
    "http://0.0.0.0:8079/v1",               # wildcard is not a target
    "http://[::2]:8079/v1",
    "https://127.0.0.1:8079/v1",            # no TLS policy exists
    "http://127.0.0.1/v1",                  # port must be explicit
    "http://127.0.0.1:8079/v1?x=1",
    "http://127.0.0.1:8079/v1#frag",
    "http://user:pw@127.0.0.1:8079/v1",
    "http://rewriter.example:8079/v1",      # a hostname would need resolution
    "http://127.0.0.1:99999/v1",
    "http://127.0.0.1:8079/ v1",
    "  http://127.0.0.1:8079/v1",
    "",
    "not a url",
])
def test_non_loopback_or_malformed_endpoints_are_refused_at_construction(endpoint):
    with pytest.raises(rc.RewriteConfigError):
        rc.RewriteClient(endpoint=endpoint, model=MODEL)


def test_endpoint_refusal_is_a_startup_error_not_a_runtime_retry():
    """A non-loopback target never becomes a client, so it cannot be retried."""
    assert issubclass(rc.RewriteConfigError, rc.RewriteClientError)
    assert issubclass(rc.RewriteClientError, EngineError)


def test_host_header_keeps_ipv6_brackets():
    assert rc.parse_endpoint("http://[::1]:8079/v1").host_header == "[::1]:8079"
    assert rc.parse_endpoint(ENDPOINT).host_header == "127.0.0.1:8079"


def test_display_url_round_trips():
    endpoint = rc.parse_endpoint("http://[::1]:8079/v1")
    assert endpoint.url == "http://[::1]:8079/v1"


@pytest.mark.parametrize("model", ["", "   ", None, 7])
def test_model_must_be_a_non_empty_string(model):
    with pytest.raises(rc.RewriteConfigError):
        rc.RewriteClient(endpoint=ENDPOINT, model=model)


@pytest.mark.parametrize("timeout", [0, -1, 4000, "180", None, True])
def test_timeout_must_be_a_positive_bounded_number(timeout):
    with pytest.raises(rc.RewriteConfigError):
        rc.RewriteClient(endpoint=ENDPOINT, model=MODEL, timeout=timeout)


# ---------------------------------------------------------------------------
# Request bytes
# ---------------------------------------------------------------------------


def test_request_body_is_exactly_the_documented_json():
    body = client().build_body(messages("a cat"), "t2i")
    document = json.loads(body)
    assert document == {
        "model": MODEL,
        "messages": [{"role": "system", "content": "system prompt"},
                     {"role": "user", "content": "a cat"}],
        "stream": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16256,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_request_body_is_compact_utf8_and_not_ascii_escaped():
    body = client().build_body(messages("雨中弹吉他的柯基"), "t2i")
    assert b"\\u" not in body
    assert "雨中弹吉他的柯基".encode("utf-8") in body
    assert b", " not in body  # compact separators
    json.loads(body)


def test_edit_budget_differs_and_is_carried():
    assert json.loads(client().build_body(messages("x"), "edit"))["max_tokens"] == 24000


def test_unsent_profile_fields_are_absent_by_default():
    """top_k/min_p/presence_penalty were never exercised against the service."""
    document = json.loads(client().build_body(messages("x"), "t2i"))
    for name in ("top_k", "min_p", "presence_penalty"):
        assert name not in document


def test_extra_sampling_sends_measured_fields_only():
    body = client().build_body(messages("x"), "t2i",
                              extra_sampling={"presence_penalty": 1.5})
    assert json.loads(body)["presence_penalty"] == 1.5


def test_extra_sampling_refuses_unknown_fields():
    with pytest.raises(rc.RewriteClientError, match="unknown sampling field"):
        client().build_body(messages("x"), "t2i", extra_sampling={"topp": 0.9})


def test_build_body_refuses_an_unknown_task_or_empty_messages():
    with pytest.raises(rw.RewriteError):
        client().build_body(messages(), "video")
    with pytest.raises(rc.RewriteClientError, match="messages"):
        client().build_body([], "t2i")


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------


def test_complete_returns_content_verbatim():
    content = '  {"rewritten_prompt": "Kept.", "wh_ratio": "1:1"}\n'
    assert client(lambda body: completion(content)).complete(messages(), "t2i") == content


def test_complete_passes_the_request_bytes_to_the_transport():
    seen = {}

    def transport(body):
        seen["body"] = body
        return completion("ok")

    client(transport).complete(messages("draft text"), "t2i")
    assert json.loads(seen["body"])["messages"][1]["content"] == "draft text"


@pytest.mark.parametrize("raw", [
    b"not json",
    b"{}",
    b'{"choices": []}',
    b'{"choices": [{"message": {"content": null}}]}',
    b'{"choices": [{"message": "string"}]}',
    b'{"choices": ["not an object"]}',
    b'{"choices": [{"message": {"content": "hi"',   # truncated mid-document
    b"[1, 2, 3]",
    b"\xff\xfe binary",
])
def test_transport_violations_are_invalid_answer(raw):
    with pytest.raises(rc.RewriteInvalidAnswer):
        client(lambda body: raw).complete(messages(), "t2i")


def test_oversize_response_is_refused_even_from_a_transport():
    """Belt and braces: a transport that ignores the cap still cannot smuggle."""
    payload = b'{"choices": [{"message": {"content": "' + b"x" * rc.MAX_RESPONSE_BYTES + b'"}}]}'
    with pytest.raises(rc.RewriteInvalidAnswer, match="cap"):
        client(lambda body: payload).complete(messages(), "t2i")


def test_typed_transport_errors_propagate_unchanged():
    for error in (rc.RewriteTimeout("slow"), rc.RewriteUnavailable("gone")):
        with pytest.raises(type(error)):
            client(lambda body, e=error: (_ for _ in ()).throw(e)).complete(
                messages(), "t2i")


def test_transport_os_errors_are_not_rewrapped():
    """Only HttpTransport maps OSError; an injected transport's errors surface."""
    with pytest.raises(OSError):
        client(lambda body: (_ for _ in ()).throw(OSError("boom"))).complete(
            messages(), "t2i")


def test_a_slow_reply_past_the_budget_is_a_timeout(monkeypatch):
    clock = {"now": 0.0}

    def fake_monotonic():
        return clock["now"]

    def transport(body):
        clock["now"] += 5.0  # took longer than the 1 s budget
        return completion("late")

    monkeypatch.setattr(rc.time, "monotonic", fake_monotonic)
    slow = client(transport, timeout=1.0)
    with pytest.raises(rc.RewriteTimeout, match="budget"):
        slow.complete(messages(), "t2i")


# ---------------------------------------------------------------------------
# Read cap (the HTTP transport's own logic, without a socket)
# ---------------------------------------------------------------------------


def test_read_capped_accepts_a_body_at_the_limit():
    payload = b"y" * 100
    assert rc._read_capped(io.BytesIO(payload), limit=100) == payload


def test_read_capped_refuses_a_body_past_the_limit():
    with pytest.raises(rc.RewriteInvalidAnswer, match="cap"):
        rc._read_capped(io.BytesIO(b"y" * 101), limit=100)


def test_read_capped_stops_reading_once_over_the_limit():
    class Counting(io.BytesIO):
        reads = 0

        def read(self, size=-1):
            Counting.reads += 1
            return super().read(size)

    payload = b"y" * (rc.MAX_RESPONSE_BYTES * 4)
    with pytest.raises(rc.RewriteInvalidAnswer):
        rc._read_capped(Counting(payload))
    # Refused as soon as the cap was crossed, not after buffering the whole
    # body: the read loop is chunked, so this is a small multiple of the cap.
    assert Counting.reads <= rc.MAX_RESPONSE_BYTES // rc._READ_CHUNK + 2
    assert Counting.reads < len(payload) // rc._READ_CHUNK


# ---------------------------------------------------------------------------
# Proxy independence and construction wiring
# ---------------------------------------------------------------------------


def test_proxy_environment_never_redirects_the_request(monkeypatch):
    """http.client ignores proxy env; only the parsed endpoint is contacted."""
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY",
                 "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://10.9.9.9:3128")
    seen = {}

    def transport(body):
        seen["called"] = True
        return completion("ok")

    assert client(transport).complete(messages(), "t2i") == "ok"
    assert seen == {"called": True}


def test_default_transport_is_the_http_one_and_no_socket_is_opened():
    assert isinstance(client()._transport.__class__, type)
    assert client()._transport.__class__ is rc.HttpTransport
    assert rw  # the contract module stays importable without the transport


def test_transport_receives_bytes_and_the_client_is_not_stateful():
    bodies = []

    def transport(body):
        bodies.append(body)
        return completion("ok")

    rewriting = client(transport)
    rewriting.complete(messages("one"), "t2i")
    rewriting.complete(messages("two"), "t2i")
    assert len(bodies) == 2
    assert json.loads(bodies[0]) != json.loads(bodies[1])


def test_no_proxy_or_tls_hook_is_exposed():
    """The module offers no way to redirect the call: no proxy switch, no TLS.

    Independence from ``HTTP(S)_PROXY`` is structural — ``http.client`` has no
    proxy support at all — so the absence of any knob is the testable part.
    """
    assert not hasattr(rc, "proxy")
    assert not hasattr(rc, "HTTPSConnection")
    with pytest.raises(rc.RewriteConfigError):
        rc.parse_endpoint("http://10.9.9.9:3128")


def test_a_body_shorter_than_content_length_is_truncated_not_accepted():
    """A cut-off reply is refused even when the fragment happens to parse."""
    body = completion("ok")
    assert rc._check_complete(body, str(len(body))) == body
    for declared in (None, "", "abc", "chunked"):  # no usable length to compare
        assert rc._check_complete(body, declared) == body
    with pytest.raises(rc.RewriteInvalidAnswer, match="truncated"):
        rc._check_complete(body, str(len(body) + 100))
    with pytest.raises(rc.RewriteInvalidAnswer, match="truncated"):
        rc._check_complete(body[:-5], str(len(body)))
