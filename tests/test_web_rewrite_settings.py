"""Guarded tests for the server-side **Prompt rewriter** settings page.

The amendment to the two-rewriter plan: there is no per-request rewriter pick
in the Create view any more. One server-side choice — saved by the server,
shared by every browser pointed at it — says which rewriter answers, and a
loopback-only Settings route changes it. Three properties carry the design and
are what these tests pin:

* the POST form is CLOSED, and nothing in it is a filesystem path or an
  executable (the binary, the GGUF and the system prompts stay launch flags);
* a request is validated, and refused, BEFORE any client is built or any file
  is written — a refusal leaves the server exactly as it was;
* no response names a path, a command or an endpoint the user did not type.

No sockets, no torch, no GPU: the local service is an injected transport.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import stat
import threading
import types

import pytest
from PIL import Image

from qwen_engine import reference_store
from qwen_engine import rewrite_client as rc
from qwen_engine.web import api
from qwen_engine.web import rewrite_service as rs
from qwen_engine.web import server

ENDPOINT = "http://127.0.0.1:8079/v1/chat/completions"
OTHER_ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "Qwen3.8"
PROMPTS = {"t2i": "system prompt for t2i", "edit": "system prompt for edit"}
BINARY = "/opt/llama/llama-server"
GGUF = "/models/pe/PE-T2I-Q8_0.gguf"


def reply(content: str = '{"rewritten_prompt": "x"}') -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant",
                                                "content": content}}]}).encode()


class FakeClient:
    """Stands in for a backend client; records that it was ever built."""

    def __init__(self, name, built, timeout=30.0):
        self.name = name
        self.built = built
        self.timeout = timeout
        self.requests = []

    def complete(self, messages, task, sampling=None):
        """The client contract: messages in, the raw answer text out."""
        self.requests.append(task)
        return json.dumps({"rewritten_prompt": f"rewritten by {self.name}"})

    def close(self):
        return None


class BrokenClient(FakeClient):
    """A backend that fails the way a failed spawn does: quoting its own
    command line, which is exactly what must never reach the browser."""

    def __init__(self, name, built, detail):
        super().__init__(name, built)
        self.detail = detail

    def complete(self, messages, task, sampling=None):
        raise rc.RewriteClientError(self.detail)


@pytest.fixture
def built():
    """Records every client the service builds, by backend id."""
    return []


def make_service(tmp_path, *, local=True, pe=False, built=None, clients=None,
                 transport=None, preferences=None, client=None,
                 **overrides) -> rs.RewriteService:
    """A service with the given backends possible, writing settings in tmp.

    ``built`` records every build in order and ``clients`` ends up holding the
    live client per backend id, so a test can see which backend was asked.
    """
    local_settings = (rs.RewriteSettings(
        backend="local-llm", endpoint=ENDPOINT, model=MODEL, prompts=dict(PROMPTS),
        timeout=30.0, **overrides) if local else rs.RewriteSettings.disabled())
    pe_settings = (rs.PESettings(llama_server=BINARY, model_path=GGUF,
                                 system_prompt="pe system prompt",
                                 timeout=300.0) if pe else None)
    made = built if built is not None else []

    def factory(backend, settings, pe_cfg, prefs):
        made.append(backend)
        instance = client if client is not None else FakeClient(backend, made)
        if clients is not None:
            clients[backend] = instance
        return instance

    return rs.RewriteService(local_settings, transport=transport,
                             pe=pe_settings, cache_dir=str(tmp_path),
                             preferences=preferences, client_factory=factory)


def body(**overrides) -> bytes:
    doc = {"schema": api.REWRITE_SETTINGS_SCHEMA, "backend": "local-llm",
           "endpoint": ENDPOINT, "model": MODEL, "timeout_s": 180.0}
    doc.update(overrides)
    return json.dumps(doc).encode()


def form(raw: bytes) -> dict:
    return api.build_rewrite_settings_request(raw)


# --------------------------------------------------------------------------
# the posted form: closed, and never a path


def test_the_canonical_form_is_accepted_and_normalised():
    checked = form(body(endpoint="  " + ENDPOINT + "  ", model="  Qwen3.8 "))
    assert checked == {"backend": "local-llm", "endpoint": ENDPOINT,
                       "model": MODEL, "timeout_s": 180.0}
    # An empty box means "no override", not "an empty address".
    blank = form(body(backend="pe-llamacpp", endpoint="", model="",
                      timeout_s=None))
    assert blank == {"backend": "pe-llamacpp", "endpoint": None,
                     "model": None, "timeout_s": None}


def test_the_form_has_exactly_five_keys_and_a_schema_tag():
    for dropped in ("schema", "backend", "endpoint", "model", "timeout_s"):
        doc = json.loads(body())
        del doc[dropped]
        with pytest.raises(api.ApiError) as exc:
            form(json.dumps(doc).encode())
        assert exc.value.status == 400
        assert exc.value.code == "invalid_envelope"


def test_a_sixth_key_is_refused_rather_than_dropped():
    doc = json.loads(body())
    doc["temperature"] = 0.5
    with pytest.raises(api.ApiError) as exc:
        form(json.dumps(doc).encode())
    assert exc.value.code == "invalid_envelope"
    assert "temperature" in exc.value.message


@pytest.mark.parametrize("key", [
    "binary", "llama_binary", "server_bin", "model_path", "gguf",
    "prompt_path", "cache_dir", "log_file", "exe", "folder", "path",
    "a/b", "C:\\Program Files\\llama.exe",
])
def test_a_path_shaped_key_is_refused_and_the_rule_is_named(key):
    """The absolute rule, not a denylist: a browser that may name one file can
    name any file on this machine, so NO key that looks like a location — by
    name or by shape — is ever accepted."""
    doc = json.loads(body(backend="pe-llamacpp"))
    doc[key] = BINARY
    with pytest.raises(api.ApiError) as exc:
        form(json.dumps(doc).encode())
    assert exc.value.status == 400
    assert exc.value.code == "invalid_envelope"
    assert "no filesystem path" in exc.value.message


def test_the_saved_document_rejects_a_path_key_too():
    """The route and the loader share ONE validator, so a hand-edited file
    cannot be wider than a request."""
    with pytest.raises(api.ApiError):
        rs.RewritePreferences.from_dict(
            {"backend": "pe-llamacpp", "endpoint": None, "model": None,
             "timeout_s": None, "binary": BINARY})


@pytest.mark.parametrize("endpoint", [
    "https://127.0.0.1:8079/v1/chat/completions",   # no TLS policy exists
    "http://example.com/v1/chat/completions",       # not loopback
    "http://127.0.0.1/chat/completions",            # no explicit port
    "http://user:pw@127.0.0.1:8079/v1",             # credentials
    "file:///etc/passwd",                           # not http at all
    "http://127.0.0.1:8079/v1?q=http://evil.test",   # not a URL
])
def test_an_endpoint_that_is_not_a_plain_loopback_url_is_refused(endpoint):
    with pytest.raises(api.ApiError) as exc:
        form(body(endpoint=endpoint))
    assert exc.value.status == 422
    assert exc.value.code == "endpoint_unsupported"
    assert any(name == "endpoint" for name, _ in exc.value.field_errors)


def test_a_path_IN_THE_URL_IS_fine_but_a_path_on_DISK_is_not():
    """The endpoint's URL path is required (``/v1/chat/completions``); what is
    forbidden is a FILESYSTEM path, so the two must not be confused."""
    assert form(body(endpoint="http://localhost:1234/a/b/c.json"))["endpoint"] == \
        "http://localhost:1234/a/b/c.json"


@pytest.mark.parametrize("backend", ["gpt-4", "pe_llamacpp", "PE-LLAMACPP",
                                     "local", "", "off "])
def test_the_backend_is_closed(backend):
    with pytest.raises(api.ApiError) as exc:
        form(body(backend=backend))
    assert exc.value.status == 422
    assert exc.value.code == "backend_unknown"


def test_off_is_a_legal_choice():
    assert form(body(backend="off", endpoint=None, model=None))["backend"] == "off"


def test_choosing_the_local_service_without_an_address_is_refused():
    with pytest.raises(api.ApiError) as exc:
        form(body(endpoint=None))
    assert exc.value.status == 422
    assert exc.value.code == "backend_incomplete"
    assert sorted(name for name, _ in exc.value.field_errors) == ["endpoint", "model"]


@pytest.mark.parametrize("timeout", [0, -1, 3601, "180", True])
def test_the_timeout_is_a_bounded_number_of_seconds(timeout):
    # ``Infinity`` is absent: the strict JSON parser rejects a non-finite
    # constant long before this validator sees it.
    with pytest.raises(api.ApiError) as exc:
        form(body(timeout_s=timeout))
    assert exc.value.status == 400
    assert exc.value.code == "invalid_envelope"
    assert any(name == "timeout_s" for name, _ in exc.value.field_errors)


def test_the_probe_request_pins_the_backend_and_keeps_the_rules():
    fields = api.build_rewrite_test_request(json.dumps(
        {"schema": api.REWRITE_TEST_REQUEST_SCHEMA, "endpoint": ENDPOINT,
         "model": MODEL, "timeout_s": 5.0}).encode())
    assert fields["backend"] == "local-llm"
    # A probe may not ask for the enhancer: that would load 9.5 GiB on a click.
    with pytest.raises(api.ApiError):
        api.build_rewrite_test_request(json.dumps(
            {"schema": api.REWRITE_TEST_REQUEST_SCHEMA, "backend": "pe-llamacpp",
             "endpoint": ENDPOINT, "model": MODEL, "timeout_s": 5.0}).encode())
    with pytest.raises(api.ApiError):
        api.build_rewrite_test_request(json.dumps(
            {"schema": api.REWRITE_TEST_REQUEST_SCHEMA, "endpoint": "http://x:1/",
             "model": MODEL, "timeout_s": 5.0}).encode())


# --------------------------------------------------------------------------
# the state document the page is built from


def state_of(service):
    return api.validate_rewrite_settings_state(json.loads(
        json.dumps(service.settings_state())))


def test_the_state_names_every_rewriter_and_where_the_choice_lives(tmp_path):
    service = make_service(tmp_path, local=True, pe=True)
    state = state_of(service)
    assert state["schema"] == api.REWRITE_SETTINGS_SCHEMA
    # Both rows always, in the server's order, with the GPU one saying so.
    assert [(row["id"], row["available"], row["uses_gpu"], row["tasks"])
            for row in state["backends"]] == [
        ("pe-llamacpp", True, True, ["t2i"]),
        ("local-llm", True, False, ["t2i", "edit"]),
    ]
    assert state["off"] == {"id": "off", "label": "Off", "available": True,
                            "reason": ""}
    assert state["persistent"] is True and "note" not in state
    # Prefill: what the LOCAL service is reached as RIGHT NOW (flags here).
    assert state["local"] == {"endpoint": ENDPOINT, "model": MODEL,
                              "timeout_s": 30.0}
    assert state["backend"] == "pe-llamacpp"       # the enhancer is standard
    assert state["saved"] == {"backend": None, "endpoint": None, "model": None,
                              "timeout_s": None}


def test_a_rewriter_the_flags_cannot_run_is_offered_greyed_with_a_reason(tmp_path):
    service = make_service(tmp_path, local=True, pe=False)
    state = state_of(service)
    pe_row = state["backends"][0]
    assert pe_row["available"] is False
    assert pe_row["reason"] and "flags" in pe_row["reason"]
    # A greyed-out choice still explains itself; the page never hides options.
    assert state["backend"] == "local-llm"


def test_the_state_never_carries_a_path_even_when_every_path_is_configured(tmp_path):
    """The binary, the GGUF, the system prompts and the enhancer log all live
    on disk; none of those names may cross the wire, because the page would
    otherwise tell a browser where the operator's files are."""
    service = make_service(tmp_path, local=True, pe=True)
    for document in (service.settings_state(), service.status_block()):
        text = json.dumps(document)
        for secret in (BINARY, GGUF, str(tmp_path), "system_prompt", ".log"):
            assert secret not in text, (secret, sorted(document))
    assert state_of(service)["backends"][0]["available"] is True


def test_a_state_with_an_extra_key_is_refused_before_it_is_sent(tmp_path):
    service = make_service(tmp_path)
    doc = dict(service.settings_state())
    for extra in ({"binary": BINARY}, {"log_path": "/tmp/x.log"},
                  {"backend_extra": 1}):
        with pytest.raises(ValueError):
            api.validate_rewrite_settings_state({**doc, **extra})


def test_the_state_rows_must_be_the_two_backends_in_order(tmp_path):
    doc = dict(make_service(tmp_path).settings_state())
    doc["backends"] = list(reversed(doc["backends"]))
    with pytest.raises(ValueError):
        api.validate_rewrite_settings_state(doc)
    doc = dict(make_service(tmp_path).settings_state())
    doc["backends"] = [doc["backends"][1]]
    with pytest.raises(ValueError):
        api.validate_rewrite_settings_state(doc)


def test_an_unpersistent_server_says_so():
    service = rs.RewriteService(rs.RewriteSettings(
        backend="local-llm", endpoint=ENDPOINT, model=MODEL,
        prompts=dict(PROMPTS), timeout=30.0),
        transport=lambda r: reply(),
        client_factory=lambda b, s, p, prefs: FakeClient(b, []))
    state = service.settings_state()
    assert state["persistent"] is False
    assert "no cache directory" in state["note"]
    # And a save here is a refusal, not a silent in-memory lie.
    with pytest.raises(OSError):
        rs.save_preferences(None, rs.RewritePreferences(backend="off"))


# --------------------------------------------------------------------------
# the saved document


def test_a_saved_choice_round_trips_and_is_written_private(tmp_path):
    service = make_service(tmp_path)
    prefs = rs.RewritePreferences(backend="pe-llamacpp", endpoint=ENDPOINT,
                                  model="Qwen3.8", timeout_s=42.5)
    service.apply_preferences(prefs)
    path = rs.preferences_path(str(tmp_path))
    assert path.endswith(rs.SETTINGS_FILENAME)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, oct(mode)
    loaded, warning = rs.load_preferences(path)
    assert warning == "" and loaded == prefs
    assert json.loads(open(path, encoding="utf-8").read())["backend"] == \
        "pe-llamacpp"
    # The file holds four values and no location.
    assert set(json.loads(open(path, encoding="utf-8").read())) == \
        set(rs.RewritePreferences.KEYS)
    assert BINARY not in open(path, encoding="utf-8").read()


def test_a_saved_choice_changes_which_rewriter_answers(tmp_path):
    built: list[str] = []
    service = make_service(tmp_path, local=True, pe=True, built=built)
    assert service.active_backend == "pe-llamacpp"
    service.apply_preferences(rs.RewritePreferences(backend="local-llm",
                                                    endpoint=ENDPOINT,
                                                    model=MODEL))
    assert service.active_backend == "local-llm"
    # Every change rebuilds BOTH available clients; the choice only decides
    # which one answers, so switching back never needs the operator.
    assert built == ["pe-llamacpp", "local-llm"] * 2
    service.apply_preferences(rs.RewritePreferences(backend="off"))
    assert service.active_backend is None
    assert service.status_block()["code"] == "rewrite_disabled"
    assert "Settings" in service.status_block()["reason"]


def test_an_unavailable_choice_is_refused_before_anything_is_built(tmp_path):
    """The amendment's ordering rule: check, then build, then write. A browser
    may not select the enhancer on a server that was started without it, and
    the attempt must leave no trace behind."""
    built: list[str] = []
    service = make_service(tmp_path, local=True, pe=False, built=built)
    built.clear()
    with pytest.raises(api.ApiError) as exc:
        service.check_preferences(rs.RewritePreferences(backend="pe-llamacpp"))
    assert exc.value.status == 422 and exc.value.code == "backend_unavailable"
    assert "flags" in exc.value.message
    assert built == [], "a refused choice must build nothing"
    with pytest.raises(api.ApiError):
        service.check_preferences(rs.RewritePreferences(backend="nonsense"))
    assert built == []
    # "off" is always allowed, and nothing is written by a refusal.
    service.check_preferences(rs.RewritePreferences(backend="off"))
    assert not os.path.exists(rs.preferences_path(str(tmp_path)))


def test_a_choice_of_off_survives_a_server_restart(tmp_path):
    """Off is a setting, not the absence of one: the flags must not silently
    re-enable a rewriter the user switched off."""
    service = make_service(tmp_path, local=True, pe=True)
    service.apply_preferences(rs.RewritePreferences(backend="off"))
    prefs, warning = rs.load_preferences(rs.preferences_path(str(tmp_path)))
    assert warning == "" and prefs is not None
    restarted = make_service(tmp_path, local=True, pe=True, preferences=prefs)
    assert restarted.active_backend is None
    assert restarted.status_block()["state"] == "unavailable"


def test_a_broken_saved_document_falls_back_and_warns(tmp_path, capsys):
    path = rs.preferences_path(str(tmp_path))
    for garbage in (b"", b"{", b'{"backend": 4}', b'{"backend": "gpt"}',
                    json.dumps({"backend": "local-llm", "endpoint": "http://x:1/",
                                "model": "m", "timeout_s": None}).encode(),
                    json.dumps({"backend": "local-llm", "endpoint": ENDPOINT,
                                "model": "m", "timeout_s": None,
                                "binary": BINARY}).encode()):
        with open(path, "wb") as handle:
            handle.write(garbage)
        prefs, warning = rs.load_preferences(path)
        assert prefs is None, garbage
        assert warning and "start-up defaults" in warning
        # One bounded line, never a traceback and never a path.
        assert "\n" not in warning.strip() and str(tmp_path) not in warning
        service = make_service(tmp_path)
        server.load_saved_preferences(service, cache_dir=str(tmp_path))
        assert service.active_backend == "local-llm", garbage
    assert "rewrite settings" in capsys.readouterr().err


def test_an_oversized_saved_document_is_ignored(tmp_path):
    path = rs.preferences_path(str(tmp_path))
    with open(path, "wb") as handle:
        handle.write(b'{"backend":"off",' + b"x" * rs.MAX_SETTINGS_BYTES)
    prefs, warning = rs.load_preferences(path)
    assert prefs is None and str(rs.MAX_SETTINGS_BYTES) in warning


def test_a_missing_file_is_not_a_warning(tmp_path):
    prefs, warning = rs.load_preferences(rs.preferences_path(str(tmp_path)))
    assert prefs is None and warning == ""
    assert rs.load_preferences(None) == (None, "")


# --------------------------------------------------------------------------
# the connection probe


def test_the_probe_asks_for_a_handful_of_tokens_and_reports_in_plain_words(tmp_path):
    seen: list[dict] = []

    def transport(request: bytes) -> bytes:
        seen.append(json.loads(request.decode("utf-8")))
        return reply("pong")

    service = make_service(tmp_path, transport=transport)
    result = service.test_connection(ENDPOINT, MODEL, 300.0)
    assert result["ok"] is True
    assert result["schema"] == api.REWRITE_TEST_SCHEMA
    assert result["detail"] == "ok: the service answered"
    assert result["elapsed_ms"] >= 0
    # The probe is a token or two, never a draft: a wrong endpoint comes back
    # in seconds rather than after a full rewrite.
    assert seen[0]["max_tokens"] == rs.TEST_MAX_TOKENS
    assert all("rewritten_prompt" not in json.dumps(sent) for sent in seen)


def test_the_probe_budget_is_capped_whatever_was_saved(tmp_path, monkeypatch):
    """A 3600-second rewrite timeout is a legitimate setting; making the page
    wait an hour to find out an address is wrong would not be."""
    captured: dict[str, object] = {}
    real = rc.RewriteClient

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(rc, "RewriteClient", spy)
    service = make_service(tmp_path, transport=lambda r: reply("pong"))
    service.test_connection(ENDPOINT, MODEL, 3600.0)
    assert captured["timeout"] == rs.TEST_TIMEOUT_CAP


def test_a_failed_probe_says_why_without_naming_a_host_or_a_binary(tmp_path):
    def transport(request: bytes) -> bytes:
        raise rc.RewriteClientError(
            "could not reach http://127.0.0.1:8079/v1/chat/completions: "
            f"connection refused (see {BINARY})")

    result = make_service(tmp_path, transport=transport).test_connection(
        ENDPOINT, MODEL, 5.0)
    assert result["ok"] is False
    assert result["schema"] == api.REWRITE_TEST_SCHEMA
    detail = result["detail"]
    assert "failed" in detail
    # The address became "the host" and no path survives: the page learns that
    # it failed and in what words, never where the service or the binary is.
    for leak in ("127.0.0.1", "://", BINARY, "/"):
        assert leak not in detail, detail


def test_a_probe_never_takes_the_rewrite_slot(tmp_path):
    """Saving a setting must not be blocked by a rewrite in flight, and the
    probe must not become a second rewrite competing for the one slot."""
    service = make_service(tmp_path, pe=True, transport=lambda r: reply("pong"))
    job = service._claim("t2i")            # the single slot, held deliberately
    try:
        assert service._current is job
        result = service.test_connection(ENDPOINT, MODEL, 5.0)
        assert service._current is job, "a probe must not disturb a rewrite"
    finally:
        with service._cond:
            service._current = None
            service._cond.notify_all()
    assert result["ok"] is True


def test_a_probe_refuses_a_non_loopback_address_before_connecting(tmp_path):
    service = make_service(tmp_path)
    with pytest.raises(rc.RewriteConfigError):
        service.test_connection("http://10.0.0.5:1234/v1/chat/completions", MODEL)


# --------------------------------------------------------------------------
# the routes


def call(raw: bytes, service=None):
    """One request through the real handler over an in-memory socket."""
    h = server._Handler.__new__(server._Handler)
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.connection = types.SimpleNamespace(getsockname=lambda: ("127.0.0.1", 8765))
    h.server = types.SimpleNamespace(
        bound_ip="127.0.0.1", bound_port=8765,
        assets={"app.js": b"console.log('x')"},
        web_runtime=api.WebRuntime(), rewrite=service)
    h.close_connection = False
    h.request_version = "HTTP/1.1"
    h.address_string = lambda: "127.0.0.1"
    h.log_message = lambda *a, **k: None
    h.handle_one_request()
    return h.wfile.getvalue().decode("latin-1")


def request(method: str, route: str, payload=None, service=None, *,
            origin=None) -> str:
    raw = b"" if payload is None else json.dumps(payload).encode()
    head = (f"{method} {route} HTTP/1.1\r\n"
            "Host: 127.0.0.1:8765\r\nX-Qwen-UI: 1\r\n")
    if origin is not None:
        head += f"Origin: {origin}\r\n"
    if payload is not None:
        head += f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n"
    head += "\r\n"
    return call(head.encode() + raw, service)


def responded(out: str) -> tuple[int, dict]:
    head, _, rest = out.partition("\r\n\r\n")
    return int(head.split("\r\n")[0].split(" ")[1]), json.loads(rest)


def test_get_settings_reads_the_choice_without_a_rewriter_enabled(tmp_path):
    """Off is precisely when the page has to be able to read and change it."""
    service = make_service(tmp_path, local=False, pe=False)
    status, payload = responded(request("GET", server.SETTINGS_ROUTE, None,
                                       service))
    assert status == 200
    assert [row["available"] for row in payload["backends"]] == [False, False]
    assert payload["backend"] is None


def test_get_settings_without_a_service_at_all_is_a_typed_503():
    status, payload = responded(request("GET", server.SETTINGS_ROUTE, None))
    assert status == 503 and payload["code"] == "settings_unavailable"


def test_methods_are_advertised_per_route():
    status, payload = responded(request("DELETE", server.SETTINGS_ROUTE))
    assert status == 405 and payload["code"] == "method_not_allowed"
    assert "GET, POST" in request("DELETE", server.SETTINGS_ROUTE)
    # The probe is POST-only: a GET must not be able to trigger a connection.
    out = request("GET", server.SETTINGS_TEST_ROUTE)
    assert " 405 " in out.split("\r\n")[0] and "Allow: POST" in out
    assert server.resolve_route("GET", server.SETTINGS_ROUTE) == (
        "settings_get", server.SETTINGS_ROUTE)
    assert server.resolve_route("POST", server.SETTINGS_ROUTE) == (
        "settings_set", server.SETTINGS_ROUTE)
    assert server.resolve_route("POST", server.SETTINGS_TEST_ROUTE) == (
        "settings_test", server.SETTINGS_TEST_ROUTE)


def test_a_foreign_origin_may_not_change_the_rewriter():
    """The Settings route is a write on a loopback-only server: the same-origin
    policy that guards uploads guards it too."""
    status, payload = responded(request(
        "POST", server.SETTINGS_ROUTE,
        json.loads(body()), origin="http://evil.test"))
    assert status == 403 and payload["code"] == "foreign_origin_denied"


def test_post_settings_switches_the_rewriter_and_answers_with_the_new_state(tmp_path):
    service = make_service(tmp_path, local=True, pe=True)
    status, payload = responded(request(
        "POST", server.SETTINGS_ROUTE,
        json.loads(body(backend="local-llm", endpoint=OTHER_ENDPOINT,
                        model="Other", timeout_s=240.0)), service))
    assert status == 200
    assert payload["backend"] == "local-llm"
    assert payload["saved"] == {"backend": "local-llm",
                                "endpoint": OTHER_ENDPOINT, "model": "Other",
                                "timeout_s": 240.0}
    # The override wins over the flags, for the page AND for the next start.
    assert payload["local"] == {"endpoint": OTHER_ENDPOINT, "model": "Other",
                                "timeout_s": 240.0}
    assert service.active_backend == "local-llm"
    saved, warning = rs.load_preferences(rs.preferences_path(str(tmp_path)))
    assert warning == "" and saved.backend == "local-llm"
    assert saved.endpoint == OTHER_ENDPOINT


def test_a_refused_post_leaves_the_server_exactly_as_it_was(tmp_path):
    """Every way the page can ask wrongly must be a NO-OP: the active rewriter,
    the choice in force and the file on disk all stay untouched, so a refused
    save can never half-apply."""
    service = make_service(tmp_path, local=True, pe=False)
    path = rs.preferences_path(str(tmp_path))
    before = service.preferences()
    for payload, code, status in (
            # The enhancer is not configured here: asking for it is a question
            # about files only the operator can place.
            (json.loads(body(backend="pe-llamacpp")), "backend_unavailable", 422),
            ({"binary": BINARY}, "invalid_envelope", 400),
            (json.loads(body(endpoint="http://0.0.0.0:1/v1")),
             "endpoint_unsupported", 422),
            (json.loads(body(model=None)), "backend_incomplete", 422),
            (json.loads(body(timeout_s=99999)), "invalid_envelope", 400)):
        got, answered = responded(request("POST", server.SETTINGS_ROUTE, payload,
                                          service))
        assert got == status, (payload, answered)
        assert answered["code"] == code, answered
        assert service.active_backend == "local-llm", code
        assert service.preferences() == before, code
        assert not os.path.exists(path), code


def test_a_save_the_cache_dir_refuses_is_reported_and_undone(tmp_path, monkeypatch):
    service = make_service(tmp_path, local=True, pe=True)

    def refuse(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(rs, "save_preferences", refuse)
    got, answered = responded(request(
        "POST", server.SETTINGS_ROUTE,
        json.loads(body(backend="local-llm")), service))
    assert got == 503 and answered["code"] == "settings_unavailable"
    # The page must not show a choice the file does not hold: the previous one
    # is back in force, so the NEXT start-up agrees with what was displayed.
    assert service.active_backend == "pe-llamacpp"
    assert service.preferences().backend is None
    assert not os.path.exists(rs.preferences_path(str(tmp_path)))


def test_post_test_reports_the_probe_result(tmp_path):
    service = make_service(tmp_path, transport=lambda r: reply("pong"))
    got, answered = responded(request(
        "POST", server.SETTINGS_TEST_ROUTE,
        {"schema": api.REWRITE_TEST_REQUEST_SCHEMA, "endpoint": ENDPOINT,
         "model": MODEL, "timeout_s": 5.0}, service))
    assert got == 200 and answered["ok"] is True
    assert answered["schema"] == api.REWRITE_TEST_SCHEMA
    got, answered = responded(request(
        "POST", server.SETTINGS_TEST_ROUTE,
        {"schema": api.REWRITE_TEST_REQUEST_SCHEMA, "endpoint": "http://x:1/",
         "model": MODEL, "timeout_s": 5.0}, service))
    assert got == 422 and answered["code"] == "endpoint_unsupported"


# --------------------------------------------------------------------------
# nothing that reaches the browser names a file or a command


def test_no_response_names_a_path_a_command_or_the_enhancer_binary(tmp_path):
    """Every document these routes can produce, swept — including the failure
    paths, which are the ones that quote an exception's text. The backend here
    fails the way a failed spawn does: naming its binary, its model and the
    options it was started with."""
    command = f"could not exec {BINARY} --no-webui -m {GGUF}: no such file"
    service = make_service(tmp_path, local=True, pe=True,
                           transport=lambda r: reply("pong"),
                           client=BrokenClient("pe-llamacpp", [], command))
    documents = [service.settings_state(), service.status_block(),
                 service.test_connection(ENDPOINT, MODEL, 5.0),
                 api.rewrite_test_result(False, command, 1)]
    try:
        service.submit({"task": "t2i", "prompt": "a cat", "draft_revision": 1,
                        "image_ids": []})
        raise AssertionError("the broken backend must fail the rewrite")
    except api.ApiError as exc:
        documents.append(exc.to_payload())
    for document in documents:
        text = json.dumps(document)
        # The PROGRAM NAME may appear (a greyed-out choice explains itself in
        # words) and so may the settings ROUTE; what may never appear is a
        # location on this machine or a fragment of the command line.
        for secret in (BINARY, GGUF, str(tmp_path), "--no-webui", "--",
                       "subprocess", "Popen", "llama_server", "model_path"):
            assert secret not in text, (secret, text[:400])




def test_the_status_block_points_at_the_route_not_at_a_file(tmp_path):
    block = make_service(tmp_path).status_block()
    assert block["settings"]["href"] == "/api/v1/settings/rewrite"
    assert block["settings"]["saved"] is False
    service = make_service(tmp_path)
    service.apply_preferences(rs.RewritePreferences(backend="off"))
    assert service.status_block()["settings"]["saved"] is True


# --------------------------------------------------------------------------
# which rewriter answers, once the choice is saved


def store_image(root, colour=(200, 30, 30)):
    """Store a real image and return its id (the store is the real store)."""
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), colour).save(buffer, "PNG")
    return reference_store.put_bytes(root, buffer.getvalue())


def rewrite(service, task="t2i", prompt="a cat", image_ids=()):
    return service.submit({"task": task, "prompt": prompt, "draft_revision": 1,
                           "image_ids": list(image_ids)}).payload


def test_an_edit_draft_uses_the_local_service_while_the_enhancer_is_active(tmp_path):
    """The enhancer has no edit weights, and the amendment refuses to answer an
    edit draft with the wrong model: the local service takes it and the answer
    says which one did. The saved choice is untouched — the text draft right
    afterwards still goes to the enhancer."""
    root = tmp_path / "store"
    ref = store_image(root)
    live: dict[str, FakeClient] = {}
    service = make_service(tmp_path, local=True, pe=True, clients=live,
                           references_dir=str(root))
    assert service.active_backend == "pe-llamacpp"
    payload = rewrite(service, "edit", "make it night", [ref])
    assert payload["backend"] == "local-llm"
    assert payload["rewritten_prompt"] == "rewritten by local-llm"
    assert live["pe-llamacpp"].requests == [], "the enhancer must not see an edit"
    assert live["local-llm"].requests == ["edit"]
    # A text draft still goes to the enhancer: the fallback is per task, not a
    # change of the saved choice.
    assert rewrite(service)["backend"] == "pe-llamacpp"
    assert service.preferences().backend is None


def test_an_edit_draft_is_refused_when_only_the_enhancer_exists(tmp_path):
    """No local service means no edit rewriter. Falling back to the enhancer
    would send reference images to weights that were never trained on them, so
    the refusal names what is missing instead of quietly guessing."""
    root = tmp_path / "store"
    ref = store_image(root)
    live: dict[str, FakeClient] = {}
    service = rs.RewriteService(
        rs.RewriteSettings.disabled(),
        pe=rs.PESettings(llama_server=BINARY, model_path=GGUF,
                         system_prompt="pe system prompt", timeout=300.0),
        cache_dir=str(tmp_path),
        client_factory=lambda b, s, p, prefs: live.setdefault(
            b, FakeClient(b, [])))
    assert service.available_backends() == ["pe-llamacpp"]
    assert service.active_backend == "pe-llamacpp"
    with pytest.raises(api.ApiError) as exc:
        rewrite(service, "edit", "make it night", [ref])
    assert exc.value.status == 422 and exc.value.code == "task_unsupported"
    assert "text-to-image" in exc.value.message
    assert "local LLM service" in exc.value.message
    assert live["pe-llamacpp"].requests == []
    # A text draft is untouched by that refusal.
    assert rewrite(service)["backend"] == "pe-llamacpp"


def test_a_settings_change_takes_effect_on_the_next_rewrite(tmp_path):
    """An in-flight rewrite finishes on the client it claimed. Rebuilding the
    map under the Condition changes what the NEXT caller gets; yanking the
    client out from under a running call would abort a rewrite the user has
    already waited thirty seconds for."""
    gate = threading.Event()
    inside = threading.Event()

    class SlowClient(FakeClient):
        def complete(self, messages, task, sampling=None):
            inside.set()
            assert gate.wait(10.0), "the rewrite was never released"
            return json.dumps({"rewritten_prompt": f"rewritten by {self.name}"})

    slow = SlowClient("pe-llamacpp", [])
    service = make_service(tmp_path, local=True, pe=True, clients={},
                           client=slow)
    assert service.active_backend == "pe-llamacpp"
    answers: list[dict] = []
    worker = threading.Thread(target=lambda: answers.append(rewrite(service)),
                              name="rewrite-under-test")
    worker.start()
    assert inside.wait(10.0), "the rewrite never reached its client"
    service.apply_preferences(rs.RewritePreferences(backend="local-llm",
                                                   endpoint=ENDPOINT,
                                                   model=MODEL))
    gate.set()
    worker.join(10.0)
    assert not worker.is_alive(), "the rewrite never finished"
    assert answers[0]["backend"] == "pe-llamacpp", "the running rewrite moved"
    # From here on the change is real, and what status reports agrees with it.
    assert service.active_backend == "local-llm"
    assert service.status_block()["backend"] == "local-llm"


def test_a_cancel_that_killed_the_backend_is_reported_as_a_cancel(tmp_path):
    """The enhancer frees the GPU by KILLING its server, so the worker can come
    back failed a moment before the waiting caller wakes. That failure is the
    consequence of the cancel: the answer the user gets is "cancelled", never
    "the rewriter did not answer before its deadline"."""
    class CancelledMidGeneration(FakeClient):
        subject = None          # set once the service exists

        def complete(self, messages, task, sampling=None):
            assert self.subject is not None
            assert self.subject.cancel() is True, "the job was not in flight"
            raise rc.RewriteTimeout("the rewrite was cancelled")

    client = CancelledMidGeneration("pe-llamacpp", [])
    subject = make_service(tmp_path, local=True, pe=True, client=client)
    client.subject = subject
    with pytest.raises(api.ApiError) as caught:
        rewrite(subject)
    assert caught.value.status == 503
    assert caught.value.code == "rewrite_cancelled"
    assert caught.value.message == "the rewrite was cancelled"
