"""Guarded tests for the server-side **Prompt rewriter** settings page.

There is no per-request rewriter pick in the Create view: the choice is
server-side, saved by the server, shared by every browser pointed at it, and it
is made **per draft kind** — one rewriter for a text-to-image draft, one for an
edit draft — plus ONE Off switch. A loopback-only Settings route reads and
changes it. Four properties carry the design and are what these tests pin:

* the POST form is CLOSED and FLAT (``t2i``, ``edit``, ``enabled`` plus the
  shared endpoint fields), and nothing in it is a filesystem path or an
  executable (the binary, both GGUFs and the system prompts stay launch flags);
* a request is validated, and refused, BEFORE any client is built or any file
  is written — a refusal leaves the server exactly as it was;
* a choice is honoured PER TASK: what the enhancer can answer here comes from
  the checkpoints that are installed, and a task nothing can answer is refused,
  never silently handed to the other task's rewriter;
* an old saved file with the single ``backend`` key still means something, and
  reading it never rewrites it.

No sockets, no torch, no GPU: the local service is an injected transport and
both enhancer clients are fakes.
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
I2I_GGUF = "/models/pe/PE-I2I-Q8_0.gguf"
MMPROJ = "/models/pe/mmproj-PE-I2I-BF16.gguf"

#: The six keys the saved/POSTed choice carries: two per-task choices, one
#: switch, and the three fields of the SHARED OpenAI-compatible endpoint.
CHOICE_KEYS = {"t2i", "edit", "enabled", "endpoint", "model", "timeout_s"}


def reply(content: str = '{"rewritten_prompt": "x"}') -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant",
                                                "content": content}}]}).encode()


class FakeClient:
    """Stands in for a backend client; records which drafts it was asked about.

    One instance is built per ``(task, backend)`` the service can run, except
    for the endpoint, which the service builds once and keys by both tasks — so
    ``requests`` shows exactly which draft kinds reached which rewriter.
    """

    def __init__(self, name, built=None, timeout=30.0):
        self.name = name
        self.built = [] if built is None else built
        self.timeout = timeout
        self.requests = []
        self.closes = 0

    def complete(self, messages, task, sampling=None):
        """The client contract: messages in, the raw answer text out."""
        self.requests.append(task)
        return json.dumps({"rewritten_prompt": f"rewritten by {self.name}"})

    def close(self):
        self.closes += 1
        return True


class BrokenClient(FakeClient):
    """A backend that fails the way a failed spawn does: quoting its own
    command line, which is exactly what must never reach the browser."""

    def __init__(self, name, built=None, detail=""):
        super().__init__(name, built)
        self.detail = detail

    def complete(self, messages, task, sampling=None):
        raise rc.RewriteClientError(self.detail)


def pe_settings(**overrides) -> rs.PESettings:
    """The enhancer's launch-flag configuration.

    ``model_path`` is the text checkpoint; the ``i2i_*`` pair is the edit
    checkpoint and its vision projector, and is what makes an edit draft
    offerable to the official enhancer at all.
    """
    base = dict(llama_server=BINARY, model_path=GGUF,
                system_prompts=dict(PROMPTS), timeout=300.0)
    base.update(overrides)
    return rs.PESettings(**base)


def make_service(tmp_path, *, local=True, pe=False, i2i=False, built=None,
                 clients=None, transport=None, preferences=None, client=None,
                 **overrides) -> rs.RewriteService:
    """A service with the given backends possible, writing settings in tmp.

    ``built`` records every build in order as ``(task, backend)`` and
    ``clients`` ends up holding the live client per built key, so a test can see
    which rewriter was asked about which draft kind.
    """
    local_settings = (rs.RewriteSettings(
        backend="local-llm", endpoint=ENDPOINT, model=MODEL, prompts=dict(PROMPTS),
        timeout=30.0, **overrides) if local else rs.RewriteSettings.disabled())
    pe_settings_value = None
    if pe:
        pe_settings_value = pe_settings(
            **({"i2i_model_path": I2I_GGUF, "i2i_mmproj_path": MMPROJ}
               if i2i else {}))
    made = built if built is not None else []

    def factory(task, backend, settings, pe_cfg, prefs):
        made.append((task, backend))
        instance = client if client is not None else FakeClient(
            f"{backend} for {task}", made)
        if clients is not None:
            clients[(task, backend)] = instance
        return instance

    return rs.RewriteService(local_settings, transport=transport,
                             pe=pe_settings_value, cache_dir=str(tmp_path),
                             preferences=preferences, client_factory=factory)


def body(**overrides) -> bytes:
    doc = {"schema": api.REWRITE_SETTINGS_SCHEMA, "t2i": "pe-llamacpp",
           "edit": "pe-llamacpp", "enabled": True, "endpoint": ENDPOINT,
           "model": MODEL, "timeout_s": 180.0}
    doc.update(overrides)
    return json.dumps(doc).encode()


def form(raw: bytes) -> dict:
    return api.build_rewrite_settings_request(raw)


def choices_of(service) -> dict:
    return service.active_by_task()


# --------------------------------------------------------------------------
# the posted form: closed, per task, and never a path


def test_the_canonical_form_is_accepted_and_normalised():
    checked = form(body(endpoint="  " + ENDPOINT + "  ", model="  Qwen3.8 "))
    assert checked == {"t2i": "pe-llamacpp", "edit": "pe-llamacpp",
                       "enabled": True, "endpoint": ENDPOINT, "model": MODEL,
                       "timeout_s": 180.0}
    # The two groups are independent: one draft on the enhancer, the other on
    # the shared endpoint, in a single save.
    split = form(body(t2i="pe-llamacpp", edit="local-llm"))
    assert (split["t2i"], split["edit"]) == ("pe-llamacpp", "local-llm")
    # An empty box means "no override", not "an empty address" — and the
    # shared fields may be left empty while no draft kind uses them.
    blank = form(body(endpoint="", model="", timeout_s=None))
    assert blank == {"t2i": "pe-llamacpp", "edit": "pe-llamacpp",
                     "enabled": True, "endpoint": None, "model": None,
                     "timeout_s": None}


def test_the_form_has_exactly_six_keys_and_a_schema_tag():
    for dropped in ("schema", "t2i", "edit", "enabled", "endpoint", "model",
                    "timeout_s"):
        doc = json.loads(body())
        del doc[dropped]
        with pytest.raises(api.ApiError) as exc:
            form(json.dumps(doc).encode())
        assert exc.value.status == 400, dropped
        assert exc.value.code == "invalid_envelope"


def test_a_seventh_key_is_refused_rather_than_dropped():
    doc = json.loads(body())
    doc["temperature"] = 0.5
    with pytest.raises(api.ApiError) as exc:
        form(json.dumps(doc).encode())
    assert exc.value.code == "invalid_envelope"
    assert "temperature" in exc.value.message


def test_the_old_single_backend_key_is_no_longer_a_wire_key():
    """The POST route never migrates: a client that still sends ``backend`` is
    told the rule, not quietly translated (only a FILE already on disk is)."""
    with pytest.raises(api.ApiError) as exc:
        api.validate_rewrite_settings({"backend": "local-llm",
                                       "t2i": "pe-llamacpp",
                                       "edit": "pe-llamacpp",
                                       "enabled": True, "endpoint": ENDPOINT,
                                       "model": MODEL, "timeout_s": None})
    assert exc.value.status == 400 and exc.value.code == "invalid_envelope"
    assert "backend" in exc.value.message
    assert "per draft kind" in exc.value.message


@pytest.mark.parametrize("key", [
    "binary", "llama_binary", "server_bin", "model_path", "gguf",
    "prompt_path", "cache_dir", "log_file", "exe", "folder", "path",
    "a/b", "C:\\Program Files\\llama.exe",
])
def test_a_path_shaped_key_is_refused_and_the_rule_is_named(key):
    """The absolute rule, not a denylist: a browser that may name one file can
    name any file on this machine, so NO key that looks like a location — by
    name or by shape — is ever accepted. That covers the edit checkpoint too."""
    doc = json.loads(body(t2i="pe-llamacpp", edit="pe-llamacpp"))
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
            {"t2i": "pe-llamacpp", "edit": "pe-llamacpp", "enabled": True,
             "endpoint": None, "model": None, "timeout_s": None,
             "binary": BINARY})


@pytest.mark.parametrize("endpoint", [
    "https://127.0.0.1:8079/v1/chat/completions",   # no TLS policy exists
    "http://example.com/v1/chat/completions",       # not loopback
    "http://127.0.0.1/chat/completions",            # no explicit port
    "http://user:pw@127.0.0.1:8079/v1",             # credentials
    "file:///etc/passwd",                           # not http at all
    "http://127.0.0.1:8079/v1?q=http://evil.test",   # not a URL
])
def test_an_endpoint_that_is_not_a_plain_loopback_url_is_refused(endpoint):
    # The endpoint is shared by both groups, so the rule is checked once for
    # whichever draft kind chose it.
    for task in ("t2i", "edit"):
        with pytest.raises(api.ApiError) as exc:
            form(body(**{task: "local-llm", "endpoint": endpoint}))
        assert exc.value.status == 422, (task, endpoint)
        assert exc.value.code == "endpoint_unsupported"
        assert any(name == "endpoint" for name, _ in exc.value.field_errors)


def test_a_path_IN_THE_URL_IS_fine_but_a_path_on_DISK_is_not():
    """The endpoint's URL path is required (``/v1/chat/completions``); what is
    forbidden is a FILESYSTEM path, so the two must not be confused."""
    assert form(body(endpoint="http://localhost:1234/a/b/c.json"))["endpoint"] == \
        "http://localhost:1234/a/b/c.json"


@pytest.mark.parametrize("task", ["t2i", "edit"])
@pytest.mark.parametrize("choice", ["gpt-4", "pe_llamacpp", "PE-LLAMACPP",
                                    "local", "", "off ", "off", True, 4])
def test_the_choice_is_closed_separately_for_each_draft_kind(task, choice):
    with pytest.raises(api.ApiError) as exc:
        form(body(**{task: choice}))
    assert exc.value.status == 422
    assert exc.value.code == "backend_unknown"
    # The sentence names WHICH group was asked, because the page shows two.
    assert task in exc.value.message


def test_off_is_a_switch_and_not_a_choice():
    """``off`` used to be a third value of one key; it is now the ``enabled``
    boolean, because "both tasks off" and "the switch is off" are one fact."""
    assert form(body(enabled=False))["enabled"] is False
    for bad in ("off", "true", 1, 0):
        with pytest.raises(api.ApiError) as exc:
            form(body(enabled=bad))
        assert exc.value.status == 400, bad
        assert exc.value.code == "invalid_envelope"
        assert any(name == "enabled" for name, _ in exc.value.field_errors)
    # A POST may leave nothing unset, not even the switch.
    with pytest.raises(api.ApiError) as exc:
        form(body(enabled=None))
    assert exc.value.status == 400 and exc.value.code == "invalid_envelope"


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_a_posted_choice_may_not_be_null_even_though_a_saved_one_may(task):
    with pytest.raises(api.ApiError) as exc:
        form(body(**{task: None}))
    assert exc.value.status == 400 and exc.value.code == "invalid_envelope"
    assert task in exc.value.message
    # A saved document, by contrast, may say "not chosen yet: use the default".
    unset = rs.RewritePreferences.from_dict(
        {"t2i": None, "edit": None, "enabled": None, "endpoint": None,
         "model": None, "timeout_s": None})
    assert unset.choice_for(task) is None and unset.off is False


def test_choosing_the_endpoint_for_either_task_without_an_address_is_refused():
    for task in ("t2i", "edit"):
        with pytest.raises(api.ApiError) as exc:
            form(body(**{task: "local-llm", "endpoint": None, "model": None}))
        assert exc.value.status == 422, task
        assert exc.value.code == "backend_incomplete"
        assert sorted(name for name, _ in exc.value.field_errors) == \
            ["endpoint", "model"]


def test_the_shared_endpoint_fields_may_be_saved_while_no_task_uses_them():
    """The address and model describe ONE service; filling them in while both
    groups use the official enhancer is how the page saves a draft of them."""
    filled = form(body(endpoint=ENDPOINT, model=MODEL))
    assert (filled["endpoint"], filled["model"]) == (ENDPOINT, MODEL)
    assert form(body(endpoint=None, model=None, timeout_s=None))["t2i"] == \
        "pe-llamacpp"


def test_a_half_empty_choice_is_completed_by_the_launch_flags(tmp_path):
    """An endpoint choice needs an address to be reachable, and the address may
    come from the flags rather than from the record. The rule therefore applies
    to the EFFECTIVE document the page posts, while the record the server holds
    (and republishes on the Settings page) may legitimately be half empty."""
    document = {"t2i": "local-llm", "edit": "pe-llamacpp", "enabled": True,
                "endpoint": None, "model": None, "timeout_s": None}
    service = make_service(tmp_path, local=True, pe=True, i2i=True,
                           preferences=rs.RewritePreferences(
                               choices={"t2i": "local-llm",
                                        "edit": "pe-llamacpp"}, enabled=True))
    assert service.active_by_task() == {"t2i": "local-llm",
                                        "edit": "pe-llamacpp"}
    # The page is publishable, and prefills the address it is really using.
    state = state_of(service)
    assert state["saved"] == document
    assert state["local"] == {"endpoint": ENDPOINT, "model": MODEL,
                              "timeout_s": 30.0}
    # The browser, by contrast, is the only place an address could come from:
    # the same six keys POSTed are a refusal, not a silent read of the flags.
    with pytest.raises(api.ApiError) as exc:
        form(json.dumps({**document,
                         "schema": api.REWRITE_SETTINGS_SCHEMA}).encode())
    assert exc.value.status == 422 and exc.value.code == "backend_incomplete"


def test_a_saved_file_may_choose_the_endpoint_without_naming_the_address(tmp_path):
    """The same record ON DISK must load: ``api.validate_rewrite_settings``
    documents that a saved record's endpoint and model are OVERRIDES, and a
    reader that demanded them would throw away a user's whole choice — and warn
    about a file that is legal by that rule."""
    document = {"t2i": "local-llm", "edit": "pe-llamacpp", "enabled": True,
                "endpoint": None, "model": None, "timeout_s": None}
    path = rs.preferences_path(str(tmp_path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    loaded, warning = rs.load_preferences(path)
    assert warning == "" and loaded is not None
    assert loaded.choice_for("t2i") == "local-llm" and loaded.endpoint is None
    service = make_service(tmp_path, local=True, pe=True, i2i=True,
                           preferences=loaded)
    assert service.active_by_task() == {"t2i": "local-llm",
                                        "edit": "pe-llamacpp"}


@pytest.mark.parametrize("timeout", [0, -1, 3601, "180", True])
def test_the_timeout_is_a_bounded_number_of_seconds(timeout):
    # ``Infinity`` is absent: the strict JSON parser rejects a non-finite
    # constant long before this validator sees it.
    with pytest.raises(api.ApiError) as exc:
        form(body(timeout_s=timeout))
    assert exc.value.status == 400
    assert exc.value.code == "invalid_envelope"
    assert any(name == "timeout_s" for name, _ in exc.value.field_errors)


def test_the_probe_carries_only_the_endpoint_fields():
    """A Test click always probes the SHARED service, so the probe cannot even
    name a draft kind, let alone ask for 9.5 GiB of enhancer on the GPU."""
    fields = api.build_rewrite_test_request(json.dumps(
        {"schema": api.REWRITE_TEST_REQUEST_SCHEMA, "endpoint": ENDPOINT,
         "model": MODEL, "timeout_s": 5.0}).encode())
    assert fields == {"endpoint": ENDPOINT, "model": MODEL, "timeout_s": 5.0}
    for extra in ({"t2i": "pe-llamacpp"}, {"enabled": True},
                  {"backend": "pe-llamacpp"}):
        with pytest.raises(api.ApiError) as exc:
            api.build_rewrite_test_request(json.dumps(
                {"schema": api.REWRITE_TEST_REQUEST_SCHEMA, "endpoint": ENDPOINT,
                 "model": MODEL, "timeout_s": 5.0, **extra}).encode())
        assert exc.value.status == 400 and exc.value.code == "invalid_envelope"
    for missing in ("endpoint", "model"):
        doc = {"schema": api.REWRITE_TEST_REQUEST_SCHEMA, "endpoint": ENDPOINT,
               "model": MODEL, "timeout_s": 5.0}
        doc[missing] = None
        with pytest.raises(api.ApiError) as exc:
            api.build_rewrite_test_request(json.dumps(doc).encode())
        assert exc.value.status == 422 and exc.value.code == "backend_incomplete"
    with pytest.raises(api.ApiError) as exc:
        api.build_rewrite_test_request(json.dumps(
            {"schema": api.REWRITE_TEST_REQUEST_SCHEMA, "endpoint": "http://x:1/",
             "model": MODEL, "timeout_s": 5.0}).encode())
    assert exc.value.status == 422 and exc.value.code == "endpoint_unsupported"


# --------------------------------------------------------------------------
# the state document the page is built from


def state_of(service):
    return api.validate_rewrite_settings_state(json.loads(
        json.dumps(service.settings_state())))


def task_choice(state, task, backend):
    row = [row for row in state["tasks"] if row["task"] == task][0]
    return [choice for choice in row["choices"] if choice["id"] == backend][0]


def test_the_state_offers_a_rewriter_per_draft_kind(tmp_path):
    service = make_service(tmp_path, local=True, pe=True, i2i=True)
    state = state_of(service)
    assert state["schema"] == api.REWRITE_SETTINGS_SCHEMA
    # One row per draft kind, in the server's order, each labelled with the
    # words this module publishes — the page never invents a heading.
    assert [(row["task"], row["label"]) for row in state["tasks"]] == [
        ("t2i", api.REWRITE_TASK_LABELS["t2i"]),
        ("edit", api.REWRITE_TASK_LABELS["edit"])]
    assert api.REWRITE_TASK_LABELS["edit"] == "Edit prompt enhance"
    for row in state["tasks"]:
        assert [choice["id"] for choice in row["choices"]] == \
            list(api.REWRITE_BACKENDS)
    # Every rewriter here is usable for both kinds, so nothing is greyed.
    for row in state["tasks"]:
        for choice in row["choices"]:
            assert choice["available"] is True and choice["reason"] == "", row
    assert task_choice(state, "edit", "local-llm")["label"] == \
        "OpenAI-compatible endpoint"
    assert task_choice(state, "t2i", "pe-llamacpp")["uses_gpu"] is True
    assert task_choice(state, "t2i", "local-llm")["uses_gpu"] is False
    # The backend rows too, in the server's order, with the GPU one saying so.
    assert [(row["id"], row["available"], row["uses_gpu"], row["tasks"])
            for row in state["backends"]] == [
        ("pe-llamacpp", True, True, ["t2i", "edit"]),
        ("local-llm", True, False, ["t2i", "edit"]),
    ]
    assert state["off"] == {"id": "off", "label": "Off", "available": True,
                            "reason": ""}
    assert state["persistent"] is True and "note" not in state
    # Which radio is checked in EACH group: a choice, never a null.
    assert state["choices"] == {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"}
    assert state["enabled"] is True
    # Prefill: what the LOCAL service is reached as RIGHT NOW (flags here), and
    # it is one shared service, so it names no rewriter at all.
    assert state["local"] == {"endpoint": ENDPOINT, "model": MODEL,
                              "timeout_s": 30.0}
    assert state["saved"] == {"t2i": None, "edit": None, "enabled": None,
                              "endpoint": None, "model": None,
                              "timeout_s": None}


def test_the_official_choice_is_greyed_for_edit_only_without_the_i2i_pair(tmp_path):
    """PE-T2I and PE-I2I are installed independently, so "no official rewriter"
    is a per-draft-kind fact and the page must be able to show exactly that."""
    service = make_service(tmp_path, local=True, pe=True)
    state = state_of(service)
    official_for_edit = task_choice(state, "edit", "pe-llamacpp")
    assert official_for_edit["available"] is False
    assert official_for_edit["reason"] and "image-edit" in official_for_edit["reason"]
    # The text group is untouched: same rewriter, still offered, no excuse.
    assert task_choice(state, "t2i", "pe-llamacpp")["available"] is True
    # A greyed-out choice still explains itself; the page never hides options.
    assert task_choice(state, "edit", "local-llm")["available"] is True
    # Capability is not choice: the group still shows the official radio
    # checked, and `active_by_task` is what says the endpoint answers edit.
    assert state["choices"] == {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"}
    assert service.active_by_task() == {"t2i": "pe-llamacpp",
                                        "edit": "local-llm"}
    # The reason renders in a browser, so it names no path — the closed
    # validator enforces that on the whole document.
    assert "/" not in official_for_edit["reason"]
    assert [row["tasks"] for row in state["backends"]] == [["t2i"],
                                                           ["t2i", "edit"]]


def test_a_rewriter_the_flags_cannot_run_is_greyed_for_every_draft_kind(tmp_path):
    service = make_service(tmp_path, local=True, pe=False)
    state = state_of(service)
    pe_row = state["backends"][0]
    assert pe_row["available"] is False
    assert pe_row["reason"] and "flags" in pe_row["reason"]
    for task in ("t2i", "edit"):
        choice = task_choice(state, task, "pe-llamacpp")
        assert choice["available"] is False and choice["reason"]
    assert service.active_by_task() == {"t2i": "local-llm",
                                        "edit": "local-llm"}


def test_a_server_with_no_rewriter_at_all_still_answers_the_page(tmp_path):
    service = make_service(tmp_path, local=False, pe=False)
    state = state_of(service)
    assert [row["available"] for row in state["backends"]] == [False, False]
    for task in ("t2i", "edit"):
        for backend in api.REWRITE_BACKENDS:
            assert task_choice(state, task, backend)["available"] is False
    # The page still needs a checked radio in both groups, so the default is
    # shown even where it cannot be honoured.
    assert state["choices"] == {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"}
    assert service.active_by_task() == {"t2i": None, "edit": None}


def test_the_state_never_carries_a_path_even_when_every_path_is_configured(tmp_path):
    """The binary, both GGUFs, the vision projector, the system prompts and the
    enhancer log all live on disk; none of those names may cross the wire,
    because the page would otherwise tell a browser where the operator's files
    are."""
    for service in (make_service(tmp_path, local=True, pe=True, i2i=True),
                    make_service(tmp_path, local=True, pe=True)):
        for document in (service.settings_state(), service.status_block()):
            text = json.dumps(document)
            for secret in (BINARY, GGUF, I2I_GGUF, MMPROJ, str(tmp_path),
                           "system_prompt", ".log"):
                assert secret not in text, (secret, sorted(document))
    assert state_of(make_service(tmp_path, local=True, pe=True,
                                 i2i=True))["backends"][0]["available"] is True


def test_a_state_with_an_extra_key_is_refused_before_it_is_sent(tmp_path):
    service = make_service(tmp_path)
    doc = dict(service.settings_state())
    for extra in ({"binary": BINARY}, {"log_path": "/tmp/x.log"},
                  {"backend": "local-llm"}, {"backend_extra": 1}):
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


def test_the_task_rows_are_closed_too(tmp_path):
    """The per-task groups are validated by the same closed-shape rule as the
    backend rows: a page that had to guess which flag gates which option could
    not be checked, so the server answers and the validator pins the answer."""
    base = dict(make_service(tmp_path, local=True, pe=True,
                             i2i=True).settings_state())
    for mutate in (
            lambda rows: list(reversed(rows)),                       # order
            lambda rows: rows[:1],                                   # one group
            lambda rows: [{**rows[0], "task": "edit"}, rows[1]],     # wrong kind
            lambda rows: [{**rows[0], "label": "Rewriter?"}, rows[1]],
            lambda rows: [{**rows[0], "choices": [rows[0]["choices"][0]]},
                          rows[1]],                                  # one option
            lambda rows: [{**rows[0], "choices": list(
                reversed(rows[0]["choices"]))}, rows[1]],            # option order
            lambda rows: [{**rows[0], "choices": [
                {**rows[0]["choices"][0], "available": False, "reason": ""},
                rows[0]["choices"][1]]}, rows[1]],                   # no excuse
            lambda rows: [{**rows[0], "choices": [
                {**rows[0]["choices"][0], "uses_gpu": False},
                rows[0]["choices"][1]]}, rows[1]],                   # lies about GPU
    ):
        doc = dict(base)
        doc["tasks"] = mutate([dict(row) for row in base["tasks"]])
        with pytest.raises(ValueError):
            api.validate_rewrite_settings_state(doc)
    for task in ("t2i", "edit"):
        doc = dict(base)
        doc["choices"] = {**base["choices"], task: None}
        with pytest.raises(ValueError):
            api.validate_rewrite_settings_state(doc)


def test_an_unpersistent_server_says_so():
    service = rs.RewriteService(rs.RewriteSettings(
        backend="local-llm", endpoint=ENDPOINT, model=MODEL,
        prompts=dict(PROMPTS), timeout=30.0),
        transport=lambda r: reply(),
        client_factory=lambda task, backend, s, p, pr: FakeClient(backend, []))
    state = service.settings_state()
    assert state["persistent"] is False
    assert "no cache directory" in state["note"]
    # And a save here is a refusal, not a silent in-memory lie.
    with pytest.raises(OSError):
        rs.save_preferences(None, rs.RewritePreferences(
            choices={"t2i": "pe-llamacpp", "edit": "pe-llamacpp"},
            enabled=False))


# --------------------------------------------------------------------------
# the saved document, and the old single-backend file it grew out of


def test_a_saved_choice_round_trips_and_is_written_private(tmp_path):
    service = make_service(tmp_path)
    saved = rs.RewritePreferences(choices={"t2i": "pe-llamacpp",
                                           "edit": "local-llm"},
                                  enabled=True, endpoint=ENDPOINT,
                                  model="Qwen3.8", timeout_s=42.5)
    service.apply_preferences(saved)
    path = rs.preferences_path(str(tmp_path))
    assert path.endswith(rs.SETTINGS_FILENAME)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, oct(mode)
    loaded, warning = rs.load_preferences(path)
    assert warning == "" and loaded == saved
    assert loaded.choice_for("t2i") == "pe-llamacpp"
    assert loaded.choice_for("edit") == "local-llm"
    on_disk = json.loads(open(path, encoding="utf-8").read())
    # The file we write is a file this server accepts with no translation step:
    # six values, and no legacy ``backend`` key.
    assert set(on_disk) == CHOICE_KEYS
    assert rs.LEGACY_BACKEND_KEY not in on_disk
    assert BINARY not in open(path, encoding="utf-8").read()


@pytest.mark.parametrize("legacy,chosen,enabled", [
    ("local-llm", ("local-llm", "local-llm"), True),
    ("pe-llamacpp", ("pe-llamacpp", "pe-llamacpp"), True),
    ("off", (None, None), False),
    (None, (None, None), None),
])
def test_an_old_single_backend_file_becomes_a_choice_for_both_tasks(
        tmp_path, legacy, chosen, enabled):
    """The old key meant one thing; per-task choices mean two. The migration is
    a pure key translation: the old value applies to BOTH kinds, ``off`` was a
    switch rather than a rewriter, and "not chosen" stays not chosen."""
    document = {"backend": legacy, "endpoint": ENDPOINT, "model": MODEL,
                "timeout_s": 60.0}
    migrated = rs.migrate_saved(document)
    assert rs.LEGACY_BACKEND_KEY not in migrated
    assert (migrated["t2i"], migrated["edit"]) == chosen
    assert migrated["enabled"] is enabled
    assert (migrated["endpoint"], migrated["model"], migrated["timeout_s"]) == \
        (ENDPOINT, MODEL, 60.0)
    loaded = rs.RewritePreferences.from_dict(migrated)
    assert (loaded.choice_for("t2i"), loaded.choice_for("edit")) == chosen
    assert loaded.off is (enabled is False)
    # And through the file loader, which is where a real server meets it.
    path = rs.preferences_path(str(tmp_path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    from_disk, warning = rs.load_preferences(path)
    assert warning == "" and from_disk == loaded


def test_an_old_off_file_switches_the_restarted_server_off(tmp_path):
    path = rs.preferences_path(str(tmp_path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"backend": "off", "endpoint": ENDPOINT, "model": MODEL,
                   "timeout_s": None}, handle)
    loaded, warning = rs.load_preferences(path)
    assert warning == "" and loaded.off is True
    service = make_service(tmp_path, local=True, pe=True, i2i=True,
                           preferences=loaded)
    assert service.active_by_task() == {"t2i": None, "edit": None}
    assert service.status_block()["state"] == "unavailable"
    # A switch, not a choice: the groups still show the official radios ready
    # for whenever the switch comes back on.
    assert service.settings_state()["choices"] == {"t2i": "pe-llamacpp",
                                                   "edit": "pe-llamacpp"}


def test_a_leftover_backend_key_beside_per_task_choices_is_dropped_not_obeyed(tmp_path):
    """A hand-merged file must not have the old key win over the new ones."""
    document = {"backend": "local-llm", "t2i": "pe-llamacpp",
                "edit": "local-llm", "enabled": True, "endpoint": ENDPOINT,
                "model": MODEL, "timeout_s": None}
    assert rs.migrate_saved(document) == {key: value for key, value
                                          in document.items()
                                          if key != rs.LEGACY_BACKEND_KEY}
    path = rs.preferences_path(str(tmp_path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    loaded, warning = rs.load_preferences(path)
    assert warning == ""
    assert (loaded.choice_for("t2i"), loaded.choice_for("edit")) == \
        ("pe-llamacpp", "local-llm")


def test_reading_an_old_file_never_rewrites_it(tmp_path):
    """Migration runs on READ. The operator's file keeps its shape until the
    USER saves again: starting a server must not have write side effects."""
    path = rs.preferences_path(str(tmp_path))
    legacy = json.dumps({"backend": "local-llm", "endpoint": ENDPOINT,
                         "model": MODEL, "timeout_s": 60.0}, sort_keys=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(legacy)
    service = make_service(tmp_path, local=True, pe=True)
    assert server.load_saved_preferences(service, cache_dir=str(tmp_path))
    assert open(path, encoding="utf-8").read() == legacy
    assert service.active_by_task() == {"t2i": "local-llm", "edit": "local-llm"}
    # Saving from Settings is what converts it, and then the old key is gone.
    service.apply_preferences(service.preferences())
    assert rs.LEGACY_BACKEND_KEY not in open(path, encoding="utf-8").read()


def test_migration_is_a_no_op_for_anything_that_is_not_an_old_choice():
    for document in (None, [], "off", {"endpoint": ENDPOINT},
                     {"t2i": "pe-llamacpp", "edit": "pe-llamacpp",
                      "enabled": True, "endpoint": None, "model": None,
                      "timeout_s": None}):
        assert rs.migrate_saved(document) == document


def test_a_saved_choice_changes_which_rewriter_answers(tmp_path):
    built: list[tuple[str, str]] = []
    service = make_service(tmp_path, local=True, pe=True, i2i=True, built=built)
    assert choices_of(service) == {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"}
    # Every change rebuilds every client this server can run; the choice only
    # decides which one answers, so switching back never needs the operator.
    per_rebuild = [("t2i", "pe-llamacpp"), ("t2i", "local-llm"),
                   ("edit", "pe-llamacpp")]
    assert built == per_rebuild
    service.apply_preferences(rs.RewritePreferences(
        choices={"t2i": "local-llm", "edit": "local-llm"}, enabled=True,
        endpoint=ENDPOINT, model=MODEL))
    assert choices_of(service) == {"t2i": "local-llm", "edit": "local-llm"}
    assert built == per_rebuild * 2
    service.apply_preferences(rs.RewritePreferences(
        choices={"t2i": "pe-llamacpp", "edit": "pe-llamacpp"}, enabled=False))
    assert choices_of(service) == {"t2i": None, "edit": None}
    assert service.active_backend is None
    assert service.status_block()["code"] == "rewrite_disabled"
    assert "Settings" in service.status_block()["reason"]


def test_the_endpoint_service_is_built_once_and_shared_by_both_groups(tmp_path):
    """One OpenAI-compatible service, two groups: a second client would mean a
    second connection pool for the same address."""
    built: list[tuple[str, str]] = []
    live: dict = {}
    service = make_service(tmp_path, local=True, pe=True, i2i=True, built=built,
                           clients=live)
    assert built == [("t2i", "pe-llamacpp"), ("t2i", "local-llm"),
                     ("edit", "pe-llamacpp")]
    assert service._clients[("t2i", "local-llm")] is \
        service._clients[("edit", "local-llm")]
    # Two checkpoints are two llama-server processes, so they are two clients.
    assert service._clients[("t2i", "pe-llamacpp")] is not \
        service._clients[("edit", "pe-llamacpp")]


def test_an_unavailable_choice_is_refused_before_anything_is_built(tmp_path):
    """The ordering rule: check, then build, then write. A browser may not
    select the enhancer on a server that was started without it, and the
    attempt must leave no trace behind."""
    built: list[tuple[str, str]] = []
    service = make_service(tmp_path, local=True, pe=False, built=built)
    built.clear()
    with pytest.raises(api.ApiError) as exc:
        service.check_preferences(rs.RewritePreferences(
            choices={"t2i": "pe-llamacpp", "edit": "local-llm"}, enabled=True,
            endpoint=ENDPOINT, model=MODEL))
    assert exc.value.status == 422 and exc.value.code == "backend_unavailable"
    assert "flags" in exc.value.message
    # The refusal says the same thing the greyed-out radio says.
    assert exc.value.message == service.unavailable_reason("pe-llamacpp", "t2i")
    assert built == [], "a refused choice must build nothing"
    # Switched off is always allowed: nothing is being asked of any rewriter.
    service.check_preferences(rs.RewritePreferences(
        choices={"t2i": "pe-llamacpp", "edit": "pe-llamacpp"}, enabled=False))
    # The endpoint answer is available here, so that save passes.
    service.check_preferences(rs.RewritePreferences(
        choices={"t2i": "local-llm", "edit": "local-llm"}, enabled=True,
        endpoint=ENDPOINT, model=MODEL))
    assert not os.path.exists(rs.preferences_path(str(tmp_path)))


def test_a_choice_the_server_cannot_honour_falls_back_and_says_so(tmp_path):
    """A SAVED choice is different from a POST: nobody is waiting for a reply,
    so startup keeps the choice, uses what works, and reports it."""
    path = rs.preferences_path(str(tmp_path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"t2i": "pe-llamacpp", "edit": "pe-llamacpp",
                   "enabled": True, "endpoint": ENDPOINT, "model": MODEL,
                   "timeout_s": None}, handle)
    loaded, warning = rs.load_preferences(path)
    assert warning == ""
    service = make_service(tmp_path, local=True, pe=True, preferences=loaded)
    # t2i is honoured; edit has no official checkpoint here, so the endpoint
    # answers it — and the saved choice is left alone.
    assert service.active_by_task() == {"t2i": "pe-llamacpp",
                                        "edit": "local-llm"}
    assert service.preferences().choice_for("edit") == "pe-llamacpp"
    assert service.effective_choices() == {"t2i": "pe-llamacpp",
                                           "edit": "pe-llamacpp"}
    assert service.backend_tasks("pe-llamacpp") == ["t2i"]
    assert service.backend_tasks("local-llm") == ["t2i", "edit"]


def test_a_choice_of_off_survives_a_server_restart(tmp_path):
    """Off is a setting, not the absence of one: the flags must not silently
    re-enable a rewriter the user switched off."""
    service = make_service(tmp_path, local=True, pe=True, i2i=True)
    service.apply_preferences(rs.RewritePreferences(
        choices={"t2i": "pe-llamacpp", "edit": "pe-llamacpp"}, enabled=False,
        endpoint=ENDPOINT, model=MODEL))
    loaded, warning = rs.load_preferences(rs.preferences_path(str(tmp_path)))
    assert warning == "" and loaded is not None and loaded.off is True
    restarted = make_service(tmp_path, local=True, pe=True, i2i=True,
                             preferences=loaded)
    assert restarted.active_by_task() == {"t2i": None, "edit": None}
    assert restarted.available is False
    block = restarted.status_block()
    assert block["state"] == "unavailable" and "switched off" in block["reason"]
    # One switch means BOTH drafts go quiet, with the off reason, not a
    # per-task refusal.
    with pytest.raises(api.ApiError) as exc:
        restarted.submit({"task": "t2i", "prompt": "a cat",
                          "draft_revision": 1, "image_ids": []})
    assert (exc.value.status, exc.value.code) == (503, "rewrite_disabled")


def test_a_broken_saved_document_falls_back_and_warns(tmp_path, capsys):
    path = rs.preferences_path(str(tmp_path))
    for garbage in (b"", b"{",
                    b'{"t2i": 4}',
                    json.dumps({"t2i": "gpt-4", "edit": "local-llm",
                                "enabled": True, "endpoint": ENDPOINT,
                                "model": "m", "timeout_s": None}).encode(),
                    json.dumps({"t2i": "local-llm", "edit": "local-llm",
                                "enabled": True, "endpoint": "http://x:1/",
                                "model": "m", "timeout_s": None}).encode(),
                    json.dumps({"t2i": "local-llm", "edit": "local-llm",
                                "enabled": True, "endpoint": ENDPOINT,
                                "model": "m", "timeout_s": None,
                                "binary": BINARY}).encode(),
                    json.dumps({"t2i": "pe-llamacpp", "edit": "pe-llamacpp",
                                "enabled": "off", "endpoint": None,
                                "model": None, "timeout_s": None}).encode(),
                    # Migration must not launder an old file that is also bad.
                    json.dumps({"backend": "local-llm",
                                "endpoint": "http://x:1/", "model": "m",
                                "timeout_s": None}).encode()):
        with open(path, "wb") as handle:
            handle.write(garbage)
        loaded, warning = rs.load_preferences(path)
        assert loaded is None, garbage
        assert warning and "start-up defaults" in warning
        # One bounded line, never a traceback and never a path.
        assert "\n" not in warning.strip() and str(tmp_path) not in warning
        service = make_service(tmp_path)
        server.load_saved_preferences(service, cache_dir=str(tmp_path))
        assert service.active_by_task() == {"t2i": "local-llm",
                                            "edit": "local-llm"}, garbage
    assert "rewrite settings" in capsys.readouterr().err


def test_an_oversized_saved_document_is_ignored(tmp_path):
    path = rs.preferences_path(str(tmp_path))
    with open(path, "wb") as handle:
        handle.write(b'{"enabled":true,' + b"x" * rs.MAX_SETTINGS_BYTES)
    loaded, warning = rs.load_preferences(path)
    assert loaded is None and str(rs.MAX_SETTINGS_BYTES) in warning


def test_a_missing_file_is_not_a_warning(tmp_path):
    loaded, warning = rs.load_preferences(rs.preferences_path(str(tmp_path)))
    assert loaded is None and warning == ""
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
    assert payload["choices"] == {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"}
    assert payload["enabled"] is True
    # Both groups are greyed everywhere, with a reason each.
    for row in payload["tasks"]:
        for choice in row["choices"]:
            assert choice["available"] is False and choice["reason"]


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
        json.loads(body(t2i="local-llm", edit="local-llm")),
        origin="http://evil.test"))
    assert status == 403 and payload["code"] == "foreign_origin_denied"


def test_post_settings_switches_the_rewriter_and_answers_with_the_new_state(tmp_path):
    service = make_service(tmp_path, local=True, pe=True, i2i=True)
    status, payload = responded(request(
        "POST", server.SETTINGS_ROUTE,
        json.loads(body(t2i="local-llm", edit="local-llm",
                        endpoint=OTHER_ENDPOINT, model="Other",
                        timeout_s=240.0)), service))
    assert status == 200
    assert payload["choices"] == {"t2i": "local-llm", "edit": "local-llm"}
    assert payload["saved"] == {"t2i": "local-llm", "edit": "local-llm",
                                "enabled": True, "endpoint": OTHER_ENDPOINT,
                                "model": "Other", "timeout_s": 240.0}
    # The override wins over the flags, for the page AND for the next start.
    assert payload["local"] == {"endpoint": OTHER_ENDPOINT, "model": "Other",
                                "timeout_s": 240.0}
    assert service.active_by_task() == {"t2i": "local-llm", "edit": "local-llm"}
    saved, warning = rs.load_preferences(rs.preferences_path(str(tmp_path)))
    assert warning == "" and saved.choice_for("edit") == "local-llm"
    assert saved.endpoint == OTHER_ENDPOINT


def test_post_settings_can_split_the_two_draft_kinds(tmp_path):
    """The whole point of the change: one save, two answers, each named."""
    service = make_service(tmp_path, local=True, pe=True, i2i=True)
    status, payload = responded(request(
        "POST", server.SETTINGS_ROUTE,
        json.loads(body(t2i="pe-llamacpp", edit="local-llm")), service))
    assert status == 200
    assert payload["choices"] == {"t2i": "pe-llamacpp", "edit": "local-llm"}
    assert service.active_by_task() == {"t2i": "pe-llamacpp",
                                        "edit": "local-llm"}
    assert service.status_block()["active_by_task"] == {
        "t2i": "pe-llamacpp", "edit": "local-llm"}


def test_a_refused_post_leaves_the_server_exactly_as_it_was(tmp_path):
    """Every way the page can ask wrongly must be a NO-OP: the rewriter
    answering each draft kind, the choice in force and the file on disk all
    stay untouched, so a refused save can never half-apply."""
    built: list[tuple[str, str]] = []
    service = make_service(tmp_path, local=True, pe=False, built=built)
    built.clear()                 # only what a REFUSAL builds is of interest
    path = rs.preferences_path(str(tmp_path))
    before = service.preferences()
    for payload, code, status in (
            # The enhancer is not configured here: asking for it, for either
            # draft kind, is a question about files only the operator can place.
            (json.loads(body(t2i="pe-llamacpp", edit="local-llm")),
             "backend_unavailable", 422),
            (json.loads(body(t2i="local-llm", edit="pe-llamacpp")),
             "backend_unavailable", 422),
            # The legacy key is not a wire key any more.
            (json.loads(body(backend="local-llm")), "invalid_envelope", 400),
            ({"binary": BINARY}, "invalid_envelope", 400),
            (json.loads(body(enabled="off")), "invalid_envelope", 400),
            (json.loads(body(t2i="local-llm",
                             endpoint="http://0.0.0.0:1/v1")),
             "endpoint_unsupported", 422),
            (json.loads(body(edit="local-llm", model=None, endpoint=None)),
             "backend_incomplete", 422),
            (json.loads(body(timeout_s=99999)), "invalid_envelope", 400)):
        got, answered = responded(request("POST", server.SETTINGS_ROUTE, payload,
                                          service))
        assert got == status, (payload, answered)
        assert answered["code"] == code, answered
        assert service.active_by_task() == {"t2i": "local-llm",
                                            "edit": "local-llm"}, code
        assert service.preferences() == before, code
        assert built == [], f"a refused save must build no client: {code}"
        assert not os.path.exists(path), code


def test_a_save_the_cache_dir_refuses_is_reported_and_undone(tmp_path, monkeypatch):
    service = make_service(tmp_path, local=True, pe=True, i2i=True)

    def refuse(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(rs, "save_preferences", refuse)
    got, answered = responded(request(
        "POST", server.SETTINGS_ROUTE,
        json.loads(body(t2i="local-llm", edit="local-llm")), service))
    assert got == 503 and answered["code"] == "settings_unavailable"
    # The page must not show a choice the file does not hold: the previous one
    # is back in force for BOTH groups, so the NEXT start-up agrees with what
    # was displayed.
    assert service.active_by_task() == {"t2i": "pe-llamacpp",
                                        "edit": "pe-llamacpp"}
    assert service.preferences().choice_for("t2i") is None
    assert service.preferences().choice_for("edit") is None
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
    got, answered = responded(request(
        "POST", server.SETTINGS_TEST_ROUTE,
        {"schema": api.REWRITE_TEST_REQUEST_SCHEMA, "endpoint": ENDPOINT,
         "model": None, "timeout_s": 5.0}, service))
    assert got == 422 and answered["code"] == "backend_incomplete"


# --------------------------------------------------------------------------
# nothing that reaches the browser names a file or a command


def test_no_response_names_a_path_a_command_or_the_enhancer_binary(tmp_path):
    """Every document these routes can produce, swept — including the failure
    paths, which are the ones that quote an exception's text. The backend here
    fails the way a failed spawn does: naming its binary, its model and the
    options it was started with."""
    command = (f"could not exec {BINARY} --no-webui -m {GGUF} "
               f"--mmproj {I2I_GGUF}: no such file")
    service = make_service(tmp_path, local=True, pe=True, i2i=True,
                           transport=lambda r: reply("pong"),
                           client=BrokenClient("pe-llamacpp", [], command))
    documents = [service.settings_state(), service.status_block(),
                 service.test_connection(ENDPOINT, MODEL, 5.0),
                 api.rewrite_test_result(False, command, 1)]
    # A server with only the text checkpoint explains itself per draft kind;
    # that sentence is rendered in a browser, so it must be path-free too.
    documents.append(make_service(tmp_path / "other", local=True,
                                  pe=True).settings_state())
    for task, images in (("t2i", []), ("edit", [store_image(tmp_path / "store")])):
        try:
            service.submit({"task": task, "prompt": "a cat",
                            "draft_revision": 1, "image_ids": images})
            raise AssertionError("the broken backend must fail the rewrite")
        except api.ApiError as exc:
            documents.append(exc.to_payload())
    for document in documents:
        text = json.dumps(document)
        # The PROGRAM NAME may appear (a greyed-out choice explains itself in
        # words) and so may the settings ROUTE; what may never appear is a
        # location on this machine or a fragment of the command line.
        for secret in (BINARY, GGUF, I2I_GGUF, MMPROJ, str(tmp_path),
                       "--no-webui", "--", "subprocess", "Popen",
                       "llama_server", "model_path", "i2i_model_path"):
            assert secret not in text, (secret, text[:400])


def test_the_status_block_points_at_the_route_not_at_a_file(tmp_path):
    block = make_service(tmp_path).status_block()
    assert block["settings"]["href"] == "/api/v1/settings/rewrite"
    assert block["settings"]["saved"] is False
    service = make_service(tmp_path)
    service.apply_preferences(rs.RewritePreferences(
        choices={"t2i": "pe-llamacpp", "edit": "pe-llamacpp"}, enabled=False))
    assert service.status_block()["settings"]["saved"] is True


# --------------------------------------------------------------------------
# which rewriter answers each draft kind, once the choice is saved


def store_image(root, colour=(200, 30, 30)):
    """Store a real image and return its id (the store is the real store)."""
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), colour).save(buffer, "PNG")
    return reference_store.put_bytes(root, buffer.getvalue())


def rewrite(service, task="t2i", prompt="a cat", image_ids=()):
    return service.submit({"task": task, "prompt": prompt, "draft_revision": 1,
                           "image_ids": list(image_ids)}).payload


def test_each_draft_kind_is_answered_by_its_own_rewriter(tmp_path):
    """The split actually routes: a t2i draft goes to the official text
    checkpoint, an edit draft to the endpoint, and neither sees the other's
    draft — a different rewriter is a different prompt contract."""
    root = tmp_path / "store"
    ref = store_image(root)
    live: dict = {}
    service = make_service(tmp_path, local=True, pe=True, i2i=True, clients=live,
                           references_dir=str(root),
                           preferences=rs.RewritePreferences(
                               choices={"t2i": "pe-llamacpp",
                                        "edit": "local-llm"},
                               enabled=True, endpoint=ENDPOINT, model=MODEL))
    assert rewrite(service)["backend"] == "pe-llamacpp"
    edit = rewrite(service, "edit", "make it night", [ref])
    assert edit["backend"] == "local-llm"
    # ONE endpoint object keyed by both draft kinds: it was built for the text
    # group, and it is the same object that answered the edit draft.
    assert service._clients[("edit", "local-llm")] is \
        service._clients[("t2i", "local-llm")]
    assert edit["rewritten_prompt"] == "rewritten by local-llm for t2i"
    assert live[("t2i", "pe-llamacpp")].requests == ["t2i"]
    assert live[("edit", "pe-llamacpp")].requests == [], \
        "the edit weights must not see a draft the choice sent elsewhere"
    assert live[("t2i", "local-llm")].requests == ["edit"], \
        "the shared endpoint client saw the edit draft once"
    # The choice is unchanged by any of this.
    assert service.preferences().choice_for("edit") == "local-llm"


def test_an_edit_draft_can_be_answered_by_the_official_edit_checkpoint(tmp_path):
    """With both files installed, the edit group's official radio is real work:
    the edit client answers, the text client stays untouched."""
    root = tmp_path / "store"
    ref = store_image(root)
    live: dict = {}
    service = make_service(tmp_path, local=True, pe=True, i2i=True, clients=live,
                           references_dir=str(root))
    payload = rewrite(service, "edit", "make it night", [ref])
    assert payload["backend"] == "pe-llamacpp"
    assert payload["rewritten_prompt"] == "rewritten by pe-llamacpp for edit"
    assert live[("edit", "pe-llamacpp")].requests == ["edit"]
    assert live[("t2i", "pe-llamacpp")].requests == []


def test_a_server_without_the_official_edit_checkpoint_routes_edit_to_the_endpoint(tmp_path):
    """PE-T2I only: the edit group's official radio is greyed, the choice still
    says official, and the endpoint answers — loudly, in the status block."""
    root = tmp_path / "store"
    ref = store_image(root)
    live: dict = {}
    service = make_service(tmp_path, local=True, pe=True, clients=live,
                           references_dir=str(root))
    state = state_of(service)
    assert task_choice(state, "edit", "pe-llamacpp")["available"] is False
    assert task_choice(state, "edit", "local-llm")["available"] is True
    edit = rewrite(service, "edit", "make it night", [ref])
    assert edit["backend"] == "local-llm"
    assert live[("t2i", "pe-llamacpp")].requests == [], \
        "an edit draft must never be handed to the text checkpoint"
    assert rewrite(service)["backend"] == "pe-llamacpp"
    block = service.status_block()
    assert api.validate_rewrite_block(block) == block
    assert block["active_by_task"] == {"t2i": "pe-llamacpp",
                                       "edit": "local-llm"}
    assert block["backend"] == "pe-llamacpp"
    # The saved choice is untouched: putting the missing file in place fixes
    # the edit group without a second visit to Settings.
    assert service.preferences().choices is None


def test_asking_for_the_official_edit_enhancer_without_its_files_is_refused(tmp_path):
    """A POST is a question with an answer waiting, so it is refused rather
    than fallen back on — and the refusal is the task-specific sentence."""
    service = make_service(tmp_path, local=True, pe=True)
    got, answered = responded(request(
        "POST", server.SETTINGS_ROUTE,
        json.loads(body(t2i="pe-llamacpp", edit="pe-llamacpp")), service))
    assert got == 422 and answered["code"] == "backend_unavailable"
    assert "image-edit" in answered["message"]
    assert service.unavailable_reason("pe-llamacpp", "edit") == \
        answered["message"]
    # The same server happily accepts official t2i plus endpoint edit.
    got, answered = responded(request(
        "POST", server.SETTINGS_ROUTE,
        json.loads(body(t2i="pe-llamacpp", edit="local-llm")), service))
    assert got == 200
    assert service.active_by_task() == {"t2i": "pe-llamacpp",
                                        "edit": "local-llm"}


def test_a_draft_kind_no_rewriter_can_answer_is_refused_never_swapped(tmp_path):
    """No endpoint and no edit checkpoint means an edit draft has no rewriter.
    Falling back to the text weights would send references to a model that was
    never trained on them, so the refusal names what is missing."""
    root = tmp_path / "store"
    ref = store_image(root)
    live: dict = {}
    service = make_service(tmp_path, local=False, pe=True, clients=live,
                           references_dir=str(root))
    assert service.available_backends() == ["pe-llamacpp"]
    assert service.active_by_task() == {"t2i": "pe-llamacpp", "edit": None}
    with pytest.raises(api.ApiError) as exc:
        rewrite(service, "edit", "make it night", [ref])
    assert exc.value.status == 422 and exc.value.code == "task_unsupported"
    assert "image-edit" in exc.value.message
    assert "OpenAI-compatible endpoint" in exc.value.message
    assert live[("t2i", "pe-llamacpp")].requests == []
    # A text draft is untouched by that refusal.
    assert rewrite(service)["backend"] == "pe-llamacpp"


def test_close_closes_each_distinct_client_exactly_once(tmp_path):
    """Two checkpoints mean two llama-server children; the endpoint is one
    object keyed by two tasks. Identity, not the key, is what dedupes."""
    live: dict = {}
    service = make_service(tmp_path, local=True, pe=True, i2i=True, clients=live)
    assert service.close(timeout_s=5) is True
    assert [client.closes for client in (
        live[("t2i", "pe-llamacpp")], live[("edit", "pe-llamacpp")],
        live[("t2i", "local-llm")])] == [1, 1, 1]


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

    slow = SlowClient("pe-llamacpp for t2i")
    service = make_service(tmp_path, local=True, pe=True, i2i=True, clients={},
                           client=slow)
    assert service.active_by_task() == {"t2i": "pe-llamacpp",
                                        "edit": "pe-llamacpp"}
    answers: list[dict] = []
    worker = threading.Thread(target=lambda: answers.append(rewrite(service)),
                              name="rewrite-under-test")
    worker.start()
    assert inside.wait(10.0), "the rewrite never reached its client"
    service.apply_preferences(rs.RewritePreferences(
        choices={"t2i": "local-llm", "edit": "local-llm"}, enabled=True,
        endpoint=ENDPOINT, model=MODEL))
    gate.set()
    worker.join(10.0)
    assert not worker.is_alive(), "the rewrite never finished"
    assert answers[0]["backend"] == "pe-llamacpp", "the running rewrite moved"
    # From here on the change is real, and what status reports agrees with it.
    assert service.active_by_task() == {"t2i": "local-llm", "edit": "local-llm"}
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

    client = CancelledMidGeneration("pe-llamacpp for t2i")
    subject = make_service(tmp_path, local=True, pe=True, i2i=True, client=client)
    client.subject = subject
    with pytest.raises(api.ApiError) as caught:
        rewrite(subject)
    assert caught.value.status == 503
    assert caught.value.code == "rewrite_cancelled"
    assert caught.value.message == "the rewrite was cancelled"
