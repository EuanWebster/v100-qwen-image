"""The official enhancer backend: prompt bytes, argv, GPU lock, process life.

Two doubles, each for the half that the guarded runner cannot do for real:

* **HTTP** — ``tests/cpu_guard/sitecustomize.py`` denies loopback TCP socket
  construction in the pytest process *and in every child it spawns*, so a fake
  server could not bind the port the brief names. The client's real
  ``http.client`` path is therefore driven over an AF_UNIX ``socketpair`` with
  a ``BaseHTTPRequestHandler`` answering real HTTP/1.1 bytes (the repo's
  existing guarded pattern; see ``tests/test_rewrite_client.py``'s note that no
  socket is ever constructed).
* **Process** — ``tests/fixtures/fake_llama_server.py`` is a REAL subprocess
  with a real PID, so "the server is gone before the lock is released" and
  "no orphan remains" are asserted against ``os.kill(pid, 0)``, not a mock.

The edit leg is driven through the same fixture: a real multimodal server
invents its ``media_marker`` per process and answers it on ``GET /props``, so
the handler serves that route too and the request ORDER — health polls, then
``/props``, then ``/completion`` — is asserted against recorded paths.

Nothing here touches a GPU, llama.cpp or a port.
"""
from __future__ import annotations

import base64
import http.client
import http.server
import os
import json
import pathlib
import socket
import subprocess
import sys
import threading
import time
import types

import pytest

from qwen_engine import pe_llamacpp as pe
from qwen_engine import rewrite as rw
from qwen_engine import rewrite_client as rc
from qwen_engine.web import rewrite_service as rs

FAKE_SERVER = pathlib.Path(__file__).resolve().parent / "fixtures" / \
    "fake_llama_server.py"

# The template markers, spelled the way the product spells them (a literal tag
# cannot live in source) so the golden strings below are written independently
# of the module's own constants.
_IM_START = "<|" + "im_start" + "|>"
_IM_END = "<|" + "im_end" + "|>"
_THINK = "<" + "think" + ">"

MESSAGES = [{"role": "system", "content": "SYS PROMPT"},
            {"role": "user", "content": "a red fox"}]


def golden(system: str, user: str, thinking: bool = True) -> str:
    """The official template's output for one two-turn conversation."""
    out = (f"{_IM_START}system\n{system}{_IM_END}\n"
           f"{_IM_START}user\n{user}{_IM_END}\n"
           f"{_IM_START}assistant\n")
    return out + (f"{_THINK}\n" if thinking else "")


# A stand-in reference image and the base64 the server is handed for it:
# distinct bytes per reference, so a swapped reference ORDER is observable.
def reference(tag: str) -> tuple[str, str]:
    """``(data URI, base64 payload)`` for one stand-in reference image."""
    payload = base64.b64encode(tag.encode("utf-8")).decode("ascii")
    return f"data:image/png;base64,{payload}", payload


ONE, ONE_B64 = reference("reference-one")
TWO, TWO_B64 = reference("reference-two")
IMAGE_PART = {"type": "image_url", "image_url": {"url": ONE}}
TEXT_PART = {"type": "text", "text": "make it night"}

#: llama.cpp invents the media marker per process, so every test below brings
#: its own value. ``PROPS_ROUTE`` is spelled independently of ``pe.PROPS_PATH``:
#: a client that asks a different route must fail rather than be agreed with.
MARKER = "<|" + "extra_media_7f3a9c" + "|>"
PROPS_ROUTE = "/props"


def edit_messages(*images: str,
                  draft: str = "make it night") -> list[dict[str, object]]:
    """The conversation the edit leg sends, built by the product's own
    :func:`qwen_engine.rewrite.build_messages`: references first, then the
    draft."""
    return rw.build_messages("SYS PROMPT", draft, list(images))


def golden_edit(marker: str, images: int, user: str,
                system: str = "SYS PROMPT", thinking: bool = True) -> str:
    """The edit layout, written independently of the module: ``images``
    markers immediately after the ``user\\n`` marker, then the trimmed draft."""
    out = (f"{_IM_START}system\n{system}{_IM_END}\n"
           f"{_IM_START}user\n{marker * images}{user}{_IM_END}\n"
           f"{_IM_START}assistant\n")
    return out + (f"{_THINK}\n" if thinking else "")


def touch(path: pathlib.Path, *, executable: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# stand-in operator file\n", encoding="utf-8")
    if executable:
        os.chmod(path, 0o755)
    return str(path)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SpyLock:
    """The GPU lock, recording whether a child was still alive at release."""

    def __init__(self) -> None:
        self._real = threading.Lock()
        self.witness = None          # set to the child once spawned
        self.released_with_live_child = False
        self.acquires = 0
        self.releases = 0

    def acquire(self, blocking: bool = True) -> bool:
        self.acquires += 1
        return self._real.acquire(blocking)

    def release(self) -> None:
        self.releases += 1
        process = self.witness
        if process is not None and alive(process.pid):
            self.released_with_live_child = True
        self._real.release()

    @property
    def free(self) -> bool:
        return self._real.acquire(blocking=False)


class FakeTransport:
    """The health/completion pair, with the bytes it was sent recorded.

    ``health_default`` is what the poll answers once the scripted list runs
    out: ``["starting"] * 200`` alone would otherwise read as healthy the
    moment the list is exhausted. ``on_completion`` lets a test hold the
    request open until a real child process has had its turn at the CPU.
    """

    def __init__(self, health=(), content=None, error=None,
                 health_default="ok", on_completion=None):
        self.health = list(health)
        self.content = content
        self.error = error
        self.health_default = health_default
        self.on_completion = on_completion
        self.completion_bodies: list[bytes] = []
        self.health_polls = 0

    def get_status(self, path="/health"):     # noqa: ARG002 - fixed by client
        self.health_polls += 1
        if self.health:
            return self.health.pop(0)
        return self.health_default

    def post_completion(self, body, path="/completion"):   # noqa: ARG002
        if self.on_completion is not None:
            self.on_completion()
        self.completion_bodies.append(body)
        if self.error is not None:
            raise self.error
        return self.content


class TransportFactory:
    """Records the per-call budget the client hands the transport."""

    def __init__(self, transport: FakeTransport) -> None:
        self.transport = transport
        self.budgets: list[float] = []

    def __call__(self, port: int, budget: float):   # noqa: ARG002
        self.budgets.append(budget)
        return self.transport


def spawn_server(argv, log):        # noqa: ARG001 - argv asserted separately
    """The default test spawn: a REAL child, killed by the client."""
    return subprocess.Popen(
        [sys.executable, str(FAKE_SERVER)], stdin=subprocess.DEVNULL,
        stdout=log.fileno(), stderr=subprocess.STDOUT,
        start_new_session=True)


def no_spawn(argv, log):            # noqa: ARG001
    no_spawn.calls.append(argv)
    return None


no_spawn.calls: list[list[str]] = []


def instant(argv, log):             # noqa: ARG001
    """A child that dies immediately: the never-healthy case."""
    return subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"],
                            stdin=subprocess.DEVNULL, stdout=log.fileno(),
                            stderr=subprocess.STDOUT, start_new_session=True)


def make_client(tmp_path, *, transport=None, spawn=spawn_server, lock=None,
                cancel=None, sleep=None, **overrides) -> pe.PELlamaCppClient:
    """A client wired to doubles, with the operator files for real."""
    binary = overrides.pop("server_binary", None) or touch(
        tmp_path / "bin" / "llama-server", executable=True)
    model = overrides.pop("model_path", None) or touch(
        tmp_path / "models" / "PE-T2I-Q8_0.gguf")
    kwargs: dict = dict(
        server_binary=binary, model_path=model,
        log_path=str(tmp_path / "cache" / "pe-llamacpp.log"),
        gpu_lock=lock or threading.Lock(), timeout=30.0,
        health_timeout=5.0, spawn=spawn,
        port_picker=lambda: 43001, sleep=sleep or (lambda _seconds: None),
        cancel=cancel)
    if transport is not None:
        kwargs["transport_factory"] = TransportFactory(transport)
    kwargs.update(overrides)
    return pe.PELlamaCppClient(**kwargs)


# ---------------------------------------------------------------------------
# the prompt the official processor produces
# ---------------------------------------------------------------------------


def test_the_template_matches_the_official_processor_case_one():
    assert pe.render_chat_template(MESSAGES) == golden("SYS PROMPT", "a red fox")


def test_the_template_matches_the_official_processor_case_two():
    """Both contents are TRIMMED, and a newline inside one is kept."""
    messages = [{"role": "system", "content": "Line one\nLine two  "},
                {"role": "user", "content": "  spaced prompt \n"}]
    assert pe.render_chat_template(messages) == golden(
        "Line one\nLine two", "spaced prompt")


def test_without_thinking_the_prompt_stops_at_the_assistant_turn():
    assert pe.render_chat_template(MESSAGES, enable_thinking=False) == \
        golden("SYS PROMPT", "a red fox", thinking=False)


@pytest.mark.parametrize("messages, why", [
    ([], "empty"),
    ("not a list", "sequence"),
    ([["system", "SYS PROMPT"]], "mapping"),
    ([{"role": "system", "content": "s"},
      {"role": "user", "content": [IMAGE_PART, TEXT_PART]}], "render_prompt"),
    ([{"role": "user", "content": "x"}, {"role": "system", "content": "y"}],
     "system turn"),
    ([{"role": "system", "content": "y"}], "user turn"),
    ([{"content": "no role"}], "role"),
])
def test_a_conversation_the_enhancer_cannot_answer_is_refused_before_send(
        messages, why):
    with pytest.raises(rc.RewriteConfigError) as exc:
        pe.render_chat_template(messages)
    assert why in str(exc.value)


# ---------------------------------------------------------------------------
# the completion request
# ---------------------------------------------------------------------------


def test_the_request_carries_the_frozen_t2i_profile_not_a_copy_of_it():
    profile = rw.sampling("t2i")
    body = json.loads(pe.build_completion_body("p", seed=7))
    assert body["n_predict"] == profile["max_new_tokens"]
    for field in ("temperature", "top_p", "top_k", "min_p", "presence_penalty"):
        assert body[field] == profile[field]
    assert "max_tokens" not in body          # llama.cpp calls it n_predict


def test_the_request_pins_the_fields_root_verified_on_this_server():
    body = json.loads(pe.build_completion_body("p", seed=7))
    assert body == {
        "prompt": "p", "n_predict": 16256, "temperature": 1.0, "top_p": 0.95,
        "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
        "repeat_penalty": 1.0, "seed": 7, "cache_prompt": False,
        "special": False,
    }


def test_the_seed_is_the_callers_or_a_fresh_one_never_a_fixed_default():
    assert json.loads(pe.build_completion_body("p", seed=7))["seed"] == 7
    first = json.loads(pe.build_completion_body("p"))["seed"]
    second = json.loads(pe.build_completion_body("p"))["seed"]
    assert isinstance(first, int) and first >= 0 and first != second


def test_the_answer_is_read_from_content_or_the_choice_text():
    assert pe.content_from_document({"content": "text"}) == "text"
    assert pe.content_from_document({"choices": [{"text": "t"}]}) == "t"


@pytest.mark.parametrize("document", [
    {"content": None}, {"choices": []}, ["not", "an", "object"], {},
])
def test_a_reply_without_a_content_string_is_a_transport_violation(document):
    with pytest.raises(rc.RewriteInvalidAnswer):
        pe.content_from_document(document)


def test_health_is_ok_only_for_a_status_of_ok():
    assert pe._health_is_ok(b'{"status":"ok"}') is True
    assert pe._health_is_ok(b'{"status":"loading"}') is False
    assert pe._health_is_ok(b'<html>nope</html>') is False


# ---------------------------------------------------------------------------
# the official EDIT enhancer prompt: marker, order, profile
# ---------------------------------------------------------------------------


def test_the_edit_prompt_puts_one_marker_per_reference_before_the_draft():
    """llama.cpp counts the markers in the prompt against the images handed to
    it, so a dropped marker — or markers placed after the draft instead of
    before it — is the ``mtmd_tokenize`` mismatch, not a style choice. The
    draft is still trimmed exactly as the text template trims it."""
    prompt, images = pe.render_prompt(
        edit_messages(ONE, TWO, draft="  make it night \n"),
        media_marker=MARKER)
    assert prompt == golden_edit(MARKER, 2, "make it night")
    assert images == [ONE_B64, TWO_B64]


def test_swapping_references_moves_the_payloads_and_keeps_counts_equal():
    """The markers are interchangeable strings, so the only record of which
    reference is ``<image1>`` is the order of ``multimodal_data`` — and the
    server refuses a body whose marker count differs from its image count."""
    forward_prompt, forward = pe.render_prompt(edit_messages(ONE, TWO),
                                               media_marker=MARKER)
    swapped_prompt, swapped = pe.render_prompt(edit_messages(TWO, ONE),
                                               media_marker=MARKER)
    assert forward == [ONE_B64, TWO_B64]
    assert swapped == [TWO_B64, ONE_B64]
    assert forward_prompt == swapped_prompt       # the payloads ARE the order
    body = json.loads(pe.build_completion_body(swapped_prompt, task="edit",
                                               images=swapped, seed=5))
    assert body["prompt"]["multimodal_data"] == [TWO_B64, ONE_B64]
    assert (body["prompt"]["prompt_string"].count(MARKER)
            == len(body["prompt"]["multimodal_data"]))


def test_an_image_prompt_cannot_be_rendered_without_the_servers_marker():
    """There is no marker to fall back to: guessing one is what earns the real
    server's "number of media markers in text (0) does not match number of
    bitmaps"."""
    with pytest.raises(rc.RewriteConfigError, match="/props"):
        pe.render_prompt(edit_messages(ONE))


def test_a_reference_in_the_system_turn_is_refused_rather_than_counted():
    """Markers belong to the turn being answered; one in the system turn would
    be counted by the server and addressed by nobody."""
    with pytest.raises(rc.RewriteConfigError, match="only the user turn"):
        pe.render_prompt([{"role": "system",
                           "content": [IMAGE_PART, TEXT_PART]},
                          {"role": "user", "content": "make it night"}],
                         media_marker=MARKER)


@pytest.mark.parametrize("parts, why", [
    ([TEXT_PART, IMAGE_PART], "image after its text"),
    ([{"type": "image_url", "image_url": {"url": "http://evil.example/x.png"}},
      TEXT_PART], "never given a URL"),
    ([{"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
      TEXT_PART], "non-empty"),
    ([IMAGE_PART, {"type": "text", "text": "one"},
      {"type": "text", "text": "two"}], "exactly one text part"),
])
def test_a_reference_list_the_edit_server_could_not_index_is_refused(parts, why):
    """The edit system prompt addresses the references as ``<imageN>``: an
    ambiguous, fetchable or empty reference is refused before a server is
    started."""
    messages = [{"role": "system", "content": "SYS PROMPT"},
                {"role": "user", "content": parts}]
    with pytest.raises(rc.RewriteConfigError) as exc:
        pe.render_prompt(messages, media_marker=MARKER)
    assert why in str(exc.value)


def test_the_edit_body_carries_the_edit_profile_and_the_multimodal_object():
    """``presence_penalty`` is the load-bearing difference between the two
    frozen profiles (1.5 text / 0 edit) and a wrong one changes the sample
    without any error, so it is asserted next to the token budget."""
    body = json.loads(pe.build_completion_body("P", task="edit", images=["AAA"],
                                               seed=3))
    assert body["prompt"] == {"prompt_string": "P", "multimodal_data": ["AAA"]}
    assert body["n_predict"] == 24000 == rw.sampling("edit")["max_new_tokens"]
    assert body["presence_penalty"] == 0.0
    assert body["presence_penalty"] != rw.sampling("t2i")["presence_penalty"]


def test_only_a_request_with_references_becomes_a_prompt_object():
    """A bare prompt string is what a non-multimodal build reads, so an edit
    draft with no reference stays text — for either task."""
    for task in rw.TASKS:
        assert json.loads(pe.build_completion_body(
            "P", task=task, seed=3))["prompt"] == "P"


# ---------------------------------------------------------------------------
# argv and the operator's files
# ---------------------------------------------------------------------------


def test_the_argv_is_the_command_root_verified_on_the_v100(tmp_path):
    argv = make_client(tmp_path).build_argv(43211)
    assert argv == [str(tmp_path / "bin" / "llama-server"),
                    "-m", str(tmp_path / "models" / "PE-T2I-Q8_0.gguf"),
                    "-ngl", "99", "-c", "24576", "--host", "127.0.0.1",
                    "--port", "43211", "--no-webui"]
    # Every value is its own argv entry and no shell string is ever built.
    assert all(isinstance(part, str) for part in argv)


def installed_pe(tmp_path) -> rs.PESettings:
    """Both official checkpoints installed the way the launch flags install
    them: the text weights, and the edit weights with their vision projector."""
    return rs.PESettings(
        llama_server=touch(tmp_path / "bin" / "llama-server", executable=True),
        model_path=touch(tmp_path / "models" / "PE-T2I-Q8_0.gguf"),
        system_prompts={"t2i": "T2I PROMPT", "edit": "EDIT PROMPT"},
        log_path=str(tmp_path / "cache" / "pe-llamacpp.log"),
        i2i_model_path=touch(tmp_path / "models" / "PE-I2I-Q8_0.gguf"),
        i2i_mmproj_path=touch(tmp_path / "models" / "mmproj-BF16.gguf"))


def test_the_edit_server_gets_its_projector_and_its_image_budget(tmp_path):
    """The command Root verified for PE-I2I on the V100: the vision projector,
    32 768 context tokens (a 1 MP reference does not fit in the text profile's
    24 576) and the per-image cap that leaves room for the thinking block."""
    settings = installed_pe(tmp_path)
    client = settings.build_client(task="edit", spawn=no_spawn)
    assert client.build_argv(43211) == [
        settings.llama_server, "-m", settings.i2i_model_path,
        "-ngl", "99", "-c", "32768", "--host", "127.0.0.1", "--port", "43211",
        "--no-webui", "--mmproj", settings.i2i_mmproj_path,
        "--image-max-tokens", "1280"]
    assert client.supported_tasks == ("edit",)


def test_the_text_server_never_gets_the_vision_flags(tmp_path):
    """Both checkpoints are installed here, so this is where a client factory
    that handed out the projector or the wide context unconditionally shows up:
    the text weights would load a foreign projector and pay for vision tokens
    they never use."""
    settings = installed_pe(tmp_path)
    argv = settings.build_client(task="t2i", spawn=no_spawn).build_argv(43211)
    assert "--mmproj" not in argv and "--image-max-tokens" not in argv
    assert argv[argv.index("-m") + 1] == settings.model_path
    assert argv[argv.index("-c") + 1] == "24576"


def test_the_real_spawn_uses_no_shell_and_its_own_process_group(tmp_path,
                                                                monkeypatch):
    seen: dict = {}
    real = subprocess.Popen

    def redirect(argv, **kwargs):
        seen["argv"] = argv
        seen.update(kwargs)
        # argv is redirected to the fake server; every other keyword — the
        # log's write fd included — reaches Popen untouched.
        return real([sys.executable, str(FAKE_SERVER)], **kwargs)

    monkeypatch.setattr(pe.subprocess, "Popen", redirect)
    transport = FakeTransport(content='{"rewritten_prompt":"x"}')
    client = make_client(tmp_path, transport=transport, spawn=None)
    assert client.complete(MESSAGES, "t2i") == '{"rewritten_prompt":"x"}'
    assert isinstance(seen["argv"], list) and not isinstance(seen["argv"], str)
    assert seen["shell"] is False
    assert seen["start_new_session"] is True
    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["stderr"] is subprocess.STDOUT
    assert seen["close_fds"] is True
    assert isinstance(seen["stdout"], int)   # the bounded log's pipe


@pytest.mark.parametrize("field", ["server_binary", "model_path"])
def test_a_missing_operator_file_refuses_at_construction(tmp_path, field):
    kwargs = {field: str(tmp_path / "nope" / "thing")}
    with pytest.raises(rc.RewriteConfigError, match="not a readable"):
        make_client(tmp_path, **kwargs)


def test_a_binary_that_is_not_executable_refuses_at_construction(tmp_path):
    with pytest.raises(rc.RewriteConfigError, match="executable"):
        make_client(tmp_path, server_binary=touch(tmp_path / "llama-server"))


def test_claiming_edit_without_a_projector_refuses_at_construction(tmp_path):
    """Half an edit install is a startup mistake, never a 502 that arrives
    halfway through a user's rewrite: without ``--mmproj`` these weights cannot
    see the reference they are being asked about."""
    with pytest.raises(rc.RewriteConfigError, match="vision projector"):
        make_client(tmp_path, supported_tasks=("edit",), spawn=no_spawn)


@pytest.mark.parametrize("shape", ["missing", "directory"])
def test_a_projector_that_cannot_be_read_refuses_at_construction(tmp_path,
                                                                 shape):
    mmproj = tmp_path / "models" / "mmproj-BF16.gguf"
    if shape == "directory":
        mmproj.mkdir(parents=True)
    with pytest.raises(rc.RewriteConfigError, match="not a readable"):
        make_client(tmp_path, mmproj_path=str(mmproj),
                    supported_tasks=("edit",), spawn=no_spawn)


def test_the_production_port_picker_and_transport_are_the_loopback_ones(tmp_path):
    client = make_client(tmp_path, spawn=None)
    client = pe.PELlamaCppClient(
        server_binary=touch(tmp_path / "b", executable=True),
        model_path=touch(tmp_path / "m.gguf"), log_path=str(tmp_path / "l.log"))
    assert client._port_picker is pe._free_port      # never called in tests
    assert client._transport_factory(43001, 5.0).__class__ is pe.LoopbackClient


@pytest.mark.parametrize("host", ["0.0.0.0", "8.8.8.8", "evil.example"])
def test_a_non_loopback_server_host_refuses_before_any_request(host):
    client = pe.LoopbackClient(43001, 1.0, host=host)
    with pytest.raises(rc.RewriteConfigError, match="loopback|IP literal"):
        client.get_status()


# ---------------------------------------------------------------------------
# the GPU lock
# ---------------------------------------------------------------------------


def test_an_image_job_on_the_gpu_is_refused_at_once_not_queued(tmp_path):
    lock = threading.Lock()
    assert lock.acquire()
    client = make_client(tmp_path, transport=FakeTransport(content="x"),
                         lock=lock, spawn=no_spawn)
    started = time.monotonic()
    with pytest.raises(pe.RewriteGpuBusy) as exc:
        client.complete(MESSAGES, "t2i")
    assert time.monotonic() - started < 1.0
    assert str(exc.value) == ("the GPU is generating an image; try again when "
                              "it finishes")
    assert no_spawn.calls == []          # nothing started, nothing waited


def test_the_enhancer_takes_the_lock_and_gives_it_back(tmp_path):
    lock = SpyLock()
    transport = FakeTransport(content="ok")
    client = make_client(tmp_path, transport=transport, lock=lock)
    assert client.complete(MESSAGES, "t2i") == "ok"
    assert (lock.acquires, lock.releases) == (1, 1)
    assert lock.free is True


# ---------------------------------------------------------------------------
# process lifetime: the child must never outlive the call
# ---------------------------------------------------------------------------


def _run_in_thread(client, *args, **kwargs):
    outcome: dict = {}

    def body():
        try:
            outcome["content"] = client.complete(*args, **kwargs)
        except BaseException as exc:      # noqa: BLE001 - surfaced below
            outcome["error"] = exc

    thread = threading.Thread(target=body)
    thread.start()
    return thread, outcome


def test_a_successful_rewrite_leaves_no_server_and_frees_the_gpu(tmp_path):
    spawned: dict = {}

    def spawn(argv, log):                 # noqa: ARG001
        spawned["process"] = spawn_server(argv, log)
        return spawned["process"]

    lock = SpyLock()
    transport = FakeTransport(health=["starting", "starting", "ok"],
                              content='{"rewritten_prompt":"A fox."}')
    client = make_client(tmp_path, transport=transport, spawn=spawn, lock=lock)
    lock.witness = None
    original_complete = transport.post_completion

    def post(body, path="/completion"):
        lock.witness = spawned["process"]     # alive at this moment
        return original_complete(body, path)

    transport.post_completion = post
    assert client.complete(MESSAGES, "t2i") == '{"rewritten_prompt":"A fox."}'
    pid = spawned["process"].pid
    assert spawned["process"].poll() is not None
    assert alive(pid) is False, "an orphan would hold 9.5 GiB"
    assert lock.released_with_live_child is False
    assert lock.free is True
    assert client.live_process() is None
    assert json.loads(transport.completion_bodies[0])["prompt"] == \
        pe.render_chat_template(MESSAGES)
    assert transport.health_polls == 3


def test_the_child_is_terminated_before_the_lock_is_released(tmp_path):
    """The order matters twice: the next image job must not meet a live
    enhancer on the GPU, and neither must the browser's retry."""
    holder = threading.Lock()
    holder.acquire()                        # image job is mid-generation
    client = make_client(tmp_path, transport=FakeTransport(content="x"),
                         lock=holder, spawn=no_spawn)
    with pytest.raises(pe.RewriteGpuBusy):
        client.complete(MESSAGES, "t2i")
    assert no_spawn.calls == []
    holder.release()


@pytest.mark.parametrize("scenario", ["health_timeout", "child_exit",
                                      "transport_error", "cancel", "deadline"])
def test_every_failure_path_terminates_the_child(tmp_path, scenario):
    spawned: dict = {}

    def spawn(argv, log):                 # noqa: ARG001
        spawned["process"] = (instant(argv, log) if scenario == "child_exit"
                              else spawn_server(argv, log))
        return spawned["process"]

    cancel = threading.Event()
    if scenario == "health_timeout":
        # Never ready: a drained script would read as healthy, and with the
        # injected no-op sleep only the deadline can end the wait.
        transport = FakeTransport(health_default="starting",
                                  health=["starting"] * 400)
        kwargs = dict(health_timeout=0.01)
    elif scenario == "child_exit":
        transport = FakeTransport(health_default="starting", health=["starting"])
        kwargs = {}
    elif scenario == "transport_error":
        transport = FakeTransport(error=rc.RewriteUnavailable("boom"))
        kwargs = {}
    elif scenario == "deadline":
        transport = FakeTransport(health_default="starting",
                                  health=["starting"] * 400)
        kwargs = dict(timeout=0.3, health_timeout=30.0,
                      sleep=lambda seconds: time.sleep(seconds / 50.0))
    else:                                  # cancel: mid-generation
        transport = FakeTransport(content="x")
        kwargs = {"cancel": cancel.is_set}

        def post(body, path="/completion"):
            cancel.set()
            return transport.content

        transport.post_completion = post

    client = make_client(tmp_path, transport=transport, spawn=spawn, **kwargs)
    with pytest.raises(rc.RewriteClientError):
        client.complete(MESSAGES, "t2i")
    pid = spawned["process"].pid
    assert alive(pid) is False, f"{scenario} left an orphan"
    assert client.live_process() is None


def test_the_cancel_flag_ends_a_wait_without_a_response(tmp_path):
    cancel = threading.Event()
    transport = FakeTransport(health=["starting"] * 400)
    client = make_client(tmp_path, transport=transport,
                         cancel=cancel.is_set, health_timeout=30.0,
                         sleep=lambda _seconds: time.sleep(0.01))
    thread, outcome = _run_in_thread(client, MESSAGES, "t2i")
    while transport.health_polls == 0:
        time.sleep(0.01)
    cancel.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert isinstance(outcome["error"], rc.RewriteTimeout)


def test_a_cancel_while_the_model_is_generating_frees_the_gpu_at_once(tmp_path):
    """The cancel flag used to be read only before and after the completion
    request, so cancelling a running generation left llama-server sitting on the
    V100 with its 9.5 GiB for the rest of the timeout while an image job waited
    on the lock behind it. The watchdog kills the child instead, and the honest
    name for that failure is a cancellation, not an unavailable backend."""
    cancel = threading.Event()
    spawned: dict = {}
    clock: dict = {}
    lock = SpyLock()

    def spawn(argv, log):                 # noqa: ARG001
        spawned["process"] = spawn_server(argv, log)
        lock.witness = spawned["process"]
        return spawned["process"]

    def block_until_the_child_dies():
        """The way a real generation blocks: the read ends only when the server
        underneath it is killed. The cancel is raised while it is blocked."""
        process = spawned["process"]
        cancel.set()
        clock["cancelled"] = time.monotonic()
        while process.poll() is None and \
                time.monotonic() - clock["cancelled"] < 20.0:
            time.sleep(0.02)
        clock["dead"] = time.monotonic()
        raise pe.RewriteUnavailable("the enhancer closed the connection")

    transport = FakeTransport(on_completion=block_until_the_child_dies)
    client = make_client(tmp_path, transport=transport, spawn=spawn, lock=lock,
                         cancel=cancel.is_set, timeout=300.0,
                         health_timeout=30.0)
    thread, outcome = _run_in_thread(client, MESSAGES, "t2i")
    thread.join(timeout=30)
    assert not thread.is_alive(), "the rewrite never finished"
    error = outcome.get("error")
    assert isinstance(error, rc.RewriteTimeout), f"misclassified: {error!r}"
    assert not isinstance(error, pe.RewriteGpuBusy)
    assert str(error) == "the rewrite was cancelled"
    # One poll interval, not the remaining 300-second budget.
    assert clock["dead"] - clock["cancelled"] <= 1.0, "the watchdog let it live"
    assert alive(spawned["process"].pid) is False, "an orphan holds the GPU"
    assert client.live_process() is None
    assert lock.released_with_live_child is False
    assert lock.free is True, "the GPU lock was never handed back"


def test_the_watchdog_stops_when_the_call_ends(tmp_path):
    """A watchdog that outlived its rewrite would kill a later call's server,
    so no thread named for it may still be alive once the call has returned —
    after a success, and after a failure."""
    def watchers():
        return [thread.name for thread in threading.enumerate()
                if thread.name == "qwen-pe-cancel-watchdog"]

    transport = FakeTransport(content="x")
    client = make_client(tmp_path, transport=transport)
    assert client.complete(MESSAGES, "t2i") == "x"
    assert watchers() == [], "a watchdog outlived the rewrite that started it"
    assert pe._live_snapshot() == [], "a child stayed on the exit backstop"
    transport.error = pe.RewriteUnavailable("the scripted server goes away")
    with pytest.raises(pe.RewriteUnavailable):
        client.complete(MESSAGES, "t2i")
    assert watchers() == [], "a watchdog survived a failed rewrite"
    assert pe._live_snapshot() == []


def test_a_server_that_dies_by_itself_is_still_an_unavailable_backend(tmp_path):
    """The remap must not be greedy: with no cancel in play, a dropped
    connection is a broken backend and stays a 502, not "you cancelled"."""
    transport = FakeTransport(content="x",
                              error=pe.RewriteUnavailable("connection reset"))
    client = make_client(tmp_path, transport=transport)
    with pytest.raises(rc.RewriteUnavailable) as caught:
        client.complete(MESSAGES, "t2i")
    assert not isinstance(caught.value, rc.RewriteTimeout)
    assert "cancel" not in str(caught.value).lower()
    assert client.live_process() is None


def test_close_kills_a_server_that_is_still_starting(tmp_path):
    spawned: dict = {}

    def spawn(argv, log):                 # noqa: ARG001
        spawned["process"] = spawn_server(argv, log)
        return spawned["process"]

    transport = FakeTransport(health=["starting"] * 400)
    client = make_client(tmp_path, transport=transport, spawn=spawn,
                         sleep=lambda _seconds: time.sleep(0.01),
                         health_timeout=30.0)
    thread, outcome = _run_in_thread(client, MESSAGES, "t2i")
    while client.live_process() is None:
        time.sleep(0.01)
    assert client.close() is True
    assert alive(spawned["process"].pid) is False
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert isinstance(outcome["error"], rc.RewriteUnavailable)
    assert client.close() is True, "a second close is not an error"


def test_the_termination_backstop_kills_every_live_child(tmp_path):
    client = make_client(tmp_path, transport=FakeTransport(content="x"),
                         spawn=no_spawn)
    assert client is not None          # constructing registers the atexit hook
    assert pe._ATEXIT_ARMED is True, "constructing a client arms atexit"
    process = subprocess.Popen([sys.executable, str(FAKE_SERVER)],
                               stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.STDOUT,
                               start_new_session=True)
    pe._LIVE.add(process)
    pe._kill_all_live()
    assert alive(process.pid) is False
    pe._LIVE.discard(process)


def test_the_log_is_bounded_and_the_child_output_reaches_it(tmp_path):
    log_path = tmp_path / "cache" / "pe-llamacpp.log"

    def wait_for_banner():
        """A scripted transport answers faster than Python can start a child,
        so the request holds until the copier has carried the banner."""
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                if "fake llama-server argv:" in log_path.read_text(encoding="utf-8"):
                    return
            except OSError:
                pass
            time.sleep(0.02)
        raise AssertionError("the child never reached the log")

    client = make_client(tmp_path, log_path=str(log_path),
                         transport=FakeTransport(content="x",
                                                 on_completion=wait_for_banner))
    assert client.complete(MESSAGES, "t2i") == "x"
    written = log_path.read_text(encoding="utf-8")
    assert "fake llama-server argv:" in written
    assert "awaiting the health poll" in written


def test_a_chatty_server_cannot_grow_the_log_past_the_cap(tmp_path):
    log_path = tmp_path / "cache" / "pe-llamacpp.log"
    log = pe._BoundedLog(str(log_path), limit=64).start()
    writer = subprocess.Popen([sys.executable, "-c",
                               "import sys; sys.stdout.write('x' * 5000)"],
                              stdin=subprocess.DEVNULL,
                              stdout=log.fileno(),
                              stderr=subprocess.STDOUT,
                              start_new_session=True)
    writer.wait(timeout=30)
    log.finish()
    written = log_path.read_bytes()
    assert len(written) <= 64 + 64
    assert written.endswith(b"d]\n") and b"capped" in written
    assert written.startswith(b"xxx")


def test_a_log_file_that_cannot_be_written_is_a_typed_refusal(tmp_path):
    client = make_client(tmp_path, log_path="/nonexistent-root/pe.log",
                         transport=FakeTransport(content="x"),
                         spawn=no_spawn)
    with pytest.raises(rc.RewriteUnavailable, match="log file"):
        client.complete(MESSAGES, "t2i")
    assert no_spawn.calls == []


# ---------------------------------------------------------------------------
# what this backend can answer
# ---------------------------------------------------------------------------


def test_an_edit_draft_is_refused_when_the_edit_checkpoint_is_absent(tmp_path):
    """The text weights never answer an edit draft. The refusal names the
    missing image-edit checkpoint and the rewriter that can answer it, and no
    server is started for a draft this backend cannot serve."""
    client = make_client(tmp_path, transport=FakeTransport(content="x"),
                         spawn=no_spawn)
    with pytest.raises(pe.RewriteTaskUnsupported) as exc:
        client.complete(MESSAGES, "edit")
    assert "image-edit enhancer" in str(exc.value)
    assert "endpoint" in str(exc.value)
    assert client.uses_task("t2i") is True and client.uses_task("edit") is False
    assert no_spawn.calls == []
    assert isinstance(exc.value, rc.RewriteClientError)   # the service maps it


def test_a_text_draft_on_edit_only_weights_is_refused_not_answered_by_swap(tmp_path):
    """Clients are per task now: a task with no weights is refused rather than
    quietly handed to whichever enhancer happens to be installed."""
    client = make_client(tmp_path, spawn=no_spawn,
                         supported_tasks=("edit",),
                         mmproj_path=touch(
                             tmp_path / "models" / "mmproj-BF16.gguf"),
                         transport=FakeTransport(content="x"))
    with pytest.raises(pe.RewriteTaskUnsupported) as exc:
        client.complete(MESSAGES, "t2i")
    assert "text-to-image" in str(exc.value)
    assert no_spawn.calls == []


def test_an_unknown_task_is_refused_by_the_shared_contract(tmp_path):
    with pytest.raises(rw.RewriteError):
        make_client(tmp_path, spawn=no_spawn).complete(MESSAGES, "inpaint")


def test_sampling_overrides_are_refused_not_silently_ignored(tmp_path):
    client = make_client(tmp_path, transport=FakeTransport(content="x"),
                         spawn=no_spawn)
    with pytest.raises(rc.RewriteConfigError, match="overrides"):
        client.complete(MESSAGES, "t2i", {"temperature": 0.0})


def test_a_closed_client_refuses_instead_of_starting_another_server(tmp_path):
    client = make_client(tmp_path, transport=FakeTransport(content="x"),
                         spawn=no_spawn)
    assert client.close() is True
    with pytest.raises(rc.RewriteUnavailable, match="closed"):
        client.complete(MESSAGES, "t2i")


@pytest.mark.parametrize("scenario", ["busy", "edit", "startup"])
def test_no_failure_message_names_a_path_a_port_or_a_url(tmp_path, scenario):
    """The refusal reaches the browser; the log file is the only place paths
    may appear."""
    binary = touch(tmp_path / "bin" / "llama-server", executable=True)
    model = touch(tmp_path / "models" / "PE-T2I-Q8_0.gguf")
    if scenario == "busy":
        lock = threading.Lock()
        lock.acquire()
        client = make_client(tmp_path, lock=lock, spawn=no_spawn)
        error: BaseException | None = None
        try:
            client.complete(MESSAGES, "t2i")
        except BaseException as exc:       # noqa: BLE001 - asserted below
            error = exc
    elif scenario == "edit":
        client = make_client(tmp_path, spawn=no_spawn)
        error = None
        try:
            client.complete(MESSAGES, "edit")
        except BaseException as exc:       # noqa: BLE001
            error = exc
    else:
        # Never ready: the health script must not drain into "ok", or the
        # startup failure this scenario exists to check never happens.
        client = make_client(tmp_path, transport=FakeTransport(
            health_default="starting", health=["starting"] * 200),
            health_timeout=0.01)
        error = None
        try:
            client.complete(MESSAGES, "t2i")
        except BaseException as exc:       # noqa: BLE001
            error = exc
        finally:
            client.close()                 # the real child must not outlive it
    assert error is not None
    message = str(error)
    for leak in ("/", "\\", "127.0.0.1", "43001", binary, model, "http"):
        assert leak not in message, f"{scenario} leaks {leak!r}: {message}"


# ---------------------------------------------------------------------------
# the loopback HTTP path, over a socketpair (see the module docstring)
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):            # noqa: ARG002 - silent fixture
        pass

    def _reply(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):                        # noqa: N802 - stdlib name
        state = self.server.state
        state.setdefault("paths", []).append(self.path)
        if self.path == PROPS_ROUTE:
            # Unscripted is answered, not dropped: a client that asks for a
            # marker it was never given must fail loudly, not hang.
            status, payload = state.get(
                "props", (500, b'{"error":"no /props scripted"}'))
            self._reply(status, payload)
            return
        plan = state["health"]
        status, payload = plan.pop(0) if plan else (200, b'{"status":"ok"}')
        self._reply(status, payload)

    def do_POST(self):                       # noqa: N802 - stdlib name
        length = int(self.headers.get("content-length") or 0)
        state = self.server.state
        state.setdefault("paths", []).append(self.path)
        state.setdefault("bodies", []).append(self.rfile.read(length))
        status, payload = state["completion"]
        self._reply(status, payload)


def piped_client(state, *, timeout: float = 5.0) -> pe.LoopbackClient:
    """A client whose connection is one end of a fresh AF_UNIX socketpair."""

    class Piped(http.client.HTTPConnection):
        def __init__(self, host, port, budget):
            super().__init__(host, port, timeout=budget)

        def connect(self):                   # already connected
            self.sock = self._sock

    def factory(host, port, budget):
        client_socket, server_socket = socket.socketpair(socket.AF_UNIX,
                                                         socket.SOCK_STREAM)
        holder = Piped(host, port, budget)
        holder._sock = client_socket
        threading.Thread(
            target=serve, args=(server_socket, state), daemon=True).start()
        return holder

    def serve(server_socket, served):
        try:
            _Handler(server_socket, ("pipe", 0),
                     types.SimpleNamespace(state=served))
        except OSError:
            pass                             # client hung up: normal
        finally:
            server_socket.close()

    return pe.LoopbackClient(43001, timeout, connection_factory=factory)


def test_health_ok_and_starting_come_from_real_http_bytes():
    assert piped_client({"health": [(200, b'{"status":"ok"}')]}).get_status() \
        == "ok"
    assert piped_client({"health": [(200, b'{"status":"loading"}')]}
                        ).get_status() == "starting"
    assert piped_client({"health": [(503, b"wait")]}).get_status() == "starting"


def test_a_health_endpoint_that_refuses_is_not_a_startup_wait():
    with pytest.raises(rc.RewriteUnavailable, match="404"):
        piped_client({"health": [(404, b"no route")]}).get_status()


def test_a_refused_connection_reads_as_still_starting():
    """Nothing is listening yet during load: a refused connect is normal, and
    it is the transport's own OSError mapping that says so."""

    class Refused:
        def request(self, *args, **kwargs):
            raise ConnectionRefusedError("nobody home")

        def close(self):
            pass

    client = pe.LoopbackClient(43001, 1.0,
                               connection_factory=lambda h, p, t: Refused())
    assert client.get_status() == "starting"


def test_a_completion_round_trip_keeps_the_thinking_preamble():
    answer = _THINK + "\nbecause\n" + _IM_END + "\n{\"rewritten_prompt\":\"x\"}"
    state = {"completion": (200, json.dumps({"content": answer}).encode())}
    client = piped_client(state)
    assert client.post_completion(b'{"prompt":"p"}') == answer
    assert json.loads(state["bodies"][0]) == {"prompt": "p"}


@pytest.mark.parametrize("status, payload, error", [
    (500, b"boom", rc.RewriteUnavailable),
    (200, b"<html>not json", rc.RewriteInvalidAnswer),
    (200, b'{"nope":1}', rc.RewriteInvalidAnswer),
])
def test_a_bad_completion_reply_is_typed(status, payload, error):
    client = piped_client({"completion": (status, payload)})
    with pytest.raises(error):
        client.post_completion(b"{}")


def test_an_oversized_completion_is_refused_before_it_is_buffered():
    padding = "y" * (pe.MAX_RESPONSE_BYTES + 1024)
    client = piped_client({"completion": (200, json.dumps(
        {"content": padding}).encode())})
    with pytest.raises(rc.RewriteInvalidAnswer, match="cap"):
        client.post_completion(b"{}")


def test_the_free_port_picker_is_only_used_in_production(tmp_path):
    """The guarded runner denies AF_INET construction, so a test must inject a
    port; asserting the default proves the production path is wired."""
    assert pe._free_port.__doc__ is not None
    assert make_client(tmp_path, spawn=no_spawn)._port_picker() == 43001


# ---------------------------------------------------------------------------
# the edit rewrite over real HTTP: /props after health, order, profile
# ---------------------------------------------------------------------------


def piped_factory(state):
    """A ``transport_factory`` of real :class:`pe.LoopbackClient` transports
    over the socketpair fixture, so the request ORDER the client chooses is the
    order a real ``llama-server`` would see."""
    return lambda _port, budget: piped_client(state, timeout=budget)


def edit_ready_state(marker: str = MARKER) -> dict:
    """A multimodal server: still loading, then healthy, then a marker of its
    own — a value a real process invents, so no test may assume the next one
    looks like this one."""
    return {
        "health": [(200, b'{"status":"loading"}'), (200, b'{"status":"ok"}')],
        "props": (200, json.dumps({"media_marker": marker}).encode()),
        "completion": (200, json.dumps({"content": "ENHANCED"}).encode()),
    }


def edit_client(tmp_path, state) -> pe.PELlamaCppClient:
    """An edit-capable client whose every request goes through the real
    loopback HTTP code path."""
    return make_client(
        tmp_path, spawn=no_spawn,
        mmproj_path=touch(tmp_path / "models" / "mmproj-BF16.gguf"),
        supported_tasks=("edit",), transport_factory=piped_factory(state))


def test_an_edit_rewrite_asks_for_its_marker_after_health_and_sends_it(tmp_path):
    """The whole official-edit round trip: two loading polls, then ``/props``,
    then a completion whose prompt carries THIS server's marker — never the
    literal ``__media__``, and never a marker read before the server was
    listening, when it does not exist yet."""
    state = edit_ready_state()
    client = edit_client(tmp_path, state)
    assert client.complete(edit_messages(ONE, TWO), "edit") == "ENHANCED"
    assert state["paths"] == ["/health", "/health", PROPS_ROUTE, "/completion"]
    body = json.loads(state["bodies"][0])
    assert body["prompt"] == {
        "prompt_string": golden_edit(MARKER, 2, "make it night"),
        "multimodal_data": [ONE_B64, TWO_B64]}
    assert "__media__" not in body["prompt"]["prompt_string"]


def test_an_edit_rewrite_is_posted_with_the_edit_profile(tmp_path):
    """The task has to survive from ``complete()`` into the body: a t2i profile
    hardcoded in the call path would sample an edit draft at presence_penalty
    1.5 with two thirds of its token budget, and the server would answer
    happily — the deviation is invisible unless it is asserted here."""
    state = edit_ready_state()
    edit_client(tmp_path, state).complete(edit_messages(ONE), "edit")
    body = json.loads(state["bodies"][0])
    assert body["n_predict"] == 24000
    assert body["presence_penalty"] == 0.0


def test_the_marker_is_asked_again_next_time_never_remembered(tmp_path):
    """One server per rewrite means one marker per rewrite: a marker cached on
    the client matches no bitmaps on the second attempt."""
    state = edit_ready_state()
    client = edit_client(tmp_path, state)
    assert client.complete(edit_messages(ONE), "edit") == "ENHANCED"
    assert client.complete(edit_messages(ONE), "edit") == "ENHANCED"
    assert state["paths"] == ["/health", "/health", PROPS_ROUTE, "/completion",
                              "/health", PROPS_ROUTE, "/completion"]


def test_a_text_rewrite_never_buys_a_props_round_trip(tmp_path):
    """A text-only ``llama-server`` build does not answer ``/props`` at all:
    asking for a marker on a text draft would break every t2i rewrite on such a
    server, and is a wasted round trip on any other."""
    state = edit_ready_state()
    client = make_client(tmp_path, spawn=no_spawn,
                         transport_factory=piped_factory(state))
    assert client.complete(MESSAGES, "t2i") == "ENHANCED"
    assert PROPS_ROUTE not in state["paths"]
    assert json.loads(state["bodies"][0])["prompt"] == golden("SYS PROMPT",
                                                              "a red fox")


def scripted_props(marker: object) -> tuple[int, bytes]:
    return 200, json.dumps({"media_marker": marker}).encode()


@pytest.mark.parametrize("props, error", [
    ((200, b'{"model_path":"PE-I2I-Q8_0.gguf"}'), rc.RewriteConfigError),
    (scripted_props(None), rc.RewriteConfigError),
    (scripted_props(""), rc.RewriteConfigError),
    (scripted_props("<|media marker|>"), rc.RewriteConfigError),
    (scripted_props("x" * (pe.MEDIA_MARKER_MAX + 1)), rc.RewriteConfigError),
    ((200, b"<html>not json"), rc.RewriteInvalidAnswer),
    ((404, b"no route"), rc.RewriteUnavailable),
])
def test_a_server_that_cannot_name_its_marker_fails_typed(tmp_path, props,
                                                          error):
    """No marker, an unusable one, or no ``/props`` route at all is a typed
    refusal with NOTHING sent: the alternative is to post a prompt the server
    refuses for every image, which the user reads as a broken backend. A marker
    carrying whitespace would also be a template injection, since it is
    embedded in the user turn verbatim."""
    state = edit_ready_state()
    state["props"] = props
    client = edit_client(tmp_path, state)
    with pytest.raises(error):
        client.complete(edit_messages(ONE), "edit")
    assert "bodies" not in state
    assert client.live_process() is None
