"""The official T2I prompt enhancer (llama.cpp + Q8_0 GGUF) as a rewriter.

Same ``complete(messages, task)`` contract as
:class:`qwen_engine.rewrite_client.RewriteClient`, so
:class:`qwen_engine.web.rewrite_service.RewriteService` can call either one.
The difference is transport: ``rewrite_client`` POSTs an OpenAI chat
completion to a service that is already running, this module **owns a
llama-server process for the duration of one rewrite** — the enhancer needs
9.5 GiB of the 16 GiB V100 (measured by Root, 2026-10-26) and an image job
13–15 GiB, so the two cannot be resident together and neither can outlive the
call that started it.

Per call:

1. take the shared GPU lock WITHOUT waiting — an image job that is generating
   now must not be delayed by a rewrite, and a rewrite that waited would be
   reported as "slow" when it is really "queued";
2. bind an ephemeral loopback port (bind-then-close: the port is passed to a
   child, so there is nothing to hold open);
3. spawn ``llama-server`` with an explicit argv list (never a shell), in its
   own process group, stdout/stderr into one bounded log file under the
   server's cache dir;
4. wait for ``GET /health`` to answer ``{"status":"ok"}`` (bounded);
5. build the raw prompt with :func:`render_chat_template` — exactly what the
   official processor produces with ``enable_thinking=True``, contents
   trimmed, thinking pre-opened;
6. ``POST /completion`` with the frozen t2i sampling profile
   (:func:`qwen_engine.rewrite.sampling`, never a copy of its numbers);
7. terminate the process group (SIGTERM, wait, SIGKILL) and REAP it;
8. release the lock, then return ``content``.

The process is gone before the lock is released, at every exit path: cancel,
deadline, health failure, HTTP failure, success. :meth:`PELlamaCppClient.close`
and an :mod:`atexit` hook cover a client that is shut down or abandoned while
a call is in flight, because an orphan llama-server holds 9.5 GiB of the only
GPU this project has.

A cancel or a deadline that lands WHILE the model is generating is honoured
too: the completion read blocks and cannot be interrupted, so a watchdog thread
polls the caller's cancel predicate every ``CANCEL_POLL_SECONDS`` and
terminates the child; the read then fails, and that failure is reported as a
cancellation rather than as an unavailable backend.

Only ``t2i`` is supported: the enhancer checkpoints are task-specific and the
edit one is not installed, so an edit rewrite is refused with a typed error
rather than silently answered by the wrong model.

Nothing that reaches the browser names a URL, port, path or host: failures are
fixed strings, and the log file (which does contain paths) is only ever
mentioned as "the server log file".
"""
from __future__ import annotations

import atexit
import http.client
import ipaddress
import json
import os
import secrets
import signal
import socket
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from qwen_engine import rewrite as rw
from qwen_engine.rewrite_client import (
    MAX_RESPONSE_BYTES,
    RewriteClientError,
    RewriteConfigError,
    RewriteInvalidAnswer,
    RewriteTimeout,
    RewriteUnavailable,
)

#: Budget for one enhancer rewrite. Root measured 14-37 s for a t2i rewrite at
#: about 70 tok/s, and the thinking block can legitimately reach the token
#: budget (16256 tokens ~= 230 s), so the default is above that worst case.
DEFAULT_TIMEOUT_SECONDS = 300.0

#: How long a server that never becomes healthy is waited for, in seconds.
#: Model load is bounded and measured: 9.5 GiB of GPU, no retries.
HEALTH_TIMEOUT_SECONDS = 120.0

#: Health poll spacing. Load is the slow phase; polling harder buys nothing.
HEALTH_POLL_SECONDS = 0.25

#: SIGTERM grace before SIGKILL, and the wait for the reaping, in seconds.
TERM_GRACE_SECONDS = 10.0
KILL_WAIT_SECONDS = 5.0

#: The loopback host the child is told to bind. Literal, never configurable:
#: the enhancer reads user drafts and must stay on this machine.
SERVER_HOST = "127.0.0.1"

#: Flags Root verified on the V100 with the sm_70 build (2026-10-26):
#: ``-ngl 99`` (full GPU offload), ``-c 24576`` (thinking + answer + prompt),
#: ``--no-webui`` (no UI routes on a port the browser can reach).
NGL = 99
CONTEXT_TOKENS = 24576

#: The enhancer writes its own stdout/stderr here. Bounded so a chatty or
#: looping model cannot fill the disk: once the cap is hit the tail is dropped
#: rather than rotated (this is a diagnostic, not a record).
MAX_LOG_BYTES = 1 << 20

#: How often the cancel watchdog looks at the caller's cancel flag while the
#: model is generating. The completion read blocks for the whole generation and
#: cannot be interrupted, so this interval IS the delay between "stop" and "the
#: V100 is free again" — a child holding 9.5 GiB must not get to sit there for
#: the rest of the timeout with an image job queued behind it.
CANCEL_POLL_SECONDS = 0.25

#: How long a call waits for its own watchdog to stop before reaping the child
#: itself. Short on purpose: waiting for a kill to finish would hold the GPU
#: lock longer than the kill needs, and ``_terminate`` is safe to call twice.
CANCEL_JOIN_SECONDS = 0.5

#: The only task these weights were trained for.
SUPPORTED_TASKS: tuple[str, ...] = ("t2i",)

#: Response ``content`` starts INSIDE the thinking block and closes it; the
#: answer JSON follows. ``rewrite.parse_answer`` handles the preamble, so the
#: full content is handed to it untouched.

#: Chat-template markers, assembled so this source carries no literal tag.
_IM_START = "<|" + "im_start" + "|>"
_IM_END = "<|" + "im_end" + "|>"
_THINK_OPEN = "<" + "think" + ">"


class RewriteGpuBusy(RewriteUnavailable):
    """The V100 is generating an image; the enhancer cannot share it."""


class RewriteTaskUnsupported(RewriteClientError):
    """This backend has no weights for that task (refused, never guessed)."""


# ---------------------------------------------------------------------------
# prompt template
# ---------------------------------------------------------------------------


def render_chat_template(messages: Sequence[Mapping[str, Any]], *,
                         enable_thinking: bool = True) -> str:
    """The raw prompt for the enhancer, byte-for-byte the official template.

    Root captured both golden cases from the official processor with
    ``enable_thinking=True`` (pinned in ``tests/test_pe_llamacpp.py``):

    * every turn is ``{role}\\n{content}
</think>

<|im_end|>\n`;
    * system and user contents are **trimmed** (`"Line one\\nLine two  "`
      → ``Line one\\nLine two``, ``"  spaced prompt \\n"`` → ``spaced
      prompt``), while newlines INSIDE a content are kept;
    * the assistant turn is only the opening ``assistant\\n``,
      plus the pre-opened thinking block ``\\n`` when
      ``enable_thinking`` — the generation starts inside the thinking block.

    Refused with :class:`RewriteConfigError`: an empty message list, a
    non-mapping entry, a missing/non-string role or content (a parts list is a
    vision request, which this backend does not implement), a conversation that
    does not start with ``system`` or ends anywhere but ``user``.
    """
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise RewriteConfigError(
            "messages must be a sequence of message mappings")
    if not messages:
        raise RewriteConfigError("messages must not be empty")
    rendered: list[str] = []
    roles: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise RewriteConfigError(
                f"message {index + 1} must be a mapping, not "
                f"{type(message).__name__}")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise RewriteConfigError(f"message {index + 1} needs a string role")
        if not isinstance(content, str):
            raise RewriteConfigError(
                f"message {index + 1} content must be a string: the official "
                "enhancer takes text only")
        roles.append(role)
        rendered.append(f"{_IM_START}{role}\n{content.strip()}{_IM_END}\n")
    if roles[0] != "system":
        raise RewriteConfigError(
            "the enhancer prompt must start with the system turn (the digest-"
            "pinned task prompt)")
    if roles[-1] != "user":
        raise RewriteConfigError(
            "the enhancer prompt must end with the user turn to be answered")
    prompt = "".join(rendered) + f"{_IM_START}assistant\n"
    return prompt + (f"{_THINK_OPEN}\n" if enable_thinking else "")


# ---------------------------------------------------------------------------
# request / response bytes
# ---------------------------------------------------------------------------


def _json_bytes(body: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(body, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):  # pragma: no cover - defensive
        raise RewriteClientError(
            "the enhancer request is not JSON-serialisable") from None


def build_completion_body(prompt: str, *, task: str = "t2i",
                          seed: int | None = None) -> bytes:
    """The exact ``POST /completion`` bytes for one enhancer call.

    The sampling numbers are :func:`qwen_engine.rewrite.sampling` for the task
    — the same frozen upstream profile the ``local-llm`` backend sends — mapped
    onto llama.cpp's names (``max_new_tokens`` → ``n_predict``).
    ``repeat_penalty``, ``cache_prompt`` and ``special`` are the values Root
    verified against this server; ``seed`` is a fresh 63-bit integer unless one
    is given, because a rewrite is not reproducible-by-request in the UI and a
    fixed seed would make every attempt for one draft identical.
    """
    profile = rw.sampling(task)
    if seed is None:
        seed = secrets.randbits(63)
    return _json_bytes({
        "prompt": prompt,
        "n_predict": profile["max_new_tokens"],
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "top_k": profile["top_k"],
        "min_p": profile["min_p"],
        "presence_penalty": profile["presence_penalty"],
        "repeat_penalty": 1.0,
        "seed": int(seed),
        "cache_prompt": False,
        "special": False,
    })


def content_from_document(document: Any) -> str:
    """``content`` from one ``/completion`` document, or a typed refusal.

    The response is not an OpenAI chat completion: the text is at the top
    level (``content``, or ``choices[0].text`` when the server is built with
    the compatibility fields). A non-string ``content`` is a transport
    violation, never an empty answer.
    """
    if not isinstance(document, dict):
        raise RewriteInvalidAnswer("enhancer response is not a JSON object")
    content = document.get("content")
    if content is None:
        choices = document.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            content = choices[0].get("text")
    if not isinstance(content, str):
        raise RewriteInvalidAnswer(
            "enhancer response has no content string")
    return content


def _read_capped(response: Any, limit: int = MAX_RESPONSE_BYTES) -> bytes:
    """Read a response fully, refusing to buffer more than ``limit`` bytes."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise RewriteInvalidAnswer(
                f"enhancer response exceeds the {limit} byte cap")
        chunks.append(chunk)
    return b"".join(chunks)


def _health_is_ok(body: bytes) -> bool:
    try:
        document = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(document, dict) and document.get("status") == "ok"


# ---------------------------------------------------------------------------
# process handling
# ---------------------------------------------------------------------------


class _BoundedLog:
    """One child's stdout/stderr into one file, capped at ``limit`` bytes.

    The cap is enforced by the READER: the child writes to a pipe and a daemon
    thread copies it to the file, so a server that talks forever cannot grow
    the file past the cap — and cannot block on a full pipe either (which
    would look exactly like a hung model load). What was dropped is stated on
    the file's last line. One file per call, replaced each time: the last
    attempt is the one an operator reads.
    """

    _CHUNK = 65536

    def __init__(self, path: str, limit: int = MAX_LOG_BYTES) -> None:
        self.path = path
        self.limit = limit
        self.dropped = 0
        self._file: Any = None
        self._read_fd: int | None = None
        self._write_fd: int | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> "_BoundedLog":
        """Open the file and the pipe, and start the copier. Raises
        :class:`RewriteUnavailable` (never ``OSError``): a cache directory the
        server cannot write to is a typed refusal, not a crash."""
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, mode=0o700, exist_ok=True)
            self._file = open(self.path, "wb")
            self._read_fd, self._write_fd = os.pipe()
        except OSError:
            self.finish()
            raise RewriteUnavailable(
                "the enhancer server log file cannot be opened in the server "
                "cache directory") from None
        self._reader = threading.Thread(target=self._copy, name="pe-llamacpp-log",
                                        daemon=True)
        self._reader.start()
        return self

    def fileno(self) -> int:
        """The descriptor the child writes to (what ``Popen`` is given)."""
        if self._write_fd is None:
            raise RewriteUnavailable("the enhancer server log is closed")
        return self._write_fd

    def _copy(self) -> None:
        """Drain the pipe until EOF, writing at most ``limit`` bytes.

        The read is the raw descriptor, not a buffered reader: a
        ``BufferedReader.read(n)`` blocks until ``n`` bytes arrive, so a
        server that talks in lines would leave its operator staring at an
        empty file for the whole model load. Each chunk is flushed as it
        lands, which is what makes ``tail -f`` on the log useful.
        """
        read_fd, self._read_fd = self._read_fd, None
        try:
            while True:
                try:
                    chunk = os.read(read_fd, self._CHUNK)
                except OSError:
                    break               # EOF, or the child took its end away
                if not chunk:
                    break
                room = max(0, self.limit - self._file.tell())
                if room:
                    self._file.write(chunk[:room])
                if len(chunk) > room:
                    self.dropped += len(chunk) - room
                self._file.flush()
        except OSError:
            self.dropped += 1   # the sink failed; say so rather than lie
        finally:
            try:
                if self.dropped:
                    self._file.write(
                        b"\n[enhancer log capped: "
                        + str(self.dropped).encode("ascii")
                        + b" further bytes dropped]\n")
                self._file.flush()
            except OSError:
                pass

    def finish(self) -> None:
        """Close the child's end, wait for the copier, close the file. Safe to
        call twice (the error path and the normal path both do)."""
        if self._write_fd is not None:
            try:
                os.close(self._write_fd)
            except OSError:
                pass
            self._write_fd = None
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.join(timeout=5.0)
        if self._read_fd is not None:
            try:
                os.close(self._read_fd)
            except OSError:
                pass
            self._read_fd = None
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None


class LoopbackClient:
    """One ``http.client`` request per call to ``127.0.0.1:{port}``.

    Same reasoning as ``rewrite_client.HttpTransport``: ``http.client``
    ignores ``HTTP(S)_PROXY`` and never follows redirects, so a draft cannot be
    sent anywhere but the process this module spawned. ``connection_factory``
    exists for the guarded tests (loopback TCP sockets are denied there) and
    is never set by production code.
    """

    def __init__(self, port: int, timeout: float, *, host: str = SERVER_HOST,
                 connection_factory: Callable[[str, int, float], Any] | None = None
                 ) -> None:
        self.port = port
        self.timeout = timeout
        self.host = host
        self._factory = connection_factory or \
            (lambda h, p, t: http.client.HTTPConnection(h, p, timeout=t))

    def _check_host(self) -> None:
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError:
            raise RewriteConfigError(
                "the enhancer server host must be an IP literal") from None
        if not address.is_loopback:
            raise RewriteConfigError(
                "the enhancer server must bind loopback: a draft is user text")

    def _request(self, method: str, path: str,
                 body: bytes | None = None) -> tuple[int, bytes]:
        self._check_host()
        connection = self._factory(self.host, self.port, self.timeout)
        try:
            headers = {"Accept": "application/json"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            status = int(response.status)
            payload = _read_capped(response)
            return status, payload
        except RewriteClientError:
            raise
        except (TimeoutError, socket.timeout):
            raise RewriteTimeout(
                f"the enhancer server did not answer {method} {path} within "
                f"{self.timeout:g}s") from None
        except OSError as exc:
            raise RewriteUnavailable(
                f"the enhancer server is unreachable ({type(exc).__name__})"
            ) from None
        finally:
            connection.close()

    def get_status(self, path: str = "/health") -> str | None:
        """``"ok"`` when health says so, ``"starting"`` while the server is
        not ready yet (a refusal or a non-JSON body — both normal during
        load), and a typed error for anything else."""
        try:
            status, body = self._request("GET", path)
        except RewriteUnavailable:
            return "starting"      # not listening yet: normal while loading
        except RewriteTimeout:
            raise                # the whole call budget expired
        if status == 200 and _health_is_ok(body):
            return "ok"
        if 200 <= status < 300 or status == 503:
            return "starting"
        raise RewriteUnavailable(
            f"the enhancer server answered HTTP {status} on its health check")

    def post_completion(self, body: bytes,
                        path: str = "/completion") -> str:
        status, payload = self._request("POST", path, body)
        if not 200 <= status < 300:
            raise RewriteUnavailable(
                f"the enhancer server answered HTTP {status}")
        try:
            document = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RewriteInvalidAnswer(
                "enhancer response is not JSON") from None
        return content_from_document(document)


def _free_port() -> int:
    """An ephemeral loopback port, released immediately before the child binds
    it. A bind failure is a host problem, not a rewrite outcome."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((SERVER_HOST, 0))
        return int(probe.getsockname()[1])
    except OSError:
        raise RewriteUnavailable(
            "no loopback port is available for the enhancer server") from None
    finally:
        probe.close()


def _terminate(process: Any, *, kill_wait: float = KILL_WAIT_SECONDS) -> bool:
    """SIGTERM the child's OWN process group, then SIGKILL, then reap.

    The group (not just the pid) is the target: llama-server can fork, and a
    surviving grandchild holding 9.5 GiB is exactly the failure this module
    exists to prevent. Returns True when the process was reaped. Never raises:
    every caller is already on a failure path and must still release the lock.
    """
    pid = getattr(process, "pid", None)
    if pid is None:
        return True
    if getattr(process, "poll", lambda: None)() is not None:
        return True
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + TERM_GRACE_SECONDS
    while time.monotonic() < deadline:
        if getattr(process, "poll", lambda: None)() is not None:
            return True
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=kill_wait)
    except Exception:  # noqa: BLE001 - a zombie is reported, never raised
        return False
    return True


#: Live children of every client in this process, for the interpreter-exit
#: backstop. Guarded by one lock: atexit runs once, at interpreter shutdown.
_LIVE: set[Any] = set()
_LIVE_LOCK = threading.Lock()
_ATEXIT_ARMED = False


def _live_snapshot() -> list[Any]:
    with _LIVE_LOCK:
        return list(_LIVE)


def _kill_all_live() -> None:
    """Last resort: an interpreter that exits with an enhancer server running
    must not leave it holding the GPU."""
    for process in _live_snapshot():
        _terminate(process)


def _arm_atexit() -> None:
    global _ATEXIT_ARMED
    with _LIVE_LOCK:
        if _ATEXIT_ARMED:
            return
        _ATEXIT_ARMED = True
        atexit.register(_kill_all_live)


class PELlamaCppClient:
    """One owned llama-server process per rewrite, same contract as
    :class:`qwen_engine.rewrite_client.RewriteClient`."""

    backend_id = "pe-llamacpp"

    def __init__(self, *, server_binary: str | os.PathLike[str],
                 model_path: str | os.PathLike[str],
                 log_path: str | os.PathLike[str],
                 gpu_lock: threading.Lock | None = None,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 health_timeout: float = HEALTH_TIMEOUT_SECONDS,
                 host: str = SERVER_HOST, context: int = CONTEXT_TOKENS,
                 ngl: int = NGL,
                 spawn: Callable[..., Any] | None = None,
                 transport_factory: Callable[[int, float], Any] | None = None,
                 port_picker: Callable[[], int] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 cancel: Callable[[], bool] | None = None) -> None:
        """``server_binary``/``model_path`` are operator files, checked at
        construction (a startup mistake is never a runtime 502). ``log_path``
        is one bounded file inside the server's cache dir. ``gpu_lock`` is the
        SAME lock the web generation executor takes: without it the two
        backends would sit on the GPU at once.

        ``spawn``, ``transport_factory``, ``port_picker`` and ``sleep`` are the
        test seams; production leaves them unset. ``cancel`` is the rewrite
        job's cancel flag, polled so an operator cancel still kills the child.
        """
        self.server_binary = self._checked_executable(server_binary,
                                                     "llama-server binary")
        self.model_path = self._checked_file(model_path, "enhancer model file")
        self.log_path = os.fspath(log_path)
        self.timeout = self._check_timeout(timeout)
        self.health_timeout = self._check_timeout(health_timeout)
        self.host = host
        self.context = self._check_positive_int(context, "context tokens")
        self.ngl = self._check_positive_int(ngl, "GPU offload layer count")
        self._gpu_lock = gpu_lock if gpu_lock is not None else threading.Lock()
        self._spawn = spawn if spawn is not None else self._default_spawn
        self._transport_factory = (transport_factory
                                   or (lambda port, budget: LoopbackClient(
                                       port, budget, host=host)))
        self._port_picker = port_picker or _free_port
        self._sleep = sleep
        self._cancel = cancel or _never
        self._lock = threading.Lock()
        self._process: Any = None
        self._closed = False
        _arm_atexit()

    # -- construction checks ------------------------------------------------

    @staticmethod
    def _check_timeout(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RewriteConfigError(
                f"enhancer timeout must be a number, got {type(value).__name__}")
        if not 0 < float(value) <= 3600:
            raise RewriteConfigError(
                f"enhancer timeout must be in (0, 3600] seconds, got {value!r}")
        return float(value)

    @staticmethod
    def _check_positive_int(value: Any, what: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RewriteConfigError(f"{what} must be a positive integer")
        return value

    @staticmethod
    def _checked_executable(path: Any, what: str) -> str:
        text = os.fspath(path) if isinstance(path, (str, os.PathLike)) else None
        if not text or not text.strip():
            raise RewriteConfigError(f"{what} must be a non-empty path")
        if not os.path.isfile(text) or not os.access(text, os.X_OK | os.R_OK):
            raise RewriteConfigError(
                f"{what} is not a readable executable file: {text}")
        return text

    @staticmethod
    def _checked_file(path: Any, what: str) -> str:
        text = os.fspath(path) if isinstance(path, (str, os.PathLike)) else None
        if not text or not text.strip():
            raise RewriteConfigError(f"{what} must be a non-empty path")
        if not os.path.isfile(text) or not os.access(text, os.R_OK):
            raise RewriteConfigError(
                f"{what} is not a readable file: {text}")
        return text

    # -- argv / spawn -------------------------------------------------------

    def build_argv(self, port: int) -> list[str]:
        """The explicit argv list for one server: no shell, no string to
        escape, and the exact flags Root verified on the V100 with the sm_70
        build (full GPU offload, 24576-token context, no web UI routes on a
        port a browser could reach)."""
        return [self.server_binary,
                "-m", self.model_path,
                "-ngl", str(self.ngl),
                "-c", str(self.context),
                "--host", self.host,
                "--port", str(port),
                "--no-webui"]

    def _default_spawn(self, argv: list[str], log: _BoundedLog) -> Any:
        """The real spawn: own process group (so the kill reaches a forked
        child), no shell, and stdout/stderr into the bounded log's pipe."""
        return subprocess.Popen(  # noqa: S603 - explicit argv list, no shell
            argv, stdin=subprocess.DEVNULL, stdout=log.fileno(),
            stderr=subprocess.STDOUT, start_new_session=True, shell=False,
            close_fds=True)

    # -- lifecycle ----------------------------------------------------------

    @property
    def available(self) -> bool:
        """The files are present; availability is decided at startup."""
        return not self._closed and os.path.isfile(self.model_path)

    def uses_task(self, task: str) -> bool:
        return task in SUPPORTED_TASKS

    def close(self) -> bool:
        """Terminate any live child. True when nothing is running afterwards."""
        with self._lock:
            self._closed = True
            process = self._process
        if process is None:
            return True
        return _terminate(process)

    def live_process(self) -> Any:
        """The child of the call in flight, or None (for checks/tests)."""
        with self._lock:
            return self._process

    # -- the call -----------------------------------------------------------

    def complete(self, messages: list[dict[str, Any]], task: str,
                 extra_sampling: Mapping[str, Any] | None = None,
                 *, seed: int | None = None) -> str:
        """Run one enhancer rewrite and return the raw ``content`` (thinking
        preamble included: ``rewrite.parse_answer`` strips it).

        Every failure is a :class:`~qwen_engine.rewrite_client.RewriteClientError`
        subclass: :class:`RewriteTaskUnsupported` for a task these weights do
        not have, :class:`RewriteGpuBusy` when an image job holds the GPU,
        :class:`RewriteUnavailable` for a server that will not start or answer,
        :class:`RewriteTimeout` when the budget expires,
        :class:`RewriteInvalidAnswer` for a body that is not a completion. In
        every case the child is terminated, reaped and the lock released
        before this returns or raises.
        """
        rw.check_task(task)
        if not self.uses_task(task):
            raise RewriteTaskUnsupported(
                "the official enhancer supports text-to-image rewrites only; "
                "the image-edit enhancer model is not installed — use the "
                "Quick rewriter for an edit draft")
        if extra_sampling is not None:
            raise RewriteConfigError(
                "the enhancer backend does not accept sampling overrides: it "
                "runs the frozen upstream profile for its task")
        if self._closed:
            raise RewriteUnavailable("the enhancer client is closed")
        if self._cancel():
            raise RewriteTimeout("the rewrite was cancelled")
        if not self._gpu_lock.acquire(blocking=False):
            raise RewriteGpuBusy(
                "the GPU is generating an image; try again when it finishes")
        try:
            return self._generate(messages, task, seed)
        finally:
            self._gpu_lock.release()

    def _generate(self, messages: list[dict[str, Any]], task: str,
                  seed: int | None) -> str:
        deadline = time.monotonic() + self.timeout
        port = self._port_picker()
        process = None
        killed = True
        log = self._open_log()
        stop_watch = threading.Event()
        watcher: threading.Thread | None = None
        try:
            process = self._spawn(self.build_argv(port), log)
            with self._lock:
                self._process = process
            with _LIVE_LOCK:
                _LIVE.add(process)
            # The completion read below blocks for as long as the model
            # generates, and a blocked read cannot be interrupted: this is the
            # only thing that can notice a cancel (or the service deadline,
            # which the same predicate reports) while a rewrite is running.
            # Without it the child sits on the V100 for the rest of the timeout
            # after the user has pressed cancel.
            watcher = threading.Thread(
                target=self._watch_cancel, args=(process, stop_watch),
                name="qwen-pe-cancel-watchdog", daemon=True)
            watcher.start()
            transport = self._transport_factory(port, self.timeout)
            self._wait_healthy(process, transport, deadline)
            prompt = render_chat_template(messages)
            body = build_completion_body(prompt, task=task, seed=seed)
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self._cancel():
                raise RewriteTimeout(
                    f"the enhancer did not answer within {self.timeout:g}s")
            content = self._transport_factory(port, remaining) \
                .post_completion(body)
            if self._cancel():
                raise RewriteTimeout("the rewrite was cancelled")
            return content
        except RewriteClientError as error:
            # The watchdog ends a cancelled rewrite by killing the child, so the
            # blocked read fails with "the server went away". The user asked for
            # this one to stop: that is a cancellation, not a broken backend.
            # A real timeout keeps its own message — the predicate reports both,
            # and only the watchdog's kill is evidence of a cancel.
            if not isinstance(error, RewriteTimeout) and self._cancel():
                raise RewriteTimeout("the rewrite was cancelled") from error
            raise
        finally:
            # The watchdog is stopped and joined before anything is reaped: it
            # belongs to THIS call, and a thread that outlived it could kill the
            # next call's child. The join is bounded short because waiting for a
            # kill to finish here would hold the GPU lock longer than the kill.
            stop_watch.set()
            if watcher is not None:
                watcher.join(CANCEL_JOIN_SECONDS)
            # The order is the whole point: the child is terminated and reaped
            # first (its log copier waits for EOF on the pipe), the log is
            # closed second, and the CALLER releases the GPU lock after this
            # returns — so the lock is never handed on while a server that
            # holds 9.5 GiB is still alive.
            if process is not None:
                killed = _terminate(process)
                with _LIVE_LOCK:
                    _LIVE.discard(process)
                with self._lock:
                    if self._process is process:
                        self._process = None
            log.finish()
            if not killed:
                # Deliberately not raised: the caller must see its own typed
                # refusal, and the group has already been SIGKILLed.
                print("qwen-pe: an enhancer server could not be reaped; check "
                      "the server log file", flush=True)

    def _open_log(self) -> _BoundedLog:
        """One bounded log file for this call, replacing the last attempt."""
        return _BoundedLog(self.log_path).start()

    def _watch_cancel(self, process: Any, stop: threading.Event) -> None:
        """Terminate ``process`` once the caller has cancelled or run out of
        budget, then stop.

        Started by :meth:`_generate` and joined by it, so it never outlives the
        call that owns the child and never sees another call's. ``cancel`` is
        the constructor's cheap total predicate; nothing else is touched from
        here — no lock is taken, because the reaper in :meth:`close` may be
        waiting on it and this thread is the one that has to get through.
        """
        while not stop.wait(CANCEL_POLL_SECONDS):
            if self._cancel():
                _terminate(process)
                return

    def _wait_healthy(self, process: Any, transport: Any,
                      deadline: float) -> None:
        """Poll ``/health`` until the server is ready. Bounded twice over: the
        health budget and the call's own deadline; a dead child is noticed
        immediately rather than at the timeout."""
        started = time.monotonic()
        health_deadline = started + min(self.health_timeout,
                                        max(0.0, deadline - started))
        while True:
            if self._cancel():
                raise RewriteTimeout("the rewrite was cancelled")
            exit_code = getattr(process, "poll", lambda: None)()
            if exit_code is not None:
                raise RewriteUnavailable(
                    "the enhancer server exited before it became healthy "
                    "(see the server log file)")
            state = transport.get_status()
            if state == "ok":
                return
            if time.monotonic() >= health_deadline:
                raise RewriteUnavailable(
                    f"the enhancer server did not become healthy within "
                    f"{self.health_timeout:g}s (see the server log file)")
            self._sleep(HEALTH_POLL_SECONDS)


def _never() -> bool:
    return False
