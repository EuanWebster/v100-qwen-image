"""Outbound call to a local OpenAI-compatible rewriter service (plan §4.2 R2).

This is the only module in the engine that speaks HTTP to a service. Everything
about it is deliberately narrow:

* **Loopback only, resolved once.** :func:`parse_endpoint` accepts ``http`` on
  ``127.0.0.0/8`` or ``::1`` and nothing else. A hostname is refused outright
  rather than resolved, because resolution can change between the check and the
  connect; ``localhost`` is the one exception, mapped to ``127.0.0.1`` before
  the check. A non-loopback endpoint is a construction-time
  :class:`RewriteConfigError` — never a runtime retry against somewhere else.
* **No proxy, no redirects.** ``http.client`` does not consult
  ``HTTP(S)_PROXY``/``ALL_PROXY`` (unlike ``urllib.request``) and does not
  follow redirects; a 3xx is a failure here, not a hop. That is the whole
  reason for using it instead of ``urllib``.
* **Bounded.** One request per call, no pooling, a fixed socket timeout, and a
  streaming read that refuses a response larger than
  :data:`MAX_RESPONSE_BYTES` rather than buffering it.
* **Transport injected.** The socket work sits behind ``send(body) -> bytes``;
  tests pass a fake and the CPU guard (``tests/cpu_guard/sitecustomize.py``)
  denies socket construction outright, so nothing here is exercised on a socket
  in the suite.

Nothing in this module reads or logs the draft text: callers pass already-built
messages, and failures carry status/length facts only.
"""
from __future__ import annotations

import http.client
import ipaddress
import json
import socket
import time
from typing import Any, Callable, Mapping, NamedTuple
from urllib.parse import urlsplit

from qwen_engine.rewrite import (
    SAMPLING_WIRE_FIELDS,
    RewriteError,
    check_task,
    sampling,
)

#: Hard cap on a response body, enforced while reading. The client's own limit
#: (``app.js:509``, 256 KiB) binds first for honest replies; this is the
#: backstop against a misbehaving backend, not a size the UI must handle.
MAX_RESPONSE_BYTES = 1024 * 1024

#: Default budget for one rewrite, in seconds. The worst measured round trip
#: against the shared service was 38 s; a CPU or GGUF backend needs this raised
#: explicitly, which is a configuration decision, never an accident.
DEFAULT_TIMEOUT_SECONDS = 180.0

#: Bytes requested per read; small enough that an oversized body is refused
#: before it is fully buffered.
_READ_CHUNK = 65536

#: The service's thinking switch. Upstream runs with thinking ON and the answer
#: contract assumes it; the probe against the shared service used the same flag.
_THINKING_ON = {"enable_thinking": True}


class RewriteClientError(RewriteError):
    """Base for typed transport failures (mapped to 5xx by the web layer)."""


class RewriteConfigError(RewriteClientError):
    """Bad endpoint/model/timeout: a startup mistake, not a runtime outcome."""


class RewriteTimeout(RewriteClientError):
    """The rewrite exceeded its budget; the draft is untouched."""


class RewriteUnavailable(RewriteClientError):
    """The service could not be reached, or answered non-2xx."""


class RewriteInvalidAnswer(RewriteClientError):
    """The service answered, but not with a chat completion."""


class Endpoint(NamedTuple):
    """A verified loopback target."""

    host: str
    port: int
    path: str

    @property
    def host_header(self) -> str:
        """``Host`` value; an IPv6 literal keeps its brackets."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{host}:{self.port}"

    @property
    def url(self) -> str:
        """Display form. Loopback only, so this leaks no operator detail."""
        return f"http://{self.host_header}{self.path}"


def parse_endpoint(endpoint: Any) -> Endpoint:
    """Verify a rewriter endpoint URL and reduce it to ``(host, port, path)``.

    Refused, each with a :class:`RewriteConfigError`: a non-``http`` scheme
    (TLS to a loopback service would need a trust decision nobody made), user
    info, a query or fragment, a path that is not absolute, a missing or out-of
    range port, an unparseable or non-loopback host, and any hostname other than
    ``localhost``. The error text never echoes credentials (there must be none).
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise RewriteConfigError("rewrite endpoint must be a non-empty string")
    if endpoint != endpoint.strip():
        raise RewriteConfigError("rewrite endpoint has surrounding whitespace")
    if any(ch.isspace() for ch in endpoint):
        raise RewriteConfigError(
            "rewrite endpoint must not contain whitespace")
    try:
        parts = urlsplit(endpoint)
        port = parts.port
    except ValueError as exc:
        raise RewriteConfigError(
            f"rewrite endpoint is not a valid URL ({exc})") from None
    if parts.scheme != "http":
        raise RewriteConfigError(
            "rewrite endpoint must use http:// (a loopback service; no TLS "
            f"policy exists for one), got {parts.scheme or 'no scheme'!r}")
    if parts.username is not None or parts.password is not None:
        raise RewriteConfigError("rewrite endpoint must not carry user info")
    if "?" in endpoint or "#" in endpoint:
        raise RewriteConfigError(
            "rewrite endpoint must not carry a query or fragment")
    if port is None:
        raise RewriteConfigError("rewrite endpoint must name an explicit port")
    if not 1 <= port <= 65535:
        raise RewriteConfigError(
            f"rewrite endpoint port must be 1-65535, got {port}")
    host = parts.hostname
    if not host:
        raise RewriteConfigError("rewrite endpoint must name a host")
    if host == "localhost":
        host = "127.0.0.1"
    else:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise RewriteConfigError(
                "rewrite endpoint host must be an IP literal or 'localhost' "
                f"(a hostname would have to be resolved), got {host!r}") from None
        if not address.is_loopback:
            raise RewriteConfigError(
                "rewrite endpoint must be loopback: the rewriter runs on this "
                f"machine and a draft is user text, got {host!r}")
    path = parts.path or "/"
    if not path.startswith("/"):
        raise RewriteConfigError(
            f"rewrite endpoint path must be absolute, got {path!r}")
    return Endpoint(host=host, port=port, path=path)


def _read_capped(fp: Any, limit: int = MAX_RESPONSE_BYTES) -> bytes:
    """Read `fp` fully, refusing to buffer more than `limit` bytes."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = fp.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise RewriteInvalidAnswer(
                f"rewriter response exceeds the {limit} byte cap")
        chunks.append(chunk)
    return b"".join(chunks)


def _check_complete(body: bytes, declared: Any) -> bytes:
    """Refuse a reply shorter than its declared length.

    A short read means the reply was cut off mid-document; the remainder is not
    coming, so whatever parsed is incomplete.
    """
    if declared is not None and declared.isdigit() and len(body) != int(declared):
        raise RewriteInvalidAnswer(
            f"rewriter response truncated: got {len(body)} of "
            f"{declared} bytes")
    return body


class HttpTransport:
    """One POST per call over a fresh ``http.client`` connection."""

    def __init__(self, endpoint: Endpoint, timeout: float) -> None:
        self._endpoint = endpoint
        self._timeout = timeout

    def __call__(self, body: bytes) -> bytes:
        connection = http.client.HTTPConnection(
            self._endpoint.host, self._endpoint.port, timeout=self._timeout)
        try:
            connection.request(
                "POST", self._endpoint.path, body=body,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json",
                         "Host": self._endpoint.host_header})
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                raise RewriteUnavailable(
                    f"rewriter service answered HTTP {response.status}")
            declared = response.getheader("Content-Length")
            return _check_complete(_read_capped(response), declared)
        except RewriteClientError:
            raise
        except (TimeoutError, socket.timeout):
            raise RewriteTimeout(
                f"rewriter service timed out after {self._timeout:g}s") from None
        except OSError as exc:
            raise RewriteUnavailable(
                f"rewriter service unreachable ({type(exc).__name__})") from None
        finally:
            # One request per connection: closing releases the socket without
            # waiting for the service to close first.
            connection.close()


class RewriteClient:
    """Calls the rewriter and returns the assistant message content verbatim."""

    def __init__(self, endpoint: str | Endpoint, model: str,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 transport: Callable[[bytes], bytes] | None = None) -> None:
        self.endpoint = (endpoint if isinstance(endpoint, Endpoint)
                         else parse_endpoint(endpoint))
        if not isinstance(model, str) or not model.strip():
            raise RewriteConfigError("rewrite model must be a non-empty string")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise RewriteConfigError(
                f"rewrite timeout must be a number, got {type(timeout).__name__}")
        if not 0 < float(timeout) <= 3600:
            raise RewriteConfigError(
                f"rewrite timeout must be in (0, 3600] seconds, got {timeout!r}")
        self.model = model
        self.timeout = float(timeout)
        self._transport = (transport if transport is not None
                           else HttpTransport(self.endpoint, self.timeout))

    def build_body(self, messages: list[dict[str, Any]], task: str,
                   extra_sampling: Mapping[str, Any] | None = None) -> bytes:
        """The exact request bytes for one rewrite.

        Sends the verified subset of the profile (``SAMPLING_WIRE_FIELDS``);
        `extra_sampling` adds further profile values under their wire names,
        which is how an operator enables ``presence_penalty`` etc. after
        measuring the service actually honours them. Unknown keys are refused —
        a silently ignored sampling field is the failure this guards against.
        """
        check_task(task)
        if not isinstance(messages, list) or not messages:
            raise RewriteClientError("messages must be a non-empty list")
        profile = sampling(task)
        body: dict[str, Any] = {"model": self.model, "messages": messages,
                                "stream": False}
        for name, wire in SAMPLING_WIRE_FIELDS.items():
            body[wire] = profile[name]
        for name, value in (extra_sampling or {}).items():
            if name not in profile:
                raise RewriteClientError(
                    f"unknown sampling field {name!r}; expected one of "
                    f"{sorted(profile)}")
            body[name] = value
        body["chat_template_kwargs"] = dict(_THINKING_ON)
        try:
            return json.dumps(body, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise RewriteClientError(
                f"rewrite request is not JSON-serialisable ({type(exc).__name__})"
            ) from None

    def complete(self, messages: list[dict[str, Any]], task: str,
                 extra_sampling: Mapping[str, Any] | None = None) -> str:
        """Send one rewrite request and return ``choices[0].message.content``.

        Raises a :class:`RewriteClientError` subclass on every failure mode, so
        a caller never has to interpret a raw response: timeout, transport,
        non-2xx, oversize, non-JSON, a document that is not an OpenAI chat
        completion, and — reserved for a transport violation only — a missing
        ``choices``/non-string ``content``. A *content* problem is not an error
        here: the caller parses it and reports an unparsed answer.
        """
        body = self.build_body(messages, task, extra_sampling)
        started = time.monotonic()
        raw = self._transport(body)
        elapsed = time.monotonic() - started
        if elapsed > self.timeout:
            # The budget expired even though bytes arrived: a slow drip must
            # not be reported as a normal answer.
            raise RewriteTimeout(
                f"rewriter exceeded its {self.timeout:g}s budget "
                f"(took {elapsed:.1f}s)")
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RewriteInvalidAnswer(
                f"rewriter response exceeds the {MAX_RESPONSE_BYTES} byte cap")
        try:
            document = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RewriteInvalidAnswer(
                "rewriter response is not JSON") from None
        if not isinstance(document, dict):
            raise RewriteInvalidAnswer(
                "rewriter response is not a JSON object")
        choices = document.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RewriteInvalidAnswer(
                "rewriter response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RewriteInvalidAnswer(
                "rewriter response has no message content string")
        return content
