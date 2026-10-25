"""Explicit-address web server with opt-in serial engine jobs.

Safety posture (reviewed W2 contract):

* Defaults to literal ``127.0.0.1``. Repeated ``--host`` allows explicit
  IPv4 LAN/Tailscale listeners sharing one job queue. Wildcard, DNS and
  public-address binds are refused; each listener validates its own Host.
* Default port 8765 (the dsh harness port 3081 is unrelated and must never
  be used); ``--port 0`` picks an ephemeral port and the actual canonical
  origin is printed after bind.
* Exactly ONE ``Host`` header, validated against the bound literal + the
  actual port (plus the ``localhost`` alias for normal browser use — DNS
  rebinding attempts carry the attacker's own hostname and are refused).
  No DNS is performed for any of it.
* Foreign ``Origin`` or a cross-site/cross-origin ``Sec-Fetch-Site`` value
  is refused on EVERY route — stricter than required, ``same-site`` is also
  denied because another port on loopback is same-site but NOT same-origin
  and no DNS is consulted to tell them apart. No CORS headers are ever
  emitted, and OPTIONS does not grant anything (plain 405).
* POST requires the exact ``X-Qwen-UI: 1`` header, exactly one plain ASCII
  decimal ``Content-Length`` (no signs/whitespace, cap 64 KiB *before* any
  read), rejects ``Transfer-Encoding``/chunked and unsupported ``Expect``.
* Static serving is a fixed allowlist map over the packaged assets — no
  directory listing, no ``translate_path``, no arbitrary host paths. API and
  error responses all carry the defensive header set. EVERY response —
  including the early malformed/unsupported-version refusals emitted inside
  ``parse_request`` before any valid version exists — carries a real
  HTTP/1.x status line: the inherited stdlib framing suppresses ALL header
  output while ``request_version`` is the HTTP/0.9 default, so that legacy
  state is framed explicitly instead of shipping a bare body.
* Legacy HTTP/0.9-class requests (no version token, or an explicit HTTP/0.x
  token) are refused with a typed, FRAMED 400 BEFORE any dispatch decision
  — never served, and their version is never normalized to make them
  servable. Supported HTTP/1.0/1.1 handling is unchanged.
* ``--help`` never binds, never writes, never imports ML.

This is a *local* development server on loopback — explicitly not a
hardened public-internet deployment. Container loopback is not the host's
(or a Tailscale peer's) loopback: reaching it from outside the sandbox would
need its own reviewed bind/auth policy (a later milestone).
"""
from __future__ import annotations

import argparse
import json
import ipaddress
import os
import re
import signal
import selectors
import queue
import socket
import stat as stat_module
import sys
import threading
from pathlib import Path
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import resources
from typing import Any

from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.web import api
from qwen_engine.web.api import ApiError, ApiResponse, WebRuntime

DEFAULT_HOST = "127.0.0.1"
#: Default bind; additional explicit private IPv4 addresses are opt-in.
ALLOWED_BIND_HOSTS = (DEFAULT_HOST,)
#: Accepted Host header names at the bound port: the bound literal plus the
#: usual browser alias for loopback. Everything else (including a rebound
#: attacker hostname) is refused.
ALLOWED_HOST_NAMES = ("127.0.0.1", "localhost")
_PRIVATE_BIND_NETWORKS = tuple(ipaddress.ip_network(value) for value in
                               ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12",
                                "192.168.0.0/16", "100.64.0.0/10"))
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 64 * 1024
#: Finite socket/read timeout applied before any header or body read.
CONNECTION_TIMEOUT_SECONDS = 30
WORKER_QUEUE_POLL_SECONDS = 1.0
#: Concurrency cap: a small pool of PERSISTENT worker threads serves this
#: single-user local development server. No thread is ever spawned per
#: connection (thread churn exhausts pid-limited containers), and extra
#: connections are refused (503) once the queue backs up.
MAX_CONCURRENT_HANDLERS = 4
MAX_QUEUED_CONNECTIONS = 16
#: Exactly application/json, optional utf-8 charset parameter — a form
#: content type is refused 415 before the body is ever read.
_JSON_CONTENT_TYPE_RE = re.compile(
    r"application/json(\s*;\s*charset\s*=\s*utf-8)?\Z", re.IGNORECASE)

STATIC_ASSETS = ("index.html", "app.css", "app.js")
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
STATUS_ROUTE = "/api/v1/status"
PLAN_ROUTE = "/api/v1/plan"
EXPORT_ROUTE = "/api/v1/config/export"
IMPORT_ROUTE = "/api/v1/config/import"
JOBS_ROUTE = "/api/v1/jobs"
_JOB_ROUTE = re.compile(r"/api/v1/jobs/([0-9a-f]{32})(?:/(image|cancel|forget))?\Z")

CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; "
       "connect-src 'self'; img-src 'self'; base-uri 'none'; "
       "frame-ancestors 'none'; form-action 'self'")

_DEFENSIVE_HEADERS = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("X-Frame-Options", "DENY"),
    ("Content-Security-Policy", CSP),
    ("Cache-Control", "no-store"),
    ("Connection", "close"),
)


# ---------------------------------------------------------------------------
# Pure request-policy helpers (no sockets — directly unit-testable)
# ---------------------------------------------------------------------------


def _port_ok(text: str) -> bool:
    return 1 <= len(text) <= 5 and text.isdigit() and text.isascii() and (text == "0" or not text.startswith("0")) \
        and 0 <= int(text) <= 65535


def valid_bind_host(host: str) -> bool:
    """Explicit IPv4 loopback, RFC1918 LAN or Tailscale/CGNAT addresses only.

    No wildcard or DNS resolution: the OS must own each requested address.
    """
    try:
        address = ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError:
        return False
    return any(address in network for network in _PRIVATE_BIND_NETWORKS)


def validate_host(values: list[str], bound_ip: str,
                  port: int) -> tuple[str | None, str | None]:
    """(expected_origin, error): exactly one Host header naming this listener
    literal (or localhost on 127.0.0.1) at the actual port; everything else —
    duplicates, userinfo, whitespace, non-ASCII, missing/foreign port, foreign
    name — is a typed error string."""
    if len(values) != 1:
        return None, f"exactly one Host header is required, got {len(values)}"
    host = values[0]
    if any(ch.isspace() for ch in host) or not host:
        return None, "Host must be non-empty without whitespace"
    if "@" in host:
        return None, "Host must not contain userinfo"
    if not host.isascii():
        return None, "Host must be ASCII"
    name, sep, port_part = host.rpartition(":")
    if not sep or not name:
        return None, "Host must include the bound port (host:port)"
    # Bounded digit conversion BEFORE int() — an arbitrary-length digit
    # string must be a typed refusal, never a ValueError -> 500.
    if not port_part.isdigit() or not port_part.isascii() \
            or len(port_part) > 5:
        return None, "Host port must be a plain ASCII decimal port (max 5 digits)"
    if len(port_part) > 1 and port_part[0] == "0":
        return None, "Host port must not have leading zeros"
    if int(port_part) != port:
        return None, "Host port must match the bound port"
    names = {bound_ip}
    if bound_ip == DEFAULT_HOST:
        names.add("localhost")
    if name.lower() not in names:
        return None, "Host name must match this listener address (or localhost on loopback)"
    return f"http://{name.lower()}:{port}", None


def header_policy(method: str, headers: dict[str, list[str]],
                  expected_origin: str) -> ApiError | None:
    """Post-Host header policy, pure. ``headers`` maps lowercase names to
    their raw values in wire order."""
    def dup(name: str) -> ApiError | None:
        values = headers.get(name, [])
        if len(values) > 1:
            return ApiError(400, "duplicate_header",
                            f"duplicate {name!r} headers are refused")
        return None

    for name in ("origin", "sec-fetch-site", "x-qwen-ui", "content-length",
                 "expect", "content-type"):
        err = dup(name)
        if err:
            return err

    site = headers.get("sec-fetch-site", [None])[0]
    if site is not None and site.lower() not in ("none", "same-origin"):
        # same-site is denied too: another port on loopback is same-site but
        # NOT same-origin, and no DNS is consulted to tell them apart.
        return ApiError(403, "cross_site_denied",
                        "this local server answers only same-origin requests")

    origin = headers.get("origin", [None])[0]
    if origin is not None and origin.lower() != expected_origin.lower():
        return ApiError(403, "foreign_origin_denied",
                        "Origin does not match this server's origin")

    if method != "POST":
        return None

    ui = headers.get("x-qwen-ui", [None])[0]
    if ui != "1":
        return ApiError(403, "ui_header_required",
                        "writes require the exact X-Qwen-UI: 1 header "
                        "(no cross-origin writes)")
    if "transfer-encoding" in headers:
        return ApiError(400, "transfer_encoding_unsupported",
                        "chunked/Transfer-Encoding requests are not supported")
    expect = headers.get("expect", [None])[0]
    if expect is not None:
        # NO Expect extension is supported — including 100-continue: the
        # server never emits an early 100 before Host/Origin/length
        # validation and never reads an unsupported Expect body.
        return ApiError(417, "expectation_failed",
                        "this server supports no Expect extensions")
    types = headers.get("content-type", [])
    if not types or not _JSON_CONTENT_TYPE_RE.match(types[0].strip()):
        return ApiError(415, "unsupported_media_type",
                        "POST body must be application/json (a form or "
                        "other content type is refused before the body is "
                        "read)")
    lengths = headers.get("content-length", [])
    if not lengths:
        return ApiError(411, "length_required",
                        "POST requires exactly one Content-Length")
    raw = lengths[0]
    if not raw.isdigit() or not raw.isascii() or (len(raw) > 1 and raw[0] == "0"):
        return ApiError(400, "invalid_content_length",
                        "Content-Length must be one plain ASCII decimal "
                        "without signs or whitespace")
    # Bounded digit conversion BEFORE int() (arbitrary-length digit strings
    # must be typed refusals, never a ValueError -> 500).
    if len(raw) > 6 or int(raw) > MAX_BODY_BYTES:
        return ApiError(413, "request_too_large",
                        f"request bodies are capped at {MAX_BODY_BYTES} bytes")
    return None


def resolve_route(method: str, target: str) -> tuple[str, str]:
    """Exact origin-form routing. Anything resembling an absolute target, a
    query, a fragment or any percent-encoding is refused BEFORE lookup; no
    path is ever translated to a filesystem path."""
    if not target.startswith("/") or target.startswith("//"):
        raise ApiError(400, "target_form_unsupported",
                       "only origin-form request targets are routed")
    if "?" in target or "#" in target or "%" in target:
        raise ApiError(400, "target_invalid",
                       "queries, fragments and encoded target variants are "
                       "not routed")
    if method == "GET":
        if target in STATIC_ROUTES:
            return "static", target
        if target == STATUS_ROUTE:
            return "status", target
        if target == JOBS_ROUTE:
            return "jobs", target
    elif method == "POST":
        if target == PLAN_ROUTE:
            return "plan", target
        if target == EXPORT_ROUTE:
            return "export", target
        if target == IMPORT_ROUTE:
            return "import", target
        if target == JOBS_ROUTE:
            return "submit", target
    match = _JOB_ROUTE.fullmatch(target)
    if match:
        action = match[2] or "job"
        allow = "POST" if action in ("cancel", "forget") else "GET"
        if method == allow:
            return action, match[1]
        raise ApiError(405, "method_not_allowed", "this route does not accept that method",
                       headers=(("Allow", allow),))
    if target == JOBS_ROUTE:
        raise ApiError(405, "method_not_allowed", "this route does not accept that method",
                       headers=(("Allow", "GET, POST"),))
    known = set(STATIC_ROUTES) | {STATUS_ROUTE, PLAN_ROUTE, EXPORT_ROUTE, IMPORT_ROUTE}
    if target in known:
        allow = "POST" if target in (PLAN_ROUTE, EXPORT_ROUTE, IMPORT_ROUTE) else "GET"
        raise ApiError(405, "method_not_allowed",
                       "this route does not accept that method",
                       headers=(("Allow", allow),))
    raise ApiError(404, "not_found", "no such route")


# ---------------------------------------------------------------------------
# Startup helpers (one-shot metadata reads; jobs verify weights separately)
# ---------------------------------------------------------------------------

_MANIFEST_LIMIT = 1 << 20  # <= 1 MiB, bounded read


def read_manifest_limited(path: str) -> tuple[bytes | None, str | None]:
    """One bounded, hang-proof read of the manifest file at startup: regular
    FD only (no FIFO open-block, no symlink follow), hard 1 MiB cap. Returns
    (bytes, None) or (None, typed reason code). Observed opened/published
    size or identity changes refuse; this is a snapshot, not verification."""
    path = Path(os.path.abspath(path))
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return None, "manifest_file_missing"
    except OSError:
        return None, api.MANIFEST_UNREADABLE
    try:
        try:
            st = os.fstat(fd)
            if not stat_module.S_ISREG(st.st_mode):
                return None, api.MANIFEST_UNREADABLE
            if st.st_size > _MANIFEST_LIMIT:
                return None, "manifest_too_large"
            # BOUNDED chunk loop to EOF (a single os.read may legitimately
            # return fewer bytes than requested); past the cap it is typed
            # too-large, never buffered further.
            chunks: list[bytes] = []
            total = 0
            while True:
                try:
                    chunk = os.read(fd, min(1 << 16, st.st_size + 1 - total))
                except OSError:
                    return None, api.MANIFEST_UNREADABLE
                if not chunk:
                    break
                total += len(chunk)
                if total > _MANIFEST_LIMIT:
                    return None, "manifest_too_large"
                if total > st.st_size:
                    return None, api.MANIFEST_UNREADABLE
                chunks.append(chunk)
            def signature(info):
                return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                        info.st_mtime_ns, info.st_ctime_ns)
            if (total != st.st_size or signature(st) != signature(os.fstat(fd))
                    or signature(st) != signature(path.stat(follow_symlinks=False))):
                return None, api.MANIFEST_UNREADABLE
        except OSError:
            return None, api.MANIFEST_UNREADABLE
        return b"".join(chunks), None
    finally:
        # A close failure must never mask the primary outcome.
        try:
            os.close(fd)
        except OSError:
            pass


def load_static_assets() -> dict[str, bytes]:
    root = resources.files("qwen_engine").joinpath("web_static")
    assets: dict[str, bytes] = {}
    for name in STATIC_ASSETS:
        assets[name] = root.joinpath(name).read_bytes()
    return assets


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwen-web/0.1"
    sys_version = ""
    timeout = CONNECTION_TIMEOUT_SECONDS

    # No request/error lines: user-controlled text never reaches logs
    # (the console only ever sees fixed startup/diagnostic lines).
    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
        pass

    def _header_map(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key, value in self.headers.items():
            out.setdefault(key.lower(), []).append(value)
        return out

    def _frame_method_token(self) -> str:
        """Method token for body suppression. ``self.command`` is set only
        once the stdlib parser has FULLY recognized the request; early
        errors carry the RAW request line only, so fall back to the same
        whitespace split the stdlib and this handler already use."""
        if self.command:
            return str(self.command)
        raw = self.raw_requestline
        if isinstance(raw, bytes):
            raw = raw.decode("iso-8859-1", "replace")
        parts = (raw or "").strip().split()
        return parts[0] if parts else ""

    def _send(self, status: int, body: bytes, ctype: str,
              extra: tuple[tuple[str, str], ...] = ()) -> None:
        if self.request_version == "HTTP/0.9":
            # FRAMING NORMALIZATION FOR THE LEGACY/ERROR STATE ONLY. Every
            # inherited header emitter (send_response_only/send_header/
            # end_headers) suppresses ALL output while request_version is
            # the HTTP/0.9 default — including the early errors emitted
            # INSIDE parse_request before any valid version was recognized
            # (broken request line, HTTP/2.x, header overflow) and the
            # legacy-refusal gate below. Those refusals must still honour
            # the all-responses contract, so frame them explicitly with a
            # real HTTP/1.1 status line and the SHARED _DEFENSIVE_HEADERS
            # constant (one builder, never a second header list). This is
            # framing ONLY: no legacy request is normalized into a
            # servable state — the parse_request gate refuses it BEFORE
            # dispatch, so this branch can only ever carry a refusal.
            try:
                phrase = HTTPStatus(status).phrase
            except ValueError:  # pragma: no cover - every code here is standard
                phrase = "Error"
            headless = self._frame_method_token() == "HEAD"
            # HEAD framing contract: headers — including the honest
            # Content-Length a GET would produce — but NEVER a payload.
            frame = _framed_bytes(f"HTTP/1.1 {status} {phrase}",
                                  b"" if headless else body,
                                  body_len=len(body), ctype=ctype,
                                  extra=extra, server=self.server_version)
            self.wfile.write(frame)
            self.close_connection = True
            return
        self.send_response_only(status)
        self.send_header("Server", self.server_version)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for name, value in _DEFENSIVE_HEADERS:
            self.send_header(name, value)
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        self.close_connection = True
        if self._frame_method_token() != "HEAD":  # HEAD: framed headers, no body
            self.wfile.write(body)

    def _send_api(self, resp: ApiResponse) -> None:
        if resp.body is not None:
            self._send(resp.status, resp.body, "application/json; charset=utf-8",
                       resp.headers)
            return
        payload = resp.payload if resp.payload is not None else \
            ApiError(500, "internal_error", "empty response").to_payload()
        # allow_nan=False: no non-finite number may ever become an invalid
        # JSON token on the wire, whatever upstream produced it.
        self._send(resp.status,
                   (json.dumps(payload, sort_keys=True,
                               allow_nan=False) + "\n").encode("utf-8"),
                   "application/json; charset=utf-8",
                   resp.headers)

    def send_error(self, code: int, message: str | None = None,
                   explain: str | None = None) -> None:
        """Even malformed-request paths get the typed JSON error + defensive
        headers — never the stock HTML error page, never a traceback."""
        err = ApiError(code, "malformed_request",
                       message or HTTPStatus(code).phrase)
        self._send_api(ApiResponse(status=code, payload=err.to_payload()))

    def parse_request(self) -> bool:
        ok = super().parse_request()
        if not ok:
            # The stdlib already emitted the error — through OUR send_error/
            # _send, which frames it properly even in the HTTP/0.9 default
            # state (see _send). Nothing more to decide here.
            return False
        # LEGACY REFUSAL GATE — BEFORE any dispatch decision. The stdlib
        # returns True with request_version LEFT at the HTTP/0.9 default
        # when the request line carries no version token (a bare
        # 'GET /path'), and keeps an explicit 'HTTP/0.x' token verbatim.
        # Serving anything in that state would emit a bare legacy body —
        # no status, no headers — so HTTP/0.x-class requests are refused
        # here with a typed, FRAMED 400. The request version is NEVER
        # normalized to make a legacy request servable: dispatch happens
        # only for HTTP/1.x. Supported HTTP/1.0/1.1 behavior is unchanged.
        if not str(self.request_version).startswith("HTTP/1."):
            self.close_connection = True
            err = ApiError(400, "unsupported_request_version",
                           "legacy HTTP/0.9-class requests are refused; "
                           "every response needs an HTTP/1.x version")
            self._send_api(ApiResponse(status=err.status,
                                       payload=err.to_payload()))
            return False
        # Keep the RAW wire target: the stdlib normalizes a leading '//'
        # in self.path before we would see it — routing must consider
        # both, so a real '//x' request is refused, never served.
        # Split EXACTLY like the stdlib (str.split() on any run of
        # whitespace): a TAB-separated request line must still yield the
        # real target token, never fall back to the normalized path.
        raw = self.raw_requestline
        if isinstance(raw, bytes):
            raw = raw.decode("iso-8859-1", "replace")
        parts = (raw or "").strip().split()
        # '' when the line was uncapturable or had no target token:
        # resolve_route then refuses — there is NO normalized fallback.
        self.raw_target = parts[1] if len(parts) >= 2 else ""
        return True

    def handle_expect_100(self) -> bool:
        # NO Expect extension is supported (including 100-continue): never
        # send an early 100 before Host/Origin/length validation, never read
        # an unsupported Expect body. The 417 itself goes through the same
        # Host/header policy first.
        headers = self._header_map()
        host_origin, host_err = validate_host(headers.get("host", []),
                                              self.server.bound_ip,
                                              self.server.bound_port)
        if host_err is not None:
            err = ApiError(400, "invalid_host", host_err)
        else:
            err = header_policy(self.command, headers, host_origin) or \
                ApiError(417, "expectation_failed",
                         "this server supports no Expect extensions")
        self._send_api(ApiResponse(status=err.status, payload=err.to_payload(),
                                   headers=err.headers))
        return False

    def handle_one_request(self) -> None:
        # EVERY parsed method — including ones with no do_* dispatcher at
        # all (TRACE, FOOBAR, ...) — goes through the SAME Host/Origin
        # header policy and the typed 405/404 JSON path, instead of the
        # stock unguarded 501 HTML error page. Structural mirror of the
        # stdlib implementation with the unknown-method branch re-pointed.
        try:
            try:
                self.raw_requestline = self.rfile.readline(65537)
                if len(self.raw_requestline) > 65536:
                    # The stock stdlib replies with a typed 414 here: keep
                    # that (defensive JSON 414 via our send_error), with
                    # request_version/command initialised so the stock
                    # HTTP/0.9 header suppression can never hide it.
                    self.requestline = ""
                    self.request_version = ""
                    self.command = ""
                    self.close_connection = True
                    self.send_error(HTTPStatus.REQUEST_URI_TOO_LONG)
                    return
                if not self.raw_requestline:
                    self.close_connection = True
                    return
                if not self.parse_request():  # an error was already sent
                    return
                name = "do_" + self.command
                method = getattr(self, name, None) if name.isidentifier() else None
                if method is not None:
                    method()
                else:
                    self._dispatch(self.command)
            except TimeoutError as exc:  # pragma: no cover - socket timeout
                self.log_error("Request timed out: %r", exc)
                self.close_connection = True
                return
            self.wfile.flush()
        finally:
            self.request_version = None
            self.command = None

    def _dispatch(self, method: str) -> None:
        try:
            self.close_connection = True  # POST/error close; static too (simple)
            headers = self._header_map()
            host_origin, host_err = validate_host(headers.get("host", []),
                                                  self.server.bound_ip,
                                                  self.server.bound_port)
            if host_err is not None:
                err = ApiError(400, "invalid_host", host_err)
                self._send_api(ApiResponse(status=err.status, payload=err.to_payload()))
                return
            err = header_policy(method, headers, host_origin)
            if err is not None:
                self._send_api(ApiResponse(status=err.status, payload=err.to_payload(),
                                           headers=err.headers))
                return
            # Route on the RAW wire target ONLY (never the stdlib-normalized
            # self.path): the stdlib folds '//' in self.path before the app
            # sees it, and TAB request lines reach self.path normalized too.
            # An uncapturable/absent raw target is '' -> resolve_route refuses.
            target = getattr(self, "raw_target", "")
            kind, arg = resolve_route(method, target)
            if kind == "static":
                asset, ctype = STATIC_ROUTES[arg]
                self._send(200, self.server.assets[asset], ctype)
            elif kind == "status":
                self._send_api(api.status_response(self.server.web_runtime))
            elif kind in ("jobs", "job", "image"):
                jobs = getattr(self.server, "generation", None)
                if jobs is None:
                    raise ApiError(503, "generation_disabled", "generation is disabled on this server")
                if kind == "image":
                    filename = jobs.image_filename(arg)
                    image_bytes = jobs.image(arg)
                    self._send(200, image_bytes, jobs.image_content_type(image_bytes),
                               (("Content-Disposition", f'inline; filename="{filename}"'),))
                else:
                    self._send_api(jobs.list_response() if kind == "jobs" else jobs.job_response(arg))
            else:
                length = int(headers["content-length"][0])
                body = self.rfile.read(length)
                if len(body) != length:
                    self.send_error(400, "request body ended early")
                    return
                if kind == "plan":
                    self._send_api(api.plan_response(self.server.web_runtime, body))
                elif kind == "export":
                    self._send_api(api.export_response(self.server.web_runtime, body))
                elif kind == "import":
                    self._send_api(api.import_response(self.server.web_runtime, body))
                else:
                    jobs = getattr(self.server, "generation", None)
                    if jobs is None:
                        raise ApiError(503, "generation_disabled", "generation is disabled on this server")
                    if kind == "submit":
                        self._send_api(jobs.submit(body))
                    else:
                        if api.parse_request_json(body) != {}:
                            raise ApiError(400, "invalid_envelope", "job actions require an empty JSON object")
                        self._send_api(jobs.job_response(arg, action=kind))
        except ApiError as exc:
            self._send_api(ApiResponse(status=exc.status, payload=exc.to_payload(),
                                       headers=exc.headers))
        except (TimeoutError, socket.timeout, ConnectionError, OSError):
            # A dead/slow socket is simply closed; nothing to report.
            self.close_connection = True
        except Exception:
            self._send_api(api.internal_error_response())

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    # EVERY other method goes through the same Host/Origin/header policy
    # BEFORE the typed 405/404 — no method bypasses the checks. OPTIONS
    # grants nothing: no CORS preflight, ever (the 405 carries Allow, never
    # an Access-Control-* header).
    def do_OPTIONS(self) -> None:
        self._dispatch("OPTIONS")

    def do_HEAD(self) -> None:
        self._dispatch("HEAD")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def do_PATCH(self) -> None:
        self._dispatch("PATCH")

    def do_TRACE(self) -> None:
        self._dispatch("TRACE")


# ---------------------------------------------------------------------------
# Server: bounded persistent worker pool with a finite, leak-free close
# ---------------------------------------------------------------------------


#: Sentinel asked of each worker at graceful shutdown (never a connection).
_WORKER_SHUTDOWN = object()


def drain_connection_queue(conn_queue: "queue.Queue[Any]") -> int:
    """Close every CONNECTION still WAITING in the queue without handling
    it (graceful shutdown must not leave queued sockets dangling). Shutdown
    SENTINELS are recognised and re-posted to task_done, never unpacked —
    draining twice (or after a worker failed to consume its sentinel) is
    safe. Returns the number of connections closed. Pure: takes a queue,
    needs no sockets of its own."""
    closed = 0
    while True:
        try:
            item = conn_queue.get_nowait()
        except queue.Empty:
            return closed
        if item is not _WORKER_SHUTDOWN:
            request, _client_address = item
            try:
                request.close()
            except OSError:  # pragma: no cover - already closed
                pass
            closed += 1
        conn_queue.task_done()


def stop_workers(conn_queue: "queue.Queue[Any]",
                 workers: list[threading.Thread],
                 put_timeout: float = 2.0,
                 join_timeout: float = 5.0) -> list[threading.Thread]:
    """Ask each started worker to exit (one sentinel per worker) and join
    with FINITE timeouts — never blocking forever. Returns the workers
    STILL ALIVE after their bounded join (the caller must keep references
    to them — no false claim that everything joined). A worker
    mid-request has an idle socket timeout, not a total request deadline;
    slow trickle clients may outlive the joins. References to still-live
    daemon threads are retained rather than claiming they all stopped."""
    for _ in workers:
        try:
            conn_queue.put(_WORKER_SHUTDOWN, timeout=put_timeout)
        except Exception:  # full/broken queue: idle workers poll _stopping
            break
    still_alive: list[threading.Thread] = []
    for t in workers:
        t.join(join_timeout)
        if t.is_alive():
            still_alive.append(t)
    return still_alive


class _Server(HTTPServer):
    """Explicit-address server with a BOUNDED POOL of persistent worker threads —
    no per-connection thread creation, finite handler idle timeout via
    _Handler.timeout, typed 503 when saturated."""
    # Allow a clean restart while old connections remain in TIME_WAIT.
    # This is SO_REUSEADDR, not SO_REUSEPORT: no shared live listeners.
    allow_reuse_address = True

    web_runtime: WebRuntime
    assets: dict[str, bytes]
    bound_ip: str
    bound_port: int

    def __init__(self, addr_info, handler) -> None:
        self._queue: "queue.Queue[Any]" = queue.Queue(
            maxsize=MAX_QUEUED_CONNECTIONS)
        self._workers: list[threading.Thread] = []
        self._pool_ready = False
        self._stopping = False
        # TCPServer calls our server_close() if bind fails inside its init.
        # The worker state must exist before entering that constructor.
        super().__init__(addr_info, handler)

    def _ensure_pool(self) -> None:
        # Lazily started on the first connection: startup itself then costs
        # no extra threads (kind to pid-limited containers). If the box
        # cannot start ANY worker thread we serve serially instead of
        # dying — still correct, just single-flight.
        if self._pool_ready or self._stopping:
            return
        self._pool_ready = True
        for i in range(MAX_CONCURRENT_HANDLERS):
            t = threading.Thread(target=self._worker,
                                 name=f"qwen-web-worker-{i}", daemon=True)
            try:
                t.start()
                self._workers.append(t)
            except RuntimeError:  # cannot start thread here — serial fallback
                break

    def process_request(self, request, client_address) -> None:
        self._ensure_pool()
        if self._stopping:
            # Graceful shutdown already started: refuse without queueing so
            # no socket is ever left dangling in the queue.
            try:
                request.sendall(_OVERLOAD_BYTES)
            except OSError:
                pass
            self.shutdown_request(request)
            return
        if not self._workers:
            # Serial fallback: handle in the accept thread (bounded by the
            # handler socket timeout), never spawn per connection.
            try:
                try:
                    self.finish_request(request, client_address)
                except Exception:
                    self.handle_error(request, client_address)
                finally:
                    self.shutdown_request(request)
            except Exception:
                pass
            return
        try:
            self._queue.put_nowait((request, client_address))
        except queue.Full:
            # Saturate guard: fixed bytes, no thread, no buffering.
            try:
                request.sendall(_OVERLOAD_BYTES)
            except OSError:
                pass
            self.shutdown_request(request)

    def _worker(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=WORKER_QUEUE_POLL_SECONDS)
            except queue.Empty:
                if self._stopping:
                    return
                continue
            except OSError:  # interpreter teardown
                return
            if item is _WORKER_SHUTDOWN:   # graceful shutdown sentinel
                self._queue.task_done()
                return
            request, client_address = item
            try:
                try:
                    self.finish_request(request, client_address)
                except Exception:
                    self.handle_error(request, client_address)
                finally:
                    self.shutdown_request(request)
            finally:
                self._queue.task_done()

    def server_close(self) -> None:
        # Graceful shutdown must not leak its OWN pool and must be IDEMPOTENT
        # (server_close can be reached more than once): close the listening
        # socket, CLOSE every still-queued connection unhandled (never
        # dangling — sentinels are recognised, never unpacked), then ask
        # each worker to exit and join with finite timeouts. Workers that
        # did NOT join keep their references (no false 'all joined' claim);
        # they are daemons, bounded by the handler socket timeout.
        if self._stopping:
            return
        self._stopping = True
        super().server_close()
        drain_connection_queue(self._queue)
        still_alive = stop_workers(self._queue, self._workers)
        self._workers = still_alive


#: Prebuilt overload refusal — built ONCE from the shared _DEFENSIVE_HEADERS
#: constant so this raw path can NEVER drift away from the defensive header
#: set every other response carries (CSP, nosniff, frame-deny, no-referrer,
#: CORP, no-store, close). It must stay far below the kernel socket buffer:
#: one sendall, never reading the (possibly oversize) request.
def _framed_bytes(status_line: str, body: bytes, *,
                  body_len: int | None = None,
                  ctype: str = "application/json; charset=utf-8",
                  extra: tuple[tuple[str, str], ...] = (),
                  server: str | None = None) -> bytes:
    """THE one status-line + shared-headers + body frame builder. Used by
    the prebuilt overload refusal and by the handler's legacy/error-state
    normalization (_send) — both from the SAME _DEFENSIVE_HEADERS constant,
    so neither can drift. latin-1 like the stdlib header emitters."""
    if body_len is None:
        body_len = len(body)
    lines = [status_line.encode("latin-1")]
    if server is not None:
        lines.append(f"Server: {server}".encode("latin-1"))
    lines.append(f"Content-Type: {ctype}".encode("latin-1"))
    lines += [f"{name}: {value}".encode("latin-1")
              for name, value in _DEFENSIVE_HEADERS]
    lines += [f"{name}: {value}".encode("latin-1") for name, value in extra]
    lines.append(f"Content-Length: {body_len}".encode("latin-1"))
    return b"\r\n".join(lines) + b"\r\n\r\n" + body


def _raw_refusal_bytes(status_line: str, body: bytes) -> bytes:
    return _framed_bytes(status_line, body)


_OVERLOAD_BODY = (
    b'{"schema":"qwen_engine.web_error.v1","code":"overloaded",'
    b'"message":"too many concurrent connections; retry"}\n'
)
_OVERLOAD_BYTES = _raw_refusal_bytes("HTTP/1.1 503 Service Unavailable",
                                     _OVERLOAD_BODY)


# ---------------------------------------------------------------------------
# CLI + entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qwen-web",
        description=(
            "Qwen Image Studio local web server: serves the workspace, "
            "status/plan/config-export API and opt-in serial generation jobs. Planning "
            "is available by default. Explicit IPv4 loopback/LAN/Tailscale binds. "
            "Omitted path options mean 'unconfigured': the server "
            "then writes nothing anywhere."
        ),
    )
    p.add_argument("--host", action="append", default=None,
                   help="explicit IPv4 loopback, LAN or Tailscale address; repeat "
                        "to share one job queue across listeners (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"TCP port (default {DEFAULT_PORT}; 0 picks an "
                        "ephemeral port and prints the actual origin)")
    p.add_argument("--manifest", default=None,
                   help="model manifest for metadata-only status summaries "
                        "(read once at startup, <=1 MiB, never verified here)")
    p.add_argument("--models-dir", default=None,
                   help="model storage root (metadata context only for W2)")
    p.add_argument("--q8-manifest", default=None,
                   help="server-owned Q8 GGUF sidecar manifest; never selectable by browser paths")
    p.add_argument("--enable-generation", action="store_true",
                   help="enable real serial engine jobs (requires manifest, models, output and cache paths)")
    p.add_argument("--attest-unbounded-host", action="store_true",
                   help="operator attestation of no cgroup RAM cap; encoder uses fresh host MemAvailable")
    for flag in ("output-dir", "cache-dir", "log-dir", "benchmarks-dir"):
        p.add_argument(f"--{flag}", default=None,
                       help=f"server runtime path {flag!r} (never settable "
                            "from the browser; omitted = unconfigured/no writes)")
    return p


def _startup_config_or_exit(paths: dict[str, str | None]) -> int:
    try:
        GenerationConfig(prompt="qwen-web startup validation", **paths)
    except ConfigError as exc:
        for err in exc.errors:
            print(f"config error: {err}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hosts = list(dict.fromkeys(args.host or [DEFAULT_HOST]))
    if len(hosts) > 8 or any(not valid_bind_host(host) for host in hosts):
        print("error: refusing to bind: use at most eight explicit IPv4 loopback, "
              "RFC1918 LAN or Tailscale addresses; wildcards, DNS names, public "
              "addresses and IPv6 are unsupported", file=sys.stderr)
        return 2
    if not 0 <= args.port <= 65535:
        print("error: --port must be in [0, 65535]", file=sys.stderr)
        return 2

    paths = {
        # SAFELY RESOLVED at startup (realpath normalizes symlinks/'..' —
        # never abspath-only, which would leave a final-symlink swap to
        # request time; the manifest FD is then opened with O_NOFOLLOW on
        # the resolved path, so the read cannot be diverted either).
        "output_dir": None if args.output_dir is None else os.path.realpath(args.output_dir),
        "cache_dir": None if args.cache_dir is None else os.path.realpath(args.cache_dir),
        "log_dir": None if args.log_dir is None else os.path.realpath(args.log_dir),
        "benchmarks_dir": None if args.benchmarks_dir is None else os.path.realpath(args.benchmarks_dir),
        "models_dir": None if args.models_dir is None else os.path.realpath(args.models_dir),
        "manifest_path": None if args.manifest is None else os.path.realpath(args.manifest),
        "q8_manifest_path": None if args.q8_manifest is None else os.path.realpath(args.q8_manifest),
    }
    status = _startup_config_or_exit(paths)
    if status:
        return status

    if paths["manifest_path"] is None:
        summary = api.summarize_manifest_bytes(None)
    else:
        raw, unreadable = read_manifest_limited(paths["manifest_path"])
        summary = api.summarize_manifest_bytes(raw, unreadable)

    runtime = WebRuntime(**paths, manifest_summary=summary,
                         generation_enabled=args.enable_generation)
    if args.enable_generation and api._blockers(runtime):
        print("error: generation requires valid manifest metadata and explicit "
              "--manifest, --models-dir, --output-dir and --cache-dir", file=sys.stderr)
        return 2
    try:
        assets = load_static_assets()
    except OSError as exc:
        print(f"error: packaged static assets are not readable: {exc.strerror or exc}",
              file=sys.stderr)
        return 1

    listeners = []
    port = args.port
    try:
        for host in hosts:
            httpd = _Server((host, port), _Handler)
            listeners.append(httpd)
            port = int(httpd.server_address[1])
            httpd.web_runtime = runtime
            httpd.assets = assets
            httpd.bound_ip = host
            httpd.bound_port = port
    except OSError as exc:
        for listener in listeners:
            listener.server_close()
        print(f"error: cannot bind http://{host}:{port}: "
              f"{exc.strerror or exc}", file=sys.stderr)
        return 1
    httpd.generation = None
    if args.enable_generation:
        from qwen_engine.web.generation import WebGeneration
        try:
            Path(runtime.output_dir).mkdir(parents=True, exist_ok=True)
            httpd.generation = WebGeneration(runtime, trust_unbounded_host=args.attest_unbounded_host)
        except (OSError, ValueError, RuntimeError):
            for listener in listeners:
                listener.server_close()
            print("error: generation output storage or worker could not be initialized", file=sys.stderr)
            return 1
        if args.attest_unbounded_host:
            print("notice: operator attests no cgroup RAM limit for generation; "
                  "the encoder still checks fresh host MemAvailable", file=sys.stderr)

    for listener in listeners:
        listener.generation = httpd.generation
        print(f"qwen-web listening on http://{listener.bound_ip}:{port}")
    mode = "generation enabled (serial jobs)" if httpd.generation else "planning only (no generation)"
    print(f"qwen-web: {mode}; explicit address binds with per-listener "
          "Host and same-origin request checks.")
    sys.stdout.flush()
    def terminate(signum, frame):
        raise KeyboardInterrupt

    old_term = signal.signal(signal.SIGTERM, terminate)
    stopped = True
    try:
        if len(listeners) == 1:
            httpd.serve_forever()
        else:
            with selectors.DefaultSelector() as selector:
                for listener in listeners:
                    selector.register(listener, selectors.EVENT_READ)
                while True:
                    for key, _ in selector.select(timeout=.5):
                        key.fileobj._handle_request_noblock()
    except KeyboardInterrupt:
        pass
    finally:
        for listener in listeners:
            listener.server_close()
        if httpd.generation is not None:
            stopped = httpd.generation.close()
        signal.signal(signal.SIGTERM, old_term)
    if not stopped:
        print("error: generation worker did not finish shutdown within 60 seconds", file=sys.stderr)
    return 0 if stopped else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
