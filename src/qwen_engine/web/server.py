"""W2 local web server: loopback-only stdlib transport for the web API.

Safety posture (reviewed W2 contract):

* Binds the **literal** ``127.0.0.1`` only. ``--host`` is refused BEFORE any
  bind for anything else — including ``::1`` (an IPv6 loopback bind needs its
  own reviewed, actually-tested milestone) and any non-loopback address.
  ``localhost`` is never resolved to decide the bind address.
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
  error responses all carry the defensive header set.
* ``--help`` never binds, never writes, never imports ML.

This is a *local* development server on loopback — explicitly not a
hardened public-internet deployment. Container loopback is not the host's
(or a Tailscale peer's) loopback: reaching it from outside the sandbox would
need its own reviewed bind/auth policy (a later milestone).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import queue
import socket
import stat as stat_module
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import resources
from typing import Any

from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.web import api
from qwen_engine.web.api import ApiError, ApiResponse, WebRuntime

DEFAULT_HOST = "127.0.0.1"
#: Only the literal loopback (W2 policy; ::1 refused until an actual
#: IPv6-tested bind milestone is reviewed). Never "localhost" — no DNS.
ALLOWED_BIND_HOSTS = (DEFAULT_HOST,)
#: Accepted Host header names at the bound port: the bound literal plus the
#: usual browser alias for loopback. Everything else (including a rebound
#: attacker hostname) is refused.
ALLOWED_HOST_NAMES = ("127.0.0.1", "localhost")
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 64 * 1024
#: Finite socket/read timeout applied before any header or body read.
CONNECTION_TIMEOUT_SECONDS = 30
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
    return text.isdigit() and text.isascii() and (text == "0" or not text.startswith("0")) \
        and 0 <= int(text) <= 65535


def validate_host(values: list[str], bound_ip: str,
                  port: int) -> tuple[str | None, str | None]:
    """(expected_origin, error): exactly one Host header naming the bound
    loopback literal (or localhost) at the actual port; everything else —
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
    if name.lower() not in ALLOWED_HOST_NAMES:
        return None, "Host name must be the bound loopback address (or localhost)"
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
    elif method == "POST":
        if target == PLAN_ROUTE:
            return "plan", target
        if target == EXPORT_ROUTE:
            return "export", target
    known = set(STATIC_ROUTES) | {STATUS_ROUTE, PLAN_ROUTE, EXPORT_ROUTE}
    if target in known:
        allow = "POST" if target in (PLAN_ROUTE, EXPORT_ROUTE) else "GET"
        raise ApiError(405, "method_not_allowed",
                       "this route does not accept that method",
                       headers=(("Allow", allow),))
    raise ApiError(404, "not_found", "no such route")


# ---------------------------------------------------------------------------
# Startup helpers (one-shot; the request path never touches the filesystem)
# ---------------------------------------------------------------------------

_MANIFEST_LIMIT = 1 << 20  # <= 1 MiB, bounded read


def read_manifest_limited(path: str) -> tuple[bytes | None, str | None]:
    """One bounded, hang-proof read of the manifest file at startup: regular
    FD only (no FIFO open-block, no symlink follow), hard 1 MiB cap. Returns
    (bytes, None) or (None, typed reason code) — never raises."""
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
            # BOUNDED chunk loop to EOF (a single os.read may legitimately
            # return fewer bytes than requested); past the cap it is typed
            # too-large, never buffered further.
            chunks: list[bytes] = []
            total = 0
            while True:
                try:
                    chunk = os.read(fd, 1 << 16)
                except OSError:
                    return None, api.MANIFEST_UNREADABLE
                if not chunk:
                    break
                total += len(chunk)
                if total > _MANIFEST_LIMIT:
                    return None, "manifest_too_large"
                chunks.append(chunk)
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

    def _send(self, status: int, body: bytes, ctype: str,
              extra: tuple[tuple[str, str], ...] = ()) -> None:
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
        if self.command != "HEAD":  # pragma: no cover - HEAD is refused anyway
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
        if ok:
            # Keep the RAW wire target: the stdlib normalizes a leading '//'
            # in self.path before we would see it — routing must consider
            # both, so a real '//x' request is refused, never served.
            raw = self.raw_requestline
            if isinstance(raw, bytes):
                raw = raw.decode("iso-8859-1", "replace")
            parts = raw.strip().split(" ")
            self.raw_target = parts[1] if len(parts) >= 2 else ""
        return ok

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
                    self.requestline = ""
                    self.request = ""
                    self.close_connection = True
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
            # Route on the RAW wire target (fall back to self.path only if
            # the raw line could not be captured): the stdlib folds '//'
            # targets before self.path, so checking raw closes that gap.
            target = getattr(self, "raw_target", None) or self.path
            kind, arg = resolve_route(method, target)
            if kind == "static":
                asset, ctype = STATIC_ROUTES[arg]
                self._send(200, self.server.assets[asset], ctype)
            elif kind == "status":
                self._send_api(api.status_response(self.server.web_runtime))
            else:
                length = int(headers["content-length"][0])
                body = self.rfile.read(length)
                if len(body) != length:
                    self.send_error(400, "request body ended early")
                    return
                if kind == "plan":
                    self._send_api(api.plan_response(self.server.web_runtime, body))
                else:
                    self._send_api(api.export_response(self.server.web_runtime, body))
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
    """Close every connection still WAITING in the queue without handling
    it (graceful shutdown must not leave queued sockets dangling). Returns
    the number closed. Pure: takes a queue, needs no sockets of its own."""
    closed = 0
    while True:
        try:
            request, _client_address = conn_queue.get_nowait()
        except queue.Empty:
            return closed
        try:
            request.close()
        except OSError:  # pragma: no cover - already closed
            pass
        conn_queue.task_done()
        closed += 1


def stop_workers(conn_queue: "queue.Queue[Any]",
                 workers: list[threading.Thread],
                 put_timeout: float = 2.0,
                 join_timeout: float = 5.0) -> None:
    """Ask each started worker to exit (one sentinel per worker) and join
    with FINITE timeouts — never blocking forever. A worker mid-request is
    already bounded by the handler socket timeout; the threads are daemons
    as the last-resort backstop. Pure except for thread join/put bounds."""
    for _ in workers:
        try:
            conn_queue.put(_WORKER_SHUTDOWN, timeout=put_timeout)
        except Exception:  # full/broken queue: worker sees _stopping instead
            break
    for t in workers:
        t.join(join_timeout)


class _Server(HTTPServer):
    """Loopback server with a BOUNDED POOL of persistent worker threads —
    no per-connection thread creation, finite handler idle timeout via
    _Handler.timeout, typed 503 when saturated."""
    allow_reuse_address = False

    web_runtime: WebRuntime
    assets: dict[str, bytes]
    bound_ip: str
    bound_port: int

    def __init__(self, addr_info, handler) -> None:
        super().__init__(addr_info, handler)
        self._queue: "queue.Queue[Any]" = queue.Queue(
            maxsize=MAX_QUEUED_CONNECTIONS)
        self._workers: list[threading.Thread] = []
        self._pool_ready = False
        self._stopping = False

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
                item = self._queue.get()
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
        # Graceful shutdown must not leak its OWN pool: close the listening
        # socket, CLOSE every still-queued connection unhandled (never
        # dangling), then ask each worker to exit and join with a finite
        # timeout. Finite handler timeouts remain the only wait inside a
        # worker; the threads are daemons as the last-resort backstop.
        self._stopping = True
        super().server_close()
        drain_connection_queue(self._queue)
        stop_workers(self._queue, self._workers)
        self._workers = []


#: Prebuilt overload refusal — built ONCE from the shared _DEFENSIVE_HEADERS
#: constant so this raw path can NEVER drift away from the defensive header
#: set every other response carries (CSP, nosniff, frame-deny, no-referrer,
#: CORP, no-store, close). It must stay far below the kernel socket buffer:
#: one sendall, never reading the (possibly oversize) request.
def _raw_refusal_bytes(status_line: str, body: bytes) -> bytes:
    lines = [status_line.encode("ascii"),
             b"Content-Type: application/json; charset=utf-8"]
    lines += [f"{name}: {value}".encode("ascii")
              for name, value in _DEFENSIVE_HEADERS]
    lines.append(f"Content-Length: {len(body)}".encode("ascii"))
    return b"\r\n".join(lines) + b"\r\n\r\n" + body


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
            "Qwen Image Studio local web server (W2): serves the static "
            "workspace plus the versioned status/plan/config-export API. "
            "Planning only — generation is NOT implemented here. Loopback "
            "only. Omitted path options mean 'unconfigured': the server "
            "then writes nothing anywhere."
        ),
    )
    p.add_argument("--host", default=DEFAULT_HOST,
                   help="bind address: only the literal 127.0.0.1 is "
                        "accepted in W2 (refused before bind otherwise)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"TCP port (default {DEFAULT_PORT}; 0 picks an "
                        "ephemeral port and prints the actual origin)")
    p.add_argument("--manifest", default=None,
                   help="model manifest for metadata-only status summaries "
                        "(read once at startup, <=1 MiB, never verified here)")
    p.add_argument("--models-dir", default=None,
                   help="model storage root (metadata context only for W2)")
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
    if args.host not in ALLOWED_BIND_HOSTS:
        hint = ""
        if args.host == "::1":
            hint = (" (IPv6 loopback needs its own reviewed, actually-tested "
                    "bind milestone — W2 refuses it)")
        print(f"error: refusing to bind {args.host!r}: W2 binds the literal "
              f"127.0.0.1 only{hint}; non-loopback exposure needs a reviewed "
              "bind/auth policy milestone", file=sys.stderr)
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
    }
    status = _startup_config_or_exit(paths)
    if status:
        return status

    if paths["manifest_path"] is None:
        summary = api.summarize_manifest_bytes(None)
    else:
        raw, unreadable = read_manifest_limited(paths["manifest_path"])
        summary = api.summarize_manifest_bytes(raw, unreadable)

    runtime = WebRuntime(**paths, manifest_summary=summary)
    try:
        assets = load_static_assets()
    except OSError as exc:
        print(f"error: packaged static assets are not readable: {exc.strerror or exc}",
              file=sys.stderr)
        return 1

    try:
        httpd = _Server((args.host, args.port), _Handler)
    except OSError as exc:
        print(f"error: cannot bind http://{args.host}:{args.port}: "
              f"{exc.strerror or exc}", file=sys.stderr)
        return 1
    httpd.web_runtime = runtime
    httpd.assets = assets
    httpd.bound_ip = args.host
    httpd.bound_port = int(httpd.server_address[1])

    origin = f"http://{args.host}:{httpd.bound_port}"
    print(f"qwen-web listening on {origin}")
    print("qwen-web: planning only (no generation); loopback-only local "
          "server. Container loopback is NOT the host's or a Tailscale "
          "peer's loopback — remote access is a later reviewed milestone.")
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
