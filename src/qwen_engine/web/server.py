"""Explicit-address web server with opt-in serial engine jobs.

Safety posture (reviewed W2 contract):

* Defaults to literal ``127.0.0.1``. Repeated ``--host`` allows explicit
  IPv4 LAN/Tailscale listeners sharing one job queue. Wildcard, DNS and
  public-address binds are refused; each listener validates its own Host.
* Default port 8765 (the dsh harness port 3081 is unrelated and must never
  be used); ``--port 0`` picks an ephemeral port and the actual canonical
  origin is printed after bind.
* Exactly ONE ``Host`` header, validated against the bound literal + the
  actual port (plus the ``localhost`` alias for normal browser use, and any
  names the operator lists with ``--host-name`` — DNS rebinding attempts
  carry the attacker's own hostname and are refused). No DNS is performed
  for any of it.
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

from qwen_engine import local_bundle
from qwen_engine import model_setup as _setup
from qwen_engine import rewrite as rw
from qwen_engine import rewrite_client as rc
from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.web import api
from qwen_engine.web import rewrite_service as _rewrite_service
from qwen_engine.web.api import ApiError, ApiResponse, WebRuntime

DEFAULT_HOST = "127.0.0.1"
#: Default bind; additional explicit private IPv4 addresses are opt-in.
ALLOWED_BIND_HOSTS = (DEFAULT_HOST,)
#: Accepted Host header names at the bound port: the bound literal plus the
#: usual browser alias for loopback. Everything else (including a rebound
#: attacker hostname) is refused.
ALLOWED_HOST_NAMES = ("127.0.0.1", "localhost")
#: Operator-listed machine names accepted on every listener, from
#: ``<cache>/web-settings.json`` and/or ``--host-name``.
MAX_HOST_NAMES = 16
WEB_SETTINGS_FILENAME = "web-settings.json"
_WEB_SETTINGS_LIMIT = 16384
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
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

#: Bundled brand artwork: byte-for-byte copies of the agreed Lightseed assets
#: under web_static/brand/. Served through the SAME fixed allowlist as the text
#: assets — the names are compile-time constants, never request input.
BRAND_ASSETS = ("brand/lightseed-logo-on-dark.svg",
                "brand/lightseed-logo-on-light.svg",
                "brand/favicon.svg",
                "brand/favicon.ico",
                "brand/favicon-180.png")
STATIC_ASSETS = ("index.html", "app.css", "app.js") + BRAND_ASSETS
#: Binary brand files carry their exact image type and NO charset; the text
#: assets keep "charset=utf-8".
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/brand/lightseed-logo-on-dark.svg":
        ("brand/lightseed-logo-on-dark.svg", "image/svg+xml"),
    "/brand/lightseed-logo-on-light.svg":
        ("brand/lightseed-logo-on-light.svg", "image/svg+xml"),
    "/brand/favicon.svg": ("brand/favicon.svg", "image/svg+xml"),
    "/brand/favicon-180.png": ("brand/favicon-180.png", "image/png"),
    # Browsers request /favicon.ico at the root on their own, so the packaged
    # file is published under that exact route (the file itself stays in the
    # brand folder; one asset, one route).
    "/favicon.ico": ("brand/favicon.ico", "image/x-icon"),
}
STATUS_ROUTE = "/api/v1/status"
PLAN_ROUTE = "/api/v1/plan"
EXPORT_ROUTE = "/api/v1/config/export"
IMPORT_ROUTE = "/api/v1/config/import"
REWRITE_ROUTE = "/api/v1/rewrite"

#: The rewriter choice is SERVER-side: one page changes what every browser
#: using this server does next, and nothing here accepts a filesystem path.
#: ``/test`` probes the local service only — the enhancer cannot be probed
#: without loading 9.5 GiB onto the GPU.
SETTINGS_ROUTE = "/api/v1/settings/rewrite"
SETTINGS_TEST_ROUTE = "/api/v1/settings/rewrite/test"
JOBS_ROUTE = "/api/v1/jobs"
#: Clearing the terminal history is its OWN route: the per-job grammar is a
#: 32-hex identifier, so nothing else may ever resolve under /api/v1/jobs.
JOBS_FORGET_ALL_ROUTE = "/api/v1/jobs/forget-all"
_JOB_ROUTE = re.compile(r"/api/v1/jobs/([0-9a-f]{32})(?:/(image|cancel|forget))?\Z")
REFERENCES_ROUTE = "/api/v1/references"
#: Strict lowercase-content-id grammar; anything else never matches (404).
_REFERENCE_ROUTE = re.compile(r"/api/v1/references/([0-9a-f]{64})(/thumb)?\Z")

#: First-run model setup. Three routes, no ids in the URL: the catalogue is read
#: with GET and one download job is started or stopped with POST.
MODELS_ROUTE = api.MODELS_HREF
MODELS_DOWNLOAD_ROUTE = api.MODELS_DOWNLOAD_HREF
MODELS_CANCEL_ROUTE = api.MODELS_CANCEL_HREF

#: The one route that accepts raw bytes instead of JSON (design §2.4).
_OCTET_STREAM_CONTENT_TYPE_RE = re.compile(r"application/octet-stream\Z",
                                           re.IGNORECASE)

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


def valid_host_name(name: str) -> bool:
    """A lowercase DNS name the operator owns (``pc``, ``pc.local``,
    ``pc.tailnet.ts.net``). No wildcards, IP literals or ``localhost``: the
    listener literal and the loopback alias are already accepted, and a name
    whose last label is numeric could be read as an address."""
    if not name or len(name) > 253 or not name.isascii() or name != name.lower():
        return False
    labels = name.split(".")
    if not all(_HOST_LABEL.fullmatch(label) for label in labels):
        return False
    return not labels[-1].isdigit() and name != "localhost"


def read_web_settings(cache_dir: str | None) -> tuple[list[str], str | None]:
    """(host names, error) from ``<cache_dir>/web-settings.json``, read once at
    startup. A missing file (or no cache dir) means no names; anything else
    that is not ``{"host_names": [str, ...]}`` is an error, so a typo stops the
    server instead of silently locking the names out."""
    if cache_dir is None:
        return [], None
    path = os.path.join(cache_dir, WEB_SETTINGS_FILENAME)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return [], None
    except OSError as exc:
        return [], f"{path} is not readable: {exc.strerror or exc}"
    try:
        if not stat_module.S_ISREG(os.fstat(fd).st_mode):
            return [], f"{path} is not a regular file"
        raw = os.read(fd, _WEB_SETTINGS_LIMIT + 1)
    except OSError as exc:
        return [], f"{path} is not readable: {exc.strerror or exc}"
    finally:
        os.close(fd)
    if len(raw) > _WEB_SETTINGS_LIMIT:
        return [], f"{path} is larger than {_WEB_SETTINGS_LIMIT} bytes"
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return [], f"{path} is not valid JSON: {exc}"
    names = doc.get("host_names") if isinstance(doc, dict) else None
    if not isinstance(doc, dict) or set(doc) != {"host_names"} or not isinstance(names, list) \
            or not all(isinstance(name, str) for name in names):
        return [], f'{path} must be {{"host_names": ["name", ...]}} and nothing else'
    return names, None


def validate_host(values: list[str], bound_ip: str, port: int,
                  host_names: frozenset[str] = frozenset()) -> tuple[str | None, str | None]:
    """(expected_origin, error): exactly one Host header naming this listener
    literal (or localhost on 127.0.0.1, or an operator-listed ``host_names``
    entry) at the actual port; everything else — duplicates, userinfo,
    whitespace, non-ASCII, missing/foreign port, foreign name — is a typed
    error string."""
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
    names = {bound_ip} | host_names
    if bound_ip == DEFAULT_HOST:
        names.add("localhost")
    if name.lower() not in names:
        return None, "Host name must match this listener address, localhost on loopback, or a --host-name"
    return f"http://{name.lower()}:{port}", None


def header_policy(method: str, headers: dict[str, list[str]],
                  expected_origin: str, *, upload: bool = False,
                  max_body_bytes: int | None = None) -> ApiError | None:
    """Post-Host header policy, pure. ``headers`` maps lowercase names to
    their raw values in wire order. ``upload`` selects the reference-upload
    route's rules: raw ``application/octet-stream`` instead of JSON and the
    store's 32 MiB cap instead of the 64 KiB JSON cap. ``max_body_bytes``
    tightens the JSON cap for ONE route (the rewriter's derived limit); it
    never loosens it and it is ignored for uploads, whose cap belongs to the
    store."""
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
    expected_type = (_OCTET_STREAM_CONTENT_TYPE_RE if upload
                     else _JSON_CONTENT_TYPE_RE)
    if not types or not expected_type.match(types[0].strip()):
        return ApiError(415, "unsupported_media_type",
                        "reference uploads must be exactly "
                        "application/octet-stream (refused before the body "
                        "is read)" if upload else
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
    if upload:
        cap, digits = api.MAX_REFERENCE_UPLOAD_BYTES, 9
    elif max_body_bytes is not None:
        # One route (the rewriter) is capped TIGHTER than the generic JSON
        # limit; a caller can never raise it above MAX_BODY_BYTES.
        cap, digits = min(max_body_bytes, MAX_BODY_BYTES), 6
    else:
        cap, digits = MAX_BODY_BYTES, 6
    if len(raw) > digits or int(raw) > cap:
        return ApiError(413, "request_too_large",
                        f"request bodies are capped at {cap} bytes")
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
        if target == SETTINGS_ROUTE:
            return "settings_get", target
        if target == MODELS_ROUTE:
            return "models", target
        if target == JOBS_ROUTE:
            return "jobs", target
        match = _REFERENCE_ROUTE.fullmatch(target)
        if match:
            return ("reference_thumb" if match[2] else "reference"), match[1]
    elif method == "POST":
        if target == PLAN_ROUTE:
            return "plan", target
        if target == EXPORT_ROUTE:
            return "export", target
        if target == IMPORT_ROUTE:
            return "import", target
        if target == REWRITE_ROUTE:
            return "rewrite", target
        if target == SETTINGS_TEST_ROUTE:
            return "settings_test", target
        if target == SETTINGS_ROUTE:
            return "settings_set", target
        if target == MODELS_CANCEL_ROUTE:
            return "models_cancel", target
        if target == MODELS_DOWNLOAD_ROUTE:
            return "models_download", target
        if target == JOBS_ROUTE:
            return "submit", target
        if target == JOBS_FORGET_ALL_ROUTE:
            return "forget_all", target
        if target == REFERENCES_ROUTE:
            return "upload", REFERENCES_ROUTE
    ref_match = _REFERENCE_ROUTE.fullmatch(target)
    if ref_match:
        raise ApiError(405, "method_not_allowed", "this route does not accept that method",
                       headers=(("Allow", "GET"),))
    if target == REFERENCES_ROUTE:
        raise ApiError(405, "method_not_allowed", "this route does not accept that method",
                       headers=(("Allow", "POST"),))
    match = _JOB_ROUTE.fullmatch(target)
    if match:
        action = match[2] or "job"
        allow = "POST" if action in ("cancel", "forget") else "GET"
        if method == allow:
            return action, match[1]
        raise ApiError(405, "method_not_allowed", "this route does not accept that method",
                       headers=(("Allow", allow),))
    if target == JOBS_FORGET_ALL_ROUTE:
        raise ApiError(405, "method_not_allowed", "this route does not accept that method",
                       headers=(("Allow", "POST"),))
    if target == JOBS_ROUTE:
        raise ApiError(405, "method_not_allowed", "this route does not accept that method",
                       headers=(("Allow", "GET, POST"),))
    if target == MODELS_ROUTE:
        raise ApiError(405, "method_not_allowed",
                       "the model catalogue is read with GET",
                       headers=(("Allow", "GET"),))
    if target in (MODELS_DOWNLOAD_ROUTE, MODELS_CANCEL_ROUTE):
        raise ApiError(405, "method_not_allowed",
                       "a model download is started or stopped with POST",
                       headers=(("Allow", "POST"),))
    known = set(STATIC_ROUTES) | {STATUS_ROUTE, PLAN_ROUTE, EXPORT_ROUTE,
                                  IMPORT_ROUTE, REWRITE_ROUTE, SETTINGS_ROUTE,
                                  SETTINGS_TEST_ROUTE, MODELS_ROUTE,
                                  MODELS_DOWNLOAD_ROUTE, MODELS_CANCEL_ROUTE}
    if target in known:
        # The Settings page reads its state with GET and writes it with POST;
        # everything else here is one method only.
        allow = ("GET, POST" if target == SETTINGS_ROUTE else
                 "POST" if target in (PLAN_ROUTE, EXPORT_ROUTE, IMPORT_ROUTE,
                                      REWRITE_ROUTE, SETTINGS_TEST_ROUTE)
                 else "GET")
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
        # An asset name is the fixed, slash-separated path INSIDE web_static
        # (brand files live in a subfolder). It never comes from a request,
        # so no user-controlled path is ever resolved here.
        assets[name] = root.joinpath(*name.split("/")).read_bytes()
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
                                              self.server.bound_port,
                                              getattr(self.server, "host_names", frozenset()))
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
                                                  self.server.bound_port,
                                                  getattr(self.server, "host_names", frozenset()))
            if host_err is not None:
                err = ApiError(400, "invalid_host", host_err)
                self._send_api(ApiResponse(status=err.status, payload=err.to_payload()))
                return
            # Route on the RAW wire target ONLY (never the stdlib-normalized
            # self.path): the stdlib folds '//' in self.path before the app
            # sees it, and TAB request lines reach self.path normalized too.
            # An uncapturable/absent raw target is '' -> resolve_route refuses.
            target = getattr(self, "raw_target", "")
            # The upload route's content-type/length rules differ, so the
            # header policy needs to know the route BEFORE it is applied.
            # The rewrite route also carries a TIGHTER body cap than the
            # generic JSON limit (its body is one bounded prompt), chosen
            # from the same constant the browser enforces locally.
            err = header_policy(
                method, headers, host_origin,
                upload=method == "POST" and target == REFERENCES_ROUTE,
                max_body_bytes=(api.MAX_REWRITE_REQUEST_BYTES
                                if method == "POST" and target == REWRITE_ROUTE
                                else None))
            if err is not None:
                self._send_api(ApiResponse(status=err.status, payload=err.to_payload(),
                                           headers=err.headers))
                return
            kind, arg = resolve_route(method, target)
            if kind == "static":
                asset, ctype = STATIC_ROUTES[arg]
                self._send(200, self.server.assets[asset], ctype)
            elif kind == "status":
                self._send_api(api.status_response(self.server.web_runtime))
            elif kind == "settings_get":
                self._send_api(settings_get_response(
                    getattr(self.server, "rewrite", None)))
            elif kind == "models":
                self._send_api(models_get_response(
                    getattr(self.server, "setup", None), self.client_address))
            elif kind in ("reference", "reference_thumb"):
                data, ctype = api.reference_bytes(self.server.web_runtime, arg,
                                                  thumb=kind == "reference_thumb")
                self._send(200, data, ctype)
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
                if kind == "rewrite":
                    # The service exists on EVERY server (a disabled one holds
                    # no client), so this is only None when the handler was
                    # built without one — the same typed refusal either way.
                    service = getattr(self.server, "rewrite", None)
                    if service is None:
                        raise ApiError(503, "rewrite_disabled",
                                       "rewriting is not enabled on this server")
                    length = int(headers["content-length"][0])
                    body = self.rfile.read(length)
                    if len(body) != length:
                        self.send_error(400, "request body ended early")
                        return
                    # Envelope + semantics are validated by the wire contract
                    # BEFORE the service sees the prompt.
                    self._send_api(service.submit(api.build_rewrite_request(body)))
                    return
                length = int(headers["content-length"][0])
                body = self.rfile.read(length)
                if len(body) != length:
                    self.send_error(400, "request body ended early")
                    return
                if kind == "settings_set":
                    self._send_api(settings_post_response(
                        getattr(self.server, "rewrite", None), body))
                    return
                if kind == "settings_test":
                    self._send_api(settings_test_response(
                        getattr(self.server, "rewrite", None), body))
                    return
                if kind == "models_download":
                    self._send_api(models_download_response(
                        getattr(self.server, "setup", None), body,
                        self.client_address))
                    return
                if kind == "models_cancel":
                    self._send_api(models_cancel_response(
                        getattr(self.server, "setup", None), body,
                        self.client_address))
                    return
                if kind == "plan":
                    self._send_api(api.plan_response(self.server.web_runtime, body))
                elif kind == "export":
                    self._send_api(api.export_response(self.server.web_runtime, body))
                elif kind == "import":
                    self._send_api(api.import_response(self.server.web_runtime, body))
                else:
                    if kind == "upload":
                        self._send_api(api.upload_reference_response(
                            self.server.web_runtime, body))
                        return
                    jobs = getattr(self.server, "generation", None)
                    if jobs is None:
                        raise ApiError(503, "generation_disabled", "generation is disabled on this server")
                    if kind == "submit":
                        self._send_api(jobs.submit(body))
                    elif kind == "forget_all":
                        self._send_api(jobs.forget_all_response(body))
                    elif kind == "forget":
                        self._send_api(jobs.forget_response(arg, body))
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
            "Lightseed Studio local web server: serves the workspace, "
            "status/plan/config-export API and opt-in serial generation jobs. Planning "
            "is available by default. Explicit IPv4 loopback/LAN/Tailscale binds. "
            "Omitted path options mean 'unconfigured': the server "
            "then writes nothing anywhere."
        ),
    )
    p.add_argument("--host", action="append", default=None,
                   help="explicit IPv4 loopback, LAN or Tailscale address; repeat "
                        "to share one job queue across listeners (default: 127.0.0.1)")
    p.add_argument("--host-name", action="append", default=None, metavar="NAME",
                   help="a DNS name of this machine (e.g. pc.local or the Tailscale "
                        "MagicDNS name) that browsers may use in the address bar; "
                        "accepted on every listener, never resolved; repeatable. "
                        f"Added to the host_names list in <cache-dir>/{WEB_SETTINGS_FILENAME}")
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
    p.add_argument("--performance", action="store_true",
                   help="write opt-in prompt-free synchronized metrics in each job directory")
    p.add_argument("--attest-unbounded-host", action="store_true",
                   help="operator attestation of no cgroup RAM cap; encoder uses fresh host MemAvailable")
    p.add_argument("--reuse-verified-assets", type=float, nargs="?",
                   metavar="SECONDS",
                   const=local_bundle.VERIFICATION_REUSE_DEFAULT_MAX_AGE_S,
                   default=None,
                   help="opt-in, generation only: re-use a recorded SUCCESSFUL "
                        "asset verification younger than SECONDS (default "
                        f"{local_bundle.VERIFICATION_REUSE_DEFAULT_MAX_AGE_S:g}s) "
                        "while every artifact's stat fingerprint is unchanged "
                        "— weaker than re-hashing content; omitted (default): "
                        "every job hashes every artifact")
    # -- prompt rewriting (opt-in; off by default) --------------------------
    p.add_argument("--rewrite-backend", choices=("off",) + _rewrite_service.BACKENDS,
                   default="off",
                   help="the rewriter to START with when nothing is saved in "
                        "Settings (default off); the enhancer is the default "
                        "whenever its flags are given, and Settings then wins "
                        "over both")
    p.add_argument("--rewrite-endpoint", default=None,
                   help="OpenAI-compatible base URL for the rewriter "
                        "(http(s) only; no credentials are accepted here)")
    p.add_argument("--rewrite-model", default=None,
                   help="model name the rewriter backend serves")
    p.add_argument("--rewrite-prompt-t2i", default=None,
                   help="system prompt file for text-to-image rewriting "
                        "(digest-pinned; a mismatch refuses to start)")
    p.add_argument("--rewrite-prompt-edit", default=None,
                   help="system prompt file for image-edit rewriting "
                        "(digest-pinned; a mismatch refuses to start)")
    p.add_argument("--rewrite-timeout", type=float,
                   default=_rewrite_service.DEFAULT_TIMEOUT_SECONDS,
                   help="seconds a single rewrite may take before the request "
                        f"times out (default {_rewrite_service.DEFAULT_TIMEOUT_SECONDS:g})")
    # -- the official prompt enhancer: llama.cpp + the Q8_0 GGUFs -----------
    # These name FILES on this machine, so they exist only as launch flags: the
    # Settings page may CHOOSE the enhancer per draft kind but can never point it
    # at a binary or a model. The two enhancers are separate checkpoints —
    # PE-T2I for a text draft, PE-I2I plus its vision projector for an edit
    # draft — installed independently, so each pair is both-or-neither.
    p.add_argument("--pe-llama-server", default=None, metavar="PATH",
                   help="llama-server binary for the official prompt enhancer "
                        "(give it together with --pe-t2i-model; never settable "
                        "from the browser)")
    p.add_argument("--pe-t2i-model", default=None, metavar="PATH",
                   help="the text enhancer's Q8_0 GGUF (give it together with "
                        "--pe-llama-server)")
    p.add_argument("--pe-i2i-model", default=None, metavar="PATH",
                   help="the edit enhancer's Q8_0 GGUF (give it together with "
                        "--pe-i2i-mmproj; without both, an edit draft has no "
                        "official rewriter)")
    p.add_argument("--pe-i2i-mmproj", default=None, metavar="PATH",
                   help="the edit enhancer's vision projector (BF16 GGUF); a "
                        "Q8 GGUF alone cannot see the reference it is asked to "
                        "describe")
    p.add_argument("--pe-timeout", type=float,
                   default=_rewrite_service.DEFAULT_PE_TIMEOUT_SECONDS,
                   metavar="SECONDS",
                   help="seconds one enhancer rewrite may take, thinking "
                        "included (default "
                        f"{_rewrite_service.DEFAULT_PE_TIMEOUT_SECONDS:g}: the "
                        "thinking block alone can run to its token budget)")
    for flag in ("output-dir", "cache-dir", "log-dir", "benchmarks-dir",
                 "references-dir"):
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
def pe_flags_error(args: argparse.Namespace) -> str:
    """One bounded startup line when the enhancer flags are wrong, else ``""``.

    Each checkpoint is a pair and each pair is both-or-neither: the binary with
    the text GGUF, the edit GGUF with its vision projector. Files must be usable
    — binary readable AND executable, models readable. A missing file would
    otherwise surface as a mysterious backend failure forty seconds into the
    first rewrite; here it is one line and exit 2. The digest-pinned system
    prompt each checkpoint reads is checked where it is loaded
    (:func:`load_system_prompt`), because a drifted prompt is the same class of
    refusal — and PE-I2I needs the EDIT prompt, not the text one.
    """
    binary, model = args.pe_llama_server, args.pe_t2i_model
    i2i_model, i2i_mmproj = args.pe_i2i_model, args.pe_i2i_mmproj
    if binary is None and model is None and i2i_model is None and i2i_mmproj is None:
        return ""
    if (i2i_model is None) != (i2i_mmproj is None):
        return ("--pe-i2i-model and --pe-i2i-mmproj must be given together: the "
                "edit enhancer's GGUF cannot see the reference it is asked to "
                "describe without its vision projector")
    if binary is None or model is None:
        return ("--pe-llama-server and --pe-t2i-model must be given together: "
                "the enhancer needs its binary and its model")
    if not os.path.isfile(binary) or not os.access(binary, os.R_OK | os.X_OK):
        return f"--pe-llama-server is not a readable executable file: {binary}"
    if not os.path.isfile(model) or not os.access(model, os.R_OK):
        return f"--pe-t2i-model is not a readable file: {model}"
    if args.rewrite_prompt_t2i is None:
        return ("the enhancer needs --rewrite-prompt-t2i: the digest-pinned "
                "system prompt its weights were trained on")
    if i2i_model is not None:
        for flag, value in (("--pe-i2i-model", i2i_model),
                            ("--pe-i2i-mmproj", i2i_mmproj)):
            if not os.path.isfile(value) or not os.access(value, os.R_OK):
                return f"{flag} is not a readable file: {value}"
        if args.rewrite_prompt_edit is None:
            return ("the edit enhancer needs --rewrite-prompt-edit: the "
                    "digest-pinned system prompt PE-I2I was trained on")
    return ""


def build_rewrite_service(args: argparse.Namespace,
                          references_dir: str | None = None,
                          *, cache_dir: str | None = None,
                          gpu_lock: threading.Lock | None = None,
                          ) -> _rewrite_service.RewriteService:
    """Construct the rewriter from the CLI flags, or refuse at startup.

    Operator errors are startup errors: a bad endpoint, a missing prompt or a
    digest mismatch prints ONE bounded line to stderr and exits 2 — the server
    never starts half-configured, and the browser never learns any of it (its
    status block is a closed set of keys that carries no location).

    The flags say what is POSSIBLE; the saved Settings document (applied after
    this by :func:`load_saved_preferences`) says which possibility is ACTIVE.
    With nothing saved the enhancer wins when it exists, because it is the
    standard rewriter. ``references_dir`` is the store edit rewrites read their
    images from; ``cache_dir`` holds the saved settings and the enhancer's
    bounded log file; ``gpu_lock`` is the SAME lock the generation executor
    takes around a job.
    """
    prompts: dict[str, str] = {}
    for task in rw.TASKS:
        flag = getattr(args, f"rewrite_prompt_{task}")
        if flag is None:
            continue
        # Digest-pinned: a drifted prompt refuses here, never at request time.
        prompts[task] = rw.load_system_prompt(flag, task)
    # The local service is configured by its OWN flags, whatever the default
    # backend is: a server that starts on the enhancer still needs it for edit
    # drafts, and Settings still needs it to have something to switch to.
    complete = (args.rewrite_endpoint is not None and args.rewrite_model is not None
                and len(prompts) == len(rw.TASKS))
    if args.rewrite_backend == "local-llm" and not complete:
        missing = [f"--rewrite-prompt-{task}" for task in rw.TASKS
                   if task not in prompts]
        if args.rewrite_endpoint is None:
            missing.append("--rewrite-endpoint")
        if args.rewrite_model is None:
            missing.append("--rewrite-model")
        raise rc.RewriteConfigError(
            f"--rewrite-backend local-llm needs {', '.join(missing)}")
    local = (_rewrite_service.RewriteSettings(
        backend="local-llm", endpoint=args.rewrite_endpoint,
        model=args.rewrite_model, prompts=prompts,
        timeout=args.rewrite_timeout, references_dir=references_dir)
        if complete else _rewrite_service.RewriteSettings.disabled())
    enhancer = None
    if args.pe_llama_server is not None and args.pe_t2i_model is not None:
        # One PESettings, both checkpoints: `tasks` reports what is actually
        # installed here, and the service builds one enhancer client per task it
        # can serve. The edit pair is optional — without it an edit draft simply
        # has no official rewriter, and Settings says so.
        enhancer = _rewrite_service.PESettings(
            llama_server=args.pe_llama_server, model_path=args.pe_t2i_model,
            system_prompts=prompts, timeout=args.pe_timeout,
            i2i_model_path=args.pe_i2i_model,
            i2i_mmproj_path=args.pe_i2i_mmproj)
    if args.rewrite_backend == "pe-llamacpp" and enhancer is None:
        raise rc.RewriteConfigError(
            "--rewrite-backend pe-llamacpp needs --pe-llama-server and "
            "--pe-t2i-model")
    return _rewrite_service.RewriteService(local, pe=enhancer,
                                           gpu_lock=gpu_lock,
                                           cache_dir=cache_dir)


def load_saved_preferences(service: _rewrite_service.RewriteService,
                           cache_dir: str | None) -> bool:
    """Apply the saved Settings document over the flag defaults.

    A corrupt or invalid file is ONE bounded stderr line and the flag defaults:
    a rewriter that cannot read its own preferences still starts, and the
    operator hears about it at start-up rather than in a browser message. The
    document is never rewritten here (``persist=False``) — loading must not
    have side effects.
    """
    prefs, warning = _rewrite_service.load_preferences(
        _rewrite_service.preferences_path(cache_dir))
    if warning:
        print(f"rewrite settings: {warning}", file=sys.stderr)
    if prefs is None:
        return False
    service.apply_preferences(prefs, persist=False)
    return True


def settings_get_response(service: _rewrite_service.RewriteService | None) -> ApiResponse:
    """``GET /api/v1/settings/rewrite``. The choice is readable even when the
    rewriter is off — that is exactly when the page has to be able to turn it
    on — and the payload is the validated state, which carries no path."""
    if service is None:
        raise ApiError(503, "settings_unavailable",
                       "this server has no rewriter settings to read")
    return ApiResponse(200, service.settings_state())


def settings_post_response(service: _rewrite_service.RewriteService | None,
                           body: bytes) -> ApiResponse:
    """``POST /api/v1/settings/rewrite``: validate, refuse the impossible,
    then adopt and persist, and answer with the state the NEXT rewrite uses.

    A change that cannot be written to the cache dir is rolled back in memory
    and reported: the page must never show a setting the file does not hold,
    because the next start-up would quietly disagree with it.
    """
    if service is None:
        raise ApiError(503, "settings_unavailable",
                       "this server has no rewriter settings to change")
    fields = api.build_rewrite_settings_request(body)
    prefs = _rewrite_service.RewritePreferences(
        choices={task: fields[task] for task in rw.TASKS},
        enabled=fields["enabled"], endpoint=fields["endpoint"],
        model=fields["model"], timeout_s=fields["timeout_s"])
    service.check_preferences(prefs)
    previous = service.preferences()
    try:
        service.apply_preferences(prefs)
    except OSError:
        # ``apply_preferences`` already put the previous choice back, so there
        # is nothing to undo here beyond saying so: a change the cache dir
        # refused to hold must not be shown as saved.
        raise ApiError(503, "settings_unavailable",
                       "the rewriter settings could not be written in the "
                       "server cache directory; nothing was changed") from None
    return ApiResponse(200, service.settings_state())


def settings_test_response(service: _rewrite_service.RewriteService | None,
                           body: bytes) -> ApiResponse:
    """``POST /api/v1/settings/rewrite/test``: one tiny completion, in plain
    words. It never touches the GPU and never takes the rewrite slot."""
    if service is None:
        raise ApiError(503, "settings_unavailable",
                       "this server has no rewriter settings to test")
    fields = api.build_rewrite_test_request(body)
    try:
        result = service.test_connection(fields["endpoint"], fields["model"],
                                         fields["timeout_s"])
    except rc.RewriteConfigError as exc:
        raise ApiError(422, "endpoint_unsupported", str(exc)) from None
    return ApiResponse(200, result)


# ---------------------------------------------------------------------------
# First-run model setup: the three routes
# ---------------------------------------------------------------------------


def models_get_response(service: Any, client_address: Any) -> ApiResponse:
    """``GET /api/v1/models``: the catalogue, the sizes and the one job.

    Every caller may READ it, including a Tailscale address: seeing what is
    missing is not a privilege. ``can_download`` carries the difference, so the
    page can explain why its buttons are not there.
    """
    if service is None:
        raise ApiError(503, "models_unavailable",
                       "this server cannot read its model catalogue")
    return ApiResponse(200, api.validate_models_state(
        {**service.state(),
         "can_download": _setup.client_is_loopback(client_address),
         "server_command": api.SETUP_SERVER_COMMAND,
         "notices": service.notices()}))


def _setup_write_allowed(client_address: Any) -> None:
    """Downloads and cancels are for the person standing at the machine."""
    if not _setup.client_is_loopback(client_address):
        raise ApiError(403, "setup_server_only",
                       "model downloads are started on the machine that runs "
                       "this server; run "
                       f"{api.SETUP_SERVER_COMMAND} there")


def models_download_response(service: Any, body: bytes,
                             client_address: Any) -> ApiResponse:
    """``POST /api/v1/models/download``: ids in, the running job out.

    Every refusal — an unknown id, too little free space, a job already
    running, no fetch tool — happens before a thread exists, so a refused
    request leaves nothing half-downloaded.
    """
    if service is None:
        raise ApiError(503, "models_unavailable",
                       "this server cannot download models")
    _setup_write_allowed(client_address)
    request = api.build_models_download_request(body)
    try:
        return ApiResponse(202, service.start(request["ids"]))
    except _setup.UnknownItem as exc:
        raise ApiError(422, "unknown_model", str(exc),
                       (("ids", "this server does not offer that model"),)) from None
    except _setup.NotEnoughSpace as exc:
        raise ApiError(507, "insufficient_space", str(exc),
                       (("disk", "free space is needed for the download and "
                                 "for the temp file it resumes from"),)) from None
    except _setup.DownloadBusy as exc:
        raise ApiError(409, "setup_busy", str(exc)) from None
    except _setup.FetcherUnavailable as exc:
        raise ApiError(503, "models_unavailable", str(exc)) from None
    except _setup.SetupError as exc:
        raise ApiError(500, "models_unavailable", str(exc)) from None


def models_cancel_response(service: Any, body: bytes,
                           client_address: Any) -> ApiResponse:
    """``POST /api/v1/models/download/cancel``: stop, keep the temp file."""
    if service is None:
        raise ApiError(503, "models_unavailable",
                       "this server has no model download to cancel")
    _setup_write_allowed(client_address)
    if api.parse_request_json(body) != {}:
        raise ApiError(400, "invalid_envelope",
                       "cancelling a model download takes an empty JSON object")
    return ApiResponse(200, service.cancel())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hosts = list(dict.fromkeys(args.host or [DEFAULT_HOST]))
    if len(hosts) > 8 or any(not valid_bind_host(host) for host in hosts):
        print("error: refusing to bind: use at most eight explicit IPv4 loopback, "
              "RFC1918 LAN or Tailscale addresses; wildcards, DNS names, public "
              "addresses and IPv6 are unsupported", file=sys.stderr)
        return 2
    saved_names, settings_err = read_web_settings(args.cache_dir)
    if settings_err is not None:
        print(f"error: {settings_err}", file=sys.stderr)
        return 2
    host_names = frozenset(name.lower().rstrip(".")
                           for name in saved_names + (args.host_name or []))
    if len(host_names) > MAX_HOST_NAMES or any(not valid_host_name(name) for name in host_names):
        print(f"error: host names ({WEB_SETTINGS_FILENAME} or --host-name) must be at most "
              f"{MAX_HOST_NAMES} plain DNS names (letters, digits, hyphens and dots; "
              "no wildcards, IP addresses or localhost)", file=sys.stderr)
        return 2
    if not 0 <= args.port <= 65535:
        print("error: --port must be in [0, 65535]", file=sys.stderr)
        return 2
    reuse_policy = None
    if args.reuse_verified_assets is not None:
        # Typed startup refusal before any bind; the policy itself owns
        # what a valid age is (positive, finite seconds).
        try:
            reuse_policy = local_bundle.VerificationReusePolicy(
                max_age_s=args.reuse_verified_assets)
        except ValueError as exc:
            print(f"error: --reuse-verified-assets: {exc}", file=sys.stderr)
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
    if args.references_dir is not None:
        paths["references_dir"] = os.path.realpath(args.references_dir)
    elif paths["cache_dir"] is not None:
        paths["references_dir"] = os.path.join(paths["cache_dir"], "references")
    else:
        paths["references_dir"] = None
    if paths["references_dir"] is not None:
        # Server-owned store: created 0700 at startup (uploads are the only
        # writes; the browser never sees this path).
        try:
            os.makedirs(paths["references_dir"], mode=0o700, exist_ok=True)
        except OSError:
            print("error: the server reference store directory cannot be created",
                  file=sys.stderr)
            return 1
    status = _startup_config_or_exit(paths)
    if status:
        return status

    pe_error = pe_flags_error(args)
    if pe_error:
        # One bounded line, before any bind: the enhancer is either fully
        # configured by the operator or not offered at all.
        print(f"error: {pe_error}", file=sys.stderr)
        return 2

    # ONE lock for the ONE V100, handed to the rewriter and to the generation
    # executor: 9.5 GiB of enhancer plus 13-15 GiB of image model does not fit
    # in 16 GiB, so whichever did not ask first must not simply load.
    gpu_lock = threading.Lock()
    try:
        rewrite = build_rewrite_service(args, paths["references_dir"],
                                       cache_dir=paths["cache_dir"],
                                       gpu_lock=gpu_lock)
        saved_warning = load_saved_preferences(rewrite, paths["cache_dir"])
    except (rc.RewriteConfigError, rw.RewriteError) as exc:
        # Bounded operator diagnostic: the message may name a path, which is
        # fine on a local console and MUST NOT reach the browser (it does not:
        # the status block is a closed set of keys and the 503 is a fixed
        # string).
        print(f"rewrite error: {exc}", file=sys.stderr)
        return 2
    if saved_warning:
        # The saved document is the user's own Settings; a broken one falls
        # back to the flags rather than taking the server down.
        print(f"rewrite settings: {saved_warning}", file=sys.stderr)
    if paths["manifest_path"] is None:
        summary = api.summarize_manifest_bytes(None)
    else:
        raw, unreadable = read_manifest_limited(paths["manifest_path"])
        summary = api.summarize_manifest_bytes(raw, unreadable)

    # First-run model setup: the catalogue comes from the pinned manifests this
    # checkout carries and the directory the operator pointed at. Building it is
    # never fatal — a server that cannot read its catalogue still serves pages,
    # and the Settings notice says so.
    setup_service = _setup.ModelSetupService.for_paths(
        models_dir=paths["models_dir"], manifest_path=paths["manifest_path"])

    runtime = WebRuntime(**paths, manifest_summary=summary,
                         generation_enabled=args.enable_generation,
                         rewrite_block=rewrite.status_block(),
                         setup_service=setup_service)
    if args.enable_generation and api.startup_blockers(runtime):
        # Missing weights are NOT a startup error: the page is how they get
        # here, and WebGeneration applies the same rule via the same helper.
        # Every other blocker still refuses to start generation.
        print("error: generation requires valid manifest metadata and explicit "
              "--manifest, --models-dir, --output-dir and --cache-dir",
              file=sys.stderr)
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
            httpd.rewrite = rewrite
            httpd.bound_ip = host
            httpd.bound_port = port
            httpd.host_names = host_names
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
            httpd.generation = WebGeneration(runtime, trust_unbounded_host=args.attest_unbounded_host,
                                            performance=args.performance,
                                            reuse_verified_assets=reuse_policy,
                                            gpu_lock=gpu_lock)
        except (OSError, ValueError, RuntimeError):
            for listener in listeners:
                listener.server_close()
            print("error: generation output storage or worker could not be initialized", file=sys.stderr)
            return 1
        if args.attest_unbounded_host:
            print("notice: operator attests no cgroup RAM limit for generation; "
                  "the encoder still checks fresh host MemAvailable", file=sys.stderr)
        if reuse_policy is not None:
            print("notice: asset-verification reuse is ON: a successful asset "
                  f"verification up to {reuse_policy.max_age_s:g}s old may be "
                  "reused while every artifact's stat fingerprint is unchanged "
                  "— a weaker guarantee than hashing every artifact per job")

    for listener in listeners:
        listener.generation = httpd.generation
        listener.rewrite = rewrite
        listener.setup = setup_service
        print(f"qwen-web listening on http://{listener.bound_ip}:{port}")
    for name in sorted(host_names):
        print(f"qwen-web also answers as http://{name}:{port}")
    mode = "generation enabled (serial jobs)" if httpd.generation else "planning only (no generation)"
    if rewrite.available:
        # Which rewriter answers EACH draft kind, and which ones could have: the
        # operator sees "edit fell back to the endpoint because PE-I2I is not
        # installed here" on the console, without reading the settings file.
        answering = ", ".join(f"{task}={backend}"
                              for task, backend in rewrite.active_by_task().items()
                              if backend)
        mode += (f"; prompt rewriting enabled ({answering}"
                 f" of {', '.join(rewrite.available_backends())})")
    else:
        mode += "; prompt rewriting off"
    print(f"qwen-web: {mode}; explicit address binds with per-listener "
          "Host and same-origin request checks.")
    if setup_service.notices():
        # A catalogue this checkout cannot build is a packaging problem, and the
        # console says so as plainly as the Settings page does.
        print("model setup: " + "; ".join(setup_service.notices()),
              file=sys.stderr)
    if setup_service.required_missing():
        # The one line a fresh machine needs: what is missing, how big, and the
        # two ways to get it. Paths stay off the wire; a console may name them.
        # The size is the REQUIRED files' own size — quoting the whole
        # catalogue's total beside one label would overstate the price of the
        # thing the sentence names.
        state = setup_service.state()
        wanted = [row for row in state["items"]
                  if row["required"] and row["installed"] != _setup.INSTALLED]
        labels = " and ".join(str(row["label"]) for row in wanted) or \
            "the base image model (this checkout's catalogue does not list it)"
        needed = sum(int(row["bytes_needed"]) for row in wanted)
        print(f"model setup: missing {labels}"
              f" ({needed / 2**30:.1f} GiB) — open the Settings page on this "
              f"machine to download them, or run "
              f"{api.SETUP_SERVER_COMMAND}", file=sys.stderr)
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
        # Bounded: a blocked backend socket is abandoned, never waited on.
        rewrite.close()
        if httpd.generation is not None:
            stopped = httpd.generation.close()
        signal.signal(signal.SIGTERM, old_term)
    if not stopped:
        print("error: generation worker did not finish shutdown within 60 seconds", file=sys.stderr)
    return 0 if stopped else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
