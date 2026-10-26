"""Guarded W2 tests: pure transport policy + entrypoint/packaging checks.

No sockets are ever created here (the guarded runner denies them anyway);
the real-socket smoke checks live in the SEPARATE integration command
``scripts/web_integration_check.mjs``. These tests exercise the pure policy
functions that the handler delegates to, plus the CLI contract.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from qwen_engine.web import api
from qwen_engine.web import server

PORT = 8765


def h(**name_to_values: list[str]) -> dict[str, list[str]]:
    out = {k.replace("_", "-"): v for k, v in name_to_values.items()}
    # POST-shaped fixtures (carrying the UI header) always carry the exact
    # JSON content type unless a test overrides it explicitly.
    if "x-qwen-ui" in out and "content-type" not in out:
        out["content-type"] = ["application/json"]
    return out


# ---------------------------------------------------------------- Host


def test_host_exact_single_and_port_bound():
    assert server.validate_host(["127.0.0.1:8765"], "127.0.0.1", 8765) \
        == ("http://127.0.0.1:8765", None)
    assert server.validate_host(["localhost:8765"], "127.0.0.1", 8765) \
        == ("http://localhost:8765", None)
    for bad in ([], ["a", "b"], ["127.0.0.1"], ["127.0.0.1:3081"],
                ["evil.example:8765"], ["127.0.0.1: 8765"], ["127.0.0.1:8765 "],
                ["user@127.0.0.1:8765"], ["127.0.0.1:+8765"], ["ünicode:8765"],
                ["127.0.0.1:99999"], [":8765"], ["127.0.0.1:08765"]):
        origin, err = server.validate_host(bad, "127.0.0.1", PORT)
        assert origin is None and err, bad


def test_default_port_is_8765_never_the_harness_port():
    assert server.DEFAULT_PORT == 8765
    assert server.DEFAULT_PORT != 3081
    assert server.DEFAULT_HOST == "127.0.0.1"
    assert server.ALLOWED_BIND_HOSTS == ("127.0.0.1",)


# ---------------------------------------------------------- other headers


def test_origin_and_sec_fetch_denied_on_every_route():
    ok_origin = "http://127.0.0.1:8765"
    for method in ("GET", "POST"):
        good = (h(origin=[ok_origin]) if method == "GET" else
                h(origin=[ok_origin], x_qwen_ui=["1"], content_length=["5"]))
        assert server.header_policy(method, good, ok_origin) is None
        # Origin/site checks come FIRST on every route, before write policy:
        err = server.header_policy(method, h(origin=["http://evil.example"]), ok_origin)
        assert err is not None and err.status == 403
        err = server.header_policy(method, h(origin=["null"]), ok_origin)
        assert err is not None and err.status == 403
        for site in ("cross-site", "cross-origin", "same-site", "different-origin"):
            err = server.header_policy(method, h(sec_fetch_site=[site]), ok_origin)
            assert err is not None and err.status == 403, site
        for site in ("none", "same-origin"):
            good_site = dict(good)
            good_site["sec-fetch-site"] = [site]
            assert server.header_policy(method, good_site, ok_origin) is None


def test_duplicate_policy_headers_refused():
    ok_origin = "http://127.0.0.1:8765"
    body_headers = h(origin=[ok_origin, ok_origin])
    assert server.header_policy("GET", body_headers, ok_origin).status == 400
    post = h(origin=[ok_origin], x_qwen_ui=["1", "1"],
             content_length=["10"])
    assert server.header_policy("POST", post, ok_origin).status == 400


def test_post_requires_exact_ui_header_and_plain_length():
    ok_origin = "http://127.0.0.1:8765"
    base = h(origin=[ok_origin])
    assert server.header_policy("POST", base, ok_origin).code == "ui_header_required"
    good = h(origin=[ok_origin], x_qwen_ui=["1"], content_length=["5"])
    assert server.header_policy("POST", good, ok_origin) is None
    assert server.header_policy("POST", h(origin=[ok_origin], x_qwen_ui=["1"],
                                          content_length=[]), ok_origin).code \
        == "length_required"
    for bad in ["+5", " 5", "5.0", "0x5", "-5", "", "05", "1e3", "5, 6"]:
        err = server.header_policy("POST", h(origin=[ok_origin], x_qwen_ui=["1"],
                                             content_length=[bad]), ok_origin)
        assert err is not None and err.code == "invalid_content_length", bad
    huge = h(origin=[ok_origin], x_qwen_ui=["1"],
             content_length=[str(server.MAX_BODY_BYTES + 1)])
    assert server.header_policy("POST", huge, ok_origin).status == 413


def test_transfer_encoding_and_expect_policy():
    ok_origin = "http://127.0.0.1:8765"
    post = h(origin=[ok_origin], x_qwen_ui=["1"], content_length=["5"])
    te = dict(post); te["transfer-encoding"] = ["chunked"]
    assert server.header_policy("POST", te, ok_origin).code == "transfer_encoding_unsupported"
    ex = dict(post); ex["expect"] = ["PGFQ 10"]
    assert server.header_policy("POST", ex, ok_origin).status == 417
    # Closed posture: NO Expect extension at all, including 100-continue —
    # the server never emits an early 100 before Host/Origin/length checks.
    for expect_value in ("100-Continue", "100-continue"):
        ex = dict(post); ex["expect"] = [expect_value]
        err = server.header_policy("POST", ex, ok_origin)
        assert err is not None and err.status == 417, expect_value


def test_content_type_exact_json_enforced_before_body():
    ok_origin = "http://127.0.0.1:8765"
    base = h(origin=[ok_origin], x_qwen_ui=["1"], content_length=["5"])
    good = dict(base); good["content-type"] = ["application/json"]
    assert server.header_policy("POST", good, ok_origin) is None
    good_cs = dict(base)
    good_cs["content-type"] = ["application/json; charset=UTF-8"]
    assert server.header_policy("POST", good_cs, ok_origin) is None
    for bad in ("text/plain;charset=UTF-8",
                "application/x-www-form-urlencoded",
                "application/json; charset=iso-8859-1",
                "application/jsonx", "application/json;charset=utf8", ""):
        wrong = dict(base); wrong["content-type"] = [bad]
        err = server.header_policy("POST", wrong, ok_origin)
        assert err is not None and err.status == 415, bad
    missing = dict(base); missing.pop("content-type")
    assert server.header_policy("POST", missing, ok_origin).status == 415
    dup = dict(base); dup["content-type"] = ["application/json", "application/json"]
    assert server.header_policy("POST", dup, ok_origin).status == 400


def test_long_digit_headers_are_typed_not_500():
    # Arbitrary-length digit strings never reach int() unguarded.
    _, err = server.validate_host(["127.0.0.1:" + "9" * 10000], "127.0.0.1", 8765)
    assert err is not None
    ok_origin = "http://127.0.0.1:8765"
    headers = h(origin=[ok_origin], x_qwen_ui=["1"], content_length=["9" * 10000],
                content_type=["application/json"])
    err = server.header_policy("POST", headers, ok_origin)
    assert err is not None and err.status in (400, 413)


def test_overload_response_is_well_formed():
    head, _, body = server._OVERLOAD_BYTES.partition(b"\r\n\r\n")
    declared = [int(line.split(b":")[1])
                for line in head.split(b"\r\n") if line.lower().startswith(b"content-length")]
    assert declared == [len(body)]
    assert body.endswith(b"\n") and b"overloaded" in body
    assert head.startswith(b"HTTP/1.1 503") and b"access-control" not in head.lower()


def test_overload_refusal_carries_the_FULL_defensive_header_set():
    # The raw sendall path must NOT drift from the header set every other
    # response carries: reuse of the shared constant is asserted, not just
    # some fixed headers.
    head = server._OVERLOAD_BYTES.partition(b"\r\n\r\n")[0]
    for name, value in server._DEFENSIVE_HEADERS:
        assert f"{name}: {value}".encode("ascii") in head, name
    assert b"Content-Security-Policy" in head and b"Referrer-Policy" in head
    assert b"Cross-Origin-Resource-Policy: same-origin" in head
    assert b"X-Frame-Options: DENY" in head
    assert not any(name.lower().startswith(b"access-control")
                   for name in (line.split(b":")[0] for line in head.split(b"\r\n")))
    # built FROM the constant (one source of truth), not a second literal
    # list. The legacy/early-error framing fix consolidated raw framing
    # into _framed_bytes: _raw_refusal_bytes must DELEGATE to it (no
    # second header list anywhere) and that one builder must use the
    # shared constant.
    import inspect
    raw_src = inspect.getsource(server._raw_refusal_bytes)
    builder_src = inspect.getsource(server._framed_bytes)
    assert "_framed_bytes" in raw_src  # delegation, never a second copy
    assert "_DEFENSIVE_HEADERS" in builder_src
    for name, _ in server._DEFENSIVE_HEADERS:
        # no defensive header NAME may appear as a literal outside the
        # shared constant's own definition (drift guard for both builders)
        assert f'"{name}"' not in raw_src and f'"{name}"' not in builder_src


class _FakeSocket:
    """Records closes only — never a real socket (the guard denies those)."""

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_drain_connection_queue_closes_every_queued_socket():
    import queue as _queue

    q: _queue.Queue = _queue.Queue()
    sockets = [_FakeSocket() for _ in range(5)]
    for s in sockets:
        q.put((s, ("127.0.0.1", 1234)))
    assert server.drain_connection_queue(q) == 5
    assert all(s.closed == 1 for s in sockets)
    assert q.empty()
    assert q.unfinished_tasks == 0          # every item was task_done'd
    # idempotent on an already-empty queue
    assert server.drain_connection_queue(q) == 0


def test_stop_workers_terminates_persistent_workers_in_finite_time():
    import queue as _queue
    import threading
    import time

    q: _queue.Queue = _queue.Queue()
    done = _queue.Queue()

    def worker():
        while True:
            item = q.get()
            if item is server._WORKER_SHUTDOWN:
                q.task_done()
                done.put("exit")
                return
            q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    t0 = time.monotonic()
    server.stop_workers(q, threads)
    assert time.monotonic() - t0 < 5.0, "shutdown must be finite"
    assert all(not t.is_alive() for t in threads), "workers must have exited"
    assert done.qsize() == 3
    # a second stop on dead workers must not raise or block forever
    server.stop_workers(q, threads)


def test_server_close_owns_its_worker_pool_lifecycle():
    # server_close must drain the queue and stop the pool (no live worker
    # leak, no dangling queued sockets) with finite bounds — verified via
    # the source wiring and the pure helpers, no sockets here.
    import inspect

    src = inspect.getsource(server._Server.server_close)
    assert "drain_connection_queue(self._queue)" in src
    assert "stop_workers(self._queue, self._workers)" in src
    assert "_stopping = True" in src
    close_src = inspect.getsource(server._Server.process_request)
    assert "_stopping" in close_src, "connections after stop must not queue"


def test_idle_worker_stops_without_sentinel_after_queue_put_failure(monkeypatch):
    import queue
    from types import SimpleNamespace

    timeouts = []

    class EmptyQueue:
        def get(self, *, timeout):
            timeouts.append(timeout)
            raise queue.Empty

    owner = SimpleNamespace(_queue=EmptyQueue(), _stopping=True)
    server._Server._worker(owner)
    assert timeouts == [server.WORKER_QUEUE_POLL_SECONDS]


@pytest.mark.parametrize("port", ["9" * 10000, "65536", "01", "٥", "", "123456"])
def test_auxiliary_port_check_bounds_conversion_and_refuses_invalid(port):
    assert server._port_ok(port) is False


def test_every_method_obeyes_common_policy_before_typing_the_rejection():
    # The catch-all routes ANY method through header_policy first: an
    # unsupported method with a foreign Origin is 403, not a 501 page.
    ok_origin = "http://127.0.0.1:8765"
    for method in ("OPTIONS", "CONNECT", "TRACE", "FOOBAR"):
        err = server.header_policy(method, h(origin=["http://evil.example"]),
                                   ok_origin)
        assert err is not None and err.status == 403, method
        assert server.header_policy(method, h(origin=[ok_origin]),
                                    ok_origin) is None, method
        # known routes answer 405 with an Allow header; unknown 404 — typed
        # JSON either way, never the stock 501 HTML page
        with pytest.raises(api.ApiError) as exc:
            server.resolve_route(method, "/api/v1/plan")
        assert exc.value.status == 405, method
        assert exc.value.headers == (("Allow", "POST"),), method
    import inspect
    handler_src = inspect.getsource(server._Handler.handle_one_request)
    assert "_dispatch(self.command)" in handler_src, \
        "methods without a do_* dispatcher must hit the policy path"


# ---------------------------------------------------------------- routes


def test_routes_exact_origin_form_only():
    assert server.resolve_route("GET", "/") == ("static", "/")
    assert server.resolve_route("GET", "/app.js") == ("static", "/app.js")
    assert server.resolve_route("GET", "/api/v1/status") == ("status", "/api/v1/status")
    assert server.resolve_route("POST", "/api/v1/plan") == ("plan", "/api/v1/plan")
    assert server.resolve_route("POST", "/api/v1/config/export") == \
        ("export", "/api/v1/config/export")
    # traversal / absolute / query / fragment / encoded variants: refused,
    # and NEVER translated to a filesystem path.
    with pytest.raises(api.ApiError) as exc:
        server.resolve_route("GET", "/../pyproject.toml")
    assert exc.value.status == 404
    for bad in ("http://127.0.0.1:8765/app.js", "//app.js", "/app.js?v=1",
                "/app.js#x", "/app%2Ejs", "/%2e%2e/app.js", "*"):
        with pytest.raises(api.ApiError) as exc:
            server.resolve_route("GET", bad)
        assert exc.value.status == 400, bad
    with pytest.raises(api.ApiError) as exc:
        server.resolve_route("POST", "/api/v1/status")
    assert exc.value.status == 405
    with pytest.raises(api.ApiError) as exc:
        server.resolve_route("GET", "/api/v1/plan")
    assert exc.value.status == 405
    with pytest.raises(api.ApiError) as exc:
        server.resolve_route("GET", "/nope")
    assert exc.value.status == 404


def test_static_map_is_exact_allowlist_over_assets():
    served = {name for name, _ in server.STATIC_ROUTES.values()}
    assert served == set(server.STATIC_ASSETS)
    assert not any(route.endswith("/") and route != "/"
                   for route in server.STATIC_ROUTES)
    # Text responses must declare their charset. The brand images carry their
    # exact image type and NOTHING else: a charset on an SVG/ICO/PNG describes
    # bytes that are not text, and the browser's own icon decoding is the thing
    # that would suffer.
    for route, (_, ctype) in server.STATIC_ROUTES.items():
        if ctype.startswith("image/"):
            assert ctype in ("image/svg+xml", "image/png", "image/x-icon"), route
        else:
            assert "charset=utf-8" in ctype, route


def test_defensive_headers_present_and_no_cors_ever():
    names = {name for name, _ in server._DEFENSIVE_HEADERS}
    assert {"X-Content-Type-Options", "Referrer-Policy",
            "Cross-Origin-Resource-Policy", "X-Frame-Options",
            "Content-Security-Policy", "Cache-Control",
            "Connection"} == names
    assert not any(name.lower().startswith("access-control") for name in names)
    assert "connect-src 'self'" in server.CSP


# --------------------------------------------------------- manifest reads


def test_read_manifest_limited_bounded_and_hang_free(tmp_path):
    good = tmp_path / "manifest.json"
    good.write_text('{"schema_version": 1, "artifacts": []}', encoding="utf-8")
    data, err = server.read_manifest_limited(str(good))
    assert err is None and b"schema_version" in data

    assert server.read_manifest_limited(str(tmp_path / "missing.json")) == \
        (None, "manifest_file_missing")
    assert server.read_manifest_limited(str(tmp_path)) == (None,
                                                           api.MANIFEST_UNREADABLE)
    link = tmp_path / "link.json"
    link.symlink_to(good)
    assert server.read_manifest_limited(str(link)) == (None, api.MANIFEST_UNREADABLE)
    fifo = tmp_path / "fifo.json"
    import os
    os.mkfifo(fifo)
    # must not block: O_NONBLOCK + regular-file check
    assert server.read_manifest_limited(str(fifo)) == (None, api.MANIFEST_UNREADABLE)
    big = tmp_path / "big.json"
    big.write_bytes(b"{ " + b" " * (1 << 20) + b"}")
    assert server.read_manifest_limited(str(big)) == (None, "manifest_too_large")


def test_manifest_reader_rejects_initial_oversize_without_read(tmp_path, monkeypatch):
    path = tmp_path / "large.json"
    path.write_bytes(b"x" * (server._MANIFEST_LIMIT + 1))

    def no_read(*args):
        raise AssertionError("oversized input was read")

    monkeypatch.setattr(server.os, "read", no_read)
    assert server.read_manifest_limited(str(path)) == (None, "manifest_too_large")


@pytest.mark.parametrize("change", ["grow", "shrink", "replace", "restore-mtime"])
def test_manifest_reader_refuses_observed_changes(tmp_path, monkeypatch, change):
    path = tmp_path / "manifest.json"
    path.write_bytes(b"abcdef")
    before = path.stat()
    real_read = server.os.read
    changed, requests, received = [], [], []

    def mutate(fd, size):
        requests.append(size)
        block = real_read(fd, size)
        received.append(len(block))
        if not changed:
            changed.append(True)
            if change == "grow":
                with path.open("ab") as out:
                    out.write(b"x" * 100)
            elif change == "shrink":
                path.write_bytes(b"a")
            elif change == "replace":
                path.rename(tmp_path / "original.json")
                path.write_bytes(b"ABCDEF")
            else:
                path.write_bytes(b"ABCDEF")
                server.os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return block

    monkeypatch.setattr(server.os, "read", mutate)
    assert server.read_manifest_limited(str(path)) == (None, api.MANIFEST_UNREADABLE)
    assert sum(received) <= before.st_size + 1
    assert max(requests) <= before.st_size + 1


# ------------------------------------------------------------- entrypoint


def test_help_is_ml_free_binds_nothing_writes_nothing(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "qwen_engine.web", "--help"],
        capture_output=True, text=True, timeout=60,
        env={"PYTHONPATH": str(Path(server.__file__).resolve().parents[2]),
             "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
             "TMPDIR": str(tmp_path)})
    assert proc.returncode == 0, proc.stderr
    assert "usage: qwen-web" in proc.stdout
    assert "planning only" in proc.stdout.lower() or "planning" in proc.stdout.lower()
    assert proc.stderr == ""


def test_main_refusals_exit_2_before_any_bind(capsys):
    assert server.main(["--host", "0.0.0.0"]) == 2
    assert server.main(["--host", "localhost"]) == 2   # no DNS decisions
    assert server.main(["--host", "::1"]) == 2          # IPv6 not reviewed/tested
    assert server.main(["--port", "70000"]) == 2
    err = capsys.readouterr().err
    assert "3081" not in err
    assert "refusing to bind" in err


def test_main_startup_config_validation_before_bind(tmp_path, capsys):
    blocker = tmp_path / "afile"
    blocker.write_text("x", encoding="utf-8")
    # a regular file is not a directory: exit 2, before any bind/write
    assert server.main(["--output-dir", str(blocker)]) == 2
    assert "config error" in capsys.readouterr().err


def test_reuse_verified_assets_parser_defaults():
    p = server.build_parser()
    assert p.parse_args([]).reuse_verified_assets is None          # off by default
    assert p.parse_args(["--reuse-verified-assets"]).reuse_verified_assets == 3600.0
    assert p.parse_args(["--reuse-verified-assets", "45"]).reuse_verified_assets == 45.0


def test_main_refuses_bad_reuse_age_before_any_bind(capsys):
    assert server.main(["--reuse-verified-assets", "0"]) == 2
    assert server.main(["--reuse-verified-assets", "-5"]) == 2
    err = capsys.readouterr().err
    assert err.count("--reuse-verified-assets") == 2 and "max_age_s" in err


def test_module_entrypoint_and_script_declared():
    pyproject = tomllib.loads(
        (Path(server.__file__).resolve().parents[3] / "pyproject.toml")
        .read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["qwen-web"] == "qwen_engine.web.server:main"
    data = pyproject["tool"]["setuptools"]["package-data"]["qwen_engine"]
    for asset in server.STATIC_ASSETS:
        entry = f"web_static/{asset}"
        assert entry in data, entry
        # Listed AND present on disk: a typo (or a glob matching nothing) would
        # ship an install that fails the moment that file is requested.
        # web/server.py -> qwen_engine/ is the package that holds web_static/.
        assert (Path(server.__file__).resolve().parents[1] / entry).is_file(), entry
    assert not any("*" in entry for entry in data), \
        "package data is an explicit list, so a missing brand file cannot pass silently"
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert find["namespaces"] is False


def test_web_import_is_ml_free_and_never_touches_weights():
    code = (
        "import sys, qwen_engine.web.server, qwen_engine.web.api;"
        "assert not [m for m in sys.modules"
        " if m.split('.')[0] in ('torch','diffusers','transformers')]"
    )
    proc = subprocess.run([sys.executable, "-B", "-c", code],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_chrome_lifecycle_protocol_selfcheck_pure():
    # Pure protocol checks for the shared Chrome helpers (no Chrome launch,
    # no fake DOM): timeouts, pending cleanup, owner-only signals, spawn
    # errors, and the documented no---no-zygote source fact.
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = (Path(__file__).resolve().parent.parent
              / "scripts" / "chrome_lifecycle_selfcheck.mjs")
    proc = subprocess.run([node, str(script)], capture_output=True,
                          text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAIL" not in proc.stdout
    # Pure protocol success must point to separate browser evidence, without
    # making a stale global claim that independently completed checks are pending.
    assert "PURE protocol checks only" in proc.stdout
    assert "chrome_browser_check.mjs" in proc.stdout
    assert "web_integration_check.mjs" in proc.stdout


def test_raw_http_framing_selfcheck_pure():
    # Pure framing protocol checks for scripts/raw_http.mjs (no server, no
    # socket, no Chrome): interim-1xx WITH headers sliced at the CRLF
    # boundary (the old regex bug), final-block content-length, capped
    # bytes, latin1 heads and UTF-8 bodies.
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = (Path(__file__).resolve().parent.parent
              / "scripts" / "raw_http_selfcheck.mjs")
    proc = subprocess.run([node, str(script)], capture_output=True,
                          text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAIL" not in proc.stdout
    assert "pure framing checks" in proc.stdout        # honesty note present


# ------------------------------------------- root-follow-up regressions

def _memory_response(raw: bytes, assets: dict[str, bytes] | None = None) -> bytes:
    """Drive _Handler.handle_one_request over an in-memory socket adapter —
    NO real socket is created (the guarded suite forbids them), while the
    FULL stdlib request-line path runs verbatim. Returns the RAW response
    bytes, so a body can be compared byte for byte."""
    import io
    import types

    h = server._Handler.__new__(server._Handler)
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.connection = types.SimpleNamespace(
        getsockname=lambda: ("127.0.0.1", 8765))
    h.server = types.SimpleNamespace(bound_ip="127.0.0.1", bound_port=8765,
                                     assets=assets if assets is not None else
                                     {"app.js": b"console.log('fixed asset')"},
                                     web_runtime=api.WebRuntime())
    h.close_connection = False
    h.request_version = "HTTP/1.1"
    h.address_string = lambda: "127.0.0.1"
    h.log_message = lambda *a, **k: None
    h.handle_one_request()
    return h.wfile.getvalue()


def _memory_handler(raw: bytes) -> str:
    return _memory_response(raw).decode("latin-1")


def test_overlong_request_line_gets_typed_414_not_silent_close():
    out = _memory_handler(b"GET /" + b"x" * 70000 + b" HTTP/1.1\r\n\r\n")
    assert "HTTP/1.1 414" in out.split("\r\n")[0]
    assert "malformed_request" in out              # typed JSON, not bare close
    assert "X-Content-Type-Options: nosniff" in out  # defensive headers present


def test_tab_separated_request_line_is_never_served_through_path_folding():
    # The stdlib normalizes the FOLDED self.path ('\t//app.js' -> '/app.js'),
    # but ROUTING uses the RAW target token: '//app.js' is an invalid form
    # and must be a typed 400 — never a 200 of app.js.
    out = _memory_handler(b"GET\t//app.js HTTP/1.1\r\n"
                          b"Host: 127.0.0.1:8765\r\n\r\n")
    first = out.split("\r\n")[0]
    assert " 400 " in first, first
    assert "malformed" in out.lower() or "target" in out.lower()


def test_unknown_method_with_foreign_host_hits_host_policy_first():
    out = _memory_handler(b"WEIRD / HTTP/1.1\r\nHost: evil.example:8765\r\n\r\n")
    assert " 400 " in out.split("\r\n")[0]
    assert "invalid_host" in out                    # policy BEFORE method typing


# --------------------- legacy/early-error HTTP framing (root 9-case review)

#: The four root-review FAIL legs: versionless request line, explicit
#: HTTP/0.9, broken request line, HTTP/2.0. ALL four must come back with a
#: REAL HTTP/1.x status line and the full shared defensive header set —
#: never a bare body (the inherited stdlib emitters suppress EVERYTHING
#: while request_version is the HTTP/0.9 default).
_LEGACY_FRAMING_CASES = (
    (b"GET /app.js\r\nHost: 127.0.0.1:8765\r\n\r\n", 400),
    (b"GET /app.js HTTP/0.9\r\nHost: 127.0.0.1:8765\r\n\r\n", 400),
    (b"BROKEN\r\n\r\n", 400),
    (b"GET /app.js HTTP/2.0\r\nHost: 127.0.0.1:8765\r\n\r\n", 505),
)


@pytest.mark.parametrize("wire,status", _LEGACY_FRAMING_CASES,
                         ids=["no-version", "explicit-0.9", "broken-line",
                              "http-2.0"])
def test_legacy_and_early_errors_are_framed_never_bare(wire, status):
    out = _memory_handler(wire)
    head, _, body = out.partition("\r\n\r\n")
    assert head.startswith(f"HTTP/1.1 {status} "), head.split("\r\n")[0]
    for name, value in server._DEFENSIVE_HEADERS:
        assert f"{name}: {value}" in head, name    # shared defensive set
    assert '"schema": "qwen_engine.web_error.v1"' in body  # typed JSON
    assert "console.log" not in out, \
        "a legacy request must NEVER be served — framed refusal only"


@pytest.mark.parametrize("wire", [
    b"POST /api/v1/plan\r\nHost: 127.0.0.1:8765\r\n\r\n",
    b"POST /api/v1/plan HTTP/0.9\r\nHost: 127.0.0.1:8765\r\n\r\n",
], ids=["no-version", "explicit-0.9"])
def test_legacy_versions_refused_before_dispatch_not_normalized(wire):
    # The gate fires in parse_request BEFORE any dispatch decision: a
    # fully header-valid POST without an HTTP/1.x version must hit the
    # version refusal (unsupported_request_version), NEVER the POST
    # write-policy chain (ui_header_required/411/415) — proof that the
    # version is refused, not normalized into a servable state.
    out = _memory_handler(wire)
    head, _, body = out.partition("\r\n\r\n")
    assert head.startswith("HTTP/1.1 400 "), head
    # Refused BEFORE dispatch either way — by the stdlib's own legacy type
    # check (no version token) or by our gate (explicit HTTP/0.9) — but
    # never by the POST write-policy chain, which only dispatch reaches.
    if b"HTTP/0.9" in wire:
        assert "unsupported_request_version" in body
    else:
        assert "Bad HTTP/0.9 request type" in body
    assert "ui_header_required" not in body and "length_required" not in body
    # parse_request must REFUSE, never reassign request_version to serve:
    import inspect
    assert "self.request_version =" not in inspect.getsource(
        server._Handler.parse_request)


def test_malformed_head_refusals_are_framed_and_payloadless():
    # HEAD framing contract under early/legacy errors: real status line and
    # headers (with an honest Content-Length) but NEVER a payload — even
    # though self.command is not set yet when the error fires.
    for wire in (b"HEAD /app.js\r\nHost: 127.0.0.1:8765\r\n\r\n",
                 b"HEAD /app.js HTTP/0.9\r\nHost: 127.0.0.1:8765\r\n\r\n"):
        out = _memory_handler(wire)
        head, _, body = out.partition("\r\n\r\n")
        assert head.startswith("HTTP/1.1 400 "), head.split("\r\n")[0]
        for name, value in server._DEFENSIVE_HEADERS:
            assert f"{name}: {value}" in head, name
        assert "content-length:" in head.lower()
        assert body == "", wire   # headers framed, payload NEVER


@pytest.mark.parametrize("version", ["HTTP/1.0", "HTTP/1.1"])
def test_supported_http_versions_behave_exactly_as_before(version):
    out = _memory_handler(f"GET /app.js {version}\r\n"
                          f"Host: 127.0.0.1:8765\r\n\r\n".encode("ascii"))
    head, _, body = out.partition("\r\n\r\n")
    assert head.startswith("HTTP/1.1 200 OK"), head.split("\r\n")[0]
    assert body == "console.log('fixed asset')"
    for name, value in server._DEFENSIVE_HEADERS:
        assert f"{name}: {value}" in head, name


def test_legacy_framing_normalization_shares_the_one_header_builder():
    # Static drift guard: the legacy/error-state branch of _send must use
    # the SAME shared builder/constant as every other response — a second
    # literal header list anywhere is exactly the bug this guards.
    import inspect
    src = inspect.getsource(server._Handler._send)
    assert 'if self.request_version == "HTTP/0.9":' in src
    assert "_framed_bytes" in src and "send_response_only" in src
    for name, _ in server._DEFENSIVE_HEADERS:
        assert f'"{name}"' not in src


def test_drain_connection_queue_recognizes_sentinels(tmp_path):
    import queue as _queue

    q: _queue.Queue = _queue.Queue()
    socks = [_FakeSocket() for _ in range(3)]
    for s in socks:
        q.put((s, ("127.0.0.1", 50000)))
    q.put(server._WORKER_SHUTDOWN)                  # leftover sentinel
    assert server.drain_connection_queue(q) == 3    # only REAL connections
    assert all(s.closed for s in socks)
    assert q.unfinished_tasks == 0                  # sentinel task_done too
    # a second drain over leftovers must not raise (no blind tuple unpack)
    q.put(server._WORKER_SHUTDOWN)
    assert server.drain_connection_queue(q) == 0


def test_server_close_is_idempotent_and_never_double_signals():
    import inspect
    src = inspect.getsource(server._Server.server_close)
    assert "if self._stopping:" in src and "return" in src, \
        "second server_close must be a no-op, not a re-drain/re-stop"
    assert "still_alive = stop_workers(self._queue, self._workers)" in src
    assert "self._workers = still_alive" in src, \
        "workers that failed to join keep references — no false all-joined claim"


# ------------------------------------------------- rewrite startup flags

def test_rewrite_flags_default_off_with_the_service_timeout():
    from qwen_engine.web import rewrite_service as rs
    args = server.build_parser().parse_args([])
    assert args.rewrite_backend == "off"
    assert args.rewrite_timeout == rs.DEFAULT_TIMEOUT_SECONDS
    for flag in ("rewrite_endpoint", "rewrite_model", "rewrite_prompt_t2i",
                 "rewrite_prompt_edit"):
        assert getattr(args, flag) is None, flag


def test_rewrite_backend_choices_are_closed():
    from qwen_engine.web import rewrite_service as rs
    parser = server.build_parser()
    for bad in ("local", "openai", "llm", ""):
        with pytest.raises(SystemExit):
            parser.parse_args(["--rewrite-backend", bad])
    args = parser.parse_args(["--rewrite-backend", "local-llm"])
    assert args.rewrite_backend in rs.BACKENDS


def test_off_backend_builds_a_real_disabled_service():
    args = server.build_parser().parse_args([])
    svc = server.build_rewrite_service(args)
    assert svc.available is False
    assert svc.status_block()["code"] == "rewrite_disabled"


def test_enabled_backend_without_prompt_paths_is_a_startup_error():
    from qwen_engine import rewrite_client as rc
    args = server.build_parser().parse_args(["--rewrite-backend", "local-llm"])
    with pytest.raises(rc.RewriteConfigError):
        server.build_rewrite_service(args)


def test_prompt_files_are_digest_pinned_at_startup(tmp_path):
    from qwen_engine import rewrite as rw
    prompt = tmp_path / "sys.txt"
    prompt.write_text("not the pinned prompt\n", encoding="utf-8")
    args = server.build_parser().parse_args([
        "--rewrite-backend", "local-llm", "--rewrite-endpoint",
        "http://127.0.0.1:8079/v1/chat/completions", "--rewrite-model", "m",
        "--rewrite-prompt-t2i", str(prompt), "--rewrite-prompt-edit", str(prompt)])
    with pytest.raises(rw.RewriteError) as excinfo:
        server.build_rewrite_service(args)
    assert "pinned digest" in str(excinfo.value)


def test_main_returns_2_for_a_rewrite_configuration_error(monkeypatch, capsys):
    # Operator error: one bounded stderr line, exit 2, never a partial server.
    # main() must fail BEFORE any bind, so no socket is created here.
    args = ["--rewrite-backend", "local-llm"]
    monkeypatch.setattr(server, "build_rewrite_service",
                        lambda _args, _refs=None, cache_dir=None, gpu_lock=None: (
                            _ for _ in ()).throw(
                            __import__("qwen_engine.rewrite_client",
                                       fromlist=["x"]).RewriteConfigError(
                                "operator misconfiguration")))
    assert server.main(args) == 2
    err = capsys.readouterr().err
    assert "rewrite error:" in err and "operator misconfiguration" in err


def test_main_hands_the_reference_store_to_the_rewriter(monkeypatch, tmp_path):
    # Edit rewrites read their images from the server's reference store, so
    # main() must pass the resolved store path, not leave it unset.
    from qwen_engine import rewrite_client as rc
    seen = []

    def fake(args, references_dir=None, cache_dir=None, gpu_lock=None):
        seen.append((references_dir, cache_dir))
        raise rc.RewriteConfigError("stop before bind")

    monkeypatch.setattr(server, "build_rewrite_service", fake)
    refs = tmp_path / "refs"
    assert server.main(["--rewrite-backend", "local-llm",
                        "--references-dir", str(refs),
                        "--cache-dir", str(tmp_path / "cache")]) == 2
    assert seen == [(os.path.realpath(str(refs)),
                     os.path.realpath(str(tmp_path / "cache")))]


# ------------------------------------------------------------- brand assets

#: route -> (packaged asset, exact content type, the magic the bytes start with)
_BRAND_ROUTES = (
    ("/brand/lightseed-logo-on-dark.svg", "brand/lightseed-logo-on-dark.svg",
     "image/svg+xml", b"<svg"),
    ("/brand/lightseed-logo-on-light.svg", "brand/lightseed-logo-on-light.svg",
     "image/svg+xml", b"<svg"),
    ("/brand/favicon.svg", "brand/favicon.svg", "image/svg+xml", b"<svg"),
    ("/brand/favicon-180.png", "brand/favicon-180.png", "image/png",
     b"\x89PNG\r\n\x1a\n"),
    # The ICO lives in the brand folder but is published at the ROOT, because
    # that is where browsers ask for it without being told to.
    ("/favicon.ico", "brand/favicon.ico", "image/x-icon", b"\x00\x00\x01\x00"),
)


@pytest.mark.parametrize("route,asset,ctype,magic", _BRAND_ROUTES,
                         ids=["logo-dark", "logo-light", "icon-svg",
                              "icon-180", "icon-ico"])
def test_brand_routes_serve_the_exact_file_bytes_with_their_own_type(route, asset,
                                                                    ctype, magic):
    """Full request path, no socket: the bytes a browser gets must BE the
    packaged file (no re-encoding, no charset on an image)."""
    out = _memory_response(f"GET {route} HTTP/1.1\r\n"
                           f"Host: 127.0.0.1:8765\r\n\r\n".encode(),
                           server.load_static_assets())
    head, _, body = out.partition(b"\r\n\r\n")
    text = head.decode("latin-1")
    assert text.split("\r\n")[0] == "HTTP/1.1 200 OK", text.split("\r\n")[0]
    assert f"Content-Type: {ctype}" in text, text
    assert "charset" not in text.split("Content-Type", 1)[1].split("\r\n")[0], \
        f"{ctype} must not claim a charset"
    assert f"Content-Length: {len(body)}" in text
    assert "X-Content-Type-Options: nosniff" in text
    assert body.startswith(magic), route
    assert body == server.load_static_assets()[asset], \
        "the served bytes must be the packaged file verbatim"
    # The CSP the page ships with already allows 'img-src self': a brand file
    # must never be the reason to loosen it.
    assert "img-src 'self'" in text


def test_brand_asset_table_matches_magic_and_stays_inside_the_folder():
    for route, (asset, ctype) in server.STATIC_ROUTES.items():
        if not asset.startswith("brand/"):
            continue
        data = server.load_static_assets()[asset]
        if ctype == "image/svg+xml":
            assert data.startswith(b"<svg"), route
        elif ctype == "image/png":
            assert data.startswith(b"\x89PNG\r\n\x1a\n"), route
        elif ctype == "image/x-icon":
            assert data[:4] == b"\x00\x00\x01\x00", route
        else:
            raise AssertionError(f"unhandled brand content type {ctype} for {route}")
        # Every brand name is a fixed one-level constant under web_static/brand/:
        # no traversal, nothing outside the folder, never a request-derived path.
        assert asset.startswith("brand/"), asset
        assert "/" not in asset[len("brand/"):], asset
    assert server.resolve_route("GET", "/favicon.ico") == ("static", "/favicon.ico")
    # An unlisted brand-looking path is still a 404, never a file read.
    with pytest.raises(api.ApiError) as exc:
        server.resolve_route("GET", "/brand/lightseed-banner.png")
    assert exc.value.status == 404


def test_load_static_assets_reads_the_brand_subfolder_byte_for_byte():
    from qwen_engine.web.server import load_static_assets

    # web/server.py -> qwen_engine/web_static (the packaged data directory).
    root = Path(server.__file__).resolve().parents[1] / "web_static"
    assets = load_static_assets()
    assert set(assets) == set(server.STATIC_ASSETS)
    for name in server.STATIC_ASSETS:
        on_disk = (root / name).read_bytes()
        assert assets[name] == on_disk, name
    assert any(name.startswith("brand/") for name in assets), \
        "the loader must reach the brand subfolder, not only the top level"
