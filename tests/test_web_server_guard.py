"""Guarded W2 tests: pure transport policy + entrypoint/packaging checks.

No sockets are ever created here (the guarded runner denies them anyway);
the real-socket smoke checks live in the SEPARATE integration command
``scripts/web_integration_check.mjs``. These tests exercise the pure policy
functions that the handler delegates to, plus the CLI contract.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from qwen_engine.web import api
from qwen_engine.web import server

PORT = 8765


def h(**name_to_values: list[str]) -> dict[str, list[str]]:
    return {k.replace("_", "-"): v for k, v in name_to_values.items()}


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
    ex = dict(post); ex["expect"] = ["100-Continue"]
    assert server.header_policy("POST", ex, ok_origin) is None


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
    for route, (_, ctype) in server.STATIC_ROUTES.items():
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


def test_module_entrypoint_and_script_declared():
    pyproject = tomllib.loads(
        (Path(server.__file__).resolve().parents[3] / "pyproject.toml")
        .read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["qwen-web"] == "qwen_engine.web.server:main"
    data = pyproject["tool"]["setuptools"]["package-data"]["qwen_engine"]
    for asset in server.STATIC_ASSETS:
        assert f"web_static/{asset}" in data
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
