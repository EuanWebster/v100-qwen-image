"""Explicit LAN/Tailscale listeners retain address and Origin checks."""
import contextlib
import io
import types

import pytest

from qwen_engine.web import server


@pytest.mark.parametrize('host', ['127.0.0.1', '192.168.68.103', '10.1.2.3', '172.16.1.2', '100.98.64.62'])
def test_explicit_private_ipv4_bind(host):
    assert server.valid_bind_host(host)
    assert server.validate_host([host + ':8765'], host, 8765) == ('http://' + host + ':8765', None)
    assert server.validate_host(['evil.invalid:8765'], host, 8765)[1]
    assert server.header_policy('GET', {'origin': ['http://evil.invalid:8765']}, 'http://' + host + ':8765').status == 403


@pytest.mark.parametrize('host', ['0.0.0.0', '::1', 'localhost', 'example.com', '8.8.8.8', '169.254.1.1', '224.0.0.1', '192.168.068.103'])
def test_wildcards_dns_public_ipv6_bind_refused(host):
    assert not server.valid_bind_host(host)


def test_other_listener_host_is_not_accepted():
    assert server.validate_host(['100.98.64.62:8765'], '192.168.68.103', 8765)[1]
    assert server.validate_host(['localhost:8765'], '192.168.68.103', 8765)[1]


def test_partial_bind_failure_closes_open_listener(monkeypatch):
    opened = []
    class Listener:
        def __init__(self, address, handler):
            if opened: raise OSError('address unavailable')
            self.server_address, self.closed = address, False
            opened.append(self)
        def server_close(self): self.closed = True
    monkeypatch.setattr(server, '_Server', Listener)
    assert server.main(['--host', '192.168.68.103', '--host', '100.98.64.62']) == 1
    assert len(opened) == 1 and opened[0].closed


def test_constructor_bind_failure_keeps_primary_error_and_closes(monkeypatch):
    closed = []
    def init(listener, *args):
        listener.server_close()
        raise OSError('address already in use')
    monkeypatch.setattr(server.HTTPServer, '__init__', init)
    monkeypatch.setattr(server.HTTPServer, 'server_close', lambda listener: closed.append(listener))
    with pytest.raises(OSError, match='address already in use'):
        server._Server(('127.0.0.1', 8765), server._Handler)
    assert len(closed) == 1


def test_listeners_share_worker_and_shutdown(tmp_path, monkeypatch):
    from qwen_engine.web import generation
    opened, workers = [], []
    class Listener:
        def __init__(self, address, handler):
            self.server_address, self.closed = (address[0], address[1] or 54321), False
            opened.append(self)
        def server_close(self): self.closed = True
    class Worker:
        def __init__(self, runtime, **kwargs):
            self.closed = False
            workers.append(self)
        def close(self):
            self.closed = True
            return True
    class Selector:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def register(self, *args): pass
        def select(self, **kwargs):
            assert len(workers) == 1
            assert all(listener.generation is workers[0] for listener in opened)
            assert all(listener.bound_port == 54321 for listener in opened)
            raise KeyboardInterrupt
    monkeypatch.setattr(server, '_Server', Listener)
    monkeypatch.setattr(generation, 'WebGeneration', Worker)
    monkeypatch.setattr(server.selectors, 'DefaultSelector', Selector)
    monkeypatch.setattr(server, 'read_manifest_limited', lambda _: (b'{}', None))
    monkeypatch.setattr(server.api, 'summarize_manifest_bytes', lambda *a: {
        'configured': True, 'valid': True, 'schema_version': 1, 'artifact_count': 24})
    result = server.main(['--host', '127.0.0.1', '--host', '192.168.68.103', '--host', '100.98.64.62',
                          '--port', '0', '--enable-generation', '--manifest', str(tmp_path / 'manifest.json'),
                          '--models-dir', str(tmp_path / 'models'), '--output-dir', str(tmp_path / 'out'),
                          '--cache-dir', str(tmp_path / 'cache')])
    assert result == 0
    assert workers[0].closed and all(listener.closed for listener in opened)


def test_reuse_flag_reaches_worker_with_path_free_notice(tmp_path, monkeypatch, capsys):
    from qwen_engine.web import generation
    workers = []

    class Listener:
        def __init__(self, address, handler):
            self.server_address, self.closed = (address[0], address[1] or 54321), False
            self.bound_ip, self.bound_port = address[0], address[1] or 54321
        def server_close(self): self.closed = True

    class Worker:
        def __init__(self, runtime, **kwargs):
            self.kwargs, self.closed = kwargs, False
            workers.append(self)
        def close(self):
            self.closed = True
            return True

    class Selector:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def register(self, *args): pass
        def select(self, **kwargs): raise KeyboardInterrupt

    monkeypatch.setattr(server, '_Server', Listener)
    monkeypatch.setattr(generation, 'WebGeneration', Worker)
    monkeypatch.setattr(server.selectors, 'DefaultSelector', Selector)
    monkeypatch.setattr(server, 'read_manifest_limited', lambda _: (b'{}', None))
    monkeypatch.setattr(server.api, 'summarize_manifest_bytes', lambda *a: {
        'configured': True, 'valid': True, 'schema_version': 1, 'artifact_count': 24})
    base = ['--host', '127.0.0.1', '--host', '192.168.68.103', '--port', '0',
            '--enable-generation', '--manifest', str(tmp_path / 'manifest.json'),
            '--models-dir', str(tmp_path / 'models'), '--output-dir', str(tmp_path / 'out'),
            '--cache-dir', str(tmp_path / 'cache')]
    assert server.main(base + ['--reuse-verified-assets', '90']) == 0
    policy = workers[0].kwargs['reuse_verified_assets']
    assert policy is not None and policy.max_age_s == 90
    out = capsys.readouterr().out
    assert 'asset-verification reuse is ON' in out and '90s' in out
    assert str(tmp_path) not in out        # the notice never names paths
    assert server.main(base) == 0
    out = capsys.readouterr().out
    assert workers[1].kwargs['reuse_verified_assets'] is None
    assert 'reuse' not in out              # off by default: nothing announced


# ------------------------------------------------------------- remote listener
# The four C1 flags are all-or-none, resolved ONCE before any bind: a
# half-configured remote port would otherwise open a listener with no token or
# no expected Origin. These tests drive the REAL ``server.main`` with fake
# socket objects, so what is pinned is the startup wiring, not a helper.

REMOTE_HEAD = ["--remote-port", "0", "--remote-host", "100.98.64.62",
               "--remote-origin", "https://studio.example.com"]


def _token_file(tmp_path, mode=0o600, secret=b"x" * 32):
    path = tmp_path / "remote-token"
    path.write_bytes(secret + b"\n")
    path.chmod(mode)
    return str(path)


def _remote_args(token):
    return REMOTE_HEAD + ["--remote-token-file", token]


@contextlib.contextmanager
def _captured():
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


def _run_main(monkeypatch, tmp_path, extra, *, bind_error=None, cache=True):
    """``server.main`` with fake listeners; returns (code, listeners, output)."""
    opened = []

    class Listener:
        def __init__(self, address, handler):
            host, port = address
            if host == bind_error:
                raise OSError("address unavailable")
            self.server_address = (host, port or 54321 + len(opened))
            self.bound_ip, self.bound_port = self.server_address
            self.closed = False
            opened.append(self)

        def server_close(self):
            self.closed = True

    class Selector:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def register(self, *args):
            pass

        def select(self, **kwargs):
            raise KeyboardInterrupt

    class Service:
        available = False
        def active_by_task(self): return {}
        def available_backends(self): return ()
        def status_block(self):
            # The closed schema api.status_response validates: this fake is
            # only here so the remote wiring can be exercised.
            return {"state": "unavailable", "code": "rewrite_disabled",
                    "reason": "prompt rewriting is not configured",
                    "backend": None}
        def close(self): return True

    monkeypatch.setattr(server, "_Server", Listener)
    monkeypatch.setattr(server.selectors, "DefaultSelector", Selector)
    monkeypatch.setattr(server, "_startup_config_or_exit", lambda paths: 0)
    monkeypatch.setattr(server, "build_rewrite_service", lambda *a, **k: Service())
    monkeypatch.setattr(server, "load_saved_preferences", lambda *a: None)
    monkeypatch.setattr(server, "read_manifest_limited", lambda _: (None, "absent"))
    monkeypatch.setattr(server._setup.ModelSetupService, "for_paths",
                        classmethod(lambda cls, **kwargs: types.SimpleNamespace(
                            notices=lambda: (), required_missing=lambda: (),
                            state=lambda: {"items": []})))
    base = ["--host", "127.0.0.1", "--port", "0"]
    if cache:
        (tmp_path / "cache").mkdir(exist_ok=True)
        base += ["--cache-dir", str(tmp_path / "cache")]
    with _captured() as streams:
        code = server.main(base + extra)
    return code, opened, streams[0].getvalue() + streams[1].getvalue()


def _settings(argv):
    return server.remote_settings_error(server.build_parser().parse_args(argv))


def test_no_remote_flags_at_all_is_not_an_error():
    assert _settings([]) is None


@pytest.mark.parametrize("argv,missing", [
    (["--remote-port", "0"], ["--remote-host", "--remote-token-file",
                              "--remote-origin"]),
    (["--remote-origin", "https://a.example.com"],
     ["--remote-port", "--remote-host", "--remote-token-file"]),
    (["--remote-port", "0", "--remote-host", "100.98.64.62",
      "--remote-token-file", "t"], ["--remote-origin"]),
])
def test_the_remote_flags_are_all_or_none(argv, missing):
    message = _settings(argv)
    assert message is not None and "missing" in message
    for name in missing:
        assert name in message


def test_remote_port_zero_counts_as_given_not_as_absent():
    # Port 0 is a legal ephemeral bind (the live tests use it). Reading it as
    # "flag omitted" would silently drop the listener the operator asked for.
    assert _settings(REMOTE_HEAD + ["--remote-token-file", "t"]) is None


@pytest.mark.parametrize("port", ["-1", "65536"])
def test_remote_port_must_be_in_range(port):
    assert _settings(["--remote-port", port, "--remote-host", "100.98.64.62",
                      "--remote-token-file", "t",
                      "--remote-origin", "https://a.example.com"]) is not None


@pytest.mark.parametrize("origin", [
    "https://studio.example.com/", "https://studio.example.com/path",
    "https://user:pw@studio.example.com", "ftp://studio.example.com",
    "studio.example.com", "https://studio.example.com?x=1", "",
    "https://ex ample.com",
])
def test_remote_origin_must_be_one_bare_http_origin(origin):
    message = _settings(REMOTE_HEAD[:4] + ["--remote-token-file", "t",
                                           "--remote-origin", origin])
    assert message is not None and "origin" in message


@pytest.mark.parametrize("origin", ["https://studio.example.com",
                                    "http://localhost:3000"])
def test_a_bare_http_origin_is_accepted(origin):
    assert _settings(REMOTE_HEAD[:4] + ["--remote-token-file", "t",
                                        "--remote-origin", origin]) is None


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "8.8.8.8", "::1"])
def test_remote_host_uses_the_local_bind_rules(host):
    message = _settings(["--remote-port", "0", "--remote-host", host,
                         "--remote-token-file", "t",
                         "--remote-origin", "https://a.example.com"])
    assert message is not None and "remote-host" in message


def test_remote_host_accepts_up_to_eight_explicit_addresses():
    many = []
    for n in range(9):
        many += ["--remote-host", f"10.0.0.{n + 1}"]
    assert _settings(["--remote-port", "0", "--remote-token-file", "t",
                      "--remote-origin", "https://a.example.com"] + many) is not None


def test_the_remote_listener_is_bound_with_token_origin_and_shared_state(
        tmp_path, monkeypatch):
    token = _token_file(tmp_path)
    code, opened, out = _run_main(monkeypatch, tmp_path, _remote_args(token))
    assert code == 0, out
    assert len(opened) == 2
    local, remote_listener = opened
    assert local.remote is None
    settings = remote_listener.remote
    assert settings.token == b"x" * 32          # the trailing newline is not secret
    assert settings.origin == "https://studio.example.com"
    # ONE worker and ONE rewriter: the remote port is another door, not another
    # GPU user.
    assert remote_listener.web_runtime is local.web_runtime
    assert remote_listener.rewrite is local.rewrite
    assert remote_listener.generation is local.generation
    assert settings.generation is local.generation
    assert f"remote listener on http://100.98.64.62:{remote_listener.bound_port}" in out
    assert all(listener.closed for listener in opened)


def test_the_allowlist_is_the_cache_dirs_users_file(tmp_path, monkeypatch):
    from qwen_engine import users as users_module
    token = _token_file(tmp_path)
    code, opened, _ = _run_main(monkeypatch, tmp_path, _remote_args(token))
    assert code == 0
    assert opened[1].remote.users.path == users_module.users_path(
        str(tmp_path / "cache"))


@pytest.mark.parametrize("mode,fragment", [
    (0o644, "group or others"),
    (0o640, "group or others"),
])
def test_a_token_file_other_users_could_read_refuses_the_whole_startup(
        tmp_path, monkeypatch, mode, fragment):
    token = _token_file(tmp_path, mode=mode)
    code, opened, out = _run_main(monkeypatch, tmp_path, _remote_args(token))
    assert code == 2 and opened == []
    assert fragment in out


def test_a_missing_or_empty_token_file_refuses_the_whole_startup(
        tmp_path, monkeypatch):
    for token in (str(tmp_path / "absent"), _token_file(tmp_path, secret=b"short")):
        code, opened, out = _run_main(monkeypatch, tmp_path, _remote_args(token))
        assert code == 2 and opened == [], out
    assert "unreadable" in out or "token" in out


def test_a_symlinked_token_file_is_refused(tmp_path, monkeypatch):
    real = _token_file(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(real)
    code, opened, out = _run_main(monkeypatch, tmp_path, _remote_args(str(link)))
    assert code == 2 and opened == []
    assert "regular file" in out


def test_the_remote_listener_needs_a_cache_dir_for_the_allowlist(
        tmp_path, monkeypatch):
    token = _token_file(tmp_path)
    code, opened, out = _run_main(monkeypatch, tmp_path, _remote_args(token),
                                  cache=False)
    assert code == 2 and opened == []
    assert "--cache-dir" in out


def test_a_remote_bind_failure_closes_the_local_listener(tmp_path, monkeypatch):
    token = _token_file(tmp_path)
    code, opened, out = _run_main(monkeypatch, tmp_path, _remote_args(token),
                                  bind_error="100.98.64.62")
    assert code == 1
    assert len(opened) == 1 and opened[0].closed
    assert "cannot bind the remote listener" in out
