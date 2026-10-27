"""Explicit LAN/Tailscale listeners retain address and Origin checks."""
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


NAMES = frozenset({'tiktak-ai', 'tiktak-ai.local', 'tiktak-ai.tail3e3fc8.ts.net'})


@pytest.mark.parametrize('host', ['127.0.0.1', '192.168.68.103', '100.98.64.62'])
@pytest.mark.parametrize('name', sorted(NAMES))
def test_listed_machine_names_are_accepted_on_every_listener(host, name):
    assert server.validate_host([name + ':8765'], host, 8765, NAMES) == ('http://' + name + ':8765', None)
    assert server.validate_host([name.upper() + ':8765'], host, 8765, NAMES) == ('http://' + name + ':8765', None)
    # The name earns no exemption from the port or Origin checks.
    assert server.validate_host([name + ':8766'], host, 8765, NAMES)[1]
    assert server.header_policy('GET', {'origin': ['http://' + host + ':8765']}, 'http://' + name + ':8765').status == 403


@pytest.mark.parametrize('name', ['evil.invalid', 'tiktak-ai.evil.invalid', 'evil-tiktak-ai.local', 'tiktak-ai.', ''])
def test_unlisted_names_are_still_refused(name):
    assert server.validate_host([name + ':8765'], '192.168.68.103', 8765, NAMES)[1]


def test_names_are_refused_when_none_are_listed():
    assert server.validate_host(['tiktak-ai.local:8765'], '127.0.0.1', 8765)[1]


@pytest.mark.parametrize('name', ['pc', 'pc.local', 'tiktak-ai.tail3e3fc8.ts.net', 'a-b.c9', 'x' * 63 + '.local'])
def test_valid_host_names(name):
    assert server.valid_host_name(name)


@pytest.mark.parametrize('name', ['', 'localhost', '*.local', 'PC.local', '192.168.68.103', '10.0.0.1', 'pc..local',
                                  '-pc.local', 'pc-.local', 'pc_1.local', 'pc.local:80', 'pc local', 'pç.local',
                                  'x' * 64 + '.local', '.'.join(['a' * 60] * 5)])
def test_invalid_host_names(name):
    assert not server.valid_host_name(name)


@pytest.mark.parametrize('bad', [['*.local'], ['localhost'], ['10.0.0.1'], ['pc_1']] + [[f'pc{i}' for i in range(17)]])
def test_bad_host_name_options_refuse_startup_before_any_bind(monkeypatch, bad, capsys):
    monkeypatch.setattr(server, '_Server', lambda *a: pytest.fail('bound despite a bad --host-name'))
    argv = [arg for name in bad for arg in ('--host-name', name)]
    assert server.main(argv) == 2
    assert '--host-name' in capsys.readouterr().err


class _Started(Exception):
    pass


def _start(monkeypatch, argv):
    """Run main() up to serving, return the listeners it built (never serves)."""
    opened = []
    class Listener:
        def __init__(self, address, handler):
            self.server_address = address
            opened.append(self)
        def server_close(self): pass
        def serve_forever(self): raise _Started
    class Selector:
        def __enter__(self): raise _Started
        def __exit__(self, *exc): return False
    monkeypatch.setattr(server, '_Server', Listener)
    monkeypatch.setattr(server.selectors, 'DefaultSelector', Selector)
    with pytest.raises(_Started):
        server.main(argv)
    return opened


def test_host_names_reach_every_listener(monkeypatch):
    opened = _start(monkeypatch, ['--host', '127.0.0.1', '--host', '192.168.68.103', '--port', '8765',
                                  '--host-name', 'PC.Local.', '--host-name', 'pc'])
    assert len(opened) == 2 and all(listener.host_names == frozenset({'pc.local', 'pc'}) for listener in opened)


def test_settings_file_names_join_the_command_line_names(monkeypatch, tmp_path):
    (tmp_path / 'web-settings.json').write_text('{"host_names": ["tiktak-ai.local", "Tiktak-AI"]}')
    opened = _start(monkeypatch, ['--cache-dir', str(tmp_path), '--port', '8765', '--host-name', 'pc'])
    assert opened[0].host_names == frozenset({'tiktak-ai.local', 'tiktak-ai', 'pc'})


def test_no_settings_file_means_no_names(monkeypatch, tmp_path):
    assert server.read_web_settings(str(tmp_path)) == ([], None)
    assert server.read_web_settings(None) == ([], None)
    assert _start(monkeypatch, ['--cache-dir', str(tmp_path), '--port', '8765'])[0].host_names == frozenset()


@pytest.mark.parametrize('text', ['not json', '[]', '{}', '{"host_names": "pc"}', '{"host_names": [1]}',
                                  '{"host_names": ["pc"], "extra": 1}', '{"host_names": ["*.local"]}',
                                  '{"host_names": ["10.0.0.1"]}', '{"host_names": ["pc"' + ', "x"' * 9000 + ']}',
                                  b'\xff\xfe'])
def test_a_bad_settings_file_refuses_startup_before_any_bind(monkeypatch, tmp_path, capsys, text):
    path = tmp_path / 'web-settings.json'
    path.write_bytes(text if isinstance(text, bytes) else text.encode())
    monkeypatch.setattr(server, '_Server', lambda *a: pytest.fail('bound despite a bad settings file'))
    assert server.main(['--cache-dir', str(tmp_path)]) == 2
    assert 'web-settings.json' in capsys.readouterr().err


def test_a_settings_file_that_is_not_a_regular_file_is_refused(tmp_path):
    (tmp_path / 'target.json').write_text('{"host_names": ["pc"]}')
    (tmp_path / 'web-settings.json').symlink_to(tmp_path / 'target.json')
    assert server.read_web_settings(str(tmp_path))[1]
    (tmp_path / 'web-settings.json').unlink()
    (tmp_path / 'web-settings.json').mkdir()
    assert server.read_web_settings(str(tmp_path))[1]
