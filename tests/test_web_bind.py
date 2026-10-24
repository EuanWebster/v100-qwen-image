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
