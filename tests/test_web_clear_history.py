"""Clearing history: the controller's forget/forget_all file opt-in, the
``POST /api/v1/jobs/forget-all`` route (CSRF, envelope, counts, live skip,
restart survival), the per-job ``delete_files`` forget, symlink refusals and
the history-full 409 wording.

Everything runs CPU-only in tmp directories with fake executors; the wire
checks drive the real ``_Handler`` over an in-memory socket (no real socket
exists — the guarded suite forbids them), exactly like test_web_server_guard.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import types
from pathlib import Path

import pytest

from qwen_engine import jobs
from qwen_engine.config import GenerationConfig
from qwen_engine.core import GenerationResult
from qwen_engine.web import api, server
from qwen_engine.web.generation import WebGeneration

PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j2ioAAAAASUVORK5CYII=')

ORIGIN = 'http://127.0.0.1:8765'


def envelope(**values):
    return json.dumps({'schema': api.WEB_PLAN_REQUEST_SCHEMA,
                       'config': {'prompt': 'clear-me', 'seed': '7', **values}}).encode()


def save(context):
    for phase in ('load', 'encode', 'denoise', 'decode', 'save'):
        context.report_phase(phase)
    path = Path(context.config.output_dir) / 'result.png'
    path.write_bytes(PNG)
    return GenerationResult(str(path))


def explode(context):
    raise RuntimeError('fixture failure')


@pytest.fixture
def root(tmp_path):
    path = tmp_path / 'web'
    path.mkdir()
    return path


@pytest.fixture
def runtime(tmp_path):
    output = tmp_path / 'images'
    output.mkdir()
    return api.WebRuntime(output_dir=str(output), cache_dir=str(tmp_path / 'cache'),
                          models_dir=str(tmp_path / 'models'),
                          manifest_path=str(tmp_path / 'manifest.json'),
                          manifest_summary={'configured': True, 'valid': True, 'schema_version': 1,
                                            'artifact_count': 24, 'verification': 'not_performed'},
                          generation_enabled=True)


@pytest.fixture
def controllers():
    started = []

    def build(root, executor=save, **kwargs):
        controller = jobs.JobController(root, executor, **kwargs)
        started.append(controller)
        return controller

    yield build
    for controller in started:
        controller.close(timeout_s=3)


@pytest.fixture
def services():
    started = []

    def build(runtime, executor=save, **kwargs):
        service = WebGeneration(runtime, executor=executor, **kwargs)
        started.append(service)
        return service

    yield build
    for service in started:
        service.close()


services_fixture = services


# ----------------------------------------------------------- controller core


def test_forget_default_still_touches_no_files(root, controllers):
    controller = controllers(root)
    done = controller.wait(controller.submit(GenerationConfig()).job_id)
    image = Path(done.config.output_dir) / done.artifact.filename
    assert controller.forget(done.job_id) == 0
    assert image.is_file() and Path(done.config.output_dir).is_dir()


def test_forget_delete_files_removes_only_that_jobs_directory(root, controllers, tmp_path):
    outside = tmp_path / 'outside.png'
    outside.write_bytes(PNG)
    controller = controllers(root)
    keeper = controller.wait(controller.submit(GenerationConfig(prompt='keeper')).job_id)
    doomed = controller.wait(controller.submit(GenerationConfig(prompt='doomed')).job_id)
    assert controller.forget(doomed.job_id, delete_files=True) == 1  # the artifact
    assert not Path(doomed.config.output_dir).exists()
    keeper_dir = Path(keeper.config.output_dir)
    assert (keeper_dir / keeper.artifact.filename).is_file()
    assert outside.read_bytes() == PNG
    assert controller.snapshots() == (keeper,)


def test_forget_rejects_non_boolean_delete_files(root, controllers):
    controller = controllers(root)
    done = controller.wait(controller.submit(GenerationConfig()).job_id)
    with pytest.raises(jobs.JobError, match="boolean"):
        controller.forget(done.job_id, delete_files='yes')
    with pytest.raises(jobs.JobError, match="boolean"):
        controller.forget_all(delete_files=1)


def test_forget_all_skips_live_jobs_then_clears(root, controllers):
    gate = threading.Event()

    def gated(context):
        gate.wait(5)
        return save(context)

    controller = controllers(root, executor=gated)
    first = controller.submit(GenerationConfig(prompt='running'))
    while controller.get(first.job_id).state is not jobs.JobState.RUNNING:
        controller.wait(first.job_id, timeout_s=0.05)
    second = controller.submit(GenerationConfig(prompt='queued'))
    assert controller.forget_all() == (0, 0, 2)  # nothing terminal yet
    gate.set()
    assert controller.wait(first.job_id).terminal and controller.wait(second.job_id).terminal
    assert controller.forget_all(delete_files=True) == (2, 2, 0)
    assert controller.snapshots() == ()
    assert not any(Path(record.config.output_dir).exists()
                   for record in (first, second))


def test_forget_all_forgets_every_terminal_job_whatever_its_shape(root, controllers):
    calls = {'n': 0}

    def mixed(context):
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('fixture failure')
        return save(context)

    controller = controllers(root, executor=mixed)
    failed = controller.wait(controller.submit(GenerationConfig(prompt='failed')).job_id)
    done = controller.wait(controller.submit(GenerationConfig(prompt='done')).job_id)
    assert (failed.state, done.state) == (jobs.JobState.FAILED, jobs.JobState.SUCCEEDED)
    # failed: its record was the only file -> 0; succeeded: its artifact -> 1
    assert controller.forget_all(delete_files=True) == (2, 1, 0)
    assert not Path(failed.config.output_dir).exists()
    assert not Path(done.config.output_dir).exists()


def test_forget_all_refuses_symlinked_entry_and_keeps_everything(root, controllers, tmp_path):
    outside = tmp_path / 'outside.png'
    outside.write_bytes(PNG)
    controller = controllers(root)
    done = controller.wait(controller.submit(GenerationConfig()).job_id)
    directory = Path(done.config.output_dir)
    link = directory / 'escape.png'
    link.symlink_to(outside)
    assert controller.forget_all(delete_files=True) == (1, 0, 0)
    assert outside.read_bytes() == PNG          # never followed, never deleted
    assert (directory / done.artifact.filename).is_file()  # whole directory kept
    assert link.is_symlink()
    assert controller.snapshots() == ()         # the RECORD is still forgotten


def test_forget_all_refuses_replaced_directory(root, controllers, tmp_path):
    controller = controllers(root)
    done = controller.wait(controller.submit(GenerationConfig()).job_id)
    directory = Path(done.config.output_dir)
    target = tmp_path / 'outside-dir'
    target.mkdir()
    keep = target / 'keep.png'
    keep.write_bytes(PNG)
    (directory / 'job.json').unlink()
    (directory / done.artifact.filename).unlink()
    directory.rmdir()
    directory.symlink_to(target)
    assert controller.forget_all(delete_files=True) == (1, 0, 0)
    assert keep.read_bytes() == PNG and target.is_dir()  # the link was never followed


# --------------------------------------------------------------- web service


def test_job_document_carries_image_bytes(runtime, services_fixture):
    service = services_fixture(runtime)
    try:
        done_id = service.submit(envelope()).payload['job_id']
        service.controller.wait(done_id)
        doc = service.job_response(done_id).payload
        assert doc['image_bytes'] == len(PNG)
        failed = WebGeneration(runtime, executor=explode)
        try:
            failed_id = failed.submit(envelope(prompt='no image')).payload['job_id']
            failed.controller.wait(failed_id)
            assert failed.job_response(failed_id).payload['image_bytes'] is None
        finally:
            assert failed.close()
    finally:
        assert service.close()


def test_forget_response_body_grammar_is_closed(runtime, services_fixture):
    service = services_fixture(runtime)
    try:
        for bad in (b'{"delete_files": "yes"}', b'{"delete_files": 1}', b'{"x": 1}',
                    b'[1]', b'{"delete_files": true, "extra": false}', b'"delete_files"'):
            job_id = service.submit(envelope()).payload['job_id']
            service.controller.wait(job_id)
            with pytest.raises(api.ApiError) as caught:
                service.forget_response(job_id, bad)
            assert (caught.value.status, caught.value.code) == (400, 'invalid_envelope')
            service.forget_response(job_id, b'{}')  # the entry stays forgetable
        assert service.list_response().payload['jobs'] == []
    finally:
        assert service.close()


def test_per_job_forget_delete_files_removes_the_directory(runtime, services_fixture):
    service = services_fixture(runtime)
    try:
        kept_id = service.submit(envelope(prompt='keep the file')).payload['job_id']
        kept = service.controller.wait(kept_id)
        gone_id = service.submit(envelope(prompt='delete the file')).payload['job_id']
        gone = service.controller.wait(gone_id)
        kept_path = Path(kept.config.output_dir) / kept.artifact.filename
        gone_path = Path(gone.config.output_dir) / gone.artifact.filename
        assert service.forget_response(kept_id, b'{}').payload == \
            {'schema': 'qwen_engine.web_jobs.v1', 'forgotten': kept_id, 'deleted_files': 0}
        assert kept_path.is_file()
        assert service.forget_response(gone_id, b'{"delete_files": true}').payload == \
            {'schema': 'qwen_engine.web_jobs.v1', 'forgotten': gone_id, 'deleted_files': 1}
        assert not gone_path.exists() and not Path(gone.config.output_dir).exists()
        assert service.list_response().payload['jobs'] == []
    finally:
        assert service.close()


def test_forget_response_refuses_live_and_unknown_jobs(runtime, services_fixture):
    service = services_fixture(runtime)
    try:
        gate = threading.Event()

        def gated(context):
            gate.wait(5)
            return save(context)

        live = WebGeneration(runtime, executor=gated)
        try:
            running = live.submit(envelope()).payload['job_id']
            while live.controller.get(running).state != 'running':
                live.controller.wait(running, timeout_s=0.05)
            with pytest.raises(api.ApiError) as caught:
                live.forget_response(running, b'{"delete_files": true}')
            assert (caught.value.status, caught.value.code) == (409, 'job_unavailable')
            gate.set()
            live.controller.wait(running)
        finally:
            assert live.close()
        with pytest.raises(api.ApiError) as caught:
            service.forget_response('a' * 32, b'{}')
        assert (caught.value.status, caught.value.code) == (409, 'job_unavailable')
    finally:
        assert service.close()


def test_forget_all_counts_and_survives_restart(runtime, services):
    gate = threading.Event()

    def gated(context):
        if context.config.prompt == 'stays':
            gate.wait(5)
        return save(context)

    blocking = services(runtime, executor=gated)
    done_id = blocking.submit(envelope(prompt='terminal')).payload['job_id']
    record = blocking.controller.wait(done_id)
    path = Path(record.config.output_dir) / record.artifact.filename
    assert path.is_file()
    running = blocking.submit(envelope(prompt='stays')).payload['job_id']
    while blocking.controller.get(running).state is not jobs.JobState.RUNNING:
        blocking.controller.wait(running, timeout_s=0.05)
    assert blocking.forget_all_response(b'{"delete_files": true}').payload == \
        {'schema': 'qwen_engine.web_jobs.v1', 'forgotten': 1, 'deleted_files': 1,
         'skipped_live': 1}
    assert not path.exists() and not Path(record.config.output_dir).exists()
    # the live job's own directory is untouched while it runs
    assert Path(blocking.controller.get(running).config.output_dir).is_dir()
    gate.set()
    assert blocking.controller.wait(running).terminal
    assert blocking.forget_all_response(b'{"delete_files": true}').payload == \
        {'schema': 'qwen_engine.web_jobs.v1', 'forgotten': 1, 'deleted_files': 1,
         'skipped_live': 0}
    assert blocking.list_response().payload['jobs'] == []
    assert blocking.controller.collect_ms(running) is None
    assert blocking.close()

    restarted = services(runtime)  # a fresh server on the same root
    assert restarted.list_response().payload['jobs'] == []
    with pytest.raises(api.ApiError) as caught:
        restarted.image(done_id)
    assert (caught.value.status, caught.value.code) == (404, 'image_unavailable')
    assert list(Path(runtime.output_dir).iterdir()) == []  # empty root, nothing orphaned
    again = restarted.submit(envelope(prompt='after the clear')).payload['job_id']
    assert restarted.controller.wait(again).state == jobs.JobState.SUCCEEDED


def test_history_full_refusal_names_the_new_delete_button(runtime, services_fixture):
    service = services_fixture(runtime)
    try:
        for index in range(32):  # the controller's default max_records
            job_id = service.submit(envelope(prompt=f'fill {index}')).payload['job_id']
            service.controller.wait(job_id)
        with pytest.raises(api.ApiError) as caught:
            service.submit(envelope(prompt='one too many'))
        assert caught.value.status == 409
        assert caught.value.code == 'job_submission_refused'
        assert caught.value.message == ('History is full — delete old entries on the '
                                        'History tab (Delete all history or Forget entry)')
        assert service.forget_all_response(b'{}').payload == \
            {'schema': 'qwen_engine.web_jobs.v1', 'forgotten': 32, 'deleted_files': 0,
             'skipped_live': 0}
        accepted = service.submit(envelope(prompt='room again')).payload['job_id']
        assert service.controller.wait(accepted).terminal
    finally:
        assert service.close()


# ---------------------------------------------------------------- the wire


def _response(raw: bytes, generation=None) -> bytes:
    handler = server._Handler.__new__(server._Handler)
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.connection = types.SimpleNamespace(getsockname=lambda: ('127.0.0.1', 8765))
    handler.server = types.SimpleNamespace(
        bound_ip='127.0.0.1', bound_port=8765, assets={},
        web_runtime=api.WebRuntime(), generation=generation)
    handler.close_connection = False
    handler.request_version = 'HTTP/1.1'
    handler.address_string = lambda: '127.0.0.1'
    handler.log_message = lambda *a, **k: None
    handler.handle_one_request()
    return handler.wfile.getvalue()


def _post(route: str, body: bytes, generation=None, *, ui='1', origin=ORIGIN,
          content_type='application/json') -> bytes:
    head = f'POST {route} HTTP/1.1\r\nHost: 127.0.0.1:8765\r\n'
    if origin is not None:
        head += f'Origin: {origin}\r\n'
    if ui is not None:
        head += f'X-Qwen-UI: {ui}\r\n'
    if content_type is not None:
        head += f'Content-Type: {content_type}\r\n'
    head += f'Content-Length: {len(body)}\r\n\r\n'
    return _response(head.encode() + body, generation)


def test_resolve_route_owns_forget_all_exclusively():
    assert server.resolve_route('POST', server.JOBS_FORGET_ALL_ROUTE) == \
        ('forget_all', '/api/v1/jobs/forget-all')
    for method in ('GET', 'HEAD', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'):
        with pytest.raises(api.ApiError) as caught:
            server.resolve_route(method, server.JOBS_FORGET_ALL_ROUTE)
        assert caught.value.status == 405
        assert caught.value.headers == (('Allow', 'POST'),)
    with pytest.raises(api.ApiError):
        server.resolve_route('POST', server.JOBS_FORGET_ALL_ROUTE + '?delete_files=true')
    # the per-job grammar is untouched and never grows a second meaning
    assert server.resolve_route('POST', '/api/v1/jobs/' + 'a' * 32 + '/forget') == \
        ('forget', 'a' * 32)


def test_forget_all_wire_speaks_the_same_csrf_and_envelope_rules(runtime, services_fixture):
    service = services_fixture(runtime)
    route = server.JOBS_FORGET_ALL_ROUTE
    try:
        out = _response(b'GET ' + route.encode() + b' HTTP/1.1\r\n'
                        b'Host: 127.0.0.1:8765\r\n\r\n', service)
        assert b' 405 ' in out.split(b'\r\n')[0] and b'Allow: POST' in out
        out = _post(route, b'{}', service, ui=None)
        assert b' 403 ' in out.split(b'\r\n')[0]
        assert b'ui_header_required' in out
        assert service.list_response().payload['jobs'] == []
        out = _post(route, b'{}', service, ui='0')
        assert b'ui_header_required' in out
        out = _post(route, b'{}', service, content_type='text/plain')
        assert b' 415 ' in out.split(b'\r\n')[0] and b'unsupported_media_type' in out
        out = _post(route, b'{"delete_files": "yes"}', service)
        assert b' 400 ' in out.split(b'\r\n')[0] and b'invalid_envelope' in out
        out = _post(route, b'{}', service, origin='http://evil.invalid')
        assert b' 403 ' in out.split(b'\r\n')[0] and b'foreign_origin_denied' in out
    finally:
        assert service.close()


def test_forget_all_wire_clears_and_reports_counts(runtime, services_fixture):
    service = services_fixture(runtime)
    try:
        done_id = service.submit(envelope()).payload['job_id']
        record = service.controller.wait(done_id)
        path = Path(record.config.output_dir) / record.artifact.filename
        out = _post(server.JOBS_FORGET_ALL_ROUTE, b'{"delete_files": true}', service)
        assert b' 200 ' in out.split(b'\r\n')[0]
        payload = json.loads(out.split(b'\r\n\r\n', 1)[1])
        assert payload == {'schema': 'qwen_engine.web_jobs.v1', 'forgotten': 1,
                           'deleted_files': 1, 'skipped_live': 0}
        assert not path.exists() and not Path(record.config.output_dir).exists()
        assert service.list_response().payload['jobs'] == []
    finally:
        assert service.close()


def test_per_job_forget_wire_accepts_the_delete_files_body(runtime, services_fixture):
    service = services_fixture(runtime)
    try:
        gone_id = service.submit(envelope(prompt='wire delete')).payload['job_id']
        gone = service.controller.wait(gone_id)
        kept_id = service.submit(envelope(prompt='wire keep')).payload['job_id']
        kept = service.controller.wait(kept_id)
        out = _post(f'/api/v1/jobs/{gone_id}/forget', b'{"delete_files": true}', service)
        payload = json.loads(out.split(b'\r\n\r\n', 1)[1])
        assert payload == {'schema': 'qwen_engine.web_jobs.v1', 'forgotten': gone_id,
                           'deleted_files': 1}
        assert not Path(gone.config.output_dir).exists()
        out = _post(f'/api/v1/jobs/{kept_id}/forget', b'{}', service)
        payload = json.loads(out.split(b'\r\n\r\n', 1)[1])
        assert payload == {'schema': 'qwen_engine.web_jobs.v1', 'forgotten': kept_id,
                           'deleted_files': 0}
        assert (Path(kept.config.output_dir) / kept.artifact.filename).is_file()
        # cancel keeps its empty-object grammar
        out = _post(f'/api/v1/jobs/{kept_id}/cancel', b'{"delete_files": true}', service)
        assert b' 400 ' in out.split(b'\r\n')[0] and b'invalid_envelope' in out
    finally:
        assert service.close()
