"""W3 lifecycle/API boundaries without sockets, weights or ML imports."""
from __future__ import annotations

import base64
import gc
import json
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_engine import cli
from qwen_engine.core import GenerationResult
from qwen_engine.errors import GenerationCancelled
from qwen_engine.jobs import JobState
from qwen_engine.web import api, server
from qwen_engine.web.generation import WebGeneration

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j2ioAAAAASUVORK5CYII=')


def envelope(**values):
    return json.dumps({'schema': api.WEB_PLAN_REQUEST_SCHEMA,
                       'config': {'prompt': 'red cube', 'seed': '18446744073709551615', **values}}).encode()


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    return api.WebRuntime(output_dir=str(root), cache_dir=str(tmp_path / 'cache'),
                          models_dir=str(tmp_path / 'models'), manifest_path=str(tmp_path / 'manifest.json'),
                          manifest_summary={'configured': True, 'valid': True, 'schema_version': 1,
                                            'artifact_count': 24, 'verification': 'not_performed'},
                          generation_enabled=True)


def save(context):
    for phase in ('load', 'encode', 'denoise', 'decode', 'save'):
        context.report_phase(phase)
    path = Path(context.config.output_dir) / 'result.png'
    path.write_bytes(PNG)
    return GenerationResult(str(path))


def test_enabled_status_means_submission_only_not_verified_memory(runtime):
    status = api.status_response(runtime).payload
    assert status['generation_available'] is True
    assert status['generation_blockers'] == []
    assert status['gates']['calibration']['state'] == 'unknown'
    assert status['gates']['hardware']['state'] == 'unknown'
    assert status['manifest']['verification'] == 'not_performed'


@pytest.mark.parametrize("performance", [False, True])
def test_real_core_executor_reports_timings_closes_and_verifies_assets(runtime, monkeypatch, performance):
    events = []

    class Adapter:
        def set_progress_callback(self, callback): self.progress = callback
        def set_cancel_event(self, event):
            assert isinstance(event, threading.Event)
        def load(self, config, assets):
            events.append(('load', assets))
        def encode(self, config): return 'conditioning'
        def denoise(self, config, conditioning):
            self.progress(1, config.steps)
            self.progress(config.steps, config.steps)
            return 'latents'
        def decode(self, latents): return PNG
        def save(self, image, config):
            path = Path(config.output_dir) / 'real-core.png'
            path.write_bytes(image)
            return path
        def close(self): events.append('closed')

    def verify(config):
        events.append('verified')
        return SimpleNamespace(resolved={'weights': ()})

    def factory(*, trust_unbounded_host):
        assert trust_unbounded_host is True
        events.append('factory')
        return Adapter()

    monkeypatch.setattr(cli, '_verify_assets', verify)
    monkeypatch.setattr(cli, '_make_adapter', factory)
    service = WebGeneration(runtime, trust_unbounded_host=True, performance=performance)
    try:
        submitted = service.submit(envelope()).payload
        record = service.controller.wait(submitted['job_id'])
        doc = service.job_response(record.job_id).payload
        assert record.state == JobState.SUCCEEDED
        assert events == ['verified', 'factory', ('load', {'weights': ()}), 'closed']
        assert set(doc['phase_ms']) == {'load', 'encode', 'denoise', 'decode', 'save'}
        assert all(value >= 0 for value in doc['phase_ms'].values())
        assert doc['elapsed_ms'] >= sum(doc['phase_ms'].values())
        assert doc['config']['seed'] == '18446744073709551615'
        assert doc['progress'] == {'completed_steps': 40, 'total_steps': 40}
        assert str(Path(runtime.output_dir).parent) not in json.dumps(doc)
        assert service.image(record.job_id) == PNG
        report = Path(record.config.output_dir) / "performance.json"
        assert report.exists() is performance
        if performance:
            metrics = json.loads(report.read_text())
            assert metrics["status"] == "succeeded"
            assert metrics["stages"][-1]["stage"] == "total_job"
            assert 'red cube' not in report.read_text()
    finally:
        assert service.close()


@pytest.mark.parametrize("performance", [False, True])
def test_assets_failure_never_creates_adapter_and_error_is_private(runtime, monkeypatch, performance):
    def fail(config): raise RuntimeError('/private/model/path secret detail')
    monkeypatch.setattr(cli, '_verify_assets', fail)
    monkeypatch.setattr(cli, '_make_adapter', lambda **kw: pytest.fail('adapter loaded'))
    service = WebGeneration(runtime, performance=performance)
    try:
        job_id = service.submit(envelope()).payload['job_id']
        service.controller.wait(job_id)
        doc = service.job_response(job_id).payload
        assert doc['state'] == 'failed' and doc['error_code'] == 'execution_failed'
        assert '/private' not in json.dumps(doc) and 'secret detail' not in json.dumps(doc)
        assert doc['image_url'] is None
        report = Path(service.controller.get(job_id).config.output_dir) / "performance.json"
        assert report.exists() is performance
        if performance:
            assert json.loads(report.read_text())["status"] == "failed"
            assert 'secret detail' not in report.read_text()
    finally:
        assert service.close()


def test_queued_and_running_cancel_keep_the_worker_serial(runtime):
    started, release = threading.Event(), threading.Event()
    active = []
    def blocking(context):
        active.append(context.job_id)
        context.report_phase('load')
        started.set()
        assert release.wait(5)
        context.check_cancelled()
        return save(context)
    service = WebGeneration(runtime, executor=blocking)
    try:
        first = service.submit(envelope()).payload['job_id']
        assert started.wait(5)
        second = service.submit(envelope()).payload['job_id']
        assert service.job_response(second, action='cancel').payload['state'] == 'cancelled'
        assert service.job_response(first, action='cancel').payload['state'] == 'cancelling'
        assert active == [first]
        release.set()
        assert service.controller.wait(first).state == JobState.CANCELLED
        assert service.job_response(first).payload['image_url'] is None
    finally:
        release.set()
        assert service.close()


@pytest.mark.parametrize('change', ['content', 'symlink', 'directory'])
def test_image_refuses_changed_artifacts(runtime, tmp_path, change):
    service = WebGeneration(runtime, executor=save)
    try:
        job_id = service.submit(envelope()).payload['job_id']
        record = service.controller.wait(job_id)
        path = Path(record.config.output_dir) / record.artifact.filename
        if change == 'content':
            path.write_bytes(b'x' * len(PNG))
        elif change == 'symlink':
            outside = tmp_path / 'outside.png'
            outside.write_bytes(PNG)
            path.unlink()
            path.symlink_to(outside)
        else:
            directory = path.parent
            directory.rename(directory.with_name(directory.name + '-moved'))
            directory.mkdir()
            path.write_bytes(PNG)
        with pytest.raises(api.ApiError) as caught:
            service.image(job_id)
        assert caught.value.code == 'image_unavailable'
    finally:
        assert service.close()


def test_forget_preserves_files_and_list_is_bounded_ids_only(runtime):
    service = WebGeneration(runtime, executor=save)
    try:
        job_id = service.submit(envelope(seed=None)).payload['job_id']
        record = service.controller.wait(job_id)
        assert isinstance(record.config.seed, int)
        assert service.list_response().payload['jobs'] == [job_id]
        path = Path(record.config.output_dir) / record.artifact.filename
        service.job_response(job_id, action='forget')
        assert path.read_bytes() == PNG
        assert service.list_response().payload['jobs'] == []
        with pytest.raises(api.ApiError): service.image(job_id)
    finally:
        assert service.close()


def test_shutdown_cancels_live_and_queued_jobs_and_releases_executor(runtime):
    started, released = threading.Event(), threading.Event()
    def execute(context):
        started.set()
        try:
            assert context._cancel.wait(5)
            context.check_cancelled()
        finally:
            released.set()
    service = WebGeneration(runtime, executor=execute)
    first = service.submit(envelope()).payload['job_id']
    assert started.wait(5)
    second = service.submit(envelope()).payload['job_id']
    assert service.close()
    assert released.is_set()
    assert service.controller.get(first).state == JobState.CANCELLED
    assert service.controller.get(second).state == JobState.CANCELLED
    with pytest.raises(api.ApiError): service.submit(envelope())


def test_partial_sampler_progress_is_visible_and_must_be_monotonic(runtime):
    ready, release = threading.Event(), threading.Event()
    def execute(context):
        service.report_steps(context.job_id, 2, context.config.steps)
        ready.set()
        assert release.wait(5)
        return save(context)
    service = WebGeneration(runtime, executor=execute)
    try:
        job_id = service.submit(envelope()).payload['job_id']
        assert ready.wait(5)
        assert service.job_response(job_id).payload['progress'] == {'completed_steps': 2, 'total_steps': 40}
        for completed, total in [(1, 40), (41, 40), (True, 40), (3, 41)]:
            with pytest.raises(ValueError): service.report_steps(job_id, completed, total)
        assert service.job_response(job_id).payload['progress']['completed_steps'] == 2
    finally:
        release.set()
        assert service.close()


def test_incomplete_generation_runtime_refuses_before_socket_or_writes(monkeypatch):
    def fail(*args): pytest.fail('server bound before runtime validation')
    monkeypatch.setattr(server, '_Server', fail)
    assert server.main(['--enable-generation']) == 2


@pytest.mark.parametrize('values,code', [
    ({'output_dir': '/tmp/injected'}, 'unknown_config_field'),
    ({'seed': 42}, 'invalid_seed'),
    ({'quant': 'q4'}, 'invalid_config'),
    ({'width': 2048, 'height': 2048}, 'tiled_decode_required'),
])
def test_bad_submissions_create_no_job_or_output(runtime, values, code):
    service = WebGeneration(runtime, executor=save)
    try:
        with pytest.raises(api.ApiError) as caught:
            service.submit(envelope(**values))
        assert caught.value.code == code
        assert not service.controller.snapshots()
        assert not list(Path(runtime.output_dir).iterdir())
    finally:
        assert service.close()


@pytest.mark.parametrize('values,field', [
    ({'recipe': 'turbo'}, 'steps'),
    ({'recipe': 'turbo', 'steps': 8, 'negative_prompt': 'blurry',
      'true_cfg_scale': 4.0}, 'true_cfg_scale'),
    ({'recipe': 'turbo', 'steps': 8, 'quant': 'q8'}, 'quant'),
])
def test_turbo_submissions_refuse_the_gated_combination_on_its_own_field(
        runtime, values, field):
    service = WebGeneration(runtime, executor=save)
    try:
        with pytest.raises(api.ApiError) as caught:
            service.submit(envelope(**values))
        # GenerationConfig itself enforces the Turbo gates, so the refusal
        # arrives from config validation (invalid_config) before the adapter
        # policy stage is reached.
        assert caught.value.code == 'invalid_config'
        # The field must be NAMED so the UI can point at the control; a job
        # is never enqueued for a combination the adapter will refuse.
        assert [f for f, _ in caught.value.field_errors] == [field]
        assert not service.controller.snapshots()
        assert not list(Path(runtime.output_dir).iterdir())
    finally:
        assert service.close()


def test_turbo_submission_is_accepted_with_the_gated_defaults(runtime):
    service = WebGeneration(runtime, executor=save)
    try:
        response = service.submit(envelope(recipe='turbo', steps=8))
        assert response.status == 202
        config = service.controller.snapshots()[0].config
        assert (config.recipe.value, config.steps, config.quant.value) == ('turbo', 8, 'fp16')
        # The public job document carries the recipe so History can label it.
        assert response.payload['config']['recipe'] == 'turbo'
    finally:
        assert service.close()


@pytest.mark.parametrize('method,suffix,kind', [
    ('GET', '', 'job'), ('GET', '/image', 'image'),
    ('POST', '/cancel', 'cancel'), ('POST', '/forget', 'forget'),
])
def test_exact_job_routes_and_method_policy(method, suffix, kind):
    job_id = 'a' * 32
    assert server.resolve_route(method, '/api/v1/jobs/' + job_id + suffix) == (kind, job_id)
    with pytest.raises(api.ApiError) as caught:
        server.resolve_route('DELETE', '/api/v1/jobs/' + job_id + suffix)
    assert caught.value.status == 405
    with pytest.raises(api.ApiError):
        server.resolve_route(method, '/api/v1/jobs/' + job_id + suffix + '?path=/etc/passwd')


class TensorOwner:
    """Weakref-able stand-in for an object owning CUDA tensors."""


@pytest.mark.parametrize('kind', ['cancelled', 'failed'])
def test_terminal_web_jobs_release_executor_frames_without_gc(runtime, monkeypatch, kind):
    refs = []

    class Adapter:
        def set_progress_callback(self, callback):
            self.progress = callback

        def set_cancel_event(self, event):
            self.event = event

        def load(self, config, assets):
            pass

        def encode(self, config):
            return 'conditioning'

        def denoise(self, config, conditioning):
            owner = TensorOwner()
            refs.append(weakref.ref(owner))
            if kind == 'cancelled':
                self.event.set()
                raise GenerationCancelled('fixture mid-denoise cancellation')
            raise RuntimeError('/private/adapter/failure secret detail')

        def decode(self, latents):
            raise AssertionError('unreachable')

        def save(self, image, config):
            raise AssertionError('unreachable')

        def close(self):
            pass

    monkeypatch.setattr(cli, '_verify_assets',
                        lambda config: SimpleNamespace(resolved={'weights': ()}))
    monkeypatch.setattr(cli, '_make_adapter', lambda **kw: Adapter())
    service = WebGeneration(runtime, trust_unbounded_host=True,
                            memory_release=lambda: None)
    try:
        gc.collect()
        gc.disable()
        job_id = service.submit(envelope()).payload['job_id']
        done = service.controller.wait(job_id)
        # the real serial worker already finished handling the exception, so
        # refcounting alone (cyclic GC off for the whole window) must have
        # freed the failing adapter frame and its tensor owner
        assert refs and all(ref() is None for ref in refs)
    finally:
        gc.enable()
    try:
        doc = service.job_response(job_id).payload
        if kind == 'cancelled':
            assert done.state == JobState.CANCELLED and doc['error_code'] is None
        else:
            assert done.state == JobState.FAILED and doc['error_code'] == 'execution_failed'
            assert 'secret detail' not in json.dumps(doc)
    finally:
        assert service.close()


def test_job_documents_expose_boundary_release_cost_only_for_non_success(runtime):
    calls = []
    collected = threading.Event()
    started, release = threading.Event(), threading.Event()

    def collect():
        calls.append(1)
        collected.set()

    def execute(context):
        if context.config.prompt == 'boom':
            raise RuntimeError('fixture web fault')
        if context.config.prompt == 'cancel':
            started.set()
            assert release.wait(5)
            context.check_cancelled()
        return save(context)

    def recorded(job_id):
        # the collect duration is recorded just after the terminal state
        # becomes visible; bounded wait, the serial worker writes it directly
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            value = service.controller.collect_ms(job_id)
            if value is not None:
                return value
            time.sleep(0.005)
        raise AssertionError('boundary release duration was never recorded')

    service = WebGeneration(runtime, executor=execute, memory_release=collect)
    try:
        failed = service.submit(envelope(prompt='boom')).payload['job_id']
        assert service.controller.wait(failed).state == JobState.FAILED
        assert collected.wait(5) and len(calls) == 1
        value = recorded(failed)
        doc = service.job_response(failed).payload
        assert doc['collect_ms'] == value and doc['error_code'] == 'execution_failed'
        collected.clear()

        cancelled = service.submit(envelope(prompt='cancel')).payload['job_id']
        assert started.wait(5)
        assert service.job_response(cancelled, action='cancel').payload['state'] == 'cancelling'
        release.set()
        assert service.controller.wait(cancelled).state == JobState.CANCELLED
        assert collected.wait(5) and len(calls) == 2
        value = recorded(cancelled)
        doc = service.job_response(cancelled).payload
        assert doc['collect_ms'] == value and doc['error_code'] is None
        assert doc['image_url'] is None
        collected.clear()

        good = service.submit(envelope(prompt='ok')).payload['job_id']
        assert service.controller.wait(good).state == JobState.SUCCEEDED
        doc = service.job_response(good).payload
        assert doc['collect_ms'] is None and doc['image_url'] is not None
        assert len(calls) == 2
    finally:
        release.set()
        assert service.close()
