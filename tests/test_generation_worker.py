"""Generation child protocol/refusal tests with no ML/assets or backend bypass flags."""
import json
import signal
from pathlib import Path

import pytest

from qwen_engine import generation_worker as w
from qwen_engine.config import GenerationConfig, ConfigError
from qwen_engine.core import GenerationResult
from qwen_engine.errors import GenerationCancelled, MemoryGuardError, HardwareAbsentError
from qwen_engine.manifest import ManifestError


@pytest.fixture
def worker_inputs(tmp_path):
    path = tmp_path / "request.json"
    response = tmp_path / "response.json"
    doc = dict(schema=w.REQUEST_SCHEMA, request_id="a" * 32, config=GenerationConfig(output_dir=tmp_path).to_dict())
    path.write_text(json.dumps(doc))
    return path, response, doc


@pytest.mark.parametrize("field,value", [("schema", "wrong"), ("request_id", "short"),
    ("request_id", 1), ("config", []), ("extra", 1)])
def test_invalid_request_envelope_or_config_before_backend(worker_inputs, field, value):
    path, response, doc = worker_inputs
    doc[field] = value
    path.write_text(json.dumps(doc))
    with pytest.raises(ConfigError):
        w.parse_request(path)
    assert not response.exists()


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b"[" * 2000 + b"0" + b"]" * 2000,
                                    b"x" * (w.MAX_REQUEST_BYTES + 1)])
def test_bounded_bad_json_is_typed_config_refusal(worker_inputs, raw):
    path, response, doc = worker_inputs
    path.write_bytes(raw)
    with pytest.raises(ConfigError):
        w.parse_request(path)


@pytest.mark.parametrize("error,code", [(ConfigError("private config"), 2), (ManifestError("private assets"), 4),
    (MemoryGuardError("private memory"), 3), (HardwareAbsentError("private hardware"), 5),
    (RuntimeError("private unexpected"), 1)])
def test_worker_failures_emit_only_fixed_codes_and_restore_signal(worker_inputs, monkeypatch, error, code):
    path, response, doc = worker_inputs
    monkeypatch.setattr(w, "install_offline_guard", lambda: None)
    def fail(*args):
        raise error
    monkeypatch.setattr(w, "run_generation", fail)
    previous = signal.getsignal(signal.SIGTERM)
    assert w.execute(path, response) == code
    actual = json.loads(response.read_text())
    assert actual["status"] == "failed" and actual["exit_code"] == code
    assert actual["error_code"] == w.ERROR_CODES[code] and actual["output_path"] is None
    assert "private" not in response.read_text() and signal.getsignal(signal.SIGTERM) == previous


def test_worker_refuses_unrequested_cancellation_ack(worker_inputs, monkeypatch):
    path, response, doc = worker_inputs
    monkeypatch.setattr(w, "install_offline_guard", lambda: None)
    monkeypatch.setattr(w, "run_generation", lambda *a: (_ for _ in ()).throw(GenerationCancelled()))
    assert w.execute(path, response) == 1
    assert json.loads(response.read_text())["error_code"] == "invalid_cancellation_acknowledgment"


def test_worker_accepts_actual_signal_cancellation_ack(worker_inputs, monkeypatch):
    path, response, doc = worker_inputs
    monkeypatch.setattr(w, "install_offline_guard", lambda: None)
    def cancelled(config, event, report):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert event.is_set()
        raise GenerationCancelled()
    monkeypatch.setattr(w, "run_generation", cancelled)
    assert w.execute(path, response) == 130
    assert json.loads(response.read_text())["status"] == "cancelled"


def test_response_collision_preserved_before_backend(worker_inputs, monkeypatch):
    path, response, doc = worker_inputs
    response.write_bytes(b"retained")
    monkeypatch.setattr(w, "install_offline_guard", lambda: None)
    monkeypatch.setattr(w, "run_generation", lambda *a: pytest.fail("backend after existing result"))
    previous = signal.getsignal(signal.SIGTERM)
    assert w.main(["--request", str(path), "--response", str(response)]) == 1
    assert response.read_bytes() == b"retained" and signal.getsignal(signal.SIGTERM) == previous


def test_valid_phase_sequence_and_result_envelope(worker_inputs, monkeypatch, capsys):
    path, response, doc = worker_inputs
    monkeypatch.setattr(w, "install_offline_guard", lambda: None)
    def fixture(config, event, report):
        for phase in w.PHASES:
            report(phase)
        return GenerationResult(str(Path(config.output_dir) / "fixture.bin"))
    monkeypatch.setattr(w, "run_generation", fixture)
    assert w.execute(path, response) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["phase"] for line in lines] == list(w.PHASES)
    assert [line["sequence"] for line in lines] == list(range(5))
    assert all(line["request_id"] == doc["request_id"] for line in lines)
    assert json.loads(response.read_text())["status"] == "succeeded"


def test_interrupt_restores_signal_and_does_not_publish_success(worker_inputs, monkeypatch):
    path, response, doc = worker_inputs
    monkeypatch.setattr(w, "install_offline_guard", lambda: None)
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(w, "run_generation", interrupted)
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        w.execute(path, response)
    assert not response.exists() and signal.getsignal(signal.SIGTERM) == previous
