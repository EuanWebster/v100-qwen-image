"""CPU tests for scripts/web_e2e_smoke.py against an in-process fake of the
REAL wire contract: the fake validates every POST body through the server's
own web/api.parse_request_json + build_config (plus generation.py's submit
gates), so it speaks the exact grammar — decimal-STRING seed, editable-field
whitelist, typed config validation — and can never accept what the real
server refuses.  The fake speaks real HTTP/1.1 bytes over an AF_UNIX
socketpair driven by the stdlib BaseHTTPRequestHandler — no listening
sockets exist anywhere (the guarded runner denies network-socket
construction), and no GPU, models or network leave the process."""
from __future__ import annotations

import hashlib
import http.client
import http.server
import importlib.util
import io
import json
import re
import socket
import threading
from pathlib import Path

import pytest

from qwen_engine.web import api as web_api

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "web_e2e_smoke.py"

_spec = importlib.util.spec_from_file_location("web_e2e_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(smoke)


def _png(width=2, height=2, mode="RGBA"):
    from PIL import Image
    color = (200, 30, 30, 120) if mode == "RGBA" else (200, 30, 30)
    image = Image.new(mode, (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _State:
    """The fake server's shared state (what web/server.py calls its runtime)."""

    def __init__(self):
        self.thumb = _png()
        self.rgba_png = _png(mode="RGBA")
        self.rgb_png = _png(mode="RGB")
        self.jobs, self.images = {}, {}
        self.counter = 0
        self.hangs = False
        self.fails = False
        self.uploaded_ids: set[str] = set()
        self.references_expected = {smoke.PROMPTS["edit-1ref"]: 1,
                                    smoke.PROMPTS["edit-2ref"]: 2}
        # Server-controlled paths exactly as startup injects them (the fake
        # never opens them; they only complete the merged config).
        self.runtime = web_api.WebRuntime(
            output_dir="/srv/qwen/out", cache_dir="/srv/qwen/cache",
            log_dir="/srv/qwen/log", benchmarks_dir="/srv/qwen/bench",
            models_dir="/srv/qwen/models",
            manifest_path="/srv/qwen/manifest.json",
            references_dir="/srv/qwen/references", generation_enabled=True)


class _Handler(http.server.BaseHTTPRequestHandler):
    """Mirrors web/server.py's client-visible rules tightly enough that a
    client breaking any of them fails the smoke, not the fixture."""

    server_version = "FakeQwen/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence
        pass

    def _send(self, status, payload: bytes, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status, code):
        self._send(status, json.dumps({"schema": "qwen_engine.web_error.v1",
                                       "code": code, "message": code}).encode())

    def _post_ok(self, *, upload=False):
        if self.headers.get("X-Qwen-UI") != "1":
            self._error(403, "ui_header_required")
            return False
        want = "application/octet-stream" if upload else "application/json"
        if self.headers.get("Content-Type") != want:
            self._error(415, "unsupported_media_type")
            return False
        return True

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, b"<!doctype html><html><body>qwen</body></html>",
                       "text/html; charset=utf-8")
            return
        match = re.fullmatch(r"/api/v1/references/([0-9a-f]{64})/thumb", self.path)
        if match:
            self._send(200, self.server.thumb, "image/png")
            return
        match = re.fullmatch(r"/api/v1/jobs/([0-9a-f]{32})/image", self.path)
        if match:
            self._send(200, self.server.images.get(match.group(1), b""),
                       "image/png")
            return
        match = re.fullmatch(r"/api/v1/jobs/([0-9a-f]{32})", self.path)
        if match:
            job_id = match.group(1)
            job = self.server.jobs.get(job_id)
            if job is None:
                self._error(404, "job_unavailable")
                return
            job["polls"] += 1
            terminal = (not self.server.hangs) and job["polls"] >= 3
            state = ("succeeded" if terminal and not self.server.fails
                     else "failed" if terminal else "running")
            if state == "failed":
                job["error_code"] = "execution_failed"
            self._send(200, json.dumps({
                "schema": "qwen_engine.web_job.v1", "job_id": job_id,
                "state": state, "phase": "denoise", "cancel_requested": False,
                "config": {}, "elapsed_ms": 5.0,
                "phase_ms": {"load": 1.0, "denoise": 2.0},
                "error_code": job["error_code"], "collect_ms": None,
                "progress": {"completed_steps": 8, "total_steps": 8},
                "image_url": (f"/api/v1/jobs/{job_id}/image"
                              if state == "succeeded" else None),
                "image_filename": "x.png" if state == "succeeded" else None,
            }).encode())
            return
        self._error(404, "not_found")

    def do_POST(self):
        upload = self.path == "/api/v1/references"
        if not self._post_ok(upload=upload):
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if upload:
            reference_id = hashlib.sha256(body).hexdigest()
            self.server.uploaded_ids.add(reference_id)
            self._send(201, json.dumps({"schema": "qwen_engine.web_reference.v1",
                                        "id": reference_id, "width": 2,
                                        "height": 2, "has_alpha": True,
                                        "bytes": len(body)}).encode())
            return
        # THE real wire grammar: the same strict parser and envelope/config
        # validator web/server.py runs (web/generation.py submit), so the
        # fake can never accept what the real server refuses — decimal-
        # STRING seed included.
        try:
            config = web_api.build_config(self.server.runtime,
                                          web_api.parse_request_json(body))
        except web_api.ApiError as exc:
            self._send(exc.status, json.dumps(exc.to_payload()).encode())
            return
        if self.path == "/api/v1/plan":
            self._send(200, json.dumps({"schema": "qwen_engine.web_plan.v1",
                                        "config": web_api.editable_config_dict(
                                            config)}).encode())
            return
        if self.path == "/api/v1/jobs":
            if (config.references
                    and not set(config.references) <= self.server.uploaded_ids):
                # the real submit reads a hash-verified blob per id first
                self._error(422, "reference_unavailable")
                return
            expected_refs = self.server.references_expected.get(config.prompt, 0)
            if len(config.references) != expected_refs:
                # the fixture knows the smoke prompts: edits must carry
                # exactly 1 / 2 uploaded ids, non-edits none
                self._error(422, "invalid_config")
                return
            from qwen_engine.adapters.qwen21_diffusers import _validate_policy
            from qwen_engine.config import ConfigError
            try:
                _validate_policy(config)
            except ConfigError as exc:
                editable_policy = ("quant", "offload", "attention_backend",
                                   "steps", "true_cfg_scale", "recipe")
                self._send(422, json.dumps({
                    "schema": "qwen_engine.web_error.v1",
                    "code": "unsupported_generation_config",
                    "message": "this configuration is not supported by the "
                               "current adapter",
                    "fields": [{"field": e.field, "message": e.message}
                               for e in exc.errors
                               if e.field in editable_policy]}).encode())
                return
            if config.latent_tokens > 4096 and not config.vae_tiled_decode:
                self._error(422, "tiled_decode_required")
                return
            self.server.counter += 1
            job_id = f"{self.server.counter:032x}"
            self.server.jobs[job_id] = {"polls": 0, "error_code": None}
            self.server.images[job_id] = (self.server.rgba_png
                                          if config.transparent_background
                                          else self.server.rgb_png)
            self._send(202, json.dumps({"schema": "qwen_engine.web_job.v1",
                                        "job_id": job_id, "state": "queued",
                                        "error_code": None}).encode())
            return
        self._error(404, "not_found")


class _PipeClient(smoke.ApiClient):
    """ApiClient whose HTTPConnection is bridged onto a fresh AF_UNIX
    socketpair; the server side is served synchronously in a thread, exactly
    one HTTP conversation per connection (what the smoke client does)."""

    def __init__(self, state: _State):
        super().__init__("127.0.0.1", 1)
        self.state = state

    def _connect(self):
        client = _BridgedConnection()
        client_sock, server_side = socket.socketpair(socket.AF_UNIX,
                                                     socket.SOCK_STREAM)
        client.attach(client_sock)
        thread = threading.Thread(
            target=_Handler, args=(server_side, ("pipe", 0), self.state),
            daemon=True)
        thread.start()
        self._server_thread = thread  # last one, joined implicitly at exit
        return client


class _BridgedConnection(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("127.0.0.1", 1, timeout=10)
        self._sock = None

    def attach(self, sock):
        self._sock = sock

    def connect(self):
        self.sock = self._sock


@pytest.fixture
def fake_state():
    return _State()


def test_happy_path_reports_every_job_and_is_prompt_free(fake_state, tmp_path,
                                                         monkeypatch):
    reference = tmp_path / "ref.png"
    reference.write_bytes(_png(mode="RGB"))
    monkeypatch.setattr(smoke, "ApiClient",
                        lambda host, port, *, timeout=30.0: _PipeClient(fake_state))
    rc = smoke.main(["--base-url", "http://127.0.0.1:8765",
                     "--reference", str(reference),
                     "--jobs", ",".join(smoke.JOB_KINDS),
                     "--poll-interval", "0.005", "--job-timeout", "10",
                     "--report", str(tmp_path / "report.json")])
    assert rc == 0
    text = (tmp_path / "report.json").read_text()
    report = json.loads(text)
    assert report["index_ok"] and report["plan_ok"] and report["ok"]
    assert report["reference"]["id"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert report["reference"]["thumb_ok"] is True
    assert [entry["kind"] for entry in report["jobs"]] == list(smoke.JOB_KINDS)
    for entry in report["jobs"]:
        assert entry["state"] == "succeeded" and entry["error_code"] is None
        assert entry["image_format"] == "png" and entry["image_bytes"] > 0
        assert entry["image_sha256"] == hashlib.sha256(
            fake_state.images[entry["job_id"]]).hexdigest()
        assert entry["phase_ms"] == {"load": 1.0, "denoise": 2.0}
    by_kind = {entry["kind"]: entry for entry in report["jobs"]}
    assert by_kind["transparent"]["alpha"] is True  # Pillow present in env
    # prompt-free: no prompt text and no "prompt" key anywhere in the report
    for prompt in smoke.PROMPTS.values():
        assert prompt not in text
    assert "prompt" not in text


def test_loopback_guard_refuses_before_connecting():
    for bad in ("https://127.0.0.1:8765", "http://example.com:80",
                "http://127.0.0.1:8765/api", "http://user@127.0.0.1:8765",
                "http://127.0.0.1:99999", "http://10.0.0.5:80"):
        with pytest.raises(ValueError):
            smoke.parse_base_url(bad)
    assert smoke.parse_base_url("http://[::1]:8765") == ("::1", 8765)
    assert smoke.parse_base_url("http://localhost:80") == ("localhost", 80)


def test_nonloopback_base_url_exits_two_without_request():
    assert smoke.main(["--base-url", "http://example.com:80"]) == 2


def test_edit_jobs_require_reference():
    assert smoke.main(["--base-url", "http://127.0.0.1:1",
                       "--jobs", "edit-1ref"]) == 2


def test_job_that_never_terminates_is_reported_timeout(fake_state):
    fake_state.hangs = True
    report, ok = smoke.run_smoke(_PipeClient(fake_state), ("t2i-turbo",), None,
                                 job_timeout=0.05, poll_s=0.005)
    assert ok is False and report["ok"] is False
    assert report["jobs"][0]["state"] == "timeout"


def test_failed_job_reports_error_code(fake_state):
    fake_state.fails = True
    report, ok = smoke.run_smoke(_PipeClient(fake_state), ("t2i-turbo",), None,
                                 job_timeout=5, poll_s=0.005)
    assert ok is False
    assert report["jobs"][0]["state"] == "failed"
    assert report["jobs"][0]["error_code"] == "execution_failed"


def test_transparent_without_alpha_channel_fails_verification(fake_state):
    fake_state.rgba_png = _png(mode="RGB")  # server lies about alpha
    report, ok = smoke.run_smoke(_PipeClient(fake_state), ("transparent",), None,
                                 job_timeout=5, poll_s=0.005)
    assert ok is False
    assert report["jobs"][0]["alpha"] is False


def test_missing_reference_ids_are_refused_by_the_wire(fake_state):
    """The fixture enforces references exactly like the real submit path;
    an edit config without ids must NOT be accepted (envelope guard)."""
    body = smoke.envelope({"prompt": smoke.PROMPTS["edit-1ref"], "seed": "42"})
    status, _ = _PipeClient(fake_state).request_json("POST", "/api/v1/jobs",
                                                     body=body)
    assert status == 422


def test_numeric_seed_is_refused_by_the_real_wire_grammar(fake_state):
    """THE regression: seed is a decimal STRING on the wire (web/api.py
    _SEED_WIRE_RE); a JSON number is a 400 invalid_seed, never accepted."""
    status, payload = _PipeClient(fake_state).request_json(
        "POST", "/api/v1/jobs",
        body=smoke.envelope({"prompt": smoke.PROMPTS["t2i-turbo"],
                             "recipe": "turbo", "steps": smoke.TURBO_STEPS,
                             "seed": 42}))
    assert status == 400 and payload["code"] == "invalid_seed"


def test_transparent_without_alpha_format_is_refused(fake_state):
    """The second script-vs-grammar gap: transparent_background with the
    default (jpg) output_format is a 422 field error on the real submit."""
    status, payload = _PipeClient(fake_state).request_json(
        "POST", "/api/v1/jobs",
        body=smoke.envelope({"prompt": "a probe", "seed": "1",
                             "transparent_background": True}))
    assert status == 422 and payload["code"] == "invalid_config"
    assert any(f["field"] == "output_format" for f in payload["fields"])


def test_runtime_path_field_on_the_wire_is_refused(fake_state):
    """output_dir is server-controlled: the real build_config answers
    400 unknown_config_field — refused, never silently stripped."""
    status, payload = _PipeClient(fake_state).request_json(
        "POST", "/api/v1/jobs",
        body=smoke.envelope({"prompt": "a probe", "seed": "42",
                             "output_dir": "/tmp/pwn"}))
    assert status == 400 and payload["code"] == "unknown_config_field"
