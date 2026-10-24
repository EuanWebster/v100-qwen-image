"""Opt-in serial web execution over the same verified engine as the CLI.

No ML imports at startup. Paths, host attestation and executor are operator
owned; the browser supplies only the shared editable config envelope.
"""
from __future__ import annotations

import json
import secrets
import sys
import threading
import time
import traceback
from dataclasses import replace

from qwen_engine.config import ConfigError
from qwen_engine.core import generate
from qwen_engine.errors import GenerationCancelled
from qwen_engine.jobs import JobCancelled, JobController, JobError
from qwen_engine.log import PhaseLogger
from qwen_engine.web.api import (
    ApiError, ApiResponse, WebRuntime, _blockers, build_config,
    editable_config_dict, parse_request_json,
)

JOB_SCHEMA = "qwen_engine.web_job.v1"
JOBS_SCHEMA = "qwen_engine.web_jobs.v1"


class WebGeneration:
    def __init__(self, runtime: WebRuntime, *, trust_unbounded_host=False, executor=None):
        if _blockers(runtime):
            raise ValueError("generation runtime is incomplete or disabled")
        self.runtime = runtime
        self.trust_unbounded_host = trust_unbounded_host
        self._lock = threading.Lock()
        self._metrics = {}
        self._executor = executor or self._execute
        self.controller = JobController(runtime.output_dir, self._run)

    def _execute(self, context):
        # Identical asset verification, adapter and host policy to --run.
        from qwen_engine.cli import _make_adapter, _verify_assets

        context.check_cancelled()
        assets = _verify_assets(context.config)
        context.check_cancelled()
        adapter = _make_adapter(trust_unbounded_host=self.trust_unbounded_host)

        def sink(line):
            record = json.loads(line)
            if record.get("event") == "phase_end" and record.get("elapsed_ms") is not None:
                with self._lock:
                    self._metrics[context.job_id]["phase_ms"][record["phase"]] = record["elapsed_ms"]

        return generate(context.config, adapter, assets.resolved, PhaseLogger(sink=sink),
                        cancel_event=context._cancel, phase_callback=context.report_phase)

    def _run(self, context):
        started = time.monotonic()
        with self._lock:
            self._metrics[context.job_id] = {"started": started, "elapsed_ms": None, "phase_ms": {}}
        try:
            return self._executor(context)
        except (JobCancelled, GenerationCancelled):
            raise
        except Exception:
            print(f"qwen-web job {context.job_id} failed:", file=sys.stderr)
            traceback.print_exc()
            raise
        finally:
            with self._lock:
                self._metrics[context.job_id]["elapsed_ms"] = (time.monotonic() - started) * 1000

    def public_job(self, record):
        with self._lock:
            metric = self._metrics.get(record.job_id)
            elapsed = None
            phases = {}
            if metric:
                elapsed = metric["elapsed_ms"]
                if elapsed is None:
                    elapsed = (time.monotonic() - metric["started"]) * 1000
                phases = dict(metric["phase_ms"])
        return {
            "schema": JOB_SCHEMA, "job_id": record.job_id,
            "state": record.state.value, "phase": record.phase,
            "cancel_requested": record.cancel_requested,
            "config": editable_config_dict(record.config),
            "elapsed_ms": elapsed, "phase_ms": phases,
            "error_code": record.error_code,
            "image_url": (f"/api/v1/jobs/{record.job_id}/image" if record.artifact else None),
        }

    def submit(self, body):
        config = build_config(self.runtime, parse_request_json(body))
        from qwen_engine.adapters.qwen21_diffusers import _validate_policy

        try:
            _validate_policy(config)
        except ConfigError as exc:
            # Path values never cross the wire. Editable policy errors are safe.
            fields = tuple((e.field, e.message) for e in exc.errors
                           if e.field in ("quant", "offload", "attention_backend"))
            raise ApiError(422, "unsupported_generation_config",
                           "this configuration is not supported by the current adapter", fields) from None
        if config.latent_tokens > 4096 and not config.vae_tiled_decode:
            raise ApiError(422, "tiled_decode_required", "enable tiled VAE decode for images above 4096 latent tokens")
        # Choose a real seed once, before enqueueing, so History can reproduce it.
        config = replace(config, output_dir=None,
                         seed=config.seed if config.seed is not None else secrets.randbits(64))
        try:
            record = self.controller.submit(config)
        except (JobError, OSError):
            raise ApiError(409, "job_submission_refused", "job queue/history is full or its output storage is unavailable") from None
        return ApiResponse(202, self.public_job(record))

    def list_response(self):
        return ApiResponse(200, {"schema": JOBS_SCHEMA,
                                 "jobs": [r.job_id for r in self.controller.snapshots()]})

    def job_response(self, job_id, *, action=None):
        try:
            if action == "cancel":
                record = self.controller.cancel(job_id)
            elif action == "forget":
                self.controller.forget(job_id)
                with self._lock:
                    self._metrics.pop(job_id, None)
                return ApiResponse(200, {"schema": JOBS_SCHEMA, "forgotten": job_id})
            else:
                record = self.controller.get(job_id)
            return ApiResponse(200, self.public_job(record))
        except (JobError, OSError):
            raise ApiError(409 if action == "forget" else 404, "job_unavailable",
                           "job is unknown or cannot perform this action") from None

    def image(self, job_id):
        try:
            data = self.controller.read_artifact(job_id)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise JobError("not a PNG")
            return data
        except (JobError, OSError):
            raise ApiError(404, "image_unavailable", "completed image is unavailable or has changed") from None

    def close(self):
        return self.controller.close(timeout_s=60)
