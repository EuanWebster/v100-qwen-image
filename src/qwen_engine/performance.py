"""Opt-in, prompt-free substage diagnostics. Importing this module is ML-free.

CUDA is injected by the adapter after hardware validation. Only leaf CUDA
stages reset peaks; enclosing CPU scopes are wall times, not additive totals.
Reports never supply calibration/admission evidence.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import json
import os
from pathlib import Path
import time

from qwen_engine.bench import host_rss_bytes
from qwen_engine._jsonsafe import strict_json_loads
from qwen_engine._local_io import read_bounded_file

SCHEMA = "qwen_engine.performance.v1"
_active = ContextVar("qwen_performance", default=None)
CHILD_STAGES = {"text_encoder_loading", "prompt_encoding", "text_encoder_cleanup"}


def current():
    return _active.get()


def stage(name, *, cuda=False):
    recorder = current()
    return recorder.stage(name, cuda=cuda) if recorder is not None else nullcontext()


def attach_cuda(torch):
    if current() is not None:
        current().torch = torch


def _rss():
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


class PerformanceRecorder:
    def __init__(self, config=None, *, clock=time.perf_counter, human_sink=None):
        self.clock, self.human_sink = clock, human_sink
        self.torch = None
        self.records, self.errors = [], []
        self.metadata = ({
            "width": config.width, "height": config.height, "seed": config.seed,
            "steps": config.steps, "precision": "fp16", "quant": config.quant.value,
            "attention_requested": config.attention_backend.value,
            "vae_tiled_decode": config.vae_tiled_decode,
        } if config is not None else {})
        self._gpu_active = False

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    def _memory(self):
        cuda = self.torch.cuda
        return dict(allocated_bytes=int(cuda.memory_allocated(0)),
                    reserved_bytes=int(cuda.memory_reserved(0)),
                    peak_allocated_bytes=int(cuda.max_memory_allocated(0)),
                    peak_reserved_bytes=int(cuda.max_memory_reserved(0)))

    @contextmanager
    def stage(self, name, *, cuda=False):
        gpu = cuda and self.torch is not None
        if gpu and self._gpu_active:
            raise ValueError("CUDA performance scopes must not nest")
        record = dict(stage=name, pid=os.getpid(), status="succeeded", cuda=None,
                      host_rss_start_bytes=_rss(), host_rss_end_bytes=None,
                      host_lifetime_peak_rss_bytes=None)
        if gpu:
            self._gpu_active = True
            try:
                self.torch.cuda.synchronize(0)
                record["cuda_start"] = self._memory()
                self.torch.cuda.reset_peak_memory_stats(0)
            except Exception as exc:
                self.errors.append(type(exc).__name__)
                gpu = False
                self._gpu_active = False
        start = self.clock()
        try:
            yield
        except BaseException:
            record["status"] = "failed"
            raise
        finally:
            try:
                if gpu:
                    try:
                        self.torch.cuda.synchronize(0)
                        record["cuda"] = self._memory()
                    except Exception as exc:
                        self.errors.append(type(exc).__name__)
                record["wall_s"] = max(0., self.clock() - start)
                record["host_rss_end_bytes"] = _rss()
                record["host_lifetime_peak_rss_bytes"] = host_rss_bytes()
                self.records.append(record)
                if self.human_sink:
                    try:
                        self.human_sink(f"[perf] {name}: {record['wall_s']:.3f}s ({record['status']})")
                    except Exception:
                        pass
            finally:
                if gpu:
                    self._gpu_active = False

    def document(self, status):
        device = None
        if self.torch is not None:
            try:
                props = self.torch.cuda.get_device_properties(0)
                device = dict(name=props.name, compute_capability=[props.major, props.minor],
                              total_bytes=int(props.total_memory))
            except Exception as exc:
                self.errors.append(type(exc).__name__)
        return dict(schema=SCHEMA, status=status, metadata=self.metadata, device=device,
                    stages=self.records, measurement_errors=self.errors,
                    memory_scope="CUDA process allocator; host RSS per PID; host peak is lifetime, not per stage",
                    timing_scope="nested CPU scopes overlap; total_job includes cleanup; not additive",
                    attention_observed=None)

    def import_encoder_report(self, path):
        """Read only the bounded child sidecar before its private scratch is removed."""
        try:
            doc = strict_json_loads(read_bounded_file(path, 64 * 1024))
            if doc.get("schema") != SCHEMA or doc.get("metadata") != {}:
                raise ValueError("unexpected child performance schema")
            records = doc["stages"]
            if not isinstance(records, list) or len(records) > 3:
                raise ValueError("unexpected child performance stage count")
            expected = {"stage", "pid", "status", "cuda", "host_rss_start_bytes",
                        "host_rss_end_bytes", "host_lifetime_peak_rss_bytes", "wall_s"}
            import math
            for record in records:
                if (not isinstance(record, dict) or set(record) != expected
                        or record["stage"] not in CHILD_STAGES or record["cuda"] is not None
                        or record["status"] not in ("succeeded", "failed")
                        or type(record["pid"]) is not int or record["pid"] <= 0
                        or type(record["wall_s"]) not in (int, float)
                        or not math.isfinite(record["wall_s"]) or record["wall_s"] < 0):
                    raise ValueError("invalid child performance stage")
                for key in ("host_rss_start_bytes", "host_rss_end_bytes", "host_lifetime_peak_rss_bytes"):
                    if record[key] is not None and (type(record[key]) is not int or record[key] < 0):
                        raise ValueError("invalid child RSS")
            self.records.extend(records)
        except Exception as exc:
            self.errors.append("encoder_report_" + type(exc).__name__)

    def write(self, handle, status):
        try:
            json.dump(self.document(status), handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            return True
        except Exception as exc:
            # A diagnostic sink must not replace cancellation/primary failure
            # or turn an already verified image into a failed generation.
            self.errors.append("report_write_" + type(exc).__name__)
            if self.human_sink:
                try:
                    self.human_sink("[perf] report write failed: " + type(exc).__name__)
                except Exception:
                    pass
            return False
