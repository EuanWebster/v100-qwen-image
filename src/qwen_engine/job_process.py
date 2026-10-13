"""Owned, bounded POSIX process executor for the internal job controller.

Always starts the fixed generation_worker module in a new session; no public
command/backend override. Pipes are bounded in memory, never unbounded log
files. Only its live owned leader/group is signalled; direct child is reaped.
If reaping fails, retain its Popen and refuse new execution. A generation
worker must itself own/reap descendants before normal exit (encoder workers
inherit this session). These are cooperative-code ownership rules, not a
hostile process sandbox or an automatic GPU/model readiness approval.
"""
from __future__ import annotations

import json
import math
import os
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from qwen_engine._jsonsafe import strict_json_loads
from qwen_engine.core import GenerationResult
from qwen_engine.jobs import JobCancelled, JobContext, JobError
from qwen_engine.config import GenerationConfig
from qwen_engine.generation_worker import (
    REQUEST_SCHEMA, RESULT_SCHEMA, EVENT_SCHEMA, PHASES, MAX_REQUEST_BYTES,
    MAX_RESULT_BYTES, MAX_EVENT_BYTES, ERROR_CODES, valid_request_id,
)


class GenerationProcessError(JobError):
    """Fixed diagnostic code; no arbitrary child output is exposed."""


def _seconds(value, maximum, *, zero=False):
    if (type(value) not in (int, float) or not 0 <= value <= maximum
            or not zero and value == 0 or not math.isfinite(value)):
        raise GenerationProcessError("invalid_process_timeout")
    return float(value)


def _protocol_json(raw):
    """Explicit depth bound even on iterative JSON parsers (Python3.13)."""
    text = raw.decode("utf-8")
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 16:
                raise ValueError("protocol JSON depth cap")
        elif char in "]}":
            depth -= 1
    return strict_json_loads(text)


def _read_result(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise GenerationProcessError("result_file_unavailable") from None
    try:
        initial = os.fstat(fd)
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1 or not 1 <= initial.st_size <= MAX_RESULT_BYTES:
            raise GenerationProcessError("invalid_result_file")
        raw = bytearray()
        while len(raw) <= initial.st_size:
            block = os.read(fd, min(65536, initial.st_size + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        signature = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
        if (len(raw) != initial.st_size or signature(initial) != signature(os.fstat(fd))
                or signature(initial) != signature(Path(path).stat(follow_symlinks=False))):
            raise GenerationProcessError("result_file_changed")
        try:
            return _protocol_json(raw)
        except (ValueError, RecursionError):
            raise GenerationProcessError("invalid_result_json") from None
    finally:
        os.close(fd)


def _result(document, context, exit_code):
    if (not isinstance(document, dict) or set(document) != {
            "schema", "request_id", "status", "exit_code", "output_path", "error_code"}
            or document["schema"] != RESULT_SCHEMA or document["request_id"] != context.job_id
            or type(document["exit_code"]) is not int or document["exit_code"] != exit_code):
        raise GenerationProcessError("invalid_result_envelope")
    status = document["status"]
    if status == "succeeded" and exit_code == 0 and document["error_code"] is None:
        path = document["output_path"]
        if type(path) is not str or not path or "\0" in path or not Path(path).is_absolute() or Path(path).parent != Path(context.config.output_dir):
            raise GenerationProcessError("invalid_result_output")
        return GenerationResult(path)  # controller freshly verifies the artifact
    if status == "cancelled" and exit_code == 130 and document["error_code"] is None and document["output_path"] is None:
        if not context.cancel_requested:
            raise GenerationProcessError("unrequested_child_cancellation")
        raise JobCancelled()
    if (status == "failed" and document["output_path"] is None and exit_code in ERROR_CODES
            and (document["error_code"] == ERROR_CODES[exit_code]
                 or exit_code == 1 and document["error_code"] == "invalid_cancellation_acknowledgment")):
        raise GenerationProcessError(document["error_code"])
    raise GenerationProcessError("incoherent_result_status")


class GenerationProcessExecutor:
    def __init__(self, *, scratch_dir, python=None, timeout_s=600, cancel_grace_s=15,
                 max_output_bytes=1024 * 1024):
        if os.name != "posix":
            raise GenerationProcessError("posix_process_ownership_required")
        self.timeout_s = _seconds(timeout_s, 86400)
        self.cancel_grace_s = _seconds(cancel_grace_s, 30, zero=True)
        if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 1024 * 1024:
            raise GenerationProcessError("invalid_process_output_bound")
        self.max_output_bytes = max_output_bytes
        chosen = sys.executable if python is None else python
        if type(chosen) is not str or not chosen or "\0" in chosen:
            raise GenerationProcessError("invalid_process_interpreter")
        self.python = str(Path(chosen).absolute()) if os.sep in chosen else shutil.which(chosen)
        if self.python is None:
            raise GenerationProcessError("process_interpreter_missing")
        self.scratch_dir = Path(scratch_dir).resolve(strict=True)
        info = self.scratch_dir.stat()
        if not stat.S_ISDIR(info.st_mode):
            raise GenerationProcessError("invalid_process_scratch_root")
        self._scratch_identity = (info.st_dev, info.st_ino)
        self._lease = threading.Lock()
        self._closing = threading.Event()
        self._active = None
        self._pending_process = None
        self._pending_scratch = None
        self.last_cleanup_error = None

    @property
    def active_pid(self):
        return None if self._active is None else self._active.pid

    def close(self, *, timeout_s=5):
        timeout = _seconds(timeout_s, 60, zero=True)
        deadline = time.monotonic() + timeout
        self._closing.set()
        if not self._lease.acquire(timeout=timeout):
            return False
        try:
            if self._active is not None:
                try:
                    self._stop(self._active, deadline=deadline)
                except GenerationProcessError:
                    return False
            try:
                self._cleanup_owned()
            except Exception:
                return False
            return self._active is None and self._pending_process is None and self._pending_scratch is None
        finally:
            self._lease.release()

    def _cleanup_owned(self):
        """Keep handles and scratch identity until cleanup is actually done.

        Never remove scratch belonging to a still-live child or a replacement
        peer. Closed admission remains closed even after successful retry.
        """
        if self._active is not None:
            return
        if self._pending_process is not None:
            failed = False
            for stream in (self._pending_process.stdout, self._pending_process.stderr):
                try:
                    stream.close()
                except Exception:
                    failed = True
            if failed:
                self.last_cleanup_error = "worker_pipe_cleanup_failed"
                raise GenerationProcessError(self.last_cleanup_error)
            self._pending_process = None
        if self._pending_scratch is not None:
            scratch, owned = self._pending_scratch
            try:
                current = scratch.stat(follow_symlinks=False)
            except FileNotFoundError:
                # Already removed: do not invent a peer target to clean up.
                self._pending_scratch = None
                return
            except OSError:
                self.last_cleanup_error = "scratch_cleanup_failed"
                raise GenerationProcessError(self.last_cleanup_error) from None
            if (current.st_dev, current.st_ino, current.st_mode) != (owned.st_dev, owned.st_ino, owned.st_mode):
                self.last_cleanup_error = "scratch_directory_changed"
                raise GenerationProcessError(self.last_cleanup_error)
            try:
                shutil.rmtree(scratch)
            except Exception:
                self.last_cleanup_error = "scratch_cleanup_failed"
                raise GenerationProcessError(self.last_cleanup_error) from None
            self._pending_scratch = None

    def _stop(self, proc, *, deadline=None):
        def remaining(bound):
            return bound if deadline is None else min(bound, max(0, deadline - time.monotonic()))
        if proc.poll() is None:
            # An unreaped live leader reserves its PID/group ID. Never signal
            # a group after poll/wait has reaped the leader (PID reuse risk).
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=remaining(self.cancel_grace_s))
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                try:
                    proc.wait(timeout=remaining(5))
                except subprocess.TimeoutExpired:
                    raise GenerationProcessError("worker_reap_pending") from None
        if proc.poll() is not None and self._active is proc:
            self._active = None

    def __call__(self, context: JobContext):
        if (type(context) is not JobContext or not valid_request_id(context.job_id)
                or type(context.config) is not GenerationConfig or context.config.output_dir is None
                or not Path(context.config.output_dir).is_absolute()
                or not isinstance(context._cancel, threading.Event) or not callable(context._report)):
            raise GenerationProcessError("invalid_job_context")
        context.check_cancelled()
        if not self._lease.acquire(blocking=False):
            raise GenerationProcessError("executor_busy")
        scratch = proc = None
        primary = None
        completed = False
        try:
            if (self._closing.is_set() or self._active is not None
                    or self._pending_process is not None or self._pending_scratch is not None):
                raise GenerationProcessError("executor_unavailable")
            current = self.scratch_dir.stat(follow_symlinks=False)
            if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != self._scratch_identity:
                raise GenerationProcessError("scratch_root_changed")
            raw = json.dumps(dict(schema=REQUEST_SCHEMA, request_id=context.job_id,
                                  config=context.config.to_dict()), allow_nan=False,
                             ensure_ascii=False).encode("utf-8")
            if len(raw) > MAX_REQUEST_BYTES:
                raise GenerationProcessError("process_request_byte_cap")
            scratch = Path(tempfile.mkdtemp(prefix="qwen-gen-", dir=self.scratch_dir))
            owned = scratch.stat(follow_symlinks=False)
            self._pending_scratch = scratch, owned
            request, response = scratch / "request.json", scratch / "response.json"
            with request.open("xb") as stream:
                stream.write(raw)
            command = [self.python, "-B", "-m", "qwen_engine.generation_worker",
                       "--request", str(request), "--response", str(response)]
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, start_new_session=True)
            self._active = proc
            self._pending_process = proc
            result = self._await(proc, response, context)
            completed = True  # clean worker publication; controller still verifies artifact
            return result
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                if proc is not None:
                    self._stop(proc)
                self._cleanup_owned()
            except Exception:
                self._closing.set()
                if self.last_cleanup_error is None:
                    self.last_cleanup_error = "worker_cleanup_pending"
                if primary is None and not completed:
                    raise
                # Preserve primary failures/interrupts and a coherent clean
                # worker result. The controller still freshly verifies files.
                # Retained ownership + closed admission make failed cleanup
                # visible/retryable, not a claim that resources were released.
            finally:
                self._lease.release()

    def _await(self, proc, response, context):
        deadline = time.monotonic() + self.timeout_s
        counts = {"stdout": 0, "stderr": 0}
        pending = bytearray()
        sequence = 0
        stopped_for = None
        with selectors.DefaultSelector() as selector:
            for label, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, label)

            def drain(wait):
                nonlocal sequence
                events = selector.select(wait)
                for key, _ in events:
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    label = key.data
                    counts[label] += len(block)
                    if counts[label] > self.max_output_bytes:
                        raise GenerationProcessError("worker_output_byte_cap")
                    if label == "stderr":
                        continue  # bounded discard, never expose arbitrary diagnostics
                    pending.extend(block)
                    while b"\n" in pending:
                        line, _, tail = pending.partition(b"\n")
                        pending[:] = tail
                        if len(line) > MAX_EVENT_BYTES:
                            raise GenerationProcessError("worker_event_byte_cap")
                        try:
                            event = _protocol_json(line)
                        except (ValueError, RecursionError):
                            raise GenerationProcessError("invalid_worker_event") from None
                        if (not isinstance(event, dict) or set(event) != {"schema", "request_id", "event", "phase", "sequence"}
                                or event["schema"] != EVENT_SCHEMA or event["request_id"] != context.job_id
                                or event["event"] != "phase_start" or type(event["sequence"]) is not int
                                or event["sequence"] != sequence or sequence >= len(PHASES)
                                or event["phase"] != PHASES[sequence]):
                            raise GenerationProcessError("invalid_worker_phase")
                        if not context.cancel_requested:
                            try:
                                context._report(event["phase"])
                            except JobCancelled:
                                if not context.cancel_requested:
                                    raise
                        sequence += 1
                    if len(pending) > MAX_EVENT_BYTES:
                        raise GenerationProcessError("worker_event_byte_cap")
                return bool(events)

            while proc.poll() is None:
                if context.cancel_requested or self._closing.is_set():
                    self._stop(proc)
                    stopped_for = "cancelled" if context.cancel_requested else "executor_closed"
                    break
                if time.monotonic() >= deadline:
                    self._stop(proc)
                    stopped_for = "worker_timeout"
                    break
                drain(min(.05, max(0, deadline - time.monotonic())))
            try:
                while drain(0):
                    pass
            except GenerationProcessError:
                if stopped_for is None:
                    raise
        if stopped_for is not None:
            # A coherent verified-result publication + clean exit wins late
            # cancellation/close/timeout. Otherwise no success is invented.
            try:
                document = _read_result(response)
                if (isinstance(document, dict) and document.get("status") == "succeeded"
                        and not pending and sequence == len(PHASES)):
                    return _result(document, context, proc.returncode)
            except (OSError, GenerationProcessError):
                pass
            if stopped_for == "cancelled":
                raise JobCancelled()
            raise GenerationProcessError(stopped_for)
        if pending:
            raise GenerationProcessError("truncated_worker_event")
        document = _read_result(response)
        if not isinstance(document, dict):
            raise GenerationProcessError("invalid_result_envelope")
        if document.get("status") == "succeeded" and sequence != len(PHASES):
            raise GenerationProcessError("success_without_all_phases")
        return _result(document, context, proc.returncode)
