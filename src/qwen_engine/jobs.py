"""Bounded CPU-testable job lifecycle, not a generation backend or web API.

One owned worker runs an explicitly injected trusted executor serially. No
default executor, implicit ML imports or synthetic success exists. Cancellation
is cooperative: running jobs stay CANCELLING until the executor returns. A
finite close reports whether the worker actually stopped; it cannot kill a
Python thread. Production execution must additionally own/reap its child and
release model resources before returning. No images/directories are deleted.
A terminal FAILED/CANCELLED job additionally gets one deterministic
boundary reference release (injectable ``release_cyclic_references``) after
its state is recorded; successful jobs and queued-only cancellations never
run it.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import stat
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from qwen_engine._jsonsafe import strict_json_loads
from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.core import GenerationResult
from qwen_engine.errors import GenerationCancelled, OutOfGpuMemoryError

MAX_REQUEST_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
#: Terminal-state history record written inside each job's own directory
#: (the user's own prompt-bearing data on the user's own disk). The request
#: document already fits MAX_REQUEST_BYTES, so the record only needs room for
#: its own bounded envelope around it.
JOB_RECORD_NAME = "job.json"
JOB_RECORD_SCHEMA = "qwen_engine.job_record.v1"
MAX_RECORD_BYTES = MAX_REQUEST_BYTES + 8 * 1024
_RECORD_KEYS = frozenset({"schema", "job_id", "state", "error_code", "cancel_requested",
                          "created_ns", "finished_ns", "config", "artifact"})
_ARTIFACT_KEYS = frozenset({"filename", "bytes", "sha256"})
#: Closed set of fixed, prompt-free failure codes a record may carry.
_ERROR_CODES = frozenset({"execution_failed", "worker_stopped",
                          "invalid_cancellation_acknowledgment", "out_of_gpu_memory"})
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
_PHASES = ("load", "encode", "denoise", "decode", "save")
_STAT_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")


#: Stdlib-only CUDA-OOM classifier (torch is NEVER imported here — jobs.py
#: must stay ML-free). Recognised at the executor boundary: the typed
#: OutOfGpuMemoryError channel, torch's OutOfMemoryError class by type
#: identity (exact class name + torch top-level module), and the legacy
#: plain-RuntimeError CUDA-OOM texts. Only explicit ``__cause__`` chains are
#: followed: an implicit ``__context__`` may carry an unrelated in-flight
#: exception (a cancellation raised while an OOM is being handled must stay
#: a cancellation). Returns a bool only; the exception object and its text
#: never escape the call, so nothing prompt-bearing or traceback-bearing is
#: stored — the recorded error_code is one of the fixed literals.
_CUDA_OOM_TEXT = re.compile(
    r"CUDA out of memory|CUDA error: out of memory|cudaErrorMemoryAllocation",
    re.IGNORECASE)


def _is_gpu_oom(exc: BaseException | None) -> bool:
    seen: set[int] = set()
    while isinstance(exc, BaseException) and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OutOfGpuMemoryError):
            return True
        kind = type(exc)
        if (kind.__name__ == "OutOfMemoryError"
                and kind.__module__.partition(".")[0] == "torch"):
            return True
        if kind.__name__ == "RuntimeError" and _CUDA_OOM_TEXT.search(str(exc)):
            return True
        exc = exc.__cause__
    return False


class JobError(ValueError):
    """Bounded, fixed diagnostics; never expose executor exception text."""


class JobCancelled(Exception):
    """Executor acknowledged cancellation after releasing its resources."""


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_TERMINAL = frozenset({JobState.CANCELLED, JobState.SUCCEEDED, JobState.FAILED})


@dataclass(frozen=True)
class JobArtifact:
    filename: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    config: GenerationConfig
    state: JobState = JobState.QUEUED
    phase: str | None = None
    cancel_requested: bool = False
    artifact: JobArtifact | None = None
    error_code: str | None = None

    @property
    def terminal(self):
        return self.state in _TERMINAL


@dataclass(frozen=True)
class JobContext:
    job_id: str
    config: GenerationConfig
    _cancel: threading.Event
    _report: Callable[[str], None]

    @property
    def cancel_requested(self):
        return self._cancel.is_set()

    def check_cancelled(self):
        if self.cancel_requested:
            raise JobCancelled()

    def report_phase(self, phase: str):
        self.check_cancelled()
        self._report(phase)


def _timeout(value):
    if type(value) not in (int, float) or not 0 <= value <= 60 or not math.isfinite(value):
        raise JobError("timeout must be finite and between 0 and 60 seconds")
    return float(value)


def _request(config):
    if type(config) is not GenerationConfig:
        raise JobError("validated GenerationConfig required")
    if config.output_dir is not None:
        raise JobError("job output_dir is controller-owned; submit it unconfigured")
    doc = config.to_dict()
    if sum(len(v) for v in doc.values() if isinstance(v, str)) > MAX_REQUEST_BYTES:
        raise JobError("job request byte cap exceeded")
    try:
        raw = json.dumps(doc, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        raise JobError("job request must be valid UTF-8 JSON") from None
    if len(raw) > MAX_REQUEST_BYTES:
        raise JobError("job request byte cap exceeded")
    # Own builtin JSON values too: caller-supplied str/int subclasses must
    # not carry custom mutable behavior into a supposedly frozen snapshot.
    doc = json.loads(raw)
    # Copy immutable values and bind relative explicitly configured paths now,
    # preserving lexical '..'/symlink semantics (not abspath normalization).
    for name in ("models_dir", "manifest_path", "q8_manifest_path", "cache_dir", "log_dir", "benchmarks_dir"):
        if doc[name] is not None and not Path(doc[name]).is_absolute():
            doc[name] = str(Path.cwd() / doc[name])
    config = GenerationConfig.from_dict(doc)
    if len(json.dumps(config.to_dict(), ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_REQUEST_BYTES:
        raise JobError("job request byte cap exceeded")
    return config


def _signature(info) -> tuple[int, ...]:
    """Identity/state fields that must not change under an open file."""
    return tuple(getattr(info, name) for name in _STAT_FIELDS)


def job_record_document(record: JobSnapshot, *, created_ns: int, finished_ns: int) -> dict:
    """The exact terminal-state record stored in a job's own directory.

    Written only for a terminal snapshot; the builder refuses anything the
    loader would later reject, so the writer and the reader share one
    contract instead of drifting apart.
    """
    if record.state not in _TERMINAL:
        raise JobError("job record requires a terminal state")
    if record.error_code is not None and record.error_code not in _ERROR_CODES:
        raise JobError("job record error code is not a fixed known code")
    if record.error_code is not None and record.state != JobState.FAILED:
        raise JobError("only a failed job records an error code")
    if (record.artifact is None) != (record.state != JobState.SUCCEEDED):
        raise JobError("only a succeeded job records an artifact")
    for name, value in (("created_ns", created_ns), ("finished_ns", finished_ns)):
        if type(value) is not int or value < 0:
            raise JobError(f"job record {name} must be a nonnegative integer")
    if finished_ns < created_ns:
        raise JobError("job record finished_ns precedes created_ns")
    artifact = record.artifact
    return {
        "schema": JOB_RECORD_SCHEMA,
        "job_id": record.job_id,
        "state": record.state.value,
        "error_code": record.error_code,
        "cancel_requested": bool(record.cancel_requested),
        "created_ns": created_ns,
        "finished_ns": finished_ns,
        "config": record.config.to_dict(),
        "artifact": None if artifact is None else {
            "filename": artifact.filename,
            "bytes": artifact.bytes,
            "sha256": artifact.sha256,
        },
    }


def load_job_record(raw: bytes, *, job_id: str, output_dir: str) -> tuple[JobSnapshot, int, int]:
    """Strictly parse one stored record; ``(snapshot, created_ns, finished_ns)``.

    Every rejection is a fixed diagnostic: the stored document may hold
    arbitrary text, so no offending value is echoed into a log line. The
    recorded config is revalidated by ``GenerationConfig`` itself (the same
    validator a submitted request passes) with the job's directory bound to
    the current root, never to the path the old process happened to use.
    """
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_RECORD_BYTES:
        raise JobError("job record size is out of bounds")
    try:
        document = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise JobError("job record is not strict UTF-8 JSON") from None
    if type(document) is not dict or set(document) != _RECORD_KEYS:
        raise JobError("job record has an unexpected field set")
    if document["schema"] != JOB_RECORD_SCHEMA:
        raise JobError("job record schema is unknown")
    if document["job_id"] != job_id:
        raise JobError("job record belongs to another job")
    state = document["state"]
    try:
        if not isinstance(state, str):
            raise ValueError
        state = JobState(state)
    except ValueError:
        raise JobError("job record state is not terminal") from None
    if state not in _TERMINAL:
        raise JobError("job record state is not terminal")
    error_code = document["error_code"]
    if state == JobState.FAILED:
        if error_code not in _ERROR_CODES:
            raise JobError("job record error code is not a fixed known code")
    elif error_code is not None:
        raise JobError("job record carries an error code for a non-failed job")
    cancel_requested = document["cancel_requested"]
    if type(cancel_requested) is not bool:
        raise JobError("job record cancel flag is not a boolean")
    created_ns, finished_ns = document["created_ns"], document["finished_ns"]
    for name, value in (("created_ns", created_ns), ("finished_ns", finished_ns)):
        if type(value) is not int or value < 0:
            raise JobError(f"job record {name} must be a nonnegative integer")
    if finished_ns < created_ns:
        raise JobError("job record finished_ns precedes created_ns")
    config = document["config"]
    if type(config) is not dict:
        raise JobError("job record config is not an object")
    config = {**config, "output_dir": output_dir}
    try:
        parsed = GenerationConfig.from_dict(config)
    except ConfigError:
        raise JobError("job record config is invalid") from None
    # GenerationConfig accepts a partial map and fills the rest with defaults,
    # so a truncated or hand-edited record would otherwise reload as a job
    # whose stored settings silently differ from the ones that ran. A record
    # must carry the complete normalized document this build writes.
    if set(config) != set(parsed.to_dict()):
        raise JobError("job record config is not the complete document")
    config = parsed
    stored_artifact = document["artifact"]
    artifact = None
    if stored_artifact is not None:
        if type(stored_artifact) is not dict or set(stored_artifact) != _ARTIFACT_KEYS:
            raise JobError("job record artifact has an unexpected field set")
        filename, size, digest = (stored_artifact["filename"], stored_artifact["bytes"],
                                 stored_artifact["sha256"])
        if (type(filename) is not str or _ARTIFACT_NAME.fullmatch(filename) is None
                or type(size) is not int or not 1 <= size <= MAX_ARTIFACT_BYTES
                or type(digest) is not str or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise JobError("job record artifact is invalid")
        artifact = JobArtifact(filename, size, digest)
    if (artifact is None) != (state != JobState.SUCCEEDED):
        raise JobError("job record artifact does not match its state")
    return (JobSnapshot(job_id, config, state=state, cancel_requested=cancel_requested,
                        artifact=artifact, error_code=error_code), created_ns, finished_ns)


def release_cyclic_references() -> None:
    """Default queue-boundary reference release: one cyclic GC pass.

    The controller calls it only after a failed/cancelled job's exception
    has been fully handled and its terminal state recorded, so this
    package's own exception-free frames are already freed by refcounting;
    this pass additionally frees reference cycles left inside third-party
    pipeline code before the next job loads models. It never touches
    device allocators (no ``empty_cache``): production adapters own
    device teardown.
    """
    gc.collect()


class JobController:
    """Bounded queue/history with explicitly owned per-job artifact directories.

    Construction requires an existing explicit root; canonical root aliases
    are intentional. Root identity and artifact FDs detect observed changes,
    not arbitrary hostile rewriting. Forget evicts only terminal metadata,
    NEVER files. Queue/history limits are not disk quotas. Executors are
    trusted local code, not untrusted plugin isolation.
    """

    def __init__(self, root: str | Path, executor: Callable[[JobContext], GenerationResult], *,
                 max_pending: int = 4, max_records: int = 32,
                 memory_release: Callable[[], object] | None = None):
        if not callable(executor):
            raise JobError("explicit executor required")
        if (type(max_pending) is not int or not 1 <= max_pending <= 32
                or type(max_records) is not int or not max_pending <= max_records <= 128):
            raise JobError("invalid queue/history limits")
        if memory_release is not None and not callable(memory_release):
            raise JobError("memory release collector must be callable or None")
        self.root = Path(root).resolve(strict=True)
        self._root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
        info = os.fstat(self._root_fd)
        self._root_identity = (info.st_dev, info.st_ino)
        self._executor = executor
        self._memory_release = release_cyclic_references if memory_release is None else memory_release
        self._max_pending, self._max_records = max_pending, max_records
        self._condition = threading.Condition()
        self._records: dict[str, JobSnapshot] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._collect_ms: dict[str, float] = {}
        self._directories: dict[str, tuple[int, int]] = {}
        self._created_ns: dict[str, int] = {}
        self._pending: deque[str] = deque()
        self._closing = False
        self._active: str | None = None
        # Rebuild the terminal history from the job directories themselves
        # before the worker exists: nothing else can touch the maps yet, so
        # no lock is needed and a later submit/snapshot sees a stable set.
        self._restore_history()
        self._thread = threading.Thread(target=self._work, name="qwen-owned-job-worker", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            os.close(self._root_fd)
            raise

    def _check_root(self):
        info = self.root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != self._root_identity:
            raise JobError("artifact root changed")

    def submit(self, config: GenerationConfig) -> JobSnapshot:
        config = _request(config)
        with self._condition:
            if self._closing:
                raise JobError("controller closed or worker unavailable")
            if len(self._records) >= self._max_records:
                raise JobError("job history full; explicitly forget terminal metadata")
            if len(self._pending) >= self._max_pending:
                raise JobError("job queue full")
            self._check_root()
            job_id = uuid.uuid4().hex
            if job_id in self._records:
                raise JobError("job identifier collision")
            config = replace(config, output_dir=str(self.root / job_id))
            if len(json.dumps(config.to_dict(), ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_REQUEST_BYTES:
                raise JobError("job request byte cap exceeded")
            # Anchored mkdir cannot redirect into a peer-created directory.
            os.mkdir(job_id, mode=0o700, dir_fd=self._root_fd)
            self._check_root()
            directory = os.stat(job_id, dir_fd=self._root_fd, follow_symlinks=False)
            if not stat.S_ISDIR(directory.st_mode):
                raise JobError("new job directory changed")
            self._directories[job_id] = (directory.st_dev, directory.st_ino)
            record = JobSnapshot(job_id, config)
            self._records[job_id] = record
            self._created_ns[job_id] = time.time_ns()
            self._cancels[job_id] = threading.Event()
            self._pending.append(job_id)
            self._condition.notify_all()
            return record

    def _record(self, job_id):
        if (type(job_id) is not str or len(job_id) != 32
                or any(c not in "0123456789abcdef" for c in job_id)
                or job_id not in self._records):
            raise JobError("unknown job")
        return self._records[job_id]

    def get(self, job_id: str) -> JobSnapshot:
        with self._condition:
            return self._record(job_id)

    def snapshots(self) -> tuple[JobSnapshot, ...]:
        with self._condition:
            return tuple(self._records.values())

    def read_artifact(self, job_id: str) -> bytes:
        """Read only a recorded, successful artifact through owned directory FDs.

        Recheck the saved digest on the actual bytes returned: replacing a file
        after job completion must never turn the image endpoint into file access.
        """
        with self._condition:
            record = self._record(job_id)
            artifact = record.artifact
            if record.state != JobState.SUCCEEDED or artifact is None:
                raise JobError("job has no completed artifact")
            self._check_root()
            directory_fd = self._open_job_directory(job_id)
            try:
                return self._verified_artifact_bytes(directory_fd, artifact)
            finally:
                os.close(directory_fd)

    def _open_job_directory(self, job_id: str) -> int:
        """Open a recorded job directory without following links and require
        the identity captured when this controller first saw it."""
        directory_fd = os.open(job_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=self._root_fd)
        try:
            info = os.fstat(directory_fd)
            if (info.st_dev, info.st_ino) != self._directories[job_id]:
                raise JobError("job directory replaced")
            return directory_fd
        except BaseException:
            os.close(directory_fd)
            raise

    @staticmethod
    def _verified_artifact_bytes(directory_fd: int, artifact: JobArtifact) -> bytes:
        """Read one artifact file and verify size + sha256 on the very bytes
        returned. Shared by the image endpoint and the startup rebuild, so a
        reloaded history entry can only be served from bytes that still match
        the record it was rebuilt from."""
        fd = os.open(artifact.filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=directory_fd)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or info.st_size != artifact.bytes):
                raise JobError("artifact changed")
            parts, count = [], 0
            while count <= artifact.bytes:
                block = os.read(fd, min(65536, artifact.bytes + 1 - count))
                if not block:
                    break
                parts.append(block)
                count += len(block)
            data = b"".join(parts)
            if count != artifact.bytes or hashlib.sha256(data).hexdigest() != artifact.sha256:
                raise JobError("artifact changed")
            if _signature(info) != _signature(os.fstat(fd)):
                raise JobError("artifact changed")
            return data
        finally:
            os.close(fd)

    def cancel(self, job_id: str) -> JobSnapshot:
        with self._condition:
            record = self._record(job_id)
            if record.terminal:
                return record
            self._cancels[job_id].set()
            if record.state == JobState.QUEUED:
                self._pending.remove(job_id)
                state = JobState.CANCELLED
                del self._cancels[job_id]
            else:
                state = JobState.CANCELLING
            record = replace(record, state=state, cancel_requested=True)
            self._records[job_id] = record
            if state == JobState.CANCELLED:
                # A queued cancellation is terminal here: no worker phase will
                # follow, so this is the one place its own record is written.
                self._write_job_record(job_id, record)
            self._condition.notify_all()
            return record

    def forget(self, job_id: str):
        with self._condition:
            if not self._record(job_id).terminal:
                raise JobError("cannot forget live job")
            # Removing the stored record is what makes a forget survive a
            # restart; the image file itself is never deleted.
            self._discard_job_record(job_id)
            del self._records[job_id]  # never delete partials or completed artifacts
            del self._directories[job_id]
            self._created_ns.pop(job_id, None)
            self._collect_ms.pop(job_id, None)  # evicted with the metadata

    def wait(self, job_id: str, *, timeout_s: float = 5) -> JobSnapshot:
        deadline = time.monotonic() + _timeout(timeout_s)
        with self._condition:
            while not self._record(job_id).terminal:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._record(job_id)

    def collect_ms(self, job_id: str) -> float | None:
        """Wall cost of the queue-boundary reference release recorded for a
        terminal CANCELLED/FAILED job, or None when no collector ran (a
        success, or a cancellation that never entered the executor)."""
        with self._condition:
            return self._collect_ms.get(job_id)

    def close(self, *, timeout_s: float = 5) -> bool:
        timeout_s = _timeout(timeout_s)
        with self._condition:
            self._closing = True
            for job_id in tuple(self._pending):
                self.cancel(job_id)
            if self._active is not None:
                self.cancel(self._active)
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def _job_directory_fd(self, job_id: str) -> int | None:
        """Open a job directory for record bookkeeping; never follows links.

        Bookkeeping is best-effort: a directory that has been replaced or
        removed since it was recorded simply gets no record written to it.
        """
        try:
            return self._open_job_directory(job_id)
        except (OSError, JobError):
            return None

    def _write_job_record(self, job_id: str, record: JobSnapshot) -> None:
        """Atomically replace this job's own ``job.json`` with its terminal state.

        Called while the controller lock is held, so a concurrent ``forget``
        cannot delete the record and then have it resurrected by this write.
        The document is built and bounded BEFORE any file is touched; any
        failure keeps the in-memory history authoritative and never kills the
        worker.
        """
        created_ns = self._created_ns.get(job_id)
        if created_ns is None:
            return
        try:
            document = job_record_document(record, created_ns=created_ns, finished_ns=time.time_ns())
            raw = json.dumps(document, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (JobError, TypeError, ValueError):
            return
        if not 1 <= len(raw) <= MAX_RECORD_BYTES:
            return
        directory_fd = self._job_directory_fd(job_id)
        if directory_fd is None:
            return
        temporary = f".{JOB_RECORD_NAME}.{uuid.uuid4().hex}.tmp"
        # /proc/self/fd keeps every step tied to the directory FD we verified
        # while also naming an absolute path (a relative name would be resolved
        # against the process CWD, which is not this job's directory).
        anchored = Path("/proc/self/fd") / str(directory_fd)
        try:
            fd = os.open(anchored / temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600)
            try:
                written = 0
                while written < len(raw):
                    written += os.write(fd, raw[written:])
                os.fsync(fd)
            finally:
                os.close(fd)
            # Replace, never truncate-in-place: a crash leaves either the old
            # record or the new one, never a half-written document.
            os.replace(anchored / temporary, anchored / JOB_RECORD_NAME)
            os.fsync(directory_fd)
        except OSError as exc:
            try:
                os.unlink(anchored / temporary)
            except OSError:
                pass
            print(f"qwen-history: job {job_id} history record not written "
                  f"({exc.strerror or 'storage error'})", file=sys.stderr)
        finally:
            os.close(directory_fd)

    def _discard_job_record(self, job_id: str) -> None:
        """Remove only this job's history record; the image file is untouched."""
        directory_fd = self._job_directory_fd(job_id)
        if directory_fd is None:
            return
        try:
            os.unlink(JOB_RECORD_NAME, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"qwen-history: job {job_id} history record not removed "
                  f"({exc.strerror or 'storage error'})", file=sys.stderr)
        finally:
            os.close(directory_fd)

    def _restore_history(self) -> None:
        """Rebuild terminal history from the job directories on disk.

        Runs before the worker thread exists, so the maps are touched without
        the lock. Only directories holding a strictly valid ``job.json`` come
        back: a job that was running when the process died has no record and
        is deliberately not resurrected. Nothing is ever deleted here; entries
        beyond ``max_records`` are simply not loaded.
        """
        try:
            names = os.listdir(self._root_fd)
        except OSError:
            return
        stored = []
        for job_id in names:
            if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
                continue
            restored = self._load_stored_job(job_id)
            if restored is not None:
                stored.append(restored)
        # Newest by SUBMISSION time: that is exactly the order a live
        # controller accumulates its records in, so selecting and re-inserting
        # on the same key makes a rebuilt History list identical to the one the
        # previous process had (the UI shows it newest-last).
        stored.sort(key=lambda item: (item[3], item[2]), reverse=True)
        for job_id, record, _finished_ns, created_ns, identity in reversed(
                stored[: self._max_records]):
            self._records[job_id] = record
            self._directories[job_id] = identity
            self._created_ns[job_id] = created_ns

    @staticmethod
    def _read_stored_record(directory_fd: int) -> bytes | None:
        """Read one bounded ``job.json`` through an owned directory FD."""
        try:
            fd = os.open(JOB_RECORD_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=directory_fd)
        except FileNotFoundError:
            return None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 \
                    or not 1 <= info.st_size <= MAX_RECORD_BYTES:
                raise JobError("job record is not a bounded regular file")
            parts, count = [], 0
            while count <= MAX_RECORD_BYTES:
                block = os.read(fd, min(65536, MAX_RECORD_BYTES + 1 - count))
                if not block:
                    break
                parts.append(block)
                count += len(block)
            data = b"".join(parts)
            if count != info.st_size or _signature(info) != _signature(os.fstat(fd)):
                raise JobError("job record changed while being read")
            return data
        finally:
            os.close(fd)

    @staticmethod
    def _check_stored_artifact(directory_fd: int, artifact: JobArtifact) -> None:
        """Prove at startup that the recorded image is still there at its
        recorded size and link count. This is advisory only: the sha256 is
        re-checked on the exact bytes every time the image is served
        (``read_artifact``), so a restarted server can never hand out a file
        that no longer matches its record even when this cheap check passes."""
        fd = os.open(artifact.filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=directory_fd)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or info.st_size != artifact.bytes):
                raise JobError("artifact changed")
        finally:
            os.close(fd)

    def _load_stored_job(self, job_id: str):
        """Strictly parse one directory's record; ``None`` when it is absent
        (normal for a job that never reached a terminal state) or unusable."""
        try:
            directory_fd = os.open(job_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                   dir_fd=self._root_fd)
        except OSError:
            return None
        try:
            directory = os.fstat(directory_fd)
            identity = (directory.st_dev, directory.st_ino)
            raw = self._read_stored_record(directory_fd)
            if raw is None:
                return None
            record, created_ns, finished_ns = load_job_record(
                raw, job_id=job_id, output_dir=str(self.root / job_id))
            if record.artifact is not None:
                try:
                    self._check_stored_artifact(directory_fd, record.artifact)
                except (JobError, OSError):
                    # The RECORD is what history is rebuilt from; the image is
                    # only ever served through the digest gate in
                    # ``read_artifact``. Dropping the entry here would hide it
                    # from the UI entirely (no way to see it, no way to forget
                    # it) while leaving its job.json on disk, so a job whose
                    # image moved or changed reloads exactly like a live job
                    # whose image changed after completion: visible, with an
                    # unavailable image.
                    print(f"qwen-history: job {job_id} image no longer matches "
                          f"its record", file=sys.stderr)
            return (job_id, record, finished_ns, created_ns, identity)
        except JobError as exc:
            print(f"qwen-history: skipped job {job_id}: {exc}", file=sys.stderr)
            return None
        except OSError:
            print(f"qwen-history: skipped job {job_id}: directory is unreadable", file=sys.stderr)
            return None
        finally:
            os.close(directory_fd)

    def _phase(self, job_id, phase):
        with self._condition:
            record = self._record(job_id)
            if self._active == job_id and record.state == JobState.CANCELLING:
                raise JobCancelled()
            if self._active != job_id or record.state != JobState.RUNNING:
                raise JobError("phase report outside running job")
            previous = -1 if record.phase is None else _PHASES.index(record.phase)
            if not isinstance(phase, str) or phase not in _PHASES or _PHASES.index(phase) != previous + 1:
                raise JobError("phase report out of order")
            self._records[job_id] = replace(record, phase=phase)
            self._condition.notify_all()

    def _artifact(self, result, record):
        if type(result) is not GenerationResult or not isinstance(result.output_path, str):
            raise JobError("executor returned invalid result")
        self._check_root()
        path = Path(result.output_path)
        directory = Path(record.config.output_dir)
        if not path.is_absolute() or path.parent != directory or path.name in ("", ".", ".."):
            raise JobError("executor artifact is outside its direct job directory")
        directory_fd = os.open(record.job_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=self._root_fd)
        try:
            opened = os.fstat(directory_fd)
            if (opened.st_dev, opened.st_ino) != self._directories[record.job_id]:
                raise JobError("job directory replaced")
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 1 <= before.st_size <= MAX_ARTIFACT_BYTES:
                    raise JobError("executor artifact is not a bounded nonempty single-link regular file")
                digest, count = hashlib.sha256(), 0
                while True:
                    block = os.read(fd, min(65536, before.st_size + 1 - count))
                    if not block:
                        break
                    digest.update(block)
                    count += len(block)
                    if count > before.st_size:
                        raise JobError("executor artifact grew during verification")
                signature = lambda info: tuple(getattr(info, name) for name in _STAT_FIELDS)
                if (count != before.st_size or signature(before) != signature(os.fstat(fd))
                        or signature(before) != signature(os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False))):
                    raise JobError("executor artifact changed during verification")
                self._check_root()
                opened, named = os.fstat(directory_fd), directory.stat(follow_symlinks=False)
                if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                    raise JobError("job directory changed during verification")
                return JobArtifact(path.name, count, digest.hexdigest())
            finally:
                os.close(fd)
        finally:
            os.close(directory_fd)

    def _work(self):
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._closing:
                        self._condition.wait()
                    if self._closing:
                        return
                    job_id = self._pending.popleft()
                    self._active = job_id
                    self._records[job_id] = replace(self._records[job_id], state=JobState.RUNNING)
                    record = self._records[job_id]
                    context = JobContext(job_id, record.config, self._cancels[job_id],
                                         lambda phase, owned=job_id: self._phase(owned, phase))
                    self._condition.notify_all()
                artifact = None
                error = None
                try:
                    context.check_cancelled()
                    self._check_root()
                    directory = Path(record.config.output_dir).stat(follow_symlinks=False)
                    if not stat.S_ISDIR(directory.st_mode) or (directory.st_dev, directory.st_ino) != self._directories[job_id]:
                        raise JobError("job directory replaced before execution")
                    result = self._executor(context)
                    artifact = self._artifact(result, record)
                    state = JobState.SUCCEEDED  # verified save wins a late cancellation request
                except (JobCancelled, GenerationCancelled):
                    if not context.cancel_requested:
                        state, error = JobState.FAILED, "invalid_cancellation_acknowledgment"
                    else:
                        state = JobState.CANCELLED
                except Exception as exc:
                    # Fixed codes only; the classifier returns a bool and the
                    # exception object dies with this except block (the
                    # post-block gc boundary below stays traceback-free).
                    state = JobState.FAILED
                    error = ("out_of_gpu_memory" if _is_gpu_oom(exc)
                             else "execution_failed")
                with self._condition:
                    self._records[job_id] = replace(self._records[job_id], state=state, artifact=artifact, error_code=error)
                    del self._cancels[job_id]
                    self._active = None
                    # Written while the lock is held so a concurrent forget
                    # cannot delete the record and then see it resurrected.
                    self._write_job_record(job_id, self._records[job_id])
                    self._condition.notify_all()
                # Deterministic queue-boundary memory release, only for
                # FAILED/CANCELLED jobs: the executor exception has now been
                # fully handled (this is outside every except scope, so this
                # thread retains no traceback) and the terminal state is
                # recorded. Nothing in this package stores a raised
                # exception any more, so refcounting has already freed our
                # frames; this one cyclic collection additionally frees
                # cycles left inside third-party pipeline code before the
                # next job loads models. Successes retain nothing here and
                # never pay; queued cancellations never entered the
                # executor. The collect timing is prompt-free bookkeeping.
                if state in (JobState.CANCELLED, JobState.FAILED):
                    started = time.monotonic()
                    try:
                        self._memory_release()
                    except Exception:
                        pass  # the state is final; never kill the worker
                    with self._condition:
                        self._collect_ms[job_id] = (time.monotonic() - started) * 1000.0
        finally:
            # Unexpected BaseException/thread death must not strand live jobs.
            # Preserve the original thread exception; expose only fixed codes.
            with self._condition:
                self._closing = True
                for job_id, record in tuple(self._records.items()):
                    if not record.terminal:
                        self._records[job_id] = replace(record, state=JobState.FAILED, error_code="worker_stopped")
                self._active = None
                self._pending.clear()
                self._cancels.clear()
                self._condition.notify_all()
                os.close(self._root_fd)
