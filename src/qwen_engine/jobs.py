"""Bounded CPU-testable job lifecycle, not a generation backend or web API.

One owned worker runs an explicitly injected trusted executor serially. No
default executor, implicit ML imports or synthetic success exists. Cancellation
is cooperative: running jobs stay CANCELLING until the executor returns. A
finite close reports whether the worker actually stopped; it cannot kill a
Python thread. Production execution must additionally own/reap its child and
release model resources before returning. No images/directories are deleted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from qwen_engine.config import GenerationConfig
from qwen_engine.core import GenerationResult
from qwen_engine.errors import GenerationCancelled

MAX_REQUEST_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_PHASES = ("load", "encode", "denoise", "decode", "save")
_STAT_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")


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
    for name in ("models_dir", "manifest_path", "cache_dir", "log_dir", "benchmarks_dir"):
        if doc[name] is not None and not Path(doc[name]).is_absolute():
            doc[name] = str(Path.cwd() / doc[name])
    config = GenerationConfig.from_dict(doc)
    if len(json.dumps(config.to_dict(), ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_REQUEST_BYTES:
        raise JobError("job request byte cap exceeded")
    return config


class JobController:
    """Bounded queue/history with explicitly owned per-job artifact directories.

    Construction requires an existing explicit root; canonical root aliases
    are intentional. Root identity and artifact FDs detect observed changes,
    not arbitrary hostile rewriting. Forget evicts only terminal metadata,
    NEVER files. Queue/history limits are not disk quotas. Executors are
    trusted local code, not untrusted plugin isolation.
    """

    def __init__(self, root: str | Path, executor: Callable[[JobContext], GenerationResult], *,
                 max_pending: int = 4, max_records: int = 32):
        if not callable(executor):
            raise JobError("explicit executor required")
        if (type(max_pending) is not int or not 1 <= max_pending <= 32
                or type(max_records) is not int or not max_pending <= max_records <= 128):
            raise JobError("invalid queue/history limits")
        self.root = Path(root).resolve(strict=True)
        self._root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
        info = os.fstat(self._root_fd)
        self._root_identity = (info.st_dev, info.st_ino)
        self._executor = executor
        self._max_pending, self._max_records = max_pending, max_records
        self._condition = threading.Condition()
        self._records: dict[str, JobSnapshot] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._directories: dict[str, tuple[int, int]] = {}
        self._pending: deque[str] = deque()
        self._closing = False
        self._active: str | None = None
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
            directory_fd = os.open(job_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                   dir_fd=self._root_fd)
            try:
                info = os.fstat(directory_fd)
                if (info.st_dev, info.st_ino) != self._directories[job_id]:
                    raise JobError("job directory replaced")
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
                    self._check_root()
                    return data
                finally:
                    os.close(fd)
            finally:
                os.close(directory_fd)

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
            self._condition.notify_all()
            return record

    def forget(self, job_id: str):
        with self._condition:
            if not self._record(job_id).terminal:
                raise JobError("cannot forget live job")
            del self._records[job_id]  # never delete partials or completed artifacts
            del self._directories[job_id]

    def wait(self, job_id: str, *, timeout_s: float = 5) -> JobSnapshot:
        deadline = time.monotonic() + _timeout(timeout_s)
        with self._condition:
            while not self._record(job_id).terminal:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._record(job_id)

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
                except Exception:
                    state, error = JobState.FAILED, "execution_failed"
                with self._condition:
                    self._records[job_id] = replace(self._records[job_id], state=state, artifact=artifact, error_code=error)
                    del self._cancels[job_id]
                    self._active = None
                    self._condition.notify_all()
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
