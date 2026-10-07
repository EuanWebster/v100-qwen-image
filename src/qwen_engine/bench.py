"""Versioned benchmark + calibration schemas (plan R6, task 1.5).

Stdlib-only, and the **no-fabrication rule** is the point of the schema:
every measured field is ``null`` when it was not measured — peak CUDA memory
in particular is recorded as ``null`` on machines without CUDA (this sandbox
included), never guessed. This module also never imports torch: the CUDA
numbers are *provided by the adapter* at record time (heavy imports stay
inside it); the host-RSS probe here is stdlib ``resource``.

Schemas
-------
* ``benchmark_result.v1`` — one run: per-phase wall times, optional step
  count (denoise-per-step is *derived*, not invented), peak host RSS, peak
  CUDA allocated/reserved (null when unavailable), environment identifiers,
  model identifiers from the manifest, and an echo of the config.
* ``benchmark_aggregate.v1`` — deterministic aggregation over results
  (min/median/max per phase; ``n_measured`` separates "absent" from "zero").
* ``calibration_observation.v1`` + ``qwen_engine.calibration.v1`` —
  the observation records feeding ``calibration.json``; each record carries
  the staleness key the memory policy (M5.3) matches on: schema version,
  model revision, attention backend, quant.
"""
from __future__ import annotations

import json
import math
import os
import platform as platform_module
import re
import resource
import socket
import statistics
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from qwen_engine.manifest import ModelManifest

BENCHMARK_SCHEMA = "benchmark_result.v1"
AGGREGATE_SCHEMA = "benchmark_aggregate.v1"
OBSERVATION_SCHEMA = "calibration_observation.v1"
CALIBRATION_FILE_SCHEMA = "qwen_engine.calibration.v1"

#: Conventional phase names (the schema does not restrict to these — a run
#: may record extra phases; they aggregate the same way).
PHASES = ("load", "encode", "denoise", "decode", "save")

_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class BenchSchemaError(ValueError):
    """A benchmark/calibration record violates its versioned schema; the
    message names every offending field."""


def _bad(errors: list[str], name: str, msg: str) -> None:
    errors.append(f"{name}: {msg}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_nonneg(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value) and value >= 0


def _check_optional_bytes(value: Any, name: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _bad(errors, name, f"must be a non-negative int or null, got {value!r}")


# ---------------------------------------------------------------------------
# Environment identifiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkEnvironment:
    """Machine/library identifiers for a run. torch/CUDA/driver/arch are
    ``None`` when not measured — **never fabricated** — and this module has
    no way to measure them without torch, so an adapter (or a future probe
    script) supplies them on GPU runs."""

    python: str
    platform: str
    hostname: str | None = None
    torch: str | None = None
    cuda: str | None = None
    driver: str | None = None
    arch_list: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        for name in ("python", "platform"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                _bad(errors, name, f"must be a non-empty string, got {value!r}")
        for name in ("hostname", "torch", "cuda", "driver"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                _bad(errors, name, f"must be a non-empty string or null, got {value!r}")
        if self.arch_list is not None and (
                isinstance(self.arch_list, (str, bytes))
                or not all(isinstance(a, str) and a.strip() for a in self.arch_list)):
            _bad(errors, "arch_list",
                 f"must be a list of non-empty strings or null, got {self.arch_list!r}")
        if errors:
            raise BenchSchemaError("invalid benchmark environment: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {"python": self.python, "platform": self.platform,
                "hostname": self.hostname, "torch": self.torch, "cuda": self.cuda,
                "driver": self.driver,
                "arch_list": list(self.arch_list) if self.arch_list is not None else None}

    @classmethod
    def from_dict(cls, data: Any) -> "BenchmarkEnvironment":
        errors: list[str] = []
        if not isinstance(data, Mapping):
            raise BenchSchemaError(
                f"environment: must be an object, got {type(data).__name__}")
        known = {"python", "platform", "hostname", "torch", "cuda", "driver",
                 "arch_list"}
        for key in sorted(set(data) - known):
            _bad(errors, key, "unknown environment key")
        arch = data.get("arch_list")
        if arch is not None and (isinstance(arch, (str, bytes))
                                 or not isinstance(arch, list)):
            _bad(errors, "arch_list", f"must be a list or null, got {arch!r}")
            arch = None
        try:
            env = cls(python=data.get("python"), platform=data.get("platform"),
                      hostname=data.get("hostname"), torch=data.get("torch"),
                      cuda=data.get("cuda"), driver=data.get("driver"),
                      arch_list=tuple(arch) if arch is not None else None)
        except BenchSchemaError as exc:
            raise BenchSchemaError(f"invalid benchmark environment: {exc}") from exc
        if errors:
            raise BenchSchemaError("invalid benchmark environment: "
                                   + "; ".join(errors))
        return env


def capture_environment(cuda_info: Mapping[str, Any] | None = None) -> BenchmarkEnvironment:
    """Capture the parts measurable **without torch** (python, platform,
    hostname). ``cuda_info`` (from an adapter that may import torch) carries
    ``{torch, cuda, driver, arch_list}``; anything absent stays ``None``."""
    try:
        hostname: str | None = socket.gethostname() or None
    except OSError:  # pragma: no cover - exotic containers
        hostname = None
    info = dict(cuda_info or {})
    unknown = sorted(set(info) - {"torch", "cuda", "driver", "arch_list"})
    if unknown:
        raise BenchSchemaError("cuda_info keys not recognised: " + ", ".join(unknown))
    arch = info.get("arch_list")
    return BenchmarkEnvironment(
        python=platform_module.python_version(),
        platform=platform_module.platform(),
        hostname=hostname,
        torch=info.get("torch"), cuda=info.get("cuda"),
        driver=info.get("driver"),
        arch_list=tuple(arch) if isinstance(arch, (list, tuple)) else arch,
    )


def host_rss_bytes() -> int | None:
    """Peak resident set size of *this process* in bytes (stdlib
    ``resource``); ``None`` where the probe does not exist — never 0."""
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (AttributeError, OSError, ValueError):  # pragma: no cover
        return None


def model_identifiers(manifest: ModelManifest) -> tuple[dict[str, Any], ...]:
    """Repo/revision/filename/sha256 per artifact, deterministic order —
    the model identity half of a benchmark record (R6)."""
    out = []
    for art in sorted(manifest.artifacts, key=lambda a: (a.role, a.identity)):
        out.append({"role": art.role, "repo": art.repo, "revision": art.revision,
                    "filename": art.filename, "sha256": art.sha256})
    return tuple(out)


# ---------------------------------------------------------------------------
# benchmark_result.v1
# ---------------------------------------------------------------------------

_RESULT_KEYS = {"schema", "run_id", "environment", "model", "config", "phases",
                "steps", "peak_host_rss_bytes", "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes"}


@dataclass(frozen=True)
class BenchmarkResult:
    schema: str = BENCHMARK_SCHEMA
    run_id: str = ""
    environment: BenchmarkEnvironment | None = None
    model: tuple[Mapping[str, Any], ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)
    #: Wall-clock milliseconds per phase name (>= 0, finite).
    phases: Mapping[str, float] = field(default_factory=dict)
    steps: int | None = None
    peak_host_rss_bytes: int | None = None
    peak_cuda_allocated_bytes: int | None = None
    peak_cuda_reserved_bytes: int | None = None

    def __post_init__(self) -> None:
        errors = validate_result_fields(self)
        if errors:
            raise BenchSchemaError("invalid benchmark result: " + "; ".join(errors))

    @property
    def denoise_ms_per_step(self) -> float | None:
        """Derived only when both parts were measured — never stored."""
        if self.steps and "denoise" in self.phases:
            return float(self.phases["denoise"]) / self.steps
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "environment": self.environment.to_dict() if self.environment else None,
            "model": [dict(m) for m in self.model],
            "config": dict(self.config),
            "phases": dict(self.phases),
            "steps": self.steps,
            "peak_host_rss_bytes": self.peak_host_rss_bytes,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Any) -> "BenchmarkResult":
        errors: list[str] = []
        if not isinstance(data, Mapping):
            raise BenchSchemaError(
                f"benchmark result: must be a JSON object, got {type(data).__name__}")
        for key in sorted(set(data) - _RESULT_KEYS):
            _bad(errors, key, "unknown benchmark result key")
        env: BenchmarkEnvironment | None = None
        raw_env = data.get("environment")
        if raw_env is None:
            _bad(errors, "environment", "required (record nulls inside it, not the whole block)")
        else:
            try:
                env = BenchmarkEnvironment.from_dict(raw_env)
            except BenchSchemaError as exc:
                _bad(errors, "environment", str(exc))
        model = data.get("model")
        model_errors = _check_model_list(model)
        errors.extend(model_errors)
        phases = data.get("phases")
        if not isinstance(phases, dict):
            _bad(errors, "phases", f"must be an object, got {phases!r}")
            phases = {}
        config = data.get("config")
        if config is not None and not isinstance(config, dict):
            _bad(errors, "config", f"must be an object, got {config!r}")
            config = {}
        if errors:
            raise BenchSchemaError("invalid benchmark result: " + "; ".join(errors))
        return cls(run_id=data.get("run_id", ""), environment=env,
                   model=tuple(model or ()), config=dict(config or {}),
                   phases=dict(phases), steps=data.get("steps"),
                   peak_host_rss_bytes=data.get("peak_host_rss_bytes"),
                   peak_cuda_allocated_bytes=data.get("peak_cuda_allocated_bytes"),
                   peak_cuda_reserved_bytes=data.get("peak_cuda_reserved_bytes"))

    @classmethod
    def from_json(cls, text: str) -> "BenchmarkResult":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BenchSchemaError(f"benchmark result is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


def _check_model_list(model: Any) -> list[str]:
    errors: list[str] = []
    if model is None:
        model = ()
    if isinstance(model, (str, bytes, Mapping)) or not isinstance(model, (list, tuple)):
        _bad(errors, "model", f"must be a list of artifact identifier objects, got {model!r}")
        return errors
    if not model:
        _bad(errors, "model", "must name at least the artifacts used "
                              "(from model_identifiers(manifest))")
    for i, m in enumerate(model):
        if not isinstance(m, Mapping):
            _bad(errors, f"model[{i}]", f"must be an object, got {m!r}")
            continue
        for key in sorted(set(m) - {"role", "repo", "revision", "filename", "sha256"}):
            _bad(errors, f"model[{i}].{key}", "unknown model identifier key")
        for key in ("role", "repo", "filename"):
            value = m.get(key)
            if not isinstance(value, str) or not value.strip():
                _bad(errors, f"model[{i}].{key}", f"must be non-empty, got {value!r}")
        rev = m.get("revision")
        if not isinstance(rev, str) or not _REVISION_RE.fullmatch(rev):
            _bad(errors, f"model[{i}].revision",
                 f"must be a pinned 40-hex commit sha, got {rev!r}")
        sha = m.get("sha256")
        if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
            _bad(errors, f"model[{i}].sha256",
                 f"must be 64 lowercase hex chars, got {sha!r}")
    return errors


def validate_result_fields(res: "BenchmarkResult") -> list[str]:
    errors: list[str] = []
    if res.schema != BENCHMARK_SCHEMA:
        _bad(errors, "schema", f"must be {BENCHMARK_SCHEMA!r}, got {res.schema!r}")
    if not isinstance(res.run_id, str) or not res.run_id.strip():
        _bad(errors, "run_id", f"must be a non-empty string, got {res.run_id!r}")
    if not isinstance(res.environment, BenchmarkEnvironment):
        _bad(errors, "environment",
             f"required (nulls inside it are fine), got {res.environment!r}")
    errors.extend(_check_model_list(list(res.model) if not isinstance(res.model,
                                                                      (str, bytes))
                                    else res.model))
    if not isinstance(res.config, Mapping):
        _bad(errors, "config", f"must be a mapping, got {type(res.config).__name__}")
    else:
        try:
            json.dumps(dict(res.config), sort_keys=True)
        except (TypeError, ValueError) as exc:
            _bad(errors, "config", f"config echo must be JSON-serializable: {exc}")
    if not isinstance(res.phases, Mapping):
        _bad(errors, "phases", f"must be a mapping, got {type(res.phases).__name__}")
    elif not res.phases:
        _bad(errors, "phases", "must be a non-empty mapping of phase -> wall_ms")
    else:
        for name, value in sorted(res.phases.items()):
            if not isinstance(name, str) or not name.strip():
                _bad(errors, "phases", f"phase name must be non-empty, got {name!r}")
            elif not _finite_nonneg(value):
                _bad(errors, f"phases.{name}",
                     f"wall time must be a finite number >= 0 ms, got {value!r}")
    if res.steps is not None and (not isinstance(res.steps, int)
                                  or isinstance(res.steps, bool) or res.steps < 1):
        _bad(errors, "steps", f"must be a positive integer or null, got {res.steps!r}")
    for name in ("peak_host_rss_bytes", "peak_cuda_allocated_bytes",
                 "peak_cuda_reserved_bytes"):
        _check_optional_bytes(getattr(res, name), name, errors)
    return errors


def save_result(result: BenchmarkResult, directory: str | Path,
                run_id: str | None = None) -> Path:
    """Atomically write ``<directory>/<run_id>.benchmark.json`` (temp file +
    ``os.replace``; directory created — this is a configured bench dir)."""
    target_dir = Path(directory)
    rid = run_id if run_id is not None else result.run_id
    if not isinstance(rid, str) or not rid.strip():
        raise BenchSchemaError(f"run_id for file name: must be non-empty, got {rid!r}")
    if "/" in rid or "\0" in rid or rid.startswith("."):
        raise BenchSchemaError(f"run_id for file name: unsafe, got {rid!r}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{rid}.benchmark.json"
    fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".bench-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def load_result(path: str | Path) -> BenchmarkResult:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise BenchSchemaError(f"benchmark file not readable: {p}: {exc}") from exc
    try:
        return BenchmarkResult.from_json(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise BenchSchemaError(f"benchmark file is not valid UTF-8: {p}: {exc}") from exc


# ---------------------------------------------------------------------------
# Deterministic aggregation over (synthetic or real) runs
# ---------------------------------------------------------------------------


def _stats(values: Sequence[float]) -> dict[str, Any]:
    vals = sorted(float(v) for v in values)
    return {"n": len(vals),
            "min": vals[0],
            "median": float(statistics.median(vals)),
            "max": vals[-1]}


def _optional_stats(values: Sequence[int]) -> dict[str, Any]:
    present = [int(v) for v in values if v is not None]
    if not present:
        # null means *not measured* — it must never aggregate into a 0.
        return {"n_measured": 0, "max": None, "median": None}
    present.sort()
    return {"n_measured": len(present), "max": present[-1],
            "median": int(statistics.median(present))}


def aggregate_results(results: Sequence[BenchmarkResult]) -> dict[str, Any]:
    """Deterministic aggregation (``benchmark_aggregate.v1``): input order
    cannot change the output, phase names sort, and unmeasured memory stays
    null instead of becoming zero."""
    items = list(results)
    for r in items:
        if not isinstance(r, BenchmarkResult):
            raise BenchSchemaError(
                f"aggregate expects BenchmarkResult, got {type(r).__name__}")
    phases: dict[str, dict[str, Any]] = {}
    phase_names = sorted({name for r in items for name in r.phases})
    for name in phase_names:
        values = [r.phases[name] for r in items if name in r.phases]
        phases[name] = _stats(values)
    per_step = [r.denoise_ms_per_step for r in items]
    per_step = [v for v in per_step if v is not None]
    return {
        "schema": AGGREGATE_SCHEMA,
        "n_runs": len(items),
        "run_ids": sorted(r.run_id for r in items),
        "phases": phases,
        "denoise_ms_per_step": (_stats(per_step) if per_step
                                else {"n": 0, "min": None, "median": None, "max": None}),
        "peak_host_rss_bytes": _optional_stats(
            [r.peak_host_rss_bytes for r in items]),
        "peak_cuda_allocated_bytes": _optional_stats(
            [r.peak_cuda_allocated_bytes for r in items]),
        "peak_cuda_reserved_bytes": _optional_stats(
            [r.peak_cuda_reserved_bytes for r in items]),
    }


# ---------------------------------------------------------------------------
# Phase timing collector (harness skeleton)
# ---------------------------------------------------------------------------


@dataclass
class PhaseCollector:
    """Accumulate wall-clock ms per phase name; injectable clock makes test
    timings exact. Repeated phases of the same name accumulate (denoise steps
    into ``denoise``)."""

    clock: Callable[[], float] = time.monotonic
    _timings: dict[str, float] = field(default_factory=dict, repr=False)

    def phase(self, name: str):
        if not isinstance(name, str) or not name.strip():
            raise BenchSchemaError(f"phase name: must be non-empty, got {name!r}")
        return self._phase(name)

    @contextmanager
    def _phase(self, name: str):
        start = self.clock()
        try:
            yield self
        finally:
            self._timings[name] = self._timings.get(name, 0.0) + (self.clock() - start) * 1000.0

    def timings(self) -> dict[str, float]:
        return {k: self._timings[k] for k in sorted(self._timings)}


# ---------------------------------------------------------------------------
# calibration observation schema (feeds M5.3 staleness checks)
# ---------------------------------------------------------------------------

_OBS_KEYS = {"schema", "observed_at", "phase", "model_revision", "backend",
             "quant", "residency", "vae_tiled_decode", "cfg_duplicated",
             "tokens", "resolution", "peak_gpu_allocated_bytes",
             "peak_gpu_reserved_bytes", "peak_host_rss_bytes"}


@dataclass(frozen=True)
class CalibrationObservation:
    """One memory observation keyed for staleness (R4): the memory policy
    matches on ``(schema, model_revision, backend, quant)`` — any mismatch
    makes an observation stale, never interpolated across. ``residency``,
    ``vae_tiled_decode``, ``cfg_duplicated``, ``tokens`` and ``resolution``
    are descriptor fields the policy conditions on; the peak-memory fields
    are the measurements (null = not measured, never 0)."""

    observed_at: float
    phase: str
    model_revision: str
    backend: str
    quant: str
    residency: str
    vae_tiled_decode: bool = False
    cfg_duplicated: bool = False
    tokens: int = 0
    resolution: tuple[int, int] = (0, 0)
    peak_gpu_allocated_bytes: int | None = None
    peak_gpu_reserved_bytes: int | None = None
    peak_host_rss_bytes: int | None = None
    schema: str = OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        errors = validate_observation_fields(self)
        if errors:
            raise BenchSchemaError("invalid calibration observation: "
                                   + "; ".join(errors))

    @property
    def staleness_key(self) -> tuple[str, str, str, str]:
        return (self.schema, self.model_revision, self.backend, self.quant)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "observed_at": self.observed_at,
            "phase": self.phase, "model_revision": self.model_revision,
            "backend": self.backend, "quant": self.quant,
            "residency": self.residency,
            "vae_tiled_decode": self.vae_tiled_decode,
            "cfg_duplicated": self.cfg_duplicated, "tokens": self.tokens,
            "resolution": list(self.resolution),
            "peak_gpu_allocated_bytes": self.peak_gpu_allocated_bytes,
            "peak_gpu_reserved_bytes": self.peak_gpu_reserved_bytes,
            "peak_host_rss_bytes": self.peak_host_rss_bytes,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CalibrationObservation":
        errors: list[str] = []
        if not isinstance(data, Mapping):
            raise BenchSchemaError(
                f"observation: must be an object, got {type(data).__name__}")
        for key in sorted(set(data) - _OBS_KEYS):
            _bad(errors, key, "unknown observation key")
        res = data.get("resolution")
        if not isinstance(res, list) or len(res) != 2:
            _bad(errors, "resolution", f"must be [width, height], got {res!r}")
            res = (0, 0)
        if errors:
            raise BenchSchemaError("invalid calibration observation: "
                                   + "; ".join(errors))
        return cls(observed_at=data.get("observed_at"), phase=data.get("phase"),
                   model_revision=data.get("model_revision"),
                   backend=data.get("backend"), quant=data.get("quant"),
                   residency=data.get("residency"),
                   vae_tiled_decode=data.get("vae_tiled_decode", False),
                   cfg_duplicated=data.get("cfg_duplicated", False),
                   tokens=data.get("tokens", 0),
                   resolution=tuple(res),
                   peak_gpu_allocated_bytes=data.get("peak_gpu_allocated_bytes"),
                   peak_gpu_reserved_bytes=data.get("peak_gpu_reserved_bytes"),
                   peak_host_rss_bytes=data.get("peak_host_rss_bytes"),
                   schema=data.get("schema", OBSERVATION_SCHEMA))


def validate_observation_fields(obs: CalibrationObservation) -> list[str]:
    errors: list[str] = []
    if obs.schema != OBSERVATION_SCHEMA:
        _bad(errors, "schema", f"must be {OBSERVATION_SCHEMA!r}, got {obs.schema!r}")
    if not _is_number(obs.observed_at) or not math.isfinite(obs.observed_at) \
            or obs.observed_at <= 0:
        _bad(errors, "observed_at",
             f"must be a finite positive epoch seconds, got {obs.observed_at!r}")
    for name in ("phase", "backend", "quant", "residency"):
        value = getattr(obs, name)
        if not isinstance(value, str) or not value.strip():
            _bad(errors, name, f"must be a non-empty string, got {value!r}")
    if not isinstance(obs.model_revision, str) \
            or not _REVISION_RE.fullmatch(obs.model_revision):
        _bad(errors, "model_revision",
             f"must be a pinned 40-hex commit sha, got {obs.model_revision!r}")
    for name in ("vae_tiled_decode", "cfg_duplicated"):
        if not isinstance(getattr(obs, name), bool):
            _bad(errors, name, f"must be a boolean, got {getattr(obs, name)!r}")
    if not isinstance(obs.tokens, int) or isinstance(obs.tokens, bool) or obs.tokens < 1:
        _bad(errors, "tokens", f"must be a positive integer, got {obs.tokens!r}")
    res = obs.resolution
    if (isinstance(res, (str, bytes)) or len(tuple(res)) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1
                       for v in res)):
        _bad(errors, "resolution", f"must be two positive ints, got {res!r}")
    for name in ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes",
                 "peak_host_rss_bytes"):
        _check_optional_bytes(getattr(obs, name), name, errors)
    return errors


@dataclass(frozen=True)
class CalibrationLog:
    """The whole ``calibration.json`` document."""

    observations: tuple[CalibrationObservation, ...] = ()

    def __post_init__(self) -> None:
        bad = [type(o).__name__ for o in self.observations
               if not isinstance(o, CalibrationObservation)]
        if bad:
            raise BenchSchemaError(
                "observations: every entry must be a CalibrationObservation, got "
                + ", ".join(bad))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CALIBRATION_FILE_SCHEMA,
                "observations": [o.to_dict() for o in self.observations]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Any) -> "CalibrationLog":
        errors: list[str] = []
        if not isinstance(data, Mapping):
            raise BenchSchemaError(
                f"calibration: must be an object, got {type(data).__name__}")
        for key in sorted(set(data) - {"schema", "observations"}):
            _bad(errors, key, "unknown calibration key")
        if data.get("schema") != CALIBRATION_FILE_SCHEMA:
            _bad(errors, "schema",
                 f"must be {CALIBRATION_FILE_SCHEMA!r}, got {data.get('schema')!r}")
        raw = data.get("observations")
        observations: list[CalibrationObservation] = []
        if not isinstance(raw, list):
            _bad(errors, "observations", f"must be a list, got {raw!r}")
        else:
            for i, entry in enumerate(raw):
                try:
                    observations.append(CalibrationObservation.from_dict(entry))
                except BenchSchemaError as exc:
                    _bad(errors, f"observations[{i}]", str(exc))
        if errors:
            raise BenchSchemaError("invalid calibration file: " + "; ".join(errors))
        return cls(observations=tuple(observations))


def load_calibration(path: str | Path) -> CalibrationLog:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise BenchSchemaError(f"calibration file not readable: {p}: {exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchSchemaError(f"calibration file is not valid UTF-8 JSON: {p}: {exc}") from exc
    return CalibrationLog.from_dict(data)


def save_calibration(log: CalibrationLog, path: str | Path) -> Path:
    """Atomic write (temp + ``os.replace``) of the whole document."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".calib-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(log.to_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def add_observation(path: str | Path, observation: CalibrationObservation) -> CalibrationLog:
    """Load-append-atomic-save; a missing file starts a fresh log."""
    try:
        log = load_calibration(path)
        observations = list(log.observations)
    except BenchSchemaError as exc:
        if Path(path).exists():
            raise
        observations = []
    observations.append(observation)
    log = CalibrationLog(observations=tuple(observations))
    save_calibration(log, path)
    return log


def matching_observations(log: CalibrationLog, *, model_revision: str,
                          backend: str, quant: str) -> tuple[CalibrationObservation, ...]:
    """Fresh-by-staleness-key selection (M5.3 consumes this): an observation
    counts only when schema, model revision, backend and quant all match —
    never interpolated across a mismatch."""
    key = (OBSERVATION_SCHEMA, model_revision, backend, quant)
    return tuple(o for o in log.observations if o.staleness_key == key)
