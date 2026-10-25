"""ML-free core for the standalone VAE tile-geometry decode benchmark.

Why this module exists
----------------------
The current 2048² decode path (FP32 tiled VAE, 1024 px tiles / 768 px stride)
peaks at a measured 15.57 GiB with allocator retry on the 16 GiB V100 — near
OOM. Fixing it means comparing real tile geometries (1024/768, 768/576,
512/384) over the **identical saved real latent tensor**, with seams, alpha and
memory/time all measured per geometry. Nothing in this module claims a smaller
geometry preserves quality; that claim requires the actual images the script
produces.

Boundary rules
--------------
* Importing this module is **ML-free**: ``torch``, ``PIL``, ``numpy``,
  ``psutil``, ``diffusers`` and NVML are never imported here. The GPU script
  imports this module and owns the heavy stack itself.
* All refusals use the project's existing typed errors (:class:`ConfigError`
  for malformed inputs, :class:`EngineError` for state/observation failures).
* The benchmark report is its OWN strict-JSON document. The production
  ``CalibrationObservation`` is deliberately NOT reused: it has no tile
  geometry field, so reusing it would silently mislabel geometry measurements.
* Byte fields are length-checked; JSON is emitted with
  ``allow_nan=False`` so a non-finite metric cannot leave the process.
* Every artifact write is collision-checked (``O_EXCL``) or atomic
  (temp + ``os.replace``); nothing is ever silently overwritten.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from qwen_engine.config import ConfigError
from qwen_engine.errors import EngineError
from qwen_engine._jsonsafe import strict_json_loads

#: Benchmark schema tag and version; every emitted document carries both.
BENCHMARK_SCHEMA = "qwen.decode-bench"
BENCHMARK_VERSION = 1

#: The reviewed baseline geometry (production constant) plus the smaller
#: geometries worth comparing against it, as validated records.

#: Hard caps on benchmark document size (bytes), mirroring the bounded-buffer
#: style of the production PNG cap.
MAX_REPORT_BYTES = 4 * 1024**2
MAX_IMAGE_BUFFER_BYTES = 4096 * 4096 * 4
MAX_SNAPSHOT_METADATA_BYTES = 64 * 1024

_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def utc_now() -> str:
    """Second-resolution UTC stamp; every observation carries its own."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str, chunk: int = 1024**2) -> str:
    """Streaming SHA-256 of one file's bytes (bounded memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of an in-memory buffer (raw tensor bytes binding)."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Collision-safe artifact writes (no output is ever silently overwritten)
# ---------------------------------------------------------------------------


def exclusive_write_bytes(path: Any, payload: bytes) -> None:
    """Write ``payload`` to a brand-new ``path``; refuse any collision.

    ``O_EXCL`` means an existing snapshot/PNG is never replaced: a second
    capture into the same directory fails typed instead of destroying the
    evidence the first run produced.
    """
    if os.path.exists(path):
        raise EngineError(f"refusing to overwrite existing artifact: {path}")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(str(path))
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Any, payload: bytes, *, prefix: str = "decode-bench") -> None:
    """Publish ``payload`` at ``path`` via temp-file + ``os.replace`` (fsynced).

    Same durability pattern as the production benchmark persistence helper;
    a reader never observes a half-written report.
    """
    path = str(path)
    parent = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=prefix + ".", suffix=".tmp")
    handle = None
    try:
        handle = os.fdopen(fd, "wb")
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    finally:
        if handle is None:
            try:
                os.close(fd)
            except OSError:
                pass


def check_output_collisions(out_dir: Any, names: Any) -> None:
    """Refuse a run whose declared artifact names already exist in ``out_dir``."""
    clashes = sorted(name for name in names if os.path.exists(os.path.join(str(out_dir), name)))
    if clashes:
        raise EngineError(
            "output directory already holds benchmark artifacts; refusing to "
            f"overwrite: {', '.join(clashes)}; choose a fresh --output-dir")


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileGeometry:
    """One validated tiled-decode geometry (tile px, stride px)."""

    tile: int
    stride: int

    @property
    def label(self) -> str:
        return f"tile{self.tile}-stride{self.stride}"


def validate_geometry(tile: Any, stride: Any) -> TileGeometry:
    """Refuse any geometry the pinned VAE tiling could not be called with.

    Exact positive ``int`` only (bools and floats are refusals, not
    coercions), multiples of 16 (the VAE spatial scale factor), stride no
    larger than the tile (a stride above the tile would skip pixels), and a
    stride large enough that blending is defined (>= 16).
    """
    errors = []
    for name, value in (("tile", tile), ("stride", stride)):
        if type(value) is not int:
            errors.append(f"{name} must be an exact int, got {value!r}")
        elif value <= 0:
            errors.append(f"{name} must be a positive int, got {value}")
        elif value % 16 != 0:
            errors.append(f"{name} must be a multiple of 16, got {value}")
    if not errors:
        if stride < 16:
            errors.append(f"stride must be at least 16, got {stride}")
        elif stride > tile:
            errors.append(f"stride {stride} must not exceed tile {tile}")
    if errors:
        raise ConfigError("invalid tile geometry: " + "; ".join(errors))
    return TileGeometry(tile=tile, stride=stride)


def validate_geometries(entries: Any) -> tuple[TileGeometry, ...]:
    """Validate a geometry list: non-empty, all valid, no duplicate labels."""
    if type(entries) is not tuple and type(entries) is not list:
        raise ConfigError("geometries must be a list/tuple of (tile, stride) pairs")
    if not entries:
        raise ConfigError("at least one tile geometry is required")
    geometries = []
    for entry in entries:
        if isinstance(entry, TileGeometry):
            entry = (entry.tile, entry.stride)
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ConfigError(f"geometry entry must be a (tile, stride) pair: {entry!r}")
        geometries.append(validate_geometry(entry[0], entry[1]))
    labels = [geometry.label for geometry in geometries]
    if len(set(labels)) != len(labels):
        raise ConfigError("duplicate tile geometries are refused")
    return tuple(geometries)


DEFAULT_GEOMETRIES = tuple(validate_geometry(*pair) for pair in
                           ((1024, 768), (768, 576), (512, 384)))


def validate_label(value: Any, field: str = "label") -> str:
    """Path-injection-free label: one safe token, never a path or control text."""
    if type(value) is not str or _LABEL_RE.fullmatch(value) is None:
        raise ConfigError(
            f"{field} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}; got {value!r}")
    return value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{field} must be a positive int, got {value!r}")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{field} must be finite, got {value!r}")
    return number


def _byte_field(value: Any, field: str, cap: int) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise ConfigError(f"{field} must be bytes")
    if len(value) == 0 or len(value) > cap:
        raise ConfigError(f"{field} length {len(value)} is outside (0, {cap}]")
    return bytes(value)


def _sha(value: Any, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ConfigError(f"{field} must be a lowercase 64-hex SHA-256 digest")
    return value


# ---------------------------------------------------------------------------
# Memory observation (fresh-allocator peak vs NVML device-wide, distinguished)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryObservation:
    """One profile's memory evidence, with every source stated explicitly.

    ``peak_allocated_bytes`` / ``peak_reserved_bytes`` are this process's
    PyTorch allocator high-water marks (freshly reset before the profile,
    after the allocator cache was emptied so the previous profile's cached
    blocks cannot contaminate the next measurement).
    ``nvml_peak_used_bytes`` is the NVML *device-wide* used maximum sampled on
    an interval — it includes the CUDA context and any other process on the
    card, so it is never interchangeable with the allocator numbers. Sampled
    peaks can miss spikes shorter than the interval; ``samples`` and
    ``interval_ms`` state that coverage limit honestly.
    ``allocator_retries`` is the ACTUAL delta of ``torch.cuda.memory_stats()
    ['num_alloc_retries']`` measured across the profile, not a placeholder.
    """

    peak_allocated_bytes: int
    peak_reserved_bytes: int
    nvml_peak_used_bytes: int | None
    nvml_total_bytes: int | None
    nvml_driver_version: str | None
    interval_ms: float
    samples: int
    allocator_retries: int
    device_uuid: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "nvml_peak_used_bytes": self.nvml_peak_used_bytes,
            "nvml_total_bytes": self.nvml_total_bytes,
            "nvml_driver_version": self.nvml_driver_version,
            "nvml_scope": ("device-wide sampled maximum; includes CUDA context "
                           "and other processes" if self.nvml_peak_used_bytes is not None
                           else "nvml unavailable; allocator peaks only"),
            "interval_ms": self.interval_ms,
            "samples": self.samples,
            "allocator_retries": self.allocator_retries,
            "allocator_retries_source": ("torch.cuda.memory_stats num_alloc_retries delta"
                                         if self.allocator_retries >= 0
                                         else "num_alloc_retries key unavailable"),
            "device_uuid": self.device_uuid,
        }


def make_memory_observation(
    *,
    peak_allocated_bytes: Any,
    peak_reserved_bytes: Any,
    interval_ms: Any,
    samples: Any,
    allocator_retries: Any,
    nvml_peak_used_bytes: Any = None,
    nvml_total_bytes: Any = None,
    nvml_driver_version: Any = None,
    device_uuid: Any = None,
) -> MemoryObservation:
    """Validate and freeze one memory observation (all ints exact, finite)."""
    peak_allocated = _positive_int(peak_allocated_bytes, "peak_allocated_bytes")
    peak_reserved = _positive_int(peak_reserved_bytes, "peak_reserved_bytes")
    if peak_reserved < peak_allocated:
        raise ConfigError("reserved peak cannot be below the allocated peak")
    interval = _finite(interval_ms, "interval_ms")
    if not 5.0 <= interval <= 1000.0:
        raise ConfigError(f"interval_ms must be in [5, 1000], got {interval}")
    if type(samples) is not int or samples < 0:
        raise ConfigError("samples must be a nonnegative integer")
    sample_count = samples
    if nvml_peak_used_bytes is not None and samples == 0:
        raise ConfigError("a sampled NVML peak requires at least one observation")
    if type(allocator_retries) is not int or allocator_retries < 0:
        raise ConfigError(
            f"allocator_retries must be a non-negative int, got {allocator_retries!r}")
    if nvml_peak_used_bytes is not None:
        nvml_peak_used_bytes = _positive_int(nvml_peak_used_bytes, "nvml_peak_used_bytes")
    if nvml_total_bytes is not None:
        nvml_total_bytes = _positive_int(nvml_total_bytes, "nvml_total_bytes")
    if nvml_driver_version is not None and type(nvml_driver_version) is not str:
        raise ConfigError("nvml_driver_version must be a string or None")
    if device_uuid is not None and type(device_uuid) is not str:
        raise ConfigError("device_uuid must be a string or None")
    return MemoryObservation(
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        nvml_peak_used_bytes=nvml_peak_used_bytes,
        nvml_total_bytes=nvml_total_bytes,
        nvml_driver_version=nvml_driver_version,
        interval_ms=interval,
        samples=sample_count,
        allocator_retries=allocator_retries,
        device_uuid=device_uuid,
    )


def allocator_retry_delta(before: Any, after: Any) -> int:
    """Actual ``num_alloc_retries`` delta between two ``memory_stats()`` reads.

    Both mappings must expose the exact ``num_alloc_retries`` key with a
    non-negative int; a missing key is a typed refusal, never a silent zero.
    A negative delta (stats reset underneath us) is also refused.
    """
    if type(before) is not dict or type(after) is not dict:
        raise EngineError("allocator retry stats must be memory_stats() dicts")
    try:
        first = before["num_alloc_retries"]
        second = after["num_alloc_retries"]
    except KeyError as exc:
        raise EngineError(
            f"torch.cuda.memory_stats() is missing {exc.args[0]!r}; "
            "allocator retries cannot be measured") from None
    for name, value in (("before", first), ("after", second)):
        if type(value) is not int or value < 0:
            raise EngineError(f"memory_stats num_alloc_retries ({name}) is malformed: {value!r}")
    if second < first:
        raise EngineError("num_alloc_retries went backwards; stats were reset mid-profile")
    return second - first


class PeakMemoryProbe:
    """One-shot allocator peak + optional NVML device-wide sampler.

    Before resetting peak stats the probe drops its own CUDA cache
    (``synchronize`` + ``empty_cache``) so the previous profile's cached
    blocks — the production decode leaves ~15 GiB reserved in the allocator
    cache — cannot inflate this profile's NVML device-wide peak.

    The NVML session uses the reviewed private ``_NvmlDevice`` from
    ``calibration`` **only** after its interface is verified against this
    exact source tree (constructor ``(uuid, total_bytes)``, ``used_bytes()``,
    ``close()``, ``driver_version``, ``total_bytes``); the import is lazy so
    this module stays ML-free and CPU tests never touch NVML. If the interface
    drifted or NVML is absent, the probe degrades to allocator-only and the
    report states that the device-wide peak was not measured — it never
    fabricates one.

    ``interval_ms`` is a declared sampling interval, not a safety budget: a
    device-wide spike shorter than the interval can be missed and the report
    says so via ``samples``/``interval_ms``.
    """

    def __init__(self, *, device_uuid: str, device_total_bytes: int,
                 interval_ms: float = 50.0, torch_module=None,
                 allow_nvml: bool = True):
        # torch is resolved lazily in start(): constructing a probe must never
        # import torch (CPU tests build probes to exercise validation only).
        self._torch = torch_module
        self._interval_s = _finite(interval_ms, "interval_ms") / 1000.0
        if not 0.005 <= self._interval_s <= 1.0:
            raise ConfigError("interval_ms must be in [5, 1000]")
        if type(device_uuid) is not str or not device_uuid:
            raise ConfigError("device_uuid must be a non-empty string")
        self._device_uuid = device_uuid
        self._total_bytes = _positive_int(device_total_bytes, "device_total_bytes")
        self._allow_nvml = allow_nvml is True
        self._nvml = None
        self._nvml_error: str | None = None
        self._thread = None
        self._stop = None
        self._lock = None
        self._nvml_peak: int | None = None
        self._samples = 0
        self._active = False
        self._closed = False
        self._peak_allocated = 0
        self._peak_reserved = 0
        self._retries_before = None

    @staticmethod
    def _resolve_torch(torch_module):
        if torch_module is not None:
            return torch_module
        try:
            import torch  # lazy: only the GPU script ever constructs this probe
        except ImportError as exc:
            raise EngineError("CUDA memory measurement requires PyTorch") from exc
        return torch

    def start(self) -> None:
        """Free this profile's starting point, then start the sampler thread."""
        if self._active or self._closed:
            raise EngineError("memory probe lifecycle violation")
        import gc
        import threading

        torch = self._resolve_torch(self._torch)
        self._torch = torch
        if not torch.cuda.is_available():
            raise EngineError("CUDA is required for a memory probe")
        gc.collect()
        torch.cuda.synchronize()
        # Empty the allocator cache FIRST: a previous profile's cached blocks
        # would otherwise sit in the NVML device-wide peak this probe samples.
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stats = dict(torch.cuda.memory_stats())
        if type(stats) is not dict or "num_alloc_retries" not in stats:
            raise EngineError(
                "torch.cuda.memory_stats() lacks num_alloc_retries; retries cannot be measured")
        self._retries_before = stats
        self._stop = threading.Event()
        self._lock = threading.Lock()
        if self._allow_nvml:
            self._nvml = self._open_nvml()
            if self._nvml is None:
                raise EngineError(self._nvml_error or "NVML measurement is unavailable")
            self._nvml_peak = self._nvml.used_bytes()
            self._samples = 1
        self._active = True
        self._thread = threading.Thread(target=self._sample_loop,
                                        name="decode-bench-nvml", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._active = False
            self._thread = None
            if self._nvml is not None:
                self._nvml.close()
                self._nvml = None
            self._closed = True
            raise

    def _open_nvml(self):
        try:
            from qwen_engine import calibration

            device_cls = getattr(calibration, "_NvmlDevice", None)
            if (device_cls is None
                    or not callable(getattr(device_cls, "used_bytes", None))
                    or not callable(getattr(device_cls, "close", None))):
                self._nvml_error = "private NVML interface drifted; device-wide peak unmeasured"
                return None
            device = device_cls(self._device_uuid, self._total_bytes)
            if not isinstance(device.used_bytes(), int):
                self._nvml_error = "NVML used_bytes returned a non-int"
                device.close()
                return None
            return device
        except Exception as exc:  # absent libnvidia-ml, bad UUID, ...
            self._nvml_error = f"NVML session unavailable: {type(exc).__name__}"
            return None

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            used = None
            if self._nvml is not None:
                try:
                    used = self._nvml.used_bytes()
                except Exception:
                    self._nvml_error = "NVML sampling failed mid-profile"
                    self._stop.set()
            with self._lock:
                self._samples += 1
                if used is not None and (self._nvml_peak is None or used > self._nvml_peak):
                    self._nvml_peak = used
            self._stop.wait(self._interval_s)

    def finish(self, *, device_uuid: str | None = None) -> MemoryObservation:
        """Synchronize, read allocator peaks + actual retries, close everything."""
        if self._closed:
            raise EngineError("memory probe is closed")
        torch = self._torch
        retries_after = None
        try:
            if self._active:
                torch.cuda.synchronize()
                self._peak_allocated = int(torch.cuda.max_memory_allocated())
                self._peak_reserved = int(torch.cuda.max_memory_reserved())
                retries_after = dict(torch.cuda.memory_stats())
        finally:
            # Lifecycle teardown runs even when the measured phase raised.
            self._active = False
            if self._stop is not None:
                self._stop.set()
            thread, self._thread = self._thread, None
            if thread is not None:
                thread.join(timeout=10.0)
                if thread.is_alive():
                    self._thread = thread
                    raise EngineError("NVML sampler cleanup is pending; session retained")
            nvml, self._nvml = self._nvml, None
            if nvml is not None:
                nvml.close()
            self._closed = True
        with self._lock:
            samples, nvml_peak = self._samples, self._nvml_peak
        driver = getattr(nvml, "driver_version", None) if nvml is not None else None
        total = getattr(nvml, "total_bytes", self._total_bytes) if nvml is not None \
            else self._total_bytes
        if self._retries_before is not None and type(retries_after) is dict:
            retries = allocator_retry_delta(self._retries_before, retries_after)
        else:
            # The measured phase raised before stats could be read; report the
            # unmeasurable state honestly instead of a fake zero.
            raise EngineError("allocator retries could not be measured")
        if self._nvml_error is not None:
            raise EngineError(self._nvml_error)
        return make_memory_observation(
            peak_allocated_bytes=self._peak_allocated,
            peak_reserved_bytes=self._peak_reserved,
            nvml_peak_used_bytes=nvml_peak,
            nvml_total_bytes=total,
            nvml_driver_version=driver,
            interval_ms=self._interval_s * 1000.0,
            samples=samples,
            allocator_retries=retries,
            device_uuid=device_uuid if device_uuid is not None else self._device_uuid,
        )

    @property
    def nvml_error(self) -> str | None:
        return self._nvml_error


# ---------------------------------------------------------------------------
# Image comparison (pure byte math; PIL/torch never imported here)
# ---------------------------------------------------------------------------


def compare_rgb_bytes(reference: Any, candidate: Any, *, channels: int = 3) -> dict[str, Any]:
    """Quantitative per-channel RGB(A) difference between two raw byte buffers.

    Both buffers must be the exact same length and channel count; every metric
    is a finite integer/float derived from ``abs(reference - candidate)``.
    ``max_channel_delta`` is on the 0..255 scale.
    """
    reference = _byte_field(reference, "reference_bytes", MAX_IMAGE_BUFFER_BYTES)
    candidate = _byte_field(candidate, "candidate_bytes", MAX_IMAGE_BUFFER_BYTES)
    if channels not in (3, 4):
        raise ConfigError("channels must be 3 (RGB) or 4 (RGBA)")
    if len(reference) != len(candidate):
        raise ConfigError("image byte buffers differ in length; not comparable")
    if len(reference) % channels != 0:
        raise ConfigError("image byte length is not a whole number of pixels")
    if len(reference) == 0:
        raise ConfigError("empty image buffers are not comparable")
    max_delta = 0
    total = 0
    changed = 0
    last_changed_pixel = -1
    per_channel_max = [0] * channels
    for index, (left, right) in enumerate(zip(reference, candidate)):
        delta = left - right
        delta = delta if delta >= 0 else -delta
        total += delta
        if delta:
            pixel = index // channels
            if pixel != last_changed_pixel:
                changed += 1
                last_changed_pixel = pixel
            if delta > max_delta:
                max_delta = delta
            slot = index % channels
            if delta > per_channel_max[slot]:
                per_channel_max[slot] = delta
    pixels = len(reference) // channels
    return {
        "channels": channels,
        "pixels": pixels,
        "max_channel_delta": max_delta,
        "mean_channel_delta": total / len(reference),
        "changed_pixel_fraction": changed / pixels,
        "per_channel_max_delta": per_channel_max,
    }


def alpha_stats(rgba_bytes: Any) -> dict[str, Any]:
    """Alpha-channel summary of one RGBA raw buffer (all-equal is common)."""
    data = _byte_field(rgba_bytes, "rgba_bytes", MAX_IMAGE_BUFFER_BYTES)
    if len(data) % 4 != 0 or not data:
        raise ConfigError("rgba byte length is not a whole number of pixels")
    alphas = data[3::4]
    first = alphas[0]
    uniform = True
    minimum = maximum = int(first)
    total = 0
    for value in alphas:
        value = int(value)
        total += value
        if value < minimum:
            minimum = value
        if value > maximum:
            maximum = value
        if value != first:
            uniform = False
    return {
        "alpha_uniform": uniform,
        "alpha_min": minimum,
        "alpha_max": maximum,
        "alpha_mean": total / len(alphas),
    }


# ---------------------------------------------------------------------------
# Snapshot metadata (binds the saved latent to the run that produced it)
# ---------------------------------------------------------------------------

_SNAPSHOT_KEYS = (
    "schema", "seed", "width", "height", "steps", "quant", "semantic_config_sha256",
    "artifact_set_sha256", "runtime_identity_sha256", "capture_geometry",
    "capture_tile", "capture_stride", "latent_shape", "latent_dtype",
    "latent_sha256", "observed_at",
)


def build_snapshot_metadata(*, seed: Any, width: Any, height: Any, steps: Any,
                            quant: Any, semantic_config_sha256: Any,
                            artifact_set_sha256: Any, runtime_identity_sha256: Any,
                            capture_geometry: Any, latent_shape: Any,
                            latent_dtype: Any, latent_sha256: Any,
                            observed_at: str | None = None) -> dict[str, Any]:
    """Exact, digest-bound metadata stored beside the captured latent.

    ``semantic_config_sha256`` must be the digest of the SEMANTIC generation
    parameters only (prompt/size/steps/seed/quant) — never the runtime paths —
    so a snapshot captured in one directory replays into a fresh output
    directory while still refusing a different generation. Artifact and
    runtime identities are bound as SEPARATE digests.

    ``latent_sha256`` must be the SHA-256 of the RAW contiguous CPU float16
    tensor bytes computed BEFORE the file was written (see
    :func:`raw_tensor_sha256`); it binds the tensor itself, not the file that
    happens to contain it.
    """
    if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
        raise ConfigError("snapshot seed must be an exact uint64 integer")
    if (type(width) is not int or type(height) is not int
            or any(not 64 <= n <= 4096 or n % 32 for n in (width, height))):
        raise ConfigError("snapshot dimensions must match the engine's aligned size range")
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ConfigError("snapshot steps must be an exact integer in [1,1000]")
    if quant not in ('fp16', 'q8') or latent_dtype != 'float16':
        raise ConfigError("snapshot must use a supported quant and float16 latents")
    if (type(latent_shape) not in (tuple, list)
            or tuple(latent_shape) != (1, (width // 16) * (height // 16), 64)):
        raise ConfigError("snapshot latent shape must match its image size")
    document = {
        "schema": BENCHMARK_SCHEMA,
        "seed": _positive_int(seed, "seed") if seed != 0 else 0,
        "width": validate_geometry(width, 16).tile,
        "height": validate_geometry(height, 16).tile,
        "steps": _positive_int(steps, "steps"),
        "quant": validate_label(quant, "quant"),
        "semantic_config_sha256": _sha(semantic_config_sha256, "semantic_config_sha256"),
        "artifact_set_sha256": _sha(artifact_set_sha256, "artifact_set_sha256"),
        "runtime_identity_sha256": _sha(runtime_identity_sha256, "runtime_identity_sha256"),
        "capture_geometry": validate_label(capture_geometry, "capture_geometry"),
        "capture_tile": None,
        "capture_stride": None,
        "latent_shape": None,
        "latent_dtype": validate_label(latent_dtype, "latent_dtype"),
        "latent_sha256": _sha(latent_sha256, "latent_sha256"),
        "observed_at": observed_at or utc_now(),
    }
    match = re.fullmatch(r"tile(\d+)-stride(\d+)", document["capture_geometry"])
    if match:
        geometry = validate_geometry(int(match.group(1)), int(match.group(2)))
        document["capture_tile"], document["capture_stride"] = geometry.tile, geometry.stride
    if type(latent_shape) not in (tuple, list) or len(latent_shape) != 3:
        raise ConfigError("latent_shape must have exactly 3 dimensions")
    document["latent_shape"] = [_positive_int(value, f"latent_shape[{index}]")
                                for index, value in enumerate(latent_shape)]
    _bounded_json(document, MAX_SNAPSHOT_METADATA_BYTES, "snapshot metadata")
    return document


def raw_tensor_sha256(tensor: Any) -> tuple[str, bytes]:
    """SHA-256 of a CPU float16 tensor's RAW contiguous bytes.

    The tensor must already be on CPU, float16, contiguous and finite; this
    is the digest recorded in snapshot metadata BEFORE the file exists, so
    the binding is never self-referential (hashing the file that contains the
    hash would be). Returns ``(digest, raw_bytes)`` so the caller can also
    verify the payload it is about to save.
    """
    if getattr(tensor, "is_cuda", False):
        raise EngineError("latent digest must be computed on CPU tensor bytes")
    if str(getattr(tensor, "dtype", "")) != "torch.float16":
        raise EngineError("latent tensor must be float16 for the raw digest")
    if not bool(getattr(tensor, "is_contiguous", lambda: False)()):
        raise EngineError("latent tensor must be contiguous for the raw digest")
    raw = bytes(bytearray(tensor.numpy().tobytes()))
    digest = sha256_bytes(raw)
    return digest, raw


def validate_snapshot_metadata(document: Any, *, expected: dict[str, Any]) -> dict[str, Any]:
    """Strict re-validation of a loaded snapshot document against expectations.

    Unknown/missing keys, wrong types and any digest disagreement are typed
    refusals; this is what stands between an arbitrary ``.pt`` file and the
    benchmark.
    """
    if type(document) is not dict or set(document) != set(_SNAPSHOT_KEYS):
        raise ConfigError("snapshot metadata keys disagree with the benchmark schema")
    seed = document["seed"]
    if (type(seed) is not int or seed < 0
            or type(document["steps"]) is not int or document["steps"] <= 0):
        raise ConfigError("snapshot metadata seed/steps are malformed")
    validate_geometry(document["width"], 16)
    validate_geometry(document["height"], 16)
    validate_label(document["quant"], "quant")
    validate_label(document["capture_geometry"], "capture_geometry")
    validate_label(document["latent_dtype"], "latent_dtype")
    for field in ("semantic_config_sha256", "artifact_set_sha256",
                  "runtime_identity_sha256", "latent_sha256"):
        _sha(document[field], field)
    if type(document["latent_shape"]) is not list or len(document["latent_shape"]) != 3:
        raise ConfigError("snapshot latent_shape is malformed")
    if any(type(value) is not int or value <= 0 for value in document["latent_shape"]):
        raise ConfigError("snapshot latent_shape entries must be positive ints")
    canonical = build_snapshot_metadata(**{
        key: document[key] for key in (
            'seed', 'width', 'height', 'steps', 'quant', 'semantic_config_sha256',
            'artifact_set_sha256', 'runtime_identity_sha256', 'capture_geometry',
            'latent_shape', 'latent_dtype', 'latent_sha256', 'observed_at')
    })
    if canonical != document:
        raise ConfigError("snapshot facts disagree with the canonical metadata")
    if type(expected) is not dict:
        raise ConfigError("snapshot expectations must be a dict")
    for field, want in expected.items():
        if field not in document:
            raise ConfigError(f"unknown snapshot expectation field: {field}")
        if document[field] != want:
            raise EngineError(
                f"snapshot {field} disagrees with this run: "
                f"{document[field]!r} != {want!r}")
    return document


# ---------------------------------------------------------------------------
# Benchmark report (its own schema; NOT CalibrationObservation)
# ---------------------------------------------------------------------------

_REPORT_KEYS = ("schema", "version", "created_at", "run", "latents", "profiles",
                "comparisons", "limitations")
_PROFILE_KEYS = ("label", "geometry", "vae_dtype", "observed_at", "duration_s",
                 "memory", "image")
_RUN_KEYS = ("prompt", "seed", "width", "height", "steps", "quant",
             "semantic_config_sha256", "artifact_set_sha256",
             "runtime_identity_sha256", "device_uuid", "nvml_error")


def _bounded_json(document: dict[str, Any], cap: int, field: str) -> bytes:
    try:
        payload = json.dumps(document, sort_keys=True, ensure_ascii=True,
                             allow_nan=False, separators=(",", ":")).encode("ascii")
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{field} is not strict JSON: {exc}") from exc
    if len(payload) > cap:
        raise ConfigError(f"{field} exceeds the {cap} byte cap")
    return payload


def _check_finite_metrics(mapping: Any, field: str) -> None:
    for key, value in mapping.items():
        if value is None and key.startswith('nvml_'):
            continue
        if key.endswith(("_s", "_bytes", "_ms")) or key in (
                "samples", "allocator_retries", "max_channel_delta",
                "mean_channel_delta", "changed_pixel_fraction", "pixels"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{field}.{key} must be numeric")
            if not math.isfinite(float(value)):
                raise ConfigError(f"{field}.{key} must be finite")


def build_benchmark_report(*, run: Any, latents: Any, profiles: Any,
                           comparisons: Any, limitations: Any) -> dict[str, Any]:
    """Assemble and validate the whole benchmark document (strict JSON-safe)."""
    if type(run) is not dict or tuple(sorted(run)) != tuple(sorted(_RUN_KEYS)):
        raise ConfigError("report run block must carry exactly the run identity fields")
    _sha(run["semantic_config_sha256"], "run.semantic_config_sha256")
    _sha(run["artifact_set_sha256"], "run.artifact_set_sha256")
    _sha(run["runtime_identity_sha256"], "run.runtime_identity_sha256")
    if type(run["prompt"]) is not str or not run["prompt"]:
        raise ConfigError("run.prompt must be a non-empty string")
    if run["seed"] is not None:
        _positive_int(run["seed"], "run.seed") if run["seed"] != 0 else None
    validate_label(run["quant"], "run.quant")
    if type(latents) is not dict:
        raise ConfigError("report latents block must be a dict")
    _sha(latents.get("latent_sha256"), "latents.latent_sha256")
    _sha(latents.get("snapshot_sha256"), "latents.snapshot_sha256")
    if type(profiles) is not list or not profiles:
        raise ConfigError("at least one geometry profile is required")
    labels = []
    for profile in profiles:
        if type(profile) is not dict or tuple(sorted(profile)) != tuple(sorted(_PROFILE_KEYS)):
            raise ConfigError("profile block must carry exactly the profile fields")
        labels.append(validate_label(profile["label"], "profile.label"))
        geometry = profile["geometry"]
        if (type(geometry) is not dict or "tile" not in geometry or "stride" not in geometry):
            raise ConfigError("profile.geometry must record tile and stride")
        validate_geometry(geometry["tile"], geometry["stride"])
        validate_label(profile["vae_dtype"], "profile.vae_dtype")
        _finite(profile["duration_s"], "profile.duration_s")
        memory = profile["memory"]
        if isinstance(memory, MemoryObservation):
            profile["memory"] = memory.to_dict()
        elif type(memory) is not dict:
            raise ConfigError("profile.memory must be a MemoryObservation or dict")
        _check_finite_metrics(profile["memory"], "profile.memory")
        if type(profile["image"]) is not dict:
            raise ConfigError("profile.image must be a dict of image facts")
    if len(set(labels)) != len(labels):
        raise ConfigError("profile labels must be unique")
    if type(comparisons) is not list or type(limitations) is not list:
        raise ConfigError("report comparisons/limitations must be lists")
    for comparison in comparisons:
        if type(comparison) is not dict:
            raise ConfigError("comparison entries must be dicts")
        _check_finite_metrics(comparison, "comparison")
    document = {
        "schema": BENCHMARK_SCHEMA,
        "version": BENCHMARK_VERSION,
        "created_at": utc_now(),
        "run": run,
        "latents": latents,
        "profiles": profiles,
        "comparisons": comparisons,
        "limitations": [str(item) for item in limitations],
    }
    _bounded_json(document, MAX_REPORT_BYTES, "benchmark report")
    return document


def write_benchmark_report(report: dict[str, Any], path: str) -> int:
    """Publish the report as strict JSON atomically; refuse over the byte cap."""
    payload = _bounded_json(report, MAX_REPORT_BYTES, "benchmark report")
    exclusive_write_bytes(path, payload + b"\n")
    return len(payload)


def read_benchmark_report(path: str) -> dict[str, Any]:
    """Read one benchmark report back with the same strictness it was written."""
    with open(path, "rb") as handle:
        payload = _byte_field(handle.read(MAX_REPORT_BYTES + 1), "report bytes", MAX_REPORT_BYTES)
    try:
        document = strict_json_loads(payload.decode("ascii"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise EngineError(f"benchmark report is not strict JSON: {exc}") from exc
    if type(document) is not dict or set(document) != set(_REPORT_KEYS):
        raise EngineError("benchmark report keys disagree with the schema")
    if document["schema"] != BENCHMARK_SCHEMA or document["version"] != BENCHMARK_VERSION:
        raise EngineError("benchmark report schema/version is not recognized")
    return document


def _reject_constant(value: str):
    raise EngineError(f"non-finite JSON constant in report: {value}")
