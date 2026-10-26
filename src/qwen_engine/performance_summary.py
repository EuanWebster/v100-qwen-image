"""Aggregate repeated ``qwen_engine.performance.v1`` reports offline (stdlib-only).

Purely diagnostic: never invokes generation, never touches GPU/ML imports.
NULL means unmeasured and is never coerced to zero. Allocated and reserved
CUDA memory stay distinct. CPU-encoder-child lifetime RSS peaks are kept
separate from the generation PID's; peaks are never summed across PIDs or
stages. Nested CPU stage wall times overlap and are never summed into a
total; ``total_job`` is aggregated independently.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import stat
import statistics
import sys
from pathlib import Path

from qwen_engine._jsonsafe import strict_json_loads
from qwen_engine.performance import SCHEMA, CHILD_STAGES

INPUT_CAP = 256 * 1024
MAX_INPUTS = 64
SUMMARY_SCHEMA = "qwen_engine.performance_summary.v1"
METADATA_KEYS = ("width", "height", "seed", "steps", "precision", "quant",
                 "attention_requested", "vae_tiled_decode",
                 "vae_tile", "vae_stride", "vae_precision",
                 "embedding_cache_hit", "runtime_identity_sha256", "artifact_set_sha256",
                 "prompt_tokens", "negative_prompt_tokens")
#: Metadata compared for cross-report compatibility. ``embedding_cache_hit``
#: is excluded so hit/miss/unknown runs of the same configuration can be
#: grouped into separate cache cohorts instead of being refused outright.
COMPARE_METADATA_KEYS = tuple(k for k in METADATA_KEYS if k != "embedding_cache_hit")
#: Stage names actually emitted by the engine (CPU scopes, CUDA leaf scopes
#: and the imported CPU-encoder-child sidecar stages). Arbitrary strings are
#: refused: a report must not smuggle private text (e.g. prompt fragments)
#: through stage names.
KNOWN_STAGE_NAMES = CHILD_STAGES | {
    "total_job", "model_initialisation", "text_encoder_request", "output_saving",
    "adapter_cleanup", "transformer_loading", "transformer_cleanup",
    "initial_vae_loading", "initial_vae_cleanup", "vae_loading",
    "vae_decoding", "output_conversion", "denoising"}
CUDA_STAGES = {"adapter_cleanup", "transformer_loading", "transformer_cleanup",
               "initial_vae_loading", "initial_vae_cleanup", "vae_loading",
               "vae_decoding", "output_conversion", "denoising"}
MAX_STAGES = 32
ROOT_KEYS = {"schema", "status", "metadata", "device", "stages", "measurement_errors",
             "memory_scope", "timing_scope", "attention_observed"}
STAGE_KEYS = {"stage", "pid", "status", "cuda", "host_rss_start_bytes",
              "host_rss_end_bytes", "host_lifetime_peak_rss_bytes", "wall_s"}
CUDA_KEYS = {"allocated_bytes", "reserved_bytes", "peak_allocated_bytes",
             "peak_reserved_bytes"}
MEMORY_METRICS = ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
                  "host_rss_end_bytes")
TOTAL_STAGE = "total_job"


class SummaryError(Exception):
    """Safe, content-free message suitable for CLI display."""


def _is_count(v):
    return type(v) is int and not isinstance(v, bool) and v >= 0


def _is_time(v):
    try:
        return type(v) in (int, float) and math.isfinite(v) and v >= 0
    except (ValueError, OverflowError):
        return False


def _load_report(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > INPUT_CAP:
                raise ValueError("input is not a bounded regular file")
            raw = handle.read(INPUT_CAP + 1)
            after = os.fstat(handle.fileno())
            if (len(raw) != before.st_size or before.st_mtime_ns != after.st_mtime_ns
                    or before.st_ctime_ns != after.st_ctime_ns or len(raw) > INPUT_CAP):
                raise ValueError("input changed")
        doc = strict_json_loads(raw)
    except Exception as exc:
        raise SummaryError(f"{Path(path).name}: unreadable or malformed input ({type(exc).__name__})") from exc
    name = Path(path).name
    if not isinstance(doc, dict) or set(doc) != ROOT_KEYS:
        raise SummaryError(f"{name}: unexpected report top-level fields")
    if doc["schema"] != SCHEMA:
        raise SummaryError(f"{name}: unsupported schema")
    if doc["status"] != "succeeded":
        raise SummaryError(f"{name}: report status is not succeeded")
    if not isinstance(doc["measurement_errors"], list) or doc["measurement_errors"]:
        raise SummaryError(f"{name}: measurement errors present")
    if doc["attention_observed"] is not None:
        raise SummaryError(f"{name}: unexpected attention_observed value")
    md = doc["metadata"]
    if not isinstance(md, dict) or set(md) - set(METADATA_KEYS):
        raise SummaryError(f"{name}: unknown metadata fields")
    for key in METADATA_KEYS:
        v = md.get(key)
        if key in ("runtime_identity_sha256", "artifact_set_sha256"):
            if key in md and (not isinstance(v, str) or len(v) != 64
                              or any(c not in "0123456789abcdef" for c in v)):
                raise SummaryError(f"{name}: invalid identity")
        elif key in ("prompt_tokens", "negative_prompt_tokens"):
            if v is not None and not _is_count(v):
                raise SummaryError(f"{name}: invalid token count")
        elif key in ("width", "height", "steps"):
            if not _is_count(v):
                raise SummaryError(f"{name}: invalid metadata {key}")
        elif key == "seed":
            # seed is None until sampling has actually finished
            if v is not None and not _is_count(v):
                raise SummaryError(f"{name}: invalid metadata {key}")
        elif key in ("vae_tiled_decode", "embedding_cache_hit"):
            # both are booleans; embedding_cache_hit may be absent (unknown)
            # in minimal reports and lands in the "unknown" cache cohort
            if key not in md:
                if key == "embedding_cache_hit":
                    continue
                raise SummaryError(f"{name}: invalid metadata {key}")
            if not isinstance(v, bool):
                raise SummaryError(f"{name}: invalid metadata {key}")
        elif key in ("vae_tile", "vae_stride"):
            # the adapter sets these to None when tiled decode is off
            if v is not None and not _is_count(v):
                raise SummaryError(f"{name}: invalid metadata {key}")
        elif key not in md or not isinstance(v, str) or not v or len(v) > 128:
            raise SummaryError(f"{name}: invalid metadata {key}")
    stages = doc["stages"]
    if not isinstance(stages, list) or not stages or len(stages) > MAX_STAGES:
        raise SummaryError(f"{name}: bad stage list")
    allowed_keys = STAGE_KEYS | {"cuda_start"}  # cuda_start only on CUDA stages
    names, pids = [], set()
    for st in stages:
        if (not isinstance(st, dict) or not STAGE_KEYS <= set(st)
                or not set(st) <= allowed_keys):
            raise SummaryError(f"{name}: unknown or missing stage fields")
        if not isinstance(st["stage"], str) or st["stage"] not in KNOWN_STAGE_NAMES:
            raise SummaryError(f"{name}: unrecognised stage name")
        if st["stage"] in names:
            raise SummaryError(f"{name}: duplicate stage name")
        names.append(st["stage"])
        if st["stage"] not in CUDA_STAGES and (st["cuda"] is not None or "cuda_start" in st):
            raise SummaryError(f"{name}: unexpected CUDA stage fields")
        if not _is_count(st["pid"]) or st["pid"] == 0:
            raise SummaryError(f"{name}: invalid stage pid")
        pids.add(st["pid"])
        if st["status"] != "succeeded":
            raise SummaryError(f"{name}: stage {st['stage']} did not succeed")
        if not _is_time(st["wall_s"]):
            raise SummaryError(f"{name}: invalid wall_s for stage {st['stage']}")
        for key in ("host_rss_start_bytes", "host_rss_end_bytes", "host_lifetime_peak_rss_bytes"):
            if st[key] is not None and not _is_count(st[key]):
                raise SummaryError(f"{name}: invalid {key}")
        for ckey in ("cuda", "cuda_start"):
            if ckey not in st:
                continue  # cuda_start is optional; cuda must always be present
            block = st[ckey]
            if block is None and ckey == "cuda":
                continue
            if not isinstance(block, dict) or set(block) != CUDA_KEYS:
                raise SummaryError(f"{name}: unknown or missing {ckey} fields")
            for key in CUDA_KEYS:
                if not _is_count(block[key]):
                    raise SummaryError(f"{name}: invalid {ckey} {key}")
            if (block["allocated_bytes"] > block["reserved_bytes"]
                    or block["allocated_bytes"] > block["peak_allocated_bytes"]
                    or block["reserved_bytes"] > block["peak_reserved_bytes"]
                    or block["peak_allocated_bytes"] > block["peak_reserved_bytes"]):
                raise SummaryError(f"{name}: inconsistent CUDA counters")
    if TOTAL_STAGE not in names:
        raise SummaryError(f"{name}: missing total_job stage")
    child = [n for n in names if n in CHILD_STAGES]
    if len(child) > 3 or len(child) != len(set(child)):
        raise SummaryError(f"{name}: unexpected child stage set")
    parent_pid = next(st["pid"] for st in stages if st["stage"] == TOTAL_STAGE)
    if any((st["pid"] == parent_pid) != (st["stage"] not in CHILD_STAGES) for st in stages):
        raise SummaryError(f"{name}: unexpected stage process ownership")
    _device(doc, name)
    return doc


def _device(doc, name="report"):
    """Validate the device block. ``None`` is only legitimate when the report
    carries no CUDA stage at all (CPU-only/child-only run); a GPU report must
    carry exact device facts so mixed GPUs can never be merged unnoticed."""
    device = doc["device"]
    if device is None:
        if any(st.get("cuda") is not None or st.get("cuda_start") is not None
               for st in doc["stages"]):
            raise SummaryError(f"{name}: CUDA stage present but device is null")
        return None
    if (not isinstance(device, dict)
            or set(device) != {"name", "compute_capability", "total_bytes"}):
        raise SummaryError(f"{name}: unexpected device fields")
    cc = device["compute_capability"]
    if (not isinstance(device["name"], str) or not device["name"] or len(device["name"]) > 256
            or not _is_count(device["total_bytes"]) or device["total_bytes"] == 0
            or not isinstance(cc, list) or len(cc) != 2 or not all(_is_count(c) for c in cc)):
        raise SummaryError(f"{name}: invalid device fields")
    return (device["name"], tuple(cc), device["total_bytes"])


def _metrics(samples):
    """samples: list of numbers or None. NULL stays excluded; empty -> nulls."""
    vals = [v for v in samples if v is not None]
    if not vals:
        return {"count": 0, "median": None, "min": None, "max": None, "pstdev": None}
    return {"count": len(vals), "median": statistics.median(vals),
            "min": min(vals), "max": max(vals),
            "pstdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0}


def _cache_group(doc):
    """Cache cohorts only: hit / miss / unknown (key absent in minimal
    reports). Device presence is NOT a cache statement and is never used as a
    cohort heuristic; device matching is a separate hard requirement."""
    md = doc["metadata"]
    if "embedding_cache_hit" not in md:
        return "embedding-cache-unknown"
    return "embedding-cache-hit" if md["embedding_cache_hit"] is True else "embedding-cache-miss"


def summarise_reports(paths, cohort=None):
    paths = [Path(p) for p in paths]
    if not paths:
        raise SummaryError("at least one report input is required")
    if len(paths) > MAX_INPUTS:
        raise SummaryError(f"too many report inputs (max {MAX_INPUTS})")
    docs = [_load_report(p) for p in paths]
    base = docs[0]["metadata"]
    base_cmp = {k: base[k] for k in COMPARE_METADATA_KEYS if k in base}
    base_dev = _device(docs[0], paths[0].name)
    for p, doc in zip(paths[1:], docs[1:]):
        md = doc["metadata"]
        if {k: md[k] for k in COMPARE_METADATA_KEYS if k in md} != base_cmp:
            raise SummaryError(f"{p.name}: incompatible run metadata")
        if _device(doc, p.name) != base_dev:
            raise SummaryError(f"{p.name}: device differs between reports")
    groups = {}
    for doc in docs:
        # --cohort labels the whole report set but never merges cache
        # cohorts: hit, miss and unknown always stay in separate groups.
        key = _cache_group(doc) if cohort is None else f"{cohort}/{_cache_group(doc)}"
        groups.setdefault(key, []).append(doc)
    summary = {"schema": SUMMARY_SCHEMA,
               "comparison_metadata": dict(sorted(base.items())),
               "cohort_label": cohort,
               "sample_count": len(docs),
               "groups": {}}
    for name in sorted(groups):
        gd = groups[name]
        stage_names = sorted({st["stage"] for doc in gd for st in doc["stages"]})
        stages_out = {}
        for stage in stage_names:
            walls, mems, pids = [], {m: [] for m in MEMORY_METRICS}, set()
            for doc in gd:
                rec = [st for st in doc["stages"] if st["stage"] == stage]
                if len(rec) != 1:
                    continue  # stage absent in this report; never fabricate
                st = rec[0]
                pids.add(st["pid"])
                walls.append(st["wall_s"])
                for m in MEMORY_METRICS:
                    if m == "cuda_peak_allocated_bytes":
                        mems[m].append(None if st["cuda"] is None else st["cuda"]["peak_allocated_bytes"])
                    elif m == "cuda_peak_reserved_bytes":
                        mems[m].append(None if st["cuda"] is None else st["cuda"]["peak_reserved_bytes"])
                    else:
                        mems[m].append(st["host_rss_end_bytes"])
            entry = {"present_in_reports": len(walls), "distinct_pids": len(pids),
                     "wall_s": _metrics(walls)}
            for m in MEMORY_METRICS:
                entry[m] = _metrics(mems[m])
            stages_out[stage] = entry
        peaks = {}
        for doc in gd:
            by_pid = {}
            for st in doc["stages"]:
                if st["host_lifetime_peak_rss_bytes"] is not None:
                    cur = by_pid.get(st["pid"])
                    if cur is None or st["host_lifetime_peak_rss_bytes"] > cur:
                        by_pid[st["pid"]] = st["host_lifetime_peak_rss_bytes"]
            child_pids = {st["pid"] for st in doc["stages"] if st["stage"] in CHILD_STAGES}
            for pid, peak in by_pid.items():
                kind = "cpu_encoder_child" if pid in child_pids else "generation_process"
                peaks.setdefault(kind, []).append(peak)
        summary["groups"][name] = {
            "report_count": len(gd),
            "note": ("lifetime host RSS peaks are per-PID maxima; never summed "
                     "across PIDs or stages; wall times of nested stages overlap "
                     "and total_job is aggregated independently, not as a sum"),
            "host_lifetime_peak_rss_bytes_by_pid_kind":
                {k: _metrics(v) for k, v in sorted(peaks.items())},
            "stages": stages_out}
    summary["interpretation"] = {
        "cache_group_meaning": "embedding-cache hit/miss cohorts only; not disk pagecache or persistent-worker warmth",
        "cuda_memory": "allocated and reserved are distinct allocator metrics; peaks are per CUDA leaf stage",
        "attention_backend": "never inferred; reports carry attention_observed=null"}
    return summary


def write_summary(summary, output):
    out = Path(output)
    try:
        text = json.dumps(summary, sort_keys=True, allow_nan=False) + "\n"
        with out.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise SummaryError("output file already exists; refusing to overwrite") from exc
    except OSError as exc:
        raise SummaryError(f"cannot write output ({type(exc).__name__})") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m qwen_engine.performance_summary",
        description="Aggregate qwen_engine performance reports (offline, diagnostic only).")
    parser.add_argument("reports", nargs="+", type=Path, metavar="REPORT.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cohort", default=None,
                        help="explicit cohort label; default groups by cache state")
    args = parser.parse_args(argv)
    try:
        summary = summarise_reports(args.reports, args.cohort)
        write_summary(summary, args.output)
    except SummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
