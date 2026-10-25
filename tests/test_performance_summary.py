"""Offline tests for performance_summary (no ML, no GPU)."""
import json
import math
from pathlib import Path

import pytest

from qwen_engine.performance_summary import (
    SummaryError, summarise_reports, write_summary)

# Real adapter metadata: quant is fp16/q8 (never "none"); the adapter adds
# vae_tile/vae_stride (None when tiling is off) and vae_precision.
META = {"width": 1024, "height": 1024, "seed": 7, "steps": 30, "precision": "fp16",
        "quant": "fp16", "attention_requested": "auto", "vae_tiled_decode": False,
        "vae_tile": None, "vae_stride": None, "vae_precision": "fp16"}

DEVICE = {"name": "Tesla V100", "compute_capability": [7, 0],
          "total_bytes": 32212254720}


def stage(name, pid, wall, cuda=None, rss_end=1000, peak=2000, status="succeeded",
          cuda_start=None):
    st = {"stage": name, "pid": pid, "status": status, "cuda": cuda,
          "host_rss_start_bytes": 500, "host_rss_end_bytes": rss_end,
          "host_lifetime_peak_rss_bytes": peak, "wall_s": wall}
    if cuda_start is not None:
        st["cuda_start"] = cuda_start
    return st


def cuda(alloc, res, pa, pr):
    return {"allocated_bytes": alloc, "reserved_bytes": res,
            "peak_allocated_bytes": pa, "peak_reserved_bytes": pr}


def report(stages, status="succeeded", errors=(), device=DEVICE, metadata=META):
    return {"schema": "qwen_engine.performance.v1", "status": status,
            "metadata": metadata, "device": device, "stages": stages,
            "measurement_errors": list(errors),
            "memory_scope": "CUDA process allocator; host RSS per PID; host peak is lifetime, not per stage",
            "timing_scope": "nested CPU scopes overlap; total_job includes cleanup; not additive",
            "attention_observed": None}


def gen_stages(wall_total=10.0, wall_denoise=6.0, pa=1000, pr=2000, rss=1000, peak=2000):
    """Matches a real GPU report: total_job is a CPU scope (cuda=None); CUDA
    leaf stages carry optional cuda_start plus cuda end counters; child
    encoder stages run in a different PID with cuda=None."""
    return [stage("total_job", 100, wall_total, None, rss, peak),
            stage("denoising", 100, wall_denoise, cuda(800, 1400, pa, pr), rss, peak,
                  cuda_start=cuda(100, 200, 100, 200)),
            stage("text_encoder_loading", 200, 1.0, None, 300, 700),
            stage("prompt_encoding", 200, 0.5, None, 350, 700),
            stage("text_encoder_cleanup", 200, 0.1, None, 360, 700)]


def write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc, sort_keys=True))
    return p


def test_median_and_pstdev_actual_numbers(tmp_path):
    walls = [1.0, 2.0, 3.0, 4.0]
    docs = [gen_stages(wall_total=w) for w in walls]
    paths = [write(tmp_path, f"r{i}.json", report(s)) for i, s in enumerate(docs)]
    out = summarise_reports(paths)
    tj = out["groups"]["embedding-cache-unknown"]["stages"]["total_job"]["wall_s"]
    assert tj == {"count": 4, "median": 2.5, "min": 1.0, "max": 4.0,
                  "pstdev": pytest.approx(math.sqrt(1.25))}


def test_null_unmeasured_never_zero(tmp_path):
    s = [stage("total_job", 1, 1.0), stage("denoising", 1, 0.5)]
    out = summarise_reports([write(tmp_path, "r.json", report(s))])
    st = out["groups"]["embedding-cache-unknown"]["stages"]
    assert st["denoising"]["cuda_peak_allocated_bytes"]["count"] == 0
    assert st["denoising"]["cuda_peak_allocated_bytes"]["median"] is None
    assert st["denoising"]["wall_s"]["count"] == 1


def test_allocated_vs_reserved_distinct(tmp_path):
    s = [stage("total_job", 1, 1.0), stage("denoising", 1, .5, cuda(10, 20, 30, 40))]
    st = summarise_reports([write(tmp_path, "r.json", report(s))])[
        "groups"]["embedding-cache-unknown"]["stages"]["denoising"]
    assert st["cuda_peak_allocated_bytes"]["median"] == 30
    assert st["cuda_peak_reserved_bytes"]["median"] == 40


def test_hit_miss_unknown_groups_separate(tmp_path):
    hit = dict(META, embedding_cache_hit=True)
    miss = dict(META, embedding_cache_hit=False)
    paths = [write(tmp_path, "h.json", report(gen_stages(), metadata=hit)),
             write(tmp_path, "m.json", report(gen_stages(), metadata=miss)),
             write(tmp_path, "u.json", report(gen_stages()))]  # key absent = unknown
    out = summarise_reports(paths)
    assert set(out["groups"]) == {"embedding-cache-hit", "embedding-cache-miss",
                                  "embedding-cache-unknown"}
    assert out["groups"]["embedding-cache-hit"]["report_count"] == 1
    assert out["groups"]["embedding-cache-unknown"]["report_count"] == 1


def test_cohort_labels_but_never_merges_cache_groups(tmp_path):
    hit = dict(META, embedding_cache_hit=True)
    miss = dict(META, embedding_cache_hit=False)
    paths = [write(tmp_path, "h.json", report(gen_stages(), metadata=hit)),
             write(tmp_path, "m.json", report(gen_stages(), metadata=miss))]
    out = summarise_reports(paths, cohort="manual")
    assert out["cohort_label"] == "manual"
    assert set(out["groups"]) == {"manual/embedding-cache-hit",
                                  "manual/embedding-cache-miss"}
    assert out["groups"]["manual/embedding-cache-hit"]["report_count"] == 1


def test_incompatible_metadata_refused(tmp_path):
    paths = [write(tmp_path, "a.json", report(gen_stages())),
             write(tmp_path, "b.json", report(gen_stages(), metadata=dict(META, seed=8)))]
    with pytest.raises(SummaryError, match="metadata"):
        summarise_reports(paths)


def test_device_must_match_even_with_cache_keys(tmp_path):
    hit = dict(META, embedding_cache_hit=True)
    other = dict(DEVICE, name="Tesla V100-SXM2")
    paths = [write(tmp_path, "h.json", report(gen_stages(), metadata=hit)),
             write(tmp_path, "g.json", report(gen_stages(), metadata=hit, device=other))]
    with pytest.raises(SummaryError, match="device"):
        summarise_reports(paths)


def test_duplicate_stage_in_one_report_refused(tmp_path):
    s = gen_stages()
    s.append(stage("denoising", 100, 1.0, cuda(1, 1, 1, 1)))
    with pytest.raises(SummaryError, match="duplicate"):
        summarise_reports([write(tmp_path, "r.json", report(s))])


def test_failed_or_incomplete_not_success(tmp_path):
    p1 = write(tmp_path, "f.json", report(gen_stages(), status="failed"))
    with pytest.raises(SummaryError, match="status"):
        summarise_reports([p1])
    p2 = write(tmp_path, "e.json", report(gen_stages(), errors=["RuntimeError"]))
    with pytest.raises(SummaryError, match="measurement"):
        summarise_reports([p2])
    s = gen_stages()
    s[1]["status"] = "failed"
    with pytest.raises(SummaryError, match="stage"):
        summarise_reports([write(tmp_path, "s.json", report(s))])


def test_invalid_numbers_bool_negative_nan(tmp_path):
    s = gen_stages()
    s[0]["wall_s"] = True
    with pytest.raises(SummaryError, match="wall_s"):
        summarise_reports([write(tmp_path, "r.json", report(s))])
    s = gen_stages()
    s[0]["wall_s"] = -1.0
    with pytest.raises(SummaryError, match="wall_s"):
        summarise_reports([write(tmp_path, "n.json", report(s))])
    s = gen_stages()
    s[1]["cuda"]["peak_allocated_bytes"] = -5
    with pytest.raises(SummaryError, match="cuda"):
        summarise_reports([write(tmp_path, "c.json", report(s))])
    s = gen_stages()
    s[1]["cuda_start"]["reserved_bytes"] = -1
    with pytest.raises(SummaryError, match="cuda_start"):
        summarise_reports([write(tmp_path, "cs.json", report(s))])
    s = gen_stages()
    s[0]["host_rss_end_bytes"] = True
    with pytest.raises(SummaryError, match="rss"):
        summarise_reports([write(tmp_path, "b.json", report(s))])
    bad = tmp_path / "nan.json"
    bad.write_text(json.dumps(report(gen_stages())).replace('"wall_s": 10.0', '"wall_s": NaN'))
    with pytest.raises(SummaryError, match="malformed"):
        summarise_reports([bad])


def test_unknown_nested_fields_reject_private_data(tmp_path):
    s = gen_stages()
    s[0]["prompt"] = "secret"
    with pytest.raises(SummaryError, match="stage fields"):
        summarise_reports([write(tmp_path, "r.json", report(s))])
    s = gen_stages()
    s[1]["cuda"]["extra"] = 1
    with pytest.raises(SummaryError, match="cuda fields"):
        summarise_reports([write(tmp_path, "x.json", report(s))])
    doc = report(gen_stages())
    doc["extra"] = 1
    with pytest.raises(SummaryError, match="top-level"):
        summarise_reports([write(tmp_path, "t.json", doc)])
    dup = tmp_path / "dup.json"
    dup.write_text('{"schema": "qwen_engine.performance.v1", '
                   '"schema": "qwen_engine.performance.v1"}')
    with pytest.raises(SummaryError, match="malformed"):
        summarise_reports([dup])


def test_stage_name_allowlist_and_cuda_start_placement(tmp_path):
    s = gen_stages()
    s[1]["stage"] = "denoise"  # actual reports say "denoising"
    with pytest.raises(SummaryError, match="stage name"):
        summarise_reports([write(tmp_path, "r.json", report(s))])
    s = gen_stages()
    s[1]["stage"] = "a prompt fragment about a cat"
    with pytest.raises(SummaryError, match="stage name"):
        summarise_reports([write(tmp_path, "p.json", report(s))])
    s = gen_stages()
    s[0]["cuda_start"] = cuda(1, 1, 1, 1)  # CPU total_job never has cuda_start
    with pytest.raises(SummaryError, match="stage fields"):
        summarise_reports([write(tmp_path, "c.json", report(s))])


def test_device_validated_and_required_with_cuda_stages(tmp_path):
    s = gen_stages()
    with pytest.raises(SummaryError, match="device"):
        summarise_reports([write(tmp_path, "n.json", report(s, device=None))])
    s = [stage("total_job", 1, 1.0), stage("text_encoder_loading", 2, 0.5)]
    out = summarise_reports([write(tmp_path, "cpu.json", report(s, device=None))])
    assert out["groups"]["embedding-cache-unknown"]["report_count"] == 1
    s = gen_stages()
    with pytest.raises(SummaryError, match="device"):
        summarise_reports([write(tmp_path, "d.json",
                                 report(s, device={"name": "V100", "total_bytes": 1}))])


def test_lifetime_peaks_separate_per_pid_kind(tmp_path):
    out = summarise_reports([write(tmp_path, "r.json", report(gen_stages()))])
    peaks = out["groups"]["embedding-cache-unknown"][
        "host_lifetime_peak_rss_bytes_by_pid_kind"]
    assert peaks["generation_process"]["median"] == 2000
    assert peaks["cpu_encoder_child"]["median"] == 700
    assert peaks["generation_process"]["count"] == 1
    assert peaks["cpu_encoder_child"]["count"] == 1  # max per PID, not sum of 3


def test_total_job_not_sum_of_stages(tmp_path):
    out = summarise_reports([write(tmp_path, "r.json", report(gen_stages()))])
    st = out["groups"]["embedding-cache-unknown"]["stages"]
    assert st["total_job"]["wall_s"]["median"] == 10.0
    assert st["total_job"]["wall_s"]["median"] != sum(
        st[n]["wall_s"]["median"] for n in st if n != "total_job")


def test_symlink_input_refused_and_write_no_clobber(tmp_path):
    real = write(tmp_path, "real.json", report(gen_stages()))
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(SummaryError, match="unreadable"):
        summarise_reports([link])
    out = summarise_reports([real])
    target = tmp_path / "sum.json"
    write_summary(out, target)
    with pytest.raises(SummaryError, match="overwrite"):
        write_summary(out, target)
    assert target.read_text()
