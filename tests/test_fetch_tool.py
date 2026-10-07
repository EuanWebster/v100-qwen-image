"""Offline fake-stream tests for ``tools/fetch_models.py`` (M2.1 fetcher task).

Everything here is CPU-only, network-free and stdlib/heavy-import-free: the
fetcher's transport is *injected* (a fake byte stream), so no socket is ever
constructed, no real weight bytes exist anywhere, and every write lands in
pytest ``tmp_path`` (R7). Tests cover: happy path, already-good and
mismatched finals, truncated / oversized / wrong-hash responses,
interrupted-then-resumed Range, ignored Range, forged/wrong Content-Range,
partial too big, symlinked partial / symlinked write path / symlinked final,
path escape, disk-space refusal, no-clobber publish, URL/host pinning and
sanitized (query-stripped) error messages, and read-only/offline plan+help.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS_PATH = ROOT / "tools" / "fetch_models.py"

_spec = importlib.util.spec_from_file_location("fetch_models", TOOLS_PATH)
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)

REPO = "Qwen/Qwen-Image-2.1"
REV = "d26bb61231c349cf6b7896fa83353113880e1ba3"
HEAVY_MODULES = ("torch", "diffusers", "transformers")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def art(blob: bytes, filename: str = "a/x.bin", role: str = "dit") -> dict:
    return {"role": role, "repo": REPO, "revision": REV, "filename": filename,
            "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}


def write_manifest(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "artifacts": entries}))
    return path


class FakeStream:
    def __init__(self, status: int, headers: dict, data: bytes,
                 fail_at: int | None = None) -> None:
        self.status = status
        self.headers = headers
        self._data = data
        self._pos = 0
        self._fail_at = fail_at

    def read(self, n: int) -> bytes:
        if self._fail_at is not None and self._pos >= self._fail_at:
            raise ConnectionError("fake stream reset by peer")
        end = self._pos + n
        if self._fail_at is not None:
            end = min(end, self._fail_at)
        chunk = self._data[self._pos:end]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        pass


class FakeTransport:
    """url -> spec; spec is a FakeStream, an Exception instance (raised), or
    a callable(start) -> spec. Records (url, start) for every call."""

    def __init__(self, plan: dict | None = None) -> None:
        self.plan = plan or {}
        self.calls: list[tuple[str, int]] = []

    def get(self, url: str, start: int) -> FakeStream:
        self.calls.append((url, start))
        spec = self.plan[url]
        if callable(spec):
            spec = spec(start)
        if isinstance(spec, Exception):
            raise spec
        return spec

    def urls(self) -> list[str]:
        return [u for u, _ in self.calls]


def ok_stream(blob: bytes):
    """A spec factory serving the full blob at offset 0 and a proper 206
    Content-Range resume at any offset."""
    def spec(start: int) -> FakeStream:
        if start == 0:
            return FakeStream(200, {}, blob)
        return FakeStream(206, {"Content-Range":
                               f"bytes {start}-{len(blob) - 1}/{len(blob)}"},
                          blob[start:])
    return spec


class ExplodingTransport(FakeTransport):
    def get(self, url: str, start: int):
        raise AssertionError(f"network transport called unexpectedly: {url}")


def run(tmp_path: Path, manifest: Path, root: Path, *, transport=None,
        fetch_mode: bool = True, disk_free=None, extra: list[str] | None = None):
    """Invoke the tool in-process with an injected (or exploding) transport."""
    cap = _Capture()
    transport = transport if transport is not None else ExplodingTransport()
    args = ["--manifest", str(manifest), "--models-dir", str(root)]
    if fetch_mode:
        args.append("--fetch")
    args += extra or []
    import io
    import contextlib
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fetch.main(args, transport=transport, disk_free_fn=disk_free)
    cap.out, cap.err = out.getvalue(), err.getvalue()
    cap.rc = rc
    return cap


class _Capture:
    rc = 0
    out = ""
    err = ""


def _no_network(monkeypatch):
    import urllib.request

    def explode(*a, **k):
        raise AssertionError("fetch tool attempted network access")
    monkeypatch.setattr(socket, "socket", explode)
    monkeypatch.setattr(socket, "create_connection", explode)
    monkeypatch.setattr(urllib.request, "urlopen", explode)


def _no_partial_left(root: Path) -> None:
    assert not list(root.rglob("*.partial")), "partial file left behind"


# ---------------------------------------------------------------------------
# CLI surface: help/usage/plan are offline and read-only
# ---------------------------------------------------------------------------


def test_help_and_usage_exit_codes(tmp_path, capsys):
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert fetch.main(["--help"]) == 0
    assert "fetch_models" in buf.getvalue()
    assert "--fetch" in buf.getvalue()
    # Missing required explicit paths: usage error (2), nothing else.
    assert fetch.main([]) == fetch.EXIT_USAGE


def test_plan_is_offline_read_only_and_socket_free(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    blob_a, blob_b = b"alpha" * 7, b"beta" * 11
    manifest = write_manifest(tmp_path, [
        art(blob_a, "a/x.bin"), art(blob_b, "b/y.bin", role="vae")])
    root = tmp_path / "assets"  # deliberately never created
    res = run(tmp_path, manifest, root, fetch_mode=False)
    assert res.rc == 0, res.err
    doc = json.loads(res.out)
    assert doc["mode"] == "plan" and doc["network"].startswith("offline")
    assert doc["artifact_count"] == 2 and doc["pending_or_bad"] == 2
    assert [r["state"] for r in doc["artifacts"]] == ["missing", "missing"]
    assert doc["estimated_bytes_still_needed"] == len(blob_a) + len(blob_b)
    # Deterministic: a second identical plan prints byte-identical JSON.
    res2 = run(tmp_path, manifest, root, fetch_mode=False)
    assert res2.out == res.out
    # Read-only: no asset root, no files, no partials, no transport calls.
    assert not root.exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_plan_urls_pinned_official_https_no_query(tmp_path):
    manifest = write_manifest(tmp_path, [art(b"z" * 3, "a/x.bin")])
    res = run(tmp_path, manifest, tmp_path / "assets", fetch_mode=False)
    assert res.rc == 0, res.err
    doc = json.loads(res.out)
    for row in doc["artifacts"]:
        url = row["url"]
        assert url == (f"https://{fetch.HF_HOST}/{REPO}/resolve/{REV}"
                       f"/{row['filename']}")
        assert "?" not in url and url.startswith("https://huggingface.co/")


def test_manifest_with_branch_revision_refused_exit_2(tmp_path):
    bad = art(b"x" * 4, "a/x.bin")
    bad["revision"] = "main"
    manifest = write_manifest(tmp_path, [bad])
    res = run(tmp_path, manifest, tmp_path / "assets")
    assert res.rc == fetch.EXIT_USAGE
    assert "fetch error:" in res.err and "40-hex" in res.err


def test_multi_segment_repo_refused_before_any_url(tmp_path):
    bad = art(b"x" * 4, "a/x.bin")
    bad["repo"] = "a/b/c"
    manifest = write_manifest(tmp_path, [bad])
    res = run(tmp_path, manifest, tmp_path / "assets", fetch_mode=False)
    assert res.rc == fetch.EXIT_USAGE
    assert "namespace/name" in res.err


def test_child_plan_run_offline_under_guard(tmp_path):
    """Fresh-interpreter plan/help: exit 0. Under ``bash run_cpu_tests.sh``
    the R7 guard rides into this child via PYTHONPATH, so exit 0 there also
    proves no socket construction and no heavy import happened."""
    manifest = ROOT / "models" / "manifest.json"
    proc = subprocess.run(
        [sys.executable, str(TOOLS_PATH), "--manifest", str(manifest),
         "--models-dir", str(tmp_path / "assets")],
        cwd=str(ROOT), env=os.environ.copy(), capture_output=True,
        text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["artifact_count"] == 24
    assert all(r["repo"] == REPO and r["revision"] == REV
               for r in doc["artifacts"])
    assert not (tmp_path / "assets").exists()  # plan wrote nothing
    proc2 = subprocess.run([sys.executable, str(TOOLS_PATH), "--help"],
                           cwd=str(ROOT), env=os.environ.copy(),
                           capture_output=True, text=True, timeout=120)
    assert proc2.returncode == 0 and "--fetch" in proc2.stdout


# ---------------------------------------------------------------------------
# --fetch: happy paths, skips, existing finals
# ---------------------------------------------------------------------------


def test_fetch_happy_path_two_artifacts(tmp_path):
    blob_a = os.urandom(1_048_600)  # > 1 MiB: crosses a chunk boundary
    blob_b = os.urandom(300)
    e_a, e_b = art(blob_a, "a/x.bin"), art(blob_b, "b/y.bin", role="vae")
    manifest = write_manifest(tmp_path, [e_a, e_b])
    root = tmp_path / "assets"
    t = FakeTransport({_url(e_a): ok_stream(blob_a),
                       _url(e_b): ok_stream(blob_b)})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == 0, res.err
    assert (root / "a" / "x.bin").read_bytes() == blob_a
    assert (root / "b" / "y.bin").read_bytes() == blob_b
    _no_partial_left(root)
    assert [s for _, s in t.calls] == [0, 0]
    assert all(u.startswith(f"https://{fetch.HF_HOST}/{REPO}/resolve/{REV}/")
               for u in t.urls())
    assert "downloaded-and-verified" in res.out
    # no heavy modules entered this process through the tool
    assert not set(HEAVY_MODULES) & set(sys.modules)


def _url(entry: dict) -> str:
    from qwen_engine.manifest import ModelArtifact
    return fetch.artifact_url(ModelArtifact(**entry))


def test_existing_good_final_skipped(tmp_path):
    blob = b"good contents" * 5
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    final = root / "a" / "x.bin"
    final.write_bytes(blob)
    before = final.stat().st_mtime_ns
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == 0, res.err
    assert "already present and verified" in res.out
    assert "nothing to fetch" in res.out
    assert final.stat().st_mtime_ns == before and final.read_bytes() == blob
    assert t.calls == []


@pytest.mark.parametrize("tamper", ["wrong-hash", "wrong-size"])
def test_existing_bad_final_never_touched_never_fetched(tmp_path, tamper):
    blob = b"right contents" * 5
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    final = root / "a" / "x.bin"
    final.write_bytes(b"X" * len(blob) if tamper == "wrong-hash" else b"short")
    data_before = final.read_bytes()
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_REFUSED
    assert "never overwrites" in res.err
    assert final.read_bytes() == data_before
    assert t.calls == []
    _no_partial_left(root)


def test_fetch_skips_good_and_downloads_missing(tmp_path):
    good, missing = b"present" * 9, b"absent" * 13
    e_good, e_missing = art(good, "a/good.bin"), art(missing, "a/new.bin")
    manifest = write_manifest(tmp_path, [e_good, e_missing])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    (root / "a" / "good.bin").write_bytes(good)
    t = FakeTransport({_url(e_missing): ok_stream(missing)})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == 0, res.err
    assert (root / "a" / "new.bin").read_bytes() == missing
    assert t.calls == [(_url(e_missing), 0)]


# ---------------------------------------------------------------------------
# --fetch: response failures never publish
# ---------------------------------------------------------------------------


def test_truncated_response_keeps_partial_for_resume(tmp_path):
    blob = b"T" * 300
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    t = FakeTransport({_url(entry): FakeStream(200, {}, blob[:40])})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_NETWORK
    assert "response ended" in res.err
    assert not (root / "a" / "x.bin").exists()
    partial = root / "a" / "x.bin.partial"
    assert partial.read_bytes() == blob[:40]
    # And the kept partial is genuinely resumable:
    t2 = FakeTransport({_url(entry): ok_stream(blob)})
    res2 = run(tmp_path, manifest, root, transport=t2)
    assert res2.rc == 0, res2.err
    assert (root / "a" / "x.bin").read_bytes() == blob
    assert t2.calls == [(_url(entry), 40)]
    _no_partial_left(root)


def test_oversized_response_rejected_partial_removed(tmp_path):
    blob = b"O" * 200
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    t = FakeTransport({_url(entry): FakeStream(200, {}, blob + b"EXTRA" * 3)})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_INTEGRITY
    assert "more than the manifest" in res.err
    assert not (root / "a" / "x.bin").exists()
    _no_partial_left(root)


def test_bad_hash_never_publishes_partial_removed(tmp_path):
    blob = b"correct" * 20
    lie = b"corrupt" * 20
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    t = FakeTransport({_url(entry): FakeStream(200, {}, lie)})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_INTEGRITY
    assert "SHA256 mismatch" in res.err and entry["sha256"] in res.err
    assert not (root / "a" / "x.bin").exists()
    _no_partial_left(root)
    # The refusal left the tree clean, so a corrected server succeeds:
    t2 = FakeTransport({_url(entry): ok_stream(blob)})
    assert run(tmp_path, manifest, root, transport=t2).rc == 0
    assert (root / "a" / "x.bin").read_bytes() == blob


def test_content_encoding_gzip_refused(tmp_path):
    blob = b"G" * 64
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    t = FakeTransport({_url(entry):
                       FakeStream(200, {"Content-Encoding": "gzip"}, blob)})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_NETWORK
    assert "Content-Encoding" in res.err
    assert not (root / "a" / "x.bin").exists()


# ---------------------------------------------------------------------------
# --fetch: interruption / resume / Range honesty
# ---------------------------------------------------------------------------


def test_interrupted_then_resumed_with_proper_range(tmp_path):
    blob = os.urandom(2500)
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    t1 = FakeTransport({_url(entry): FakeStream(200, {}, blob, fail_at=1000)})
    res = run(tmp_path, manifest, root, transport=t1)
    assert res.rc == fetch.EXIT_NETWORK
    assert "interrupted" in res.err
    partial = root / "a" / "x.bin.partial"
    assert partial.stat().st_size == 1000 and partial.read_bytes() == blob[:1000]
    assert t1.calls == [(_url(entry), 0)]

    t2 = FakeTransport({_url(entry): ok_stream(blob)})
    res2 = run(tmp_path, manifest, root, transport=t2)
    assert res2.rc == 0, res2.err
    assert "resume" in res2.out and "resumed-and-verified" in res2.out
    assert t2.calls == [(_url(entry), 1000)]
    assert (root / "a" / "x.bin").read_bytes() == blob
    _no_partial_left(root)


def test_resume_rehashes_existing_partial(tmp_path):
    """A partial with a WRONG prefix (server lied earlier / corruption) must
    be caught by the full-file rehash, never published, and cleaned up as
    unrecoverable."""
    blob = b"truth" * 100
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.bin.partial").write_bytes(b"garbage"[:len(blob) // 10]
                                               + blob[len(blob) // 10:400])
    tail_plan = lambda start: FakeStream(  # noqa: E731
        206, {"Content-Range": f"bytes {start}-{len(blob) - 1}/{len(blob)}"},
        blob[start:])
    t = FakeTransport({_url(entry): tail_plan})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_INTEGRITY
    assert "SHA256 mismatch" in res.err
    assert not (root / "a" / "x.bin").exists()
    _no_partial_left(root)


def test_ignored_range_never_concatenated(tmp_path):
    blob = b"R" * 400
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    partial = root / "a" / "x.bin.partial"
    partial.write_bytes(blob[:150])
    t = FakeTransport({_url(entry): FakeStream(200, {}, blob)})  # Range ignored
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_NETWORK
    assert "ignored the Range" in res.err
    assert partial.read_bytes() == blob[:150]  # untouched, not concatenated
    assert not (root / "a" / "x.bin").exists()


@pytest.mark.parametrize("headers,fragment", [
    ({"Content-Range": "bytes 0-399/400"}, "starts at 0"),      # wrong start
    ({"Content-Range": "bytes 150-399/999999"}, "total"),       # wrong total
    ({"Content-Range": "bytes 150-999/400"}, "coherent"),       # bad span
    ({}, "missing or malformed Content-Range"),                 # absent
])
def test_forged_or_wrong_content_range_refused(tmp_path, headers, fragment):
    blob = b"S" * 400
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    partial = root / "a" / "x.bin.partial"
    partial.write_bytes(blob[:150])
    t = FakeTransport({_url(entry): FakeStream(206, headers, blob[150:])})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_NETWORK, res.err
    assert fragment in res.err
    assert partial.read_bytes() == blob[:150]
    assert not (root / "a" / "x.bin").exists()


def test_partial_larger_than_manifest_refused_untouched(tmp_path):
    blob = b"P" * 100
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    partial = root / "a" / "x.bin.partial"
    partial.write_bytes(b"junk" * 60)  # 240 > 100
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_INTEGRITY
    assert "larger than the manifest" in res.err
    assert partial.stat().st_size == 240
    assert t.calls == []


def test_symlinked_partial_refused_external_untouched(tmp_path):
    blob = b"Q" * 120
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"do not touch")
    os.symlink(outside, root / "a" / "x.bin.partial")
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_REFUSED
    assert "symlink" in res.err
    assert outside.read_bytes() == b"do not touch"
    assert t.calls == []


# ---------------------------------------------------------------------------
# Path/symlink escape refusals
# ---------------------------------------------------------------------------


def test_symlinked_directory_on_write_path_refused(tmp_path):
    blob = b"E" * 50
    entry = art(blob, "link/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir(parents=True)
    os.symlink(outside, root / "link")
    t = FakeTransport({_url(entry): ok_stream(blob)})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_REFUSED
    assert "symlinked directory" in res.err
    assert not (outside / "x.bin").exists()
    assert not list(outside.iterdir())
    assert t.calls == []


def test_final_that_is_symlink_to_outside_refused_never_overwritten(tmp_path):
    blob = b"F" * 60
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(blob)  # even a CONTENT-CORRECT external target
    os.symlink(outside, root / "a" / "x.bin")
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_REFUSED
    assert outside.read_bytes() == blob
    assert t.calls == []


# ---------------------------------------------------------------------------
# Disk-space refusal and no-clobber publish
# ---------------------------------------------------------------------------


def test_disk_space_refusal_before_any_write_or_network(tmp_path):
    blob = b"D" * 5000
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t,
              disk_free=lambda _p: 100)
    assert res.rc == fetch.EXIT_DISK
    assert "insufficient disk space" in res.err and "free" in res.err
    assert not root.exists()
    assert t.calls == []


def test_publish_link_fails_on_existing_final_never_clobbers(tmp_path):
    dest = tmp_path / "final.bin"
    dest.write_bytes(b"sentinel-not-to-be-touched")
    partial = tmp_path / "final.bin.partial"
    partial.write_bytes(b"verified-bytes")
    with pytest.raises(fetch.RefuseClobberError):
        fetch._atomic_no_clobber_publish(partial, dest)
    assert dest.read_bytes() == b"sentinel-not-to-be-touched"
    assert partial.exists()  # partial kept for the operator to inspect


# ---------------------------------------------------------------------------
# Error-message hygiene
# ---------------------------------------------------------------------------


def test_http_error_message_strips_signed_query(tmp_path):
    import urllib.error
    err = urllib.error.HTTPError(
        "https://cas-bridge.xethub.hf.co/x/abc?token=SU-PER-SECRET&sig=deadbeef",
        403, "Forbidden", {}, None)
    msg = fetch._fmt_http_error(err)
    assert "SU-PER-SECRET" not in msg and "deadbeef" not in msg and "?" not in msg
    assert "HTTP 403" in msg and "/x/abc" in msg
    assert fetch._clean_url("https://h.co/p?q=SECRET#frag") == "https://h.co/p"


def test_http_redirect_downgrade_refused_without_leaking_query():
    """The HTTPS-only redirect handler refuses https→http downgrades and the
    error message never echoes the (possibly signed) redirect query."""
    import urllib.request as ur
    req = ur.Request(f"https://{fetch.HF_HOST}/Qwen/Qwen-Image-2.1/resolve/"
                     f"{REV}/a/x.bin?signature=SEKRET")
    handler = fetch._HttpsOnlyRedirect()
    with pytest.raises(fetch.NetworkError) as excinfo:
        handler.redirect_request(req, None, 302, "Found", {},
                                 "http://cdn.example/x?token=SEKRET")
    assert "non-HTTPS" in str(excinfo.value)
    assert "SEKRET" not in str(excinfo.value)


def test_no_heavy_modules_after_inprocess_activity(tmp_path):
    blob = b"H" * 32
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    t = FakeTransport({_url(entry): ok_stream(blob)})
    assert run(tmp_path, manifest, root, transport=t).rc == 0
    assert run(tmp_path, manifest, root, fetch_mode=False).rc == 0
    assert not set(HEAVY_MODULES) & set(sys.modules)
