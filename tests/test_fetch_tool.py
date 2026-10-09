"""Offline fake-stream tests for ``tools/fetch_models.py`` (M2.1 fetcher task).

Everything here is CPU-only, network-free and stdlib/heavy-import-free: the
fetcher's transport is *injected* (a fake byte stream), so no socket is ever
constructed, no real weight bytes exist anywhere, and every write lands in
pytest ``tmp_path`` (R7). Tests cover: happy path, already-good and
mismatched finals, truncated / oversized / wrong-hash responses,
interrupted-then-resumed Range, ignored Range, forged/wrong Content-Range
(incl. wildcard total and reversed span), partial too big, symlinked partial
/ symlinked write path / symlinked final / symlinked --models-dir root (and
its existing ancestors), path escape, disk-space refusal, no-clobber publish,
byte-complete-partial rehash-and-publish, nofollow/O_EXCL/size-drift partial
opens, short-``os.write`` completion, URL/host pinning and sanitized
(query-, userinfo- and reason-stripped) error messages, read-only/offline
plan+help, and (review2) the symlink+``..`` lexical-collapse escape from
``--models-dir`` plus ``.``/``..`` Hub repo segments, and (review items
5–6) hard-linked staging partials (pre-transport and opened-FD link-count
refusals, publish's own temporary link still valid) and staged-path
ancestor-directory manifest collisions.
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
    ({"Content-Range": "bytes 150-399/*"}, "wildcard"),         # hidden total
    ({"Content-Range": "bytes 150-999/400"}, "coherent"),       # bad span
    ({"Content-Range": "bytes 150-149/400"}, "coherent"),       # reversed span
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


# ---------------------------------------------------------------------------
# M2.1 fix pass — symlinked --models-dir root refused before anything happens
# ---------------------------------------------------------------------------


def test_symlinked_models_root_refused_external_untouched(tmp_path):
    blob = b"W" * 60
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "assets"
    os.symlink(outside, root)  # the explicit lexical root IS a symlink
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_REFUSED, res.err
    assert "symlink" in res.err
    assert not list(outside.iterdir())  # external dir untouched, no dirs made
    assert root.is_symlink()
    assert t.calls == []


def test_symlinked_ancestor_of_models_root_refused(tmp_path):
    blob = b"W" * 60
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    os.symlink(outside, link)
    root = link / "assets"  # final component does not exist (no speculation)
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_REFUSED, res.err
    assert "symlink" in res.err
    assert not list(outside.iterdir())
    assert t.calls == []


def test_missing_nonsymlink_root_components_are_not_refused(tmp_path):
    """Legitimate multi-component roots that simply do not exist yet are
    created and fetched into — the ancestor check refuses REAL symlinks,
    not absent parent paths by assumption."""
    blob = b"N" * 70
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "deep" / "nested" / "assets"
    t = FakeTransport({_url(entry): ok_stream(blob)})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == 0, res.err
    assert (root / "a" / "x.bin").read_bytes() == blob


# ---------------------------------------------------------------------------
# M2.1 fix pass — complete-partial recovery (no Range request from EOF)
# ---------------------------------------------------------------------------


def test_byte_complete_partial_rehashed_and_published_no_network(tmp_path):
    blob = os.urandom(400)
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.bin.partial").write_bytes(blob)  # exactly art.bytes
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == 0, res.err
    assert "partial-complete-and-verified" in res.out
    assert (root / "a" / "x.bin").read_bytes() == blob
    assert t.calls == []  # no impossible Range: bytes=<size>- request
    _no_partial_left(root)


def test_byte_complete_partial_wrong_hash_fails_never_published(tmp_path):
    blob = b"truth" * 40
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    partial = root / "a" / "x.bin.partial"
    partial.write_bytes(b"X" * len(blob))  # right size, wrong content
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_INTEGRITY, res.err
    assert "complete partial failed" in res.err
    assert not (root / "a" / "x.bin").exists()
    _no_partial_left(root)  # removed deliberately: unrecoverable
    assert t.calls == []


# ---------------------------------------------------------------------------
# M2.1 fix pass — nofollow / O_EXCL / size-drift partial opens, short writes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"),
                    reason="platform without O_NOFOLLOW")
def test_open_partial_resume_never_follows_symlink_to_target(tmp_path):
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"do not touch" * 8)
    link = tmp_path / "x.bin.partial"
    os.symlink(victim, link)
    with pytest.raises(fetch.UnsafePathError):
        fetch._open_partial(link, 10)
    assert victim.read_bytes() == b"do not touch" * 8


def test_open_partial_excl_refuses_concurrently_appeared_partial(tmp_path):
    p = tmp_path / "x.bin.partial"
    p.write_bytes(b"appeared by another process")
    with pytest.raises(fetch.RefuseClobberError):
        fetch._open_partial(p, 0)  # start==0 → O_EXCL create must not truncate
    assert p.read_bytes() == b"appeared by another process"


def test_open_partial_refuses_size_drift_without_truncating(tmp_path):
    p = tmp_path / "x.bin.partial"
    p.write_bytes(b"0" * 30)
    with pytest.raises(fetch.RefuseClobberError):
        fetch._open_partial(p, 10)  # lstat said 10, the FD says 30
    assert p.read_bytes() == b"0" * 30  # untouched, not truncated to 10


def test_open_partial_refuses_non_regular_file(tmp_path):
    not_a_file = tmp_path / "x.bin.partial"  # a DIRECTORY on the partial path
    not_a_file.mkdir()
    with pytest.raises(fetch.UnsafePathError):
        fetch._open_partial(not_a_file, 0)
    with pytest.raises(fetch.UnsafePathError):
        fetch._open_partial(not_a_file, 10)


def test_write_all_loops_until_chunk_fully_written(tmp_path, monkeypatch):
    real_write = os.write

    def capped(fd, b):
        return real_write(fd, memoryview(b)[:7])  # every call truncates

    monkeypatch.setattr(os, "write", capped)
    target = tmp_path / "out.bin"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fetch._write_all(fd, b"z" * 50)
    finally:
        os.close(fd)
    assert target.read_bytes() == b"z" * 50


def test_short_writes_end_to_end_still_verify(tmp_path, monkeypatch):
    """A filesystem/OS that accepts fewer bytes per os.write must still get
    every byte down: the loop continues until the chunk is fully written."""
    blob = os.urandom(900)
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    real_write = os.write

    def capped(fd, b):
        return real_write(fd, memoryview(b)[:64])

    monkeypatch.setattr(os, "write", capped)
    t = FakeTransport({_url(entry): ok_stream(blob)})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == 0, res.err
    assert (root / "a" / "x.bin").read_bytes() == blob
    _no_partial_left(root)


# ---------------------------------------------------------------------------
# M2.1 fix pass — diagnostics never echo userinfo, reasons or header values
# ---------------------------------------------------------------------------


def test_clean_url_strips_userinfo_query_and_fragment():
    s = fetch._clean_url(
        "https://user:SU-PER-SECRET@hf.example:443/repo/x?X-Amz-Signature"
        "=deadbeef#frag")
    assert s == "https://hf.example/repo/x"
    assert "SU-PER-SECRET" not in s and "deadbeef" not in s
    assert "@" not in s and "?" not in s and "#" not in s


def test_http_error_reason_carrying_signed_url_is_not_echoed():
    import urllib.error
    err = urllib.error.HTTPError(
        "https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/" + REV + "/a/x.bin",
        503, "Redirect to https://cas-bridge.xethub.hf.co/x?token=SIGNING-SECRET",
        {}, None)
    msg = fetch._fmt_http_error(err)
    assert "SIGNING-SECRET" not in msg and "cas-bridge" not in msg
    assert "HTTP 503" in msg and "huggingface.co/Qwen/Qwen-Image-2.1" in msg


def test_urlerror_reason_text_never_echoed(monkeypatch):
    import urllib.error
    tr = fetch.UrllibTransport()

    def boom(request, timeout=None):
        raise urllib.error.URLError(
            "cannot reach https://cas-bridge.example/p?token=SIGNING-SECRET")

    monkeypatch.setattr(tr._opener, "open", boom)
    url = (f"https://{fetch.HF_HOST}/{REPO}/resolve/{REV}/a/x.bin")
    with pytest.raises(fetch.NetworkError) as ei:
        tr.get(url, 0)
    assert "SIGNING-SECRET" not in str(ei.value)
    assert "URLError" in str(ei.value)
    assert url in str(ei.value)


def test_urlerror_oserror_reason_shows_only_type(monkeypatch):
    import urllib.error
    tr = fetch.UrllibTransport()

    def boom(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(
            111, "refused https://evil.example/?sig=SECRET-TEXT"))

    monkeypatch.setattr(tr._opener, "open", boom)
    with pytest.raises(fetch.NetworkError) as ei:
        tr.get(f"https://{fetch.HF_HOST}/{REPO}/resolve/{REV}/a/x.bin", 0)
    assert "SECRET-TEXT" not in str(ei.value)
    assert "ConnectionRefusedError" in str(ei.value)


def test_malformed_content_range_value_never_echoed(tmp_path):
    blob = b"C" * 400
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    partial = root / "a" / "x.bin.partial"
    partial.write_bytes(blob[:150])
    t = FakeTransport({_url(entry): FakeStream(
        206, {"Content-Range":
              "bytes garbage https://cas-bridge.example/x?token=HEADER-SECRET"},
        blob[150:])})
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_NETWORK
    assert "HEADER-SECRET" not in res.err
    assert partial.read_bytes() == blob[:150]  # untouched
    assert not (root / "a" / "x.bin").exists()


# ---------------------------------------------------------------------------
# M2.1 review2 — the symlink + '..' lexical-collapse escape, closed
# ---------------------------------------------------------------------------


def test_dotdot_after_symlink_in_models_dir_refused_no_external_write(tmp_path):
    """Reproduces the reported bypass and proves it is closed.

    ``in/link`` -> ``out/deep``; the supplied root is ``in/link/../assets``.
    ``os.path.abspath`` collapses that *lexically* to ``in/assets`` (a path
    with no symlink, which the old check inspected), while the filesystem
    resolves it through the link to ``out/assets`` — where makedirs/open
    would have written. The fetcher must refuse the '..' component BEFORE
    any mkdir or network call, leaving both trees untouched."""
    blob = b"X" * 80
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    inside = tmp_path / "in"
    inside.mkdir()
    outside_deep = tmp_path / "out" / "deep"
    outside_deep.mkdir(parents=True)
    os.symlink(outside_deep, inside / "link")
    root = inside / "link" / ".." / "assets"
    # The two normalizations this bypass exploited really do disagree:
    assert Path(os.path.abspath(str(root))) == inside / "assets"
    assert root.resolve() == tmp_path / "out" / "assets"

    t = FakeTransport({_url(entry): ok_stream(blob)})  # would write if reached
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_REFUSED, res.err
    assert "'..'" in res.err
    assert t.calls == []                                   # refused pre-network
    assert not (tmp_path / "out" / "assets").exists()      # no external write
    assert not list(outside_deep.iterdir())
    assert not (inside / "assets").exists()                # no mkdir happened


def test_dotdot_in_models_dir_refused_before_mkdir_and_network(tmp_path):
    entry = art(b"D" * 40, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets" / ".." / "elsewhere"
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_REFUSED, res.err
    assert "'..'" in res.err
    assert not (tmp_path / "elsewhere").exists()
    assert not (tmp_path / "assets").exists()
    assert t.calls == []


@pytest.mark.parametrize("root,refused", [
    ("a/../b", True), ("../a", True), ("/abs/x/..", True),
    ("/abs/x/../y", True), ("a/b", False), (".", False), ("a/./b", False),
])
def test_root_no_dotdot_check_is_exact(root, refused):
    if refused:
        with pytest.raises(fetch.UnsafePathError):
            fetch._check_root_no_dotdot(Path(root))
    else:
        fetch._check_root_no_dotdot(Path(root))


def test_plan_read_only_still_allows_dotdot_root_and_writes_nothing(
        tmp_path, monkeypatch):
    """Scope discipline: only the --fetch write path gained the '..' refusal.
    The offline read-only plan stays unaffected (it resolves like the
    resolver does and touches nothing)."""
    _no_network(monkeypatch)
    blob = b"P" * 10
    manifest = write_manifest(tmp_path, [art(blob, "a/x.bin")])
    root = tmp_path / "assets" / ".." / "other"
    res = run(tmp_path, manifest, root, fetch_mode=False)
    assert res.rc == 0, res.err
    doc = json.loads(res.out)
    assert doc["mode"] == "plan" and doc["network"].startswith("offline")
    assert not (tmp_path / "other").exists()
    assert not (tmp_path / "assets").exists()


# ---------------------------------------------------------------------------
# M2.1 review2 — '.' and '..' are not valid Hub repo namespace/name segments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("repo", ["Qwen/.", "Qwen/..", "./Qwen", "../Qwen"])
def test_dot_repo_segments_refused_no_url_no_network(tmp_path, repo):
    entry = art(b"Z" * 10, "a/x.bin")
    entry["repo"] = repo
    manifest = write_manifest(tmp_path, [entry])
    from qwen_engine.manifest import ModelArtifact
    with pytest.raises(fetch.UsageError):
        fetch.artifact_url(ModelArtifact(**entry))
    # Plan mode: malformed-manifest refusal, no URL ever printed.
    res = run(tmp_path, manifest, tmp_path / "assets", fetch_mode=False)
    assert res.rc == fetch.EXIT_USAGE, res.err
    assert "namespace/name" in res.err
    # Fetch mode: the URL is refused before any transport call; no asset file.
    t = ExplodingTransport()
    res2 = run(tmp_path, manifest, tmp_path / "assets2", transport=t)
    assert res2.rc == fetch.EXIT_USAGE, res2.err
    assert t.calls == []
    assert not (tmp_path / "assets2" / "a" / "x.bin").exists()
    _no_partial_left(tmp_path / "assets2")


def test_dotted_but_valid_repo_segments_still_accepted(tmp_path):
    """'.hidden' or 'a.b' are legitimate names — only the WHOLE segment
    being exactly '.' or '..' is refused."""
    from qwen_engine.manifest import ModelArtifact
    entry = art(b"V" * 4, "a/x.bin")
    for repo in ("a.b/c", ".hidden/c", "org/a.b-c_1"):
        url = fetch.artifact_url(ModelArtifact(**{**entry, "repo": repo}))
        assert url == (f"https://{fetch.HF_HOST}/{repo}/resolve/{REV}/a/x.bin")


# ---------------------------------------------------------------------------
# M2.1 review3 — valid-manifest final/staging-name collision preflight
# ---------------------------------------------------------------------------

# ``_partial_path(dest)`` stages at ``<final>.partial`` in the same
# directory, and schema-v1 rejects only duplicate *final* filenames: a
# schema-valid manifest may list both ``foo`` and ``foo.partial``. If
# ``foo.partial`` is an already-verified final and ``foo`` is missing,
# pre-fix code skipped the former, mistook it for ``foo``'s resume
# partial, appended to it and deleted it on the hash mismatch — breaking
# the never-overwrite/never-delete promise for verified finals. The
# fetcher must now refuse such a manifest BEFORE any mkdir/network/write.

_COLLISION_ORDERS = ["foo-first", "partial-first"]


def _collision_entries(tmp_path: Path) -> tuple[list[dict], bytes, bytes]:
    missing_blob = b"M" * 64            # artifact 'a/foo', absent on disk
    final_blob = b"P" * 32             # verified final 'a/foo.partial'
    entries = [art(missing_blob, "a/foo"), art(final_blob, "a/foo.partial")]
    return entries, missing_blob, final_blob


@pytest.mark.parametrize("order", _COLLISION_ORDERS)
def test_staging_collision_verified_final_preserved_before_transport(
        tmp_path, order):
    """Required regression, BOTH manifest orders: the verified final
    'a/foo.partial' must survive byte-for-byte and the fake transport
    (an exploding one) must never be reached."""
    entries, _missing_blob, final_blob = _collision_entries(tmp_path)
    if order == "partial-first":
        entries = list(reversed(entries))
    manifest = write_manifest(tmp_path, entries)
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    victim = root / "a" / "foo.partial"
    victim.write_bytes(final_blob)
    before = victim.stat().st_mtime_ns

    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_USAGE, res.err
    assert "collision" in res.err and "foo.partial" in res.err and "foo'" in res.err
    assert t.calls == []                                 # refused pre-network
    assert victim.read_bytes() == final_blob             # byte-for-byte intact
    assert victim.stat().st_mtime_ns == before           # never even opened
    assert not (root / "a" / "foo").exists()             # nothing fetched
    assert not (root / "a" / "foo.partial.partial").exists()  # no staging
    assert sorted(p.name for p in root.rglob("*") if p.is_file()) \
        == ["foo.partial"]                               # tree unchanged


@pytest.mark.parametrize("order", _COLLISION_ORDERS)
def test_staging_collision_refused_before_mkdir(tmp_path, order):
    """Nothing on disk yet: refusal happens before the asset root exists."""
    entries, _b1, _b2 = _collision_entries(tmp_path)
    if order == "partial-first":
        entries = list(reversed(entries))
    manifest = write_manifest(tmp_path, entries)
    root = tmp_path / "assets"
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_USAGE, res.err
    assert "collision" in res.err
    assert t.calls == []
    assert not root.exists()                             # no mkdir happened
    assert not list(tmp_path.rglob("*.partial"))


def test_staging_collision_refused_even_when_everything_verifies(tmp_path):
    """The check is static: even a fully-populated store (both finals
    present and correct, so 'nothing to fetch' would otherwise exit 0) is
    refused — the reusable tool contract must never look fetch-safe."""
    entries, blob_foo, blob_partial = _collision_entries(tmp_path)
    manifest = write_manifest(tmp_path, entries)
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    (root / "a" / "foo").write_bytes(blob_foo)
    (root / "a" / "foo.partial").write_bytes(blob_partial)
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_USAGE, res.err
    assert "collision" in res.err
    assert t.calls == []
    assert (root / "a" / "foo").read_bytes() == blob_foo
    assert (root / "a" / "foo.partial").read_bytes() == blob_partial


def test_plan_reports_staging_collision_read_only(tmp_path, monkeypatch):
    """Plan mode: typed usage error (exit 2) naming both filenames; the
    read-only offline path still performs no socket access and no writes."""
    _no_network(monkeypatch)
    entries, _b1, _b2 = _collision_entries(tmp_path)
    manifest = write_manifest(tmp_path, entries)
    root = tmp_path / "assets"
    res = run(tmp_path, manifest, root, fetch_mode=False)
    assert res.rc == fetch.EXIT_USAGE, res.err
    assert "collision" in res.err and "foo.partial" in res.err
    assert not root.exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_partial_named_final_alone_is_not_a_false_positive(tmp_path):
    """'a/foo.partial' as the ONLY artifact is a legitimate final whose own
    staging path is 'a/foo.partial.partial' — it must fetch normally."""
    blob = b"S" * 48
    entry = art(blob, "a/foo.partial")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    t = FakeTransport({_url(entry): ok_stream(blob)})
    res = run(tmp_path, manifest, root, transport=t,
              disk_free=lambda _p: 1 << 40)
    assert res.rc == fetch.EXIT_OK, res.err
    assert (root / "a" / "foo.partial").read_bytes() == blob
    assert not (root / "a" / "foo.partial.partial").exists()
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file()) == ["a/foo.partial"]


def test_cross_directory_partial_names_are_not_collisions(tmp_path):
    """'a/x' stages to 'a/x.partial'; a final 'b/x.partial' lives in a
    different directory — no collision, both fetch normally."""
    blob_x, blob_y = b"C" * 20, b"D" * 24
    e_x = art(blob_x, "a/x.bin")
    e_y = art(blob_y, "b/x.bin.partial")
    manifest = write_manifest(tmp_path, [e_x, e_y])
    root = tmp_path / "assets"
    t = FakeTransport({_url(e_x): ok_stream(blob_x),
                       _url(e_y): ok_stream(blob_y)})
    res = run(tmp_path, manifest, root, transport=t,
              disk_free=lambda _p: 1 << 40)
    assert res.rc == fetch.EXIT_OK, res.err
    assert (root / "a" / "x.bin").read_bytes() == blob_x
    assert (root / "b" / "x.bin.partial").read_bytes() == blob_y
    assert not (root / "a" / "x.bin.partial").exists()   # no staging created
    assert not (root / "b" / "x.bin.partial.partial").exists()


def test_file_directory_prefix_collision_refused_before_write(tmp_path):
    """Usability preflight: 'a' (file) and 'a/b' (needs dir 'a') can never
    both download — refused statically before mkdir/network/write."""
    blob_a, blob_b = b"F" * 8, b"G" * 8
    manifest = write_manifest(tmp_path,
                              [art(blob_a, "a"), art(blob_b, "a/b")])
    root = tmp_path / "assets"
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_USAGE, res.err
    assert "directory" in res.err
    assert t.calls == []
    assert not root.exists()


@pytest.mark.parametrize("filenames,refused", [
    (("foo", "foo.partial"), True),
    (("foo.partial", "foo"), True),
    (("a/x", "a/x.partial"), True),
    (("a/x.partial", "a/b/x"), False),
    (("a/x", "b/x.partial"), False),
    (("a/x.partial",), False),
    (("a/x", "a/x.part"), False),
    (("a", "a/b"), True),
    (("a/b", "a"), True),
    (("a/b", "a/b/c"), True),
    (("a/b", "a_x/c"), False),
    (("d/x", "d/x.partial", "d/y"), True),
    (("a/x", "a/x.partial/y"), True),
    (("a/x.partial/y", "a/x"), True),
    (("a/x", "b/x.partial/y"), False),
    (("a/x", "a/xpartial/y"), False),
    (("a/x.partial/y",), False),
])
def test_manifest_collision_check_is_exact(filenames, refused):
    from qwen_engine.manifest import ModelArtifact, ModelManifest
    arts = tuple(ModelArtifact(**art(b"T" * 4, fn)) for fn in filenames)
    manifest = ModelManifest(schema_version=1, artifacts=arts)
    if refused:
        with pytest.raises(fetch.UsageError):
            fetch._check_manifest_filename_collisions(manifest)
    else:
        fetch._check_manifest_filename_collisions(manifest)


def test_committed_manifest_stays_accepted():
    """The committed 24-entry manifest must still pass the extended static
    preflight — a read-only acceptance check, no disk state, no schema
    change."""
    fetch.load_manifest_checked(str(ROOT / "models" / "manifest.json"))


# ---------------------------------------------------------------------------
# Review fix 5 — a hard-linked staging partial must never be resumed into
# ---------------------------------------------------------------------------

# Root's /tmp/v100-fetch-hardlink-review.py: a verified final 'victim.bin'
# (b'prefix') hard-linked to 'new.bin.partial' let a resume append b'tail'
# THROUGH THE SHARED INODE and publish rc0 — silently rewriting victim.bin
# to b'prefixtail'. Staging files must be the sole name of their inode;
# empty, partial-length and byte-complete stagings refuse identically,
# before transport and again on the opened FD.

@pytest.mark.parametrize("shape", ["partial", "empty", "byte-complete"])
def test_hardlinked_staging_partial_refused_names_preserved(tmp_path, shape):
    root = tmp_path / "assets"
    root.mkdir()
    if shape == "empty":
        # 0-byte staging inode shared with an unrelated name.
        entries = [art(b"tail", "new.bin")]
        partial = root / "new.bin.partial"
        partial.write_bytes(b"")
        os.link(partial, root / "shared.bin")
        names_before = {"new.bin.partial": b"", "shared.bin": b""}
    elif shape == "partial":
        # The reproduced corruption: the verified final itself is the link.
        entries = [art(b"prefix", "victim.bin"), art(b"prefixtail", "new.bin")]
        (root / "victim.bin").write_bytes(b"prefix")
        os.link(root / "victim.bin", root / "new.bin.partial")
        names_before = {"victim.bin": b"prefix", "new.bin.partial": b"prefix"}
    else:  # byte-complete: full expected length on a shared inode — no
        # rehash read, no Range-from-EOF publish, refusal regardless.
        entries = [art(b"prefix", "victim.bin"), art(b"prefix", "new.bin")]
        (root / "victim.bin").write_bytes(b"prefix")
        os.link(root / "victim.bin", root / "new.bin.partial")
        names_before = {"victim.bin": b"prefix", "new.bin.partial": b"prefix"}
    manifest = write_manifest(tmp_path, entries)
    mtimes = {n: (root / n).stat().st_mtime_ns for n in names_before}
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t,
              disk_free=lambda _p: 1 << 40)
    assert res.rc == fetch.EXIT_REFUSED, res.err
    assert "hard link" in res.err
    assert t.calls == []                            # refused pre-transport
    for name, blob in names_before.items():         # every name + bytes kept
        assert (root / name).read_bytes() == blob
        assert (root / name).stat().st_mtime_ns == mtimes[name]
    assert (root / "new.bin.partial").stat().st_nlink == 2  # never replaced
    assert not (root / "new.bin").exists()          # nothing published


def test_link_added_after_preflight_refused_on_opened_fd_before_write(
        tmp_path):
    """The FD recheck closes the ordinary race: a concurrent link() landing
    between the preflight lstat and the opened-file fstat is refused BEFORE
    any truncate/seek/write — no bytes of either name change."""
    blob = b"prefixtail"
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    partial = root / "a" / "x.bin.partial"
    partial.write_bytes(b"prefix")                  # 6 of 10 bytes, nlink 1
    raced = root / "a" / "raced.bin"

    def attach_link(start):                         # the ordinary race
        os.link(partial, raced)
        return ok_stream(blob)(start)

    t = FakeTransport({_url(entry): attach_link})
    res = run(tmp_path, manifest, root, transport=t,
              disk_free=lambda _p: 1 << 40)
    assert res.rc == fetch.EXIT_REFUSED, res.err
    assert "hard link" in res.err
    assert len(t.calls) == 1 and t.calls[0][1] == 6  # only the resume probe
    assert partial.read_bytes() == b"prefix"         # not truncated/appended
    assert raced.read_bytes() == b"prefix"           # second name untouched
    assert not (root / "a" / "x.bin").exists()       # nothing published


def test_normal_publish_temporary_second_link_still_valid(tmp_path):
    """Integrity-verified publish creating its OWN temporary second link is
    the allowed, documented exception: resume → rehash → link → unlink must
    still succeed, leaving the final singly linked with no staging debris."""
    blob = b"prefixtail"
    entry = art(blob, "a/x.bin")
    manifest = write_manifest(tmp_path, [entry])
    root = tmp_path / "assets"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.bin.partial").write_bytes(b"prefix")  # sole link
    t = FakeTransport({_url(entry): ok_stream(blob)})
    res = run(tmp_path, manifest, root, transport=t,
              disk_free=lambda _p: 1 << 40)
    assert res.rc == 0, res.err
    final = root / "a" / "x.bin"
    assert final.read_bytes() == blob
    assert final.stat().st_nlink == 1
    _no_partial_left(root)


# ---------------------------------------------------------------------------
# Review fix 6 — staged path as another artifact's ancestor directory
# ---------------------------------------------------------------------------

# 'a/x' stages at 'a/x.partial'; 'a/x.partial/y' needs that very path as a
# DIRECTORY. Neither manifest order can ever download both — refuse the
# static conflict for BOTH plan and fetch, before mkdir/network/write.

_STAGING_ANCESTOR_ENTRIES = lambda: [art(b"K" * 8, "a/x"),
                                     art(b"L" * 16, "a/x.partial/y")]


@pytest.mark.parametrize("order", ["x-first", "y-first"])
def test_staging_ancestor_collision_refused_before_mkdir(tmp_path, order):
    entries = _STAGING_ANCESTOR_ENTRIES()
    if order == "y-first":
        entries.reverse()
    manifest = write_manifest(tmp_path, entries)
    root = tmp_path / "assets"
    t = ExplodingTransport()
    res = run(tmp_path, manifest, root, transport=t)
    assert res.rc == fetch.EXIT_USAGE, res.err
    assert "collision" in res.err and "a/x.partial" in res.err
    assert t.calls == []
    assert not root.exists()                        # no mkdir happened
    assert not list(tmp_path.rglob("*.partial"))


def test_plan_reports_staging_ancestor_collision_read_only(tmp_path,
                                                           monkeypatch):
    _no_network(monkeypatch)
    manifest = write_manifest(tmp_path, _STAGING_ANCESTOR_ENTRIES())
    root = tmp_path / "assets"
    res = run(tmp_path, manifest, root, fetch_mode=False)
    assert res.rc == fetch.EXIT_USAGE, res.err
    assert "collision" in res.err
    assert not root.exists()
    assert not list(tmp_path.rglob("*.partial"))
