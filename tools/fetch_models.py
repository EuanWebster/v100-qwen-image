#!/usr/bin/env python3
"""Explicit operator asset fetcher for the committed model manifest (task M2.1 follow-on).

Standalone, **pure stdlib** (plus the pure-stdlib ``qwen_engine.manifest``
parser resolved from ``<repo>/src`` when the package is not installed): no
torch / diffusers / transformers, no model load, no inference. It exists so a
download is an *unmistakable operator action*, never a side effect of imports,
the resolver, or a CLI run — those paths stay no-download and network-free.

Contract (see ``docs/models.md`` §4 and ``/tmp/v100-m2-fetch-task.md``):

* Default (and ``--help`` / plan) is **read-only and offline**: it verifies
  disk state against the manifest and prints a deterministic JSON report; no
  socket is ever constructed. The ONLY network/write path is an explicit
  ``--fetch`` with explicit ``--manifest`` and ``--models-dir`` — there are no
  implicit defaults, no ambient HF cache, no branch/tag fallback (the
  manifest's pinned 40-hex commit revision is the only revision accepted,
  enforced by the schema-v1 parser).
* Fixed HTTPS official Hub: every URL is
  ``https://huggingface.co/<repo>/resolve/<revision>/<filename>`` — the host
  is a constant, repo segments are restricted to ``[A-Za-z0-9._-]``, and
  redirects to non-HTTPS targets are refused. Signed redirect query
  parameters are never printed: URLs shown anywhere are query-stripped.
* Bounded streaming: chunks of at most 1 MiB; never buffers a whole file.
  The expected byte count is enforced as an upper bound *while reading*.
* Publish only on an exact size+SHA256 match, atomically and no-clobber
  (``os.link`` of the verified ``<file>.partial`` next to the target, then
  unlink; never ``rename``/``replace`` over an existing file). A wrong,
  truncated or oversized response can never become a final asset.
* Resume: a partial lives in the same asset directory as ``<file>.partial``.
  A resume request sends ``Range: bytes=<offset>-`` and is accepted only on
  status 206 with a well-formed ``Content-Range`` whose start equals the
  requested offset (and total equal to the manifest size). A server that
  ignores Range (200) or misstates Content-Range is refused — bytes are never
  silently concatenated, and the partial is left untouched. Final
  verification always **rehashes the full assembled file from disk**
  (existing partial prefix included), so a garbage partial can never publish.
* Existing files: a final file that already verifies is skipped; a final file
  whose identity does not match the manifest (or that is not a regular file /
  escapes the asset root) is refused and NEVER deleted or overwritten. A
  symlinked partial or symlinked directory on the write path is refused; the
  tool never follows a symlink it is about to write through.
* Before any transfer, a bounded disk estimate (remaining manifest bytes plus
  a 1 MiB headroom per pending file) is compared with fresh ``statvfs`` free
  space. No RAM/GPU fit claim is made anywhere — capacity evidence belongs
  to M5 calibration, not to this tool.
* ``sandbox.yaml`` and everything outside the exact ``--models-dir`` asset
  root are never touched.

Partial cleanup (the only deletions this tool performs, always on its own
``*.partial`` files, never on finals): an oversized stream or a failed final
hash proves the partial is unrecoverable, so it is removed; an interrupted or
truncated stream keeps the partial for resume.

Exit codes (TOOL-SPECIFIC, deliberately distinct from the cli_gen.py engine
contract): ``0`` success · ``2`` usage/manifest error · ``3`` insufficient
disk space · ``4`` network/HTTP error (including a refused Range response) ·
``5`` downloaded-bytes integrity failure (size/hash) · ``6`` refusal to
clobber or an unsafe path.

This assignment ships CODE + OFFLINE TESTS ONLY: no real 33.1 GB transfer has
been executed; running ``--fetch`` against the live Hub is a separate
lead/operator decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat as _stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

# ---------------------------------------------------------------------------
# Import the pure-stdlib manifest parser (repo layout fallback, no install).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
try:  # pragma: no cover - trivial import preference
    import qwen_engine.manifest as _manifest_mod  # noqa: F401
except Exception:  # not installed: fall back to the repo's src tree
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    import qwen_engine.manifest as _manifest_mod  # noqa: E402

from qwen_engine.manifest import (  # noqa: E402
    ManifestError,
    ModelArtifact,
    ModelManifest,
    STATE_PRESENT,
    load_manifest,
    resolve_manifest,
)

#: The ONE official host this tool will ever talk to.
HF_HOST = "huggingface.co"
#: Max bytes held in memory per streamed chunk.
CHUNK_BYTES = 1 << 20
#: Disk headroom charged per pending file in the bounded estimate.
DISK_HEADROOM_PER_FILE = 1 << 20
#: Per-read socket timeout for the real transport.
DEFAULT_TIMEOUT_S = 60.0

_REPO_SEGMENT_RE = re.compile(r"[A-Za-z0-9._-]+")
_CONTENT_RANGE_RE = re.compile(
    r"^bytes (\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DISK = 3
EXIT_NETWORK = 4
EXIT_INTEGRITY = 5
EXIT_REFUSED = 6


# ---------------------------------------------------------------------------
# Typed errors (single-line messages, no traceback on the console)
# ---------------------------------------------------------------------------


class FetchError(Exception):
    exit_code = EXIT_USAGE


class UsageError(FetchError):
    exit_code = EXIT_USAGE


class DiskSpaceError(FetchError):
    exit_code = EXIT_DISK


class NetworkError(FetchError):
    exit_code = EXIT_NETWORK


class IntegrityError(FetchError):
    exit_code = EXIT_INTEGRITY


class UnsafePathError(FetchError):
    exit_code = EXIT_REFUSED


class RefuseClobberError(FetchError):
    exit_code = EXIT_REFUSED


# ---------------------------------------------------------------------------
# URL construction and sanitation
# ---------------------------------------------------------------------------


def _clean_url(url: str) -> str:
    """Strip query/fragment so a signed-URL redirect target can never be
    echoed (credentials and signatures live in the query)."""
    try:
        parts = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc,
                                        parts.path, "", ""))
    except ValueError:
        return "<unparseable url: query stripped>"


def artifact_url(art: ModelArtifact) -> str:
    """Canonical pinned download URL — host is a constant, so a tampered
    manifest can never steer this tool to another host."""
    segments = art.repo.split("/")
    if len(segments) != 2 or not all(_REPO_SEGMENT_RE.fullmatch(s) for s in segments):
        raise UsageError(
            f"manifest repo {art.repo!r} is not a single 'namespace/name' "
            "Hub repository id — refusing to build a URL from it")
    path = "/".join(urllib.parse.quote(seg, safe="")
                    for seg in art.filename.split("/"))
    return (f"https://{HF_HOST}/{segments[0]}/{segments[1]}"
            f"/resolve/{art.revision}/{path}")


# ---------------------------------------------------------------------------
# Transport (the ONLY network object in the program; tests inject a fake)
# ---------------------------------------------------------------------------


class Stream:
    """Protocol the fetcher relies on: status int, case-insensitive headers
    mapping, bounded ``read(n)``, ``close()``."""

    status = 200
    headers: Any = {}

    def read(self, n: int) -> bytes:  # pragma: no cover - protocol
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - protocol
        pass


def _fmt_http_error(exc: urllib.error.HTTPError) -> str:
    return (f"HTTP {exc.code} for {_clean_url(exc.url)}"
            f"{' (' + exc.reason + ')' if exc.reason else ''} "
            "— query parameters stripped from this message")


class UrllibTransport:
    """HTTPS-only transport against the official Hub (urllib, stdlib)."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_HttpsOnlyRedirect())

    def get(self, url: str, start: int) -> Stream:
        headers = {"User-Agent": "v100-qwen-image-fetch/1",
                   # Never let a transparent gzip corrupt manifest-pinned bytes:
                   "Accept-Encoding": "identity"}
        if start > 0:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            resp = self._opener.open(request, timeout=self._timeout)
        except urllib.error.HTTPError as exc:
            raise NetworkError(_fmt_http_error(exc)) from None
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            detail = " ".join(str(reason).split())[:200]
            raise NetworkError(
                f"network failure for {_clean_url(url)}: {type(reason).__name__}: {detail}"
            ) from None
        except OSError as exc:  # socket-level mid-connect failures
            raise NetworkError(
                f"network failure for {_clean_url(url)}: "
                f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}") from None
        return _UrllibStream(resp)


class _UrllibStream(Stream):
    def __init__(self, resp) -> None:
        self._resp = resp
        self.status = getattr(resp, "status", None) or resp.getcode()
        self.headers = resp.headers

    def read(self, n: int) -> bytes:
        try:
            return self._resp.read(n)
        except urllib.error.HTTPError as exc:
            raise NetworkError(_fmt_http_error(exc)) from None
        except OSError as exc:
            raise NetworkError(
                f"connection interrupted: {type(exc).__name__}") from None

    def close(self) -> None:
        try:
            self._resp.close()
        except OSError:
            pass


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise NetworkError(
                f"redirect from {_clean_url(req.full_url)} to a non-HTTPS "
                "target refused (official Hub is HTTPS-only)")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human(n: int) -> str:
    return f"{n:,} B"


def _default_disk_free(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(str(probe)).free
    except OSError as exc:
        raise UnsafePathError(f"cannot read free space near {probe}: {exc}") from None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        while chunk := os.read(fd, CHUNK_BYTES):
            digest.update(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


def _partial_path(dest: Path) -> Path:
    return dest.parent / (dest.name + ".partial")


def _check_write_path_symlinks(root: Path, filename: str) -> None:
    """Refuse ANY symlink on the directory path we are about to create/write
    through (stricter than the read-side resolver, which only requires
    containment)."""
    node = root
    for part in PurePosixPath(filename).parts[:-1]:
        node = node / part
        if os.path.islink(node):
            raise UnsafePathError(
                f"symlinked directory on the write path: {node} — refusing "
                "to write through symlinks (path escape risk)")
        if node.exists() and not node.is_dir():
            raise UnsafePathError(f"not a directory on the write path: {node}")


def _resume_offset(art: ModelArtifact, partial: Path) -> int:
    """Size of a resumable partial; raises on anything we refuse to append to."""
    try:
        st = partial.lstat()
    except FileNotFoundError:
        return 0
    except OSError as exc:
        raise UnsafePathError(f"cannot stat partial {partial}: {exc}") from None
    if _stat.S_ISLNK(st.st_mode):
        raise UnsafePathError(f"partial is a symlink, refusing to resume: {partial}")
    if not _stat.S_ISREG(st.st_mode):
        raise UnsafePathError(f"partial is not a regular file: {partial}")
    if st.st_size > art.bytes:
        raise IntegrityError(
            f"partial for '{art.filename}' is {_human(st.st_size)} — larger "
            f"than the manifest's {_human(art.bytes)}; the partial is corrupt. "
            f"Delete it deliberately and rerun: {partial}")
    return st.st_size


def _validate_response(art: ModelArtifact, stream: Stream, url: str,
                       start: int) -> None:
    """Range/encoding gates applied BEFORE a single byte is written."""
    encoding = (stream.headers.get("Content-Encoding") or "identity").strip()
    if encoding.lower() not in ("identity",):
        raise NetworkError(
            f"server applied Content-Encoding {encoding!r} for {_clean_url(url)}; "
            "refusing to hash transformed bytes")
    if start == 0:
        if stream.status != 200:
            raise NetworkError(
                f"expected HTTP 200 for {_clean_url(url)}, got {stream.status}")
        return
    if stream.status != 206:
        raise NetworkError(
            f"server ignored the Range request for '{art.filename}' "
            f"(asked to resume at offset {start}, got HTTP {stream.status}); "
            f"refusing to concatenate a full body onto the partial. The "
            f"partial is preserved untouched — rerun after the server "
            f"supports Range, or delete the partial deliberately.")
    raw = stream.headers.get("Content-Range")
    m = _CONTENT_RANGE_RE.match(raw.strip()) if isinstance(raw, str) else None
    if m is None:
        raise NetworkError(
            f"missing or malformed Content-Range on the 206 resume response "
            f"for '{art.filename}' (got {raw!r}); refusing to guess the "
            "resume offset")
    first, last, total = m.group(1), m.group(2), m.group(3)
    if int(first) != start:
        raise NetworkError(
            f"Content-Range starts at {first}, not the requested offset "
            f"{start}, for '{art.filename}' — refusing to concatenate a "
            "range the server did not actually resume from")
    if total not in ("*", str(art.bytes)):
        raise NetworkError(
            f"Content-Range total {total!r} disagrees with the manifest size "
            f"{art.bytes} for '{art.filename}'")
    if total != "*" and int(last) + 1 != int(total):
        raise NetworkError(
            f"Content-Range {raw!r} is not a coherent byte span")


def _download(art: ModelArtifact, dest: Path, transport,
              disk_free_fn) -> str:
    """Fetch one artifact into ``dest`` (verify → stream → rehash → publish).
    Returns a short outcome string; raises FetchError on any refusal."""
    url = artifact_url(art)
    partial = _partial_path(dest)
    start = _resume_offset(art, partial)

    # Disk estimate for THIS run's remaining bytes, bounded and honest:
    remaining = art.bytes - start
    need = remaining + DISK_HEADROOM_PER_FILE
    probe_dir = dest.parent if dest.parent.exists() else dest.parent.parent
    free = disk_free_fn(probe_dir if probe_dir.exists() else dest.parent)
    if free < need:
        raise DiskSpaceError(
            f"'{art.filename}': still needs ≈ {_human(need)} (remaining "
            f"{_human(remaining)} + 1 MiB headroom) but only {_human(free)} "
            f"free on the filesystem holding {probe_dir} — nothing written")

    mode = "wb" if start == 0 else "ab"
    if start == 0 and partial.exists():
        partial.unlink()  # it was 0 bytes (only path here); fresh start
    if mode == "ab":
        print(f"  resume '{art.filename}' at offset {start:,} "
              f"of {_human(art.bytes)}", flush=True)
    else:
        print(f"  fetch '{art.filename}' -> {_clean_url(url)}", flush=True)

    written = start
    try:
        stream = transport.get(url, start)
    except FetchError:
        raise
    except Exception as exc:  # transport contract is FetchError; be defensive
        raise NetworkError(f"transport failure for {_clean_url(url)}: "
                           f"{type(exc).__name__}") from None
    try:
        _validate_response(art, stream, url, start)
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if start else os.O_TRUNC)
        fd = os.open(partial, flags, 0o644)
        try:
            if start:
                os.ftruncate(fd, start)  # atomic truncate-to-EOF, no seek races
            else:
                os.ftruncate(fd, 0)
            while True:
                try:
                    chunk = stream.read(CHUNK_BYTES)
                except FetchError:
                    raise
                except Exception as exc:
                    raise NetworkError(
                        f"connection interrupted mid-stream at {written:,} B "
                        f"for '{art.filename}': {type(exc).__name__} — the "
                        f"partial is preserved for resume") from None
                if not chunk:
                    break
                written += len(chunk)
                if written > art.bytes:
                    os.close(fd)
                    fd = -1
                    try:
                        partial.unlink()
                    except OSError:
                        pass
                    raise IntegrityError(
                        f"'{art.filename}': server sent more than the manifest's "
                        f"{_human(art.bytes)} bytes — response rejected, no "
                        "final written and the unrecoverable partial removed")
                os.write(fd, chunk)
        finally:
            if fd != -1:
                os.close(fd)
    finally:
        stream.close()

    if written != art.bytes:
        raise NetworkError(
            f"'{art.filename}': response ended at {written:,} of "
            f"{_human(art.bytes)} bytes — partial preserved at "
            f"{partial} for a resumed retry")

    # Final verification ALWAYS rehashes every byte on disk (existing partial
    # prefix included), never just the freshly streamed tail:
    try:
        actual = _hash_file(partial)
    except OSError as exc:
        raise IntegrityError(f"cannot rehash partial {partial}: {exc}") from None
    if actual != art.sha256:
        try:
            partial.unlink()
        except OSError:
            pass
        raise IntegrityError(
            f"'{art.filename}': SHA256 mismatch after full rehash "
            f"(expected {art.sha256}, computed {actual}) — no final written, "
            "the unrecoverable partial removed")

    _atomic_no_clobber_publish(partial, dest)
    return "resumed-and-verified" if start else "downloaded-and-verified"


def _atomic_no_clobber_publish(partial: Path, dest: Path) -> None:
    """os.link is atomic and fails on an existing target (no clobber, never
    rename/replace over anything); the partial link is then removed."""
    try:
        os.link(partial, dest)
    except FileExistsError:
        raise RefuseClobberError(
            f"final file appeared during verification, refusing to clobber: "
            f"{dest} — the verified partial stays at {partial}") from None
    except OSError as exc:
        raise UnsafePathError(f"atomic publish failed for {dest}: {exc}") from None
    try:
        partial.unlink()
    except OSError as exc:
        print(f"warning: verified but could not remove partial {partial}: {exc}",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Manifest / plan plumbing
# ---------------------------------------------------------------------------


def load_manifest_checked(path: str) -> ModelManifest:
    try:
        return load_manifest(path)
    except ManifestError as exc:
        raise UsageError(f"manifest error: {exc}") from None


def plan_rows(manifest: ModelManifest, models_dir: str) -> tuple[list[dict], int]:
    """Read-only, offline per-artifact report rows + estimated bytes still
    needed (manifest metadata minus verified-good files and existing partials)."""
    root = Path(models_dir)
    report = resolve_manifest(manifest, root)
    rows: list[dict] = []
    estimate = 0
    for status in sorted(report.statuses,
                         key=lambda s: s.artifact.identity):
        art = status.artifact
        row: dict[str, Any] = {
            "filename": art.filename,
            "role": art.role,
            "repo": art.repo,
            "revision": art.revision,
            "bytes": art.bytes,
            "sha256": art.sha256,
            "state": status.state,
            "url": artifact_url(art),
        }
        if status.state == STATE_PRESENT:
            row["partial_bytes"] = 0
        else:
            partial = _partial_path(root / art.filename)
            partial_bytes = 0
            try:
                pst = partial.lstat()
                if _stat.S_ISREG(pst.st_mode):
                    partial_bytes = min(pst.st_size, art.bytes)
            except OSError:
                partial_bytes = 0
            row["partial_bytes"] = partial_bytes
            estimate += art.bytes - partial_bytes
        rows.append(row)
    return rows, estimate


def do_plan(manifest_path: str, models_dir: str) -> int:
    manifest = load_manifest_checked(manifest_path)
    rows, estimate = plan_rows(manifest, models_dir)
    doc = {
        "mode": "plan",
        "manifest": str(manifest_path),
        "models_dir": str(models_dir),
        "schema_version": manifest.schema_version,
        "network": "offline — plan performs no socket access and no writes",
        "artifact_count": len(rows),
        "pending_or_bad": sum(1 for r in rows if r["state"] != STATE_PRESENT),
        "estimated_bytes_still_needed": estimate,
        "artifacts": rows,
    }
    print(json.dumps(doc, sort_keys=True, indent=2))
    return EXIT_OK


def do_fetch(manifest_path: str, models_dir: str, transport,
             disk_free_fn) -> int:
    manifest = load_manifest_checked(manifest_path)
    root = Path(models_dir)
    print(f"fetch plan: {len(manifest.artifacts)} pinned artifact(s) from "
          f"https://{HF_HOST} at their pinned revisions into {root}",
          flush=True)

    # Refuse ALL unsafe paths before creating a single directory:
    for art in manifest.artifacts:
        _check_write_path_symlinks(root, art.filename)

    report = resolve_manifest(manifest, root)
    statuses = {s.artifact.identity: s for s in report.statuses}
    pending: list[ModelArtifact] = []
    for art in manifest.artifacts:
        status = statuses[art.identity]
        if status.state == STATE_PRESENT:
            print(f"skip  '{art.filename}': already present and verified",
                  flush=True)
            continue
        if status.state == "missing":
            pending.append(art)
            continue
        # present-but-bad / path escape / unreadable: never touch, never fetch:
        raise RefuseClobberError(
            f"existing file for '{art.filename}' does not verify "
            f"({status.state}: {status.error}) — this tool never overwrites "
            "or deletes a final asset; resolve it deliberately (inspect and "
            "remove it by hand) and rerun")

    if not pending:
        print("nothing to fetch: every pinned artifact is present and verified",
              flush=True)
        return EXIT_OK

    remaining = 0
    for art in pending:
        partial = _partial_path(root / art.filename)
        start = _resume_offset(art, partial)
        remaining += art.bytes - start
    need = remaining + DISK_HEADROOM_PER_FILE * len(pending)
    probe = root if root.exists() else root.parent
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = disk_free_fn(probe)
    print(f"disk check: ~{_human(need)} needed for {len(pending)} file(s) "
          f"(manifest bytes + 1 MiB headroom each); {_human(free)} free on "
          f"{probe} — this is a disk-space estimate only, no RAM/GPU fit is "
          f"claimed here", flush=True)
    if free < need:
        raise DiskSpaceError(
            f"insufficient disk space: ~{_human(need)} needed, {_human(free)} "
            f"free on {probe} — nothing was written")

    if not root.exists():
        os.makedirs(root, exist_ok=True)

    done = 0
    for art in pending:
        dest = root / art.filename
        _check_write_path_symlinks(root, art.filename)
        os.makedirs(dest.parent, exist_ok=True)
        if os.path.islink(dest):
            raise UnsafePathError(
                f"final path is a symlink: {dest} — refusing to write through it")
        if os.path.islink(_partial_path(dest)):
            raise UnsafePathError(
                f"partial path is a symlink: {_partial_path(dest)} — refusing")
        print(f"[{done + 1}/{len(pending)}] {art.filename} "
              f"(expected {_human(art.bytes)})", flush=True)
        outcome = _download(art, dest, transport, disk_free_fn)
        print(f"  done: {outcome}", flush=True)
        done += 1
    print(f"fetch complete: {done} fetched, "
          f"{len(manifest.artifacts) - done} skipped/verified", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fetch_models.py",
        description=(
            "Explicit operator fetcher for the committed schema-v1 model "
            "manifest: pinned repo/revision/filenames from the official "
            "HTTPS Hub, bounded streaming, SHA256+size verification, atomic "
            "no-clobber publish. Without --fetch this is a read-only offline "
            "plan. Never called implicitly by the engine."),
        epilog=(
            "Tool-specific exit codes (NOT the cli_gen.py engine contract): "
            "0 ok · 2 usage/manifest · 3 disk space · 4 network/HTTP "
            "(incl. refused Range response) · 5 downloaded-bytes integrity "
            "(size/hash) · 6 refusal to clobber / unsafe path."),
    )
    p.add_argument("--manifest", required=True,
                   help="committed model manifest path (explicit, no default)")
    p.add_argument("--models-dir", required=True,
                   help="asset root to write under (explicit, no default; "
                        "upstream layout preserved verbatim beneath it)")
    p.add_argument("--fetch", action="store_true",
                   help="ACTUALLY download/verify into --models-dir (the only "
                        "network and write path; a real 33.1 GB transfer is a "
                        "separate lead/operator decision)")
    return p


def main(argv: list[str] | None = None, *, transport=None,
         disk_free_fn=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: 0 = help, 2 = usage
        return int(exc.code or 0)
    try:
        if not args.fetch:
            return do_plan(args.manifest, args.models_dir)
        return do_fetch(args.manifest, args.models_dir,
                        transport if transport is not None else UrllibTransport(),
                        disk_free_fn if disk_free_fn is not None else _default_disk_free)
    except FetchError as exc:
        print(f"fetch error: {exc}", file=sys.stderr)
        return exc.exit_code
    except ManifestError as exc:
        print(f"fetch error: manifest error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as exc:
        print(f"fetch error: unexpected I/O failure: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
