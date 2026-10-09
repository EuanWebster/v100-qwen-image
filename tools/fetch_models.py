#!/usr/bin/env python3
"""Explicit operator asset fetcher for the committed model manifest (task M2.1 follow-on).

Standalone, **pure stdlib** (plus the pure-stdlib ``qwen_engine.manifest``
parser resolved from ``<repo>/src`` when the package is not installed): no
torch / diffusers / transformers, no model load, no inference. It exists so a
download is an *unmistakable operator action*, never a side effect of imports,
the resolver, or a CLI run — those paths stay no-download and network-free.

Contract (see ``docs/models.md`` §4):

* Default (and ``--help`` / plan) is **read-only and offline**: it verifies
  disk state against the manifest and prints a deterministic JSON report; no
  socket is ever constructed. The ONLY network/write path is an explicit
  ``--fetch`` with explicit ``--manifest`` and ``--models-dir`` — there are no
  implicit defaults, no ambient HF cache, no branch/tag fallback (the
  manifest's pinned 40-hex commit revision is the only revision accepted,
  enforced by the schema-v1 parser).
* Fixed HTTPS official Hub: every URL is
  ``https://huggingface.co/<repo>/resolve/<revision>/<filename>`` — the host
  is a constant, repo segments are restricted to ``[A-Za-z0-9._-]`` whole
  segments that are never ``.`` or ``..`` (a dot-segment would collapse
  inside the URL path), and redirects to non-HTTPS targets are refused.
  Signed redirect query parameters are never printed: URLs shown anywhere
  are query-stripped.
* Bounded streaming: chunks of at most 1 MiB; never buffers a whole file.
  The expected byte count is enforced as an upper bound *while reading*.
* Publish only on an exact size+SHA256 match, atomically and no-clobber
  (``os.link`` of the verified ``<file>.partial`` next to the target, then
  unlink; never ``rename``/``replace`` over an existing file). A wrong,
  truncated or oversized response can never become a final asset.
* Resume: a partial lives in the same asset directory as ``<file>.partial``.
  A resume request sends ``Range: bytes=<offset>-`` and is accepted only on
  status 206 with a well-formed numeric ``Content-Range`` whose start equals
  the requested offset, whose total equals the manifest size exactly (a
  wildcard ``*`` total is refused), and whose span is coherent
  (``start <= end < total``). A server that ignores Range (200) or misstates
  Content-Range is refused — bytes are never silently concatenated, and the
  partial is left untouched. A partial already at exactly the manifest byte
  length is rehashed instead of resumed: if it verifies it is published
  without any request (never an impossible Range from EOF); if it does not,
  integrity fails without publishing. Final verification always **rehashes
  the full assembled file from disk** (existing partial prefix included), so
  a garbage partial can never publish.
* Existing files: a final file that already verifies is skipped; a final file
  whose identity does not match the manifest (or that is not a regular file /
  escapes the asset root) is refused and NEVER deleted or overwritten. The
  ``--models-dir`` root itself (and any existing component of its path) must
  not be a symlink, and the root must contain no ``..`` component (a lexical
  ``..`` collapse after a symlinked step resolves elsewhere than the
  filesystem path, which would blind the symlink check) — both checked before
  any mkdir, network access or write, because a resolved symlinked root would
  otherwise define its own boundary.
  A symlinked partial or symlinked directory on the write path is refused;
  partials are opened ``O_NOFOLLOW`` (where the platform provides it), a new
  partial is created ``O_EXCL`` (a concurrently appeared partial is refused,
  never truncated), and the opened FD is re-verified as the expected regular
  file at the expected resume size before any byte is written. The tool
  never follows a symlink it is about to write through. A staging partial
  must be the SOLE name of its inode: a multiply hard-linked partial is
  refused (exit 6) before any transport or write — empty, partial-length
  or byte-complete alike, because resuming into a shared inode silently
  rewrites every other name sharing it, verified finals included — and the
  link count is re-checked on the opened FD before truncate/seek/write so
  an ordinary concurrent ``link()`` cannot slip past the preflight.
  Publish-time linking of a fully verified partial is the one deliberate,
  integrity-checked exception (see the publish bullet above).
* Manifest-safety preflight (static — no disk state, no network): before any
  mkdir/network/write the manifest is refused (usage error, exit 2) if one
  artifact's staged partial path ``<final>.partial`` equals another
  artifact's final filename — schema-v1 accepts a manifest listing both
  ``foo`` and ``foo.partial`` (only duplicate finals are rejected), yet
  fetching ``foo`` would then mistake the verified final ``foo.partial``
  for its own partial, append to it and delete it on a hash mismatch. A
  filename that is simultaneously another's directory (``a`` and ``a/b``)
  is refused the same way as a usability preflight, and so is an artifact's
  STAGED path being another artifact's ancestor directory (``a/x`` stages
  at ``a/x.partial``, which ``a/x.partial/y`` needs as a directory —
  neither manifest order can ever download both). The read-only offline
  plan reports this typed usage error too — never a write, never a socket.
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
#: Character-legal but semantically invalid whole-segment repo ids: '.' and
#: '..' would collapse inside the URL path and break the fixed
#: '<repo>/resolve/<revision>/<file>' shape, so they are never accepted.
_DOT_SEGMENTS = frozenset({".", ".."})
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


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _clean_url(url: str) -> str:
    """Render ONLY scheme, host/port and path: no userinfo (credentials may
    sit there), no query (signed-URL signatures live there), no fragment.
    Never echo the raw netloc — rebuild it from parsed host/port only."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
        if not host:
            return "<url without host: userinfo/query/fragment stripped>"
        if ":" in host:  # IPv6 literal needs its brackets back
            host = f"[{host}]"
        try:
            port = parts.port
        except ValueError:
            port = None
        if port is not None and port == _DEFAULT_PORTS.get(
                parts.scheme.lower()):
            port = None
        netloc = host if port is None else f"{host}:{port}"
        return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path,
                                        "", ""))
    except ValueError:
        return "<unparseable url: userinfo/query/fragment stripped>"


def artifact_url(art: ModelArtifact) -> str:
    """Canonical pinned download URL — host is a constant, so a tampered
    manifest can never steer this tool to another host."""
    segments = art.repo.split("/")
    if len(segments) != 2 or not all(
            s not in _DOT_SEGMENTS and _REPO_SEGMENT_RE.fullmatch(s)
            for s in segments):
        raise UsageError(
            f"manifest repo {art.repo!r} is not a single 'namespace/name' "
            "Hub repository id ('.' and '..' are not valid namespace/name "
            "components) — refusing to build a URL from it")
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
    # The server-supplied reason is untrusted text (it can carry a signed
    # URL or credentials), so it is NEVER echoed — only our own rendering.
    return (f"HTTP {exc.code} for {_clean_url(str(exc.url))} — host and path "
            "only; URL userinfo/query and the server reason text are not "
            "shown")


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
            # reason is untrusted (may embed a signed URL or credentials):
            # render only its TYPE, never its text.
            raise NetworkError(
                f"network failure for {_clean_url(url)}: "
                f"URLError({type(reason).__name__}) — reason text not "
                "shown (may contain signed URLs or credentials)") from None
        except OSError as exc:  # socket-level mid-connect failures
            raise NetworkError(
                f"network failure for {_clean_url(url)}: "
                f"{type(exc).__name__} — error text not shown (may contain "
                "signed URLs or credentials)") from None
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


def _check_root_no_dotdot(root: Path) -> None:
    """Refuse any ``..`` component in the operator-supplied ``--models-dir``
    BEFORE any normalization, mkdir, network access or write.

    ``os.path.abspath`` collapses ``..`` *lexically*, while the filesystem
    collapses it only after following symlinks: with ``in/link/../assets``
    where ``link`` is a symlink to ``out/deep``, the checker would inspect
    ``in/assets`` (no symlink) while ``os.makedirs``/``os.open`` would write
    under ``out/assets`` — outside the directory the operator named.
    Rejecting ``..`` up front keeps the component-by-component symlink
    check below sound. The read-only offline plan does not write and stays
    unaffected."""
    if any(part == ".." for part in root.parts):
        raise UnsafePathError(
            f"--models-dir contains a '..' component: {root} — '..' after a "
            "symlinked path component resolves on the filesystem somewhere "
            "else than its lexical collapse, so the fetcher refuses it; pass "
            "the asset root as a plain path without '..'")


def _check_root_symlinks(root: Path) -> None:
    """Refuse a symlinked asset root BEFORE any mkdir, network access or
    write. The read-side resolver treats the *resolved* root as its boundary,
    so an explicit lexical root that is (or sits under) a symlink would
    silently redirect every write outside the directory the operator named.
    Every EXISTING component of the given root path is checked; components
    that do not exist yet cannot be symlinks and are created as plain dirs.
    Sound only because ``_check_root_no_dotdot`` has already rejected any
    ``..`` component whose lexical collapse could hide a symlinked step."""
    node = Path(os.path.abspath(str(root)))
    while True:
        if os.path.islink(node):
            raise UnsafePathError(
                f"--models-dir root or one of its existing path components "
                f"is a symlink: {node} — refusing to fetch through it (a "
                "resolved symlinked root would define its own boundary and "
                "redirect writes outside)")
        if node.parent == node:
            return
        node = node.parent


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


def _resume_offset(art: ModelArtifact, partial: Path) -> tuple[int, bool]:
    """Size of a resumable partial plus whether one exists at all; raises on
    anything we refuse to append to. ``(0, False)`` means no partial. A
    multiply hard-linked partial is refused BEFORE any transport/write —
    resume appends through the shared inode and would silently corrupt every
    other name linked to it (verified finals included)."""
    try:
        st = partial.lstat()
    except FileNotFoundError:
        return 0, False
    except OSError as exc:
        raise UnsafePathError(f"cannot stat partial {partial}: {exc}") from None
    if _stat.S_ISLNK(st.st_mode):
        raise UnsafePathError(f"partial is a symlink, refusing to resume: {partial}")
    if not _stat.S_ISREG(st.st_mode):
        raise UnsafePathError(f"partial is not a regular file: {partial}")
    if st.st_nlink > 1:
        raise UnsafePathError(
            f"partial {partial} has {st.st_nlink} hard links — a resume "
            "staging file must be the sole name of its inode: appending "
            "through a shared inode would silently rewrite every other "
            "name linked to it, verified finals included. Refused before "
            "any read, transport or write; every name and its bytes are "
            "untouched — inspect the links and de-duplicate deliberately "
            "before rerunning")
    if st.st_size > art.bytes:
        raise IntegrityError(
            f"partial for '{art.filename}' is {_human(st.st_size)} — larger "
            f"than the manifest's {_human(art.bytes)}; the partial is corrupt. "
            f"Delete it deliberately and rerun: {partial}")
    return st.st_size, True


def _open_partial(partial: Path, start: int) -> int:
    """Open the partial for writing WITHOUT ever following a symlink, and
    re-verify on the opened FD that it is a REGULAR, SOLELY LINKED file of
    exactly the expected resume size (ordinary concurrent filesystem changes
    between the earlier lstat and this open — a size drift or a new hard
    link — are refused, not truncated or appended over):

    * ``start == 0``: create with ``O_CREAT | O_EXCL`` — a partial that
      appeared concurrently is refused, never truncated;
    * ``start  > 0``: open the existing file only (no ``O_CREAT``).

    ``O_NOFOLLOW`` (POSIX) closes the lstat→open symlink race; platforms
    without it keep the pre-existing lstat-based refusal above."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if start == 0:
        try:
            fd = os.open(partial,
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                         0o644)
        except FileExistsError:
            try:
                st = partial.lstat()
            except OSError:
                st = None
            if st is not None and not _stat.S_ISREG(st.st_mode):
                raise UnsafePathError(
                    f"the partial path {partial} is not a regular file — "
                    "refusing to write through it") from None
            raise RefuseClobberError(
                f"a partial appeared at {partial} while this fetch was "
                "starting — refusing to truncate a file that appeared "
                "concurrently; inspect or remove it deliberately") from None
        except OSError as exc:
            raise UnsafePathError(
                f"cannot create partial {partial}: "
                f"{type(exc).__name__} — refusing (symlink or path issue)") from None
    else:
        try:
            fd = os.open(partial, os.O_WRONLY | nofollow)
        except FileNotFoundError:
            raise RefuseClobberError(
                f"the partial vanished while resuming: {partial} — refusing "
                "to create it implicitly; rerun from scratch deliberately"
            ) from None
        except OSError as exc:
            raise UnsafePathError(
                f"cannot open partial {partial}: "
                f"{type(exc).__name__} — refusing (symlink or path issue)") from None
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            raise UnsafePathError(
                f"the opened partial {partial} is not a regular file — "
                "refusing to write through it")
        if st.st_nlink > 1:
            raise UnsafePathError(
                f"the opened partial {partial} has {st.st_nlink} hard links — "
                "refusing to truncate or append through an inode that is "
                "shared with another name (a link appeared after the "
                "pre-flight check, or the create path raced with a link); "
                "appending would silently rewrite every name sharing it")
        if st.st_size != start:
            raise RefuseClobberError(
                f"the partial {partial} changed while this fetch was "
                f"starting: expected {_human(start)} bytes, the opened file "
                f"has {_human(st.st_size)} — refusing to truncate or append "
                "over a file we no longer recognize")
        os.ftruncate(fd, start)  # no-op on a verified match; no seek races
        if start:
            os.lseek(fd, start, os.SEEK_SET)  # append AFTER the verified prefix
    except BaseException:
        os.close(fd)
        raise
    return fd


def _write_all(fd: int, data: bytes) -> None:
    """os.write may accept fewer bytes than offered: keep writing until this
    bounded chunk is fully on disk (a non-advancing write is an error, not
    an infinite loop)."""
    view = memoryview(data)
    while len(view) > 0:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("local write made no progress")
        view = view[n:]


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
            f"for '{art.filename}' (header value not echoed — untrusted); "
            "refusing to guess the resume offset")
    first, last, total = m.group(1), m.group(2), m.group(3)
    if int(first) != start:
        raise NetworkError(
            f"Content-Range starts at {first}, not the requested offset "
            f"{start}, for '{art.filename}' — refusing to concatenate a "
            "range the server did not actually resume from")
    if total == "*":
        raise NetworkError(
            f"Content-Range total is the wildcard '*' for '{art.filename}': "
            f"refusing to resume against a server that hides the total size "
            f"(the manifest pins exactly {_human(art.bytes)} bytes)")
    total_n = int(total)
    if total_n != art.bytes:
        raise NetworkError(
            f"Content-Range total disagrees with the manifest size "
            f"{art.bytes} for '{art.filename}' (header value not echoed)")
    last_n = int(last)
    if not (start <= last_n < total_n):
        raise NetworkError(
            f"Content-Range is not a coherent byte span "
            f"(need start <= end < total, total {total_n}) for "
            f"'{art.filename}' (header value not echoed)")


def _download(art: ModelArtifact, dest: Path, transport,
              disk_free_fn) -> str:
    """Fetch one artifact into ``dest`` (verify → stream → rehash → publish).
    Returns a short outcome string; raises FetchError on any refusal."""
    url = artifact_url(art)
    partial = _partial_path(dest)
    start, present = _resume_offset(art, partial)

    # A partial that already holds every byte the manifest expects is
    # REHASHED, never "resumed" with an impossible Range from EOF: valid →
    # publish with no request at all; invalid → integrity failure, published
    # never, the unrecoverable partial removed deliberately.
    if present and start == art.bytes:
        print(f"  partial '{art.filename}' already holds all "
              f"{_human(art.bytes)} — rehashing, no Range request", flush=True)
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
                f"'{art.filename}': the byte-complete partial failed the "
                f"full SHA256 rehash (expected {art.sha256}, computed "
                f"{actual}) — no final written, the unrecoverable partial "
                "removed")
        _atomic_no_clobber_publish(partial, dest)
        return "partial-complete-and-verified"

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

    if start == 0 and present:
        # A 0-byte partial is the only present size that reaches here with
        # start == 0: remove it so _open_partial can create the real file
        # with O_EXCL. Anything that appeared concurrently is refused there,
        # never truncated.
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
    if start:
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
        fd = _open_partial(partial, start)
        try:
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
                try:
                    _write_all(fd, chunk)
                except OSError as exc:
                    raise NetworkError(
                        f"local write failure at {written:,} B for "
                        f"'{art.filename}': {type(exc).__name__} — the "
                        f"partial is preserved for resume") from None
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
    rename/replace over anything); the partial link is then removed.

    This is the one place a verified partial is *allowed* to hold two names:
    the second link is created only AFTER the full SHA256 rehash passed, and
    the partial name is unlinked immediately. If that unlink fails the extra
    ``*.partial`` name is recoverable debris, not a hazard — it shares (and
    therefore proves) the already-verified final's inode; a later fetch
    refuses to resume into a multiply-linked partial, and removing the stale
    name loses nothing while the final name stands."""
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


def _check_manifest_filename_collisions(manifest: ModelManifest) -> None:
    """Static, disk-free preflight refusal of manifest filename collisions,
    BEFORE any mkdir, network access or write (schema-valid is not the same
    as fetch-safe; the current committed manifest passes — this keeps the
    generic tool contract safe for any schema-v1 manifest).

    * **Staging collision (destructive, refused):** the staged partial of an
      artifact is ``<final>.partial`` in the same directory, and schema-v1
      rejects only duplicate *final* filenames — so a valid manifest may
      also list ``foo.partial`` as its own final artifact. Fetching ``foo``
      would then see the verified final ``foo.partial`` as ``foo``'s resume
      partial, append to it, and delete it on a hash mismatch — breaking the
      never-overwrite/never-delete promise for verified finals. Any pair
      ``(foo, foo.partial)`` (either order, any depth) is refused outright.
    * **File-versus-directory collision (refused as a usability preflight):**
      filenames ``a`` and ``a/b`` cannot both be fetched — whichever runs
      second hits ``makedirs`` over an existing file mid-run, after the
      first has already transferred. Refusing up front names the conflict
      while nothing has happened yet.
    * **Staged-path-versus-directory collision (same preflight, static):**
      ``a/x`` stages at ``a/x.partial``; an artifact ``a/x.partial/y`` needs
      ``a/x.partial`` to be a DIRECTORY. Neither order can ever download
      both — the first leaves a file where the second must mkdir — so any
      manifest whose artifact's staged partial path is a proper ancestor
      directory of another artifact's final (either order) is refused here,
      before mkdir/network/write.
    """
    identities = [art.identity for art in manifest.artifacts]
    # Schema-v1 guarantees uniqueness, so the identity -> index map is exact.
    index = {ident: i for i, ident in enumerate(identities)}
    staged_index = {f"{ident}.partial": i for i, ident in enumerate(identities)}
    for i, ident in enumerate(identities):
        staged = f"{ident}.partial"
        j = index.get(staged)
        if j is not None:
            raise UsageError(
                f"manifest filename collision: the staged partial of "
                f"artifacts[{i}] '{ident}' is '{staged}', which is also the "
                f"final filename of artifacts[{j}] — fetching '{ident}' "
                f"would treat the final file '{staged}' as a resume "
                "partial (append to it, delete it on a hash mismatch), "
                "which violates the never-overwrite/never-delete promise "
                "for finals; rename or remove one of the two artifacts "
                "and rerun")
        # Every proper ancestor directory of this artifact must not itself
        # be a final file somewhere else in the manifest:
        parts = ident.split("/")
        for depth in range(1, len(parts)):
            ancestor = "/".join(parts[:depth])
            k = index.get(ancestor)
            if k is not None:
                raise UsageError(
                    f"manifest filename collision: artifacts[{k}] "
                    f"'{ancestor}' is a final file while artifacts[{i}] "
                    f"'{ident}' needs '{ancestor}' as a directory — the two "
                    "can never both download; fix the manifest")
            # ... nor the staged partial path of another artifact (which
            # lands exactly there the moment that artifact is fetched):
            k = staged_index.get(ancestor)
            if k is not None and k != i:
                raise UsageError(
                    f"manifest filename collision: artifacts[{i}] "
                    f"'{ident}' needs '{ancestor}' as a directory, but "
                    f"fetching artifacts[{k}] '{identities[k]}' stages its "
                    f"resume partial at exactly that path — the two can "
                    "never both download; fix the manifest")


def load_manifest_checked(path: str) -> ModelManifest:
    try:
        manifest = load_manifest(path)
    except ManifestError as exc:
        raise UsageError(f"manifest error: {exc}") from None
    # Schema-parsing alone is not enough for the FETCHER's safety promise
    # (a v1-valid manifest can still stage one artifact's partial onto
    # another artifact's final); the static refusal below runs for BOTH the
    # read-only plan and --fetch, before any mkdir/network/write.
    _check_manifest_filename_collisions(manifest)
    return manifest


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
    # Refuse '..' in the asset root and symlinked root components BEFORE any
    # normalization, mkdir, network access or write:
    _check_root_no_dotdot(root)
    _check_root_symlinks(root)
    print(f"fetch plan: {len(manifest.artifacts)} pinned artifact(s) from "
          f"https://{HF_HOST} at their pinned revisions into {root}",
          flush=True)

    # Refuse ALL remaining unsafe paths before creating a single directory:
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
        start, _present = _resume_offset(art, partial)
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
