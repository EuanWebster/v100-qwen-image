"""First-run model setup: the fixed catalogue of files this app needs, and the
one worker thread that is allowed to fetch them (task ``q-model-setup.md``).

The old rule was "this app never downloads". It holds everywhere except here:
someone who has just cloned the repo has an empty ``models/`` and no way to
find out what is missing except by reading ``tools/fetch_models.py``. This
module answers three questions and nothing else:

* **what is needed** — a catalogue built from the committed manifests plus the
  enhancer and prompt pins below. It is fixed in code: the browser can never
  name a repository, a revision, a URL or a path — it ticks items by ``id``;
* **what is present** — a ``stat`` of each pinned file (name and expected byte
  size only). It NEVER hash-verifies: streaming 20 GB of weights on every
  status poll would stall the card for nothing, and a real generation job
  verifies every artifact itself. Size matches → installed; wrong size →
  ``partial``; absent → ``missing``;
* **how to get it** — the transfer itself belongs to
  :mod:`tools.fetch_models` (pinned HTTPS host, Range resume, SHA256 check,
  atomic no-clobber publish, symlink refusals). This module adds only the job
  bookkeeping, the cancellable byte counter, and the second pin the Qwen prompt
  files carry (the md5 in :mod:`qwen_engine.rewrite`).

Two deliberate limits: the worker never touches the GPU and never imports torch,
and one download runs at a time — 13 GiB into one directory is not a thing
worth doing twice at once.
"""
from __future__ import annotations

import hashlib
import importlib.util
import ipaddress
import json
import os
import stat as _stat
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from qwen_engine import manifest as _manifest
from qwen_engine import rewrite as _rewrite
#: target on its own: every write goes through a :class:`SetupItem`'s own root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Wire schemas (the browser-facing documents this module produces)
# ---------------------------------------------------------------------------

MODELS_SCHEMA = "qwen_engine.web_models.v1"
DOWNLOAD_REQUEST_SCHEMA = "qwen_engine.web_models_download_request.v1"
JOB_SCHEMA = "qwen_engine.web_models_job.v1"

# ---------------------------------------------------------------------------
# Item ids — a closed list. The POST body carries these and nothing else.
# ---------------------------------------------------------------------------

BASE = "base"
TURBO = "turbo"
Q8 = "q8"
PE_TEXT = "pe-t2i"
PE_EDIT = "pe-edit"
PROMPTS = "rewrite-prompts"
ITEM_IDS: tuple[str, ...] = (BASE, TURBO, Q8, PE_TEXT, PE_EDIT, PROMPTS)
#: Without these the app cannot make a picture at all, so they are always in a
#: download request and Generate stays disabled while any of them is absent.
REQUIRED_IDS: tuple[str, ...] = (BASE,)

#: An item whose files a download cannot fully satisfy: the official enhancers
#: also need a ``llama-server`` built for sm_70, which is not a Hub file. The
#: page says so in plain words instead of promising a working enhancer.
NEEDS_BINARY = "llama-server"

#: Charged per pending file in the free-space check, mirroring the fetcher's own
#: ``DISK_HEADROOM_PER_FILE`` so the web path never promises a transfer the
#: fetcher would refuse one byte later.
DISK_HEADROOM_BYTES = 1 << 20

#: The largest catalogue the route will ever serialise (six items today; the
#: cap is a bound, not a plan).
MAX_ITEMS = 16
MAX_FILES_PER_ITEM = 64

# ---------------------------------------------------------------------------
# Pinned extra sources (enhancers and the licence-restricted prompts)
#
# Where each value came from — these are identity pins, checked again after
# every download:
#   * the three GGUF sha256s are the operator's ``sha256sum`` of the copies
#     already installed on this machine (``output/coders/pe-sha256.txt``);
#   * the two revisions of the GGUF repos are the ``.cache/huggingface/trees``
#     entries of those same installs;
#   * the prompt files come from the OFFICIAL Qwen repos, at the pinned
#     revisions, and each is verified twice: the sha256 below (the installed
#     copy, hashed) and the md5 in :data:`qwen_engine.rewrite.SYSTEM_PROMPT_MD5`
#     (the pin that proves the operator handed us the prompt the weights were
#     trained on). The edit prompt's blob id at its pinned revision is
#     ``424fcfbe2968973c57fe67e3d629df1b0a1e1550``, identical to the copy
#     inside the enhancer checkpoint, so the two sources agree byte for byte.
# ---------------------------------------------------------------------------

PE_T2I_REPO = "lightx2v/Qwen-Image-2.1-PE-T2I-GGUF-Q8_0"
PE_T2I_REVISION = "b74e77d01fe90c8537e21d1c2f7b9bc886a17bf2"
PE_T2I_FILE = "Qwen-Image-2.1-PE-T2I-Q8_0.gguf"
PE_T2I_SHA256 = "57808ab9fa2959498ef107d442afc81435b71427ecf7ddf13fc616860af196b1"
PE_T2I_BYTES = 9_527_501_536

PE_I2I_REPO = "lightx2v/Qwen-Image-2.1-PE-I2I-GGUF-Q8_0"
PE_I2I_REVISION = "a7a26d38f4a13483ee5a740e48be9dd6f563f6bd"
PE_I2I_FILE = "Qwen-Image-2.1-PE-I2I-Q8_0.gguf"
PE_I2I_SHA256 = "77458dff91be9bd6b773d5cf95c60a3d9dbd264a6a699d617f867172646cf6cf"
PE_I2I_BYTES = 9_527_501_536
PE_I2I_MMPROJ_FILE = "mmproj-Qwen-Image-2.1-PE-I2I-BF16.gguf"
PE_I2I_MMPROJ_SHA256 = "8dedb71dbc3092dc47de9108ad373d68a12854e2527d59bd9399601738f3bce1"
PE_I2I_MMPROJ_BYTES = 921_704_608

PROMPT_T2I_REPO = "Qwen/Qwen-Image-2.1-PE-T2I"
PROMPT_T2I_REVISION = "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b"
PROMPT_I2I_REPO = "Qwen/Qwen-Image-2.1-PE-I2I"
PROMPT_I2I_REVISION = "72927bc08afc99b7888ceb7d7d51a12db3700bbd"
PROMPT_SOURCE_FILE = "system_prompt.txt"
PROMPT_SHA256: Mapping[str, str] = {
    "t2i": "a77c9a06c59b120741141d9514b95682bb8761d02bec49ca61def7b2b3d9fb99",
    "edit": "e378fea686a1431581ba4c654d332ae96adad633f144ae738ec8ce9c4fd66439",
}
PROMPT_BYTES: Mapping[str, int] = {"t2i": 10_045, "edit": 18_344}
#: Where the server keeps them, matching the launch flags the rewriter reads.
PROMPT_DEST: Mapping[str, str] = {"t2i": "system_prompt_t2i.txt",
                                  "edit": "system_prompt_edit.txt"}


# ---------------------------------------------------------------------------
# Typed failures (single-line messages; never a traceback, never prompt text)
# ---------------------------------------------------------------------------


class SetupError(Exception):
    """A refusal the caller can name. ``code`` is a stable wire token."""

    code = "setup_failed"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnknownItem(SetupError):
    code = "unknown_id"


class DownloadBusy(SetupError):
    code = "download_busy"


class NotEnoughSpace(SetupError):
    code = "insufficient_disk_space"


class FetcherUnavailable(SetupError):
    code = "fetcher_unavailable"


class VerificationFailed(SetupError):
    code = "verification_failed"


#: The stream stopped for a reason that is not the user's doing: a dead socket,
#: a refused connection, a 500 from the CDN. One class so the job's error code is
#: the same token whether the failure surfaces here or in the worker loop.
class DownloadFailed(SetupError):
    code = "download_failed"


class DownloadCancelled(SetupError):
    code = "cancelled"


# ---------------------------------------------------------------------------
# The configured layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetupRoots:
    """Where each family of files lives, derived from the launch flags.

    ``models_parent`` is the ``models/`` directory: the deployed layout is
    ``models/assets`` (base), ``models/turbo/assets``, ``models/pe/<ckpt>``,
    ``models/rewrite`` and ``models/q8`` side by side, which is exactly what
    ``start-web.sh`` passes. Nothing here is ever taken from a request.
    """

    models_dir: str | None
    manifest_path: str | None
    q8_manifest_path: str | None
    models_parent: str | None
    turbo_manifest_path: str | None
    turbo_models_dir: str | None
    pe_dir: str | None
    q8_dir: str | None
    rewrite_dir: str | None

    @classmethod
    def for_paths(cls, *, models_dir: str | None,
                  manifest_path: str | None,
                  q8_manifest_path: str | None = None) -> "SetupRoots":
        parent: str | None = None
        if models_dir:
            parent = os.path.dirname(os.path.abspath(models_dir))
        elif manifest_path:
            parent = os.path.dirname(os.path.abspath(manifest_path))
        if parent is None:
            return cls(None, None, None, None, None, None, None, None, None)
        joined = os.path.join
        return cls(models_dir=models_dir, manifest_path=manifest_path,
                   q8_manifest_path=q8_manifest_path, models_parent=parent,
                   turbo_manifest_path=joined(parent, "turbo", "manifest.json"),
                   turbo_models_dir=joined(parent, "turbo", "assets"),
                   pe_dir=joined(parent, "pe"),
                   q8_dir=parent,
                   rewrite_dir=joined(parent, "rewrite"))


def display_dir(path: str | None, anchor: str | None = None) -> str:
    """A display-only directory name ("models/assets") — never an absolute path.

    The anchor is the ``models/`` directory the server was pointed at, so the
    string is stable whatever the operator's home directory is: files inside
    that directory are shown as ``models/<wherever>``, and anything outside it
    by its last two components, which is still enough for a person to find it
    on their own disk without the page publishing a path.
    """
    if not path:
        return ""
    absolute = os.path.abspath(path)
    if anchor:
        base = os.path.abspath(anchor)
        name = os.path.basename(base) or "models"
        try:
            relative = os.path.relpath(absolute, base)
        except ValueError:  # pragma: no cover - different drive on Windows
            relative = ""
        if relative == ".":
            return name
        if relative and not relative.startswith(os.pardir):
            return (name + "/" + relative).replace(os.sep, "/")
    try:
        relative = os.path.relpath(absolute, str(REPO_ROOT))
    except ValueError:  # pragma: no cover - different drive on Windows
        relative = ""
    if relative and not relative.startswith(os.pardir):
        return relative.replace(os.sep, "/")
    return "/".join(Path(absolute).parts[-2:]) or Path(absolute).name


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetupFile:
    """One pinned file: the name in its source repo, the name it lands under,
    and the two numbers that prove it is the right one."""

    source: str
    dest: str
    bytes: int
    sha256: str
    repo: str
    revision: str
    role: str
    md5: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.dest, "bytes": self.bytes}


@dataclass(frozen=True)
class SetupItem:
    id: str
    label: str
    required: bool
    reason: str
    root: str
    dest: str
    files: tuple[SetupFile, ...]
    #: ``""`` or :data:`NEEDS_BINARY`: something a download cannot supply.
    extra: str = ""

    def paths(self) -> list[tuple[SetupFile, Path]]:
        return [(f, Path(self.root) / f.dest) for f in self.files] if self.root else []


def _manifest_files(manifest: _manifest.ModelManifest) -> tuple[SetupFile, ...]:
    return tuple(SetupFile(source=a.filename, dest=a.filename, bytes=a.bytes,
                           sha256=a.sha256, repo=a.repo, revision=a.revision,
                           role=a.role)
                 for a in manifest.artifacts)


def _load_manifest(path: str | None) -> _manifest.ModelManifest | None:
    if not path:
        return None
    try:
        return _manifest.load_manifest(path)
    except (_manifest.ManifestError, OSError, ValueError):
        # A broken manifest is reported by the existing status gate; here it
        # simply means "these files cannot be listed", never a crash.
        return None


def _q8_files(sidecar_path: str | None) -> tuple[SetupFile, ...]:
    """The Q8 GGUF, if the sidecar actually describes a fetchable file.

    ``models/q8_manifest.json`` is NOT a download manifest — it is the Q8
    artifact sidecar (``qwen_engine.q8_artifact.v1``). It is usable here only
    when it carries every field the fetcher needs (repo, 40-hex revision, safe
    filename, 64-hex sha256, positive byte size); anything else leaves the Q8
    item out of the catalogue entirely rather than offering a guess.
    """
    if not sidecar_path:
        return ()
    try:
        raw = Path(sidecar_path).read_bytes()
        document = json.loads(raw)
    except (OSError, ValueError):
        return ()
    if not isinstance(document, dict):
        return ()
    values = tuple(document.get(key) for key in
                   ("repo", "revision", "filename", "sha256", "bytes"))
    repo, revision, filename, sha256, size = values
    if not all(isinstance(v, str) and v for v in values[:4]):
        return ()
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        return ()
    try:
        artifact = _manifest.ModelArtifact(role="dit", repo=repo,
                                           revision=revision, filename=filename,
                                           sha256=sha256, bytes=size)
    except _manifest.ManifestError:
        return ()
    return (SetupFile(source=artifact.filename, dest=artifact.filename,
                      bytes=artifact.bytes, sha256=artifact.sha256,
                      repo=artifact.repo, revision=artifact.revision,
                      role="dit"),)


def catalogue(roots: SetupRoots) -> tuple[SetupItem, ...]:
    """Every item this server could install, in display order.

    Required first. An item is listed whenever its pinned source is knowable,
    even when its files are already on disk — the page has to be able to say
    "installed" rather than leave a gap the user reads as a missing feature.
    """
    items: list[SetupItem] = []
    parent = roots.models_parent
    base_manifest = _load_manifest(roots.manifest_path)
    if roots.models_dir and base_manifest:
        files = _manifest_files(base_manifest)
        items.append(SetupItem(
            id=BASE, label="Qwen-Image-2.1 (required)", required=True,
            reason="The image model itself: the transformer, the text encoder, "
                   "the VAE and the processor files. Nothing can be drawn "
                   "without them.",
            root=roots.models_dir, dest=display_dir(roots.models_dir, parent),
            files=files))
    turbo = _load_manifest(roots.turbo_manifest_path)
    if turbo and roots.turbo_models_dir:
        items.append(SetupItem(
            id=TURBO, label="Turbo transformer", required=False,
            reason="A second set of transformer weights for the eight-step "
                   "Turbo recipe: faster images from the same text encoder and "
                   "VAE you already downloaded.",
            root=roots.turbo_models_dir,
            dest=display_dir(roots.turbo_models_dir, parent),
            files=_manifest_files(turbo)))
    q8_files = _q8_files(roots.q8_manifest_path)
    if q8_files and parent:
        items.append(SetupItem(
            id=Q8, label="Q8 transformer", required=False,
            reason="A quantized transformer that fits more of the card, for "
                   "larger pictures at the cost of some detail.",
            root=roots.q8_dir or parent,
            dest=display_dir(roots.q8_dir or parent, parent),
            files=q8_files))
    pe = Path(roots.pe_dir) if roots.pe_dir else None
    if pe is not None:
        items.append(SetupItem(
            id=PE_TEXT, label="Official text prompt enhancer", required=False,
            reason="Turns a short written idea into a detailed prompt with the "
                   "official Q8 model instead of your own language service. "
                   "Needs a llama.cpp llama-server built for this card.",
            root=str(pe / "PE-T2I-Q8_0"),
            dest=display_dir(str(pe / "PE-T2I-Q8_0"), parent),
            files=(SetupFile(source=PE_T2I_FILE, dest=PE_T2I_FILE,
                             bytes=PE_T2I_BYTES, sha256=PE_T2I_SHA256,
                             repo=PE_T2I_REPO, revision=PE_T2I_REVISION,
                             role="pe"),),
            extra=NEEDS_BINARY))
        items.append(SetupItem(
            id=PE_EDIT, label="Official edit prompt enhancer", required=False,
            reason="The same idea for picture edits: it reads your reference "
                   "image and rewrites the edit instruction. Needs a llama.cpp "
                   "llama-server built for this card.",
            root=str(pe / "PE-I2I-Q8_0"),
            dest=display_dir(str(pe / "PE-I2I-Q8_0"), parent),
            files=(SetupFile(source=PE_I2I_FILE, dest=PE_I2I_FILE,
                             bytes=PE_I2I_BYTES, sha256=PE_I2I_SHA256,
                             repo=PE_I2I_REPO, revision=PE_I2I_REVISION,
                             role="pe"),
                   SetupFile(source=PE_I2I_MMPROJ_FILE, dest=PE_I2I_MMPROJ_FILE,
                             bytes=PE_I2I_MMPROJ_BYTES,
                             sha256=PE_I2I_MMPROJ_SHA256, repo=PE_I2I_REPO,
                             revision=PE_I2I_REVISION, role="pe"),),
            extra=NEEDS_BINARY))
    if roots.rewrite_dir:
        items.append(SetupItem(
            id=PROMPTS, label="Rewriter system prompts", required=False,
            reason="The instruction sheets the official enhancers were trained "
                   "with. Any prompt rewriter needs them; they carry a "
                   "research licence, so they are yours to fetch, not ours to "
                   "ship.",
            root=roots.rewrite_dir, dest=display_dir(roots.rewrite_dir, parent),
            files=tuple(
                SetupFile(source=PROMPT_SOURCE_FILE, dest=PROMPT_DEST[task],
                          bytes=PROMPT_BYTES[task], sha256=PROMPT_SHA256[task],
                          repo=PROMPT_T2I_REPO if task == "t2i" else PROMPT_I2I_REPO,
                          revision=(PROMPT_T2I_REVISION if task == "t2i"
                                    else PROMPT_I2I_REVISION),
                          role="prompt",
                          md5=_rewrite.SYSTEM_PROMPT_MD5[task])
                for task in ("t2i", "edit"))))
    return tuple(items)


def catalogue_notices(roots: SetupRoots) -> list[str]:
    """Why an item cannot be offered here, in words. Empty on a normal checkout.

    These are packaging problems, not "you have not downloaded it yet": the
    pinned list travels with the code, so a manifest the page cannot read means
    the checkout is incomplete. The console prints them and the Settings page
    shows a notice, because a silently short list would look like the server
    simply does not offer Turbo.
    """
    notices: list[str] = []
    if not roots.models_dir or not roots.manifest_path:
        notices.append("the base asset manifest or its folder is not configured")
    elif not os.path.isfile(roots.manifest_path) or (
            _load_manifest(roots.manifest_path) is None):
        notices.append("the pinned base manifest cannot be read, so the "
                       "required image model cannot be listed")
    if (not roots.turbo_manifest_path
            or not os.path.isfile(roots.turbo_manifest_path)
            or _load_manifest(roots.turbo_manifest_path) is None):
        notices.append("the Turbo manifest cannot be read, so the faster "
                       "Turbo weights are not offered")
    q8_path = roots.q8_manifest_path or (
        Path(roots.models_parent) / "q8_manifest.json" if roots.models_parent else None)
    if q8_path is None or not os.path.isfile(q8_path) or not _q8_files(str(q8_path)):
        notices.append("the quantized transformer manifest cannot be read, so "
                       "that option is not offered")
    return notices


# ---------------------------------------------------------------------------
# What is on disk (stat only — never a hash)
# ---------------------------------------------------------------------------


INSTALLED = "installed"
PARTIAL = "partial"
MISSING = "missing"

#: The three words the Settings page renders; anything else is a bug.
INSTALL_STATES = (INSTALLED, PARTIAL, MISSING)


def scan(item: SetupItem) -> tuple[str, list[str], int]:
    """``(status, missing_dest_names, bytes_still_needed)`` for one item.

    A file is counted present only when it exists, is a regular file and is
    exactly the pinned size. That is deliberately weaker than verification —
    verification costs a full read of every weight and belongs to a job — and
    deliberately stronger than "the directory is not empty".
    """
    if not item.files:
        # Nothing pinned, nothing installed: an item built from a manifest that
        # could not be read must never be reported as satisfied.
        return MISSING, [], 0
    missing: list[str] = []
    needed = 0
    for setup_file, path in item.paths():
        try:
            info = os.stat(path, follow_symlinks=False)
        except OSError:
            missing.append(setup_file.dest)
            needed += setup_file.bytes
            continue
        if not _is_regular(info) or info.st_size != setup_file.bytes:
            missing.append(setup_file.dest)
            needed += setup_file.bytes
    if not missing:
        return INSTALLED, [], 0
    if len(missing) == len(item.files):
        return MISSING, missing, needed
    return PARTIAL, missing, needed


def _is_regular(info: os.stat_result) -> bool:
    return _stat.S_ISREG(info.st_mode) and info.st_nlink == 1


def free_bytes(path: str | None) -> int | None:
    """Bytes available to this user at ``path``, or ``None`` if unknowable."""
    probe = path
    if probe and not os.path.isdir(probe):
        probe = os.path.dirname(os.path.abspath(probe)) or None
    if not probe:
        return None
    try:
        info = os.statvfs(probe)
    except OSError:
        return None
    return info.f_bavail * info.f_frsize


# ---------------------------------------------------------------------------
# The download worker
# ---------------------------------------------------------------------------

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"
JOB_STATES = (STATE_IDLE, STATE_RUNNING, STATE_COMPLETED, STATE_FAILED,
              STATE_CANCELLED)

#: Where the hardened transfer lives. It is a repository tool, not an installed
#: module, so it is loaded by path the same way ``tests/test_fetch_tool.py``
#: loads it — and only when a download is actually asked for.
FETCH_TOOL = REPO_ROOT / "tools" / "fetch_models.py"

_fetch_module: Any = None
_fetch_lock = threading.Lock()


def load_fetch_tool() -> Any:
    """The ``tools/fetch_models.py`` module, loaded once, lazily.

    Raises :class:`FetcherUnavailable` (a typed refusal the page can show) when
    the tool is not there, instead of failing the whole server at import.
    """
    global _fetch_module
    with _fetch_lock:
        if _fetch_module is not None:
            return _fetch_module
        if not FETCH_TOOL.is_file():
            raise FetcherUnavailable(
                "the fetch tool is not available in this install; run "
                "python tools/fetch_models.py on the server instead")
        spec = importlib.util.spec_from_file_location("qwen_fetch_models",
                                                      str(FETCH_TOOL))
        if spec is None or spec.loader is None:
            raise FetcherUnavailable("the fetch tool could not be loaded")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            raise FetcherUnavailable(
                f"the fetch tool could not be loaded: {type(exc).__name__}") from None
        _fetch_module = module
        return module


class _ProgressStream:
    """Wraps the fetcher's stream to count bytes and to notice a cancel.

    The fetcher keeps the partial file on any interruption, so a cancelled
    download resumes where it stopped instead of starting the gigabytes again.
    """

    def __init__(self, inner: Any, on_bytes: Callable[[int], None],
                 cancelled: Callable[[], bool]) -> None:
        self._inner = inner
        self._on_bytes = on_bytes
        self._cancelled = cancelled
        self.status = getattr(inner, "status", 200)
        self.headers = getattr(inner, "headers", {})

    def read(self, n: int) -> bytes:
        if self._cancelled():
            raise DownloadCancelled("the download was stopped")
        chunk = self._inner.read(n)
        if chunk:
            self._on_bytes(len(chunk))
        return chunk

    def close(self) -> None:
        try:
            self._inner.close()
        except Exception:  # pragma: no cover - best effort on a dead socket
            pass


class _ProgressTransport:
    """The fetcher's transport, with the stream wrapped per request."""

    def __init__(self, inner: Any, on_bytes: Callable[[int], None],
                 cancelled: Callable[[], bool]) -> None:
        self._inner = inner
        self._on_bytes = on_bytes
        self._cancelled = cancelled

    def get(self, url: str, start: int) -> Any:
        return _ProgressStream(self._inner.get(url, start), self._on_bytes,
                               self._cancelled)


def verify_md5(path: Path, expected: str) -> bool:
    digest = hashlib.md5(usedforsecurity=False)
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == expected


class ModelSetupService:
    """The catalogue, the free-space check and the one download worker.

    Everything the network-touching part does is behind ``fetch`` so the guarded
    suite can drive a whole job with fake bytes and no sockets. The default
    ``fetch`` reuses :func:`tools.fetch_models._download`, which is where the
    resume, the SHA256 check and the atomic publish live.
    """

    def __init__(self, roots: SetupRoots, *,
                 fetch: Callable[[SetupFile, Path, Callable[[int], None]],
                                 None] | None = None,
                 disk_free: Callable[[str | None], int | None] | None = None,
                 items: tuple[SetupItem, ...] | None = None) -> None:
        self.roots = roots
        self.items = catalogue(roots) if items is None else items
        self._fetch = fetch
        self._disk_free = disk_free or free_bytes
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._job: dict[str, Any] | None = None

    @classmethod
    def for_paths(cls, *, models_dir: str | None, manifest_path: str | None,
                  q8_manifest_path: str | None = None, **kwargs: Any) -> "ModelSetupService":
        """The normal construction: the two paths the server was given, and
        everything else derived from their parent (see
        :meth:`SetupRoots.for_paths`)."""
        return cls(SetupRoots.for_paths(models_dir=models_dir,
                                       manifest_path=manifest_path,
                                       q8_manifest_path=q8_manifest_path),
                   **kwargs)

    # -- reads ---------------------------------------------------------------

    def notices(self) -> list[str]:
        """Packaging problems that make part of the catalogue unavailable."""
        return catalogue_notices(self.roots)

    def required_missing(self) -> bool:
        """True while any required item is not fully on disk.

        A required id that is not in the catalogue at all counts as missing: an
        unreadable base manifest is a worse state than a download that has not
        run, and a page saying "nothing required is missing" while listing no
        base files would be the more misleading of the two.
        """
        listed = {item.id for item in self.items}
        if any(item_id not in listed for item_id in REQUIRED_IDS):
            return True
        for item in self.items:
            if item.required and scan(item)[0] != INSTALLED:
                return True
        return False

    def state(self) -> dict[str, Any]:
        """The ``GET /api/v1/models`` document body (minus the loopback flag,
        which only the transport knows)."""
        catalogue_rows: list[dict[str, Any]] = []
        total_needed = 0
        for item in self.items:
            status, missing, needed = scan(item)
            total_needed += needed
            catalogue_rows.append({
                "id": item.id, "label": item.label,
                "required": item.required, "reason": item.reason,
                "files": [f.as_dict() for f in item.files],
                "dest": item.dest, "installed": status,
                "missing_files": missing, "bytes_needed": needed,
                "extra": item.extra,
            })
        with self._lock:
            job = None if self._job is None else json.loads(
                json.dumps(self._job))
        return {"schema": MODELS_SCHEMA,
                "items": catalogue_rows,
                "bytes_needed": total_needed,
                "free_bytes": self._disk_free(self.roots.models_parent),
                "required_missing": self.required_missing(),
                "job": job}

    def job_state(self) -> dict[str, Any]:
        with self._lock:
            if self._job is None:
                return {"schema": JOB_SCHEMA, "state": STATE_IDLE,
                        "ids": [], "files": [], "bytes_done": 0,
                        "bytes_total": 0, "error": None}
            return json.loads(json.dumps(self._job))

    # -- the download --------------------------------------------------------

    def start(self, ids: list[str]) -> dict[str, Any]:
        """Begin one download job. Only ids are honoured; required items are
        always added, and nothing is fetched when any part of the request is
        refused — including a request whose ids are fine but whose server has
        no fetch tool: that is refused here, not after a thread has started."""
        wanted = list(dict.fromkeys(ids))
        for item_id in wanted:
            if item_id not in ITEM_IDS:
                raise UnknownItem(f"unknown model id {item_id!r}")
        for required in REQUIRED_IDS:
            if required not in wanted:
                wanted.append(required)
        chosen = [item for item in self.items if item.id in wanted]
        missing_ids = [item_id for item_id in wanted
                       if not any(i.id == item_id for i in self.items)]
        if missing_ids:
            raise UnknownItem("this server cannot offer: "
                              + ", ".join(sorted(missing_ids)))
        pending = [(item, f, path) for item in chosen
                   for f, path in self._pending(item)]
        if pending and self._fetch is None:
            # Refuse BEFORE a thread exists: a machine whose fetch tool cannot
            # be loaded must say so to the person who pressed the button, not
            # start a job that fails one byte in.
            load_fetch_tool()
        needed = sum(f.bytes for _item, f, _path in pending)
        needed_headroom = needed + DISK_HEADROOM_BYTES * len(pending)
        free = self._disk_free(self.roots.models_parent)
        if free is not None and pending and free < needed_headroom:
            raise NotEnoughSpace(
                f"this download needs {needed_headroom} bytes and the disk has "
                f"{free} free")
        with self._lock:
            if self._job is not None and self._job["state"] == STATE_RUNNING:
                raise DownloadBusy("a download is already running")
            self._cancel.clear()
            self._job = {
                "schema": JOB_SCHEMA, "state": STATE_RUNNING,
                "ids": [item.id for item in chosen],
                "files": [{"item": item.id, "name": f.dest, "done": 0,
                           "total": f.bytes} for item, f, _ in pending],
                "bytes_done": 0, "bytes_total": needed, "error": None,
            }
        self._worker = threading.Thread(target=self._run,
                                        args=(pending,), name="model-setup",
                                        daemon=True)
        self._worker.start()
        return self.job_state()

    def _pending(self, item: SetupItem) -> list[tuple[SetupFile, Path]]:
        present = set(scan(item)[1])
        return [(f, path) for f, path in item.paths() if f.dest in present]

    def _run(self, pending: list[tuple[SetupItem, SetupFile, Path]]) -> None:
        try:
            for item, setup_file, path in pending:
                if self._cancel.is_set():
                    raise DownloadCancelled("the download was stopped")
                try:
                    os.makedirs(path.parent, mode=0o755, exist_ok=True)
                except OSError as exc:
                    self._finish(STATE_FAILED, "write_refused",
                                 f"{setup_file.dest}: the destination folder "
                                 f"could not be created ({type(exc).__name__})")
                    return
                try:
                    self._fetch_one(setup_file, path)
                except DownloadCancelled:
                    raise
                except SetupError as exc:
                    self._finish(STATE_FAILED, exc.code,
                                 f"{setup_file.dest}: {exc.message}")
                    return
                except Exception as exc:
                    # The fetcher's own message may name a URL or a local path;
                    # the browser gets the type, the console gets the line.
                    print(f"model download failed for {setup_file.dest}: {exc}",
                          file=sys.stderr, flush=True)
                    self._finish(STATE_FAILED, "download_failed",
                                 f"{setup_file.dest}: the download failed "
                                 f"({type(exc).__name__})")
                    return
                if setup_file.md5 is not None and not verify_md5(
                        path, setup_file.md5):
                    self._remove(path)
                    self._finish(STATE_FAILED, VerificationFailed.code,
                                 f"{setup_file.dest}: the file arrived but it "
                                 "is not the prompt the rewriter was built for; "
                                 "it was deleted")
                    return
            self._finish(STATE_COMPLETED, "", "")
        except DownloadCancelled:
            self._finish(STATE_CANCELLED, "", "")
        except Exception as exc:  # pragma: no cover - a bug, not a user error
            self._finish(STATE_FAILED, "download_failed",
                         f"the download stopped unexpectedly ({type(exc).__name__})")

    def _tool_fetch(self, setup_file: SetupFile, dest: Path,
                    on_bytes: Callable[[int], None]) -> None:
        """The real transfer: :func:`tools.fetch_models._download` with the byte
        counter and the cancel flag wrapped around its stream. Resume, the
        SHA256 check and the atomic no-clobber publish all live in there.

        The fetcher's own exceptions are translated into the typed setup errors
        so a verification failure reads as ``verification_failed`` on the page
        rather than as a generic download error. Nothing here publishes a file:
        an integrity failure leaves the verified bytes unpublished, and the
        fetcher removes the unrecoverable partial itself.
        """
        tool = load_fetch_tool()
        transport = _ProgressTransport(tool.UrllibTransport(), on_bytes,
                                       self._cancel.is_set)
        artifact = _manifest.ModelArtifact(
            role=setup_file.role, repo=setup_file.repo,
            revision=setup_file.revision, filename=setup_file.source,
            sha256=setup_file.sha256, bytes=setup_file.bytes)
        try:
            tool._download(artifact, dest, transport, tool._default_disk_free)
        except tool.IntegrityError as exc:
            raise VerificationFailed(
                "the bytes that arrived are not the ones this build pins") from exc
        except tool.DiskSpaceError as exc:
            raise NotEnoughSpace(
                "the disk filled up before this file could be written") from exc
        except tool.NetworkError as exc:
            raise DownloadFailed("the download could not be completed") from exc
        except tool.FetchError as exc:
            raise DownloadFailed(
                f"the download was refused ({type(exc).__name__})") from exc

    def _fetch_one(self, setup_file: SetupFile, path: Path) -> None:
        def on_bytes(count: int) -> None:
            with self._lock:
                if self._job is None:
                    return
                for row in self._job["files"]:
                    if row["name"] == setup_file.dest:
                        row["done"] = min(row["total"], row["done"] + count)
                        break
                self._job["bytes_done"] = min(
                    self._job["bytes_total"],
                    self._job["bytes_done"] + count)

        fetch = self._fetch or self._tool_fetch
        try:
            fetch(setup_file, path, on_bytes)
        except SetupError:
            raise
        except Exception as exc:
            if self._cancel.is_set():
                raise DownloadCancelled("the download was stopped") from None
            raise DownloadFailed(f"the download failed ({type(exc).__name__})") from None
        if self._cancel.is_set():
            raise DownloadCancelled("the download was stopped")

    def _finish(self, state: str, code: str, message: str) -> None:
        with self._lock:
            if self._job is None:
                return
            self._job["state"] = state
            self._job["error"] = None if not code else {"code": code,
                                                        "message": message}

    def _remove(self, path: Path) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    def cancel(self) -> dict[str, Any]:
        """Ask the running job to stop; a second press, or one while idle, is a
        no-op. The partial file stays on purpose — that is the resume.

        The worker notices within one stream chunk and publishes nothing, so the
        answer is the job as it is right now, still ``running``, and the next poll
        reports ``cancelled``.
        """
        self._cancel.set()
        return self.job_state()

    def close(self, join_timeout: float = 5.0) -> None:
        """Stop the worker on shutdown: ask, then wait a bounded time."""
        self._cancel.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=join_timeout)


def client_is_loopback(address: Any) -> bool:
    """True for ``127.0.0.1``/``::1`` — the only clients that may write.

    ``address`` is a socket peer tuple; anything unparseable is NOT loopback, so
    an unexpected form refuses rather than granting a download.
    """
    try:
        host = address[0]
    except (TypeError, IndexError, KeyError):
        return False
    if not isinstance(host, str):
        return False
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False
