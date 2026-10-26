"""Rewrite service: one in-flight rewrite, behind a typed refusal.

Two backends share one slot:

* ``pe-llamacpp`` — **the official prompt enhancer** (``Qwen/Qwen-Image-2.1-PE-T2I``
  as a Q8_0 GGUF under llama.cpp, :mod:`qwen_engine.pe_llamacpp`). The standard
  rewriter: best results, t2i only, and it owns the V100 for the duration of a
  rewrite.
* ``local-llm`` — the **local LLM service** (an OpenAI-compatible endpoint on
  this machine, :class:`qwen_engine.rewrite_client.RewriteClient`). Never
  touches the GPU, and the only backend that can rewrite an *edit* draft.
* ``off`` — not a backend: the rewriter is switched off.

Which one runs is **server-side, saved state** (Settings → Prompt rewriter),
not a per-request choice: the settings apply to every browser using this
server. What is *possible* is decided by the launch flags — the llama-server
binary, the GGUF, the digest-pinned system prompts and the PE timeout stay
operator-owned and are never accepted from the browser.

**Entry-point contract** (what ``web/server.py`` and ``web/api.py`` call)::

    settings = RewriteSettings(backend="local-llm", endpoint=..., model=...,
                               prompts={"t2i": ..., "edit": ...}, timeout=180.0)
    pe = PESettings(llama_server=..., model_path=..., system_prompt=...,
                    timeout=300.0, log_path=...)
    service = RewriteService(settings, pe=pe, gpu_lock=lock, cache_dir=cache)
    prefs, warning = load_preferences(preferences_path(cache))
    service.apply_preferences(prefs)          # the saved file wins over flags

``settings = RewriteSettings.disabled()`` is "no local-llm configuration";
``pe=None`` is "no enhancer". A service with neither reports the canonical
disabled block.

*Per request* (body already parsed to a mapping by ``api.build_rewrite_request``;
this module re-validates, cheaply and defensively)::

    request = {"task": "t2i", "prompt": "<draft>", "draft_revision": 7,
               "image_ids": []}
    response = service.submit(request)   # -> ApiResponse(200, payload)

Body keys are exactly ``task``/``prompt``/``draft_revision``/``image_ids``; any
other key is refused rather than ignored. ``draft_revision`` is echoed verbatim
so the UI can discard a stale suggestion. The browser does **not** pick a
backend: the active choice does, and an edit draft falls back to the local
service when the enhancer is active but a local service is configured
(:meth:`RewriteService._pick_client`), with the response naming the backend
that actually answered.

``image_ids`` is the ORDERED list of reference-store ids the rewrite is about:
``[]`` for a t2i rewrite, and at least one for an ``edit`` rewrite. The edit
system prompt addresses references as ``<imageN>``, so the order is the
conditioning order and reordering changes the request's meaning. Each id is
read from the server reference store, decoded through the store's own gates,
alpha-flattened onto white, downscaled so no side exceeds
``REWRITE_IMAGE_MAX_DIM``, and sent as a PNG data URI — the backend is never
given a path or a URL. Forcing this work keeps a t2i draft from carrying
pictures it has no ``<imageN>`` to name (``422 task_unsupported``), and an edit
rewrite with no pictures from silently becoming a text-only rewrite
(``422 reference_required``).

Response payload (``WEB_REWRITE_SCHEMA``): ``schema``, ``task``,
``draft_revision``, ``original_prompt`` (exactly what was sent, bounded by the
cap, so the UI can prove staleness), ``rewritten_prompt``, ``parse_ok``,
``unparsed_answer`` (only when ``parse_ok`` is false), ``ratio``,
``ratio_conflict``, ``backend`` (the one that answered), ``elapsed_ms``. The
server never maps a ratio to width/height — ``PRESETS`` in ``app.js`` is the
single source of truth.

*Typed failures* — :class:`~qwen_engine.web.api.ApiError`, messages bounded and
**never containing prompt text, a path, a port or a URL**::

    503 rewrite_disabled         no backend configured, or the choice is off
    503 rewrite_busy             another rewrite is in flight (never a queue)
    503 rewrite_gpu_busy         an image job holds the GPU; the enhancer
                                 cannot share it, and it does not wait
    504 rewrite_timeout          the deadline expired; the draft is untouched
    502 rewrite_unavailable      transport failure / non-2xx from the backend
    502 rewrite_invalid_answer   transport violation (not a completion)
    422 prompt_too_long          over ``PROMPT_MAX_CHARS``
    422 task_unsupported         unknown task, a t2i rewrite with images, or an
                                 edit draft with no backend that supports edit
    422 reference_required       an edit rewrite with no reference images
    422 too_many_references      more ids than the edit-job cap (``REF_MAX``)
    404 reference_unknown        an id is unknown or its blob is unreadable
    400 invalid_envelope         malformed body: unknown/missing key, bad types
    503 rewrite_cancelled        cancelled by shutdown or operator; see below

*The GPU lock.* One :class:`threading.Lock` is created by the server and shared
with ``WebGeneration``, which takes it around the executor call. The two
orders are deliberately different: an image job that holds it makes a rewrite
refuse **at once** (503 ``rewrite_gpu_busy`` — a rewrite that quietly waited
would be reported as slow when it is really queued), while an image job
submitted during a rewrite simply waits for it (tens of seconds, which the
user has already accepted by pressing Generate). The enhancer terminates and
reaps its llama-server child *before* releasing the lock, so nothing meets a
live 9.5 GiB process on the way in.

*Cancellation is cooperative, not an abort.* :meth:`RewriteService.cancel`
prevents a pending answer being applied and releases the waiter immediately.
For the local service it cannot interrupt a blocked socket read (the client's
own timeout bounds wall time); for the enhancer a watchdog polls the flag while
the model generates and terminates the child within one poll interval, so the
V100 is free again at once — and since that termination is what ends the call,
the caller is told it was cancelled, never that it timed out.

*Single flight, no queue.* One slot across BOTH backends: a second caller is
refused at once rather than waiting behind the first or transparently
retrying. The call runs on a worker thread so the HTTP handlers are released.
This service never touches ``JobController`` or the executor lease — a rewrite
must not queue behind, or stall, an image job.

*No prompt in logs or errors.* Nothing here logs. :meth:`RewriteService.event`
returns the only fields the caller may log — ``task``, ``backend``,
``elapsed_ms``, ``prompt_chars``, ``parse_ok``, ``outcome`` — deliberately no
prompt, no endpoint, no path, and no reference id.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import qwen_engine.pe_llamacpp as pe_client
from qwen_engine import reference_store as _reference_store
from qwen_engine import rewrite as rw
from qwen_engine import rewrite_client as rc
from qwen_engine.web import api as web_api
from qwen_engine.web.api import ApiError, ApiResponse, _safe

#: Re-exported so a caller that already imports the service never needs a
#: second import for the wire tags. The DEFINITION lives in ``web/api.py``
#: (the transport's contract boundary), so the two can never drift apart.
REWRITE_REQUEST_SCHEMA = web_api.REWRITE_REQUEST_SCHEMA
WEB_REWRITE_SCHEMA = web_api.WEB_REWRITE_SCHEMA
REWRITE_SETTINGS_SCHEMA = web_api.REWRITE_SETTINGS_SCHEMA
REWRITE_TEST_REQUEST_SCHEMA = web_api.REWRITE_TEST_REQUEST_SCHEMA
REWRITE_TEST_SCHEMA = web_api.REWRITE_TEST_SCHEMA

#: Route body cap, re-exported under the name the service contract used.
#: Derived from ``PROMPT_MAX_CHARS`` in one place only (``web/api.py``).
MAX_REQUEST_BYTES = web_api.MAX_REWRITE_REQUEST_BYTES

#: The backends that can be *configured*. ``off`` is not in here: it is the
#: absence of a backend, and it is reachable from Settings but never as a
#: launch-flag backend id that builds a client.
BACKENDS = ("pe-llamacpp", "local-llm")

#: Everything the Settings page may choose, including switching off.
SETTINGS_BACKENDS = BACKENDS + ("off",)

#: The friendly names the UI shows. A label is display text only: nothing here
#: is a path, an address or a model identifier.
BACKEND_LABELS: Mapping[str, str] = {
    "pe-llamacpp": "Official enhancer (Q8, uses GPU)",
    "local-llm": "Local LLM service",
}

#: What each backend can rewrite. The enhancer checkpoints are task-specific
#: and only the text-to-image one is installed; the local service handles both.
BACKEND_TASKS: Mapping[str, tuple[str, ...]] = {
    "pe-llamacpp": pe_client.SUPPORTED_TASKS,
    "local-llm": rw.TASKS,
}

#: Whether using this backend takes the V100. The UI says so plainly, because
#: "uses GPU" means "waits for, or blocks, image jobs".
BACKEND_USES_GPU: Mapping[str, bool] = {"pe-llamacpp": True, "local-llm": False}

#: Default deadline for the local service. Worst measured round trip against
#: the shared service was 38 s; a slower backend needs this raised explicitly,
#: as a flag decision.
DEFAULT_TIMEOUT_SECONDS = rc.DEFAULT_TIMEOUT_SECONDS

#: Default deadline for one enhancer rewrite: Root measured 14-37 s at about
#: 70 tok/s, and the thinking block may legitimately reach its token budget.
DEFAULT_PE_TIMEOUT_SECONDS = pe_client.DEFAULT_TIMEOUT_SECONDS

#: The name of the saved Settings document, inside the server's cache dir.
SETTINGS_FILENAME = "rewrite-settings.json"

#: The largest document we will even try to parse. The four fields cannot make
#: a legal file anywhere near this, so anything bigger is a hand-edited mess or
#: someone's attempt to make the server read a megabyte of JSON at start-up.
MAX_SETTINGS_BYTES = 16384

#: The only key the enhancer's log file may be written under.
PE_LOG_FILENAME = "pe-llamacpp.log"

#: Longest side of a reference image as sent to the rewriter, in pixels.
#: The rewriter only has to SEE the references to describe them, and a
#: downscaled copy keeps the request small enough for a local service; 768 was
#: chosen to stay above the 512 px Root verified landing in 0.6 s while keeping
#: fine text legible. The ORIGINAL is never sent: only these copies leave the
#: server, as data URIs, so the backend can never fetch a URL or read a path.
REWRITE_IMAGE_MAX_DIM = 768

#: PNG, because it is lossless for the flat art and text the rewriter reasons
#: about and needs no quality knob.
_IMAGE_URI_PREFIX = "data:image/png;base64,"

#: The Test-connection call is a probe, not a rewrite: a tiny completion, so a
#: wrong endpoint is reported in seconds rather than after the user's timeout.
TEST_MAX_TOKENS = 16
TEST_TIMEOUT_CAP = 30.0
_TEST_MESSAGES = ({"role": "system", "content": "Reply with the single word OK."},
                  {"role": "user", "content": "connection test"})


class _Cancelled(Exception):
    """Internal: the answer arrived but was cancelled before it could be used."""


# ---------------------------------------------------------------------------
# configuration: what the launch flags make possible
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewriteSettings:
    """Startup-owned local-service configuration; validated once, never per
    request. ``backend="off"`` means "this server has no local service"."""

    backend: str
    endpoint: str | None = None
    model: str | None = None
    prompts: Mapping[str, str] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    #: The server-owned reference store, so an edit rewrite can read the
    #: user's reference images. ``None`` means the store is not configured, and
    #: an edit rewrite is then refused rather than sent without its pictures.
    references_dir: str | None = None

    @classmethod
    def disabled(cls) -> "RewriteSettings":
        return cls(backend="off")

    @property
    def local_complete(self) -> bool:
        """Everything the local service needs is here — and ``backend`` is set
        to ``"local-llm"`` by :func:`build_rewrite_service` exactly when it is,
        so this is one fact stated once rather than re-derived per caller.
        """
        return self.backend == "local-llm"

    def __post_init__(self) -> None:
        if self.backend != "off" and self.backend not in BACKENDS:
            raise rc.RewriteConfigError(
                f"unknown rewrite backend {self.backend!r}; choose 'off' or "
                f"one of {list(BACKENDS)}")
        if self.backend == "off":
            if self.endpoint or self.model or self.prompts:
                raise rc.RewriteConfigError(
                    "a disabled rewriter must not carry an endpoint, model or "
                    "system prompt: an operator who set those meant to enable it")
            return
        if self.references_dir is not None:
            if not isinstance(self.references_dir, str) or not self.references_dir.strip():
                raise rc.RewriteConfigError(
                    "rewrite references_dir must be a non-empty path or None")
        # Validate eagerly: a bad endpoint is a startup refusal, not a 503.
        rc.parse_endpoint(self.endpoint)
        if not isinstance(self.model, str) or not self.model.strip():
            raise rc.RewriteConfigError("rewrite model must be a non-empty string")
        missing = [task for task in rw.TASKS
                   if not isinstance(self.prompts.get(task), str)
                   or not self.prompts[task].strip()]
        if missing:
            raise rc.RewriteConfigError(
                f"system prompt missing for {missing}; every task needs the "
                "prompt the weights were trained on")
        if (isinstance(self.timeout, bool)
                or not isinstance(self.timeout, (int, float))
                or not 0 < float(self.timeout) <= 3600):
            raise rc.RewriteConfigError(
                f"rewrite timeout must be in (0, 3600] seconds, got {self.timeout!r}")

    def build_client(self,
                     transport: Callable[[bytes], bytes] | None = None,
                     *, endpoint: str | None = None,
                     model: str | None = None,
                     timeout: float | None = None) -> rc.RewriteClient:
        """The client for this configuration, with the saved Settings values
        applied over the flag defaults. `transport` is for tests only."""
        return rc.RewriteClient(endpoint=endpoint or self.endpoint,
                                model=model or self.model,
                                timeout=timeout or self.timeout,
                                transport=transport)


@dataclass(frozen=True)
class PESettings:
    """Startup-owned enhancer configuration: operator files plus a budget.

    Every field here is a launch flag. None of it is ever accepted from the
    browser: a binary path, a GGUF path and a system prompt are the three
    things this project refuses to take over HTTP.
    """

    llama_server: str
    model_path: str
    system_prompt: str
    timeout: float = DEFAULT_PE_TIMEOUT_SECONDS
    #: Where the child's bounded log goes: one file under the server's cache
    #: dir, replaced by each attempt.
    log_path: str | None = None

    def __post_init__(self) -> None:
        for name, value, what in (("llama_server", self.llama_server,
                                   "llama-server binary"),
                                  ("model_path", self.model_path,
                                   "enhancer model file")):
            del name
            if not isinstance(value, str) or not value.strip():
                raise rc.RewriteConfigError(f"{what} must be a non-empty path")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise rc.RewriteConfigError(
                "the enhancer needs the digest-pinned text-to-image system "
                "prompt it was trained on")
        if (isinstance(self.timeout, bool)
                or not isinstance(self.timeout, (int, float))
                or not 0 < float(self.timeout) <= 3600):
            raise rc.RewriteConfigError(
                f"enhancer timeout must be in (0, 3600] seconds, got {self.timeout!r}")
        if self.log_path is not None and (not isinstance(self.log_path, str)
                                          or not self.log_path.strip()):
            raise rc.RewriteConfigError("the enhancer log path must be a "
                                        "non-empty path or None")

    def build_client(self, *, gpu_lock: threading.Lock | None = None,
                     cancel: Callable[[], bool] | None = None,
                     **seams: Any) -> pe_client.PELlamaCppClient:
        """The enhancer client. ``cancel`` is the service's single-slot
        cancel probe (see :meth:`RewriteService._current_cancelled`)."""
        if self.log_path is None:
            raise rc.RewriteConfigError(
                "the enhancer needs a log path inside the server cache dir")
        return pe_client.PELlamaCppClient(
            server_binary=self.llama_server, model_path=self.model_path,
            log_path=self.log_path, gpu_lock=gpu_lock,
            timeout=self.timeout, cancel=cancel, **seams)


# ---------------------------------------------------------------------------
# saved preferences: what the user chose in Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewritePreferences:
    """The saved choice. ``None`` on a field means "not saved: use the flag".

    Only four things are ever saved, and none of them is a path: which backend
    is active, and the local service's endpoint, model and timeout.
    """

    backend: str | None = None
    endpoint: str | None = None
    model: str | None = None
    timeout_s: float | None = None

    #: The keys of the saved document, in its canonical order.
    KEYS = ("backend", "endpoint", "model", "timeout_s")

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.KEYS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RewritePreferences":
        """Validate and build, as a SAVED document: a null backend means "not
        chosen yet, use the start-up default", which a POST may not carry."""
        checked = web_api.validate_rewrite_settings(data, allow_unset_backend=True)
        return cls(backend=checked["backend"], endpoint=checked["endpoint"],
                   model=checked["model"], timeout_s=checked["timeout_s"])


def preferences_path(cache_dir: str | None) -> str | None:
    """Where the saved Settings live, or ``None`` when the server has no
    writable cache dir (then settings are process-lifetime only)."""
    if not cache_dir:
        return None
    return os.path.join(cache_dir, SETTINGS_FILENAME)


def load_preferences(path: str | None) -> tuple[RewritePreferences | None, str]:
    """Read the saved choice. Never raises: a bad file is a bounded warning.

    Returns ``(preferences or None, warning)``; the warning is one line for
    stderr, empty when all is well. A corrupt or invalid document must not
    take the rewriter down — the flag defaults are used and the operator is
    told once, at startup.
    """
    if not path:
        return None, ""
    try:
        with open(path, "rb") as handle:
            # Read PAST the cap so an over-long file is recognised as one
            # rather than truncated into a misleading parse error.
            raw = handle.read(MAX_SETTINGS_BYTES + 1)
    except FileNotFoundError:
        return None, ""
    except OSError as exc:
        return None, (f"saved rewrite settings are unreadable "
                      f"({exc.strerror or type(exc).__name__}); using the "
                      "start-up defaults")
    if len(raw) > MAX_SETTINGS_BYTES:
        return None, (f"saved rewrite settings are longer than "
                      f"{MAX_SETTINGS_BYTES} bytes; using the start-up defaults")
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, ("saved rewrite settings are not valid JSON; using the "
                      "start-up defaults")
    try:
        return RewritePreferences.from_dict(document), ""
    except (ApiError, ValueError) as exc:
        return None, (f"saved rewrite settings are invalid "
                      f"({_safe(exc.message if isinstance(exc, ApiError) else str(exc))}); "
                      "using the start-up defaults")


def save_preferences(path: str | None, prefs: RewritePreferences) -> None:
    """Write the choice atomically, mode 0600, inside the server cache dir.

    Temp file in the SAME directory then :func:`os.replace`: a reader sees the
    old document or the new one, never a half-written file. Raises ``OSError``
    (the caller turns it into a typed refusal) — a settings change that cannot
    be persisted is reported, not silently accepted.
    """
    if not path:
        raise OSError("the server has no cache directory to save settings in")
    payload = json.dumps(prefs.as_dict(), sort_keys=True,
                         ensure_ascii=False).encode("utf-8")
    directory = os.path.dirname(path) or "."
    descriptor, temp_path = tempfile.mkstemp(prefix=".rewrite-settings-",
                                            suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# reference images (edit rewrites)
# ---------------------------------------------------------------------------


class _Job:
    """One in-flight rewrite's hand-off slot, guarded by the service Condition."""

    __slots__ = ("cancel", "finished", "payload", "error", "elapsed_ms",
                 "client", "backend", "timeout")

    def __init__(self, client: Any, backend: str, timeout: float) -> None:
        # Per-job, not per-service: a cancel aimed at one rewrite must never
        # poison the next one.
        self.cancel = threading.Event()
        self.finished = False
        self.payload: dict[str, Any] | None = None
        self.error: BaseException | None = None
        self.elapsed_ms: int | None = None
        # Bound at claim time, so a settings change cannot swap the client
        # under a rewrite that is already running: it finishes on the client
        # it started with.
        self.client = client
        self.backend = backend
        self.timeout = timeout


def _flatten_alpha(image):
    """RGB copy of `image` with any transparency composited onto WHITE.

    The edit system prompt describes the references as they appear in the
    rendered image, and a generated image has no transparency, so alpha must be
    resolved rather than dropped: pasting onto black would turn any cut-out
    subject into a different-looking picture, and discarding the channel
    outright makes Pillow leave the RGB values of transparent pixels as-is —
    usually arbitrary and often a colour that was never visible.
    """
    pil = _reference_store._load_pil()
    if image.mode == "RGB":
        return image
    has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
    if not has_alpha:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = pil.new("RGB", rgba.size, (255, 255, 255))
    canvas.paste(rgba, mask=rgba.getchannel("A"))
    return canvas


def _reference_image_uri(root: str | None, reference_id: str, ordinal: int) -> str:
    """One reference as a downscaled PNG data URI, ready for the user turn.

    Reads the stored blob (hash-verified), applies the store's own decode
    gates, flattens alpha onto white, and shrinks the longest side to
    :data:`REWRITE_IMAGE_MAX_DIM`. Returns ``data:image/png;base64,...`` —
    never a path and never a URL, so the backend cannot be made to fetch
    anything.

    Every failure names the reference by ORDINAL only: the store's own errors
    quote paths and (truncated) ids, and neither belongs in a browser-visible
    message.
    """
    if root is None:
        raise ApiError(422, "reference_unavailable",
                       "the server reference store is not configured")
    try:
        raw = _reference_store.read_blob(root, reference_id)
        image = _reference_store._decode_checks(raw)
    except _reference_store.ReferenceStoreError:
        raise ApiError(404, "reference_unknown",
                       f"reference {ordinal} is not available in the server "
                       "reference store") from None
    pil = _reference_store._load_pil()
    image = _flatten_alpha(image)
    if max(image.size) > REWRITE_IMAGE_MAX_DIM:
        image.thumbnail((REWRITE_IMAGE_MAX_DIM, REWRITE_IMAGE_MAX_DIM),
                        pil.LANCZOS)
    buffer = io.BytesIO()
    try:
        image.save(buffer, format="PNG")
    except Exception:  # noqa: BLE001 - any encoder failure is the same refusal
        raise ApiError(404, "reference_unknown",
                       f"reference {ordinal} cannot be prepared for the "
                       "rewriter") from None
    return _IMAGE_URI_PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")


def _reference_image_uris(root: str | None, reference_ids: list[str]) -> list[str]:
    """The ordered data URIs for `reference_ids`, in the user's order.

    Order is load-bearing: ``build_messages`` puts image N first and the edit
    system prompt refers to it as ``<imageN>``, so reordering here would
    silently re-point every reference in the rewrite.
    """
    return [_reference_image_uri(root, reference_id, ordinal)
            for ordinal, reference_id in enumerate(reference_ids, start=1)]


# ---------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------


class RewriteService:
    """Single-slot rewrite executor over two backends. One instance per server."""

    def __init__(self, settings: RewriteSettings, transport: Callable | None = None,
                 *, pe: PESettings | None = None,
                 gpu_lock: threading.Lock | None = None,
                 cache_dir: str | None = None,
                 preferences: RewritePreferences | None = None,
                 client_factory: Callable[[str, RewriteSettings, PESettings,
                                           RewritePreferences], Any] | None = None
                 ) -> None:
        """`settings` is the local-service configuration (or
        :meth:`RewriteSettings.disabled`); `pe` the enhancer's, or None.

        ``gpu_lock`` is the SAME lock ``WebGeneration`` takes around its
        executor: with no lock the enhancer and an image job would load onto
        the V100 together and one of them would fail with an out-of-memory.
        ``cache_dir`` is where the saved Settings document and the enhancer log
        live. ``client_factory`` builds clients and exists for tests only.
        """
        self.settings = settings
        self._pe = pe
        self._transport = transport
        self._gpu_lock = gpu_lock if gpu_lock is not None else threading.Lock()
        self._cache_dir = cache_dir
        self._settings_path = preferences_path(cache_dir)
        if pe is not None and pe.log_path is None:
            pe = replace(pe, log_path=(None if cache_dir is None
                                       else os.path.join(cache_dir, PE_LOG_FILENAME)))
        self.pe = pe
        self._preferences = preferences or RewritePreferences()
        self._client_factory = client_factory or self._build_client
        self._cond = threading.Condition()
        self._current: _Job | None = None
        self._closing = False
        self._clients: dict[str, Any] = {}
        self._active: str | None = None
        self._last_error = ""
        self._rebuild_clients()

    # -- capability ---------------------------------------------------------

    def _backend_available(self, backend: str) -> bool:
        """What the LAUNCH FLAGS make possible.

        The enhancer needs its binary and its GGUF; the local service needs an
        address, a model name and its two system prompt files. Neither can be
        widened by a saved choice — a browser never sends a binary, a GGUF or a
        prompt file. What the flags do NOT decide is which one is the DEFAULT:
        ``--rewrite-backend`` only seeds the choice, so a server that starts on
        the enhancer can still answer an edit draft with the local service, and
        Settings can still switch to it.
        """
        if backend == "pe-llamacpp":
            return self.pe is not None
        if backend == "local-llm":
            return self.settings.local_complete
        return False

    def available_backends(self) -> list[str]:
        return [backend for backend in BACKENDS if self._backend_available(backend)]

    def backend_rows(self, only_available: bool = False) -> list[dict[str, Any]]:
        """The ``backends`` list, in preference order (enhancer first).

        Every row is ``{"id", "label", "tasks", "uses_gpu"}`` — labels and task
        names only, never a path, an address or a model identifier. The status
        block publishes the available ones; the Settings payload publishes all
        of them, with ``available`` and ``reason`` added.
        """
        rows = [{"id": backend, "label": BACKEND_LABELS[backend],
                 "tasks": list(BACKEND_TASKS[backend]),
                 "uses_gpu": BACKEND_USES_GPU[backend]}
                for backend in BACKENDS
                if not only_available or self._backend_available(backend)]
        return rows

    def unavailable_reason(self, backend: str) -> str:
        """Why a radio is greyed out. No paths, ever — the reason is rendered
        in the browser, and a path here would say where the operator's files
        live."""
        if backend == "pe-llamacpp":
            return ("this server was started without the enhancer flags, so "
                    "the llama-server binary and its Q8 model are not "
                    "configured")
        if backend == "local-llm":
            return ("this server was started without the local rewriting "
                    "service, so its endpoint, model or system prompt files "
                    "are not configured")
        return ""


    def preferences(self) -> RewritePreferences:
        """The choice in force. Read under the lock so a caller that has to
        roll a change back after a failed save gets the document that was in
        force BEFORE it, not a half-adopted one."""
        with self._cond:
            return self._preferences

    @property
    def active_backend(self) -> str | None:
        """The backend a rewrite would use right now, or None when off."""
        with self._cond:
            return self._active

    @property
    def available(self) -> bool:
        """True when a rewrite can actually be attempted."""
        with self._cond:
            return self._active is not None

    def status_block(self) -> dict[str, Any]:
        """The additive status block. Never carries a host, port or path.

        It says which backend is active, which ones this server could run, and
        where the choice is changed — a ROUTE plus whether a saved document is
        on disk. The on-disk path itself is for the operator's console only.
        """
        with self._cond:
            active = self._active
            choice = self._preferences.backend
            detail = self._last_error
            configured = bool(self._fallback_locked())
            backends = self.backend_rows(only_available=True)
        pointer = {"href": web_api.REWRITE_SETTINGS_HREF,
                   "saved": bool(self._settings_path) and os.path.exists(
                       self._settings_path)}
        if active is None:
            if choice == "off":
                block = dict(web_api.REWRITE_DISABLED_BLOCK)
                block["reason"] = ("rewriting is switched off here; choose a "
                                   "rewriter in Settings")
                block["settings"] = pointer
                return block
            if not configured:
                # The operator started this server with neither rewriter's
                # flags; Settings cannot conjure a binary or a prompt file.
                return {**web_api.REWRITE_DISABLED_BLOCK, "settings": pointer}
            # Something is configured but no client could be built (a bad
            # saved endpoint, for instance): say so without disclosing where
            # the backend lives. The full line is for the console only.
            block = dict(web_api.REWRITE_DISABLED_BLOCK)
            block["reason"] = _public_reason(detail)
            block["settings"] = pointer
            return block
        # Built through the ONE validator (web_api.validate_rewrite_block), so
        # this block can never be a shape the browser parser rejects — and can
        # never carry a host, port or path in `reason`.
        return web_api.validate_rewrite_block(
            {"state": "available", "code": "rewrite_available",
             "backend": active, "reason": "", "backends": backends,
             "settings": pointer})

    def settings_state(self) -> dict[str, Any]:
        """The ``GET /api/v1/settings/rewrite`` payload.

        It carries the saved choice and the local service's endpoint — the user
        typed both — plus which backends exist and why one does not. It never
        carries a filesystem path: no binary, no GGUF, no prompt file, no log
        file.
        """
        with self._cond:
            active = self._active
            saved = self._preferences
            rows = self.backend_rows()
            persist = self._settings_path is not None
        # The effective local service: the saved override where there is one,
        # else the launch flags. The page prefills from this, and the Test
        # button probes exactly what it shows.
        local = {"backend": None,
                 "endpoint": saved.endpoint or self.settings.endpoint,
                 "model": saved.model or self.settings.model,
                 "timeout_s": saved.timeout_s or self.settings.timeout}
        payload = {"schema": REWRITE_SETTINGS_SCHEMA,
                   "backend": active,
                   "saved": saved.as_dict(),
                   "local": local,
                   "backends": [{**row,
                                 "available": self._backend_available(row["id"]),
                                 "reason": ("" if self._backend_available(row["id"])
                                            else self.unavailable_reason(row["id"]))}
                                for row in rows],
                   "off": {"id": "off", "label": "Off", "available": True,
                           "reason": ""},
                   "persistent": persist}
        if not persist:
            payload["note"] = ("this server has no cache directory, so a change "
                               "here lasts until it is restarted")
        return web_api.validate_rewrite_settings_state(payload)

    # -- client map ---------------------------------------------------------

    def _build_client(self, backend: str, settings: RewriteSettings,
                      pe: PESettings | None, prefs: RewritePreferences) -> Any:
        """The real client for one backend under the current preferences."""
        if backend == "local-llm":
            endpoint = prefs.endpoint or settings.endpoint
            model = prefs.model or settings.model
            timeout = prefs.timeout_s or settings.timeout
            try:
                rc.parse_endpoint(endpoint)
            except rc.RewriteConfigError as exc:
                self._last_error = _safe(exc)
            if not settings.prompts.get("t2i") or not settings.prompts.get("edit"):
                # The system prompts are launch-flag-only: no saved choice can
                # make the local service possible without them.
                raise rc.RewriteConfigError(
                    "the local rewriter has no system prompt files loaded")
            return settings.build_client(transport=self._transport,
                                         endpoint=endpoint, model=model,
                                         timeout=timeout)
        assert pe is not None
        return pe.build_client(gpu_lock=self._gpu_lock,
                               cancel=self._current_cancelled)

    def _rebuild_clients(self, prefs: RewritePreferences | None = None) -> None:
        """Adopt `prefs` (default: the current ones) and rebuild the client map.

        Called at construction and after every saved-settings change. A rewrite
        already in flight keeps the client it claimed (bound on its job), so
        applying a change never disturbs a running rewrite — it takes effect on
        the next one. A backend whose client cannot be built is dropped and the
        reason kept for the status block; the server never dies over a bad
        saved endpoint.
        """
        prefs = self._preferences if prefs is None else prefs
        clients: dict[str, Any] = {}
        self._last_error = ""
        for backend in BACKENDS:
            if not self._backend_available(backend):
                continue
            try:
                clients[backend] = self._client_factory(backend, self.settings,
                                                        self.pe, prefs)
            except (rc.RewriteConfigError, ApiError) as exc:
                self._last_error = _safe(
                    exc.message if isinstance(exc, ApiError) else str(exc))
        with self._cond:
            # Prefs, clients and the active id move together: a concurrent
            # settings_state() never sees a new choice beside old clients.
            self._preferences = prefs
            self._clients = clients
            self._active = self._choose_active_locked()

    def _choose_active_locked(self) -> str | None:
        """The active backend, given the saved choice and availability."""
        chosen = self._preferences.backend
        if chosen == "off":
            return None
        if chosen is not None and chosen in self._clients:
            return chosen
        fallback = self._fallback_locked()
        if chosen is not None:
            # The saved choice names something this server cannot run (the
            # enhancer flags are gone, or the local system prompts were never
            # loaded). Fall back rather than silently going quiet, and say why
            # in the status block.
            self._last_error = (
                f"the saved rewriter choice {chosen!r} is not available on this "
                f"server; using {fallback}" if fallback else
                "the saved rewriter choice is not available on this server, and "
                "no other rewriter is configured")
        return fallback

    def _fallback_locked(self) -> str | None:
        """The default when nothing was chosen: enhancer first, then local."""
        for backend in BACKENDS:
            if backend in self._clients:
                return backend
        return None

    def check_preferences(self, prefs: RewritePreferences) -> None:
        """Refuse a choice this server cannot run, BEFORE it is saved.

        The browser may ask for the enhancer on a server that was started
        without its flags: that is not a setting, it is a question about files
        only the operator can place, so the answer is a refusal that says why.
        (A saved document naming a backend that has since disappeared is
        different: startup falls back and states it in the status block,
        because there is no user waiting for an answer.)
        """
        backend = prefs.backend
        if backend is not None and backend != "off" and not self._backend_available(backend):
            raise ApiError(422, "backend_unavailable", self.unavailable_reason(backend))

    def apply_preferences(self, prefs: RewritePreferences, *,
                          persist: bool = True) -> None:
        """Adopt a new choice (and save it when a cache dir exists).

        Raises whatever ``save_preferences`` raises: a change that cannot be
        written is a typed failure, never a lie about having been applied. A
        rewrite already in flight is not touched — it finishes on the client it
        claimed.
        """
        previous = self._preferences
        self._rebuild_clients(prefs)
        if not persist:
            return
        try:
            save_preferences(self._settings_path, prefs)
        except Exception:
            # The file did not take; neither may the in-memory choice.
            self._rebuild_clients(previous)
            raise

    def submit(self, request: Mapping[str, Any]) -> ApiResponse:
        """Run one rewrite to completion (or refuse). Never queues."""
        fields = self._validate(request)
        job = self._claim(fields["task"])
        thread = threading.Thread(target=self._run, args=(job, fields),
                                  name="rewrite", daemon=True)
        thread.start()
        return self._await(job, fields)

    def _pick_client(self, task: str) -> tuple[Any, str]:
        """The client for `task`, under the Condition, or a typed refusal.

        Order: the active backend first; if it cannot do this task, the other
        available one (an edit draft with the enhancer active uses the local
        service, which is what the user asked for by configuring it). Only when
        no available backend supports the task is the request refused — with
        the reason, because "the rewriter cannot do this" is information, not a
        failure to explain.
        """
        with self._cond:
            active, clients = self._active, dict(self._clients)
        if active is None:
            raise ApiError(503, "rewrite_disabled",
                           "rewriting is switched off in Settings")
        order = [active] + [backend for backend in BACKENDS if backend != active]
        for backend in order:
            client = clients.get(backend)
            if client is not None and task in BACKEND_TASKS[backend]:
                return client, backend
        raise ApiError(
            422, "task_unsupported",
            "the official enhancer rewrites text-to-image drafts only, and no "
            "local LLM service is configured for edit drafts")

    def _claim(self, task: str) -> _Job:
        """Take the single slot or refuse. The job owns its own cancel flag."""
        client, backend = self._pick_client(task)
        timeout = float(getattr(client, "timeout", self.settings.timeout))
        with self._cond:
            if self._closing:
                raise ApiError(503, "rewrite_disabled",
                               "the server is shutting down")
            if self._current is not None:
                raise ApiError(503, "rewrite_busy",
                               "another rewrite is already in progress")
            job = _Job(client, backend, timeout)
            self._current = job
            return job

    def _await(self, job: _Job, fields: dict[str, Any]) -> ApiResponse:
        """Wait for `job`, honouring cancel and the deadline.

        The deadline is the chosen backend's own budget: the enhancer may
        legitimately think for minutes, the local service should answer in
        seconds. Leaving early does NOT free the slot: the worker still owns
        it, and it releases the slot in its own ``finally``. So a timed-out or
        cancelled rewrite cannot overlap the next one — the following attempt is
        refused 503 until the backend call actually returns (and, for the
        enhancer, until its server process is gone).
        """
        deadline = time.monotonic() + job.timeout
        with self._cond:
            while not job.finished:
                if job.cancel.is_set():
                    raise ApiError(503, "rewrite_cancelled",
                                   "the rewrite was cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    job.cancel.set()  # discard whatever still arrives
                    raise ApiError(
                        504, "rewrite_timeout",
                        f"the rewriter did not answer within "
                        f"{job.timeout:g} s")
                self._cond.wait(remaining)
        if job.error is not None:
            if job.cancel.is_set():
                # The enhancer's watchdog ends a cancelled rewrite by killing
                # its server, so the worker can fail and notify a moment before
                # this thread wakes. The failure is the CONSEQUENCE of the
                # cancel: say what happened instead of reporting a timeout the
                # user never waited for.
                raise ApiError(503, "rewrite_cancelled",
                               "the rewrite was cancelled")
            raise _as_api_error(job.error, fields["prompt"], job.backend)
        if job.payload is None:  # internal invariant, never a backend failure
            raise ApiError(502, "rewrite_unavailable",
                           "the rewrite completed without a result")
        return ApiResponse(status=200, payload=job.payload)

    def _run(self, job: _Job, fields: dict[str, Any]) -> None:
        """Worker body. Owns the slot release; never lets an exception escape."""
        started = time.monotonic()
        try:
            task = fields["task"]
            if job.cancel.is_set():
                raise _Cancelled
            # Decode and re-encode the references HERE, on the worker thread:
            # _claim() took the slot under ``self._cond`` and released the lock
            # before this thread existed, so no imaging work runs while the
            # service holds its Condition, and a slow decode cannot block
            # status reads or the next caller's refusal.
            images = _reference_image_uris(self.settings.references_dir,
                                           fields["image_ids"])
            if job.cancel.is_set():
                raise _Cancelled
            system_prompt = (self.pe.system_prompt
                             if job.backend == "pe-llamacpp"
                             else self.settings.prompts[task])
            messages = rw.build_messages(system_prompt, fields["prompt"],
                                         images=images)
            content = job.client.complete(messages, task)
            if job.cancel.is_set():
                raise _Cancelled
            elapsed_ms = int((time.monotonic() - started) * 1000)
            job.payload = _payload(task, fields, content,
                                   job.backend, elapsed_ms)
            job.elapsed_ms = elapsed_ms
        except BaseException as exc:  # noqa: BLE001 - surfaced to the waiter
            job.payload = None
            job.error = exc
        finally:
            with self._cond:
                if self._current is job:
                    self._current = None
                job.finished = True
                self._cond.notify_all()

    # -- connection probe ---------------------------------------------------

    def test_connection(self, endpoint: str, model: str,
                        timeout_s: float | None = None) -> dict[str, Any]:
        """Ask the LOCAL service one tiny question and report it in plain words.

        No GPU, no rewrite, no draft, no slot: a probe must not block the user
        from saving the settings they are testing. ``max_new_tokens`` is
        overridden to a handful of tokens and the budget is capped, so a wrong
        endpoint comes back in seconds. The result never names a path.
        """
        rc.parse_endpoint(endpoint)
        budget = min(float(timeout_s or DEFAULT_TIMEOUT_SECONDS), TEST_TIMEOUT_CAP)
        client = rc.RewriteClient(endpoint=endpoint, model=model, timeout=budget,
                                  transport=self._transport)
        started = time.monotonic()
        try:
            client.complete([dict(message) for message in _TEST_MESSAGES], "t2i",
                            {"max_new_tokens": TEST_MAX_TOKENS})
        except rc.RewriteClientError as exc:
            return web_api.rewrite_test_result(
                False, f"failed: {_safe(exc) or 'the service did not answer'}",
                (time.monotonic() - started) * 1000)
        return web_api.rewrite_test_result(
            True, "ok: the service answered",
            (time.monotonic() - started) * 1000)

    # -- cancellation / shutdown -------------------------------------------

    def _current_cancelled(self) -> bool:
        """The enhancer's cancel probe: is THE in-flight rewrite cancelled?

        Correct only because the service holds exactly one slot: there is never
        a second rewrite this could be answering for. It is how a cancel or a
        shutdown reaches a llama-server child that is halfway through thinking
        (the waiter cannot interrupt it, but the child can be killed).
        """
        with self._cond:
            job = self._current
        return bool(job is not None and job.cancel.is_set())

    def cancel(self) -> bool:
        """Cancel the in-flight rewrite's result. False if nothing was running.

        Cooperative: the waiter returns immediately. For the local service a
        blocked socket read keeps running until the client's own timeout; for
        the enhancer this flag is polled and the server process is terminated.
        """
        with self._cond:
            if self._current is None:
                return False
            self._current.cancel.set()
            self._cond.notify_all()
            return True

    def close(self, timeout_s: float = 5.0) -> bool:
        """Cancel and wait briefly, then make sure no enhancer survives.

        Returns True when idle (no thread left) and no live child. A rewrite
        abandoned on a blocked socket still ends via the client's own timeout —
        but the enhancer's child is killed here regardless, because an orphan
        holding 9.5 GiB outlives the server that started it.
        """
        with self._cond:
            self._closing = True
            if self._current is not None:
                self._current.cancel.set()
            self._cond.notify_all()
            deadline = time.monotonic() + timeout_s
            while self._current is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            idle = self._current is None
            clients = dict(self._clients)
        pe_client = clients.get("pe-llamacpp")
        kill = getattr(pe_client, "close", None)
        if kill is not None:
            idle = bool(kill()) and idle
        return idle

    # -- logging surface ----------------------------------------------------

    def event(self, task: str, *, prompt_chars: int, elapsed_ms: int | None,
              parse_ok: bool | None, outcome: str,
              backend: str | None = None) -> dict[str, Any]:
        """The only rewrite fields that may be logged. No prompt, ever."""
        return {"event": "rewrite", "task": task,
                "backend": backend or self.active_backend,
                "elapsed_ms": elapsed_ms, "prompt_chars": prompt_chars,
                "parse_ok": parse_ok, "outcome": outcome}

    # -- validation (defensive: api.py validates too) -----------------------

    @staticmethod
    def _validate(request: Mapping[str, Any]) -> dict[str, Any]:
        # ONE definition of a legal request, shared with the transport.
        return web_api.validate_rewrite_fields(request)


def _payload(task: str, fields: Mapping[str, Any], content: str,
             backend: str, elapsed_ms: int) -> dict[str, Any]:
    """The 200 payload. `original_prompt` is echoed so the UI can prove staleness."""
    parsed = rw.parse_answer(content, task)
    return {
        "schema": WEB_REWRITE_SCHEMA,
        "task": task,
        "draft_revision": fields["draft_revision"],
        "original_prompt": fields["prompt"],
        "rewritten_prompt": parsed.prompt,
        "parse_ok": parsed.parse_ok,
        "unparsed_answer": parsed.unparsed_answer,
        "ratio": parsed.ratio,
        "ratio_conflict": parsed.ratio_conflict,
        "backend": backend,
        "elapsed_ms": elapsed_ms,
    }


def _as_api_error(exc: BaseException, prompt: str = "",
                  backend: str = "") -> ApiError:
    """Map a worker failure to a typed refusal: no prompt, no location.

    A backend or client error may quote the request it failed on, so the draft
    is scrubbed out of whatever text is forwarded. A transport error also
    quotes the address it could not reach and an enhancer error can quote the
    binary it tried to run — a path, a URL or an address must never reach the
    browser, so the location-free copy the status block uses is sent there,
    and the full line is printed once on the server console for the operator,
    who is the only reader that can do anything about it.
    """
    def console(detail: str) -> None:
        print(f"rewrite failed ({backend or 'unknown'}): {_safe(detail)}",
              file=sys.stderr)

    if isinstance(exc, _Cancelled):
        return ApiError(503, "rewrite_cancelled", "the rewrite was cancelled")
    if isinstance(exc, ApiError):
        return _scrub_error(exc, prompt)
    if isinstance(exc, pe_client.RewriteGpuBusy):
        # Checked before the generic client errors: this one is a capacity
        # answer the user can act on, not a backend fault.
        return ApiError(503, "rewrite_gpu_busy", _scrub(_public_reason(
            _safe(exc), "the GPU is generating an image; try again when it "
                        "finishes"), prompt))
    if isinstance(exc, pe_client.RewriteTaskUnsupported):
        return ApiError(422, "task_unsupported", _scrub(_public_reason(
            _safe(exc), "this rewriter cannot handle that task"), prompt))
    if isinstance(exc, rc.RewriteTimeout):
        return ApiError(504, "rewrite_timeout",
                        _scrub("the rewriter did not answer before its "
                               "deadline", prompt))
    if isinstance(exc, rc.RewriteInvalidAnswer):
        console(exc)
        return ApiError(502, "rewrite_invalid_answer",
                        _scrub(_public_reason(
                            _safe(exc), "the rewriter's reply was not a "
                                        "completion"), prompt))
    if isinstance(exc, rc.RewriteClientError):
        console(exc)
        return ApiError(502, "rewrite_unavailable",
                        _scrub(_public_reason(
                            _safe(exc), "the rewriter service is "
                                        "unavailable"), prompt))
    if isinstance(exc, rw.RewriteError):
        # Our own message building refused the request: a caller bug, not a
        # backend fault, and the text can only describe shape or length.
        return ApiError(400, "invalid_envelope", _scrub(_safe(exc), prompt))
    return ApiError(502, "rewrite_unavailable",
                    f"the rewrite failed ({type(exc).__name__})")


def _scrub(text: str, prompt: str) -> str:
    """Remove the draft from `text`. The one leak this path must not have."""
    for needle in (prompt, prompt.strip()):
        if needle:
            text = text.replace(needle, "[prompt]")
    return text


def _scrub_error(exc: ApiError, prompt: str) -> ApiError:
    """A nested ApiError may still be quoting the draft."""
    return ApiError(exc.status, exc.code, _scrub(exc.message, prompt),
                    tuple((f, _scrub(m, prompt)) for f, m in exc.field_errors))


def _public_reason(detail: str, fallback: str = "") -> str:
    """The browser's copy of a rewriter problem: bounded, and location-free.

    The detailed line stays for the console. A transport error quotes the
    address it could not reach, an enhancer error can quote the binary it
    tried to exec, and a saved-but-broken endpoint quotes the URL — while the
    status-block validator refuses a slash, a URL scheme or an address in a
    reason. So the markers are stripped HERE rather than trusting whatever the
    upstream message happened to quote; `fallback` is the sentence used when
    nothing survives the stripping.
    """
    text = web_api.strip_command(detail)
    if not text:
        return fallback or ("the configured rewriter could not be started; "
                            "check the server console for one bounded line")
    return text if len(text) <= 200 else text[:199].rstrip() + "\u2026"
