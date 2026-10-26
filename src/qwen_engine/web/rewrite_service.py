"""Rewrite service: one in-flight rewrite, behind a typed refusal.

**Entry-point contract** (what ``web/server.py`` and ``web/api.py`` call; the
route itself is R3's later half and is not built here):

*Build once at startup, from flags — never inside a request.* ::

    settings = RewriteSettings(backend="local-llm", endpoint="http://127.0.0.1:8079/v1",
                               model="Qwen3.8", prompts={"t2i": <system prompt text>},
                               timeout=180.0)
    settings = RewriteSettings.disabled()          # the default: flag ``off``
    service = RewriteService(settings)             # transport= injected in tests only

Construction is where a bad endpoint, model or missing prompt fails: those are
operator errors and the server must refuse to start rather than 503 later.
:meth:`RewriteService.status_block` is then served from the cached status.

*Per request* (body already parsed to a mapping by ``api.build_rewrite_request``;
this module re-validates, cheaply and defensively): ::

    request = {"task": "t2i", "prompt": "<draft>", "draft_revision": 7,
               "image_ids": []}
    response = service.submit(request)   # -> ApiResponse(200, payload)

Body keys are exactly ``task``/``prompt``/``draft_revision``/``image_ids``; any
other key is refused rather than ignored. ``draft_revision`` is echoed verbatim
so the UI can discard a stale suggestion.

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
``ratio_conflict``, ``backend``, ``elapsed_ms``. The server never maps a ratio
to width/height — ``PRESETS`` in ``app.js`` is the single source of truth.

*Typed failures* — :class:`~qwen_engine.web.api.ApiError`, messages bounded and
**never containing prompt text**::

    503 rewrite_disabled         flag is off, or the service was never configured
    503 rewrite_busy             another rewrite is in flight (never a queue)
    504 rewrite_timeout          the deadline expired; the draft is untouched
    502 rewrite_unavailable      transport failure / non-2xx from the backend
    502 rewrite_invalid_answer   transport violation (not a chat completion)
    422 prompt_too_long          over ``PROMPT_MAX_CHARS``
    422 task_unsupported         unknown task, or a t2i rewrite with images
    422 reference_required       an edit rewrite with no reference images
    422 too_many_references      more ids than the edit-job cap (``REF_MAX``)
    404 reference_unknown        an id is unknown or its blob is unreadable
    400 invalid_envelope         malformed body: unknown/missing key, bad types
    503 rewrite_cancelled        cancelled by shutdown or operator; see below

``rewrite_cancelled`` is an **addition** to the plan's typed set (the plan has
no cancel route); it exists so shutdown can release a waiter with a typed
refusal instead of a lie about time or capacity. Root can drop the code once
the route decides how cancel is exposed.

*Cancellation is cooperative, not an abort.* :meth:`RewriteService.cancel`
prevents a pending answer being applied and releases the waiter immediately,
but it cannot interrupt a blocked socket read — the client's connect/read
timeout is the hard bound on wall time. Do not describe cancel as stopping the
backend.

*Single flight, no queue.* One slot: a second caller is refused at once rather
than waiting behind the first or transparently retrying. The call runs on a
worker thread so the four HTTP handlers (``MAX_CONCURRENT_HANDLERS``) are
released. This service never touches ``JobController`` or the executor lease —
a rewrite must not queue behind, or stall, an image job.

*No prompt in logs or errors.* Nothing here logs. :meth:`RewriteService.event`
returns the only fields the caller may log — ``task``, ``backend``,
``elapsed_ms``, ``prompt_chars``, ``parse_ok``, ``outcome`` — deliberately no
prompt, no endpoint, no path, and no reference id.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import base64
import io

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

#: Route body cap, re-exported under the name the service contract used.
#: Derived from ``PROMPT_MAX_CHARS`` in one place only (``web/api.py``).
MAX_REQUEST_BYTES = web_api.MAX_REWRITE_REQUEST_BYTES

#: The only backend that exists. ``off`` is not a backend, it is the default.
BACKENDS = ("local-llm",)

#: Default deadline. Worst measured round trip against the shared service was
#: 38 s; a slower backend needs this raised explicitly, as a flag decision.
DEFAULT_TIMEOUT_SECONDS = rc.DEFAULT_TIMEOUT_SECONDS

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


class _Cancelled(Exception):
    """Internal: the answer arrived but was cancelled before it could be used."""


@dataclass(frozen=True)
class RewriteSettings:
    """Startup-owned configuration; validated once, never per request."""

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
                     transport: Callable[[bytes], bytes] | None = None) -> rc.RewriteClient:
        """The client for this configuration. `transport` is for tests only."""
        return rc.RewriteClient(endpoint=self.endpoint, model=self.model,
                                timeout=self.timeout, transport=transport)


class _Job:
    """One in-flight rewrite's hand-off slot, guarded by the service Condition."""

    __slots__ = ("cancel", "finished", "payload", "error", "elapsed_ms")

    def __init__(self) -> None:
        # Per-job, not per-service: a cancel aimed at one rewrite must never
        # poison the next one.
        self.cancel = threading.Event()
        self.finished = False
        self.payload: dict[str, Any] | None = None
        self.error: BaseException | None = None
        self.elapsed_ms: int | None = None


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


class RewriteService:
    """Single-slot rewrite executor. One instance per server."""

    def __init__(self, settings: RewriteSettings,
                 transport: Callable[[bytes], bytes] | None = None) -> None:
        self.settings = settings
        self._client = (None if settings.backend == "off"
                        else settings.build_client(transport=transport))
        self._cond = threading.Condition()
        self._current: _Job | None = None
        self._closing = False

    # -- capability ---------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when a rewrite can actually be attempted (flag on + configured)."""
        return self._client is not None

    def status_block(self) -> dict[str, Any]:
        """The additive status block. Never carries a host, port or path."""
        if self._client is None:
            return dict(web_api.REWRITE_DISABLED_BLOCK)
        # Built through the ONE validator (web_api.validate_rewrite_block), so
        # this block can never be a shape the browser parser rejects — and can
        # never carry a host, port or path in `reason`.
        return web_api.validate_rewrite_block(
            {"state": "available", "code": "rewrite_available",
             "backend": self.settings.backend, "reason": ""})

    # -- execution ----------------------------------------------------------

    def submit(self, request: Mapping[str, Any]) -> ApiResponse:
        """Run one rewrite to completion (or refuse). Never queues."""
        if self._client is None:
            raise ApiError(503, "rewrite_disabled",
                           "rewriting is not enabled on this server")
        fields = self._validate(request)
        job = self._claim()
        thread = threading.Thread(target=self._run, args=(job, fields),
                                  name="rewrite", daemon=True)
        thread.start()
        return self._await(job, fields)

    def _claim(self) -> _Job:
        """Take the single slot or refuse. The job owns its own cancel flag."""
        with self._cond:
            if self._closing:
                raise ApiError(503, "rewrite_disabled",
                               "the server is shutting down")
            if self._current is not None:
                raise ApiError(503, "rewrite_busy",
                               "another rewrite is already in progress")
            job = _Job()
            self._current = job
            return job

    def _await(self, job: _Job, fields: dict[str, Any]) -> ApiResponse:
        """Wait for `job`, honouring cancel and the deadline.

        Leaving early does NOT free the slot: the worker still owns it, and it
        releases the slot in its own ``finally``. So a timed-out or cancelled
        rewrite cannot overlap the next one — the following attempt is refused
        503 until the backend call actually returns.
        """
        deadline = time.monotonic() + self.settings.timeout
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
                        f"{self.settings.timeout:g} s")
                self._cond.wait(remaining)
        if job.error is not None:
            raise _as_api_error(job.error, fields["prompt"])
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
            messages = rw.build_messages(self.settings.prompts[task],
                                         fields["prompt"], images=images)
            content = self._client.complete(messages, task)
            if job.cancel.is_set():
                raise _Cancelled
            elapsed_ms = int((time.monotonic() - started) * 1000)
            job.payload = _payload(task, fields, content,
                                   self.settings.backend, elapsed_ms)
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

    # -- cancellation / shutdown -------------------------------------------

    def cancel(self) -> bool:
        """Cancel the in-flight rewrite's result. False if nothing was running.

        Cooperative: the waiter returns immediately, but a blocked socket read
        keeps running until the client's own timeout — no thread is killed, and
        the slot stays taken until it does.
        """
        with self._cond:
            if self._current is None:
                return False
            self._current.cancel.set()
            self._cond.notify_all()
            return True

    def close(self, timeout_s: float = 5.0) -> bool:
        """Cancel and wait briefly. Returns True when idle (no thread left)."""
        with self._cond:
            self._closing = True
            if self._current is not None:
                self._current.cancel.set()
            self._cond.notify_all()
            deadline = time.monotonic() + timeout_s
            while self._current is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    # -- logging surface ----------------------------------------------------

    def event(self, task: str, *, prompt_chars: int, elapsed_ms: int | None,
              parse_ok: bool | None, outcome: str) -> dict[str, Any]:
        """The only rewrite fields that may be logged. No prompt, ever."""
        return {"event": "rewrite", "task": task,
                "backend": self.settings.backend,
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


def _as_api_error(exc: BaseException, prompt: str = "") -> ApiError:
    """Map a worker failure to a typed refusal. Messages carry no prompt.

    A backend or client error may quote the request it failed on, so the draft
    is scrubbed out of whatever text is forwarded. The diagnostics that matter
    for an operator ("truncated: got 45 of 145 bytes", "unreachable") never
    contain the draft, so scrubbing costs nothing real.
    """
    if isinstance(exc, _Cancelled):
        return ApiError(503, "rewrite_cancelled", "the rewrite was cancelled")
    if isinstance(exc, ApiError):
        return _scrub_error(exc, prompt)
    if isinstance(exc, rc.RewriteTimeout):
        return ApiError(504, "rewrite_timeout",
                        _scrub("the rewriter did not answer before its "
                               "deadline", prompt))
    if isinstance(exc, rc.RewriteInvalidAnswer):
        return ApiError(502, "rewrite_invalid_answer",
                        _scrub(_safe(exc) or
                               "the rewriter's reply was not a completion", prompt))
    if isinstance(exc, rc.RewriteClientError):
        return ApiError(502, "rewrite_unavailable",
                        _scrub(_safe(exc) or
                               "the rewriter service is unavailable", prompt))
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
