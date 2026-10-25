"""Pure W2 web contract: status / plan / editable-config-export (no sockets).

Every function here is deterministic and ML-free. The STATUS response is
served from metadata CACHED at startup — it performs no filesystem I/O at
request time. PLAN and EXPORT run the SHARED config validator, which may
STAT configured runtime-path metadata (directories only, never weight
bytes) while validating the request against the manifest. No path here
ever READS model weights or computes hashes. The only filesystem knowledge
enters through :class:`WebRuntime`, which the server validates and
canonicalises once at startup. Browser JSON can never select a runtime path — unknown
path fields are refused, not stripped.

Wire rules that are load-bearing (reviewed W2 contract):

* Public seeds are **decimal strings or null** in plan/status — never numeric
  JSON (JS ``Number`` would round past 2^53). The config export is the one
  deliberate exception: raw bytes with the exact Python uint64 number,
  downloaded as a blob and never re-serialized by the browser.
* Request seed wire: decimal ``[0-9]{1,20}`` string or null; floats, bools,
  numbers and over-long digit strings are refused typed — never parsed
  through a float and never silently rounded.
* Responses carry only schema-versioned, bounded, safe content: no runtime
  paths, no manifest filenames, no hashes, no tracebacks, no raw internal
  exception text.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from qwen_engine import _jsonsafe
from qwen_engine import reference_store as _reference_store
from qwen_engine import rewrite as _rewrite
from qwen_engine.config import SEED_MAX, ConfigError, GenerationConfig
from qwen_engine.core import plan as core_plan

#: Wire schema tags (versioned like every other qwen_engine document).
WEB_STATUS_SCHEMA = "qwen_engine.web_status.v1"
WEB_PLAN_SCHEMA = "qwen_engine.web_plan.v1"
WEB_PLAN_REQUEST_SCHEMA = "qwen_engine.web_plan_request.v1"
WEB_ERROR_SCHEMA = "qwen_engine.web_error.v1"
WEB_REFERENCE_SCHEMA = "qwen_engine.web_reference.v1"
#: Prompt-rewriting wire tags (plan §4.1). Defined HERE, next to the other
#: wire tags, because this module is the transport's contract boundary; the
#: service imports them from here rather than keeping a second copy.
REWRITE_REQUEST_SCHEMA = "qwen_engine.web_rewrite_request.v1"
WEB_REWRITE_SCHEMA = "qwen_engine.web_rewrite.v1"

#: Route body cap for /api/v1/rewrite. DERIVED, never chosen independently:
#: the longest legal draft (``PROMPT_MAX_CHARS``) at 4 bytes/char UTF-8 plus
#: a ≤1 KiB envelope. The browser applies the same formula locally, so this
#: is a backstop; ``MAX_BODY_BYTES`` (64 KiB) is larger than this and is NOT
#: the binding cap for this route.
MAX_REWRITE_REQUEST_BYTES = _rewrite.PROMPT_MAX_CHARS * 4 + 1024

#: Route cap for raw reference uploads — the store's own byte cap is the
#: single source of truth; JSON routes keep the small 64 KiB MAX_BODY_BYTES.
MAX_REFERENCE_UPLOAD_BYTES = _reference_store.MAX_REFERENCE_BYTES

#: Sniffed-store-format -> response Content-Type (design §2.4; identical
#: magic table, one source of truth: the store).
_REFERENCE_CONTENT_TYPES = {"png": "image/png", "jpeg": "image/jpeg",
                            "webp": "image/webp"}

#: The browser-editable GenerationConfig fields — exactly these, no more.
#: Runtime paths (output/cache/log/benchmarks/models/manifest) are
#: server-controlled and must never arrive from browser JSON.
EDITABLE_FIELDS = frozenset({
    "prompt", "negative_prompt", "width", "height", "steps",
    "true_cfg_scale", "seed", "quant", "offload", "attention_backend",
    "vae_tiled_decode", "output_format", "transparent_background",
    "references",
})

#: Server-controlled runtime path fields (never editable from the browser).
RUNTIME_PATH_FIELDS = ("output_dir", "cache_dir", "log_dir",
                       "benchmarks_dir", "models_dir", "manifest_path",
                       "q8_manifest_path", "references_dir")

#: GenerationConfig fields that are deliberately NOT on the browser wire YET.
#: Each entry is an announced deferral with its owning milestone, never a
#: silent carve-out: the partition guard below FAILS for any field that is
#: neither editable, a runtime path, nor listed here — so a new config field
#: still cannot slip past this contract unnoticed.
#:
#: ``recipe`` — engine/CLI landed in the Turbo milestone; the browser-side
#: recipe (EDITABLE_KEYS, locked steps under Turbo, export round-trip) is the
#: separate web follow-up in docs/turbo-plan.md §6. Sending it before the UI
#: validates it would let a stale page select a recipe it cannot represent.
DEFERRED_WIRE_FIELDS = frozenset({"recipe"})

#: Seed wire grammar: ASCII decimal digits only, at most 20 characters
#: (the full uint64 range stays an exact string; range is validated later
#: against ``config.SEED_MAX``). Leading zeros are allowed; whitespace,
#: signs, exponents and non-ASCII digits are not.
_SEED_WIRE_RE = re.compile(r"[0-9]{1,20}\Z")

#: Bound on any single public error message (safe, single-line, truncated).
MAX_MESSAGE_CHARS = 200

#: The rewrite status block is exactly these four keys (plan §4.1): the UI
#: parser is strict, so the server never widens its own contract — an extra
#: key would be a silent wire change.
_REWRITE_BLOCK_KEYS = frozenset({"state", "code", "backend", "reason"})

#: The CLOSED state/code pairs, mirroring ``REWRITE_STATES`` in app.js: the
#: code is the key, the state is the value it must pair with. Rewriting is
#: optional and never a gate, so these are deliberately NOT in ``_gates()``.
_REWRITE_PAIRS: Mapping[str, str] = {
    "rewrite_available": "available",
    "rewrite_disabled": "unavailable",
}

#: Shapes that must never reach the browser: the status document is served to
#: a page, and a reason is operator-facing prose, not a place to leak where the
#: backend lives. Refusing is cheap; a leaked loopback address is not.
_LOCATION_MARKERS = ("://", "/", "\\")
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

#: A backend is a short label the UI displays, matching the CLI's own names
#: (``local-llm``); anything address-shaped is a configuration error.
_BACKEND_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def validate_rewrite_block(block: Any) -> dict[str, Any]:
    """The ONE definition of a legal rewrite status block, pure.

    Shared by :class:`WebRuntime` (which refuses a bad block at construction,
    never filtering it silently) and by the rewrite service (which builds its
    block through here), so the server cannot emit a document its own UI
    parser would reject. Raises :class:`ValueError` with a bounded message.

    The rules are the UI's rules: exactly four keys, a CLOSED state/code pair,
    a backend IFF available, a reason IFF unavailable, and no location in the
    reason.
    """
    if not isinstance(block, Mapping):
        raise ValueError("web rewrite block must be a JSON mapping")
    if set(block) != _REWRITE_BLOCK_KEYS:
        raise ValueError(
            "web rewrite block must carry exactly "
            f"{sorted(_REWRITE_BLOCK_KEYS)}; got {sorted(map(str, block))}")
    state, code = block["state"], block["code"]
    backend, reason = block["backend"], block["reason"]
    if _REWRITE_PAIRS.get(code) != state:
        raise ValueError(
            f"web rewrite block state {state!r} does not pair with code {code!r}")
    if not isinstance(reason, str):
        raise ValueError("web rewrite block reason must be a string")
    if state == "available":
        if not isinstance(backend, str) or not backend or len(backend) > 64:
            raise ValueError(
                "an available rewrite block needs a non-empty backend (<= 64 chars)")
        # The backend is echoed to the browser too: it is a LABEL ("local-llm"),
        # never an address or path.
        if not _BACKEND_LABEL_RE.match(backend):
            raise ValueError(
                "a rewrite backend must be a label (letters, digits, . _ -), "
                "not an address or path")
        if reason != "":
            raise ValueError("an available rewrite block carries no reason")
    else:
        if backend is not None:
            raise ValueError("an unavailable rewrite block has backend null")
        if not reason.strip():
            raise ValueError("an unavailable rewrite block needs a reason")
    for marker in _LOCATION_MARKERS:
        if marker in reason:
            raise ValueError(
                "a rewrite reason must not disclose a location (URL or path)")
    if _IPV4_RE.search(reason):
        raise ValueError("a rewrite reason must not disclose a location (address)")
    return dict(block)


def _safe(text: Any, limit: int = MAX_MESSAGE_CHARS) -> str:
    """One-line, length-bounded public message: control characters (and any
    newline smuggled through a repr) collapse to a space; over-length text
    gets an explicit ellipsis instead of an internal dump."""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[:limit - 1] + "\u2026"


class ApiError(Exception):
    """Typed, transport-ready failure: HTTP status + stable code + bounded
    safe message (+ optional per-field list). Never carries raw internals."""

    def __init__(self, status: int, code: str, message: str,
                 field_errors: tuple[tuple[str, str], ...] = (),
                 headers: tuple[tuple[str, str], ...] = ()) -> None:
        super().__init__(f"{code}: {message}")
        self.status = status
        self.code = code
        self.message = _safe(message)
        self.field_errors = tuple((f, _safe(m)) for f, m in field_errors)
        self.headers = headers  # e.g. Allow on 405; transport-relevant only

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": WEB_ERROR_SCHEMA,
            "code": self.code,
            "message": self.message,
        }
        if self.field_errors:
            out["fields"] = [{"field": f, "message": m}
                             for f, m in self.field_errors]
        return out


@dataclass(frozen=True)
class WebRuntime:
    """Server-controlled runtime context, canonicalised at startup.

    ``manifest_summary`` is the metadata-only summary computed once when the
    server started (see :func:`summarize_manifest_bytes` / the server's
    bounded one-shot file read). It is never re-read per request and never
    triggers artifact verification — that would stream-hash gigabytes.
    """

    output_dir: str | None = None
    cache_dir: str | None = None
    log_dir: str | None = None
    benchmarks_dir: str | None = None
    models_dir: str | None = None
    manifest_path: str | None = None
    q8_manifest_path: str | None = None
    #: Server-owned content-addressed reference store (uploads land here;
    #: the browser exchanges ids only, never paths).
    references_dir: str | None = None
    manifest_summary: Mapping[str, Any] = field(default_factory=dict)
    generation_enabled: bool = False
    #: The rewriter capability block, computed ONCE at startup from the
    #: service configuration (never per request, never a filesystem read).
    #: ``None`` means "rewriting is not available here" and is rendered as
    #: the canonical disabled block — the same thing an unconfigured server
    #: reports. Additive to the status document: the schema tag does not
    #: change (plan §4.1).
    rewrite_block: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.generation_enabled) is not bool:
            raise ValueError("generation_enabled must be a bool")
        errors: list[str] = []
        summary = _jsonsafe.freeze_json(self.manifest_summary, "manifest_summary", errors)
        if errors or not isinstance(summary, Mapping):
            raise ValueError("web manifest summary must be a strict JSON mapping")
        object.__setattr__(self, "manifest_summary", summary)
        if self.rewrite_block is not None:
            block = _jsonsafe.freeze_json(self.rewrite_block, "rewrite_block", errors)
            if errors or not isinstance(block, Mapping):
                raise ValueError("web rewrite block must be a strict JSON mapping")
            # ONE validator, so the server can never emit a block its own UI
            # parser would reject (and never leak a backend location).
            object.__setattr__(self, "rewrite_block", validate_rewrite_block(block))

    def path_fields(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in RUNTIME_PATH_FIELDS}


# ---------------------------------------------------------------------------
# Strict JSON parsing (shared typed boundary for every POST body)
# ---------------------------------------------------------------------------


class _DuplicateKey(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(f"duplicate JSON key {key!r}")
        self.key = key


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(key)
        out[key] = value
    return out


def _finite_float(raw: str) -> float:
    value = float(raw)  # float("1e999999...") yields inf, never raises
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number literal {raw!r}")
    return value


def _iter_strings(value: Any):
    """Iterative (never recursive) walk so no parsed shape can blow the
    stack while we look for unpaired surrogates."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            for ch in item:
                if "\ud800" <= ch <= "\udfff":
                    return True
        elif isinstance(item, Mapping):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


def parse_request_json(raw: bytes) -> Any:
    """Bytes -> strict JSON value with every hostile shape typed away:
    invalid UTF-8, duplicate keys (recursively), NaN/Inf literals, float
    overflow to infinity, non-finite results, unpaired surrogates, and
    structures beyond the shared depth bound. Never a raw traceback."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ApiError(400, "invalid_utf8",
                       "request body is not valid UTF-8") from None
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_pairs,
                          parse_constant=_jsonsafe.reject_constant,
                          parse_float=_finite_float)
    except _DuplicateKey as exc:
        raise ApiError(400, "duplicate_json_key", _safe(exc)) from None
    except RecursionError:
        raise ApiError(400, "json_too_deep",
                       "request JSON nests deeper than the parser accepts") from None
    except ValueError as exc:
        # JSONDecodeError, the int-string digit limit, our float guard and
        # reject_constant all surface as typed parse failures here.
        raise ApiError(400, "invalid_json",
                       _safe(f"request body is not valid JSON: {exc}")) from None
    violations = _jsonsafe.json_violations(data, at="request")
    if violations:
        raise ApiError(400, "invalid_json_structure", _safe(violations[0]))
    if _iter_strings(data):
        raise ApiError(400, "invalid_text",
                       "request contains unpaired Unicode surrogates")
    return data


# ---------------------------------------------------------------------------
# Envelope + config validation (server runtime paths injected here only)
# ---------------------------------------------------------------------------


def build_config(runtime: WebRuntime, data: Any) -> GenerationConfig:
    """Validate the exact request envelope and build the merged
    ``GenerationConfig`` (editable browser fields + server runtime paths)."""
    if not isinstance(data, Mapping):
        raise ApiError(400, "invalid_envelope",
                       "request must be a JSON object with exactly the keys "
                       "'schema' and 'config'")
    if any(not isinstance(key, str) for key in data):
        raise ApiError(400, "invalid_envelope", "request keys must be strings")
    missing = sorted({"schema", "config"} - set(data))
    extra = sorted(set(data) - {"schema", "config"})
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing envelope key(s): " + ", ".join(missing))
        if extra:
            parts.append("unexpected envelope key(s): " + ", ".join(extra))
        raise ApiError(400, "invalid_envelope", "; ".join(parts))
    if data["schema"] != WEB_PLAN_REQUEST_SCHEMA:
        raise ApiError(400, "invalid_envelope",
                       f"schema must be {WEB_PLAN_REQUEST_SCHEMA!r}")
    raw_config = data["config"]
    if not isinstance(raw_config, Mapping):
        raise ApiError(400, "invalid_envelope",
                       "'config' must be a JSON object")
    if any(not isinstance(key, str) for key in raw_config):
        raise ApiError(400, "unknown_config_field", "config keys must be strings")
    unknown = sorted(set(raw_config) - EDITABLE_FIELDS)
    if unknown:
        raise ApiError(
            400, "unknown_config_field",
            "config fields are not browser-editable (unknown or runtime "
            "path fields are refused, never stripped): " + ", ".join(unknown),
        )
    merged: dict[str, Any] = dict(runtime.path_fields())
    for key, value in raw_config.items():
        merged[key] = _seed_wire_to_int(value) if key == "seed" else value
    try:
        return GenerationConfig.from_dict(merged)
    except ConfigError as exc:
        # Errors on browser-editable fields are reported verbatim (they
        # describe only browser data). Runtime-path fields NEVER surface:
        # startup validation is not a guarantee (TOCTOU — e.g. a configured
        # dir replaced by a file afterwards), and ConfigError messages for
        # path fields can contain the raw absolute path. A runtime-path
        # mismatch is therefore ONE generic code with no path, no stat
        # detail and no raw message.
        public: list[tuple[str, str]] = []
        runtime_hit = False
        for err in exc.errors:
            if err.field in RUNTIME_PATH_FIELDS:
                runtime_hit = True
            else:
                public.append((err.field, err.message))
        code = "invalid_config"
        if runtime_hit:
            code = "server_runtime_configuration"
            public.append(("server_runtime_configuration",
                           "one or more server runtime paths configured at "
                           "startup no longer validate; a server operator "
                           "must fix or restart qwen-web"))
        # Final leak guard: no configured runtime path value may appear in
        # any message, whatever upstream produced it.
        path_values = [v for v in runtime.path_fields().values() if v]
        if any(v in message for _, message in public for v in path_values):
            public = [("server_runtime_configuration",
                       "one or more server runtime paths configured at "
                       "startup no longer validate; a server operator "
                       "must fix or restart qwen-web")]
            code = "server_runtime_configuration"
        raise ApiError(422, code, "configuration is invalid",
                       tuple(public)) from None


def _seed_wire_to_int(value: Any) -> int | None:
    """Decimal-string/null seed wire -> Python int, typed. Numbers, bools,
    over-long digit strings and any non-ASCII digits are refused — never
    coerced through a float and never rounded."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str):
        raise ApiError(400, "invalid_seed",
                       "seed must be a decimal string (at most 20 ASCII "
                       "digits) or null — numeric JSON is refused because "
                       "JavaScript numbers would round past 2^53")
    if not value:
        raise ApiError(400, "invalid_seed",
                       "seed string must contain at least one ASCII digit")
    if _SEED_WIRE_RE.fullmatch(value):
        return int(value)
    raise ApiError(400, "invalid_seed",
                    "seed must be at most 20 ASCII decimal digits "
                    "(0 to 18446744073709551615); the text was refused "
                    "exactly as entered")


def _public_seed(config: GenerationConfig) -> str | None:
    """Public seed wire: decimal STRING or null — never numeric JSON."""
    if config.seed is None:
        return None
    return str(int(config.seed))


def editable_config_dict(config: GenerationConfig) -> dict[str, Any]:
    """The editable subset of a validated config with the public seed wire
    (decimal string / null). This is what plan echoes back.

    Deferred fields are dropped for the same reason runtime paths are: the
    wire is only ever what the browser can send back. Echoing a field that
    ``build_config`` refuses would break export -> import round-trips with a
    config the server itself produced.
    """
    out = config.to_dict()
    for name in (*RUNTIME_PATH_FIELDS, *DEFERRED_WIRE_FIELDS):
        out.pop(name, None)
    out["seed"] = _public_seed(config)
    out["references"] = list(config.references)
    return out


# ---------------------------------------------------------------------------
# Manifest metadata (never verification — no streaming hashes for the UI)
# ---------------------------------------------------------------------------

#: Stable reason codes for the manifest gate.
MANIFEST_UNCONFIGURED = "manifest_unconfigured"
MANIFEST_FILE_MISSING = "manifest_file_missing"
MANIFEST_INVALID = "manifest_invalid"
MANIFEST_UNREADABLE = "manifest_unreadable"


def summarize_manifest_bytes(raw: bytes | None, unreadable: str | None = None) -> dict[str, Any]:
    """Metadata-only summary of the (already bounded) manifest file bytes.

    Parses with the existing strict :func:`manifest_from_dict` — no
    ``resolve_manifest`` / ``manifest_status`` anywhere near the UI path,
    because verification would stream-hash every weight. Reports identity
    *counts and revisions* only: never filenames, sha256 hashes or paths.
    ``unreadable`` is the typed read-failure code from the server's one-shot
    bounded read (missing / not a regular file / too large / ...).
    """
    base: dict[str, Any] = {"configured": True, "valid": False,
                            "verification": "not_performed"}
    if unreadable is not None:
        # CLOSED reason codes only: whatever the caller's string was, an
        # arbitrary value (which could embed a path) never reaches the wire.
        base["reason"] = unreadable if unreadable in (
            MANIFEST_FILE_MISSING, MANIFEST_INVALID, MANIFEST_UNREADABLE,
            MANIFEST_UNCONFIGURED) else MANIFEST_UNREADABLE
        return base
    if raw is None:
        return {"configured": False, "valid": False, "reason": MANIFEST_UNCONFIGURED,
                "verification": "not_performed"}
    from qwen_engine.manifest import ManifestFormatError, manifest_from_dict
    try:
        manifest = manifest_from_dict(_jsonsafe.strict_json_loads(raw.decode("utf-8")))
    except (ManifestFormatError, UnicodeDecodeError, ValueError, RecursionError):
        base["reason"] = MANIFEST_INVALID
        return base
    return {
        "configured": True,
        "valid": True,
        "schema_version": manifest.schema_version,
        "artifact_count": len(manifest.artifacts),
        "revisions": sorted({a.revision for a in manifest.artifacts}),
        "verification": "not_performed",
    }


def _manifest_core(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Metadata-only manifest block shared by status and plan: never
    filenames, hashes or paths, and always verification:"not_performed"."""
    state: dict[str, Any] = {
        "configured": summary.get("configured") is True,
        "valid": False,
        "verification": "not_performed",
    }
    schema, count = summary.get("schema_version"), summary.get("artifact_count")
    state["valid"] = (state["configured"] and summary.get("valid") is True
                      and type(schema) is int and schema == 1
                      and type(count) is int and count > 0)
    if state["valid"]:
        state.update(schema_version=schema, artifact_count=count)
    else:
        reason = summary.get("reason")
        reasons = (MANIFEST_UNCONFIGURED, MANIFEST_FILE_MISSING,
                   MANIFEST_INVALID, MANIFEST_UNREADABLE)
        state["reason"] = (reason if isinstance(reason, str) and reason in reasons
                           else MANIFEST_INVALID if state["configured"]
                           else MANIFEST_UNCONFIGURED)
    return state


def manifest_plan_state(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The plan's explicit metadata-only manifest state: NOT verified, fit
    not evaluated, never 'ready' — an invalid/unconfigured manifest is a
    blocker, but planning itself still proceeds."""
    state = _manifest_core(summary)
    state["fit"] = "not_evaluated"
    state["ready"] = False
    return state


def _manifest_blocker(summary: Mapping[str, Any]) -> str | None:
    state = _manifest_core(summary)
    if not state["configured"]:
        return "manifest_unconfigured"
    if not state["valid"]:
        reason = state["reason"]
        return {"manifest_file_missing": MANIFEST_FILE_MISSING,
                MANIFEST_INVALID: MANIFEST_INVALID,
                MANIFEST_UNREADABLE: MANIFEST_UNREADABLE}.get(reason, MANIFEST_INVALID)
    return None


#: Capabilities/evidence, separate from static submission blockers.
def _gates() -> dict[str, dict[str, str]]:
    return {
        "adapter": {
            "state": "available",
            "code": "adapter_available",
            "reason": "the FP16 resident Qwen adapter is implemented; actual "
                      "load and resource checks run inside each job",
        },
        "hardware": {
            "state": "unknown",
            "code": "hardware_not_probed",
            "reason": "status/plan do not probe hardware; each actual job "
                      "validates its device, so submission availability is "
                      "not hardware verification",
        },
        "assets": {
            "state": "not_verified",
            "code": "assets_unverified",
            "reason": "artifact verification is not performed by the status "
                      "or plan path (it would stream-hash every weight); "
                      "verification belongs to an actual run",
        },
        "calibration": {
            "state": "unknown",
            "code": "calibration_unknown",
            "reason": "calibration is not VERIFIED for this server: this "
                      "status/plan path does not inspect any benchmark "
                      "records; memory fitness is unknown here, never "
                      "claimed established or absent",
        },
    }


def _blockers(runtime: WebRuntime) -> list[str]:
    """Static submission blockers, never a claim of measured memory fitness.

    Hardware/assets are checked by the actual engine job, just as in the CLI.
    Exact calibration preflight remains a separate, unimplemented integration.
    """
    codes = [] if runtime.generation_enabled else ["generation_disabled"]
    if any(getattr(runtime, name) is None for name in
           ("output_dir", "cache_dir", "models_dir", "manifest_path")):
        codes.append("runtime_unconfigured")
    blocker = _manifest_blocker(runtime.manifest_summary)
    if blocker:
        codes.append(blocker)
    return codes


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApiResponse:
    """Transport-ready result: status + JSON payload, or raw bytes for the
    attachment export (never re-serialized by the transport)."""

    status: int
    payload: Mapping[str, Any] | None = None
    body: bytes | None = None
    headers: tuple[tuple[str, str], ...] = ()


def status_response(runtime: WebRuntime) -> ApiResponse:
    summary = dict(runtime.manifest_summary)
    payload = {
        "schema": WEB_STATUS_SCHEMA,
        "planning_available": True,
        "generation_available": not _blockers(runtime),
        "gates": _gates(),
        "manifest": _manifest_core(summary),
        "generation_blockers": _blockers(runtime),
        # Additive and optional (plan §4.1): the schema tag is NOT bumped, and
        # nothing here participates in the generation gate.
        "rewrite": rewrite_status_block(runtime),
    }
    return ApiResponse(status=200, payload=payload)


def _plan_core_document(runtime: WebRuntime, config: GenerationConfig) -> dict[str, Any]:
    """core.plan runs on the MERGED config; the raw document contains runtime
    paths and a numeric seed, so it is only ever an internal input to the
    WEB PROJECTION below — never returned verbatim."""
    return core_plan(config, manifest=dict(runtime.manifest_summary))


def plan_response(runtime: WebRuntime, raw_body: bytes) -> ApiResponse:
    try:
        config = build_config(runtime, parse_request_json(raw_body))
    except ApiError as exc:
        return ApiResponse(status=exc.status, payload=exc.to_payload())
    core_doc = _plan_core_document(runtime, config)
    payload = {
        "schema": WEB_PLAN_SCHEMA,
        "planning_available": True,
        "generation_available": not _blockers(runtime),
        "config": editable_config_dict(config),
        "cfg_active": core_doc["cfg_active"],
        "latent_tokens": core_doc["latent_tokens"],
        "hardware": {"status": core_doc["hardware"].get("status", "unknown"),
                     "detail": "this qwen-web server has not probed hardware; "
                               "unknown here, never claimed present, absent "
                               "or verified"},
        "calibration": {"status": core_doc["calibration"].get("status", "unknown"),
                        "detail": "calibration is not verified for this "
                                  "server; the status/plan path does not "
                                  "inspect benchmark records"},
        "manifest": manifest_plan_state(runtime.manifest_summary),
        "generation_blockers": _blockers(runtime),
    }
    return ApiResponse(status=200, payload=payload)


def export_response(runtime: WebRuntime, raw_body: bytes) -> ApiResponse:
    """Raw-attachment export of the browser-editable GenerationConfig
    fields. Runtime paths are OMITTED and the seed is the EXACT uint64
    Python number in JSON — these bytes are downloaded as a blob and must
    never be JSON-parsed/reserialized by the browser."""
    try:
        config = build_config(runtime, parse_request_json(raw_body))
    except ApiError as exc:
        return ApiResponse(status=exc.status, payload=exc.to_payload())
    out = config.to_dict()
    for name in (*RUNTIME_PATH_FIELDS, *DEFERRED_WIRE_FIELDS):
        out.pop(name, None)
    body = (json.dumps(out, sort_keys=True) + "\n").encode("utf-8")
    return ApiResponse(
        status=200, body=body,
        headers=(("Content-Disposition",
                  'attachment; filename="qwen-editable-config.json"'),),
    )


def import_response(runtime: WebRuntime, raw_body: bytes) -> ApiResponse:
    """Validate an editable export without rounding its numeric uint64 seed.

    Files contain raw JSON config; Python parses integers exactly before the
    ordinary browser envelope validator rejects runtime/unknown fields.
    """
    try:
        data = parse_request_json(raw_body)
        if not isinstance(data, Mapping):
            raise ApiError(400, "invalid_config_file", "configuration file must contain a JSON object")
        config = dict(data)
        if type(config.get("seed")) is int:
            config["seed"] = str(config["seed"])
        envelope = {"schema": WEB_PLAN_REQUEST_SCHEMA, "config": config}
        # Reuse plan projection so the browser has one authoritative config
        # parser, with paths omitted and seeds always decimal strings.
        return plan_response(runtime, json.dumps(envelope, ensure_ascii=False).encode("utf-8"))
    except ApiError as exc:
        return ApiResponse(exc.status, exc.to_payload())


def upload_reference_response(runtime: WebRuntime, raw: bytes) -> ApiResponse:
    """Store one raw uploaded reference image (design §2.4).

    Validation is the store's (magic sniff, Pillow decode, size/dimension
    caps, store fullness); every refusal is typed and prompt-free: 415 for
    non-images, 409 when the store is at its cap, 503 when the server has no
    usable store — never a path, never a library message. Idempotent: the
    same bytes return the same id.
    """
    if runtime.references_dir is None:
        raise ApiError(503, "reference_store_unconfigured",
                       "the server reference store directory is not configured")
    if not raw:
        raise ApiError(415, "unsupported_media_type",
                       "an empty body is not an image")
    try:
        # The server-owned store directory: created 0700 at startup by the
        # server, re-created here if an operator removed it mid-run (uploads
        # are the only writers; content-addressed, so no clobbering).
        os.makedirs(runtime.references_dir, mode=0o700, exist_ok=True)
    except OSError:
        raise ApiError(503, "reference_store_unavailable",
                       "the reference store directory is not usable") from None
    try:
        reference_id = _reference_store.put_bytes(runtime.references_dir, raw)
        sidecar = _reference_store.read_sidecar(runtime.references_dir, reference_id)
    except _reference_store.ReferenceStoreError as exc:
        text = str(exc)
        if text.startswith(("cannot publish", "reference root is not")):
            raise ApiError(503, "reference_store_unavailable",
                           "the reference store could not accept the upload") from None
        if "store is full" in text:
            raise ApiError(409, "reference_store_full",
                           "the reference store is at its cap; an operator must "
                           "prune it") from None
        raise ApiError(415, "unsupported_media_type",
                       "the body must be a valid PNG, JPEG or WebP image "
                       "within the size and dimension caps") from None
    return ApiResponse(201, {"schema": WEB_REFERENCE_SCHEMA, "id": reference_id,
                             "width": sidecar["width"], "height": sidecar["height"],
                             "has_alpha": bool(sidecar["has_alpha"]),
                             "bytes": len(raw)})


def reference_bytes(runtime: WebRuntime, reference_id: str, *,
                    thumb: bool = False) -> tuple[bytes, str]:
    """Hash-verified stored bytes (original or PNG thumbnail) + sniffed
    Content-Type. Unknown/malformed ids and unreadable blobs are ONE 404 —
    the client learns nothing beyond absence."""
    if runtime.references_dir is None:
        raise ApiError(404, "reference_unknown", "no such reference")
    try:
        data = (_reference_store.read_thumb(runtime.references_dir, reference_id)
                if thumb else
                _reference_store.read_blob(runtime.references_dir, reference_id))
    except _reference_store.ReferenceStoreError:
        raise ApiError(404, "reference_unknown", "no such reference") from None
    if thumb:
        return data, "image/png"
    try:
        return data, _REFERENCE_CONTENT_TYPES[_reference_store.sniff_format(data)]
    except _reference_store.ReferenceStoreError:
        raise ApiError(404, "reference_unknown", "no such reference") from None


def build_rewrite_request(raw: bytes) -> dict[str, Any]:
    """``/api/v1/rewrite`` body -> the exact four-key request mapping.

    The wire envelope carries EXACTLY
    ``schema``/``task``/``prompt``/``draft_revision``/``image_ids``, so a typo
    is a typed refusal rather than a silently ignored field. Envelope shape is
    checked here; field SEMANTICS live in :func:`validate_rewrite_fields`,
    which the service calls too — one implementation, so the transport and the
    executor cannot disagree about what a legal request is.

    The prompt is validated for shape and size only; it is never logged, never
    echoed into an error message, and never copied anywhere but the single
    ``original_prompt`` field of a successful response.
    """
    data = parse_request_json(raw)
    if not isinstance(data, Mapping):
        raise ApiError(400, "invalid_envelope",
                       "the rewrite request must be a JSON object")
    if any(not isinstance(key, str) for key in data):
        raise ApiError(400, "invalid_envelope", "request keys must be strings")
    allowed = ("schema", "task", "prompt", "draft_revision", "image_ids")
    missing = sorted(set(allowed) - set(data))
    extra = sorted(set(data) - set(allowed))
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing key(s): " + ", ".join(missing))
        if extra:
            parts.append("unexpected key(s): " + ", ".join(extra))
        raise ApiError(400, "invalid_envelope", "; ".join(parts))
    if data["schema"] != REWRITE_REQUEST_SCHEMA:
        raise ApiError(400, "invalid_envelope",
                       f"schema must be {REWRITE_REQUEST_SCHEMA!r}")
    return validate_rewrite_fields({k: v for k, v in data.items() if k != "schema"})


def validate_rewrite_fields(request: Mapping[str, Any]) -> dict[str, Any]:
    """Semantic validation of the four request fields, pure.

    Shared by the transport (after the envelope check) and by
    ``RewriteService.submit`` (which is also called directly by tests), so
    there is exactly ONE definition of a legal rewrite request.
    """
    if not isinstance(request, Mapping):
        raise ApiError(400, "invalid_envelope",
                       "the rewrite request must be a JSON object")
    allowed = {"task", "prompt", "draft_revision", "image_ids"}
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ApiError(400, "invalid_envelope",
                       "unknown rewrite field(s): "
                       + ", ".join(map(str, unknown)))
    missing = sorted(allowed - set(request))
    if missing:
        raise ApiError(400, "invalid_envelope",
                       "missing rewrite field(s): " + ", ".join(missing))
    try:
        task = _rewrite.check_task(request["task"])
    except _rewrite.RewriteError as exc:
        raise ApiError(422, "task_unsupported", str(exc)) from None
    image_ids = request["image_ids"]
    if not isinstance(image_ids, list) or any(
            isinstance(i, bool) or not isinstance(i, int) for i in image_ids):
        raise ApiError(400, "invalid_envelope",
                       "image_ids must be a list of integers")
    if image_ids:
        # Reference-image rewriting is gated on edit support landing.
        raise ApiError(422, "task_unsupported",
                       "rewriting with reference images is not supported yet")
    revision = request["draft_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ApiError(400, "invalid_envelope",
                       "draft_revision must be a non-negative integer")
    prompt = request["prompt"]
    if not isinstance(prompt, str):
        raise ApiError(400, "invalid_envelope", "prompt must be a string")
    if len(prompt) > _rewrite.PROMPT_MAX_CHARS:
        # Length is the only prompt-derived fact ever put in a message.
        raise ApiError(422, "prompt_too_long",
                       f"the prompt is {len(prompt)} characters; the limit is "
                       f"{_rewrite.PROMPT_MAX_CHARS}")
    if not prompt.strip():
        raise ApiError(400, "invalid_envelope", "prompt must be non-empty")
    return {"task": task, "prompt": prompt, "draft_revision": revision,
            "image_ids": []}


#: The canonical "no rewriter here" block. One source of truth: an
#: unconfigured server, a disabled flag and a failed service all report this
#: exact shape, so the UI's strict parser has one thing to recognise.
REWRITE_DISABLED_BLOCK: Mapping[str, Any] = {
    "state": "unavailable",
    "code": "rewrite_disabled",
    "backend": None,
    "reason": "rewriting is off by default; start the server with "
              "--rewrite-backend local-llm to enable it",
}


def rewrite_status_block(runtime: WebRuntime) -> dict[str, Any]:
    """The additive, OPTIONAL ``rewrite`` status block (plan §4.1).

    Four keys, closed value sets, never a host, port or path: this document
    is served to the browser, and the reason string is operator-facing prose
    that must not disclose where the backend lives. ``GATE_STATES`` does not
    grow — rewriting is an optional capability and must never block (or
    appear to block) generation.

    ``WebRuntime`` already refuses any block that is not legal, so this is a
    copy, not a filter; :func:`validate_rewrite_block` is that single rule.
    """
    block = runtime.rewrite_block
    if block is None:
        return dict(REWRITE_DISABLED_BLOCK)
    return dict(block)


def internal_error_response() -> ApiResponse:
    """Catch-all typed failure: no traceback, no exception text, no paths."""
    exc = ApiError(500, "internal_error",
                   "the server failed to handle the request; check the "
                   "server console for a bounded diagnostic")
    return ApiResponse(status=500, payload=exc.to_payload())


__all__ = [
    "EDITABLE_FIELDS", "RUNTIME_PATH_FIELDS", "ApiError", "ApiResponse",
    "WEB_ERROR_SCHEMA", "WEB_PLAN_REQUEST_SCHEMA", "WEB_PLAN_SCHEMA",
    "WEB_STATUS_SCHEMA", "build_config", "editable_config_dict",
    "export_response", "internal_error_response", "manifest_plan_state",
    "parse_request_json", "plan_response", "status_response",
    "summarize_manifest_bytes",
    "MAX_REFERENCE_UPLOAD_BYTES", "WEB_REFERENCE_SCHEMA",
    "reference_bytes", "upload_reference_response",
    # Prompt rewriting (plan §4.1): the wire contract, not the transport.
    "REWRITE_REQUEST_SCHEMA", "WEB_REWRITE_SCHEMA",
    "MAX_REWRITE_REQUEST_BYTES", "REWRITE_DISABLED_BLOCK",
    "build_rewrite_request", "validate_rewrite_fields", "validate_rewrite_block",
    "rewrite_status_block",
]


# Guard against accidental field drift between config.py and this contract.
def _check_editable_subset() -> None:
    names = {f.name for f in fields(GenerationConfig)}
    assert EDITABLE_FIELDS <= names, "EDITABLE_FIELDS references unknown config fields"
    assert not (EDITABLE_FIELDS & set(RUNTIME_PATH_FIELDS)), \
        "a field cannot be both browser-editable and server-controlled"
    assert names - EDITABLE_FIELDS - set(RUNTIME_PATH_FIELDS) == set(DEFERRED_WIRE_FIELDS), \
        ("every GenerationConfig field must be browser-editable, a runtime "
         "path, or an announced deferral — unclassified: "
         f"{sorted(names - EDITABLE_FIELDS - set(RUNTIME_PATH_FIELDS) - set(DEFERRED_WIRE_FIELDS))}")
    assert DEFERRED_WIRE_FIELDS <= names, \
        f"DEFERRED_WIRE_FIELDS names unknown config fields: {sorted(DEFERRED_WIRE_FIELDS - names)}"


_check_editable_subset()
