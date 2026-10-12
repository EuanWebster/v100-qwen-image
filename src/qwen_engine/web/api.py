"""Pure W2 web contract: status / plan / editable-config-export (no sockets).

Every function here is deterministic and ML-free; the RESPONSE path performs
no filesystem I/O. (Startup validation and the shared config validator may
stat configured path metadata — directories, never weight bytes; a plan or
status call itself reads nothing.) The only filesystem knowledge enters
through :class:`WebRuntime`, which the server validates and canonicalises
once at startup. Browser JSON can never select a runtime path — unknown
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
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from qwen_engine import _jsonsafe
from qwen_engine.config import SEED_MAX, ConfigError, GenerationConfig
from qwen_engine.core import plan as core_plan

#: Wire schema tags (versioned like every other qwen_engine document).
WEB_STATUS_SCHEMA = "qwen_engine.web_status.v1"
WEB_PLAN_SCHEMA = "qwen_engine.web_plan.v1"
WEB_PLAN_REQUEST_SCHEMA = "qwen_engine.web_plan_request.v1"
WEB_ERROR_SCHEMA = "qwen_engine.web_error.v1"

#: The browser-editable GenerationConfig fields — exactly these, no more.
#: Runtime paths (output/cache/log/benchmarks/models/manifest) are
#: server-controlled and must never arrive from browser JSON.
EDITABLE_FIELDS = frozenset({
    "prompt", "negative_prompt", "width", "height", "steps",
    "true_cfg_scale", "seed", "quant", "offload", "attention_backend",
    "vae_tiled_decode",
})

#: Server-controlled runtime path fields (never editable from the browser).
RUNTIME_PATH_FIELDS = ("output_dir", "cache_dir", "log_dir",
                       "benchmarks_dir", "models_dir", "manifest_path")

#: Seed wire grammar: ASCII decimal digits only, at most 20 characters
#: (the full uint64 range stays an exact string; range is validated later
#: against ``config.SEED_MAX``). Leading zeros are allowed; whitespace,
#: signs, exponents and non-ASCII digits are not.
_SEED_WIRE_RE = re.compile(r"[0-9]{1,20}\Z")

#: Bound on any single public error message (safe, single-line, truncated).
MAX_MESSAGE_CHARS = 200


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
    manifest_summary: Mapping[str, Any] = field(default_factory=dict)

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
    (decimal string / null). This is what plan echoes back."""
    out = config.to_dict()
    for name in RUNTIME_PATH_FIELDS:
        out.pop(name, None)
    out["seed"] = _public_seed(config)
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
        manifest = manifest_from_dict(json.loads(raw.decode("utf-8")))
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
        "configured": bool(summary.get("configured", False)),
        "valid": bool(summary.get("valid", False)),
        "verification": "not_performed",
    }
    if state["valid"]:
        for key in ("schema_version", "artifact_count"):
            if key in summary:
                state[key] = summary[key]
    else:
        state["reason"] = summary.get("reason", MANIFEST_UNCONFIGURED)
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
    if not summary.get("configured", False):
        return "manifest_unconfigured"
    if not summary.get("valid", False):
        reason = summary.get("reason", MANIFEST_INVALID)
        return {"manifest_file_missing": MANIFEST_FILE_MISSING,
                MANIFEST_INVALID: MANIFEST_INVALID,
                MANIFEST_UNREADABLE: MANIFEST_UNREADABLE}.get(reason, MANIFEST_INVALID)
    return None


#: The honest W2 gates: every one is explicitly NOT satisfied.
def _gates() -> dict[str, dict[str, str]]:
    return {
        "adapter": {
            "state": "not_implemented",
            "code": "adapter_not_implemented",
            "reason": "no pipeline adapter is implemented (M4+); generation "
                      "would fail before touching a model",
        },
        "hardware": {
            "state": "unknown",
            "code": "hardware_not_probed",
            "reason": "no hardware probe has run for this server; the GPU is "
                      "unknown here — not claimed present, absent or verified",
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
    codes = [gate["code"] for gate in _gates().values()]
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
        "generation_available": False,
        "gates": _gates(),
        "manifest": _manifest_core(summary),
        "generation_blockers": _blockers(runtime),
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
        "generation_available": False,
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
    for name in RUNTIME_PATH_FIELDS:
        out.pop(name, None)
    body = (json.dumps(out, sort_keys=True) + "\n").encode("utf-8")
    return ApiResponse(
        status=200, body=body,
        headers=(("Content-Disposition",
                  'attachment; filename="qwen-editable-config.json"'),),
    )


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
]


# Guard against accidental field drift between config.py and this contract.
def _check_editable_subset() -> None:
    names = {f.name for f in fields(GenerationConfig)}
    assert EDITABLE_FIELDS <= names, "EDITABLE_FIELDS references unknown config fields"
    assert names - EDITABLE_FIELDS == set(RUNTIME_PATH_FIELDS), \
        "editable + runtime-path fields must partition GenerationConfig fields"


_check_editable_subset()
