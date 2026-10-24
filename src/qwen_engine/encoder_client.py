"""Production CPU encoder boundary: fresh child identity BEFORE cache access.

Stdlib-only parent; fixed module commands, no generation/web activation.
Projection makes no memory/asset claim. On a miss the separate loading child
still enforces actual memory, package identity and fresh narrow asset gates.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from qwen_engine._jsonsafe import plain_json
from qwen_engine.config import GenerationConfig
from qwen_engine.encoder import (
    EncodedPayload, EmbeddingCache, EncoderProtocolError, EncoderRequest,
    EncoderRequestError, EncoderSetupError, EncoderWorker, cache_key,
)
from qwen_engine.encoder_identity import project_encoder_seed, validate_completed_seed
from qwen_engine.encoder_production import validate_production_request, validate_serialized_payload
from qwen_engine.encoder_worker import MAX_REQUEST_BYTES, MAX_PROJECTION_RESPONSE_BYTES, PROJECTION_SCHEMA
from qwen_engine.log import PhaseLogger
from qwen_engine.manifest import load_manifest


def _absolute_path(value, field):
    if not isinstance(value, (str, Path)):
        raise EncoderSetupError(f"{field} must be a local path string or Path")
    raw = os.fspath(value)
    if not raw.strip() or "\0" in raw:
        raise EncoderSetupError(f"{field} must be nonblank without NUL")
    # Lexical only: choosing a model root must not access it before memory fit.
    return Path(os.path.abspath(raw))


def _request_budget(request, rid):
    # Match generic runner's current wire serialization, including newline.
    raw = (json.dumps(request.to_request_dict(rid), sort_keys=True,
                      ensure_ascii=True) + "\n").encode("utf-8")
    if len(raw) > MAX_REQUEST_BYTES:
        raise EncoderRequestError("production encoder request exceeds the worker input byte cap")


def _completed_request(payload, seed, rid, manifest):
    if (not isinstance(payload, EncodedPayload) or payload.blobs
            or set(payload.metadata) != {"schema", "request"}
            or payload.metadata["schema"] != PROJECTION_SCHEMA):
        raise EncoderProtocolError("invalid metadata-only identity projection response")
    try:
        request, response_id = EncoderRequest.from_request_dict(plain_json(payload.metadata["request"]))
        if response_id != rid:
            raise EncoderRequestError("projection request_id disagrees with the current request")
        validate_completed_seed(seed, request, manifest)
        _request_budget(request, rid)
    except EncoderRequestError as exc:
        raise EncoderProtocolError(f"invalid completed projection request: {exc}") from exc
    return request


@dataclass(frozen=True)
class ProductionEncoding:
    """Opaque payload plus the exact fresh request needed for tensor decoding.

    Structural/identity validation only; callers still use payload_to_branches
    with expect_request=request for actual tensor/mask/finiteness validation.
    """
    request: EncoderRequest
    payload: EncodedPayload

    def __post_init__(self):
        validate_production_request(self.request)
        validate_serialized_payload(self.payload, expect_request=self.request)


class ProductionEncoder:
    """Cache-aware encoder with a freshly projected selected-child identity.

    No caller-supplied command, revision override or runtime-policy bypass.
    No cache access (even lock creation) until projection succeeds. Cached
    bytes are structurally validated before returning; misses are validated
    before publication. Semantic corruption is a typed refusal, not overwrite
    or automatic fallback. Generic corruption retains its quarantine behavior.
    """
    def __init__(self, *, manifest_path, models_dir, headroom_bytes: int,
                 timeout_s: float, scratch_dir, python: str | None = None,
                 cache: EmbeddingCache | None = None, projection_timeout_s: float = 10.0,
                 cancel_grace_s: float = 5.0, logger: PhaseLogger | None = None,
                 attested_unbounded_host: bool = False):
        if type(headroom_bytes) is not int or headroom_bytes < 0:
            raise EncoderSetupError("headroom_bytes must be an explicit nonnegative integer")
        if cache is not None and not isinstance(cache, EmbeddingCache):
            raise EncoderSetupError("cache must be an EmbeddingCache or None")
        if not isinstance(attested_unbounded_host, bool):
            raise EncoderSetupError("attested_unbounded_host must be an explicit boolean")
        chosen = sys.executable if python is None else python
        if not isinstance(chosen, str) or not chosen.strip() or "\0" in chosen:
            raise EncoderSetupError("python must be a nonblank interpreter path/name")
        chosen = os.path.abspath(chosen) if os.sep in chosen else shutil.which(chosen)
        if chosen is None:
            raise EncoderSetupError("selected Python interpreter was not found")
        #: Last cache decision of this encoder instance (None until encoded).
        #: Only an observation aid for the measured-memory collector; it never
        #: changes cache behaviour or the returned payload.
        self.last_cache_hit: bool | None = None
        self.manifest_path = _absolute_path(manifest_path, "manifest_path")
        self.models_dir = _absolute_path(models_dir, "models_dir")
        scratch_dir = _absolute_path(scratch_dir, "scratch_dir")
        self.cache = cache
        common = dict(python=chosen, scratch_dir=scratch_dir,
                      cancel_grace_s=cancel_grace_s, logger=logger)
        command = ["{python}", "-B", "-m", "qwen_engine.encoder_worker",
                   "--request", "{request}", "--response", "{response}",
                   "--manifest", str(self.manifest_path)]
        self._projector = EncoderWorker(command=[*command, "--project-identity"],
            timeout_s=projection_timeout_s, max_response_bytes=MAX_PROJECTION_RESPONSE_BYTES, **common)
        try:
            headroom_argument = str(headroom_bytes)
        except ValueError as exc:
            raise EncoderSetupError("headroom_bytes is not representable as a CLI argument") from exc
        loading_flags = ["--models-dir", str(self.models_dir),
                         "--headroom-bytes", headroom_argument]
        if attested_unbounded_host:
            loading_flags.append("--attest-unbounded-host")
        self._loader = EncoderWorker(command=[*command, *loading_flags],
                                     timeout_s=timeout_s, **common)

    def encode(self, config: GenerationConfig, *, encoder_dtype: str,
               cancel_event: threading.Event | None = None) -> ProductionEncoding:
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise EncoderRequestError("cancel_event must be a threading.Event or None")
        manifest = load_manifest(self.manifest_path)
        seed = project_encoder_seed(config, manifest, encoder_dtype=encoder_dtype)
        rid = uuid.uuid4().hex
        _request_budget(seed, rid)
        projected = self._projector.run(seed, request_id=rid, cancel_event=cancel_event)
        request = _completed_request(projected, seed, rid, manifest)
        self._loader._event("production_encoder_identity", cache_key=cache_key(request),
                            request_id=rid, python=self._projector.python)
        self.last_cache_hit = None
        if self.cache is None:
            # No cache configured behaves as a miss: the child encoder runs.
            self.last_cache_hit = False
            return ProductionEncoding(request, self._loader.run(request, request_id=rid, cancel_event=cancel_event))
        key = cache_key(request)
        with self.cache.lock_for(key):
            payload = self.cache.get(key)
            if payload is not None:
                self._loader._event("encoder_cache_hit", cache_key=key)
                self.last_cache_hit = True
                return ProductionEncoding(request, payload)
            self._loader._event("encoder_cache_miss", cache_key=key)
            result = ProductionEncoding(request, self._loader.run(request, request_id=rid, cancel_event=cancel_event))
            self.cache.put(key, result.payload)
            self.last_cache_hit = False
            return result
