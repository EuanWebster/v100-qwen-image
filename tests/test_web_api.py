"""Guarded W2 tests: the pure web API contract (no sockets, no ML, no writes).

Everything here calls the pure dispatch functions directly with bytes — the
real-socket smoke test is the SEPARATE integration command
(``scripts/web_integration_check.mjs``), never this guarded suite.
"""
from __future__ import annotations

import json

import pytest

from qwen_engine.config import SEED_MAX
from qwen_engine.web import api

ENVELOPE = api.WEB_PLAN_REQUEST_SCHEMA


def runtime(**overrides) -> api.WebRuntime:
    summary = overrides.pop("manifest_summary", None)
    if summary is None:
        summary = api.summarize_manifest_bytes(None)
    return api.WebRuntime(manifest_summary=summary, **overrides)


def plan_payload(body: bytes, rt: api.WebRuntime | None = None) -> dict:
    return api.plan_response(rt or runtime(), body).payload


def plan_bytes(config: dict) -> bytes:
    return json.dumps({"schema": ENVELOPE, "config": config}).encode()


def error_shape(payload: dict) -> None:
    assert payload["schema"] == api.WEB_ERROR_SCHEMA
    assert isinstance(payload["code"], str) and payload["code"]
    assert isinstance(payload["message"], str)
    assert len(payload["message"]) <= api.MAX_MESSAGE_CHARS + 1
    assert "\n" not in payload["message"]
    text = json.dumps(payload)
    for forbidden in ("Traceback", "File \"", ".py", "qwen_engine/web"):
        assert forbidden not in text, f"internal detail leaked: {forbidden}"


# ---------------------------------------------------------------- status


def test_status_schema_booleans_and_explicit_gates():
    resp = api.status_response(runtime())
    assert resp.status == 200
    s = resp.payload
    assert s["schema"] == "qwen_engine.web_status.v1"
    assert s["planning_available"] is True
    assert s["generation_available"] is False
    codes = {g["code"] for g in s["gates"].values()}
    assert codes == {"adapter_available", "hardware_not_probed",
                     "assets_unverified", "calibration_unknown"}
    assert s["gates"]["adapter"]["state"] == "available"
    assert s["gates"]["hardware"]["state"] == "unknown"   # NOT absent/verified
    assert set(s["generation_blockers"]) == {"generation_disabled", "runtime_unconfigured", "manifest_unconfigured"}
    # metadata-only manifest state, never verification
    assert s["manifest"]["verification"] == "not_performed"


def test_status_never_leaks_paths_or_hashes(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version": 1, "artifacts": []}', encoding="utf-8")
    rt = runtime(output_dir=str(tmp_path / "out"), manifest_path=str(manifest),
                 manifest_summary=api.summarize_manifest_bytes(
                     manifest.read_bytes()))
    text = json.dumps(api.status_response(rt).payload)
    assert str(tmp_path) not in text
    assert "out" not in json.dumps(api.status_response(rt).payload["manifest"])


# ---------------------------------------------------------------- plan

def test_plan_happy_path_projection_and_seed_string():
    payload = plan_payload(plan_bytes({"prompt": "a fox",
                                       "seed": "18446744073709551615"}))
    assert payload["schema"] == "qwen_engine.web_plan.v1"
    assert payload["planning_available"] is True
    assert payload["generation_available"] is False
    assert payload["config"]["seed"] == "18446744073709551615"  # STRING
    assert isinstance(payload["config"]["seed"], str)
    assert payload["latent_tokens"] == (1024 // 16) ** 2
    assert payload["cfg_active"] is False
    assert payload["manifest"]["ready"] is False
    assert payload["manifest"]["verification"] == "not_performed"
    assert payload["manifest"]["fit"] == "not_evaluated"
    assert payload["hardware"]["status"] == "unknown"
    assert payload["calibration"]["status"] == "unknown"
    assert "generation_disabled" in payload["generation_blockers"]


def test_plan_token_math_and_cfg():
    p = plan_payload(plan_bytes({"prompt": "x", "width": 2048, "height": 2048}))
    assert p["latent_tokens"] == 16384
    p = plan_payload(plan_bytes({"prompt": "x", "negative_prompt": "bad",
                                 "true_cfg_scale": 4.0}))
    assert p["cfg_active"] is True


def test_plan_config_echo_is_editable_only():
    payload = plan_payload(plan_bytes({"prompt": "x"}))
    assert set(payload["config"]) == set(api.EDITABLE_FIELDS)
    for path_field in api.RUNTIME_PATH_FIELDS:
        assert path_field not in json.dumps(payload)


def test_plan_echoes_transparent_background_and_refuses_jpg():
    ok = plan_payload(plan_bytes({"prompt": "x", "transparent_background": True,
                                  "output_format": "webp"}))
    assert ok["config"]["transparent_background"] is True
    assert ok["config"]["output_format"] == "webp"
    # Absent from the envelope: defaults to false, exactly as config.py does.
    assert plan_payload(plan_bytes({"prompt": "x"}))["config"]["transparent_background"] is False
    # JPEG cannot store alpha: a typed FIELD error, never a silent rewrite.
    refused = plan_payload(plan_bytes({"prompt": "x", "transparent_background": True,
                                       "output_format": "jpg"}))
    error_shape(refused)
    assert refused["code"] == "invalid_config"
    assert [f["field"] for f in refused["fields"]] == ["output_format"]
    # A non-boolean is a typed field error on its own field, not a coercion.
    typed = plan_payload(plan_bytes({"prompt": "x", "transparent_background": 1}))
    assert typed["code"] == "invalid_config"
    assert [f["field"] for f in typed["fields"]] == ["transparent_background"]


def test_export_import_roundtrips_transparent_background():
    rt = runtime()
    exported = api.export_response(rt, plan_bytes({"prompt": "a jar",
                                                   "transparent_background": True,
                                                   "output_format": "png"}))
    assert exported.status == 200
    data = json.loads(exported.body)
    assert data["transparent_background"] is True
    imported = api.import_response(rt, exported.body)
    assert imported.status == 200, imported.payload
    assert imported.payload["config"]["transparent_background"] is True
    assert imported.payload["config"]["output_format"] == "png"
    # Legacy export without the key stays importable and means false.
    legacy = api.import_response(rt, b'{"prompt": "old", "output_format": "webp"}')
    assert legacy.status == 200
    assert legacy.payload["config"]["transparent_background"] is False
    # A legacy file asking for jpg is still refused (no silent format change).
    refused = api.import_response(rt, b'{"prompt": "old", "transparent_background": true}')
    assert refused.status == 422
    assert [f["field"] for f in refused.payload["fields"]] == ["output_format"]


def test_plan_never_leaks_server_runtime_paths(tmp_path):
    rt = runtime(output_dir=str(tmp_path / "out"), cache_dir=str(tmp_path / "cache"))
    text = json.dumps(api.plan_response(rt, plan_bytes({"prompt": "x"})).payload)
    assert str(tmp_path) not in text


def test_plan_seed_wire_grammar():
    ok = {"18446744073709551615", "0", "42", "000042", "0" * 20}
    for s in ok:
        assert plan_payload(plan_bytes({"prompt": "x", "seed": s}))["config"]["seed"] \
            == str(int(s))
    assert plan_payload(plan_bytes({"prompt": "x", "seed": None}))["config"]["seed"] is None
    for bad in ("18446744073709551616"[:20], "9" * 21, "1e3", " 42", "4.5", "",
                "١٢٣", "-1", "0x2a", "١"):
        payload = plan_payload(plan_bytes({"prompt": "x", "seed": bad}))
        assert payload["schema"] == api.WEB_ERROR_SCHEMA, bad
        assert payload["code"] in ("invalid_seed", "invalid_config"), bad


@pytest.mark.parametrize("bad", [5, 5.0, 1.5, True, False, [1], {}, " 42"])
def test_plan_numeric_seed_never_accepted(bad):
    payload = plan_payload(plan_bytes({"prompt": "x", "seed": bad}))
    error_shape(payload)
    assert payload["code"] == "invalid_seed"


def test_plan_over_uint64_is_typed_range_error():
    payload = plan_payload(plan_bytes({"prompt": "x", "seed": "9" * 20}))
    assert payload["code"] == "invalid_config"
    assert payload["fields"][0]["field"] == "seed"
    assert int(SEED_MAX) < int("9" * 20)


def test_envelope_exact_keys():
    for body in (b'{"config":{}}',
                 b'{"schema":"' + ENVELOPE.encode() + b'"}',
                 b'{"schema":"' + ENVELOPE.encode() + b'","config":{},"extra":1}',
                 b'{"schema":"other","config":{}}',
                 b'{"schema":"' + ENVELOPE.encode() + b'","config":[]}',
                 b'[]', b'3', b'"x"', b'null'):
        payload = plan_payload(body)
        error_shape(payload)
        assert payload["code"] == "invalid_envelope", body


def test_unknown_and_path_config_fields_refused_not_stripped():
    for field in sorted(api.RUNTIME_PATH_FIELDS) + ["bogus"]:
        payload = plan_payload(plan_bytes({"prompt": "x", field: "/tmp/nope"}))
        error_shape(payload)
        assert payload["code"] == "unknown_config_field"
        assert field in payload["message"]


# ---------------------------------------------------------- hostile JSON


def test_duplicate_json_keys_rejected_recursively():
    outer = plan_payload(b'{"schema": "a", "schema": "b"}')
    assert outer["code"] == "duplicate_json_key"
    body = json.dumps({"schema": ENVELOPE, "config": {"prompt": "x"}}).encode()
    body = body.replace(b'"prompt"', b'"prompt": "y", "prompt"', 1)
    payload = plan_payload(body)
    assert payload["code"] == "duplicate_json_key"


@pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity"])
def test_nonfinite_constants_rejected(literal):
    body = ('{"schema": "' + ENVELOPE + '", "config": {"width": ').encode() + literal + b'}}'
    payload = plan_payload(body)
    assert payload["code"] == "invalid_json"


def test_float_overflow_to_infinity_rejected():
    payload = plan_payload(plan_bytes({"prompt": "x", "true_cfg_scale": None}))
    # null is typed as a config error, not an exception:
    assert payload["code"] == "invalid_config"
    body = ('{"schema": "' + ENVELOPE + '", "config": {"true_cfg_scale": 1e999}}').encode()
    payload = plan_payload(body)
    assert payload["code"] == "invalid_json"


def test_deeply_nested_json_typed():
    body = ('{"a":' * 500 + "1" + "}" * 500).encode()
    payload = plan_payload(body)
    assert payload["code"] in ("json_too_deep", "invalid_json_structure")
    error_shape(payload)


def test_oversized_int_literal_typed():
    body = (b'{"schema": "' + ENVELOPE.encode() + b'", "config": {"steps": '
            + b"9" * 5000 + b"}}")
    payload = plan_payload(body)
    assert payload["code"] in ("invalid_json", "invalid_config")
    error_shape(payload)


def test_lone_surrogate_prompt_typed():
    body = ('{"schema": "' + ENVELOPE + '", "config": {"prompt": "\\ud800"}}').encode()
    payload = plan_payload(body)
    assert payload["code"] == "invalid_text"
    error_shape(payload)


def test_invalid_utf8_typed():
    payload = plan_payload(b"\xff\xfe{\x00")
    assert payload["code"] == "invalid_utf8"
    error_shape(payload)


def test_finite_json_only_in_responses():
    payload = plan_payload(plan_bytes({"prompt": "x"}))
    text = json.dumps(payload)
    assert "Infinity" not in text and "NaN" not in text
    json.loads(text)  # round-trip proves finiteness


def test_config_validation_errors_are_typed_fields():
    payload = plan_payload(plan_bytes({"prompt": "  ", "width": 33, "steps": 0}))
    assert payload["code"] == "invalid_config"
    assert payload["schema"] == api.WEB_ERROR_SCHEMA
    fields = {f["field"] for f in payload["fields"]}
    assert {"prompt", "width", "steps"} <= fields


def test_internal_error_response_has_no_internals():
    resp = api.internal_error_response()
    assert resp.status == 500
    error_shape(resp.payload)


# ---------------------------------------------------------------- export


def test_export_raw_bytes_exact_uint64_and_no_paths(tmp_path):
    rt = runtime(output_dir=str(tmp_path / "out"), cache_dir=str(tmp_path / "log"))
    resp = api.export_response(rt, plan_bytes({"prompt": "a fox",
                                               "seed": "18446744073709551615"}))
    assert resp.status == 200
    assert resp.payload is None and resp.body is not None
    assert any(name == "Content-Disposition" and "attachment" in value
               for name, value in resp.headers)
    body = resp.body.decode("utf-8")
    assert str(tmp_path) not in body
    for field in api.RUNTIME_PATH_FIELDS:
        assert f'"{field}"' not in body
    data = json.loads(body)
    assert data["seed"] == 18446744073709551615      # exact uint64 NUMBER
    assert isinstance(data["seed"], int) and not isinstance(data["seed"], bool)
    assert set(data) == set(api.EDITABLE_FIELDS)
    # deterministic canonical bytes
    assert resp.body == api.export_response(rt, plan_bytes({"prompt": "a fox",
        "seed": "18446744073709551615"})).body


def test_export_null_seed_stays_null(tmp_path):
    rt = runtime(output_dir=str(tmp_path / "o"))
    body = api.export_response(rt, plan_bytes({"prompt": "x", "seed": None})).body
    assert json.loads(body)["seed"] is None


def test_export_rejects_invalid_envelope_with_typed_error():
    resp = api.export_response(runtime(), b'{"nope": 1}')
    assert resp.body is None
    assert resp.payload["schema"] == api.WEB_ERROR_SCHEMA


# ------------------------------------------------------ manifest metadata


def _valid_manifest_bytes() -> bytes:
    return json.dumps({"schema_version": 1, "artifacts": [{
        "role": "transformer", "repo": "Qwen/Qwen-Image",
        "revision": "0" * 40, "filename": "model-00001.safetensors",
        "sha256": "a" * 64, "bytes": 12345,
    }]}).encode()


def test_manifest_summary_metadata_only_no_names_hashes_paths():
    summary = api.summarize_manifest_bytes(_valid_manifest_bytes())
    assert summary == {"configured": True, "valid": True, "schema_version": 1,
                       "artifact_count": 1, "revisions": ["0" * 40],
                       "verification": "not_performed"}
    text = json.dumps(summary)
    assert "model-00001" not in text and "aaaa" not in text


def test_manifest_summary_invalid_and_unconfigured_reasons():
    assert api.summarize_manifest_bytes(b"{not json")["reason"] == "manifest_invalid"
    assert api.summarize_manifest_bytes(None)["reason"] == "manifest_unconfigured"
    assert api.summarize_manifest_bytes(b"{}",
        unreadable=api.MANIFEST_UNREADABLE)["reason"] == api.MANIFEST_UNREADABLE


def test_invalid_manifest_still_plans_with_blocker():
    rt = runtime(manifest_summary=api.summarize_manifest_bytes(b"broken"))
    payload = api.plan_response(rt, plan_bytes({"prompt": "x"})).payload
    assert payload["schema"] == api.WEB_PLAN_SCHEMA          # planning proceeds
    assert "manifest_invalid" in payload["generation_blockers"]
    assert payload["manifest"]["ready"] is False
    status = api.status_response(rt).payload
    assert "manifest_invalid" in status["generation_blockers"]


def test_unconfigured_manifest_blocker():
    status = api.status_response(runtime()).payload
    assert "manifest_unconfigured" in status["generation_blockers"]


# --------------------------------------------------- root-review regressions


def test_runtime_path_never_leaks_after_startup_dir_becomes_file(tmp_path):
    # TOCTOU: startup validation is not a guarantee. A configured runtime
    # dir replaced by a file afterwards must fail with the GENERIC
    # server_runtime_configuration code and no path in any field.
    target = tmp_path / "private-runtime-path"
    target.mkdir()
    rt = runtime(output_dir=str(target))
    assert api.plan_response(rt, plan_bytes({"prompt": "x"})).status == 200
    target.rmdir()
    target.touch()
    for operation in (api.plan_response, api.export_response):
        response = operation(rt, plan_bytes({"prompt": "x"}))
        assert response.status == 422
        text = json.dumps(response.payload)
        assert str(target) not in text
        assert response.payload["code"] == "server_runtime_configuration"
        fields = {f["field"] for f in response.payload.get("fields", [])}
        assert fields == {"server_runtime_configuration"}


def test_manifest_unreadable_reason_is_a_closed_code():
    # Whatever string the caller passed (which could embed a path), the
    # projection only ever carries a closed reason code.
    summary = api.summarize_manifest_bytes(
        None, unreadable="/private/model/config.json: access denied")
    rt = runtime(manifest_summary=summary)
    for response in (api.status_response(rt),
                     api.plan_response(rt, plan_bytes({"prompt": "x"}))):
        text = json.dumps(response.payload)
        assert "/private/" not in text
        assert "manifest_unreadable" in text


def test_calibration_honest_unknown_not_global_absence():
    status = api.status_response(runtime()).payload
    gate = status["gates"]["calibration"]
    assert gate["code"] == "calibration_unknown"
    assert "not VERIFIED for this server" in gate["reason"]
    plan = api.plan_response(runtime(), plan_bytes({"prompt": "x"})).payload
    assert plan["calibration"]["status"] == "unknown"
    assert "not verified for this server" in plan["calibration"]["detail"]
    assert plan["hardware"]["status"] == "unknown"
    assert "has not probed hardware" in plan["hardware"]["detail"]


@pytest.mark.parametrize("replace", [
    (b'"schema_version": 1', b'"schema_version": 99, "schema_version": 1'),
    (b'"bytes": 12345', b'"bytes": 1, "bytes": 12345'),
    (b'"role": "transformer"', b'"role": "wrong", "role": "transformer"'),
])
def test_manifest_metadata_refuses_duplicate_keys_like_loader(replace):
    raw = _valid_manifest_bytes().replace(*replace)
    summary = api.summarize_manifest_bytes(raw)
    assert summary["valid"] is False
    assert summary["reason"] == api.MANIFEST_INVALID
    assert "manifest_invalid" in api.status_response(runtime(manifest_summary=summary)).payload["generation_blockers"]


def test_runtime_metadata_is_an_owned_deep_snapshot():
    summary = api.summarize_manifest_bytes(_valid_manifest_bytes())
    rt = runtime(manifest_summary=summary)
    before = api.status_response(rt).payload
    summary["valid"] = False
    summary["revisions"].append("1" * 40)
    assert api.status_response(rt).payload == before
    assert rt.manifest_summary["revisions"] == ("0" * 40,)
    with pytest.raises(TypeError):
        rt.manifest_summary["valid"] = False


@pytest.mark.parametrize("summary", [[], {"bad": float("nan")}, {1: "bad"}])
def test_runtime_invalid_summary_has_generic_typed_constructor_error(summary):
    with pytest.raises(ValueError, match="strict JSON mapping"):
        api.WebRuntime(manifest_summary=summary)


@pytest.mark.parametrize("data", [
    {"schema": ENVELOPE, "config": {}, 1: "bad", "extra": "bad"},
    {"schema": ENVELOPE, "config": {1: "bad", "extra": "bad"}},
])
def test_programmatic_mixed_keys_are_typed_not_sorting_errors(data):
    with pytest.raises(api.ApiError) as caught:
        api.build_config(runtime(), data)
    assert caught.value.status == 400


@pytest.mark.parametrize("summary", [
    {"configured": True, "valid": False, "reason": "/private/secret.json"},
    {"configured": True, "valid": True, "schema_version": 1, "artifact_count": "/private/secret.json"},
    {"configured": True, "valid": True, "schema_version": "/private/secret.json", "artifact_count": 1},
    {"configured": True, "valid": True, "schema_version": True, "artifact_count": 1},
])
def test_manifest_public_projection_is_closed_and_blocker_agrees(summary):
    rt = runtime(manifest_summary=summary)
    for response in (api.status_response(rt), api.plan_response(rt, plan_bytes({"prompt": "x"}))):
        assert "/private/" not in json.dumps(response.payload)
        assert response.payload["manifest"]["valid"] is False
        assert response.payload["manifest"]["reason"] == api.MANIFEST_INVALID
        assert "manifest_invalid" in response.payload["generation_blockers"]
