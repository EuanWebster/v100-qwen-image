"""Reference endpoints (design §2.4/T3): upload, serve, envelope, submit gate.

No sockets (pure dispatch/policy functions, per the W2/W3 pattern) and real
tiny Pillow images. The store is the REAL store — only Pillow and tmp dirs.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib

import pytest
from PIL import Image

from qwen_engine import reference_store
from qwen_engine.config import GenerationConfig
from qwen_engine.core import GenerationResult
from qwen_engine.web import api, server
from qwen_engine.web.generation import WebGeneration

ID_A = "a" * 64


def image_bytes(mode="RGBA", size=(8, 6), fmt="PNG"):
    buffer = io.BytesIO()
    Image.new(mode, size, (255, 0, 0, 128) if mode == "RGBA" else (200, 30, 30)).save(buffer, fmt)
    return buffer.getvalue()


def runtime(tmp_path, **overrides):
    summary = api.summarize_manifest_bytes(None)
    values = dict(manifest_summary=summary, references_dir=str(tmp_path / "store"))
    values.update(overrides)
    return api.WebRuntime(**values)


def post_headers(content_type="application/json", length=b"17"):
    return {"x-qwen-ui": ["1"], "content-type": [content_type],
            "content-length": [length]}


# --- upload endpoint -----------------------------------------------------------


def test_upload_201_shape_ids_and_blob_roundtrip(tmp_path):
    rt = runtime(tmp_path)
    raw = image_bytes()
    response = api.upload_reference_response(rt, raw)
    assert response.status == 201
    assert set(response.payload) == {"schema", "id", "width", "height",
                                     "has_alpha", "bytes"}
    assert response.payload == {
        "schema": "qwen_engine.web_reference.v1",
        "id": hashlib.sha256(raw).hexdigest(),
        "width": 8, "height": 6, "has_alpha": True, "bytes": len(raw)}
    stored = api.reference_bytes(rt, response.payload["id"])
    assert stored[0] == raw and stored[1] == "image/png"
    # The public payload NEVER carries the store path.
    assert rt.references_dir not in json.dumps(response.payload)


def test_upload_content_types_and_alpha_flag(tmp_path):
    rt = runtime(tmp_path)
    jpeg = api.upload_reference_response(rt, image_bytes("RGB", (6, 4), "JPEG"))
    webp = api.upload_reference_response(rt, image_bytes("RGB", (6, 4), "WEBP"))
    assert jpeg.payload["has_alpha"] is False
    assert api.reference_bytes(rt, jpeg.payload["id"])[1] == "image/jpeg"
    assert api.reference_bytes(rt, webp.payload["id"])[1] == "image/webp"


def test_upload_is_idempotent_deduped_by_content(tmp_path):
    rt = runtime(tmp_path)
    raw = image_bytes("RGB", (4, 4), "JPEG")
    first = api.upload_reference_response(rt, raw).payload["id"]
    second = api.upload_reference_response(rt, raw).payload["id"]
    assert first == second


@pytest.mark.parametrize("raw", [b"", b"not an image at all",
                                 b"\x89PNG\r\n\x1a\ntruncated lies"])
def test_upload_non_images_are_415(tmp_path, raw):
    with pytest.raises(api.ApiError) as excinfo:
        api.upload_reference_response(runtime(tmp_path), raw)
    assert excinfo.value.status == 415
    assert excinfo.value.code == "unsupported_media_type"


def test_upload_store_full_is_409(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_store, "MAX_STORE_BLOBS", 0)
    with pytest.raises(api.ApiError) as excinfo:
        api.upload_reference_response(runtime(tmp_path), image_bytes())
    assert excinfo.value.status == 409
    assert excinfo.value.code == "reference_store_full"


def test_upload_without_configured_store_is_503(tmp_path):
    with pytest.raises(api.ApiError) as excinfo:
        api.upload_reference_response(runtime(tmp_path, references_dir=None),
                                      image_bytes())
    assert excinfo.value.status == 503
    assert excinfo.value.code == "reference_store_unconfigured"


# --- serving -------------------------------------------------------------------


def test_thumb_serves_stored_png_preview(tmp_path):
    rt = runtime(tmp_path)
    raw = image_bytes(size=(300, 400))  # larger than the preview cap
    reference_id = api.upload_reference_response(rt, raw).payload["id"]
    thumb, ctype = api.reference_bytes(rt, reference_id, thumb=True)
    assert ctype == "image/png"
    assert thumb != raw
    assert max(Image.open(io.BytesIO(thumb)).size) <= 256


@pytest.mark.parametrize("reference_id", [ID_A, "A" * 64, "abc", "", "a" * 63])
def test_unknown_or_unreadable_references_are_one_404(tmp_path, reference_id):
    with pytest.raises(api.ApiError) as excinfo:
        api.reference_bytes(runtime(tmp_path), reference_id)
    assert excinfo.value.status == 404
    assert excinfo.value.code == "reference_unknown"


def test_tampered_blob_is_404_not_leak(tmp_path):
    rt = runtime(tmp_path)
    raw = image_bytes()
    reference_id = api.upload_reference_response(rt, raw).payload["id"]
    pathlib.Path(rt.references_dir, reference_id + ".img").write_bytes(b"tampered")
    with pytest.raises(api.ApiError) as excinfo:
        api.reference_bytes(rt, reference_id)
    assert excinfo.value.status == 404
    assert rt.references_dir not in json.dumps(excinfo.value.to_payload())


# --- routing and header policy (pure) ------------------------------------------


def test_routing_reference_routes():
    assert server.resolve_route("POST", "/api/v1/references") == ("upload", "/api/v1/references")
    assert server.resolve_route("GET", f"/api/v1/references/{ID_A}") == ("reference", ID_A)
    assert server.resolve_route("GET", f"/api/v1/references/{ID_A}/thumb") == ("reference_thumb", ID_A)


@pytest.mark.parametrize("target", [
    f"/api/v1/references/{'A'*64}", f"/api/v1/references/{'a'*63}",
    f"/api/v1/references/{'a'*64}/extra", "/api/v1/references//thumb"])
def test_routing_noncanonical_ids_are_404(target):
    with pytest.raises(api.ApiError) as excinfo:
        server.resolve_route("GET", target)
    assert excinfo.value.status == 404


@pytest.mark.parametrize("method,target,allow", [
    ("GET", "/api/v1/references", "POST"),
    ("POST", f"/api/v1/references/{ID_A}", "GET"),
    ("POST", f"/api/v1/references/{ID_A}/thumb", "GET"),
])
def test_routing_wrong_method_is_405_with_allow(method, target, allow):
    with pytest.raises(api.ApiError) as excinfo:
        server.resolve_route(method, target)
    assert excinfo.value.status == 405
    assert (allow,) == tuple(v for n, v in excinfo.value.headers if n == "Allow")


def test_upload_header_policy_octet_only_with_big_cap():
    origin = "http://127.0.0.1:8765"
    big = str(api.MAX_REFERENCE_UPLOAD_BYTES)
    assert server.header_policy("POST", post_headers(
        "application/octet-stream", big.encode()), origin, upload=True) is None
    assert server.header_policy("POST", post_headers(
        "application/json", big.encode()), origin, upload=True).status == 415
    refused = server.header_policy("POST", post_headers(
        "application/octet-stream", b"100000"), origin)
    assert refused.status == 415  # octet-stream only on the upload route
    assert server.header_policy("POST", post_headers(
        "application/json", b"100000"), origin).status == 413  # 64 KiB JSON cap
    over = server.header_policy("POST", post_headers(
        "application/octet-stream", str(api.MAX_REFERENCE_UPLOAD_BYTES + 1).encode()),
        origin, upload=True)
    assert over.status == 413
    no_ui = post_headers("application/octet-stream", b"10")
    del no_ui["x-qwen-ui"]
    assert server.header_policy("POST", no_ui, origin, upload=True).code == "ui_header_required"
    dup = post_headers("application/octet-stream", b"10")
    dup["content-length"] = [b"10", b"10"]
    assert server.header_policy("POST", dup, origin, upload=True).code == "duplicate_header"


# --- envelope and echo ----------------------------------------------------------


def plan_payload(rt, config):
    body = json.dumps({"schema": api.WEB_PLAN_REQUEST_SCHEMA, "config": config}).encode()
    return api.plan_response(rt, body)


def test_envelope_accepts_ordered_ids_and_old_envelopes_import_empty(tmp_path):
    rt = runtime(tmp_path)
    ordered = [f"{i:064x}" for i in range(3)]
    response = plan_payload(rt, {"prompt": "edit", "references": ordered})
    assert response.status == 200
    assert response.payload["config"]["references"] == ordered  # order preserved
    legacy = plan_payload(rt, {"prompt": "plain"})
    assert legacy.payload["config"]["references"] == []
    assert "references_dir" not in json.dumps(response.payload)
    assert rt.references_dir not in json.dumps(response.payload)


def test_envelope_refuses_over_cap_and_bad_ids_publicly(tmp_path):
    rt = runtime(tmp_path)
    too_many = plan_payload(rt, {"prompt": "x",
                                 "references": [f"{i:064x}" for i in range(11)]})
    assert too_many.status == 422
    assert any(field["field"] == "references" for field in too_many.payload["fields"])
    bad = plan_payload(rt, {"prompt": "x", "references": ["not-hex"]})
    assert bad.status == 422
    assert any(field["field"] == "references" for field in bad.payload["fields"])


def test_export_and_import_carry_references(tmp_path):
    rt = runtime(tmp_path)
    ordered = ["b" * 64, "a" * 64]
    body = json.dumps({"schema": api.WEB_PLAN_REQUEST_SCHEMA,
                       "config": {"prompt": "edit", "references": ordered}}).encode()
    exported = api.export_response(rt, body)
    assert json.loads(exported.body)["references"] == ordered
    imported = api.import_response(rt, exported.body)
    assert imported.payload["config"]["references"] == ordered


def test_ten_id_job_envelope_stays_under_the_ipc_cap(tmp_path):
    from qwen_engine.job_process import MAX_REQUEST_BYTES
    config = plan_payload(runtime(tmp_path),
                          {"prompt": "edit",
                           "references": [f"{i:064x}" for i in range(10)]})
    ids = config.payload["config"]["references"]
    merged = GenerationConfig(prompt="edit", references=tuple(ids),
                              references_dir="/server/store")
    assert len(json.dumps(merged.to_dict())) < MAX_REQUEST_BYTES


# --- submit gate ------------------------------------------------------------------


@pytest.fixture
def gen_runtime(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    return api.WebRuntime(output_dir=str(root), cache_dir=str(tmp_path / "cache"),
                          models_dir=str(tmp_path / "models"),
                          manifest_path=str(tmp_path / "manifest.json"),
                          references_dir=str(tmp_path / "store"),
                          manifest_summary={"configured": True, "valid": True,
                                            "schema_version": 1, "artifact_count": 24,
                                            "verification": "not_performed"},
                          generation_enabled=True)


def envelope(**values):
    return json.dumps({"schema": api.WEB_PLAN_REQUEST_SCHEMA,
                       "config": {"prompt": "edit", "seed": "1", **values}}).encode()


def make_service(gen_runtime):
    def executor(context):
        path = pathlib.Path(context.config.output_dir) / "result.png"
        path.write_bytes(image_bytes("RGB", (2, 2), "PNG"))
        return GenerationResult(str(path))
    return WebGeneration(gen_runtime, executor=executor)


def test_submit_resolves_stored_reference_ids(gen_runtime):
    stored = api.upload_reference_response(
        gen_runtime, image_bytes()).payload["id"]
    service = make_service(gen_runtime)
    try:
        response = service.submit(envelope(references=[stored, stored]))
        assert response.status == 202
        assert response.payload["config"]["references"] == [stored, stored]
        assert service.controller.wait(response.payload["job_id"]).state.value == "succeeded"
    finally:
        service.close()


def test_submit_missing_reference_is_422_ordinal_only(gen_runtime):
    stored = api.upload_reference_response(
        gen_runtime, image_bytes()).payload["id"]
    service = make_service(gen_runtime)
    try:
        with pytest.raises(api.ApiError) as excinfo:
            service.submit(envelope(references=[stored, ID_A]))
    finally:
        service.close()
    assert excinfo.value.status == 422
    assert excinfo.value.code == "reference_unavailable"
    assert "reference 2" in excinfo.value.message
    assert ID_A not in excinfo.value.message
    assert gen_runtime.references_dir not in excinfo.value.message


def test_submit_tampered_reference_is_422(gen_runtime):
    stored = api.upload_reference_response(
        gen_runtime, image_bytes()).payload["id"]
    pathlib.Path(gen_runtime.references_dir, stored + ".img").write_bytes(b"swapped bytes")
    service = make_service(gen_runtime)
    try:
        with pytest.raises(api.ApiError) as excinfo:
            service.submit(envelope(references=[stored]))
    finally:
        service.close()
    assert excinfo.value.status == 422
    assert "reference 1" in excinfo.value.message
