"""Import the raw editable export without JS uint64 rounding or path injection."""
import json

import pytest

from qwen_engine.web import api, server
from qwen_engine.web.generation import WebGeneration


def test_export_import_uint64_max_and_new_format_roundtrip():
    runtime = api.WebRuntime()
    request = json.dumps({'schema': api.WEB_PLAN_REQUEST_SCHEMA,
                          'config': {'prompt': 'a room', 'seed': '18446744073709551615', 'output_format': 'webp'}}).encode()
    exported = api.export_response(runtime, request)
    assert b'18446744073709551615' in exported.body
    imported = api.import_response(runtime, exported.body)
    assert imported.status == 200
    assert imported.payload['config']['seed'] == '18446744073709551615'
    assert imported.payload['config']['output_format'] == 'webp'
    assert imported.payload['schema'] == api.WEB_PLAN_SCHEMA


def test_legacy_export_defaults_to_jpg():
    response = api.import_response(api.WebRuntime(), b'{"prompt":"legacy file","seed":42}')
    assert response.status == 200
    assert response.payload['config']['output_format'] == 'jpg'


def test_raw_export_carries_transparency_and_legacy_raw_file_defaults_false():
    runtime = api.WebRuntime()
    exported = api.export_response(runtime, json.dumps({
        'schema': api.WEB_PLAN_REQUEST_SCHEMA,
        'config': {'prompt': 'a jar', 'transparent_background': True,
                   'output_format': 'png'}}).encode())
    assert json.loads(exported.body)['transparent_background'] is True
    imported = api.import_response(runtime, exported.body)
    assert imported.status == 200, imported.payload
    assert imported.payload['config']['transparent_background'] is True
    legacy = api.import_response(runtime, b'{"prompt":"legacy file"}')
    assert legacy.payload['config']['transparent_background'] is False
    # Raw import of an alpha request still cannot smuggle jpg through.
    refused = api.import_response(runtime,
        b'{"prompt":"a","transparent_background":true,"output_format":"jpg"}')
    assert refused.status == 422
    assert [f['field'] for f in refused.payload['fields']] == ['output_format']


def test_legacy_export_without_recipe_imports_as_the_reference_recipe():
    response = api.import_response(api.WebRuntime(), b'{"prompt":"legacy file"}')
    assert response.status == 200
    assert response.payload['config']['recipe'] == 'base'


@pytest.mark.parametrize('data', [
    b'{"prompt":"a","seed":1,"seed":2}', b'{"prompt":"a","seed":true}',
    b'{"prompt":"a","seed":1.5}', b'{"prompt":"a","seed":18446744073709551616}',
    b'{"prompt":"a","output_format":"tiff"}', b'null', b'{"prompt":"   "}',
    b'{"prompt":"a","output_dir":"/private/output"}',
    b'{"prompt":"a","q8_manifest_path":"/private/model"}',
])
def test_invalid_import_is_a_typed_refusal(data):
    response = api.import_response(api.WebRuntime(), data)
    assert response.status in (400, 422)
    assert response.payload['schema'] == api.WEB_ERROR_SCHEMA
    assert '/private' not in json.dumps(response.payload)


def test_import_route_is_post_only():
    assert server.resolve_route('POST', '/api/v1/config/import')[0] == 'import'
    with pytest.raises(api.ApiError) as caught:
        server.resolve_route('GET', '/api/v1/config/import')
    assert caught.value.status == 405


@pytest.mark.parametrize('data,mime', [(b'\xff\xd8\xffxxx', 'image/jpeg'),
    (b'\x89PNG\r\n\x1a\nxxx', 'image/png'), (b'RIFFxxxxWEBPxx', 'image/webp')])
def test_image_response_types(data, mime):
    assert WebGeneration.image_content_type(data) == mime
