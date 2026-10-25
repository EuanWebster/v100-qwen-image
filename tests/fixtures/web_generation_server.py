"""Test-only W3 HTTP fixture. Synthetic PNGs NEVER enter the product path."""
import signal
import struct
import sys
import zlib
from pathlib import Path

from qwen_engine.core import GenerationResult
from qwen_engine.web.api import WebRuntime
from qwen_engine.web.generation import WebGeneration
from qwen_engine.web.server import _Handler, _Server, load_static_assets


def png():
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 32, 32, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0' + b'\xc0\x30\x30' * 32) * 32)) + chunk(b'IEND', b''))


def execute(context):
    for phase in ('load', 'encode', 'denoise', 'decode', 'save'):
        context.report_phase(phase)
        if phase == 'denoise':
            for step in range(context.config.steps):
                context.check_cancelled()
                httpd.generation.report_steps(context.job_id, step + 1, context.config.steps)
                context._cancel.wait(.20)
        context._cancel.wait(.35)
        context.check_cancelled()
    from PIL import Image
    from qwen_engine.image_output import save_output_image
    path = save_output_image(Image.new('RGB', (32, 32), (192, 48, 48)),
                             context.config.output_dir, context.config.output_format.value,
                             context.config.seed)
    return GenerationResult(str(path))


root = Path(sys.argv[1])
runtime = WebRuntime(output_dir=str(root), cache_dir=str(root / 'cache'),
                     models_dir=str(root / 'models'), manifest_path=str(root / 'manifest.json'),
                     manifest_summary={'configured': True, 'valid': True, 'schema_version': 1,
                                       'artifact_count': 24}, generation_enabled=True)
httpd = _Server(('127.0.0.1', 0), _Handler)
httpd.web_runtime, httpd.assets = runtime, load_static_assets()
httpd.bound_ip, httpd.bound_port = httpd.server_address
httpd.generation = WebGeneration(runtime, executor=execute)


def stop(*args):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, stop)
print(f'qwen-web listening on http://127.0.0.1:{httpd.bound_port}', flush=True)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    httpd.server_close()
    if not httpd.generation.close():
        raise RuntimeError('fixture worker failed to stop')
