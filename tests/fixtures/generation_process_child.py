"""PRIVATE subprocess fixture. Never imported or selected by product code."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from qwen_engine import generation_worker as worker
from qwen_engine.core import generate
from qwen_engine.errors import GenerationCancelled

mode = os.environ["QWEN_PROCESS_TEST_MODE"]


class FixtureAdapter:
    def set_cancel_event(self, event):
        self.cancel = event

    def load(self, config, assets):
        self.directory = Path(config.output_dir)

    def encode(self, config):
        return None

    def denoise(self, config, conditioning):
        if mode == "ignore_term":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            while True:
                time.sleep(.01)
        if mode == "block":
            self.cancel.wait(10)
            raise GenerationCancelled()
        if mode == "descendant":
            child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(10)"])
            (self.directory / "descendant.pid").write_text(str(child.pid))
            try:
                self.cancel.wait(10)
                raise GenerationCancelled()
            finally:
                child.terminate()
                child.wait(timeout=2)
        return None

    def decode(self, latents):
        return None

    def save(self, image, config):
        path = self.directory / "fixture.bin"
        path.write_bytes(b"explicit synthetic fixture, not an image")
        return path

    def close(self):
        pass


def fake_generation(config, cancel, report):
    if mode in ("stdout_flood", "stderr_flood"):
        fd = 1 if mode == "stdout_flood" else 2
        while not cancel.is_set():
            os.write(fd, b"private-child-data" * 4096)
        raise GenerationCancelled()
    if mode == "bad_phase":
        rid = worker.parse_request(sys.argv[sys.argv.index("--request") + 1])[0]
        print(json.dumps(dict(schema=worker.EVENT_SCHEMA, request_id=rid,
                             event="phase_start", phase="save", sequence=0)), flush=True)
        cancel.wait(10)
        raise GenerationCancelled()
    if mode == "no_result":
        os._exit(7)
    return generate(config, FixtureAdapter(), {}, cancel_event=cancel, phase_callback=report)


worker.run_generation = fake_generation  # test-only supplied adapter; NEVER a product switch
code = worker.main()
response = Path(sys.argv[sys.argv.index("--response") + 1])
if mode == "bad_result":
    response.write_bytes(b'{"schema":"wrong"}')
elif mode == "symlink_result":
    peer = response.with_name("peer.json")
    response.rename(peer)
    response.symlink_to(peer)
elif mode == "fifo_result":
    response.unlink()
    os.mkfifo(response)
elif mode == "oversize_result":
    response.write_bytes(b"x" * (worker.MAX_RESULT_BYTES + 1))
elif mode == "publish_then_wait":
    # Deliberate late-cancellation fixture: valid committed result already
    # exists, but the process has not exited. TERM then exits CLEANLY.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    doc = json.loads(response.read_text())
    Path(doc["output_path"]).with_name("published.marker").write_bytes(b"ready")
    time.sleep(10)
raise SystemExit(code)
