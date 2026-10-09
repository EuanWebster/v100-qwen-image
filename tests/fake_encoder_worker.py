"""Fake encoder worker for tests only — a scripted stub, no model (plan R5).

Driven entirely by underscore-prefixed keys in the request ``kwargs``:

* ``_sleep``: float — seconds to sleep before answering (timeout/cancel tests)
* ``_ignore_sigterm``: true — survive terminate() so the kill path runs
* ``_fail_exit``: int — print to stderr and exit with that code, no response
* ``_stderr``: str — extra text to emit on stderr
* ``_worker_error``: {type?, message} — write an ``error`` result, exit 0
* ``_garbage_response``: true — write non-JSON, exit 0
* ``_no_response``: true — exit 0 without writing anything
* ``_bad_status``: true — write an unknown status
* ``_result_patch``: object — merged into the result document (wire
  violations: unknown fields, NaN metadata, malformed descriptors). NaN
  arrives as the literal ``NaN`` token because the writer uses ``allow_nan``
  exactly like a sloppy worker would.
* ``_response_bytes_file``: path — also write the exact response *bytes*
  there (outside the scratch dir), so tests can set a
  ``max_response_bytes`` cap at the byte-exact boundary. The document is
  deterministic apart from the 32-hex ``request_id`` (fixed length), so
  repeated runs of the same request produce same-LENGTH responses.
* ``_counter_file``: path — append one line per *encode* (double-encode detector)
* ``_pid_file``: path — record the worker pid (kill-path verification)

An otherwise plain request answers ``ok`` with a deterministic blob derived
from the prompt, so cache round-trips are byte-reproducible.

Request schema v2 may carry an ``implementation_identity`` block; the fake
ignores it entirely (it is a *generic* worker — only a production worker
requires an implementation identity).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--response", required=True)
    args = ap.parse_args()

    with open(args.request, encoding="utf-8") as fh:
        req = json.load(fh)
    kwargs = req.get("kwargs", {})

    if pid_file := kwargs.get("_pid_file"):
        with open(pid_file, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
    if kwargs.get("_ignore_sigterm"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if text := kwargs.get("_stderr"):
        print(text, file=sys.stderr, flush=True)
    if counter := kwargs.get("_counter_file"):
        with open(counter, "a", encoding="utf-8") as fh:
            fh.write(f"encode {req['request_id']}\n")
    if (sleep := kwargs.get("_sleep")) is not None:
        deadline = time.monotonic() + sleep
        while time.monotonic() < deadline:
            time.sleep(0.01)
    if (code := kwargs.get("_fail_exit")) is not None:
        print(f"fake worker: failing with exit {code}", file=sys.stderr, flush=True)
        return int(code)
    if kwargs.get("_garbage_response"):
        with open(args.response, "wb") as fh:
            fh.write(b"\xff\xfe not json at all")
        return 0
    if kwargs.get("_no_response"):
        return 0

    doc = {
        "schema": "qwen_engine.encoder_result.v1",
        "request_id": req["request_id"],
    }
    if kwargs.get("_bad_status"):
        doc["status"] = "maybe"
    elif err := kwargs.get("_worker_error"):
        doc.update({"status": "error", "error": err})
    else:
        prompt = req["prompt"]
        negative = req.get("negative_prompt", "")
        blob = b"FAKEEMBED:" + prompt.encode("utf-8") + b"|" + negative.encode("utf-8")
        metadata = {
            "dtype": req["dtype"],
            "prompt_chars": len(prompt),
            "source": "fake-worker",
        }
        if kwargs.get("_nan_metadata"):
            metadata["nan_field"] = float("nan")  # sloppy worker: NaN literal
        doc.update({
            "status": "ok",
            "metadata": metadata,
            "blobs": [{"name": "prompt_embeds",
                       "data_b64": base64.b64encode(blob).decode("ascii")}],
        })
    doc.update(kwargs.get("_result_patch") or {})
    data = json.dumps(doc, sort_keys=True).encode("utf-8")
    if record := kwargs.get("_response_bytes_file"):
        with open(record, "wb") as fh:
            fh.write(data)
    tmp = args.response + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, args.response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
