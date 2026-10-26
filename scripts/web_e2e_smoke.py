#!/usr/bin/env python3
"""End-to-end smoke client for a REAL running qwen web server (stdlib only).

Drives the public HTTP API exactly as the browser does — the ``X-Qwen-UI: 1``
write header, the exact ``application/json`` / ``application/octet-stream``
content types, the ``{"schema", "config"}`` envelope with only the fifteen
browser-editable fields — against ``--base-url``, which MUST be loopback
(``http://127.0.0.1``/``localhost``/``[::1]``; anything else, including TLS
and any path, is refused before a socket is opened).

Sequence: GET ``/`` and POST ``/api/v1/plan``; upload ``--reference`` to
POST ``/api/v1/references`` and fetch its thumb; submit a fixed job list
(``t2i-turbo``, ``transparent``, ``edit-1ref``, ``edit-2ref``; edits at
1024x960, seed 42) SEQUENTIALLY; poll each to a terminal state within
``--job-timeout``; download and verify the image (magic bytes, non-empty,
sha256; for ``transparent`` also an alpha-channel presence check via Pillow
when importable). It NEVER cancels or forgets anything, reads only its own
jobs, and writes a prompt-free JSON report (kinds, states, error codes,
elapsed + phase timings, image byte counts and digests).

Exit codes: ``0`` everything succeeded, ``1`` any job failed/timed out or a
verification failed, ``2`` usage/connection refusal (non-loopback URL,
missing reference file, server unreachable).
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import time
import urllib.parse
from pathlib import Path

REPORT_SCHEMA = "qwen_engine.web_e2e_smoke.v1"
# Same envelope schema the browser uses for plan AND submit (web/api.py).
WIRE_SCHEMA = "qwen_engine.web_plan_request.v1"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
JOB_KINDS = ("t2i-turbo", "transparent", "edit-1ref", "edit-2ref")
TURBO_STEPS = 8  # the recipe owns its step count (config.TURBO_STEPS)
MAX_RESPONSE_BYTES = 512 * 1024 * 1024

#: Fixed prompts for the smoke jobs. They go to the server and NOWHERE
#: else — the report never carries a prompt.
PROMPTS = {
    "t2i-turbo": "a red ceramic cube on a white table, studio light",
    "transparent": "a single green apple, centered",
    "edit-1ref": "make the scene snowy",
    "edit-2ref": "blend these two scenes",
}


def sniff_image(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def parse_base_url(text: str) -> tuple[str, int]:
    """Loopback-only HTTP origin -> (host, port); ValueError otherwise."""
    parts = urllib.parse.urlsplit(text)
    if (parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS
            or parts.path not in ("", "/") or parts.query or parts.fragment
            or parts.username or parts.password):
        raise ValueError("base-url must be a bare loopback http origin "
                         "(http://127.0.0.1:PORT, http://localhost:PORT or "
                         "http://[::1]:PORT) — refusing anything else")
    try:
        port = parts.port if parts.port is not None else 80
    except ValueError:  # malformed port digits
        raise ValueError("base-url has an invalid port") from None
    if not 1 <= port <= 65535:
        raise ValueError("base-url has an out-of-range port")
    return parts.hostname, port


class ApiClient:
    """Tiny strict client: every POST carries the browser's exact headers."""

    def __init__(self, host: str, port: int, *, timeout: float = 30.0):
        self.host, self.port, self.timeout = host, port, timeout

    def _connect(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port,
                                          timeout=self.timeout)

    def request(self, method: str, path: str, *,
                body: bytes | None = None,
                content_type: str | None = None) -> tuple[int, bytes]:
        headers: dict[str, str] = {}
        if method == "POST":
            headers["X-Qwen-UI"] = "1"  # exactly "1", the CSRF-like gate
            headers["Content-Type"] = content_type or "application/json"
        connection = self._connect()
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise ValueError("server response exceeded the smoke cap")
            return response.status, data
        finally:
            connection.close()

    def request_json(self, method, path, body=None, content_type=None):
        status, data = self.request(method, path, body=body,
                                    content_type=content_type)
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        return status, payload


def job_config(kind: str, reference_ids: list[str]) -> dict:
    config: dict = {"prompt": PROMPTS[kind], "seed": 42,
                    "width": 1024, "height": 1024}
    if kind == "t2i-turbo":
        config.update(recipe="turbo", steps=TURBO_STEPS)
    elif kind == "transparent":
        config.update(recipe="base", transparent_background=True)
    elif kind == "edit-1ref":
        config.update(recipe="base", width=1024, height=960,
                      references=list(reference_ids[:1]))
    elif kind == "edit-2ref":
        config.update(recipe="base", width=1024, height=960,
                      references=list(reference_ids) * 2)  # duplicates legal
    return config


def envelope(config: dict) -> bytes:
    return json.dumps({"schema": WIRE_SCHEMA, "config": config},
                      sort_keys=True).encode("utf-8")


def alpha_presence(data: bytes):
    """True/False via Pillow when importable, "skipped" without it."""
    try:
        import io

        from PIL import Image
    except ImportError:
        return "skipped"
    with Image.open(io.BytesIO(data)) as image:
        return "A" in image.getbands()


def poll_job(client: ApiClient, job_id: str, *, timeout: float,
             poll_s: float) -> tuple[dict | None, bool]:
    """Poll ONLY this smoke-submitted job (GET) to a terminal state.
    Returns (last record, timed_out); never cancels, never forgets."""
    deadline = time.monotonic() + timeout
    record: dict | None = None
    while True:
        status, payload = client.request_json("GET", f"/api/v1/jobs/{job_id}")
        if status == 200 and isinstance(payload, dict):
            record = payload
            if payload.get("state") in TERMINAL_STATES:
                return record, False
        if time.monotonic() >= deadline:
            return record, True
        time.sleep(poll_s)


def run_smoke(client: ApiClient, kinds: tuple[str, ...], reference: bytes | None,
              *, job_timeout: float, poll_s: float) -> tuple[dict, bool]:
    report: dict = {"schema": REPORT_SCHEMA, "host": client.host,
                    "port": client.port, "index_ok": False, "plan_ok": False,
                    "reference": None, "jobs": [], "ok": False}

    status, data = client.request("GET", "/")
    report["index_ok"] = status == 200 and b"<" in data[:4096]

    status, plan = client.request_json("POST", "/api/v1/plan",
                                       body=envelope(job_config("t2i-turbo", [])))
    report["plan_ok"] = status == 200 and isinstance(plan, dict)

    reference_ids: list[str] = []
    if reference is not None:
        status, payload = client.request_json(
            "POST", "/api/v1/references", body=reference,
            content_type="application/octet-stream")
        if status == 201 and isinstance(payload, dict) \
                and isinstance(payload.get("id"), str):
            reference_id = payload["id"]
            tstatus, thumb = client.request("GET",
                                            f"/api/v1/references/{reference_id}/thumb")
            report["reference"] = {
                "id": reference_id, "bytes": len(reference),
                "thumb_bytes": len(thumb),
                "thumb_format": sniff_image(thumb),
                "thumb_ok": tstatus == 200 and sniff_image(thumb) == "png",
            }
            reference_ids = [reference_id]

    ok = report["index_ok"] and report["plan_ok"]
    if any(kind.startswith("edit") for kind in kinds) and not reference_ids:
        # edits are unrunnable without an uploaded reference: refuse early
        report["ok"] = False
        return report, False
    if reference is not None and not report["reference"]:
        ok = False

    for kind in kinds:
        entry: dict = {"kind": kind, "job_id": None, "state": "submit-refused",
                       "error_code": None, "elapsed_s": None, "phase_ms": None,
                       "image_bytes": 0, "image_format": None,
                       "image_sha256": None, "alpha": None}
        report["jobs"].append(entry)
        status, submitted = client.request_json("POST", "/api/v1/jobs",
                                                body=envelope(job_config(kind, reference_ids)))
        if (status != 202 or not isinstance(submitted, dict)
                or not isinstance(submitted.get("job_id"), str)):
            entry["error_code"] = (submitted or {}).get("code", "submit_failed") \
                if isinstance(submitted, dict) else "submit_failed"
            ok = False
            continue
        job_id = submitted["job_id"]
        entry["job_id"] = job_id
        started = time.monotonic()
        record, timed_out = poll_job(client, job_id, timeout=job_timeout,
                                     poll_s=poll_s)
        entry["elapsed_s"] = round(time.monotonic() - started, 2)
        if record is not None:
            entry["state"] = record.get("state")
            entry["error_code"] = record.get("error_code")
            entry["phase_ms"] = record.get("phase_ms")
        if timed_out:
            entry["state"] = "timeout"
            ok = False
            continue
        if entry["state"] != "succeeded":
            ok = False
            continue
        status, image = client.request("GET", f"/api/v1/jobs/{job_id}/image")
        entry["image_bytes"] = len(image)
        entry["image_format"] = sniff_image(image)
        if status != 200 or not image or entry["image_format"] is None:
            entry["error_code"] = entry["error_code"] or "image_verification_failed"
            ok = False
            continue
        entry["image_sha256"] = hashlib.sha256(image).hexdigest()
        if kind == "transparent":
            entry["alpha"] = alpha_presence(image)
            if entry["alpha"] is False:
                ok = False  # "skipped" stays a note, not a failure
    report["ok"] = bool(ok)
    return report, ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True,
                        help="loopback http origin, e.g. http://127.0.0.1:8765")
    parser.add_argument("--reference", default=None,
                        help="image file to upload for the edit jobs")
    parser.add_argument("--jobs", default=",".join(JOB_KINDS),
                        help="comma list from " + ",".join(JOB_KINDS))
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--report", default="web_e2e_smoke_report.json")
    args = parser.parse_args(argv)

    try:
        host, port = parse_base_url(args.base_url)
    except ValueError as exc:
        print(f"web_e2e_smoke: {exc}", file=sys.stderr)
        return 2
    kinds = tuple(part.strip() for part in args.jobs.split(",") if part.strip())
    if not kinds or any(kind not in JOB_KINDS for kind in kinds):
        print(f"web_e2e_smoke: --jobs must be a subset of {','.join(JOB_KINDS)}",
              file=sys.stderr)
        return 2
    needs_reference = any(kind.startswith("edit") for kind in kinds)
    reference: bytes | None = None
    if args.reference is not None:
        try:
            reference = Path(args.reference).read_bytes()
        except OSError as exc:
            print(f"web_e2e_smoke: reference file unreadable: {exc.strerror}",
                  file=sys.stderr)
            return 2
    elif needs_reference:
        print("web_e2e_smoke: edit jobs need --reference", file=sys.stderr)
        return 2

    client = ApiClient(host, port)
    try:
        report, ok = run_smoke(client, kinds, reference,
                               job_timeout=args.job_timeout,
                               poll_s=args.poll_interval)
    except (ConnectionRefusedError, ConnectionResetError, TimeoutError,
            OSError, ValueError) as exc:
        print(f"web_e2e_smoke: server unreachable or response invalid "
              f"({type(exc).__name__})", file=sys.stderr)
        return 2
    destination = Path(args.report)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for entry in report["jobs"]:
        print(f"{entry['kind']}: {entry['state']}"
              + (f" ({entry['error_code']})" if entry["error_code"] else ""))
    print(f"report: {destination}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
