#!/usr/bin/env python3
"""Fixed text-to-image quality suite for the V100 profile.

This is a **measurement harness**, not a test and not a calibration record. It
runs the real CLI once per fixed prompt (same seeds, same steps, untiled decode
unless ``--tiled-decode`` is passed), then writes:

* one PNG per case in ``--output-dir``,
* the CLI's own ``engine.jsonl`` phase timings per case,
* a contact sheet with captions,
* ``report.json`` with SHA-256 hashes, phase timings, wall time and sampled
  device-wide VRAM / host ``MemAvailable`` peaks.

It samples ``nvidia-smi`` and ``/proc/meminfo`` from this process; it does not
read or write GPU counters itself and it does not claim the sampled peak is a
calibration measurement.

Prompts are FIXED so runs are comparable. Changing one changes the suite
identity; report the prompt text with any result.

Usage:
    .venv/bin/python tools/quality_suite.py \
        --output-dir /tmp/qwen-quality-2026-10-24 \
        --steps 40 --attest-unbounded-host
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    category: str
    prompt: str
    seed: int
    notes: str = ""


#: Fixed suite. The categories are deliberate coverage, not a benchmark of the
#: prompt distribution: object, text legibility, composition, counting, style,
#: lighting.
CASES: tuple[Case, ...] = (
    Case("object", "object",
         "a red cube on a white table, studio photo, sharp focus, soft daylight",
         seed=42, notes="anchor prompt used in the generation measurements"),
    Case("text-open", "text",
         'a hand-painted wooden sign reading "OPEN" leaning against a driftwood '
         "post on a misty beach at dawn, photorealistic",
         seed=7, notes="one short, correctly spelled word; report any garble"),
    Case("composition", "composition",
         "a lone lighthouse on a rocky cliff at the edge of a calm sea, wide "
         "establishing shot, rule of thirds, overcast sky",
         seed=11, notes="horizon placement and subject scale"),
    Case("counting-four-mugs", "counting",
         "exactly four ceramic mugs in a row on a wooden shelf, product photo, "
         "soft window light",
         seed=23, notes="count must be exactly four; duplicates fail the check"),
    Case("style-ukiyoe-fox", "style",
         "a fox in a snowy forest, Japanese ukiyo-e woodblock print, flat "
         "colors, visible paper grain",
         seed=31, notes="style adherence without collapsing into a photo"),
    Case("lighting-glass", "lighting",
         "a glass of water on a windowsill, strong low afternoon sunlight, "
         "long shadows, dust motes in the air, macro photo",
         seed=47, notes="directional light, shadow direction, highlight detail"),
)


@dataclass
class Sampler:
    """Cheap device-wide / host samples around one child process."""

    interval: float = 0.25
    peak_gpu_mib: int = 0
    min_host_available_kib: int = 1 << 62
    samples: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _read_gpu(self) -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False)
            first = out.stdout.strip().splitlines()[0]
            return int(first.strip())
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _read_host() -> int | None:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
        except OSError:
            return None
        return None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            gpu = self._read_gpu()
            host = self._read_host()
            self.samples += 1
            if gpu is not None:
                self.peak_gpu_mib = max(self.peak_gpu_mib, gpu)
            if host is not None:
                self.min_host_available_kib = min(self.min_host_available_kib, host)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def result(self) -> dict[str, Any]:
        return {
            "peak_gpu_used_mib": self.peak_gpu_mib or None,
            "min_host_available_kib": (
                self.min_host_available_kib
                if self.min_host_available_kib < (1 << 62) else None),
            "samples": self.samples,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase_timings(log_path: Path) -> dict[str, float]:
    phases: dict[str, float] = {}
    try:
        lines = log_path.read_text().splitlines()
    except OSError:
        return phases
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "phase_end" and isinstance(event.get("phase"), str):
            elapsed = event.get("elapsed_ms")
            if isinstance(elapsed, (int, float)):
                phases[event["phase"]] = round(float(elapsed), 1)
    return phases


def _run_case(case: Case, args: argparse.Namespace, root: Path,
              output_dir: Path) -> dict[str, Any]:
    case_dir = output_dir / case.name
    log_dir = output_dir / f"{case.name}-logs"
    case_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    command = [
        args.python, "-m", "qwen_engine", "--run",
        "--manifest", str(root / "models/manifest.json"),
        "--models-dir", str(root / "models/assets"),
        "--prompt", case.prompt,
        "--width", str(args.width), "--height", str(args.height),
        "--steps", str(args.steps), "--seed", str(case.seed),
        "--output-dir", str(case_dir),
        "--cache-dir", str(args.cache_dir),
        "--log-dir", str(log_dir),
    ]
    if args.tiled_decode:
        command.append("--tiled-decode")
    if args.attest_unbounded_host:
        command.append("--attest-unbounded-host")
    sampler = Sampler()
    started = time.time()
    sampler.start()
    completed = subprocess.run(
        command, cwd=root, env=env, capture_output=True, text=True,
        timeout=args.timeout_s, check=False)
    wall_s = round(time.time() - started, 1)
    sampler.stop()
    (output_dir / f"{case.name}.stdout").write_text(completed.stdout)
    (output_dir / f"{case.name}.stderr").write_text(completed.stderr)
    record: dict[str, Any] = {
        "name": case.name, "category": case.category, "prompt": case.prompt,
        "seed": case.seed, "width": args.width, "height": args.height,
        "steps": args.steps, "tiled_decode": args.tiled_decode,
        "notes": case.notes, "exit_code": completed.returncode,
        "wall_s": wall_s, "phases_ms": _phase_timings(log_dir / "engine.jsonl"),
        "command": command, "sample": sampler.result(),
        "output_path": None, "sha256": None,
    }
    if completed.returncode == 0:
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            produced = Path(payload["output_path"])
            if produced.exists():
                record["output_path"] = str(produced)
                record["sha256"] = _sha256(produced)
        except (ValueError, KeyError, IndexError, OSError):
            record["exit_code"] = 1
            record["error"] = "could not parse the CLI result/output file"
    else:
        tail = completed.stderr.strip().splitlines()[-1:] or [""]
        record["error"] = tail[0][:500]
    print(
        f"{case.name:<18} rc={record['exit_code']} wall={wall_s:>6.1f}s "
        f"peak={record['sample']['peak_gpu_used_mib']} MiB "
        f"{record['sha256'][:12] if record['sha256'] else '-'}",
        flush=True)
    return record


def _contact_sheet(records: list[dict[str, Any]], path: Path,
                   columns: int = 2) -> None:
    from PIL import Image, ImageDraw, ImageFont

    thumb = 512
    caption_h = 64
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + caption_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    for index, record in enumerate(records):
        column, row = index % columns, index // columns
        x, y = column * thumb, row * (thumb + caption_h)
        if record.get("output_path"):
            image = Image.open(record["output_path"]).convert("RGB")
            image.thumbnail((thumb, thumb))
            sheet.paste(image, (x, y))
        label = (f"{record['name']} / {record['category']} · seed {record['seed']} · "
                 f"{record['steps']} steps")
        draw.text((x + 8, y + thumb + 6), label, fill="black", font=font)
        detail = (f"sha256 {record['sha256'][:16]}…" if record.get("sha256")
                  else f"FAILED rc={record['exit_code']}")
        draw.text((x + 8, y + thumb + 30), detail, fill="#444444", font=font)
    sheet.save(path)
    print(f"contact sheet: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="where PNGs, logs and report.json are written")
    parser.add_argument("--python", default=str(Path(".venv/bin/python")),
                        help="interpreter for the engine child")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/tmp/qwen-cache"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--tiled-decode", action="store_true",
                        help="required for 2048x2048; not recommended at 1024")
    parser.add_argument("--attest-unbounded-host", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=3600)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    python = Path(args.python)
    if not python.is_absolute():
        python = root / python
    args.python = str(python)
    if not python.exists():
        parser.error(f"interpreter not found: {args.python}")

    records = [_run_case(case, args, root, args.output_dir) for case in CASES]
    _contact_sheet(records, args.output_dir / "contact-sheet.png")
    report = {
        "suite": "fixed-quality-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "width": args.width, "height": args.height, "steps": args.steps,
        "tiled_decode": args.tiled_decode, "cases": records,
        "failures": [r["name"] for r in records if r["exit_code"] != 0],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.output_dir / "report.json"),
                      "failures": report["failures"]}))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
