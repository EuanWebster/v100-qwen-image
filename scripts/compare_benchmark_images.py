#!/usr/bin/env python3
"""Compare SAVED benchmark images byte-for-byte (CPU-only, no GPU/ML imports).

Decodes already-saved images, refuses anything non-comparable, and writes ONE
strict JSON report; it never generates, downloads, or replaces an input. Nothing
is rescaled, cropped or alpha-flattened: bounded regular files (existing 64 MiB
raw pixel cap), 4096 px per side, full decode required (truncated/corrupt/
decompression-bomb refused), PNG/JPEG/WEBP RGB or RGBA only (palette/gray/CMYK
is refused, never converted), and each candidate must match the reference size
and mode — a JPEG against an RGBA PNG is refused with the reason, not silently
re-encoded. Same-mode pairs in different containers are measured and flagged
because codec error is inside those deltas. Numeric deltas do NOT establish
perceptual equivalence; see docs/image-quality-benchmark.md.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_engine._local_io import read_bounded_file  # noqa: E402
from qwen_engine.config import ConfigError  # noqa: E402
from qwen_engine.decode_bench import (  # noqa: E402
    MAX_IMAGE_BUFFER_BYTES, MAX_REPORT_BYTES, alpha_stats, compare_rgb_bytes,
    exclusive_write_bytes, sha256_bytes,
)
from qwen_engine.errors import EngineError, ExitCode  # noqa: E402

SCHEMA = "qwen.image-compare"
VERSION = 1
MAX_SIDE = 4096
ALLOWED_FORMATS = ("PNG", "JPEG", "WEBP")
_CHANNELS = {"RGB": 3, "RGBA": 4}
_IDENTITY = ("width", "height", "mode", "format", "encoded_sha256", "pixel_sha256")

LIMITATIONS = [
    "numerical deltas do not establish perceptual equivalence; inspect the saved "
    "images at 1:1, including tile boundaries, before any quality claim",
    "encoded_sha256 covers container bytes (PNG text metadata included); "
    "pixel_sha256 covers decoded pixels only",
    "mean_channel_delta averages absolute deltas over every channel sample "
    "(0..255), not over changed pixels",
    f"inputs are capped at {MAX_IMAGE_BUFFER_BYTES} bytes / {MAX_SIDE} px per side; "
    "larger images are refused, never scaled",
    "JPEG is lossy; WebP may be lossy or lossless depending on the encoder (this "
    "engine writes lossless WebP). Compare lossless PNGs from one run to attribute "
    "deltas to decode geometry",
]


def _decode(path: Any) -> dict[str, Any]:
    """Read, cap-check and FULLY decode one saved image (Pillow imported lazily)."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment without Pillow
        raise ConfigError(f"Pillow is required to read saved images: {exc}") from exc
    try:
        payload = read_bounded_file(path, MAX_IMAGE_BUFFER_BYTES)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read image input: {exc}") from exc
    try:
        with Image.open(io.BytesIO(payload)) as image:
            fmt, (width, height), mode = image.format, image.size, image.mode
            # Both caps are checked against the HEADER, before any pixel buffer exists.
            if fmt not in ALLOWED_FORMATS:
                raise ConfigError(
                    f"unsupported image format {fmt!r}; saved engine images are PNG/JPEG/WEBP")
            if mode not in _CHANNELS:
                raise ConfigError(
                    f"unsupported image mode {mode!r}; only RGB and RGBA are comparable "
                    "(no palette/grayscale/CMYK conversion)")
            if not (0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE
                    and width * height * _CHANNELS[mode] <= MAX_IMAGE_BUFFER_BYTES):
                raise ConfigError(
                    f"image {width}x{height} {mode} exceeds the supported {MAX_SIDE} px per "
                    f"side / {MAX_IMAGE_BUFFER_BYTES} byte decoded-pixel cap")
            image.load()  # force the full decode before any metric sees the pixels
            if image.size != (width, height) or image.mode != mode:
                raise ConfigError("image dimensions/mode changed while decoding")
            pixels = image.tobytes()
    except ConfigError:
        raise
    except Image.DecompressionBombError as exc:
        raise ConfigError(f"refusing a decompression-bomb image: {exc}") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise ConfigError(f"not a readable image (truncated or corrupt?): {exc}") from exc
    channels = _CHANNELS[mode]
    if len(pixels) != width * height * channels:
        raise ConfigError("decoded pixel buffer size disagrees with the image dimensions")
    return {"format": fmt, "width": width, "height": height, "mode": mode,
            "channels": channels, "pixels": pixels,
            "encoded_sha256": sha256_bytes(payload),
            "pixel_sha256": sha256_bytes(pixels),
            "alpha": alpha_stats(pixels) if channels == 4 else None}


def _identity(record: dict[str, Any]) -> dict[str, Any]:
    """Report-safe facts (no paths/prompts/metadata); alpha only when RGBA."""
    block = {key: record[key] for key in _IDENTITY}
    if record["alpha"] is not None:
        block["alpha"] = record["alpha"]
    return block


def _compare(reference: dict[str, Any], other: dict[str, Any], what: str) -> dict[str, Any]:
    """Refuse a non-comparable pair, then return the measured delta block."""
    if (other["width"], other["height"]) != (reference["width"], reference["height"]):
        raise ConfigError(
            f"{what} is {other['width']}x{other['height']} but the reference is "
            f"{reference['width']}x{reference['height']}; no scaling or cropping is applied")
    if other["mode"] != reference["mode"]:
        raise ConfigError(
            f"{what} is {other['format']} {other['mode']} but the reference is "
            f"{reference['format']} {reference['mode']}: the modes differ, so comparing "
            "them would need an implicit channel conversion (adding/dropping alpha or "
            "flattening it onto a background); save both images with the same mode from "
            "the same run instead")
    delta = compare_rgb_bytes(reference["pixels"], other["pixels"],
                              channels=reference["channels"])
    return _identity(other) | {
        "exact_pixels_equal": other["pixel_sha256"] == reference["pixel_sha256"],
        "max_channel_delta": delta["max_channel_delta"],
        "mean_channel_delta": delta["mean_channel_delta"],
        "per_channel_max_delta": delta["per_channel_max_delta"],
        "changed_pixel_fraction": delta["changed_pixel_fraction"],
        "encoding_note": (
            None if other["format"] == reference["format"] else
            f"{what} is {other['format']} and the reference is {reference['format']}; "
            "decoded pixel deltas may include codec differences, so they are not "
            "attributable to decode geometry alone (JPEG is lossy; WebP may be lossy "
            "or lossless depending on the encoder)")}


def compare_images(reference_path: Any, candidate_paths: Any, *,
                   expected_size: tuple[int, int] | None = None) -> dict[str, Any]:
    """Compare saved images against one reference; returns a strict-JSON report."""
    if not candidate_paths:
        raise ConfigError("at least one --candidate is required")
    reference = _decode(reference_path)
    if expected_size is not None and (reference["width"], reference["height"]) != expected_size:
        raise ConfigError(
            f"reference is {reference['width']}x{reference['height']} but "
            f"{expected_size[0]}x{expected_size[1]} was requested; no scaling is applied")
    candidates = [{"label": f"candidate-{index}"} | _compare(reference, _decode(path),
                                                            f"candidate-{index}")
                  for index, path in enumerate(candidate_paths, start=1)]
    return {"schema": SCHEMA, "version": VERSION, "reference": _identity(reference),
            "candidates": candidates, "limitations": list(LIMITATIONS)}


def write_report(report: dict[str, Any], path: Any) -> None:
    """Publish the report with the reviewed no-clobber helper (``O_EXCL`` + fsync).

    ``exclusive_write_bytes`` refuses any existing destination and cleans up its
    own partial file on failure, so an earlier report or an input image is never
    replaced. The payload is bounded and strict-JSON before that call, so a
    non-finite metric cannot reach the filesystem.
    """
    try:
        payload = json.dumps(report, sort_keys=True, ensure_ascii=True,
                             allow_nan=False, separators=(",", ":")).encode("ascii")
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"comparison report is not strict JSON: {exc}") from exc
    if len(payload) > MAX_REPORT_BYTES:
        raise ConfigError(f"comparison report exceeds the {MAX_REPORT_BYTES} byte cap")
    try:
        exclusive_write_bytes(path, payload + b"\n")
    except OSError as exc:
        raise EngineError(f"cannot publish comparison report: {exc}") from exc


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare saved benchmark images offline (no GPU, no ML imports).")
    parser.add_argument("--reference", required=True, type=Path,
                        help="reference image (the production-geometry output)")
    parser.add_argument("--candidate", required=True, action="append", type=Path,
                        metavar="PATH", help="candidate image; repeatable")
    parser.add_argument("--output", required=True, type=Path, metavar="NEW.json",
                        help="new report path; an existing file is never replaced")
    parser.add_argument("--expect-size", default=None, metavar="WxH",
                        help="refuse images that are not exactly this pixel size")
    args = parser.parse_args(argv)
    try:
        expected = None
        if args.expect_size:
            parts = args.expect_size.lower().split("x")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ConfigError(f"--expect-size must be WxH, got {args.expect_size!r}")
            expected = tuple(int(part) for part in parts)
            if not all(0 < side <= MAX_SIDE for side in expected):
                raise ConfigError(f"--expect-size must be within 1..{MAX_SIDE} px per side")
        output = os.path.realpath(os.path.abspath(str(args.output)))
        if any(os.path.realpath(os.path.abspath(str(path))) == output
               for path in (args.reference, *args.candidate)):
            raise ConfigError("--output must be a new file, never one of the input images")
        report = compare_images(args.reference, args.candidate, expected_size=expected)
        write_report(report, args.output)
    except ConfigError as exc:
        print(f"comparison refused: {exc}", file=sys.stderr)
        return int(ExitCode.CONFIG)
    except EngineError as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return int(ExitCode.UNEXPECTED)
    print(json.dumps({"report": args.output.name, "candidates": len(report["candidates"])},
                     sort_keys=True))
    return int(ExitCode.OK)


if __name__ == "__main__":
    sys.exit(main())
