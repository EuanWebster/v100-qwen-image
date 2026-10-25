"""ML-free image output helper: validated, capped, no-clobber publication.

Filename contract: ``qi21-DDMMYYYY-seed.<jpg|png|webp>``. Repeats in the SAME
directory append ``-2``, ``-3`` ...; different job directories are independent.
Publication reuses the anchored-FD atomic no-clobber hardlink helpers from
``encoder_worker``; existing files, symlinks and FIFOs are never overwritten.
Pillow is imported lazily inside the function; nothing here touches the network
or any ML stack at import time.
"""
from __future__ import annotations

import datetime as _dt
import io
import os
from pathlib import Path

from qwen_engine.encoder_worker import (
    EncoderSetupError,
    _parent,
    _publish_response,
)
from qwen_engine.errors import EngineError

MAX_ENCODED_BYTES = 256 * 1024 * 1024
_EXT_TO_FORMAT = {"jpg": "JPEG", "png": "PNG", "webp": "WEBP"}
_VALID_MODES = {"RGB", "RGBA"}


class _CappedBytesIO(io.BytesIO):
    """In-memory buffer that refuses to grow past ``cap`` bytes."""

    def __init__(self, cap: int):
        super().__init__()
        self.cap = cap

    def writable(self):
        return True

    def write(self, data):
        if max(self.tell() + len(data), len(self.getbuffer())) > self.cap:
            raise ValueError("encoded image exceeds the output byte cap")
        return super().write(data)


def _validate_seed(seed):
    if type(seed) is bool or type(seed) is not int or not 0 <= seed <= 2 ** 64 - 1:
        raise EngineError("seed must be an actual int in uint64 range [0, 2**64-1], not a bool")
    return seed


def _validate_date(date):
    if date is None:
        return _dt.date.today()
    # datetime.datetime passes isinstance(date) but is rejected: exact date type.
    if type(date) is not _dt.date:
        raise EngineError("date must be a datetime.date instance or None")
    return date


def _validate_image(image):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment without Pillow
        raise EngineError(f"Pillow is required to save images: {exc}") from exc
    if not isinstance(image, Image.Image):
        raise EngineError("image must be a PIL.Image.Image instance")
    if image.mode not in _VALID_MODES:
        raise EngineError(f"image mode {image.mode!r} is not supported (need RGB/RGBA-like)")
    if image.width < 1 or image.height < 1:
        raise EngineError("image must have positive width and height")
    return image


def _encode(image, fmt: str, cap: int) -> bytes:
    buffer = _CappedBytesIO(cap)
    if fmt == "JPEG":
        # JPEG cannot carry alpha: flatten transparency onto white explicitly.
        if image.mode in ("RGBA", "LA", "P"):
            flat = image.convert("RGBA")
            from PIL import Image as _Image
            background = _Image.new("RGB", image.size, (255, 255, 255))
            background.paste(flat, mask=flat.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=95, subsampling=0)
    elif fmt == "PNG":
        image.save(buffer, format="PNG")  # alpha preserved as-is
    else:  # WEBP: lossless keeps alpha intact
        image.save(buffer, format="WEBP", lossless=True, quality=100)
    return buffer.getvalue()


def _candidate_names(stem: str, ext: str):
    yield f"{stem}.{ext}"
    suffix = 2
    while suffix <= 1000:
        yield f"{stem}-{suffix}.{ext}"
        suffix += 1


def save_output_image(image, output_dir, output_format, seed, *, date=None,
                      check_cancelled=None, max_bytes=None) -> Path:
    """Encode ``image`` (PIL) and publish it into ``output_dir``; return the path.

    Raises typed ``EngineError`` subclasses for invalid inputs, byte-cap
    overflow, or unsafe destinations; propagates ``GenerationCancelled`` from
    ``check_cancelled`` before encoding and before publishing.
    """
    if output_dir is None:
        raise EngineError("output_dir is required")
    # output_format must be a plain str: bool/int/None are invalid inputs.
    if type(output_format) is not str or output_format not in _EXT_TO_FORMAT:
        raise EngineError(f"unsupported output_format {output_format!r} (use jpg/png/webp)")
    if max_bytes is None:
        max_bytes = MAX_ENCODED_BYTES
    if type(max_bytes) is not int or max_bytes < 1:
        raise EngineError("max_bytes must be a positive int")
    if check_cancelled is not None and not callable(check_cancelled):
        raise EngineError("check_cancelled must be a callable or None")
    seed = _validate_seed(seed)
    day = _validate_date(date)
    image = _validate_image(image)
    fmt = _EXT_TO_FORMAT[output_format]
    stem = f"qi21-{day.strftime('%d%m%Y')}-{seed}"
    out_dir = Path(output_dir)
    if not out_dir.is_dir():
        raise EngineError(f"output_dir does not exist or is not a directory: {out_dir}")

    if check_cancelled is not None:
        check_cancelled()
    try:
        payload = _encode(image, fmt, max_bytes)
    except ValueError as exc:
        raise EngineError(f"encoded image refused: {exc}") from exc
    except EngineError:
        raise
    except Exception as exc:
        raise EngineError(f"image encoding failed: {exc}") from exc

    if check_cancelled is not None:
        check_cancelled()
    try:
        with _parent(out_dir / "image-output-placeholder") as (parent, _unused, directory):
            for name in _candidate_names(stem, output_format):
                try:
                    os.stat(name, dir_fd=directory, follow_symlinks=False)
                    exists = True
                except FileNotFoundError:
                    exists = False
                if exists:
                    continue  # same-dir repeat: try the next -N suffix
                try:
                    _publish_response(parent, name, directory, payload)
                except EncoderSetupError as exc:
                    # _publish_response wraps FileExistsError (lost race or an
                    # unsafe existing target): inspect __cause__ to tell a
                    # retry collision from a real refusal, then try next name.
                    if not isinstance(exc.__cause__, FileExistsError):
                        raise
                    continue
                return parent / name
    except EngineError:
        raise
    except EncoderSetupError as exc:
        raise EngineError(f"cannot publish output image: {exc} (cause: {exc.__cause__!r})") from exc
    except (OSError, ValueError, TypeError) as exc:
        raise EngineError(f"cannot publish output image: {exc}") from exc
    raise EngineError(
        f"refusing to overwrite more than 1000 existing qi21 variants in {out_dir}"
    )
