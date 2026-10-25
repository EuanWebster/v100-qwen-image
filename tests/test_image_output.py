"""CPU-only guarded tests for the ML-free image output helper."""
from __future__ import annotations

import datetime
import io
import os
import stat
from types import SimpleNamespace

import pytest

from PIL import Image

from qwen_engine import image_output
from qwen_engine.errors import EngineError, GenerationCancelled
from qwen_engine.image_output import save_output_image

DATE = datetime.date(2025, 3, 9)
STEM = "qi21-09032025-42"


def rgb_image(size=(9, 5)):
    return Image.new("RGB", size, (10, 120, 240))


def rgba_image(size=(7, 4)):
    return Image.new("RGBA", size, (255, 0, 0, 0))  # fully transparent red


def decoded(path):
    with Image.open(path) as got:
        got.load()
        fmt, size, mode = got.format, got.size, got.mode
    # Reopen after close: .format is only set while the file handle is open.
    with Image.open(path) as again:
        again.load()
        assert again.format == fmt and again.size == size
        image = again.copy()
        return SimpleNamespace(image=image, format=fmt, size=size, mode=mode, getpixel=image.getpixel)


def test_jpg_roundtrip_exact_seed_filename(tmp_path):
    out = save_output_image(rgb_image(), tmp_path, "jpg", 42, date=DATE)
    assert out.name == f"{STEM}.jpg"
    assert out == tmp_path / out.name
    got = decoded(out)
    assert got.format == "JPEG" and got.size == (9, 5) and got.mode == "RGB"


def test_uint64_max_seed_and_date_in_name(tmp_path):
    out = save_output_image(rgb_image((3, 3)), tmp_path, "png", 2 ** 64 - 1,
                            date=datetime.date(1999, 12, 31))
    assert out.name == "qi21-31121999-18446744073709551615.png"
    assert decoded(out).format == "PNG"


def test_default_date_is_today(tmp_path, monkeypatch):
    today = datetime.date.today()
    out = save_output_image(rgb_image((2, 2)), tmp_path, "webp", 7)
    assert out.name in {f"qi21-{day:%d%m%Y}-7.webp" for day in (today, datetime.date.today())}
    assert decoded(out).format == "WEBP"


def test_same_dir_repeat_appends_suffix_no_overwrite(tmp_path):
    first = save_output_image(rgb_image((4, 4)), tmp_path, "png", 42, date=DATE)
    before = first.stat()
    second = save_output_image(rgba_image((4, 4)), tmp_path, "png", 42, date=DATE)
    third = save_output_image(rgb_image((4, 4)), tmp_path, "png", 42, date=DATE)
    assert [first.name, second.name, third.name] == [f"{STEM}.png", f"{STEM}-2.png", f"{STEM}-3.png"]
    assert first.stat().st_ino == before.st_ino  # never replaced
    assert decoded(second).mode == "RGBA"


def test_different_dirs_independent_same_name(tmp_path):
    a, b = tmp_path / "jobA", tmp_path / "jobB"
    a.mkdir(); b.mkdir()
    one = save_output_image(rgb_image((3, 3)), a, "png", 42, date=DATE)
    two = save_output_image(rgb_image((3, 3)), b, "png", 42, date=DATE)
    assert one.name == two.name == f"{STEM}.png"
    assert os.stat(one).st_ino != os.stat(two).st_ino


def test_cancellation_before_encoding_and_before_publish(tmp_path):
    calls = []

    def cancel_once():
        calls.append(1)
        if len(calls) == 2:  # survive the pre-encode probe, refuse at publish
            raise GenerationCancelled("cancelled before publish")

    with pytest.raises(GenerationCancelled):
        save_output_image(rgb_image(), tmp_path, "png", 42, date=DATE,
                          check_cancelled=cancel_once)
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []  # no partial final files


def test_png_webp_preserve_alpha_jpg_flattens_white(tmp_path):
    png = save_output_image(rgba_image(), tmp_path, "png", 1, date=DATE)
    webp = save_output_image(rgba_image(), tmp_path, "webp", 1, date=DATE)
    assert decoded(png).mode == "RGBA"
    assert decoded(webp).mode == "RGBA"
    jpg = save_output_image(rgba_image(), tmp_path, "jpg", 1, date=DATE)
    flat = decoded(jpg)
    assert flat.format == "JPEG" and flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 255, 255)  # transparent -> white


def _varying_alpha_image():
    """Every pixel a DIFFERENT alpha value: a channel dropped or flattened
    anywhere in the encode path is visible, not averaged away."""
    image = Image.new("RGBA", (4, 2))
    alphas = [0, 1, 17, 64, 127, 128, 200, 255]
    for index, alpha in enumerate(alphas):
        image.putpixel((index % 4, index // 4), (30 + index * 10, 200 - index * 5,
                                                90, alpha))
    return image


@pytest.mark.parametrize("fmt", ["png", "webp"])
def test_alpha_channel_survives_lossless_roundtrip_exactly(tmp_path, fmt):
    source = _varying_alpha_image()
    out = save_output_image(source, tmp_path, fmt, 3, date=DATE)
    got = decoded(out)
    assert got.mode == "RGBA" and got.size == source.size
    pixels = [(x, y) for y in range(source.height) for x in range(source.width)]
    expected = [source.getpixel(p) for p in pixels]
    # A fully transparent pixel's RGB is not observable and WebP may zero it;
    # every NON-ZERO alpha pixel must round-trip with exact alpha and RGB.
    actual = [got.image.convert("RGBA").getpixel(p) for p in pixels]
    assert len(actual) == len(expected)
    assert sorted(pixel[3] for pixel in actual) == sorted(pixel[3] for pixel in expected)
    for want, have in zip(expected, actual):
        if want[3] == 0:
            assert have[3] == 0
            continue
        assert have == want


def test_opaque_rgb_png_webp_bytes_unchanged_by_alpha_support(tmp_path):
    # The pre-existing opaque path must be untouched: the module publishes
    # exactly what Pillow encodes for the same RGB image and settings.
    source = rgb_image()
    for fmt, pillow_format, kwargs in (("png", "PNG", {}),
                                       ("webp", "WEBP", {"lossless": True, "quality": 100})):
        direct = io.BytesIO()
        source.save(direct, format=pillow_format, **kwargs)
        out = save_output_image(source.copy(), tmp_path, fmt, 11, date=DATE)
        assert out.read_bytes() == direct.getvalue(), f"{fmt} bytes drifted"


def test_existing_file_never_overwritten(tmp_path):
    target = tmp_path / f"{STEM}.png"
    target.write_bytes(b"sentinel")
    out = save_output_image(rgb_image((3, 3)), tmp_path, "png", 42, date=DATE)
    assert out.name == f"{STEM}-2.png"
    assert target.read_bytes() == b"sentinel"


def test_symlink_and_fifo_targets_refused(tmp_path):
    link = tmp_path / f"{STEM}.png"
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"do-not-touch")
    link.symlink_to(victim.name)
    out = save_output_image(rgb_image((3, 3)), tmp_path, "png", 42, date=DATE)
    assert out.name == f"{STEM}-2.png"
    assert victim.read_bytes() == b"do-not-touch"
    fifo = tmp_path / f"{STEM}-2.png"
    fifo.unlink()
    os.mkfifo(fifo)
    out2 = save_output_image(rgb_image((3, 3)), tmp_path, "png", 42, date=DATE)
    assert out2.name == f"{STEM}-3.png"
    assert stat.S_ISFIFO(os.lstat(fifo).st_mode)


def test_lost_race_publish_collision_survives(tmp_path, monkeypatch):
    real = image_output._publish_response

    def flaky(parent, name, directory, raw):
        if name == f"{STEM}.png":
            raise image_output.EncoderSetupError("wrapped") from FileExistsError("race")
        return real(parent, name, directory, raw)

    monkeypatch.setattr(image_output, "_publish_response", flaky)
    out = save_output_image(rgb_image((3, 3)), tmp_path, "png", 42, date=DATE)
    assert out.name == f"{STEM}-2.png"


def test_bad_inputs_raise_typed_engine_error(tmp_path):
    img = rgb_image((2, 2))
    with pytest.raises(EngineError):
        save_output_image(img, tmp_path, "jpeg", 1, date=DATE)  # alias not accepted
    with pytest.raises(EngineError):
        save_output_image(img, tmp_path, "tiff", 1, date=DATE)
    with pytest.raises(EngineError):
        save_output_image(img, tmp_path, "png", True, date=DATE)  # bool seed
    with pytest.raises(EngineError):
        save_output_image(img, tmp_path, "png", 1.5, date=DATE)
    with pytest.raises(EngineError):
        save_output_image(img, tmp_path, "png", 2 ** 64, date=DATE)
    with pytest.raises(EngineError):
        save_output_image(img, tmp_path, "png", -1, date=DATE)
    with pytest.raises(EngineError):
        save_output_image(img, tmp_path, "png", 1, date="2025-03-09")
    with pytest.raises(EngineError):
        save_output_image(img, tmp_path, "png", 1, date=datetime.datetime(2025, 3, 9))
    with pytest.raises(EngineError):
        save_output_image("not-an-image", tmp_path, "png", 1, date=DATE)
    with pytest.raises(EngineError):
        save_output_image(Image.new("CMYK", (2, 2)), tmp_path, "png", 1, date=DATE)
    with pytest.raises(EngineError):
        save_output_image(img, tmp_path / "missing-dir", "png", 1, date=DATE)
    with pytest.raises(EngineError):
        save_output_image(img, tmp_path, "png", 1, date=DATE, check_cancelled="nope")


def test_byte_cap_refuses_overflow_no_files_left(tmp_path, monkeypatch):
    monkeypatch.setattr(image_output, "MAX_ENCODED_BYTES", 64)  # docs: cap constant
    with pytest.raises(EngineError):
        save_output_image(rgb_image((64, 64)), tmp_path, "png", 42, date=DATE)
    with pytest.raises(EngineError):
        save_output_image(rgb_image((64, 64)), tmp_path, "png", 42, date=DATE, max_bytes=64)
    assert list(tmp_path.iterdir()) == []


def test_module_import_is_ml_free():
    import subprocess
    import sys
    code = ("import sys; import qwen_engine.image_output;"
            " sys.exit(1 if any(m.startswith(('torch','PIL')) for m in sys.modules) else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
