"""CPU-only guarded tests for the offline saved-image comparison tool.

Real (tiny) Pillow images are written to tmp_path and compared through the
actual CLI; the module itself must stay free of torch/diffusers imports.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from qwen_engine.config import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "compare_benchmark_images.py"


def load_module(monkeypatch=None):
    spec = importlib.util.spec_from_file_location("compare_images_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, image, **save_kwargs):
    image.save(path, format="PNG", **save_kwargs)
    return path


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT), *map(str, args)],
        capture_output=True, text=True)


def test_module_never_imports_torch_or_pillow_at_import():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "bad = [n for n in sys.modules if n.startswith(('torch', 'diffusers', 'PIL'))]\n"
        "sys.exit(1 if bad else 0)\n")
    assert subprocess.run([sys.executable, "-B", "-c", code]).returncode == 0


def test_equal_pixels_despite_different_png_metadata(tmp_path):
    pixels = Image.new("RGB", (4, 3))
    pixels.putdata([(x * 7 % 256, 9, 200) for x in range(12)])
    info = PngInfo()
    info.add_text("generator", "run-a")
    reference = save(tmp_path / "ref.png", pixels, pnginfo=info)
    other = PngInfo()
    other.add_text("generator", "run-b")
    candidate = save(tmp_path / "cand.png", pixels, pnginfo=other)
    module = load_module()
    report = module.compare_images(reference, [candidate])
    entry = report["candidates"][0]
    assert entry["exact_pixels_equal"] is True
    assert entry["max_channel_delta"] == 0
    assert entry["mean_channel_delta"] == 0.0
    assert entry["changed_pixel_fraction"] == 0.0
    assert entry["encoded_sha256"] != report["reference"]["encoded_sha256"]
    assert entry["pixel_sha256"] == report["reference"]["pixel_sha256"]


def test_one_changed_pixel_metrics_are_exact(tmp_path):
    base = [(0, 0, 0)] * 4
    changed = list(base)
    changed[2] = (10, 20, 30)
    reference_pixels = Image.new("RGB", (2, 2))
    reference_pixels.putdata(base)
    candidate_pixels = Image.new("RGB", (2, 2))
    candidate_pixels.putdata(changed)
    reference = save(tmp_path / "ref.png", reference_pixels)
    candidate = save(tmp_path / "cand.png", candidate_pixels)
    report = load_module().compare_images(reference, [candidate])
    entry = report["candidates"][0]
    assert (entry["max_channel_delta"], entry["mean_channel_delta"]) == (30, 60 / 12)
    assert entry["per_channel_max_delta"] == [10, 20, 30]
    assert entry["changed_pixel_fraction"] == 1 / 4
    assert entry["exact_pixels_equal"] is False


def test_alpha_is_preserved_and_measured_only_for_rgba(tmp_path):
    rgb = Image.new("RGBA", (2, 2), (10, 20, 30, 255))
    other = rgb.copy()
    other.putpixel((1, 1), (10, 20, 30, 0))  # identical RGB, different alpha edge
    reference = save(tmp_path / "ref.png", rgb)
    candidate = save(tmp_path / "cand.png", other)
    report = load_module().compare_images(reference, [candidate])
    entry = report["candidates"][0]
    assert entry["alpha"] == {"alpha_uniform": False, "alpha_min": 0,
                              "alpha_max": 255, "alpha_mean": 3 * 255 / 4}
    assert report["reference"]["alpha"]["alpha_uniform"] is True
    assert entry["max_channel_delta"] == 255
    assert entry["changed_pixel_fraction"] == 1 / 4


def test_alpha_free_rgb_report_omits_alpha(tmp_path):
    reference = save(tmp_path / "ref.png", Image.new("RGB", (2, 2)))
    report = load_module().compare_images(reference, [reference])
    assert "alpha" not in report["candidates"][0]
    assert "alpha" not in report["reference"]


@pytest.mark.parametrize("size", [(5, 4), (4, 5)])
def test_mismatched_dimensions_are_refused(tmp_path, size):
    reference = save(tmp_path / "ref.png", Image.new("RGB", (4, 4)))
    candidate = save(tmp_path / "cand.png", Image.new("RGB", size))
    with pytest.raises(ConfigError, match="no scaling or cropping"):
        load_module().compare_images(reference, [candidate])


def test_mode_mismatch_is_refused_with_neutral_explanation(tmp_path):
    # RGB PNG vs RGBA PNG: identical dimensions, differ only in mode, both lossless.
    reference = save(tmp_path / "ref.png", Image.new("RGBA", (3, 3)))
    candidate = save(tmp_path / "cand.png", Image.new("RGB", (3, 3)))
    result = run_cli("--reference", reference, "--candidate", candidate,
                     "--output", tmp_path / "out.json")
    assert result.returncode == 2
    assert "the modes differ" in result.stderr
    assert "implicit channel conversion" in result.stderr
    assert "lossy" not in result.stderr
    assert not (tmp_path / "out.json").exists()


def test_non_rgb_modes_are_refused_not_converted(tmp_path):
    reference = save(tmp_path / "ref.png", Image.new("P", (2, 2)))
    with pytest.raises(ConfigError, match="only RGB and RGBA"):
        load_module().compare_images(reference, [reference])


def test_non_image_and_truncated_files_are_refused(tmp_path):
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"this is not an image at all")
    with pytest.raises(ConfigError, match="not a readable image"):
        load_module().compare_images(junk, [junk])
    noisy = Image.effect_noise((32, 32), 64).convert("RGB")
    full = save(tmp_path / "full.png", noisy)
    truncated = tmp_path / "truncated.png"
    payload = full.read_bytes()
    truncated.write_bytes(payload[: max(16, len(payload) // 2)])
    with pytest.raises(ConfigError, match="truncated or corrupt"):
        load_module().compare_images(truncated, [truncated])


def _png_header(width, height):
    """Tiny PNG whose IHDR only CLAIMS a huge size (no pixel payload)."""
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b""))


def test_decompression_bomb_and_oversized_claims_are_refused(tmp_path):
    bomb = tmp_path / "bomb.png"
    bomb.write_bytes(_png_header(50000, 50000))
    with pytest.raises(ConfigError, match="decompression-bomb"):
        load_module().compare_images(bomb, [bomb])
    tall = tmp_path / "tall.png"
    tall.write_bytes(_png_header(10, 5000))
    with pytest.raises(ConfigError, match="exceeds the supported 4096"):
        load_module().compare_images(tall, [tall])


def test_input_byte_cap_and_dimension_cap_are_enforced(tmp_path, monkeypatch):
    reference = save(tmp_path / "ref.png", Image.new("RGB", (8, 8)))
    module = load_module()
    monkeypatch.setattr(module, "MAX_IMAGE_BUFFER_BYTES", 64)
    with pytest.raises(ConfigError, match="cannot read image input"):
        module.compare_images(reference, [reference])
    long_strip = Image.new("RGB", (4097, 1))
    strip_path = save(tmp_path / "strip.png", long_strip)
    with pytest.raises(ConfigError, match="exceeds the supported 4096"):
        load_module().compare_images(strip_path, [strip_path])


def test_different_container_is_flagged_but_lossless_webp_stays_exact(tmp_path):
    reference = save(tmp_path / "ref.png", Image.new("RGB", (4, 4), (7, 8, 9)))
    jpeg = tmp_path / "cand.jpg"
    Image.new("RGB", (4, 4), (7, 8, 9)).save(jpeg, format="JPEG", quality=95)
    webp = tmp_path / "cand.webp"
    # the engine's own WebP path is LOSSLESS (image_output.py: lossless=True)
    Image.new("RGB", (4, 4), (7, 8, 9)).save(webp, format="WEBP", lossless=True)
    entries = load_module().compare_images(reference, [jpeg, webp])["candidates"]
    assert entries[0]["format"] == "JPEG"
    assert "may include codec differences" in entries[0]["encoding_note"]
    assert "WebP may be lossy or lossless" in entries[0]["encoding_note"]
    assert entries[1]["format"] == "WEBP"
    assert entries[1]["exact_pixels_equal"] is True
    assert entries[1]["max_channel_delta"] == 0
    assert entries[1]["encoding_note"] is not None  # still flagged: container differs


def test_report_is_index_labelled_and_free_of_paths_prompts_and_seeds(tmp_path):
    reference = save(tmp_path / "ref.png", Image.new("RGB", (2, 2)))
    first = save(tmp_path / "a.png", Image.new("RGB", (2, 2)))
    second = tmp_path / "b.png"
    Image.new("RGB", (2, 2)).save(second, format="PNG")
    output = tmp_path / "report.json"
    result = run_cli("--reference", reference, "--candidate", first,
                     "--candidate", second, "--output", output, "--expect-size", "2x2")
    assert result.returncode == 0, result.stderr
    text = output.read_text()
    report = json.loads(text)
    assert [entry["label"] for entry in report["candidates"]] == ["candidate-1", "candidate-2"]
    for absent in ("seed", "prompt", str(tmp_path), "candidate-1.png"):
        assert absent not in text
    assert report["schema"] == "qwen.image-compare"
    assert report["limitations"]


def test_wrong_expected_size_and_output_collision_are_refused(tmp_path):
    reference = save(tmp_path / "ref.png", Image.new("RGB", (4, 4)))
    before = reference.read_bytes()
    output = tmp_path / "taken.json"
    output.write_text("earlier evidence")
    result = run_cli("--reference", reference, "--candidate", reference,
                     "--output", output)
    assert result.returncode != 0
    assert output.read_text() == "earlier evidence"
    # only the two intended files exist: the refused report left no partial/temp
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ref.png", "taken.json"]
    partial = tmp_path / "size.json"
    result = run_cli("--reference", reference, "--candidate", reference,
                     "--output", partial, "--expect-size", "8x8")
    assert result.returncode == 2
    assert not partial.exists()
    assert run_cli("--reference", reference, "--candidate", reference,
                   "--output", reference).returncode == 2
    # the input image is byte-for-byte untouched by the aliasing attempt
    assert reference.read_bytes() == before
