from pathlib import Path

import pytest
from PIL import Image

from src.image_io import load_image


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_accepts_decoded_formats_not_file_extension(tmp_path, fmt):
    path = tmp_path / "upload.bin"
    Image.new("RGB", (100, 60), "red").save(path, format=fmt)
    image = load_image(path)
    assert image.mode == "RGB"
    assert image.size == (100, 60)
    path.unlink()
    assert image.getpixel((0, 0))[0] > 200


def test_transparency_composited_on_white(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (64, 64), (0, 0, 255, 0)).save(path)
    assert load_image(path).getpixel((0, 0)) == (255, 255, 255)


def test_applies_exif_orientation(tmp_path):
    path = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (80, 40), "blue").save(path, exif=exif)
    assert load_image(path).size == (40, 80)
    assert load_image(path).getexif().get(274) is None


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_rejects_unsupported_format(tmp_path, fmt):
    path = tmp_path / "unsupported.png"
    Image.new("RGB", (64, 64)).save(path, format=fmt)
    with pytest.raises(ValueError, match="JPEG, PNG or WebP"):
        load_image(path)


def test_rejects_multiframe_png(tmp_path):
    path = tmp_path / "animated.png"
    frames = [Image.new("RGB", (64, 64), color) for color in ("red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    with pytest.raises(ValueError, match="single-frame"):
        load_image(path)


def test_rejects_corruption_and_truncation(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Invalid or corrupted"):
        load_image(path)
    Image.new("RGB", (64, 64)).save(path)
    path.write_bytes(path.read_bytes()[:50])
    with pytest.raises(ValueError, match="Invalid or corrupted"):
        load_image(path)


def test_size_limits_checked_before_decode(tmp_path):
    path = tmp_path / "large.png"
    Image.new("RGB", (64, 64)).save(path)
    with pytest.raises(ValueError, match="bytes"):
        load_image(path, max_bytes=10)
    with pytest.raises(ValueError, match="pixels"):
        load_image(path, max_pixels=100)
    with pytest.raises(ValueError, match="32"):
        Image.new("RGB", (31, 64)).save(path)
        load_image(path)


def test_decompression_bomb_rejected(tmp_path, monkeypatch):
    path = tmp_path / "bomb.png"
    Image.new("RGB", (64, 64)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 3000)
    with pytest.raises(ValueError, match="decompression"):
        load_image(path)


def test_only_local_regular_files_accepted(tmp_path):
    with pytest.raises(ValueError, match="local file"):
        load_image(tmp_path)
    with pytest.raises(ValueError, match="local file"):
        load_image("https://example.com/photo.png")
