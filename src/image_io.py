"""Bounded, local-only image decoding before model inference."""

from pathlib import Path
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MIN_IMAGE_DIMENSION = 32
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


def load_image(
    path: str | Path,
    *,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
    min_dimension: int = MIN_IMAGE_DIMENSION,
) -> Image.Image:
    """Return an independent, oriented RGB image with white alpha compositing.

    Limits are checked before loading pixels; Pillow's decompression-bomb
    warnings are also errors. Only a single still frame is accepted.
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError("Image must be an existing local file (JPEG, PNG or WebP).")
    try:
        with source.open("rb") as file, warnings.catch_warnings():
            file.seek(0, 2)
            size_bytes = file.tell()
            file.seek(0)
            if size_bytes > max_bytes:
                raise ValueError(f"Image exceeds the {max_bytes} bytes limit.")
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(file) as opened:
                if opened.format not in SUPPORTED_FORMATS:
                    raise ValueError("Only JPEG, PNG or WebP images are supported.")
                width, height = opened.size
                if width * height > max_pixels:
                    raise ValueError(f"Image exceeds the {max_pixels} pixels limit.")
                if min(width, height) < min_dimension:
                    raise ValueError(f"Each image dimension must be at least {min_dimension} pixels.")
                if getattr(opened, "n_frames", 1) != 1:
                    raise ValueError("Only single-frame images are supported.")
                opened.verify()

            file.seek(0)
            with Image.open(file) as opened:
                oriented = ImageOps.exif_transpose(opened)
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    result = Image.alpha_composite(background, rgba).convert("RGB")
                else:
                    result = oriented.convert("RGB")
                result.load()
                # Do not propagate EXIF or other source metadata to public outputs.
                result.info.clear()
                return result
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError("Image rejected by decompression-bomb protection.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("Invalid or corrupted image; supply a complete JPEG, PNG or WebP file.") from exc
