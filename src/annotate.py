"""Draw the canonical detection list on the original oriented image."""

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from src.schema import DetectedObject


FONT_PATH = Path(__file__).resolve().parent.parent / "assets/fonts/DejaVuSans.ttf"
COLORS = ((0, 110, 210), (190, 55, 25), (15, 125, 65), (135, 55, 180), (145, 95, 0))


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _glyphs(
    text: str,
    font: ImageFont.FreeTypeFont,
    image_size: tuple[int, int],
) -> Image.Image:
    """Rasterize the complete label, scaling rather than dropping characters."""
    padding = 3
    bounds = font.getbbox(text)
    glyph_width = max(1, bounds[2] - bounds[0])
    glyph_height = max(1, bounds[3] - bounds[1])
    glyphs = Image.new("RGBA", (glyph_width, glyph_height))
    ImageDraw.Draw(glyphs).text((-bounds[0], -bounds[1]), text, font=font, fill="white")
    available_width = max(1, image_size[0] - 2 * padding)
    available_height = max(1, image_size[1] - 2 * padding)
    scale = min(1.0, available_width / glyph_width, available_height / glyph_height)
    if scale < 1.0:
        glyphs = glyphs.resize(
            (max(1, int(glyph_width * scale)), max(1, int(glyph_height * scale))),
            Image.Resampling.LANCZOS,
        )
    return glyphs


def _overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    return max(0, min(first[2], second[2]) - max(first[0], second[0]) + 1) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]) + 1
    )


def _placement(
    glyphs: Image.Image,
    image_size: tuple[int, int],
    preferred: list[tuple[int, int]],
    occupied: list[tuple[int, int, int, int]],
    *,
    allow_overlap: bool = True,
) -> tuple[int, int, int, int] | None:
    """Try local positions then free card edges; minimize overlap if space runs out."""
    width, height = glyphs.width + 6, glyphs.height + 6
    canvas_width, canvas_height = image_size
    candidates = list(preferred)
    candidates.extend(((0, 0), (canvas_width - width, 0), (0, canvas_height - height), (canvas_width - width, canvas_height - height)))
    for left, top, right, bottom in occupied:
        candidates.extend(((left, bottom + 2), (right + 2, top), (left - width - 1, top), (left, top - height - 1)))
    best = None
    best_overlap = float("inf")
    seen = set()
    for x, y in candidates:
        x = max(0, min(x, canvas_width - width))
        y = max(0, min(y, canvas_height - height))
        if (x, y) in seen:
            continue
        seen.add((x, y))
        rect = (x, y, x + width - 1, y + height - 1)
        overlap = sum(_overlap(rect, other) for other in occupied)
        if not overlap:
            return rect
        if overlap < best_overlap:
            best, best_overlap = rect, overlap
    return best if allow_overlap else None


def _paint_label(
    image: Image.Image,
    glyphs: Image.Image,
    rect: tuple[int, int, int, int],
    background: tuple[int, int, int],
) -> None:
    ImageDraw.Draw(image).rectangle(rect, fill=background)
    image.paste(glyphs, (rect[0] + 3, rect[1] + 3), glyphs)


def annotate_image(image: Image.Image, objects: Sequence["DetectedObject"]) -> Image.Image:
    """Copy the image and draw boxes with collision-aware category/ID labels.

    Labels try nearby positions and free card edges, with leader lines when
    displaced. Very crowded or tiny images may have insufficient free space:
    the fallback minimizes overlap and cannot guarantee all text is readable.
    Long labels retain every character but may shrink below readable size.
    JSON in ZIP mode preserves the exact labels and IDs in either case.
    """
    annotated = image.convert("RGB").copy()
    annotated.info.clear()
    draw = ImageDraw.Draw(annotated)
    scale = max(image.size)
    stroke = max(2, round(scale / 500))
    font = _font(max(12, min(48, round(scale / 45))))
    occupied = []
    labels = []
    for index, obj in enumerate(objects):
        color = COLORS[index % len(COLORS)]
        box = obj.bbox
        x1 = max(0, min(image.width - 1, box["x1"]))
        y1 = max(0, min(image.height - 1, box["y1"]))
        x2 = max(x1, min(image.width - 1, box["x2"]))
        y2 = max(y1, min(image.height - 1, box["y2"]))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=stroke)
        number = obj.id.removeprefix("obj_")
        text = f"{obj.label} #{number}"
        glyphs = _glyphs(text, font, image.size)
        height = glyphs.height + 6
        rect = _placement(glyphs, image.size, [(x1, y1 - height - 2), (x1, y2 + 2), (x2 + 2, y1)], occupied)
        occupied.append(rect)
        labels.append((glyphs, rect, color, (x1, y1)))

    # Draw all boxes and leader lines before cards, so later outlines cannot
    # paint over an earlier object's category or ID.
    for _, rect, color, anchor in labels:
        endpoint = (max(rect[0], min(anchor[0], rect[2])), max(rect[1], min(anchor[1], rect[3])))
        draw.line((anchor, endpoint), fill=color, width=stroke)
    for glyphs, rect, color, _ in labels:
        _paint_label(annotated, glyphs, rect, color)

    # The optional total never hides an object label.
    summary = _glyphs(f"Total detections: {len(objects)}", font, image.size)
    summary_rect = _placement(summary, image.size, [(4, 4)], occupied, allow_overlap=False)
    if summary_rect is not None:
        _paint_label(annotated, summary, summary_rect, (25, 30, 38))
    return annotated
