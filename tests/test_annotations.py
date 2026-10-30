from types import SimpleNamespace

from PIL import Image, ImageChops, ImageDraw

from src.annotate import annotate_image


def test_annotations_draw_boxes_ids_labels_and_matching_total(monkeypatch):
    image = Image.new("RGB", (400, 300), "white")
    objects = [SimpleNamespace(id="obj_001", label="red helmet", bbox={"x1": 50, "y1": 100, "x2": 160, "y2": 210})]
    calls = []
    original = ImageDraw.ImageDraw.text

    def record(self, xy, text, *args, **kwargs):
        calls.append((xy, text))
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    result = annotate_image(image, objects)
    assert result.size == image.size
    assert result.mode == "RGB"
    assert result.getpixel((50, 160)) != (255, 255, 255)
    assert image.getpixel((50, 160)) == (255, 255, 255)
    assert "red helmet #001" in [text for _, text in calls]
    assert "Total detections: 1" in [text for _, text in calls]


def test_zero_detection_annotation_and_long_edge_labels_remain_in_bounds(monkeypatch):
    image = Image.new("RGB", (64, 64), "white")
    seen = []
    texts = []
    original = ImageDraw.ImageDraw.rectangle
    original_text = ImageDraw.ImageDraw.text

    def record(self, xy, *args, **kwargs):
        seen.append(xy)
        return original(self, xy, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "rectangle", record)
    def record_text(self, xy, text, *args, **kwargs):
        texts.append(text)
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    obj = SimpleNamespace(id="obj_123", label="very long custom category " * 4, bbox={"x1": 62, "y1": 63, "x2": 64, "y2": 64})
    result = annotate_image(image, [obj])
    assert f"{obj.label} #123" in texts
    assert all(0 <= x1 <= x2 < 64 and 0 <= y1 <= y2 < 64 for x1, y1, x2, y2 in seen)
    assert result.size == (64, 64)
    assert ImageChops.difference(image, annotate_image(image, [])).getbbox()


def test_nearby_top_edge_objects_keep_distinct_visible_label_cards(monkeypatch):
    image = Image.new("RGB", (360, 220), "white")
    objects = [
        SimpleNamespace(id="obj_001", label="red helmet", bbox={"x1": 20, "y1": 0, "x2": 80, "y2": 80}),
        SimpleNamespace(id="obj_002", label="red helmet", bbox={"x1": 25, "y1": 0, "x2": 100, "y2": 100}),
        SimpleNamespace(id="obj_003", label="forklift", bbox={"x1": 30, "y1": 0, "x2": 180, "y2": 160}),
    ]
    cards = []
    text_calls = []
    paint_order = []
    rectangle = ImageDraw.ImageDraw.rectangle
    text = ImageDraw.ImageDraw.text

    def record_rectangle(self, xy, *args, **kwargs):
        if kwargs.get("fill") is not None:
            cards.append(tuple(xy))
            paint_order.append("label")
        elif kwargs.get("outline") is not None:
            paint_order.append("box")
        return rectangle(self, xy, *args, **kwargs)

    def record_text(self, xy, value, *args, **kwargs):
        text_calls.append(value)
        return text(self, xy, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "rectangle", record_rectangle)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    annotated = annotate_image(image, objects)
    assert annotated.size == image.size
    assert {"red helmet #001", "red helmet #002", "forklift #003"} <= set(text_calls)
    assert len(cards) >= 3
    assert paint_order[:3] == ["box"] * 3
    for index, first in enumerate(cards):
        for second in cards[index + 1:]:
            assert first[2] < second[0] or second[2] < first[0] or first[3] < second[1] or second[3] < first[1]
