"""Contrast-class suppression: extra classes compete, then never reach the output."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.detector import MAX_MODEL_CLASSES, YOLOEDetector, contrast_classes
from src.postprocess import normalize_detections


class TensorValues:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeYOLOE:
    """Returns the boxes queued for the next predict call, labeled by class index."""

    boxes = ([], [], [])

    def __init__(self, checkpoint):
        self.classes = []
        self.calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def set_classes(self, classes):
        self.classes = list(classes)
        self.calls.append(("classes", list(classes)))

    def predict(self, **kwargs):
        boxes, confidences, indices = FakeYOLOE.boxes
        return [SimpleNamespace(
            names=dict(enumerate(self.classes)),
            boxes=SimpleNamespace(
                xyxy=TensorValues(boxes),
                conf=TensorValues(confidences),
                cls=TensorValues(indices),
            ),
        )]


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    checkpoint = tmp_path / "yoloe-26x-seg.pt"
    checkpoint.write_bytes(b"test checkpoint")
    encoder = tmp_path / "mobileclip2_b.ts"
    encoder.write_bytes(b"test encoder")
    # Every test starts with no queued boxes so fake state cannot leak.
    monkeypatch.setattr(FakeYOLOE, "boxes", ([], [], []))
    return YOLOEDetector(checkpoint, encoder=encoder, model_factory=FakeYOLOE, device="cpu")


def detect(adapter, classes, boxes, confidences, indices):
    FakeYOLOE.boxes = (boxes, confidences, indices)
    return adapter.detect(Image.new("RGB", (200, 200)), classes, 0.25, 640)


def test_woman_prompt_suppresses_overlapping_man_box(adapter):
    # Index 0 = requested "woman", index 1 = contrast "man".
    objects = detect(adapter, ["woman"], [[10, 10, 100, 180], [10, 10, 100, 180]], [0.62, 0.95], [0, 1])
    assert adapter.model.classes == ["woman", "man", "girl", "boy"]
    assert objects == []


def test_contrast_box_never_appears_without_overlap(adapter):
    # Woman at left, man at right: both are real, but only the woman is requested.
    objects = detect(adapter, ["woman"], [[10, 10, 90, 100], [110, 10, 190, 100]], [0.7, 0.9], [0, 1])
    assert [o.label for o in objects] == ["woman"]
    assert objects[0].prompt_class_index == 0


def test_lower_confidence_contrast_box_keeps_requested_box(adapter):
    objects = detect(adapter, ["woman"], [[10, 10, 100, 180], [12, 12, 98, 178]], [0.8, 0.5], [0, 1])
    assert [o.label for o in objects] == ["woman"]


def test_prompt_without_contrast_class_is_unchanged(adapter):
    objects = detect(adapter, ["forklift"], [[10, 10, 100, 180]], [0.8], [0])
    assert adapter.model.classes == ["forklift"]
    assert [o.label for o in objects] == ["forklift"]


def test_requesting_both_halves_sends_no_duplicate_contrast(adapter):
    boxes = [[10, 10, 90, 100], [110, 10, 190, 100]]
    objects = detect(adapter, ["man", "woman"], boxes, [0.9, 0.8], [0, 1])
    assert adapter.model.classes[:2] == ["man", "woman"]
    assert "man" not in adapter.model.classes[2:] and "woman" not in adapter.model.classes[2:]
    assert [o.label for o in objects] == ["man", "woman"]
    assert [o.prompt_class_index for o in objects] == [0, 1]


def test_class_list_stays_within_model_limit_and_is_deterministic():
    assert contrast_classes(["woman"]) == ["man", "girl", "boy"]
    assert contrast_classes([]) == []
    assert len(contrast_classes(["woman", "man", "cat", "dog", "car", "truck"])) <= MAX_MODEL_CLASSES


def test_clipboard_prompt_can_suppress_a_laptop_label(adapter):
    # "laptop" on a clipboard shot was scored as a laptop with no competitor.
    boxes = [[10, 10, 100, 180], [12, 12, 98, 178]]
    objects = detect(adapter, ["laptop"], boxes, [0.55, 0.91], [0, 1])
    assert adapter.model.classes == ["laptop", "clipboard", "tablet", "book", "folder", "notebook"]
    assert objects == []


@pytest.mark.parametrize("requested,expected", [
    ("clipboard", ["laptop", "tablet", "book", "folder", "notebook"]),
    ("tablet", ["laptop", "phone", "clipboard", "notebook"]),
    ("phone", ["tablet", "remote"]),
    ("remote", ["phone", "tablet"]),
    ("bottle", ["cup", "can"]),
    ("cup", ["bottle", "mug", "can"]),
    ("chair", ["sofa", "bench"]),
    ("sofa", ["chair", "bed"]),
    ("bag", ["backpack", "suitcase"]),
    ("backpack", ["bag", "suitcase"]),
])
def test_common_confusions_have_curated_contrasts(requested, expected):
    assert contrast_classes([requested]) == expected


def test_every_contrast_entry_is_self_consistent():
    # No entry may list itself, repeat a name, or overflow the per-request cap.
    from src.detector import _CONTRAST
    for name, extras in _CONTRAST.items():
        assert name not in extras, name
        assert len(extras) == len(set(extras)), name
        assert 1 + len(contrast_classes([name])) <= MAX_MODEL_CLASSES, name


def test_contrast_classes_must_not_duplicate_requested_classes():
    with pytest.raises(ValueError):
        normalize_detections([[0, 0, 5, 5]], [0.9], [0], {0: "woman", 1: "woman"},
                             ["woman"], 100, 100, 0.25, ["woman"])
    with pytest.raises(ValueError):
        normalize_detections([], [], [], {}, ["woman"], 100, 100, 0.25, ["man", "man"])


def test_contrast_mapping_is_validated_against_extended_names():
    with pytest.raises(ValueError, match="mapping"):
        normalize_detections([[0, 0, 5, 5]], [0.9], [1], {0: "woman", 1: "boy"},
                             ["woman"], 100, 100, 0.25, ["man"])


def test_contrast_ids_and_indices_are_never_exposed():
    # Requested box survives, so output IDs restart at obj_001 with requested labels only.
    objects = normalize_detections(
        [[10, 10, 90, 100], [110, 10, 190, 100], [130, 110, 190, 190]],
        [0.7, 0.9, 0.95], [0, 1, 2], {0: "woman", 1: "man", 2: "girl"},
        ["woman"], 200, 200, 0.25, ["man", "girl"],
    )
    assert [(o.id, o.label, o.prompt_class_index) for o in objects] == [("obj_001", "woman", 0)]
