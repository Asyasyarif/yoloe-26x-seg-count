"""Automatic contrast selection: nearest generic vocabulary, never synced to the request."""
from types import SimpleNamespace

import pytest
from PIL import Image

from src.contrast import (
    CONTRAST_PER_REQUEST,
    SYNONYM_SIMILARITY,
    cosine,
    load_vocab,
    normalize_label,
    select_contrast,
)
from src.detector import MAX_MODEL_CLASSES, YOLOEDetector
from src.postprocess import normalize_detections

# Tiny fake text space: cosine similarity stands in for MobileCLIP2 similarity.
# Angles are spread out so only the intended pairs sit above the duplicate cut.
FAKE_TEXT_SPACE = {
    "woman": [1.0, 0.0, 0.0, 0.0],
    "girl": [0.985, 0.174, 0.0, 0.0],   # synonym of the request, above the threshold
    "boy": [0.707, 0.707, 0.0, 0.0],    # nearest usable competitor (cos .71)
    "man": [0.342, 0.940, 0.0, 0.0],    # next nearest, cos .91 to boy: kept
    "goat": [0.0, 0.0, 1.0, 0.0],       # unrelated, chosen only to fill the quota
    "laptop": [0.0, 0.0, 0.0, 1.0],
    "clipboard": [0.0, 0.1, 0.6, 0.794],
    "tablet": [0.0, 0.2, 0.7, 0.68],
}


def embed(text: str) -> list[float]:
    """Deterministic stand-in for the model text encoder."""
    return FAKE_TEXT_SPACE.get(normalize_label(text), [0.0, 0.0, 0.0, 0.0])


def embeddings_for(names):
    return [embed(name) for name in names]


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

    def get_text_pe(self, texts):
        # Real shape is (1, len(texts), embed_dim); the adapter keeps the row.
        return [[TensorValues(embed(name)) for name in texts]]

    def set_classes(self, classes, embeddings=None):
        self.classes = list(classes)
        self.calls.append(("classes", list(classes), embeddings))

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
    detector = YOLOEDetector(checkpoint, encoder=encoder, model_factory=FakeYOLOE, device="cpu")
    # Production caches the shipped vocabulary in warmup; pin the fixture instead.
    detector.vocab = ["woman", "girl", "boy", "man", "goat", "laptop", "clipboard", "tablet"]
    detector.warmup()
    return detector


def detect(adapter, classes, boxes, confidences, indices):
    FakeYOLOE.boxes = (boxes, confidences, indices)
    return adapter.detect(Image.new("RGB", (200, 200)), classes, 0.25, 640)


# --- selection -------------------------------------------------------------


def test_near_duplicate_is_excluded_and_remaining_entries_are_ranked():
    vocab = ["woman", "girl", "boy", "man", "goat"]
    chosen = select_contrast(["woman"], embeddings_for(["woman"]), vocab, embeddings_for(vocab))
    assert chosen == ["boy", "man", "goat"]
    assert "girl" not in chosen  # cosine 0.999 >= SYNONYM_SIMILARITY


def test_synonym_threshold_is_what_removes_the_near_duplicate():
    similarity = {name: cosine(embed("woman"), embed(name)) for name in ("girl", "boy")}
    assert similarity["girl"] >= SYNONYM_SIMILARITY > similarity["boy"]
    assert select_contrast(["woman"], embeddings_for(["woman"]), ["girl", "boy"],
                           embeddings_for(["girl", "boy"])) == ["boy"]


def test_exact_and_plural_matches_are_excluded_even_with_a_distinct_embedding():
    # Plural spellings are exact matches after normalizing, so they never compete.
    vocab = ["Girls", "goat"]
    text_space = {"girl": [0.0, 1.0, 0.0], "goat": [0.7, 0.7, 0.0]}
    embeddings = [text_space[normalize_label(name)] for name in vocab]
    selected = select_contrast(["girl"], embeddings_for(["girl"]), vocab, embeddings, limit=8)
    assert selected == ["goat"]


def test_selection_is_deterministic_and_never_repeats_a_requested_name():
    vocab = ["woman", "girl", "boy", "man", "goat"]
    args = (["woman"], embeddings_for(["woman"]), vocab, embeddings_for(vocab))
    assert select_contrast(*args) == select_contrast(*args) == ["boy", "man", "goat"]
    assert select_contrast(*args, blocked=["boy"]) == ["man", "goat"]


def test_cap_and_empty_inputs():
    vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    embeddings = [[1.0, float(index), 0.0, 0.0] for index in range(len(vocab))]
    request = [[1.0, 0.0, 0.0, 0.0]]
    assert len(select_contrast(["woman"], request, vocab, embeddings, limit=3)) == 3
    assert select_contrast([], [], vocab, embeddings) == []
    assert select_contrast(["woman"], request, vocab, embeddings, limit=0) == []
    with pytest.raises(ValueError):
        select_contrast(["woman", "goat"], request, vocab, embeddings)


def test_multi_class_prompts_never_offer_a_requested_category_as_contrast():
    vocab = ["woman", "girl", "boy", "man", "laptop", "clipboard", "tablet"]
    embeddings = embeddings_for(vocab)
    selected = select_contrast(["woman", "laptop"], embeddings_for(["woman", "laptop"]), vocab, embeddings)
    assert "laptop" not in selected and "woman" not in selected
    # Both requested categories still get a competitor.
    assert {"boy", "clipboard"} <= set(selected)


def test_shipped_vocabulary_is_generic_and_distinct():
    vocab = load_vocab()
    assert len(vocab) > 1000
    assert len({normalize_label(name) for name in vocab}) == len(vocab)
    for generic in ("goat", "helmet", "clipboard"):
        assert generic in vocab


# --- suppression -----------------------------------------------------------


def test_lookalike_suppresses_overlapping_requested_box(adapter):
    # Index 0 = requested "woman", index 1 = automatic competitor "boy".
    objects = detect(adapter, ["woman"], [[10, 10, 100, 180], [10, 10, 100, 180]], [0.62, 0.95], [0, 1])
    assert adapter.model.classes[:1] == ["woman"]
    assert adapter.model.classes[1] == "boy"
    assert objects == []


def test_contrast_box_never_appears_without_overlap(adapter):
    objects = detect(adapter, ["woman"], [[10, 10, 90, 100], [110, 10, 190, 100]], [0.7, 0.9], [0, 1])
    assert [o.label for o in objects] == ["woman"]
    assert objects[0].prompt_class_index == 0


def test_lower_confidence_contrast_box_keeps_requested_box(adapter):
    objects = detect(adapter, ["woman"], [[10, 10, 100, 180], [12, 12, 98, 178]], [0.8, 0.5], [0, 1])
    assert [o.label for o in objects] == ["woman"]


def test_class_list_stays_within_model_limit(adapter):
    classes = [f"prompt{index}" for index in range(8)]
    detect(adapter, classes, [], [], [])
    sent = adapter.model.classes
    assert sent[:len(classes)] == classes
    assert len(sent) <= MAX_MODEL_CLASSES
    assert len(sent) - len(classes) <= CONTRAST_PER_REQUEST


def test_set_classes_receives_names_only_so_ultralytics_builds_the_prompt_tensor(adapter):
    detect(adapter, ["woman"], [], [], [])
    _, classes, embeddings = adapter.model.calls[-1]
    assert classes[0] == "woman"
    assert len(classes) > 1  # automatic contrast tail present
    # Ultralytics embeds these <=16 names itself and owns the 3-D prompt tensor.
    assert embeddings is None


def test_detection_without_warmup_fails_before_inference(tmp_path):
    detector = YOLOEDetector(_checkpoint(tmp_path), encoder=_encoder(tmp_path),
                             model_factory=FakeYOLOE, device="cpu")
    with pytest.raises(RuntimeError, match="warmup"):
        detector.detect(Image.new("RGB", (64, 64)), ["woman"], 0.25, 640)


def _checkpoint(tmp_path):
    path = tmp_path / "yoloe-26x-seg.pt"
    path.write_bytes(b"test checkpoint")
    return path


def _encoder(tmp_path):
    path = tmp_path / "mobileclip2_b.ts"
    path.write_bytes(b"test encoder")
    return path


# --- postprocess guards ----------------------------------------------------


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
