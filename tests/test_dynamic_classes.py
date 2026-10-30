"""CPU tests exercise the adapter boundary without downloading model weights."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest
from PIL import Image

from src.detector import YOLOEDetector


class TensorValues:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeYOLOE:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.classes = []
        self.calls = []
        self.active = False
        self.lock = threading.Lock()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True
        return self

    def set_classes(self, classes):
        with self.lock:
            assert not self.active, "set_classes raced with predict"
        self.classes = list(classes)
        self.calls.append(("classes", list(classes)))

    def predict(self, **kwargs):
        with self.lock:
            assert not self.active
            self.active = True
        try:
            time.sleep(0.005)
            self.calls.append(("predict", kwargs))
            return [SimpleNamespace(
                names=dict(enumerate(self.classes)),
                boxes=SimpleNamespace(
                    xyxy=TensorValues([[4, 5, 40, 50]]),
                    conf=TensorValues([0.8]), cls=TensorValues([0.0]),
                ),
            )]
        finally:
            with self.lock:
                self.active = False


@pytest.fixture
def adapter(tmp_path):
    checkpoint = tmp_path / "yoloe-26x-seg.pt"
    checkpoint.write_bytes(b"test checkpoint")
    encoder = tmp_path / "mobileclip2_b.ts"
    encoder.write_bytes(b"test encoder")
    return YOLOEDetector(checkpoint, encoder=encoder, model_factory=FakeYOLOE, device="cpu")


def test_sequential_prompts_refresh_dynamic_class_zero(adapter):
    image = Image.new("RGB", (64, 64))
    first = adapter.detect(image, ["forklift", "person"], 0.25, 640)
    second = adapter.detect(image, ["red helmet"], 0.25, 960)
    assert first[0].label == "forklift"
    assert second[0].label == "red helmet"
    assert first[0].prompt_class_index == second[0].prompt_class_index == 0
    assert adapter.model.evaluating
    assert [c[1] for c in adapter.model.calls if c[0] == "classes"] == [
        ["forklift", "person"], ["red helmet"],
    ]


def test_threshold_and_resolution_reach_one_inference(adapter):
    image = Image.new("RGB", (64, 64))
    assert adapter.detect(image, ["person"], 0.9, 1280) == []
    calls = [c[1] for c in adapter.model.calls if c[0] == "predict"]
    assert len(calls) == 1
    assert calls[0]["source"] is image
    assert calls[0]["conf"] == 0.9
    assert calls[0]["imgsz"] == 1280
    assert calls[0]["device"] == "cpu"
    assert calls[0]["save"] is False
    assert calls[0]["nms"] is None
    assert calls[0]["iou"] == 0.45
    assert calls[0]["agnostic_nms"] is False


def test_warmup_builds_two_distinct_prompt_embeddings_without_inference(adapter):
    adapter.warmup()
    assert adapter.model.calls == [
        ("classes", ["person"]), ("classes", ["forklift", "red helmet"]),
    ]


def test_concurrent_requests_cannot_mutate_each_others_classes(adapter):
    image = Image.new("RGB", (64, 64))
    classes = [["forklift"], ["red helmet"]] * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda names: adapter.detect(image, names, 0.25, 640), classes))
    assert [objects[0].label for objects in results] == [names[0] for names in classes]


@pytest.mark.parametrize("confidence,size", [(0,640), (float("nan"),640), (1,640), (0.25,641)])
def test_invalid_quality_controls_fail_before_model_mutation(adapter, confidence, size):
    with pytest.raises(ValueError):
        adapter.detect(Image.new("RGB", (64,64)), ["person"], confidence, size)
    assert adapter.model.calls == []


def test_missing_assets_fail_before_model_initialization(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_assets"):
        YOLOEDetector(tmp_path / "yoloe-26x-seg.pt", model_factory=FakeYOLOE)


def test_prompt_free_checkpoint_rejected(tmp_path):
    checkpoint = tmp_path / "yoloe-26x-seg-pf.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="yoloe-26x-seg.pt"):
        YOLOEDetector(checkpoint, model_factory=FakeYOLOE)
