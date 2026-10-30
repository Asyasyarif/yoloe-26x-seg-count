"""Runner contract tests use the actual Cog SDK and a lightweight detector boundary."""
import inspect
import json
from pathlib import Path as FilePath
import zipfile

from cog import BaseRunner, Path
from PIL import Image
import pytest

from run import Runner
import run as runner_module
from src.schema import DetectedObject


class Detector:
    def __init__(self):
        self.calls = []

    def detect(self, image, classes, confidence, image_size):
        self.calls.append((image.mode, classes, confidence, image_size))
        return [DetectedObject("obj_001", classes[0], 0, 0.8, {"x1":4,"y1":5,"x2":40,"y2":50})]


def invoke(runner, image, prompt, **kwargs):
    defaults = {name: parameter.default.default for name, parameter in inspect.signature(Runner.run).parameters.items()
                if name in {"output_zip", "confidence", "image_size"}}
    return runner.run(image=Path(image), prompt=prompt, **(defaults | kwargs))


@pytest.fixture
def runner_and_image(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGBA", (80, 64), (255, 0, 0, 80)).save(image)
    runner = Runner()
    runner.detector = Detector()
    return runner, image


def test_runner_public_signature_is_one_file_with_exact_inputs():
    assert issubclass(Runner, BaseRunner)
    signature = inspect.signature(Runner.run)
    assert list(signature.parameters) == ["self", "image", "prompt", "output_zip", "confidence", "image_size"]
    assert signature.return_annotation is Path
    assert signature.parameters["output_zip"].default.default is False
    assert signature.parameters["image_size"].default.default == 1280


def test_setup_honors_explicit_cpu_smoke_environment_and_warms(monkeypatch):
    calls = []

    class SetupDetector:
        def __init__(self, *, device):
            calls.append(device)

        def warmup(self):
            calls.append("warmup")

    monkeypatch.setenv("YOLO_DEVICE", "cpu")
    monkeypatch.setattr(runner_module, "YOLOEDetector", SetupDetector)
    Runner().setup()
    assert calls == ["cpu", "warmup"]


def test_default_returns_existing_png_and_no_json(runner_and_image):
    runner, image = runner_and_image
    output = invoke(runner, image, "hitung orang dan mobil")
    assert isinstance(output, Path)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert sorted(p.name for p in output.parent.iterdir()) == ["annotated.png"]
    assert runner.detector.calls == [("RGB", ["person", "car"], 0.25, 1280)]


def test_zip_returns_one_file_and_matching_json(runner_and_image):
    runner, image = runner_and_image
    output = invoke(runner, image, "red helmet, forklift", output_zip=True, confidence=0.5, image_size=960)
    assert output.is_file()
    assert output.read_bytes().startswith(b"PK")
    with zipfile.ZipFile(output) as archive:
        assert sorted(archive.namelist()) == ["annotated.png", "result.json"]
        report = json.loads(archive.read("result.json"))
        assert report["summary"] == {"total_objects":1,"counts_by_class":{"red helmet":1,"forklift":0}}
        assert report["input"]["image_size"] == 960
    assert runner.detector.calls[0][2:] == (0.5, 960)


def test_unsupported_prompt_fails_before_inference(runner_and_image):
    runner, image = runner_and_image
    with pytest.raises(ValueError):
        invoke(runner, image, "people standing next to cars")
    assert runner.detector.calls == []
