import io
import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from src.packaging import create_output


@dataclass
class Detection:
    id: str
    label: str
    prompt_class_index: int
    confidence: float
    bbox: dict[str, int]


def object_one():
    return Detection("obj_001", "forklift", 1, .956377, {"x1": 30, "y1": 40, "x2": 60, "y2": 90})


@pytest.mark.parametrize("explicit", [False, True])
def test_png_only_is_one_existing_path_without_json(explicit):
    args = {"output_zip": False} if explicit else {}
    path = create_output(Image.new("RGB", (128, 128)), [object_one()], ["red helmet", "forklift"], "red helmet, forklift", .25, 1280, **args)
    assert isinstance(path, Path)
    assert path.name == "annotated.png"
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert sorted(p.name for p in path.parent.iterdir()) == ["annotated.png"]
    with Image.open(path) as image:
        assert image.size == (128, 128)


def test_zip_exact_members_json_invariants_and_png():
    path = create_output(Image.new("RGB", (128, 128)), [object_one()], ["red helmet", "forklift"], "red helmet, forklift", .25, 1280, True)
    assert path.name == "results.zip"
    assert path.read_bytes().startswith(b"PK\x03\x04")
    with ZipFile(path) as archive:
        assert sorted(archive.namelist()) == ["annotated.png", "result.json"]
        assert all(info.compress_type == ZIP_DEFLATED for info in archive.infolist())
        with Image.open(io.BytesIO(archive.read("annotated.png"))) as image:
            assert image.format == "PNG"
        data = json.loads(archive.read("result.json"))
    assert data["schema_version"] == "1.1"
    assert data["model"]["checkpoint"] == "yoloe-26x-seg.pt"
    assert data["input"]["image_width"] == data["input"]["image_height"] == 128
    assert data["input"]["prompt"] == "red helmet, forklift"
    assert data["input"]["resolved_classes"] == ["red helmet", "forklift"]
    assert data["summary"]["counts_by_class"] == {"red helmet": 0, "forklift": 1}
    assert data["summary"]["total_objects"] == len(data["objects"]) == sum(data["summary"]["counts_by_class"].values())
    assert data["objects"][0]["confidence"] == .9564
    assert data["objects"][0]["id"] == "obj_001"
    assert data["objects"][0]["prompt_class_index"] == 1


def test_zero_objects_and_requests_have_separate_directories():
    image = Image.new("RGB", (64, 64))
    first = create_output(image, [], ["person"], "person", .25, 640, True)
    second = create_output(image, [], ["person"], "person", .25, 640)
    assert first.parent != second.parent
    with ZipFile(first) as archive:
        data = json.loads(archive.read("result.json"))
    assert data["objects"] == []
    assert data["summary"] == {"total_objects": 0, "counts_by_class": {"person": 0}}


def test_zip_records_runtime_image_identity(monkeypatch):
    monkeypatch.setenv("REPLICATE_DOCKER_IMAGE_URI", "r8.im/test/counter@sha256:" + "a" * 64)
    monkeypatch.setenv("REPLICATE_VERSION_ID", "b" * 64)
    path = create_output(Image.new("RGB", (64, 64)), [], ["person"], "person", .25, 640, True)
    with ZipFile(path) as archive:
        data = json.loads(archive.read("result.json"))
    assert data["deployment"]["version_id"] == "b" * 64
    assert data["deployment"]["docker_image_uri"].endswith("a" * 64)
