"""One-file PNG/ZIP outputs, derived from the same detection records."""

from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

from src.annotate import annotate_image
from src.prompts import PARSER_VERSION

if TYPE_CHECKING:
    from src.schema import DetectedObject


def create_output(
    image: Image.Image,
    objects: Sequence["DetectedObject"],
    classes: Sequence[str],
    prompt: str,
    confidence: float,
    image_size: int,
    output_zip: bool = False,
) -> Path:
    """Return one persistent request-specific artifact for Cog to serve.

    ZIP mode has exactly two members; PNG mode never emits a JSON artifact.
    The serving runtime owns cleanup of successful returned paths.
    """
    directory = Path(tempfile.mkdtemp(prefix="image-counter-"))
    try:
        annotated = annotate_image(image, objects)
        if not output_zip:
            path = directory / "annotated.png"
            annotated.save(path, format="PNG")
            return path

        counts = dict.fromkeys(classes, 0)
        records = []
        for obj in objects:
            counts[obj.label] += 1
            record = asdict(obj)
            record["confidence"] = round(obj.confidence, 4)
            records.append(record)
        report = {
            "schema_version": "1.1",
            "parser_version": PARSER_VERSION,
            "deployment": {
                "docker_image_uri": os.environ.get("REPLICATE_DOCKER_IMAGE_URI"),
                "version_id": os.environ.get("REPLICATE_VERSION_ID"),
            },
            "model": {
                "name": "yoloe-26x",
                "checkpoint": "yoloe-26x-seg.pt",
                "task": "open-vocabulary-object-detection",
            },
            "input": {
                "prompt": prompt,
                "resolved_classes": list(classes),
                "confidence_threshold": confidence,
                "image_size": image_size,
                "image_width": image.width,
                "image_height": image.height,
            },
            "summary": {"total_objects": len(records), "counts_by_class": counts},
            "objects": records,
        }
        png = io.BytesIO()
        annotated.save(png, format="PNG")
        path = directory / "results.zip"
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("annotated.png", png.getvalue())
            archive.writestr("result.json", json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
        return path
    except Exception:
        shutil.rmtree(directory)
        raise
