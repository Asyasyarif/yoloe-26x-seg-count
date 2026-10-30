"""Replicate/Cog public interface: one image in, exactly one PNG or ZIP out."""
import os

from cog import BaseRunner, Input, Path

from src.detector import YOLOEDetector, validate_quality
from src.image_io import load_image
from src.packaging import create_output
from src.prompts import resolve_target_phrases


class Runner(BaseRunner):
    def setup(self) -> None:
        self.detector = YOLOEDetector(device=os.environ.get("YOLO_DEVICE", "cuda:0"))
        self.detector.warmup()

    def run(
        self,
        image: Path = Input(description="One JPEG, PNG, or WebP image (up to 20 MB and 20 megapixels)"),
        prompt: str = Input(description="Object categories, e.g. 'red helmet, forklift'", min_length=1, max_length=256),
        output_zip: bool = Input(description="Return one ZIP containing annotated.png and result.json", default=False),
        confidence: float = Input(description="Minimum detection confidence", default=0.25, ge=0.05, le=0.95),
        image_size: int = Input(description="Inference resolution; larger is not always more accurate", default=1280, choices=[640, 960, 1280]),
    ) -> Path:
        validate_quality(confidence, image_size)
        if not isinstance(output_zip, bool):
            raise ValueError("output_zip must be true or false.")
        classes = resolve_target_phrases(prompt)
        source = load_image(image)
        objects = self.detector.detect(source, classes, confidence, image_size)
        return Path(create_output(source, objects, classes, prompt, confidence, image_size, output_zip))
