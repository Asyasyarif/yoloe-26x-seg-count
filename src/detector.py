"""Serialized dynamic-prompt inference with the original YOLOE-26x checkpoint."""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from threading import Lock
from typing import Callable

from PIL import Image

from src.postprocess import normalize_detections
from src.schema import DetectedObject


def validate_quality(confidence: float, image_size: int) -> None:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number between 0.05 and 0.95.")
    if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.95:
        raise ValueError("confidence must be between 0.05 and 0.95.")
    if isinstance(image_size, bool) or image_size not in (640, 960, 1280):
        raise ValueError("image_size must be 640, 960, or 1280.")


class YOLOEDetector:
    """One mutable model per worker; class mutation and inference share a lock.

    ``model_factory`` is an injection boundary for CPU contract tests. Production
    uses the real Ultralytics YOLOE class and requires already provisioned assets.
    MobileCLIP2 is resolved by Ultralytics relative to the working directory.
    """

    def __init__(
        self,
        checkpoint: str | Path = "weights/yoloe-26x-seg.pt",
        *,
        encoder: str | Path = "mobileclip2_b.ts",
        device: str = "cuda:0",
        model_factory: Callable | None = None,
    ) -> None:
        checkpoint, encoder = Path(checkpoint), Path(encoder)
        if checkpoint.name != "yoloe-26x-seg.pt":
            raise ValueError("Text prompting requires the original yoloe-26x-seg.pt checkpoint.")
        for asset in (checkpoint, encoder):
            if not asset.is_file() or asset.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Missing model asset: {asset}. Run python scripts/prepare_assets.py before building."
                )
        if model_factory is None:
            # Prevent Ultralytics from silently installing dependencies at runtime.
            os.environ["YOLO_AUTOINSTALL"] = "false"
            if importlib.util.find_spec("clip") is None:
                raise RuntimeError("The pinned ultralytics/CLIP tokenizer must be installed at build time.")
            import torch
            from ultralytics import YOLOE

            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("A CUDA GPU is required; use device='cpu' only for offline smoke tests.")
            model_factory = YOLOE
        self.device = device
        self._lock = Lock()
        self.model = model_factory(str(checkpoint))
        self.model.to(device)
        self.model.eval()

    def warmup(self) -> None:
        """Load the tokenizer/text encoder in setup using two distinct class sets."""
        with self._lock:
            self.model.set_classes(["person"])
            self.model.set_classes(["forklift", "red helmet"])

    def detect(
        self, image: Image.Image, classes: list[str], confidence: float, image_size: int
    ) -> list[DetectedObject]:
        validate_quality(confidence, image_size)
        if not classes or len(classes) > 8 or any(not isinstance(name, str) or not name.strip() for name in classes):
            raise ValueError("Provide between one and eight nonempty category phrases.")
        with self._lock:
            self.model.set_classes(list(classes))
            results = self.model.predict(
                source=image,
                imgsz=image_size,
                conf=confidence,
                device=self.device,
                verbose=False,
                save=False,
                # External, class-aware NMS. IoU .45 is a provisional baseline
                # after a real-image duplicate sweep, not a crowd accuracy claim.
                nms=None,
                iou=0.45,
                agnostic_nms=False,
            )
            if len(results) != 1:
                raise RuntimeError("Expected exactly one YOLOE result for the input image.")
            result = results[0]
            if result.boxes is None:
                return []
            return normalize_detections(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.names,
                classes,
                image.width,
                image.height,
                confidence,
            )
