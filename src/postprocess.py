"""Filter model boxes once so IDs, annotations, and counts cannot diverge.

Cross-class boxes with IoU >= 0.90 are treated as near-identical duplicates;
the highest-confidence candidate survives (lexical class/geometry breaks ties).
Same-class NMS belongs to YOLOE. Lower overlaps and nested boxes survive because
co-located categories can be distinct physical objects. This geometric heuristic
can still merge almost perfectly aligned different objects; benchmark this policy
for overlapping category prompts before claiming counting accuracy.
"""

from collections.abc import Mapping, Sequence
import math

from src.schema import DetectedObject

CROSS_CLASS_DUPLICATE_IOU = 0.90


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def normalize_detections(
    boxes: Sequence[Sequence[float]],
    confidences: Sequence[float],
    class_indices: Sequence[float],
    names: Mapping[int, str] | Sequence[str],
    classes: Sequence[str],
    width: int,
    height: int,
    confidence: float,
) -> list[DetectedObject]:
    """Validate request-relative labels, clip/filter, deduplicate, sort, assign IDs.

    Invalid geometry/scores are discarded. Invalid model class mappings raise an
    error rather than silently labeling a prediction with a stale request class.
    Confidence is filtered before any serialization rounding. Pixel boxes use
    floor(minimum)/ceil(maximum) after clipping to retain subpixel detections.
    """
    if (not isinstance(width, int) or isinstance(width, bool) or width <= 0
            or not isinstance(height, int) or isinstance(height, bool) or height <= 0
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("Image dimensions and confidence threshold are invalid.")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Requested classes must be nonempty and unique.")
    if not len(boxes) == len(confidences) == len(class_indices):
        raise ValueError("Model box, confidence, and class arrays must have equal lengths.")
    candidates: list[tuple[tuple[float, ...], float, int, str]] = []
    for box, score, raw_index in zip(boxes, confidences, class_indices):
        try:
            index_float = float(raw_index)
            if not math.isfinite(index_float) or not index_float.is_integer():
                raise ValueError
            index = int(index_float)
            if not 0 <= index < len(classes) or names[index] != classes[index]:
                raise ValueError
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Model class mapping does not match the current requested categories.") from exc
        try:
            score = float(score)
            coords = tuple(float(value) for value in box)
        except (TypeError, ValueError, OverflowError):
            continue
        if (len(coords) != 4 or not all(math.isfinite(value) for value in coords)
                or not math.isfinite(score) or not confidence <= score <= 1):
            continue
        x1, y1, x2, y2 = coords
        clipped = (max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1)),
                   max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2)))
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        candidates.append((clipped, score, index, classes[index]))

    candidates.sort(key=lambda item: (-item[1], item[3], item[0], item[2]))
    surviving: list[tuple[tuple[float, ...], float, int, str]] = []
    for candidate in candidates:
        if any(candidate[2] != other[2] and _iou(candidate[0], other[0]) >= CROSS_CLASS_DUPLICATE_IOU
               for other in surviving):
            continue
        surviving.append(candidate)

    records: list[tuple[dict[str, int], float, int, str]] = []
    for box, score, index, label in surviving:
        pixel_box = dict(zip(("x1", "y1", "x2", "y2"),
                             (math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3]))))
        records.append((pixel_box, score, index, label))
    records.sort(key=lambda item: (item[0]["y1"], item[0]["x1"], item[3], -item[1],
                                   item[0]["y2"], item[0]["x2"], item[2]))
    return [DetectedObject(f"obj_{number:03d}", label, index, score, box)
            for number, (box, score, index, label) in enumerate(records, start=1)]
