"""Evaluate a legally usable, manually reviewed corpus on the actual model.

Each JSONL row: image (relative to corpus), prompt, counts (resolved class ->
integer), reviewed=true, license (source/license reference). At least 50 unique
images are required. No fabricated accuracy report is produced without them.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    errors = [abs(row["predicted_count"] - row["ground_truth_count"]) for row in rows]
    result = {
        "samples": len(rows),
        "count_mae": statistics.mean(errors),
        "exact_count_match_rate": sum(error == 0 for error in errors) / len(errors),
        "mean_seconds": statistics.mean(row["seconds"] for row in rows),
    }
    class_errors = [error for row in rows for error in row.get("absolute_error_by_class", {}).values()]
    if class_errors:
        result["class_count_mae"] = statistics.mean(class_errors)
        result["all_classes_exact_match_rate"] = sum(
            all(error == 0 for error in row["absolute_error_by_class"].values()) for row in rows
        ) / len(rows)
    return result


def validate_corpus(rows: list[dict]) -> None:
    if len(rows) < 50:
        raise ValueError("Accuracy evaluation requires at least 50 manually reviewed images")
    if len({str(Path(row["image"]).resolve()) for row in rows}) < 50:
        raise ValueError("Accuracy evaluation requires at least 50 unique images")
    for row in rows:
        if row.get("reviewed") is not True or not row.get("license"):
            raise ValueError("Every image needs reviewed=true and a license/source reference")
        if not row.get("counts") or any(type(n) is not int or n < 0 for n in row["counts"].values()):
            raise ValueError("Counts must be nonnegative integers for all resolved classes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/accuracy.json"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    validate_corpus(rows)
    import torch
    from src.detector import YOLOEDetector
    from src.image_io import load_image
    from src.prompts import resolve_target_phrases

    started = time.perf_counter()
    detector = YOLOEDetector(device=args.device)
    setup_seconds = time.perf_counter() - started
    results = []
    for size in (640, 960, 1280):
        for confidence in (0.15, 0.25, 0.35, 0.5):
            measurements = []
            for row in rows:
                classes = resolve_target_phrases(row["prompt"])
                if set(classes) != set(row["counts"]):
                    raise ValueError(f"Ground-truth classes differ from parsed prompt: {row['image']}")
                image = load_image(args.corpus.parent / row["image"])
                if args.device != "cpu":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                started = time.perf_counter()
                objects = detector.detect(image, classes, confidence, size)
                if args.device != "cpu":
                    torch.cuda.synchronize()
                predicted = Counter(obj.label for obj in objects)
                measurement = {
                    "image": row["image"], "prompt": row["prompt"],
                    "predicted_count": len(objects), "ground_truth_count": sum(row["counts"].values()),
                    "predicted_counts": {c: predicted[c] for c in classes}, "ground_truth_counts": row["counts"],
                    "absolute_error_by_class": {c: abs(predicted[c] - row["counts"][c]) for c in classes},
                    "seconds": time.perf_counter() - started,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated() if args.device != "cpu" else None,
                }
                measurements.append(measurement)
            results.append({"image_size": size, "confidence": confidence, **summarize(measurements), "measurements": measurements})
    report = {"setup_seconds": setup_seconds, "device": args.device,
              "gpu": torch.cuda.get_device_name() if args.device != "cpu" else None,
              "object_precision_recall": None, "object_precision_recall_note": "Requires reviewed bounding-box ground truth; count annotations cannot measure FP/FN or AP.",
              "cost": None, "cost_note": "Record actual billed Replicate runs separately.", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
