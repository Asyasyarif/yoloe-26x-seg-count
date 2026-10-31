"""Run real sequential dynamic prompts without network; record measured resources.

Run inside the built image with Docker ``--network none`` for an independent
network-isolation check. No synthetic result is used by this smoke test.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import importlib.metadata
import os
from pathlib import Path
import shutil
import socket
import sys
import time
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["YOLO_AUTOINSTALL"] = "false"


@contextmanager
def no_network():
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(f"Offline smoke test blocked network connection to {address!r}")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(f"Offline smoke test blocked network connection to {address!r}")
        return original_connect_ex(sock, address)

    with patch.object(socket.socket, "connect", connect), patch.object(socket.socket, "connect_ex", connect_ex):
        yield


def verify_artifact(path: Path, classes: list[str], zipped: bool) -> int | None:
    if not path.is_file():
        raise AssertionError("Runner returned a missing file")
    if not zipped:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert sorted(item.name for item in path.parent.iterdir()) == ["annotated.png"]
        return None
    assert path.read_bytes().startswith(b"PK\x03\x04")
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == ["annotated.png", "result.json"]
        assert archive.read("annotated.png").startswith(b"\x89PNG\r\n\x1a\n")
        report = json.loads(archive.read("result.json"))
    assert report["input"]["resolved_classes"] == classes
    objects, summary = report["objects"], report["summary"]
    assert summary["total_objects"] == len(objects) == sum(summary["counts_by_class"].values())
    assert set(summary["counts_by_class"]) == set(classes)
    assert [obj["id"] for obj in objects] == [f"obj_{index:03d}" for index in range(1, len(objects) + 1)]
    for obj in objects:
        assert obj["label"] == classes[obj["prompt_class_index"]]
    return len(objects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("examples/sample.jpg"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sizes", "--image-size", dest="sizes", type=int, nargs="+", default=[640, 960, 1280])
    parser.add_argument("--report", "--output", dest="report", type=Path, default=Path("offline-report.json"))
    args = parser.parse_args()
    import torch
    from cog import Path as CogPath
    from run import Runner
    from src.prompts import resolve_target_phrases

    with no_network():
        runner = Runner()
        start = time.perf_counter()
        with patch.dict(os.environ, {"YOLO_DEVICE": args.device}):
            runner.setup()
        setup_seconds = time.perf_counter() - start
        measurements = []
        for size in args.sizes:
            for prompt, zipped in [("person, bus", False), ("forklift, red helmet", True)]:
                if args.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                start = time.perf_counter()
                output = runner.run(image=CogPath(args.image), prompt=prompt, output_zip=zipped, confidence=0.25, image_size=size)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                total = verify_artifact(Path(output), resolve_target_phrases(prompt), zipped)
                measurements.append({
                    "prompt":prompt,"image_size":size,"output_zip":zipped,"seconds":seconds,
                    "total_objects":total,
                    "peak_allocated_vram_bytes":torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
                    "peak_reserved_vram_bytes":torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None,
                })
                shutil.rmtree(Path(output).parent)
    report = {
        "device":args.device,
        "gpu":torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "setup_seconds":setup_seconds,"measurements":measurements,
        "accuracy_evaluated":False,"cost_evaluated":False,
        # coglet is pinned separately from the cog SDK; record both so a report
        # identifies the exact inference server that produced the outputs.
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("ultralytics", "torch", "torchvision", "Pillow", "cog", "coglet")},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
