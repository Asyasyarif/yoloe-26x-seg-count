"""Verify a published Replicate version and both single-file output modes.

Uses the documented HTTP API; token comes only from REPLICATE_API_TOKEN.
Never print authorization headers or save the token in build artifacts.
"""

import argparse
import base64
from collections import Counter
import io
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import zipfile

API = "https://api.replicate.com/v1"


def api_request(path: str, body: dict | None = None) -> dict:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise ValueError("REPLICATE_API_TOKEN is required")
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(API + path, data=data, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Replicate API {path} returned HTTP {error.code}") from None


def validate_output_url(output: object) -> str:
    """Return the artifact URL, accepting a one-element list from coglet 0.23.x.

    coglet 0.23.x sends a non-terminal `output` webhook whose `output` is the
    accumulated outputs array even for a scalar `Path` (replicate/cog#3205).
    A client that applies webhooks in arrival order can therefore end a
    successful prediction holding `["https://..."]` instead of the URL. The
    model's Output schema is still one URI, so this is a runtime quirk, not a
    model contract change: both shapes describe the same single artifact.
    """
    if isinstance(output, list) and len(output) == 1:
        output = output[0]
    if not isinstance(output, str) or not output.startswith("https://"):
        raise ValueError("Expected exactly one HTTPS artifact URL, never zero, many, or a dict")
    return output


def validate_artifact(payload: bytes, zipped: bool, expected_classes: list[str] | None = None) -> dict | None:
    png_magic = b"\x89PNG\r\n\x1a\n"
    if not zipped:
        if not payload.startswith(png_magic):
            raise ValueError("Expected one PNG output")
        return None
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sorted(archive.namelist()) != ["annotated.png", "result.json"]:
            raise ValueError("ZIP must contain exactly annotated.png and result.json")
        if not archive.read("annotated.png").startswith(png_magic):
            raise ValueError("ZIP annotation must be PNG")
        result = json.loads(archive.read("result.json"))
    objects, summary = result["objects"], result["summary"]
    classes = result["input"]["resolved_classes"]
    if expected_classes is not None and classes != expected_classes:
        raise ValueError("Output classes differ from requested classes")
    if result["schema_version"] != "1.1":
        raise ValueError("Unexpected result schema")
    if summary["total_objects"] != len(objects) or len(objects) != sum(summary["counts_by_class"].values()):
        raise ValueError("Counts must match canonical objects")
    counts = Counter(obj["label"] for obj in objects)
    if summary["counts_by_class"] != {name: counts[name] for name in classes}:
        raise ValueError("Per-class counts must match objects")
    for i, obj in enumerate(objects, 1):
        if obj["id"] != f"obj_{i:03d}" or obj["label"] != classes[obj["prompt_class_index"]]:
            raise ValueError("Class/ID mapping mismatch")
    return result


def verify_deployment_identity(result: dict, image: str, version_id: str) -> None:
    deployment = result.get("deployment", {})
    if deployment.get("docker_image_uri") != image:
        raise ValueError("Running image differs from immutable cog push result")
    if deployment.get("version_id") != version_id:
        raise ValueError("Running version differs from requested Replicate version")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, default=Path("examples/sample.jpg"))
    parser.add_argument("--report", type=Path, default=Path("reports/replicate.json"))
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--push-result", type=Path)
    parser.add_argument("--previous-model", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[a-z0-9_-]+", args.model):
        raise ValueError("Model name must be owner/model")
    model = api_request(f"/models/{args.model}")
    if args.check_model:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"model": args.model,
                                          "previous_version_id": (model.get("latest_version") or {}).get("id")}) + "\n")
        print(f"Verified target model: {args.model} ({model['visibility']})")
        return
    if not args.push_result or not args.previous_model:
        raise ValueError("Pass --push-result and --previous-model to identify this publication")
    pushed = json.loads(args.push_result.read_text())
    pushed_image = pushed.get("image", "")
    if pushed.get("version") != 1 or not re.fullmatch(re.escape("r8.im/" + args.model) + r"@sha256:[0-9a-f]{64}", pushed_image):
        raise ValueError("Invalid immutable image reference in cog push result")
    previous_id = json.loads(args.previous_model.read_text())["previous_version_id"]
    deadline = time.monotonic() + 180
    while not model.get("latest_version") or model["latest_version"]["id"] == previous_id:
        if time.monotonic() >= deadline:
            raise TimeoutError("Replicate has not exposed the new version within three minutes")
        time.sleep(3)
        model = api_request(f"/models/{args.model}")
    version = model.get("latest_version")
    if not version:
        raise RuntimeError("Target model has no published version")
    schema = version["openapi_schema"]["components"]["schemas"]["Input"]
    expected = {"image", "prompt", "output_zip", "confidence", "image_size"}
    if set(schema["properties"]) != expected or set(schema.get("required", [])) != {"image", "prompt"}:
        raise ValueError("Published input schema does not match the five-input contract")
    image_url = "data:image/jpeg;base64," + base64.b64encode(args.image.read_bytes()).decode()
    measurements = []
    # ZIP comes first so runtime identity is checked before the PNG-only run.
    for prompt, zipped, classes in [("forklift, red helmet", True, ["forklift", "red helmet"]),
                                    ("hitung orang dan mobil", False, ["person", "car"])]:
        inputs = {"image": image_url, "prompt": prompt, "confidence": 0.25, "image_size": 1280}
        if zipped:
            inputs["output_zip"] = True  # First run deliberately verifies omitted/default behavior.
        started = time.monotonic()
        prediction = api_request("/predictions", {"version": version["id"], "input": inputs})
        while prediction["status"] not in {"succeeded", "failed", "canceled"}:
            if time.monotonic() - started > 900:
                api_request(f"/predictions/{prediction['id']}/cancel", {})
                raise TimeoutError("Remote smoke test exceeded 15 minutes and was canceled")
            time.sleep(3)
            prediction = api_request(f"/predictions/{prediction['id']}")
        if prediction["status"] != "succeeded":
            raise RuntimeError(f"Prediction {prediction['id']} {prediction['status']}: {prediction.get('error')}")
        output = prediction.get("output")
        output_url = validate_output_url(output)
        if not isinstance(output, str):
            # Record the quirk so a green workflow never hides a runtime regression.
            measurements_output_shape = "array"
        else:
            measurements_output_shape = "string"
        with urllib.request.urlopen(output_url, timeout=120) as response:
            payload = response.read(50 * 1024 * 1024 + 1)
        if len(payload) > 50 * 1024 * 1024:
            raise ValueError("Smoke output exceeded the 50 MB verification limit")
        result = validate_artifact(payload, zipped, expected_classes=classes)
        if result:
            verify_deployment_identity(result, pushed_image, version["id"])
            if result["input"]["prompt"] != prompt or result["input"]["image_size"] != 1280:
                raise ValueError("Output request metadata differs from submitted prompt/resolution")
        measurements.append({"prediction_id": prediction["id"], "prompt": prompt, "output_zip": zipped,
                             "output_shape": measurements_output_shape,
                             "wall_seconds": time.monotonic() - started, "metrics": prediction.get("metrics"),
                             "summary": result["summary"] if result else None})
    report = {"model": args.model, "version": version["id"], "image": pushed_image, "measurements": measurements,
              "accuracy_evaluated": False, "cost": None, "peak_vram": None}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Published version {version['id']} passed PNG and ZIP smoke tests")


if __name__ == "__main__":
    main()
