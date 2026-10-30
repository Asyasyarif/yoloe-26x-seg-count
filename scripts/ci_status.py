"""Write a small, credential-free CI report for monitoring private repos via Git."""

import json
import os
from pathlib import Path


def make_report(directory: Path, environment: dict) -> dict:
    report = {key.lower(): environment.get(key) for key in (
        "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "JOB_STATUS",
        "IMAGE_BUILD_OUTCOME", "OFFLINE_OUTCOME", "PUSH_OUTCOME", "CLOUD_SMOKE_OUTCOME", "MODEL_NAME",
    )}
    report["run_url"] = f"https://github.com/{environment.get('GITHUB_REPOSITORY')}/actions/runs/{environment.get('GITHUB_RUN_ID')}"
    secret = environment.get("CLI_TOKEN", "")
    for filename in ("push.json", "offline.json", "pushed-offline.json", "replicate.json"):
        path = directory / filename
        if path.is_file() and path.stat().st_size:
            try:
                report[filename] = json.loads(path.read_text())
            except ValueError:
                pass
    for filename in ("build.log", "push.log"):
        path = directory / filename
        if path.is_file():
            text = path.read_text(errors="replace")[-24000:]
            report[filename] = text.replace(secret, "[REDACTED]") if secret else text
    # A second pass also covers any credential accidentally present in JSON.
    serialized = json.dumps(report)
    if secret:
        serialized = serialized.replace(secret, "[REDACTED]")
    return json.loads(serialized)


if __name__ == "__main__":
    directory = Path("reports")
    directory.mkdir(exist_ok=True)
    (directory / "ci-status.json").write_text(json.dumps(make_report(directory, dict(os.environ)), indent=2) + "\n")
