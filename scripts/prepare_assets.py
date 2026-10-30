"""Provision official, digest-pinned assets BEFORE running cog build.

This script deliberately uses only Python's standard library. Cog build.run
cannot read repository scripts, so provisioning happens on the CI host.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def verify(path: Path, asset: dict) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != asset["sha256"] or path.stat().st_size != asset["size"]:
        raise ValueError(f"Asset checksum/size mismatch: {asset['path']}")


def provision_asset(asset: dict, root: Path, check_only: bool = False) -> Path:
    target = root / asset["path"]
    if target.is_file():
        verify(target, asset)
        return target
    if check_only:
        raise FileNotFoundError(f"Missing {target}; run python scripts/prepare_assets.py")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "object-counter-build"})
            with urllib.request.urlopen(request, timeout=120) as response:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > asset["size"]:
                        raise ValueError(f"Asset exceeds pinned size: {asset['path']}")
                    output.write(chunk)
        verify(temporary, asset)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing assets without any network access")
    args = parser.parse_args()
    for asset in json.loads((ROOT / "weights/manifest.json").read_text()):
        path = provision_asset(asset, ROOT, args.check)
        print(f"Verified {path.relative_to(ROOT)} sha256:{asset['sha256']}")


if __name__ == "__main__":
    main()
