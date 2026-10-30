import hashlib
from pathlib import Path

import pytest

from scripts.prepare_assets import provision_asset


def test_cached_asset_checked_without_downloading(tmp_path):
    payload = b"verified checkpoint"
    target = tmp_path / "weights" / "model.pt"
    target.parent.mkdir()
    target.write_bytes(payload)
    asset = {"path": "weights/model.pt", "url": "https://example.test/model", "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    assert provision_asset(asset, tmp_path, check_only=True) == target


def test_corrupt_cached_asset_cannot_pass_verification(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"bad")
    asset = {"path": "model.pt", "url": "https://example.test/model", "sha256": "0" * 64, "size": 3}
    with pytest.raises(ValueError, match="checksum"):
        provision_asset(asset, tmp_path, check_only=True)


def test_missing_asset_check_does_not_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        provision_asset({"path": "missing", "sha256": "0" * 64, "size": 1}, tmp_path, check_only=True)


def test_manifest_has_exact_required_assets_with_sha256():
    import json
    manifest = json.loads(Path("weights/manifest.json").read_text())
    assert {a["path"] for a in manifest} == {"weights/yoloe-26x-seg.pt", "mobileclip2_b.ts"}
    assert all(len(a["sha256"]) == 64 and a["size"] > 0 for a in manifest)
