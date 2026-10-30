import io
import json
import zipfile

import pytest

from scripts.replicate_smoke import validate_artifact, validate_output_url, verify_deployment_identity


def test_single_url_is_unchanged():
    assert validate_output_url("https://x/a.png") == "https://x/a.png"


def test_one_element_array_is_the_same_single_artifact():
    # coglet 0.23.x progress webhook shape (replicate/cog#3205).
    assert validate_output_url(["https://x/a.png"]) == "https://x/a.png"


@pytest.mark.parametrize("output", [None, [], ["https://a", "https://b"], {"url": "https://a"}, "http://x", ""])
def test_every_other_shape_is_rejected(output):
    with pytest.raises(ValueError, match="HTTPS"):
        validate_output_url(output)


def test_png_response_is_one_png_and_not_a_list():
    validate_artifact(b"\x89PNG\r\n\x1a\n", False)
    with pytest.raises(ValueError, match="PNG"):
        validate_artifact(b"not png", False)


def test_remote_zip_schema_counts_and_members_checked():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("annotated.png", b"\x89PNG\r\n\x1a\n")
        archive.writestr("result.json", json.dumps({
            "schema_version": "1.1", "objects": [],
            "input": {"resolved_classes": ["forklift"]},
            "summary": {"total_objects": 0, "counts_by_class": {"forklift": 0}},
        }))
    result = validate_artifact(output.getvalue(), True)
    assert result is not None
    assert result["summary"]["total_objects"] == 0
    with pytest.raises(ValueError, match="classes"):
        validate_artifact(output.getvalue(), True, expected_classes=["person"])


def test_published_version_must_match_exact_pushed_image():
    result = {"deployment": {"docker_image_uri": "r8.im/a/b@sha256:" + "a" * 64, "version_id": "b" * 64}}
    verify_deployment_identity(result, "r8.im/a/b@sha256:" + "a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="image"):
        verify_deployment_identity(result, "r8.im/a/b@sha256:" + "c" * 64, "b" * 64)
    with pytest.raises(ValueError, match="version"):
        verify_deployment_identity(result, "r8.im/a/b@sha256:" + "a" * 64, "d" * 64)
