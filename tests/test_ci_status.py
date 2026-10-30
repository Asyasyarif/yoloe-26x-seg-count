from scripts.ci_status import make_report
import json
import pytest


def test_ci_report_only_uses_allowed_metadata_and_redacts_token(tmp_path):
    (tmp_path / "push.log").write_text("failure testing example-private-token")
    report = make_report(tmp_path, {"CLI_TOKEN": "example-private-token", "UNRELATED_SECRET": "not-for-report",
                                       "PUSH_OUTCOME": "failure", "GITHUB_RUN_ID": "123"})
    assert report["push_outcome"] == "failure"
    assert "example-private-token" not in str(report)
    assert "not-for-report" not in str(report)
    assert "[REDACTED]" in report["push.log"]


@pytest.mark.parametrize("secret", ['alpha"secret', 'alpha\\secret'])
def test_ci_report_redacts_decoded_json_before_escaping(tmp_path, secret):
    (tmp_path / "push.json").write_text(json.dumps({"message": secret}))
    report = make_report(tmp_path, {"CLI_TOKEN": secret})
    assert report["push.json"]["message"] == "[REDACTED]"
