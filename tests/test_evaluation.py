import pytest

from scripts.evaluate import summarize, validate_corpus


def test_accuracy_metrics_are_derived_from_reviewed_counts():
    result = summarize([
        {"predicted_count": 3, "ground_truth_count": 2, "seconds": 1.0},
        {"predicted_count": 0, "ground_truth_count": 0, "seconds": 3.0},
    ])
    assert result["count_mae"] == 0.5
    assert result["exact_count_match_rate"] == 0.5
    assert result["mean_seconds"] == 2.0


def test_unreviewed_small_or_duplicate_corpus_cannot_claim_accuracy():
    with pytest.raises(ValueError, match="50"):
        validate_corpus([])
    rows = [{"image": "same.jpg", "prompt": "person", "counts": {"person": 0}, "reviewed": True, "license": "CC0"}] * 50
    with pytest.raises(ValueError, match="unique"):
        validate_corpus(rows)


def test_total_count_does_not_hide_class_swap():
    result = summarize([{"predicted_count": 2, "ground_truth_count": 2, "seconds": 1,
                         "absolute_error_by_class": {"person": 1, "car": 1}}])
    assert result["exact_count_match_rate"] == 1
    assert result["all_classes_exact_match_rate"] == 0
    assert result["class_count_mae"] == 1
