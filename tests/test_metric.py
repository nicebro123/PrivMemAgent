import pytest

from evaluation.metric import (
    _maximum_weight_assignment,
    cosine_similarity,
    evaluate_privacy,
    match_items,
)


def test_wrong_span_cannot_score_from_type_and_level_only():
    prediction = [
        {
            "original_text": "not present",
            "privacy_type": "Email",
            "privacy_level": "PL2",
        }
    ]
    reference = [
        {
            "original_text": "alice@example.com",
            "privacy_type": "Email",
            "privacy_level": "PL2",
        }
    ]
    pred_scores, _, subscores = match_items(
        prediction,
        reference,
        "alice@example.com",
        {},
        mode="mean",
    )
    assert pred_scores == [0.0]
    assert subscores[0] == (0.0, 1.0, 1.0)


def test_cosine_similarity_is_bounded():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_hungarian_matching_finds_global_optimum():
    weights = [[0.9, 0.8], [0.85, 0.1]]
    assignment = _maximum_weight_assignment(weights)
    assert sum(weights[row][column] for row, column in assignment) == pytest.approx(1.65)


def test_missing_reference_penalizes_subscore_means():
    messages = [{"role": "user", "content": "alice@example.com and 555-0101"}]
    prediction = [
        {
            "original_text": "alice@example.com",
            "privacy_type": "Email",
            "privacy_level": "PL2",
        }
    ]
    reference = [
        prediction[0],
        {
            "original_text": "555-0101",
            "privacy_type": "Phone Number",
            "privacy_level": "PL2",
        },
    ]
    result = evaluate_privacy(messages, prediction, reference, mode="product")
    assert result["overall"]["precision"] == 1.0
    assert result["overall"]["recall"] == 0.5
    assert result["mean_subscores"]["original_text_mean"] == 0.5
