from __future__ import annotations

import json

from src.sufficiency_selector import RepresentationCandidate, SelectorConfig
from src.utility_leakage_selector import (
    LearnedUtilityLeakageSelector,
    LinearUtilityLeakageRanker,
)


def candidate(text, specificity, utility, leakage, tokens):
    return RepresentationCandidate(
        text=text,
        specificity=specificity,
        utility_score=utility,
        leakage_score=leakage,
        token_count=tokens,
        representation_type=text,
    )


def test_learned_selector_prefers_higher_ranked_candidate_within_budgets():
    selector = LearnedUtilityLeakageSelector(
        config=SelectorConfig(utility_floor=0.5, max_leakage=0.4, max_public_tokens=20),
        ranker=LinearUtilityLeakageRanker(
            {
                "utility_score": 2.0,
                "leakage_score": -1.0,
                "abstraction_level": -0.01,
            }
        ),
    )

    result = selector.select(
        [
            candidate("private detail", 1, 0.6, 0.05, 2),
            candidate("contact information", 2, 0.9, 0.2, 2),
        ],
        privacy_item={
            "original_text": "alice@example.com",
            "privacy_type": "Email",
            "privacy_level": "PL2",
        },
    )

    assert result.selected.text == "contact information"
    assert result.reason == "highest learned utility-leakage score within hard budgets"


def test_learned_selector_fails_closed_on_exact_leak():
    selector = LearnedUtilityLeakageSelector(
        config=SelectorConfig(utility_floor=0.1, max_leakage=1.0, max_public_tokens=20)
    )

    result = selector.select(
        [candidate("alice@example.com", 1, 1.0, 0.1, 1)],
        privacy_item={
            "original_text": "alice@example.com",
            "privacy_type": "Email",
            "privacy_level": "PL2",
        },
    )

    assert result.selected is None
    assert "exact leak" in result.reason


def test_learned_selector_fails_closed_on_private_fragment_leak():
    selector = LearnedUtilityLeakageSelector(
        config=SelectorConfig(utility_floor=0.1, max_leakage=1.0, max_public_tokens=20)
    )

    result = selector.select(
        [candidate("contact domain example.com", 1, 1.0, 0.1, 3)],
        privacy_item={
            "original_text": "alice@example.com",
            "privacy_type": "Email",
            "privacy_level": "PL2",
        },
    )

    assert result.selected is None
    assert "exact leak" in result.reason


def test_learned_selector_fails_closed_on_numeric_suffix_leak():
    selector = LearnedUtilityLeakageSelector(
        config=SelectorConfig(utility_floor=0.1, max_leakage=1.0, max_public_tokens=20)
    )

    result = selector.select(
        [candidate("phone ending 7843", 1, 1.0, 0.1, 3)],
        privacy_item={
            "original_text": "+1-617-492-7843",
            "privacy_type": "Phone Number",
            "privacy_level": "PL2",
        },
    )

    assert result.selected is None
    assert "exact leak" in result.reason


def test_linear_ranker_loads_weights_from_artifact(tmp_path):
    artifact = tmp_path / "selector.json"
    artifact.write_text(
        json.dumps({"weights": {"utility_score": 3.0, "leakage_score": -2.0}}),
        encoding="utf-8",
    )

    selector = LearnedUtilityLeakageSelector.from_artifact(
        artifact,
        config=SelectorConfig(utility_floor=0.1, max_leakage=1.0, max_public_tokens=20),
    )
    result = selector.select(
        [
            candidate("low utility", 1, 0.2, 0.1, 1),
            candidate("high utility", 2, 0.9, 0.2, 1),
        ],
        privacy_item={"privacy_level": "PL2", "privacy_type": "Preference", "original_text": ""},
    )

    assert result.selected.text == "high utility"
