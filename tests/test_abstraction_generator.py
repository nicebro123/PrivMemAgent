from __future__ import annotations

import json

from src.abstraction_generator import (
    AbstractionInput,
    ArtifactBackedAbstractionGenerator,
    RuleBasedAbstractionGenerator,
)


def abstraction_input(privacy_type="Email", original="alice@example.com"):
    return AbstractionInput(
        user_id="u1",
        message_id="m1",
        role="user",
        message_text=f"My value is {original}",
        privacy_item={
            "original_text": original,
            "privacy_type": privacy_type,
            "privacy_level": "PL2",
        },
    )


def test_rule_generator_produces_safe_category_abstraction():
    candidates = RuleBasedAbstractionGenerator().generate(abstraction_input())

    texts = [candidate.text for candidate in candidates]
    assert "alice@example.com" not in texts
    assert "contact information" in texts
    assert any(candidate.representation_type == "drop" for candidate in candidates)


def test_artifact_generator_filters_exact_private_leaks(tmp_path):
    artifact = tmp_path / "abstractions.json"
    artifact.write_text(
        json.dumps(
            {
                "templates": [
                    {
                        "privacy_type": "Email",
                        "privacy_level": "PL2",
                        "text": "alice@example.com",
                        "abstraction_level": 2,
                        "utility_score": 0.99,
                        "leakage_score": 0.99,
                        "representation_type": "unsafe_copy",
                    },
                    {
                        "privacy_type": "Email",
                        "privacy_level": "PL2",
                        "text": "reachable through private contact information",
                        "abstraction_level": 2,
                        "utility_score": 0.9,
                        "leakage_score": 0.2,
                        "representation_type": "learned_category",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = ArtifactBackedAbstractionGenerator(artifact).generate(abstraction_input())

    texts = [candidate.text for candidate in candidates]
    assert "alice@example.com" not in texts
    assert "reachable through private contact information" in texts
