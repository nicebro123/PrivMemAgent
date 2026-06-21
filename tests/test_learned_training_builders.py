from __future__ import annotations

import json

from training.build_abstraction_data import build_examples
from training.build_selector_data import build_selector_examples
from training.train_abstraction_generator import distill_templates
from training.train_utility_leakage_selector import fit_simple_weights


def test_build_abstraction_examples_from_model_annotations(tmp_path):
    input_path = tmp_path / "predictions.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "uuid": "u1",
                "dialogues": [
                    {
                        "role": "user",
                        "content": "Email alice@example.com.",
                        "privacy_info_llm": [
                            {
                                "original_text": "alice@example.com",
                                "privacy_type": "Email",
                                "privacy_level": "PL2",
                            }
                        ],
                    }
                ],
                "questions": [{"question": "How should I contact the user?"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = build_examples(input_path, annotation_source="model")

    assert len(examples) == 1
    assert examples[0]["privacy_item"]["privacy_type"] == "Email"
    assert any(c["text"] == "contact information" for c in examples[0]["candidates"])
    assert all(c["text"] != "alice@example.com" for c in examples[0]["candidates"])


def test_build_selector_examples_from_abstraction_examples(tmp_path):
    abstraction_path = tmp_path / "abstractions.jsonl"
    abstraction_path.write_text(
        json.dumps(
            {
                "user_id": "u1",
                "message_id": "m1",
                "privacy_item": {"privacy_type": "Email", "privacy_level": "PL2"},
                "candidates": [
                    {
                        "text": "contact information",
                        "level": 2,
                        "representation_type": "category_abstract",
                        "utility_score": 0.82,
                        "leakage_score": 0.2,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_selector_examples(abstraction_path)

    assert len(rows) == 1
    assert rows[0]["label_positive"] is True
    assert rows[0]["label_score"] > 0


def test_train_abstraction_generator_distills_template_artifact(tmp_path):
    examples = tmp_path / "abstractions.jsonl"
    examples.write_text(
        json.dumps(
            {
                "privacy_item": {"privacy_type": "Email", "privacy_level": "PL2"},
                "candidates": [
                    {
                        "text": "contact information",
                        "level": 2,
                        "representation_type": "category_abstract",
                        "utility_score": 0.8,
                        "leakage_score": 0.2,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    templates = distill_templates(examples)

    assert templates[0]["privacy_type"] == "Email"
    assert templates[0]["text"] == "contact information"
    assert templates[0]["support"] == 1


def test_train_selector_fits_weight_artifact(tmp_path):
    examples = tmp_path / "selector.jsonl"
    examples.write_text(
        json.dumps({"label_positive": True, "utility_score": 0.9, "leakage_score": 0.2})
        + "\n"
        + json.dumps({"label_positive": False, "utility_score": 0.2, "leakage_score": 0.8})
        + "\n",
        encoding="utf-8",
    )

    weights = fit_simple_weights(examples)

    assert weights["utility_score"] > 1.0
    assert weights["leakage_score"] < -1.0
