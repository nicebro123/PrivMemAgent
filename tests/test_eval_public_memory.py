import json

from evaluation.eval_public_memory import compile_dataset


def test_public_memory_runner_never_serializes_raw_secret(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "public.jsonl"
    metrics_path = tmp_path / "metrics.json"
    cloud_safe_path = tmp_path / "cloud-safe.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "uuid": "u1",
                "dialogues": [
                    {
                        "content": "Email alice@example.com. Code 829417.",
                        "privacy_info": [
                            {
                                "original_text": "alice@example.com",
                                "privacy_type": "Email",
                                "privacy_level": "PL2",
                            },
                            {
                                "original_text": "829417",
                                "privacy_type": "Verification Code",
                                "privacy_level": "PL4",
                            },
                        ],
                    },
                    {
                        "content": "Use alice@example.com again and discard 829417.",
                        "privacy_info": [],
                    },
                ],
                "questions": [
                    {
                        "question": "Should I use alice@example.com or code 829417?",
                        "answer": "Use alice@example.com.",
                        "evidence": "The email is alice@example.com and the code is 829417.",
                        "all_options": ["alice@example.com", "829417"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "public_memory": {
            "policy_version": "test-v1",
            "policy": {
                "pl2": "public_abstract",
                "pl3": "local_only",
                "pl4": "drop",
            },
            "alias_scope": {
                "pl2": "task",
                "pl3": "session",
                "pl4": "turn",
            },
            "selector": {
                "utility_floor": 0.75,
                "max_leakage": 0.35,
                "max_public_tokens": 128,
            },
            "leakage_budget": {
                "exact_recovery": 0.0,
                "cross_scope_linkability": 0.0,
                "pl4_public_retention": 0.0,
                "minimum_token_reduction": -1.0,
            },
        }
    }

    result = compile_dataset(
        input_path=input_path,
        output_path=output_path,
        metrics_path=metrics_path,
        state_dir=tmp_path / "state",
        config=config,
        annotation_source="oracle",
        cloud_safe_dataset_path=cloud_safe_path,
    )

    serialized = output_path.read_text(encoding="utf-8")
    cloud_serialized = cloud_safe_path.read_text(encoding="utf-8")
    assert "u1" not in serialized
    assert "u1:message" not in serialized
    assert "alice@example.com" not in serialized
    assert "829417" not in serialized
    assert "original_text" not in serialized
    assert "source_fingerprint" not in serialized
    assert "privacy_type" not in serialized
    assert "privacy_level" not in serialized
    assert "provenance_id" not in serialized
    assert "alice@example.com" not in cloud_serialized
    assert "829417" not in cloud_serialized
    assert json.loads(cloud_serialized)["uuid"].startswith("User-")
    assert result["audit"]["exact_recovery_rate"] == 0.0
    assert result["audit"]["pl4_public_retention_rate"] == 0.0
    assert result["audit"]["cross_scope_linkability_rate"] is None
    assert result["audit"]["cross_scope_linkability_applicable"] is False
    assert result["cloud_safe_dataset"]["audit"]["exact_recovery_rate"] == 0.0
    assert result["cloud_safe_dataset"]["audit"]["passed"] is True
    assert result["invalid_annotation_count"] == 0
    assert result["utility_proxy"]["oracle_type_local_recoverability"] == 1.0
    assert result["utility_proxy"]["local_recoverability_applicable"] is True


def test_public_memory_runner_excludes_invalid_annotations_from_audit(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "uuid": "u1",
                "dialogues": [
                    {
                        "content": "No secret appears here.",
                        "privacy_info": [
                            {
                                "original_text": "missing-secret",
                                "privacy_type": "Password",
                                "privacy_level": "PL4",
                            }
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "public_memory": {
            "policy": {"pl2": "public_abstract", "pl3": "local_only", "pl4": "drop"},
            "selector": {},
            "leakage_budget": {"minimum_token_reduction": -1.0},
        }
    }

    result = compile_dataset(
        input_path=input_path,
        output_path=tmp_path / "public.jsonl",
        metrics_path=tmp_path / "metrics.json",
        state_dir=tmp_path / "state",
        config=config,
        annotation_source="oracle",
    )

    assert result["invalid_annotation_count"] == 1
    assert result["audit"]["total_sensitive_items"] == 0
    assert result["audit"]["total_pl4_items"] == 0


def test_public_memory_runner_scrubs_residual_user_level_sensitive_terms(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "public.jsonl"
    metrics_path = tmp_path / "metrics.json"
    cloud_safe_path = tmp_path / "cloud-safe.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "uuid": "u1",
                "metadata": {"user_name": "David"},
                "dialogues": [
                    {
                        "role": "user",
                        "content": "My Amex card ending in 8865 is private.",
                        "privacy_info": [
                            {
                                "original_text": "Amex card ending in 8865",
                                "privacy_type": "Financial Account",
                                "privacy_level": "PL3",
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "David should never expose Amex in cloud memory.",
                        "privacy_info": [],
                    },
                ],
                "questions": [
                    {
                        "question": "What did David ask about Amex?",
                        "answer": "David asked about Amex.",
                        "evidence": "David mentioned Amex.",
                        "all_options": ["David", "Amex", "public option"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "public_memory": {
            "policy": {"pl2": "public_abstract", "pl3": "local_only", "pl4": "drop"},
            "selector": {},
            "leakage_budget": {"minimum_token_reduction": -1.0},
        }
    }

    compile_dataset(
        input_path=input_path,
        output_path=output_path,
        metrics_path=metrics_path,
        state_dir=tmp_path / "state",
        config=config,
        annotation_source="oracle",
        cloud_safe_dataset_path=cloud_safe_path,
    )

    serialized = output_path.read_text(encoding="utf-8")
    cloud_serialized = cloud_safe_path.read_text(encoding="utf-8")
    assert "David" not in serialized
    assert "Amex" not in serialized
    assert "8865" not in serialized
    assert "David" not in cloud_serialized
    assert "Amex" not in cloud_serialized
    assert "8865" not in cloud_serialized
    assert "public option" in cloud_serialized



def test_public_memory_runner_keeps_pl1_public_preferences(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "public.jsonl"
    metrics_path = tmp_path / "metrics.json"
    cloud_safe_path = tmp_path / "cloud-safe.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "uuid": "u1",
                "dialogues": [
                    {
                        "role": "user",
                        "content": "I prefer vegetarian restaurants for team dinners.",
                        "privacy_info": [
                            {
                                "original_text": "vegetarian restaurants",
                                "privacy_type": "Preference",
                                "privacy_level": "PL1",
                            }
                        ],
                    }
                ],
                "questions": [
                    {
                        "question": "What kind of restaurants does the user prefer?",
                        "answer": "The user prefers vegetarian restaurants.",
                        "evidence": "The user said they prefer vegetarian restaurants.",
                        "all_options": ["vegetarian restaurants", "steakhouses"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "public_memory": {
            "policy": {"pl1": "public", "pl2": "public_abstract", "pl3": "local_only", "pl4": "drop"},
            "selector": {},
            "leakage_budget": {"minimum_token_reduction": -1.0},
        }
    }

    compile_dataset(
        input_path=input_path,
        output_path=output_path,
        metrics_path=metrics_path,
        state_dir=tmp_path / "state",
        config=config,
        annotation_source="oracle",
        cloud_safe_dataset_path=cloud_safe_path,
    )

    serialized = output_path.read_text(encoding="utf-8")
    cloud_serialized = cloud_safe_path.read_text(encoding="utf-8")
    assert "vegetarian restaurants" in serialized
    assert "vegetarian restaurants" in cloud_serialized
