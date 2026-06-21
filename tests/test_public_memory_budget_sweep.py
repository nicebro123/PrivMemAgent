import json

from tools.public_memory_budget_sweep import run_budget_sweep


def _config():
    return {
        "public_memory": {
            "policy_version": "test-v1",
            "policy": {"pl1": "public", "pl2": "public_abstract", "pl3": "local_only", "pl4": "drop"},
            "selector": {"utility_floor": 0.75, "max_leakage": 0.35, "max_public_tokens": 128},
            "leakage_budget": {"minimum_token_reduction": -1.0},
        }
    }


def test_budget_sweep_outputs_pareto_metrics(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "uuid": "u1",
                "dialogues": [
                    {
                        "role": "user",
                        "content": "I prefer quiet vegetarian restaurants. My email is alice@example.com.",
                        "privacy_info": [
                            {
                                "original_text": "alice@example.com",
                                "privacy_type": "Email",
                                "privacy_level": "PL2",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": "The temporary verification code is 829417.",
                        "privacy_info": [
                            {
                                "original_text": "829417",
                                "privacy_type": "Verification Code",
                                "privacy_level": "PL4",
                            }
                        ],
                    },
                ],
                "questions": [
                    {
                        "question": "What restaurants does the user prefer?",
                        "answer": "The user prefers quiet vegetarian restaurants.",
                        "all_options": ["quiet vegetarian restaurants", "loud steakhouses"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_budget_sweep(
        input_path=input_path,
        output_dir=tmp_path / "sweep",
        config=_config(),
        budgets=[8, 32],
        annotation_source="oracle",
        minimum_token_reduction=-1.0,
    )

    assert summary["budgets"] == [8, 32]
    assert len(summary["runs"]) == 2
    for run in summary["runs"]:
        assert run["audit_passed"] is True
        assert run["cloud_audit_passed"] is True
        assert run["adversarial_passed"] is True
        assert run["adversarial_failure_count"] == 0
        assert run["exact_recovery_rate"] == 0.0
        assert run["pl4_public_retention_rate"] == 0.0
        assert run["public_token_count"] <= run["source_token_count"]
        assert "non_private_answer_token_recall" in run
        assert (tmp_path / "sweep" / f"budget_{run['max_public_tokens']}" / "metrics.json").exists()
        assert (
            tmp_path
            / "sweep"
            / f"budget_{run['max_public_tokens']}"
            / "adversarial_audit.json"
        ).exists()


def test_budget_sweep_validates_budget_list(tmp_path):
    try:
        run_budget_sweep(
            input_path=tmp_path / "missing.jsonl",
            output_dir=tmp_path / "sweep",
            config=_config(),
            budgets=[],
            annotation_source="oracle",
        )
    except ValueError as exc:
        assert "at least one budget" in str(exc)
    else:
        raise AssertionError("expected ValueError")
