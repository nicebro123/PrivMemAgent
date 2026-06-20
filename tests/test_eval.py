import json

from evaluation.eval import run_evaluation


def test_failed_prediction_is_scored_instead_of_dropped(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "annotated.jsonl"
    metrics_path = tmp_path / "metrics.json"
    input_path.write_text(
        json.dumps(
            {
                "uuid": "user-1",
                "metadata": {"user_name": "Alice"},
                "dialogues": [
                    {
                        "role": "user",
                        "content": "Email alice@example.com",
                        "privacy_info": [
                            {
                                "original_text": "alice@example.com",
                                "privacy_type": "Email",
                                "privacy_level": "PL2",
                            }
                        ],
                    }
                ],
                "questions": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_evaluation(
        input_path,
        output_path,
        metrics_path,
        writer=lambda _prompt, _query: "not-json",
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["false_prediction_count"] == 1
    assert len(metrics["product"]) == 1
    assert metrics["product"][0]["overall"]["recall"] == 0.0
    annotated = json.loads(output_path.read_text(encoding="utf-8"))
    assert annotated["dialogues"][0]["privacy_info_llm"] == []
