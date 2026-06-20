import json

from tools.validate_dataset import validate_dataset


def test_dataset_validator_reports_missing_spans(tmp_path):
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "uuid": "u1",
                "dialogues": [
                    {
                        "content": "Email alice@example.com",
                        "privacy_info": [
                            {"original_text": "missing@example.com"},
                            {"original_text": "alice@example.com"},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = validate_dataset(dataset)
    assert len(issues) == 1
    assert issues[0]["reason"] == "span_not_in_message"
