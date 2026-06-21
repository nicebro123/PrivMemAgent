import json

from tools.learned_attribute_attack import run_learned_attribute_attack


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_learned_attribute_attack_detects_attribute_signal(tmp_path):
    source = tmp_path / "source.jsonl"
    artifact = tmp_path / "public.jsonl"
    source_records = []
    artifact_records = []
    for index in range(6):
        is_positive = index in {0, 1, 2}
        source_records.append(
            {
                "uuid": f"u{index}",
                "dialogues": [
                    {
                        "content": "x",
                        "privacy_info": (
                            [
                                {
                                    "original_text": "asthma",
                                    "privacy_type": "Medical Health",
                                    "privacy_level": "PL3",
                                }
                            ]
                            if is_positive
                            else []
                        ),
                    }
                ],
            }
        )
        artifact_records.append(
            {
                "user_id": f"User-{index}",
                "public_text": (
                    "clinic inhaler treatment respiratory support"
                    if is_positive
                    else "gardening recipes travel photography"
                ),
            }
        )
    _write_jsonl(source, source_records)
    _write_jsonl(artifact, artifact_records)

    report = run_learned_attribute_attack(
        source_dataset=source,
        artifacts=[artifact],
        target="medical",
        target_terms=["medical"],
        privacy_levels={"PL3"},
        test_ratio=1 / 3,
    )

    assert report.applicable is True
    assert report.auc == 1.0
    assert report.accuracy == 1.0
    assert "clinic" in report.top_positive_tokens


def test_learned_attribute_attack_marks_tiny_or_single_class_data_not_applicable(tmp_path):
    source = tmp_path / "source.jsonl"
    artifact = tmp_path / "public.jsonl"
    _write_jsonl(
        source,
        [
            {
                "uuid": "u1",
                "dialogues": [
                    {
                        "content": "x",
                        "privacy_info": [
                            {
                                "original_text": "asthma",
                                "privacy_type": "Medical Health",
                                "privacy_level": "PL3",
                            }
                        ],
                    }
                ],
            }
        ],
    )
    _write_jsonl(artifact, [{"user_id": "User-1", "public_text": "clinic"}])

    report = run_learned_attribute_attack(
        source_dataset=source,
        artifacts=[artifact],
        target="medical",
        target_terms=["medical"],
    )

    assert report.applicable is False
    assert report.auc is None
