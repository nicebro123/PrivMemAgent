import json
from pathlib import Path

from evaluation.eval_public_memory import compile_dataset
from src.privacy_masking import validate_privacy_items


def test_memory_eval_smoke_dataset_is_self_contained():
    path = Path("data/memory_eval_smoke.jsonl")
    users = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    assert len(users) == 1
    user = users[0]
    assert len(user["dialogues"]) == 4
    assert len(user["questions"]) == 1
    assert user["questions"][0]["answer"] == "A"
    for message in user["dialogues"]:
        validate_privacy_items(
            message["privacy_info"],
            dialogue_text=message["content"],
            strict=True,
        )


def test_minimal_public_smoke_preserves_answer_fact(tmp_path):
    config = {
        "public_memory": {
            "policy": {"pl2": "public_abstract", "pl3": "local_only", "pl4": "drop"},
            "selector": {},
            "leakage_budget": {"minimum_token_reduction": -1.0},
        }
    }
    cloud_safe_path = tmp_path / "cloud-safe.jsonl"
    compile_dataset(
        input_path=Path("data/memory_eval_smoke.jsonl"),
        output_path=tmp_path / "public.jsonl",
        metrics_path=tmp_path / "metrics.json",
        state_dir=tmp_path / "state",
        config=config,
        annotation_source="oracle",
        cloud_safe_dataset_path=cloud_safe_path,
    )
    cloud_user = json.loads(cloud_safe_path.read_text())
    cloud_dialogue = " ".join(
        message["content"] for message in cloud_user["dialogues"]
    )

    assert "birdwatching" in cloud_dialogue.lower()
    assert 'birdwatching."' in cloud_user["questions"][0]["evidence"]
    assert "avery.smoke@example.com" not in cloud_dialogue
    assert "829417" not in cloud_dialogue
