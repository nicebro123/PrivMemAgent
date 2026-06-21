import json

import pytest

from evaluation.cloud_safe_guard import (
    enforce_no_mask_input_safety,
    validate_cloud_safe_dataset,
)


def test_cloud_safe_guard_accepts_public_dataset(tmp_path):
    path = tmp_path / "cloud.jsonl"
    path.write_text(
        json.dumps(
            {
                "uuid": "User-abc123",
                "metadata": {"user_name": "User-abc123", "language": "en"},
                "dialogues": [{"role": "user", "content": "public preference"}],
                "questions": [{"question": "q", "answer": "a", "all_options": ["a"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_cloud_safe_dataset(path)

    assert report.passed is True
    assert enforce_no_mask_input_safety(path, is_mask=False).passed is True


def test_cloud_safe_guard_rejects_raw_dataset_for_no_mask(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps(
            {
                "uuid": "raw-user-id",
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

    with pytest.raises(ValueError, match="Refusing to run with --no-mask"):
        enforce_no_mask_input_safety(path, is_mask=False)


def test_cloud_safe_guard_skips_when_masking_or_explicitly_overridden(tmp_path):
    missing_path = tmp_path / "missing.jsonl"

    assert enforce_no_mask_input_safety(missing_path, is_mask=True) is None
    assert (
        enforce_no_mask_input_safety(
            missing_path,
            is_mask=False,
            allow_unsafe_no_mask=True,
        )
        is None
    )


def test_cloud_safe_guard_respects_user_limit(tmp_path):
    path = tmp_path / "mixed.jsonl"
    safe = {
        "uuid": "User-safe",
        "metadata": {"user_name": "User-safe"},
        "dialogues": [{"content": "public"}],
        "questions": [],
    }
    raw = {
        "uuid": "raw-user",
        "metadata": {"user_name": "Alice"},
        "dialogues": [{"content": "secret", "privacy_info": []}],
        "questions": [],
    }
    path.write_text(
        json.dumps(safe) + "\n" + json.dumps(raw) + "\n",
        encoding="utf-8",
    )

    assert validate_cloud_safe_dataset(path, user_limit=1).passed is True
    assert validate_cloud_safe_dataset(path).passed is False
