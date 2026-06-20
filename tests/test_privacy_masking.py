import concurrent.futures
import hashlib
import hmac
import os
import sqlite3

import pytest
from cryptography.fernet import Fernet

from evaluation.privacy_masking import (
    collect_user_privacy_items,
    get_privacy_items,
    protect_known_values,
)
from src.privacy_masking import (
    PrivacyStore,
    _validate_llm_endpoint,
    complete_mask_dialogue,
    mask_dialogue,
    unmask_dialogue,
    validate_privacy_items,
)


def make_store(tmp_path, namespace="user-a", mask_mode="type_specific"):
    return PrivacyStore(
        db_path=str(tmp_path / "privacy.db"),
        key_path=str(tmp_path / "privacy.key"),
        namespace=namespace,
        mask_mode=mask_mode,
    )


def test_remote_detector_is_rejected_by_default():
    with pytest.raises(ValueError, match="Refusing"):
        _validate_llm_endpoint(
            {"llm": {"base_url": "https://api.example.com/v1", "allow_remote": False}}
        )


def test_local_detector_is_allowed():
    _validate_llm_endpoint({"llm": {"base_url": "http://127.0.0.1:8000/v1", "allow_remote": False}})


def test_store_encrypts_original_text_at_rest(tmp_path):
    secret = "otp-829417-super-secret"
    store = make_store(tmp_path)
    store.get_or_create(secret, "Verification Code", "PL4")
    store.close()

    database_bytes = (tmp_path / "privacy.db").read_bytes()
    assert secret.encode() not in database_bytes
    assert os.stat(tmp_path / "privacy.db").st_mode & 0o077 == 0
    assert os.stat(tmp_path / "privacy.key").st_mode & 0o077 == 0


def test_namespace_isolation_uses_distinct_masks(tmp_path):
    key = Fernet.generate_key().decode()
    db_path = str(tmp_path / "shared.db")
    first = PrivacyStore(db_path, namespace="alice", encryption_key=key)
    second = PrivacyStore(db_path, namespace="bob", encryption_key=key)
    first_mask = first.get_or_create("same@example.com", "Email", "PL2")
    second_mask = second.get_or_create("same@example.com", "Email", "PL2")

    assert first_mask != second_mask
    assert first.query_by_mask(second_mask) is None
    assert second.query_by_mask(first_mask) is None
    first.close()
    second.close()


def test_same_value_can_have_distinct_semantic_types_without_level_downgrade(tmp_path):
    store = make_store(tmp_path)
    email_mask = store.get_or_create("shared-value", "Email", "PL4")
    username_mask = store.get_or_create("shared-value", "Account ID/Username", "PL2")
    repeated_email_mask = store.get_or_create("shared-value", "Email", "PL2")

    assert email_mask != username_mask
    assert repeated_email_mask == email_mask
    email_record = store.query_by_privacy_type("Email")[0]
    assert email_record["privacy_level"] == "PL4"
    store.close()


def test_mask_round_trip_supports_punctuation_in_type(tmp_path):
    store = make_store(tmp_path)
    text = "Rotate API key sk-live-123 immediately."
    items = [
        {
            "original_text": "sk-live-123",
            "privacy_type": "API-Key",
            "privacy_level": "PL4",
        }
    ]
    masked = mask_dialogue(text, items, store, ["PL4"])

    assert "sk-live-123" not in masked
    assert unmask_dialogue(masked, store) == text
    store.close()


def test_empty_and_hallucinated_spans_are_rejected(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        mask_dialogue(
            "abc",
            [{"original_text": "", "privacy_type": "Email", "privacy_level": "PL2"}],
            store,
            ["PL2"],
        )
    with pytest.raises(ValueError, match="does not occur"):
        validate_privacy_items(
            [
                {
                    "original_text": "missing",
                    "privacy_type": "Email",
                    "privacy_level": "PL2",
                }
            ],
            dialogue_text="abc",
        )
    store.close()


def test_overlapping_spans_prefer_longest_match(tmp_path):
    store = make_store(tmp_path)
    text = "Code 123456 is active."
    items = [
        {
            "original_text": "123456",
            "privacy_type": "Verification Code",
            "privacy_level": "PL4",
        },
        {
            "original_text": "345",
            "privacy_type": "Verification Code",
            "privacy_level": "PL4",
        },
    ]
    masked = mask_dialogue(text, items, store, ["PL4"])
    assert masked.count("<MPM_") == 1
    assert unmask_dialogue(masked, store) == text
    store.close()


def test_duplicate_span_labels_choose_deterministic_most_sensitive_type(tmp_path):
    items = [
        {
            "original_text": "shared",
            "privacy_type": "Email",
            "privacy_level": "PL2",
        },
        {
            "original_text": "shared",
            "privacy_type": "Password",
            "privacy_level": "PL4",
        },
    ]
    first = PrivacyStore(
        str(tmp_path / "first.db"),
        key_path=str(tmp_path / "first.key"),
        namespace="user",
    )
    second = PrivacyStore(
        str(tmp_path / "second.db"),
        key_path=str(tmp_path / "second.key"),
        namespace="user",
    )
    mask_dialogue("shared", items, first, ["PL2", "PL4"])
    mask_dialogue("shared", list(reversed(items)), second, ["PL2", "PL4"])

    assert [record["privacy_type"] for record in first.get_all()] == ["Password"]
    assert [record["privacy_type"] for record in second.get_all()] == ["Password"]
    first.close()
    second.close()


def test_complete_mask_uses_same_validation():
    with pytest.raises(ValueError, match="does not occur"):
        complete_mask_dialogue(
            "safe",
            [
                {
                    "original_text": "missing",
                    "privacy_type": "Password",
                    "privacy_level": "PL4",
                }
            ],
            ["PL4"],
        )


def test_concurrent_get_or_create_is_atomic(tmp_path):
    key = Fernet.generate_key().decode()
    db_path = str(tmp_path / "concurrent.db")

    def create_mask(_):
        store = PrivacyStore(db_path, namespace="same-user", encryption_key=key)
        try:
            return store.get_or_create("alice@example.com", "Email", "PL2")
        finally:
            store.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        masks = list(executor.map(create_mask, range(32)))
    assert len(set(masks)) == 1

    store = PrivacyStore(db_path, namespace="same-user", encryption_key=key)
    assert len(store.get_all()) == 1
    assert store.get_all()[0]["mask"] == masks[0]
    store.close()


def test_legacy_plaintext_database_is_migrated_and_scrubbed(tmp_path):
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE privacy_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT NOT NULL UNIQUE,
            privacy_type TEXT NOT NULL,
            privacy_level TEXT NOT NULL,
            mask TEXT NOT NULL UNIQUE
        )
        """
    )
    connection.execute(
        "INSERT INTO privacy_items "
        "(original_text, privacy_type, privacy_level, mask) VALUES (?, ?, ?, ?)",
        ("legacy-secret", "API-Key", "PL4", "<API-Key_1>"),
    )
    connection.commit()
    connection.close()

    store = PrivacyStore(
        str(db_path),
        namespace="legacy-user",
        key_path=str(tmp_path / "legacy.key"),
    )
    assert store.query_by_mask("<API-Key_1>")["original_text"] == "legacy-secret"
    assert unmask_dialogue("Use <API-Key_1>", store) == "Use legacy-secret"
    store.close()
    assert b"legacy-secret" not in db_path.read_bytes()


def test_secure_v1_database_migrates_to_type_aware_schema(tmp_path):
    key = Fernet.generate_key()
    db_path = tmp_path / "secure-v1.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE privacy_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT NOT NULL,
            original_hash TEXT NOT NULL,
            original_ciphertext BLOB NOT NULL,
            privacy_type TEXT NOT NULL,
            privacy_level TEXT NOT NULL,
            mask TEXT NOT NULL,
            UNIQUE(namespace, original_hash),
            UNIQUE(namespace, mask)
        )
        """
    )
    original_hash = hmac.new(
        key,
        b"user-a\0shared-value",
        hashlib.sha256,
    ).hexdigest()
    connection.execute(
        """
        INSERT INTO privacy_items (
            namespace, original_hash, original_ciphertext,
            privacy_type, privacy_level, mask
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "user-a",
            original_hash,
            Fernet(key).encrypt(b"shared-value"),
            "Email",
            "PL2",
            "<MPM_Email_1_aaaaaaaaaaaa>",
        ),
    )
    connection.commit()
    connection.close()

    store = PrivacyStore(
        str(db_path),
        namespace="user-a",
        encryption_key=key.decode(),
    )
    second_mask = store.get_or_create("shared-value", "Account ID/Username", "PL2")
    assert second_mask != "<MPM_Email_1_aaaaaaaaaaaa>"
    assert len(store.get_all()) == 2
    store.close()


def test_evaluation_annotation_source_never_silently_falls_back(monkeypatch):
    message = {"privacy_info": [{"original_text": "oracle"}]}
    monkeypatch.setenv("MEMPRIVACY_ANNOTATION_SOURCE", "model")
    with pytest.raises(KeyError, match="privacy_info_llm"):
        get_privacy_items(message)
    monkeypatch.setenv("MEMPRIVACY_ANNOTATION_SOURCE", "oracle")
    assert get_privacy_items(message) == message["privacy_info"]


def test_evaluation_query_protection_reuses_ingestion_alias(tmp_path):
    store = make_store(tmp_path)
    item = {
        "original_text": "alice@example.com",
        "privacy_type": "Email",
        "privacy_level": "PL2",
    }
    ingested = mask_dialogue(
        "Email alice@example.com",
        [item],
        store,
        ["PL2"],
    )
    protected_query = protect_known_values(
        "Should I email alice@example.com?",
        [item],
        ["PL2"],
        "type_specific",
        store,
    )

    alias = store.get_all()[0]["mask"]
    assert alias in ingested
    assert alias in protected_query
    assert "alice@example.com" not in protected_query
    store.close()


def test_complete_query_protection_does_not_require_local_storage():
    protected = protect_known_values(
        "Use recovery code 829417.",
        [
            {
                "original_text": "829417",
                "privacy_type": "Recovery Code",
                "privacy_level": "PL4",
            }
        ],
        ["PL4"],
        "complete",
    )

    assert protected == "Use recovery code ***."


def test_collect_user_privacy_items_uses_selected_annotation_source(monkeypatch):
    monkeypatch.setenv("MEMPRIVACY_ANNOTATION_SOURCE", "oracle")
    user = {
        "metadata": {"user_name": "Alice Example"},
        "dialogues": [
            {
                "content": "Email alice@example.com",
                "privacy_info": [
                    {
                        "original_text": "alice@example.com",
                        "privacy_type": "Email",
                        "privacy_level": "PL2",
                    },
                    {
                        "original_text": "missing@example.com",
                        "privacy_type": "Email",
                        "privacy_level": "PL2",
                    },
                ],
            }
        ]
    }

    assert collect_user_privacy_items(user) == [
        {
            "original_text": "alice@example.com",
            "privacy_type": "Email",
            "privacy_level": "PL2",
        },
        {
            "original_text": "Alice Example",
            "privacy_type": "Real Name",
            "privacy_level": "PL2",
        },
    ]
