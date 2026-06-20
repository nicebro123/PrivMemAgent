"""Evaluation wrappers around the strict production privacy implementation."""

import os
from typing import Dict, Iterable, List, Optional

from src.privacy_masking import PrivacyStore, unmask_dialogue, validate_privacy_items
from src.privacy_masking import complete_mask_dialogue as _complete_mask_dialogue
from src.privacy_masking import mask_dialogue as _mask_dialogue


def get_privacy_items(message: dict) -> List[dict]:
    source = os.getenv("MEMPRIVACY_ANNOTATION_SOURCE", "model")
    if source not in {"model", "oracle"}:
        raise ValueError("MEMPRIVACY_ANNOTATION_SOURCE must be 'model' or 'oracle'")
    key = "privacy_info_llm" if source == "model" else "privacy_info"
    if key not in message:
        raise KeyError(
            f"Missing {key}. Run extraction first or explicitly use "
            "--annotation-source oracle for an oracle-only experiment."
        )
    return message[key]


def mask_dialogue(
    dialogue_text: str,
    privacy_items: List[Dict],
    store: PrivacyStore,
    mask_levels: Optional[List[str]] = None,
) -> str:
    """Skip malformed benchmark annotations after emitting a warning."""
    return _mask_dialogue(
        dialogue_text,
        privacy_items,
        store,
        mask_levels,
        strict=False,
    )


def complete_mask_dialogue(
    dialogue_text: str,
    privacy_items: List[Dict],
    mask_levels: Optional[List[str]] = None,
) -> str:
    """Skip malformed benchmark annotations after emitting a warning."""
    return _complete_mask_dialogue(
        dialogue_text,
        privacy_items,
        mask_levels,
        strict=False,
    )


def collect_user_privacy_items(user_data: dict) -> List[dict]:
    """Collect valid, deduplicated annotations available to a benchmark user."""
    collected = []
    seen = set()
    for message in user_data.get("dialogues", []):
        for item in validate_privacy_items(
            get_privacy_items(message),
            dialogue_text=message.get("content", ""),
            strict=False,
        ):
            key = (
                item["original_text"],
                item["privacy_type"],
                item["privacy_level"],
            )
            if key not in seen:
                seen.add(key)
                collected.append(item)
    user_name = str(user_data.get("metadata", {}).get("user_name", "")).strip()
    if user_name:
        name_item = {
            "original_text": user_name,
            "privacy_type": "Real Name",
            "privacy_level": "PL2",
        }
        key = (user_name, "Real Name", "PL2")
        if key not in seen:
            collected.append(name_item)
    return collected


def protect_known_values(
    text: str,
    privacy_items: Iterable[Dict],
    mask_levels: Optional[List[str]],
    mask_mode: str,
    store: Optional[PrivacyStore] = None,
) -> str:
    """Protect known local values before any evaluation text leaves the edge."""
    applicable = [
        item
        for item in privacy_items
        if item.get("original_text")
        and item["original_text"] in text
        and (mask_levels is None or item.get("privacy_level") in mask_levels)
    ]
    if not applicable:
        return text
    if mask_mode == "complete":
        return complete_mask_dialogue(text, applicable, mask_levels)
    if store is None:
        raise ValueError("a privacy store is required for reversible query protection")
    return mask_dialogue(text, applicable, store, mask_levels)


__all__ = [
    "PrivacyStore",
    "collect_user_privacy_items",
    "complete_mask_dialogue",
    "get_privacy_items",
    "mask_dialogue",
    "protect_known_values",
    "unmask_dialogue",
    "validate_privacy_items",
]
