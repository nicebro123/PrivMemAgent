from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

_RAW_METADATA_KEYS = {
    "privacy_info",
    "privacy_info_llm",
    "source_fingerprint",
    "source_message_id",
    "source_item_index",
    "scope_id",
    "provenance_id",
    "items",
}
_REQUIRED_TOP_LEVEL_KEYS = {"uuid", "dialogues", "questions"}


@dataclass(frozen=True)
class CloudSafeValidationReport:
    checked_users: int
    issue_count: int
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.issue_count == 0


def _iter_mappings(value: object, path: str = "$") -> Iterable[tuple[str, Mapping]]:
    if isinstance(value, Mapping):
        yield path, value
        for key, child in value.items():
            yield from _iter_mappings(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_mappings(child, f"{path}[{index}]")


def _has_raw_metadata(record: Mapping) -> list[str]:
    issues = []
    for path, mapping in _iter_mappings(record):
        for key in mapping:
            if key in _RAW_METADATA_KEYS:
                issues.append(f"{path}.{key} contains raw/debug metadata")
    return issues


def validate_cloud_safe_dataset(path: Path, user_limit: int | None = None) -> CloudSafeValidationReport:
    issues: list[str] = []
    checked_users = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if user_limit is not None and checked_users >= user_limit:
                break
            if not line.strip():
                continue
            checked_users += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            missing = sorted(_REQUIRED_TOP_LEVEL_KEYS - set(record))
            if missing:
                issues.append(f"line {line_number}: missing required keys {missing}")
            uuid = str(record.get("uuid", ""))
            if not uuid.startswith("User-"):
                issues.append(
                    f"line {line_number}: uuid is not an anonymized User-* alias"
                )
            metadata = record.get("metadata", {})
            if isinstance(metadata, Mapping):
                user_name = str(metadata.get("user_name", ""))
                if user_name and not user_name.startswith("User-"):
                    issues.append(
                        f"line {line_number}: metadata.user_name is not anonymized"
                    )
            else:
                issues.append(f"line {line_number}: metadata is not an object")
            for issue in _has_raw_metadata(record):
                issues.append(f"line {line_number}: {issue}")
            for message_index, message in enumerate(record.get("dialogues", []) or []):
                if not isinstance(message, Mapping):
                    issues.append(
                        f"line {line_number}: dialogues[{message_index}] is not an object"
                    )
                    continue
                if "content" not in message:
                    issues.append(
                        f"line {line_number}: dialogues[{message_index}] missing content"
                    )
            for question_index, question in enumerate(record.get("questions", []) or []):
                if not isinstance(question, Mapping):
                    issues.append(
                        f"line {line_number}: questions[{question_index}] is not an object"
                    )
    if checked_users == 0:
        issues.append("dataset contains no users")
    return CloudSafeValidationReport(
        checked_users=checked_users,
        issue_count=len(issues),
        issues=tuple(issues),
    )


def enforce_no_mask_input_safety(
    input_path: str | Path,
    is_mask: bool,
    allow_unsafe_no_mask: bool = False,
    user_limit: int | None = None,
) -> CloudSafeValidationReport | None:
    """Prevent accidental raw benchmark uploads when running memory systems unmasked."""
    if is_mask or allow_unsafe_no_mask:
        return None
    report = validate_cloud_safe_dataset(Path(input_path), user_limit=user_limit)
    if report.passed:
        return report
    preview = "\n".join(f"- {issue}" for issue in report.issues[:10])
    raise ValueError(
        "Refusing to run with --no-mask because the input does not look like a "
        "cloud-safe public-memory dataset. Generate one with "
        "evaluation.eval_public_memory --cloud-safe-dataset-output, or pass "
        "--allow-unsafe-no-mask for trusted local debugging only.\n"
        f"Issues:\n{preview}"
    )
