from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_CRITICAL_TYPE_WORDS = (
    "password",
    "passcode",
    "otp",
    "verification code",
    "recovery code",
    "api key",
    "access token",
    "secret key",
    "private key",
    "credential",
)
_SENSITIVE_TYPE_WORDS = (
    "medical",
    "health",
    "diagnosis",
    "prescription",
    "financial",
    "bank",
    "income",
    "debt",
    "loan",
    "credit",
    "religion",
    "ethnicity",
    "precise location",
    "address",
    "biometric",
    "password",
    "verification code",
    "recovery code",
    "api key",
    "token",
)
_PROMPT_INJECTION_RE = re.compile(
    r"\b(?:ignore (?:all )?(?:previous|prior) instructions|"
    r"reveal (?:the )?(?:system|developer) prompt|system prompt|developer message|"
    r"exfiltrate|bypass (?:the )?(?:policy|safety)|send .*secret|dump .*memory)\b",
    flags=re.IGNORECASE,
)
_ALIAS_RE = re.compile(r"<MPM_[A-Za-z0-9_]+_\d+_[a-f0-9]{12}>")
_LINKAGE_FIELD_PARTS = (
    "source_fingerprint",
    "source_message_id",
    "scope_id",
    "provenance_id",
)

_SEMANTIC_LOCATION_PARTS = (
    "public_text",
    "content",
    "question",
    "answer",
    "evidence",
    "all_options",
)


def _is_semantic_location(location: str) -> bool:
    return any(part in location for part in _SEMANTIC_LOCATION_PARTS)


def _has_text_boundary(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not before.isalnum() and not after.isalnum()


def _contains_exact_value(text: str, value: str, location: str) -> bool:
    if not value or not _is_semantic_location(location):
        return False
    start = text.find(value)
    while start >= 0:
        end = start + len(value)
        if len(value) >= 4 or _has_text_boundary(text, start, end):
            return True
        start = text.find(value, start + 1)
    return False


@dataclass(frozen=True)
class Exposure:
    artifact: str
    location: str
    exposure_type: str
    value: str
    privacy_level: str | None = None
    privacy_type: str | None = None
    severity: str = "failure"


@dataclass(frozen=True)
class AdversarialAuditReport:
    artifact_count: int
    source_secret_count: int
    exact_exposure_count: int
    pl4_exposure_count: int
    user_identifier_exposure_count: int
    attribute_exposure_count: int
    linkage_exposure_count: int
    membership_exposure_count: int
    canary_exposure_count: int
    prompt_injection_risk_count: int
    warning_count: int
    failure_count: int
    exposures: tuple[Exposure, ...]
    passed: bool


def _iter_strings(value: object, prefix: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_strings(child, f"{prefix}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for index, child in enumerate(value):
            yield from _iter_strings(child, f"{prefix}[{index}]")


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_line_number"] = line_number
            records.append(record)
    return records


def _normalize_type_term(value: str) -> str:
    return re.sub(r"[_\-]+", " ", value).strip().lower()


def _type_terms(privacy_type: str, privacy_level: str) -> set[str]:
    normalized = _normalize_type_term(privacy_type)
    terms = {normalized} if len(normalized) >= 4 else set()
    if privacy_level in {"PL3", "PL4"}:
        terms.add(privacy_level.lower())
    for term in _SENSITIVE_TYPE_WORDS:
        if term in normalized or (privacy_level in {"PL3", "PL4"} and term == normalized):
            terms.add(term)
    for term in _CRITICAL_TYPE_WORDS:
        if term in normalized:
            terms.add(term)
    return {term for term in terms if len(term) >= 4 or re.fullmatch(r"pl\d", term)}


def _is_canary_like(value: str, privacy_type: str) -> bool:
    normalized_type = _normalize_type_term(privacy_type)
    if any(term in normalized_type for term in _CRITICAL_TYPE_WORDS):
        return True
    patterns = (
        r"CANARY[-_:]?[A-Za-z0-9]+",
        r"RC[-_ ]?\d{3,}",
        r"(?:sk|pk|api)[-_][A-Za-z0-9]{12,}",
        r"\b\d{6,8}\b",
    )
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def _collect_source_signals(dataset_records: Iterable[Mapping]) -> dict[str, object]:
    secrets_by_key = {}
    user_identifiers = set()
    membership_markers: dict[str, bool] = {}
    attribute_terms_by_key: dict[tuple[str, str], set[str]] = {}
    canary_values = set()
    for user in dataset_records:
        user_id = str(user.get("uuid", "")).strip()
        if user_id:
            user_identifiers.add(user_id)
        metadata = user.get("metadata", {})
        if isinstance(metadata, Mapping):
            user_name = str(metadata.get("user_name", "")).strip()
            if user_name:
                user_identifiers.add(user_name)
        for message in user.get("dialogues", []):
            content = str(message.get("content", "")).strip()
            privacy_info = list(message.get("privacy_info", []))
            has_sensitive_marker = any(
                str(item.get("original_text", "")).strip()
                and (
                    str(item.get("original_text", "")).strip() in content
                    or str(item.get("privacy_level", "")) in {"PL3", "PL4"}
                )
                for item in privacy_info
            )
            if len(content) >= 24:
                membership_markers[content] = has_sensitive_marker
            for item in privacy_info:
                original = str(item.get("original_text", "")).strip()
                if not original:
                    continue
                privacy_level = str(item.get("privacy_level", ""))
                privacy_type = str(item.get("privacy_type", ""))
                key = (original, privacy_level, privacy_type)
                secrets_by_key[key] = {
                    "original_text": original,
                    "privacy_level": privacy_level,
                    "privacy_type": privacy_type,
                }
                if privacy_level in {"PL3", "PL4"}:
                    attribute_terms_by_key.setdefault(
                        (privacy_level, privacy_type),
                        set(),
                    ).update(_type_terms(privacy_type, privacy_level))
                if _is_canary_like(original, privacy_type):
                    canary_values.add(original)
    return {
        "secrets": list(secrets_by_key.values()),
        "user_identifiers": user_identifiers,
        "membership_markers": membership_markers,
        "attribute_terms_by_key": attribute_terms_by_key,
        "canary_values": canary_values,
    }


def _contains_case_insensitive(text: str, value: str) -> bool:
    return value.casefold() in text.casefold()


def _record_identity(record: Mapping) -> str:
    return str(record.get("user_id") or record.get("uuid") or "").strip()


def _signals_for_artifact_record(
    record: Mapping,
    artifact: Path,
    source_signals_by_index: Sequence[dict[str, object]],
    global_signals: dict[str, object],
    identity_orders: dict[str, list[str]],
    identity_to_source_index: dict[tuple[str, str], int],
) -> dict[str, object]:
    identity = _record_identity(record)
    if not identity:
        return global_signals

    artifact_key = str(artifact)
    key = (artifact_key, identity)
    if key not in identity_to_source_index:
        ordered = identity_orders.setdefault(artifact_key, [])
        if identity not in ordered:
            ordered.append(identity)
        identity_to_source_index[key] = ordered.index(identity)

    source_index = identity_to_source_index[key]
    if source_index < len(source_signals_by_index):
        return source_signals_by_index[source_index]
    return global_signals


def _add_exposure(
    exposures: list[Exposure],
    artifact: Path,
    line: int,
    location: str,
    exposure_type: str,
    value: str,
    privacy_level: str | None = None,
    privacy_type: str | None = None,
    severity: str = "failure",
) -> None:
    exposures.append(
        Exposure(
            artifact=str(artifact),
            location=f"line:{line}:{location}",
            exposure_type=exposure_type,
            value=value,
            privacy_level=privacy_level,
            privacy_type=privacy_type,
            severity=severity,
        )
    )


def audit_artifacts(
    source_dataset: Path,
    artifacts: Sequence[Path],
    source_user_limit: int | None = None,
) -> AdversarialAuditReport:
    source_records = _load_jsonl(source_dataset)
    if source_user_limit is not None:
        if source_user_limit < 0:
            raise ValueError("source_user_limit must be non-negative")
        source_records = source_records[:source_user_limit]
    global_signals = _collect_source_signals(source_records)
    source_signals_by_index = [
        _collect_source_signals([record]) for record in source_records
    ]
    exposures: list[Exposure] = []
    alias_locations: dict[str, set[str]] = {}
    identity_orders: dict[str, list[str]] = {}
    identity_to_source_index: dict[tuple[str, str], int] = {}

    for artifact in artifacts:
        for record in _load_jsonl(artifact):
            line = record.pop("_line_number")
            signals = _signals_for_artifact_record(
                record,
                artifact,
                source_signals_by_index,
                global_signals,
                identity_orders,
                identity_to_source_index,
            )
            secrets = signals["secrets"]
            user_identifiers = signals["user_identifiers"]
            membership_markers = signals["membership_markers"]
            attribute_terms_by_key = signals["attribute_terms_by_key"]
            canary_values = signals["canary_values"]
            for location, text in _iter_strings(record):
                for secret in secrets:
                    original = secret["original_text"]
                    if _contains_exact_value(text, original, location):
                        _add_exposure(
                            exposures,
                            artifact,
                            line,
                            location,
                            "exact_secret",
                            original,
                            privacy_level=secret["privacy_level"],
                            privacy_type=secret["privacy_type"],
                        )
                for identifier in user_identifiers:
                    if identifier and identifier in text:
                        _add_exposure(
                            exposures,
                            artifact,
                            line,
                            location,
                            "user_identifier",
                            identifier,
                        )
                for marker, has_sensitive_marker in membership_markers.items():
                    if marker and marker == text.strip():
                        _add_exposure(
                            exposures,
                            artifact,
                            line,
                            location,
                            "membership_marker",
                            marker[:160],
                            severity=("failure" if has_sensitive_marker else "warning"),
                        )
                for canary in canary_values:
                    if _contains_exact_value(text, canary, location):
                        _add_exposure(
                            exposures,
                            artifact,
                            line,
                            location,
                            "canary_exposure",
                            canary,
                        )
                for (privacy_level, privacy_type), terms in attribute_terms_by_key.items():
                    for term in terms:
                        if not _contains_case_insensitive(text, term):
                            continue
                        severity = (
                            "failure"
                            if location.lower().endswith((".privacy_type", ".privacy_level"))
                            else "warning"
                        )
                        _add_exposure(
                            exposures,
                            artifact,
                            line,
                            location,
                            "sensitive_attribute",
                            term,
                            privacy_level=privacy_level,
                            privacy_type=privacy_type,
                            severity=severity,
                        )
                if _PROMPT_INJECTION_RE.search(text):
                    _add_exposure(
                        exposures,
                        artifact,
                        line,
                        location,
                        "prompt_injection_risk",
                        text[:160],
                    )
                if any(part in location for part in _LINKAGE_FIELD_PARTS) and text.strip():
                    _add_exposure(
                        exposures,
                        artifact,
                        line,
                        location,
                        "linkage_metadata",
                        text[:96],
                    )
                for alias in _ALIAS_RE.findall(text):
                    alias_locations.setdefault(alias, set()).add(f"{artifact}:{line}:{location}")

    for alias, locations in alias_locations.items():
        if len(locations) > 1:
            _add_exposure(
                exposures,
                Path("<cross-artifact>"),
                0,
                ",".join(sorted(locations)[:3]),
                "cross_record_alias_reuse",
                alias,
                severity="warning",
            )

    exact_exposures = [item for item in exposures if item.exposure_type == "exact_secret"]
    pl4_exposures = [item for item in exact_exposures if item.privacy_level == "PL4"]
    user_identifier_exposures = [
        item for item in exposures if item.exposure_type == "user_identifier"
    ]
    attribute_exposures = [
        item for item in exposures if item.exposure_type == "sensitive_attribute"
    ]
    linkage_exposures = [
        item
        for item in exposures
        if item.exposure_type in {"linkage_metadata", "cross_record_alias_reuse"}
    ]
    membership_exposures = [
        item for item in exposures if item.exposure_type == "membership_marker"
    ]
    canary_exposures = [item for item in exposures if item.exposure_type == "canary_exposure"]
    prompt_injection_exposures = [
        item for item in exposures if item.exposure_type == "prompt_injection_risk"
    ]
    warning_count = sum(item.severity == "warning" for item in exposures)
    failure_count = sum(item.severity == "failure" for item in exposures)
    return AdversarialAuditReport(
        artifact_count=len(artifacts),
        source_secret_count=len(global_signals["secrets"]),
        exact_exposure_count=len(exact_exposures),
        pl4_exposure_count=len(pl4_exposures),
        user_identifier_exposure_count=len(user_identifier_exposures),
        attribute_exposure_count=len(attribute_exposures),
        linkage_exposure_count=len(linkage_exposures),
        membership_exposure_count=len(membership_exposures),
        canary_exposure_count=len(canary_exposures),
        prompt_injection_risk_count=len(prompt_injection_exposures),
        warning_count=warning_count,
        failure_count=failure_count,
        exposures=tuple(exposures),
        passed=failure_count == 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic adversarial leakage audit against public/cloud-safe "
            "artifacts. This checks exact recovery, source identifiers, sensitive "
            "metadata, linkability markers, canaries, membership markers, and "
            "prompt-injection strings in exported JSONL."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="source benchmark JSONL")
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        required=True,
        help="public/cloud-safe JSONL artifact to audit; repeat for multiple files",
    )
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--source-user-limit",
        type=int,
        help="audit only the first N source users; align this with eval_public_memory --user-limit",
    )
    args = parser.parse_args()

    report = audit_artifacts(
        source_dataset=args.source.expanduser().resolve(),
        artifacts=[path.expanduser().resolve() for path in args.artifact],
        source_user_limit=args.source_user_limit,
    )
    payload = asdict(report)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
