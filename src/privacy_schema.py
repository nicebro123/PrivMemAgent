from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PrivacyLevel = Literal["PL1", "PL2", "PL3", "PL4"]
AbstractionLevel = Literal["L0", "L1", "L2", "L3", "L4", "L5"]
RetentionMode = Literal["local_only", "session_only", "no_retention"]

VALID_PRIVACY_LEVELS = {"PL1", "PL2", "PL3", "PL4"}
VALID_ABSTRACTION_LEVELS = {"L0", "L1", "L2", "L3", "L4", "L5"}
VALID_RETENTION_MODES = {"local_only", "session_only", "no_retention"}

STRICT_RECONSTRUCTION_TYPES = {
    "Verification Code",
    "Password",
    "API Key",
    "Recovery Code",
    "Government ID",
    "Financial Account",
    "Email",
    "Phone Number",
    "Detailed Address",
    "Internal IP",
}


class PolicyValidationError(ValueError):
    """Raised when a candidate violates the configured privacy policy."""


@dataclass
class PrivacyItem:
    original_text: str
    privacy_type: str
    privacy_level: PrivacyLevel

    def __post_init__(self) -> None:
        validate_privacy_item(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PrivacyItem:
        return cls(
            original_text=str(data.get("original_text", "")),
            privacy_type=str(data.get("privacy_type", "")),
            privacy_level=str(data.get("privacy_level", "")),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrivateResidue:
    raw: str
    privacy_type: str
    privacy_level: PrivacyLevel
    retention: RetentionMode = "local_only"

    def __post_init__(self) -> None:
        validate_private_residue(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PrivateResidue:
        return cls(
            raw=str(data.get("raw", "")),
            privacy_type=str(data.get("privacy_type", "")),
            privacy_level=str(data.get("privacy_level", "")),  # type: ignore[arg-type]
            retention=str(data.get("retention", "local_only")),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AbstractionTrace:
    raw: str
    public_abstraction: str
    reason: str

    def __post_init__(self) -> None:
        validate_abstraction_trace(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AbstractionTrace:
        return cls(
            raw=str(data.get("raw", "")),
            public_abstraction=str(data.get("public_abstraction", "")),
            reason=str(data.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AbstractionCandidate:
    candidate_id: str
    source_id: str
    level: AbstractionLevel
    public_memory: str
    private_residue: list[PrivateResidue] = field(default_factory=list)
    abstraction_trace: list[AbstractionTrace] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_abstraction_candidate(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AbstractionCandidate:
        return cls(
            candidate_id=str(data.get("candidate_id", "")),
            source_id=str(data.get("source_id", "")),
            level=str(data.get("level", "")),  # type: ignore[arg-type]
            public_memory=str(data.get("public_memory", "")),
            private_residue=[
                PrivateResidue.from_dict(item)
                for item in data.get("private_residue", [])
            ],
            abstraction_trace=[
                AbstractionTrace.from_dict(item)
                for item in data.get("abstraction_trace", [])
            ],
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "level": self.level,
            "public_memory": self.public_memory,
            "private_residue": [item.to_dict() for item in self.private_residue],
            "abstraction_trace": [item.to_dict() for item in self.abstraction_trace],
            "metadata": self.metadata,
        }


@dataclass
class AttackResult:
    attack_type: str
    original_text: str
    privacy_type: str
    privacy_level: PrivacyLevel
    guess: str
    success: bool
    match_score: float
    rationale: str = ""
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.privacy_level not in VALID_PRIVACY_LEVELS:
            raise ValueError(f"invalid attack privacy_level: {self.privacy_level}")
        if not 0.0 <= self.match_score <= 1.0:
            raise ValueError("attack match_score must be between 0 and 1")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("attack confidence must be between 0 and 1")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttackResult:
        return cls(
            attack_type=str(data.get("attack_type", "")),
            original_text=str(data.get("original_text", "")),
            privacy_type=str(data.get("privacy_type", "")),
            privacy_level=str(data.get("privacy_level", "")),  # type: ignore[arg-type]
            guess=str(data.get("guess", "")),
            success=bool(data.get("success", False)),
            match_score=float(data.get("match_score", 0.0)),
            rationale=str(data.get("rationale", "")),
            confidence=(
                float(data["confidence"])
                if data.get("confidence") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UtilityRecord:
    question: str
    reference_answer: str
    predicted_answer: str
    score: float
    valid: bool
    question_type: str = "unknown"

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("utility record score must be between 0 and 1")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UtilityRecord:
        return cls(
            question=str(data.get("question", "")),
            reference_answer=str(data.get("reference_answer", "")),
            predicted_answer=str(data.get("predicted_answer", "")),
            score=float(data.get("score", 0.0)),
            valid=bool(data.get("valid", False)),
            question_type=str(data.get("question_type", "unknown")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UtilityScore:
    mcq_accuracy: float
    answer_consistency: float | None = None
    num_questions: int = 0
    num_valid: int = 0
    is_proxy: bool = False
    records: list[UtilityRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        for value, label in (
            (self.mcq_accuracy, "mcq_accuracy"),
            (self.answer_consistency, "answer_consistency"),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1")
        if self.num_questions < 0 or self.num_valid < 0:
            raise ValueError("question counts cannot be negative")
        if self.num_valid > self.num_questions:
            raise ValueError("num_valid cannot exceed num_questions")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UtilityScore:
        values = dict(data)
        values["records"] = [
            UtilityRecord.from_dict(item) for item in data.get("records", [])
        ]
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "records": [record.to_dict() for record in self.records],
        }


@dataclass
class PrivacyScore:
    exact_reconstruction_rate: float
    attribute_inference_rate: float | None = None
    semantic_leakage_score: float | None = None
    per_type: dict[str, dict[str, float | int | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, label in (
            (self.exact_reconstruction_rate, "exact_reconstruction_rate"),
            (self.attribute_inference_rate, "attribute_inference_rate"),
            (self.semantic_leakage_score, "semantic_leakage_score"),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateScore:
    candidate_id: str
    source_id: str
    utility: UtilityScore
    privacy: PrivacyScore
    attacks: list[AttackResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.source_id:
            raise ValueError("candidate score identifiers must be non-empty")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateScore:
        return cls(
            candidate_id=str(data.get("candidate_id", "")),
            source_id=str(data.get("source_id", "")),
            utility=UtilityScore.from_dict(data.get("utility", {})),
            privacy=PrivacyScore(**data.get("privacy", {})),
            attacks=[AttackResult.from_dict(item) for item in data.get("attacks", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "utility": self.utility.to_dict(),
            "privacy": self.privacy.to_dict(),
            "attacks": [item.to_dict() for item in self.attacks],
        }


def stable_id(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def validate_privacy_item(item: PrivacyItem) -> None:
    if not item.original_text.strip():
        raise ValueError("privacy item original_text must be non-empty")
    if not item.privacy_type.strip():
        raise ValueError("privacy item privacy_type must be non-empty")
    if item.privacy_level not in VALID_PRIVACY_LEVELS:
        raise ValueError(f"invalid privacy_level: {item.privacy_level}")


def validate_private_residue(item: PrivateResidue) -> None:
    if not item.raw.strip():
        raise ValueError("private residue raw must be non-empty")
    if not item.privacy_type.strip():
        raise ValueError("private residue privacy_type must be non-empty")
    if item.privacy_level not in VALID_PRIVACY_LEVELS:
        raise ValueError(f"invalid private residue privacy_level: {item.privacy_level}")
    if item.retention not in VALID_RETENTION_MODES:
        raise ValueError(f"invalid retention: {item.retention}")


def validate_abstraction_trace(trace: AbstractionTrace) -> None:
    if not trace.raw.strip():
        raise ValueError("abstraction trace raw must be non-empty")
    if not trace.public_abstraction.strip():
        raise ValueError("abstraction trace public_abstraction must be non-empty")
    if not trace.reason.strip():
        raise ValueError("abstraction trace reason must be non-empty")


def validate_abstraction_candidate(candidate: AbstractionCandidate) -> None:
    if not candidate.candidate_id:
        raise ValueError("candidate_id must be non-empty")
    if not candidate.source_id:
        raise ValueError("source_id must be non-empty")
    if candidate.level not in VALID_ABSTRACTION_LEVELS:
        raise ValueError(f"invalid abstraction level: {candidate.level}")
    if candidate.level != "L5" and not candidate.public_memory.strip():
        raise ValueError("public_memory must be non-empty except for L5")


def contains_raw_value(
    public_memory: str,
    privacy_items: Sequence[PrivacyItem],
) -> bool:
    normalized_public = normalize_text(public_memory)
    return any(
        normalize_text(item.original_text) in normalized_public
        for item in privacy_items
        if normalize_text(item.original_text)
    )


def _item_key(item: PrivacyItem) -> tuple[str, str, str]:
    return (item.original_text, item.privacy_type, item.privacy_level)


def _policy_allowed_levels(
    item: PrivacyItem,
    policy: Mapping[str, Any],
) -> set[str]:
    overrides = policy.get("type_overrides", {})
    configured = overrides.get(item.privacy_type, {}).get(
        "allowed_levels",
        policy.get("allowed_levels", {}).get(item.privacy_level, ["L5"]),
    )
    allowed = {str(level) for level in configured}
    if not allowed or not allowed <= VALID_ABSTRACTION_LEVELS:
        raise PolicyValidationError(
            f"invalid allowed levels for {item.privacy_type}: {sorted(allowed)}"
        )
    return allowed


def _expected_retention(
    item: PrivacyItem,
    policy: Mapping[str, Any],
) -> str:
    return str(
        policy.get("type_overrides", {})
        .get(item.privacy_type, {})
        .get("retention", "local_only")
    )


def validate_candidate_against_policy(
    candidate: AbstractionCandidate,
    privacy_items: Sequence[PrivacyItem],
    policy: Mapping[str, Any],
) -> None:
    """Validate schema, raw-value isolation, residue alignment, and item policy."""

    validate_abstraction_candidate(candidate)
    if not privacy_items:
        if candidate.private_residue or candidate.abstraction_trace:
            raise PolicyValidationError(
                "non-private candidate cannot contain residue or abstraction trace"
            )
        return

    item_by_key = {_item_key(item): item for item in privacy_items}
    raw_values = {item.original_text for item in privacy_items}

    if candidate.level != "L0" and contains_raw_value(
        candidate.public_memory, privacy_items
    ):
        raise PolicyValidationError("public_memory contains a raw private value")

    residues = {
        (item.raw, item.privacy_type, item.privacy_level): item
        for item in candidate.private_residue
    }
    unknown_residues = set(residues) - set(item_by_key)
    if unknown_residues:
        raise PolicyValidationError(
            f"private residue does not align with privacy items: {unknown_residues}"
        )

    effective_levels_raw = candidate.metadata.get("effective_levels", [])
    effective_levels: dict[tuple[str, str], str] = {}
    for record in effective_levels_raw:
        if isinstance(record, Mapping):
            effective_levels[
                (str(record.get("raw", "")), str(record.get("privacy_type", "")))
            ] = str(record.get("level", ""))

    for item in privacy_items:
        expected_retention = _expected_retention(item, policy)
        residue = residues.get(_item_key(item))
        if expected_retention in {"local_only", "session_only"} and residue is None:
            raise PolicyValidationError(
                f"required private residue missing for {item.privacy_type}"
            )
        if residue is not None and residue.retention != expected_retention:
            raise PolicyValidationError(
                f"retention mismatch for {item.privacy_type}: "
                f"{residue.retention} != {expected_retention}"
            )

        effective_level = effective_levels.get(
            (item.original_text, item.privacy_type),
            candidate.level,
        )
        if effective_level not in _policy_allowed_levels(item, policy):
            raise PolicyValidationError(
                f"level {effective_level} is not allowed for {item.privacy_type}"
            )
        if (
            item.privacy_level == "PL4"
            and item.original_text in candidate.public_memory
        ):
            raise PolicyValidationError("raw PL4 value appears in public_memory")

    trace_raw = {trace.raw for trace in candidate.abstraction_trace}
    if candidate.level != "L0" and trace_raw != raw_values:
        missing = raw_values - trace_raw
        unknown = trace_raw - raw_values
        raise PolicyValidationError(
            f"abstraction trace mismatch; missing={missing}, unknown={unknown}"
        )
