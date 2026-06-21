from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from src.sufficiency_selector import RepresentationCandidate


def estimate_tokens(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\x00-\x7F]", text)
    return max(1, len(tokens))


def _normalize_type(value: str) -> str:
    return re.sub(r"[_\-]+", " ", value).strip().lower()


def _contains_exact_private_text(candidate: str, privacy_item: Mapping[str, str]) -> bool:
    original = str(privacy_item.get("original_text", "")).strip()
    if not original:
        return False
    return original.casefold() in candidate.casefold()


@dataclass(frozen=True)
class AbstractionInput:
    user_id: str
    message_id: str
    role: str
    message_text: str
    privacy_item: Mapping[str, str]
    neighboring_context: tuple[Mapping[str, str], ...] = ()
    question_hints: tuple[str, ...] = ()
    policy: Mapping[str, object] = field(default_factory=dict)
    exact_required_types: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class AbstractionCandidate:
    text: str
    abstraction_level: int
    representation_type: str
    utility_score: float
    leakage_score: float
    contains_alias: bool = False
    generator_score: float | None = None

    def __post_init__(self) -> None:
        if self.abstraction_level < 0:
            raise ValueError("abstraction_level must be non-negative")
        if not 0.0 <= self.utility_score <= 1.0:
            raise ValueError("utility_score must be between 0 and 1")
        if not 0.0 <= self.leakage_score <= 1.0:
            raise ValueError("leakage_score must be between 0 and 1")
        if self.generator_score is not None and not 0.0 <= self.generator_score <= 1.0:
            raise ValueError("generator_score must be between 0 and 1")

    def to_representation_candidate(self) -> RepresentationCandidate | None:
        if not self.text:
            return None
        return RepresentationCandidate(
            text=self.text,
            specificity=self.abstraction_level,
            utility_score=self.utility_score,
            leakage_score=self.leakage_score,
            token_count=estimate_tokens(self.text),
            representation_type=self.representation_type,
        )


class AbstractionGenerator(Protocol):
    def generate(self, item: AbstractionInput) -> list[AbstractionCandidate]:
        ...


class SafeCandidateFilter:
    """Reject candidates that copy the exact private value into public memory."""

    def __init__(self, *, allow_aliases: bool = True):
        self.allow_aliases = allow_aliases

    def filter(
        self,
        candidates: Iterable[AbstractionCandidate],
        privacy_item: Mapping[str, str],
    ) -> list[AbstractionCandidate]:
        safe: list[AbstractionCandidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            if candidate.contains_alias and not self.allow_aliases:
                continue
            if candidate.text and _contains_exact_private_text(candidate.text, privacy_item):
                continue
            key = (candidate.text.casefold(), candidate.representation_type)
            if key in seen:
                continue
            seen.add(key)
            safe.append(candidate)
        return safe


class RuleBasedAbstractionGenerator:
    """Deterministic fallback generator for learned-module development."""

    CATEGORY_RULES = (
        (
            ("email", "phone", "address", "contact", "zip"),
            "contact information",
        ),
        (
            ("medical", "health", "diagnosis", "prescription", "biometric"),
            "health constraint relevant to future assistance",
        ),
        (
            ("financial", "transaction", "income", "asset", "debt", "loan", "credit"),
            "financial constraint relevant to future assistance",
        ),
        (
            ("job", "employer", "company", "school", "identity background"),
            "professional or educational context",
        ),
        (
            ("location", "itinerary", "trajectory", "accommodation"),
            "location or itinerary constraint",
        ),
        (
            ("account", "username", "network", "device", "ip"),
            "account or device reference",
        ),
        (
            ("relationship", "third-party", "real name", "name"),
            "person relevant to the user",
        ),
    )

    def __init__(self, candidate_filter: SafeCandidateFilter | None = None):
        self.candidate_filter = candidate_filter or SafeCandidateFilter()

    def category_abstraction(self, privacy_type: str) -> str:
        normalized = _normalize_type(privacy_type)
        for keywords, abstraction in self.CATEGORY_RULES:
            if any(keyword in normalized for keyword in keywords):
                return abstraction
        return "private contextual detail"

    def generate(self, item: AbstractionInput) -> list[AbstractionCandidate]:
        privacy_type = str(item.privacy_item["privacy_type"])
        is_required = privacy_type in item.exact_required_types
        category_text = self.category_abstraction(privacy_type)
        candidates = [
            AbstractionCandidate(
                text="",
                abstraction_level=0,
                utility_score=0.0,
                leakage_score=0.0,
                representation_type="drop",
            ),
            AbstractionCandidate(
                text="private detail",
                abstraction_level=1,
                utility_score=0.40 if is_required else 0.68,
                leakage_score=0.05,
                representation_type="generic_abstract",
            ),
            AbstractionCandidate(
                text=category_text,
                abstraction_level=2,
                utility_score=0.92 if is_required else 0.82,
                leakage_score=0.20,
                representation_type="category_abstract",
            ),
            AbstractionCandidate(
                text=f"private {privacy_type.lower()} value",
                abstraction_level=3,
                utility_score=0.96,
                leakage_score=0.45,
                representation_type="typed_abstract",
            ),
        ]
        return self.candidate_filter.filter(candidates, item.privacy_item)


class ArtifactBackedAbstractionGenerator:
    """Load learned or distilled abstraction rules from a JSON artifact.

    The artifact is intentionally simple so early training scripts can emit it
    without a heavy runtime dependency:

    {
      "templates": [
        {
          "privacy_type": "Email",
          "privacy_level": "PL2",
          "text": "contact information",
          "abstraction_level": 2,
          "utility_score": 0.85,
          "leakage_score": 0.1,
          "representation_type": "learned_category"
        }
      ]
    }
    """

    def __init__(
        self,
        artifact_path: str | Path,
        fallback: AbstractionGenerator | None = None,
        candidate_filter: SafeCandidateFilter | None = None,
    ):
        self.artifact_path = Path(artifact_path).expanduser().resolve()
        self.fallback = fallback or RuleBasedAbstractionGenerator()
        self.candidate_filter = candidate_filter or SafeCandidateFilter()
        self.templates = self._load_templates(self.artifact_path)

    @staticmethod
    def _load_templates(path: Path) -> list[dict]:
        data = json.loads(path.read_text(encoding="utf-8"))
        templates = data.get("templates")
        if not isinstance(templates, list):
            raise ValueError("abstraction artifact must contain a templates list")
        return templates

    @staticmethod
    def _matches(template: Mapping[str, object], item: AbstractionInput) -> bool:
        privacy_item = item.privacy_item
        privacy_type = template.get("privacy_type")
        privacy_level = template.get("privacy_level")
        if privacy_type not in {None, "*"} and str(privacy_type) != privacy_item["privacy_type"]:
            return False
        if privacy_level not in {None, "*"} and str(privacy_level) != privacy_item["privacy_level"]:
            return False
        return True

    @staticmethod
    def _candidate_from_template(template: Mapping[str, object]) -> AbstractionCandidate:
        text = str(template.get("text", ""))
        return AbstractionCandidate(
            text=text,
            abstraction_level=int(template.get("abstraction_level", 2)),
            utility_score=float(template.get("utility_score", 0.75)),
            leakage_score=float(template.get("leakage_score", 0.20)),
            representation_type=str(template.get("representation_type", "learned_abstract")),
            contains_alias=bool(template.get("contains_alias", False)),
            generator_score=(
                float(template["generator_score"]) if "generator_score" in template else None
            ),
        )

    def generate(self, item: AbstractionInput) -> list[AbstractionCandidate]:
        learned = [
            self._candidate_from_template(template)
            for template in self.templates
            if self._matches(template, item)
        ]
        fallback = self.fallback.generate(item)
        return self.candidate_filter.filter([*learned, *fallback], item.privacy_item)


class AbstractorAdapter:
    """Adapter exposing the legacy ``candidates`` interface used by the compiler."""

    def __init__(self, generator: AbstractionGenerator):
        self.generator = generator

    def candidates(
        self,
        privacy_item: Mapping[str, str],
        required_types: Iterable[str],
        *,
        user_id: str = "",
        message_id: str = "",
        role: str = "user",
        message_text: str = "",
        neighboring_context: Sequence[Mapping[str, str]] = (),
        policy: Mapping[str, object] | None = None,
    ) -> list[RepresentationCandidate]:
        abstraction_input = AbstractionInput(
            user_id=user_id,
            message_id=message_id,
            role=role,
            message_text=message_text,
            privacy_item=privacy_item,
            neighboring_context=tuple(neighboring_context),
            policy=policy or {},
            exact_required_types=frozenset(required_types),
        )
        return [
            candidate
            for generated in self.generator.generate(abstraction_input)
            if (candidate := generated.to_representation_candidate()) is not None
        ]


def load_abstraction_generator(config: Mapping[str, object] | None) -> AbstractionGenerator:
    config = dict(config or {})
    mode = str(config.get("mode", "rule")).lower()
    if mode in {"rule", "deterministic", "fallback"}:
        return RuleBasedAbstractionGenerator()
    if mode in {"artifact", "learned"}:
        path = config.get("artifact_path")
        if not path:
            raise ValueError("learned abstraction mode requires artifact_path")
        return ArtifactBackedAbstractionGenerator(str(path))
    raise ValueError(f"unsupported abstraction generator mode: {mode}")
