from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from src.sufficiency_selector import (
    RepresentationCandidate,
    SelectionResult,
    SelectorConfig,
)


@dataclass(frozen=True)
class SelectorFeatures:
    candidate_text: str
    representation_type: str
    abstraction_level: int
    privacy_type: str
    privacy_level: str
    token_count: int
    utility_score: float
    leakage_score: float
    embedding_relevance: float = 0.0
    lexical_answer_overlap: float = 0.0
    exact_leak_flag: bool = False
    attribute_risk_score: float = 0.0
    linkability_scope: str | None = None

    @classmethod
    def from_candidate(
        cls,
        candidate: RepresentationCandidate,
        *,
        privacy_item: Mapping[str, str] | None = None,
        exact_leak_flag: bool = False,
        embedding_relevance: float = 0.0,
        lexical_answer_overlap: float = 0.0,
        attribute_risk_score: float | None = None,
        linkability_scope: str | None = None,
    ) -> "SelectorFeatures":
        privacy_item = privacy_item or {}
        privacy_level = str(privacy_item.get("privacy_level", ""))
        privacy_type = str(privacy_item.get("privacy_type", ""))
        if attribute_risk_score is None:
            attribute_risk_score = 0.4 if privacy_level in {"PL3", "PL4"} else 0.1
        return cls(
            candidate_text=candidate.text,
            representation_type=candidate.representation_type,
            abstraction_level=candidate.specificity,
            privacy_type=privacy_type,
            privacy_level=privacy_level,
            token_count=candidate.token_count,
            utility_score=candidate.utility_score,
            leakage_score=candidate.leakage_score,
            embedding_relevance=embedding_relevance,
            lexical_answer_overlap=lexical_answer_overlap,
            exact_leak_flag=exact_leak_flag,
            attribute_risk_score=attribute_risk_score,
            linkability_scope=linkability_scope,
        )


@dataclass(frozen=True)
class SelectorDecision:
    action: str
    score: float
    reason: str
    candidate: RepresentationCandidate | None = None
    feasible_count: int = 0


class CandidateRanker(Protocol):
    def score(self, features: SelectorFeatures) -> float:
        ...


class LinearUtilityLeakageRanker:
    """Small learned-artifact ranker.

    Training code can fit these weights and write them as JSON. The default
    weights intentionally mirror the method objective:

    utility - lambda * leakage - beta * tokens - gamma * linkability.
    """

    DEFAULT_WEIGHTS = {
        "bias": 0.0,
        "utility_score": 1.0,
        "leakage_score": -1.0,
        "token_count": -0.01,
        "abstraction_level": -0.05,
        "embedding_relevance": 0.25,
        "lexical_answer_overlap": 0.25,
        "attribute_risk_score": -0.30,
        "exact_leak_flag": -100.0,
        "pl4_flag": -100.0,
        "task_scope_flag": -0.05,
        "session_scope_flag": -0.15,
        "persistent_scope_flag": -0.35,
    }

    def __init__(self, weights: Mapping[str, float] | None = None):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self.weights.update({key: float(value) for key, value in (weights or {}).items()})

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearUtilityLeakageRanker":
        data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        weights = data.get("weights")
        if not isinstance(weights, Mapping):
            raise ValueError("selector artifact must contain a weights mapping")
        return cls({str(key): float(value) for key, value in weights.items()})

    def score(self, features: SelectorFeatures) -> float:
        scope = features.linkability_scope or ""
        values = {
            "bias": 1.0,
            "utility_score": features.utility_score,
            "leakage_score": features.leakage_score,
            "token_count": math.log1p(features.token_count),
            "abstraction_level": features.abstraction_level,
            "embedding_relevance": features.embedding_relevance,
            "lexical_answer_overlap": features.lexical_answer_overlap,
            "attribute_risk_score": features.attribute_risk_score,
            "exact_leak_flag": 1.0 if features.exact_leak_flag else 0.0,
            "pl4_flag": 1.0 if features.privacy_level == "PL4" else 0.0,
            "task_scope_flag": 1.0 if scope == "task" else 0.0,
            "session_scope_flag": 1.0 if scope == "session" else 0.0,
            "persistent_scope_flag": 1.0 if scope == "persistent" else 0.0,
        }
        return sum(self.weights.get(name, 0.0) * value for name, value in values.items())


class LearnedUtilityLeakageSelector:
    """Budget-aware selector backed by learned ranking weights."""

    def __init__(
        self,
        config: SelectorConfig | None = None,
        ranker: CandidateRanker | None = None,
        *,
        score_threshold: float = float("-inf"),
        fail_closed_on_exact_leak: bool = True,
        fail_closed_on_pl4: bool = True,
    ):
        self.config = config or SelectorConfig()
        self.ranker = ranker or LinearUtilityLeakageRanker()
        self.score_threshold = score_threshold
        self.fail_closed_on_exact_leak = fail_closed_on_exact_leak
        self.fail_closed_on_pl4 = fail_closed_on_pl4

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        config: SelectorConfig | None = None,
        *,
        score_threshold: float = float("-inf"),
    ) -> "LearnedUtilityLeakageSelector":
        return cls(
            config=config,
            ranker=LinearUtilityLeakageRanker.from_json(artifact_path),
            score_threshold=score_threshold,
        )

    def _candidate_features(
        self,
        candidate: RepresentationCandidate,
        privacy_item: Mapping[str, str] | None,
    ) -> SelectorFeatures:
        exact_leak = False
        if privacy_item:
            original = str(privacy_item.get("original_text", "")).strip()
            exact_leak = bool(original and original.casefold() in candidate.text.casefold())
        return SelectorFeatures.from_candidate(
            candidate,
            privacy_item=privacy_item,
            exact_leak_flag=exact_leak,
        )

    def select(
        self,
        candidates: list[RepresentationCandidate],
        *,
        privacy_item: Mapping[str, str] | None = None,
        features: list[SelectorFeatures] | None = None,
    ) -> SelectionResult:
        decision = self.decide(
            candidates,
            privacy_item=privacy_item,
            features=features,
        )
        return SelectionResult(
            selected=decision.candidate,
            reason=decision.reason,
            feasible_count=decision.feasible_count,
        )

    def decide(
        self,
        candidates: list[RepresentationCandidate],
        *,
        privacy_item: Mapping[str, str] | None = None,
        features: list[SelectorFeatures] | None = None,
    ) -> SelectorDecision:
        if features is not None and len(features) != len(candidates):
            raise ValueError("features and candidates must have the same length")
        candidate_features = features or [
            self._candidate_features(candidate, privacy_item) for candidate in candidates
        ]
        feasible: list[tuple[float, RepresentationCandidate]] = []
        rejected_reasons = []
        for candidate, item_features in zip(candidates, candidate_features, strict=True):
            if self.fail_closed_on_exact_leak and item_features.exact_leak_flag:
                rejected_reasons.append("exact leak")
                continue
            if self.fail_closed_on_pl4 and item_features.privacy_level == "PL4":
                rejected_reasons.append("PL4 public candidate")
                continue
            if candidate.utility_score < self.config.utility_floor:
                rejected_reasons.append("below utility floor")
                continue
            if candidate.leakage_score > self.config.max_leakage:
                rejected_reasons.append("above leakage budget")
                continue
            if candidate.token_count > self.config.max_public_tokens:
                rejected_reasons.append("above token budget")
                continue
            score = self.ranker.score(item_features)
            if score < self.score_threshold:
                rejected_reasons.append("below learned score threshold")
                continue
            feasible.append((score, candidate))

        if not feasible:
            reason = "no learned-selector candidate satisfies hard budgets"
            if rejected_reasons:
                reason += f" ({', '.join(sorted(set(rejected_reasons)))})"
            return SelectorDecision(
                action="drop",
                score=float("-inf"),
                reason=reason,
                candidate=None,
                feasible_count=0,
            )

        score, candidate = max(
            feasible,
            key=lambda item: (
                item[0],
                -item[1].specificity,
                -item[1].leakage_score,
                -item[1].token_count,
                item[1].text,
            ),
        )
        return SelectorDecision(
            action="select",
            score=score,
            reason="highest learned utility-leakage score within hard budgets",
            candidate=candidate,
            feasible_count=len(feasible),
        )


def load_utility_leakage_selector(
    config: Mapping[str, object] | None,
    selector_config: SelectorConfig | None = None,
) -> LearnedUtilityLeakageSelector | None:
    config = dict(config or {})
    mode = str(config.get("mode", "disabled")).lower()
    if mode in {"disabled", "none", "rule"}:
        return None
    if mode in {"linear", "artifact", "learned"}:
        artifact_path = config.get("artifact_path")
        ranker = (
            LinearUtilityLeakageRanker.from_json(str(artifact_path))
            if artifact_path
            else LinearUtilityLeakageRanker()
        )
        return LearnedUtilityLeakageSelector(
            config=selector_config,
            ranker=ranker,
            score_threshold=float(config.get("score_threshold", float("-inf"))),
            fail_closed_on_exact_leak=bool(config.get("fail_closed_on_exact_leak", True)),
            fail_closed_on_pl4=bool(config.get("fail_closed_on_pl4", True)),
        )
    raise ValueError(f"unsupported utility-leakage selector mode: {mode}")
