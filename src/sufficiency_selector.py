from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class RepresentationCandidate:
    text: str
    specificity: int
    utility_score: float
    leakage_score: float
    token_count: int
    representation_type: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("candidate text must be non-empty")
        if self.specificity < 0:
            raise ValueError("specificity must be non-negative")
        if not 0.0 <= self.utility_score <= 1.0:
            raise ValueError("utility_score must be between 0 and 1")
        if not 0.0 <= self.leakage_score <= 1.0:
            raise ValueError("leakage_score must be between 0 and 1")
        if self.token_count < 1:
            raise ValueError("token_count must be positive")


@dataclass(frozen=True)
class SelectorConfig:
    utility_floor: float = 0.75
    max_leakage: float = 0.35
    max_public_tokens: int = 128

    def __post_init__(self) -> None:
        if not 0.0 <= self.utility_floor <= 1.0:
            raise ValueError("utility_floor must be between 0 and 1")
        if not 0.0 <= self.max_leakage <= 1.0:
            raise ValueError("max_leakage must be between 0 and 1")
        if self.max_public_tokens < 1:
            raise ValueError("max_public_tokens must be positive")


@dataclass(frozen=True)
class SelectionResult:
    selected: Optional[RepresentationCandidate]
    reason: str
    feasible_count: int


class SufficiencySelector:
    """Select the least specific representation that satisfies hard budgets."""

    def __init__(self, config: Optional[SelectorConfig] = None):
        self.config = config or SelectorConfig()

    def select(self, candidates: Iterable[RepresentationCandidate]) -> SelectionResult:
        candidate_list = list(candidates)
        feasible = [
            candidate
            for candidate in candidate_list
            if candidate.utility_score >= self.config.utility_floor
            and candidate.leakage_score <= self.config.max_leakage
            and candidate.token_count <= self.config.max_public_tokens
        ]
        if not feasible:
            return SelectionResult(
                selected=None,
                reason="no representation satisfies utility, leakage, and token budgets",
                feasible_count=0,
            )

        selected = min(
            feasible,
            key=lambda candidate: (
                candidate.specificity,
                candidate.leakage_score,
                candidate.token_count,
                -candidate.utility_score,
                candidate.text,
            ),
        )
        return SelectionResult(
            selected=selected,
            reason="least-specific feasible representation",
            feasible_count=len(feasible),
        )
