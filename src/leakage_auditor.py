from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from src.public_memory_compiler import CompiledMemory

_ALIAS_PATTERN = re.compile(r"<MPM_[A-Za-z0-9_]+_\d+_[a-f0-9]{12}>")


@dataclass(frozen=True)
class AuditThresholds:
    exact_recovery: float = 0.01
    cross_scope_linkability: float = 0.01
    pl4_public_retention: float = 0.0
    minimum_token_reduction: float = 0.30


@dataclass(frozen=True)
class LeakageReport:
    exact_recovery_rate: float
    cross_scope_linkability_rate: Optional[float]
    cross_scope_linkability_applicable: bool
    pl4_public_retention_rate: float
    corpus_token_reduction: float
    average_token_reduction: float
    source_token_count: int
    public_token_count: int
    exact_recovered_items: int
    total_sensitive_items: int
    linked_scope_pairs: int
    total_scope_pairs: int
    pl4_public_items: int
    total_pl4_items: int
    passed: bool
    failures: tuple[str, ...]


class LeakageAuditor:
    def __init__(self, thresholds: Optional[AuditThresholds] = None):
        self.thresholds = thresholds or AuditThresholds()

    @staticmethod
    def _safe_rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def audit(
        self,
        records: Sequence[CompiledMemory],
        source_items: Mapping[str, Iterable[Mapping[str, str]]],
    ) -> LeakageReport:
        total_sensitive = 0
        exact_recovered = 0
        total_pl4 = 0
        pl4_public = 0

        for record in records:
            source_for_message = list(source_items.get(record.message_id, []))
            public_text = record.public_text
            for item in source_for_message:
                original = item.get("original_text", "")
                if not original:
                    continue
                total_sensitive += 1
                if original in public_text:
                    exact_recovered += 1
                if item.get("privacy_level") == "PL4":
                    total_pl4 += 1
                    if original in public_text:
                        pl4_public += 1

        aliases_by_fingerprint: Dict[str, List[tuple[str, str]]] = {}
        for record in records:
            for item in record.items:
                if item.public_value and _ALIAS_PATTERN.fullmatch(item.public_value):
                    scope = f"{item.alias_scope}:{item.scope_id}"
                    aliases_by_fingerprint.setdefault(item.source_fingerprint, []).append(
                        (scope, item.public_value)
                    )

        total_scope_pairs = 0
        linked_scope_pairs = 0
        for observations in aliases_by_fingerprint.values():
            unique_observations = list(dict.fromkeys(observations))
            for first, second in combinations(unique_observations, 2):
                if first[0] == second[0]:
                    continue
                total_scope_pairs += 1
                if first[1] == second[1]:
                    linked_scope_pairs += 1

        exact_rate = self._safe_rate(exact_recovered, total_sensitive)
        linkability_rate = (
            linked_scope_pairs / total_scope_pairs if total_scope_pairs else None
        )
        pl4_rate = self._safe_rate(pl4_public, total_pl4)
        source_token_count = sum(record.source_tokens for record in records)
        public_token_count = sum(record.public_tokens for record in records)
        corpus_reduction = (
            1.0 - public_token_count / source_token_count if source_token_count else 0.0
        )
        average_reduction = (
            sum(record.token_reduction for record in records) / len(records) if records else 0.0
        )

        failures = []
        if exact_rate > self.thresholds.exact_recovery:
            failures.append("exact_recovery")
        if (
            linkability_rate is not None
            and linkability_rate > self.thresholds.cross_scope_linkability
        ):
            failures.append("cross_scope_linkability")
        if pl4_rate > self.thresholds.pl4_public_retention:
            failures.append("pl4_public_retention")
        if corpus_reduction < self.thresholds.minimum_token_reduction:
            failures.append("token_reduction")

        return LeakageReport(
            exact_recovery_rate=exact_rate,
            cross_scope_linkability_rate=linkability_rate,
            cross_scope_linkability_applicable=bool(total_scope_pairs),
            pl4_public_retention_rate=pl4_rate,
            corpus_token_reduction=corpus_reduction,
            average_token_reduction=average_reduction,
            source_token_count=source_token_count,
            public_token_count=public_token_count,
            exact_recovered_items=exact_recovered,
            total_sensitive_items=total_sensitive,
            linked_scope_pairs=linked_scope_pairs,
            total_scope_pairs=total_scope_pairs,
            pl4_public_items=pl4_public,
            total_pl4_items=total_pl4,
            passed=not failures,
            failures=tuple(failures),
        )
