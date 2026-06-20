from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

from src.alias_router import ScopedAliasRouter
from src.public_memory_compiler import CompiledMemory


def _tokens(text: str) -> set[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9_]+|[^\x00-\x7F]", text.lower())
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "is",
        "was",
        "it",
        "for",
        "on",
        "what",
        "how",
        "when",
        "where",
        "who",
        "which",
    }
    return {token for token in raw_tokens if token not in stopwords}


@dataclass(frozen=True)
class UtilityProxyReport:
    question_count: int
    questions_with_exact_private_facts: int
    exactly_referenced_private_facts: int
    policy_eligible_private_facts: int
    recoverable_private_facts: int
    oracle_type_local_recoverability: Optional[float]
    local_recoverability_applicable: bool
    non_private_answer_token_recall: Optional[float]
    pl4_local_retention_rate: float
    proxy_only: bool = True


class UtilityProxyAuditor:
    """Pre-flight utility proxy; it is not a replacement for downstream QA."""

    @staticmethod
    def _safe_rate(
        numerator: int,
        denominator: int,
        empty_value: Optional[float],
    ) -> Optional[float]:
        return numerator / denominator if denominator else empty_value

    def audit(
        self,
        records: Sequence[CompiledMemory],
        questions_by_user: Mapping[str, Iterable[Mapping]],
        privacy_items_by_user: Mapping[str, Iterable[Mapping[str, str]]],
        alias_router: ScopedAliasRouter,
    ) -> UtilityProxyReport:
        public_tokens_by_user: Dict[str, set[str]] = {}
        for record in records:
            public_tokens_by_user.setdefault(record.user_id, set()).update(
                _tokens(record.public_text)
            )

        question_count = 0
        questions_with_private = 0
        referenced_private = 0
        policy_eligible_private = 0
        recoverable_private = 0
        answer_token_total = 0
        answer_token_recovered = 0
        total_pl4 = 0
        retained_pl4 = 0

        for user_id, questions in questions_by_user.items():
            selected_items = {}
            for item in privacy_items_by_user.get(user_id, []):
                original = item.get("original_text", "")
                privacy_type = item.get("privacy_type", "")
                privacy_level = item.get("privacy_level", "")
                if not original:
                    continue
                candidate = (original, privacy_type, privacy_level)
                current = selected_items.get(original)
                candidate_rank = (-int(privacy_level[-1]), privacy_type)
                current_rank = (
                    (-int(current[2][-1]), current[1]) if current is not None else None
                )
                if current is None or candidate_rank < current_rank:
                    selected_items[original] = candidate
            unique_items = sorted(
                selected_items.values(),
                key=lambda item: (-len(item[0]), item[0], item[1], item[2]),
            )
            local_cache = {
                privacy_type: alias_router.query_local(user_id, privacy_type)
                for _, privacy_type, _ in unique_items
            }
            for original, privacy_type, privacy_level in unique_items:
                if privacy_level == "PL4":
                    total_pl4 += 1
                    if any(
                        record["original_text"] == original for record in local_cache[privacy_type]
                    ):
                        retained_pl4 += 1

            for question in questions:
                question_count += 1
                answer = str(question.get("answer", ""))
                evidence = str(question.get("evidence", ""))
                evaluation_text = f"{answer}\n{evidence}"
                referenced = [
                    item for item in unique_items if item[0] and item[0] in evaluation_text
                ]
                if referenced:
                    questions_with_private += 1
                referenced_private += len(referenced)
                for original, privacy_type, privacy_level in referenced:
                    if privacy_level == "PL4":
                        continue
                    policy_eligible_private += 1
                    if any(
                        record["original_text"] == original for record in local_cache[privacy_type]
                    ):
                        recoverable_private += 1

                non_private_answer = answer
                for original, _, _ in referenced:
                    non_private_answer = non_private_answer.replace(original, "")
                answer_tokens = _tokens(non_private_answer)
                answer_token_total += len(answer_tokens)
                answer_token_recovered += len(
                    answer_tokens & public_tokens_by_user.get(user_id, set())
                )

        return UtilityProxyReport(
            question_count=question_count,
            questions_with_exact_private_facts=questions_with_private,
            exactly_referenced_private_facts=referenced_private,
            policy_eligible_private_facts=policy_eligible_private,
            recoverable_private_facts=recoverable_private,
            oracle_type_local_recoverability=self._safe_rate(
                recoverable_private,
                policy_eligible_private,
                None,
            ),
            local_recoverability_applicable=bool(policy_eligible_private),
            non_private_answer_token_recall=self._safe_rate(
                answer_token_recovered,
                answer_token_total,
                None,
            ),
            pl4_local_retention_rate=self._safe_rate(
                retained_pl4,
                total_pl4,
                0.0,
            ),
        )

    @staticmethod
    def to_dict(report: UtilityProxyReport) -> dict:
        return asdict(report)
