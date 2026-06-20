from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from .privacy_auditor import AdversarialMemoryAuditor, aggregate_attacks_by_type
from .privacy_schema import (
    AbstractionCandidate,
    AttackResult,
    CandidateScore,
    PrivacyScore,
    UtilityRecord,
    UtilityScore,
)


class UtilityEvaluator(Protocol):
    def evaluate(
        self,
        public_memory: str,
        questions: Sequence[Mapping[str, Any]],
    ) -> UtilityScore: ...


def normalize_mcq_answer(value: str) -> str | None:
    text = value.strip()
    match = re.search(r"(?:^|[\s(<{\[])([A-Da-d])(?:$|[\s)>}\].,:])", text)
    if match:
        return match.group(1).lower()
    if len(text) == 1 and text.lower() in {"a", "b", "c", "d"}:
        return text.lower()
    return None


def verify_mcq_answer(prediction: str, reference: str) -> tuple[bool, bool]:
    predicted = normalize_mcq_answer(prediction)
    expected = normalize_mcq_answer(reference)
    if predicted is None or expected is None:
        return False, False
    return predicted == expected, True


class QuestionAnswerUtilityEvaluator:
    """Measures utility from actual QA outputs rather than abstraction-level priors."""

    def __init__(
        self,
        answer_fn: Callable[[str, Mapping[str, Any]], str],
        judge_fn: (
            Callable[[Mapping[str, Any], str, str], float | tuple[float, bool]] | None
        ) = None,
    ):
        self.answer_fn = answer_fn
        self.judge_fn = judge_fn

    def evaluate(
        self,
        public_memory: str,
        questions: Sequence[Mapping[str, Any]],
    ) -> UtilityScore:
        records: list[UtilityRecord] = []
        for question in questions:
            prompt = str(question.get("question", ""))
            reference = str(question.get("answer", ""))
            if not prompt or not reference:
                records.append(
                    UtilityRecord(
                        question=prompt,
                        reference_answer=reference,
                        predicted_answer="",
                        score=0.0,
                        valid=False,
                    )
                )
                continue
            prediction = str(self.answer_fn(public_memory, question))
            if question.get("all_options"):
                correct, valid = verify_mcq_answer(prediction, reference)
                score = 1.0 if correct else 0.0
                question_type = "mcq"
            elif self.judge_fn is not None:
                judged = self.judge_fn(question, prediction, reference)
                if isinstance(judged, tuple):
                    score, valid = judged
                else:
                    score, valid = judged, True
                score = max(0.0, min(1.0, float(score)))
                question_type = str(question.get("question_type", "open_qa"))
            else:
                valid = bool(prediction.strip())
                score = (
                    1.0
                    if prediction.strip().casefold() == reference.strip().casefold()
                    else 0.0
                )
                question_type = str(question.get("question_type", "exact_qa"))
            records.append(
                UtilityRecord(
                    question=prompt,
                    reference_answer=reference,
                    predicted_answer=prediction,
                    score=score,
                    valid=valid,
                    question_type=question_type,
                )
            )
        valid_records = [record for record in records if record.valid]
        total_score = sum(record.score for record in records)
        accuracy = total_score / len(records) if records else 0.0
        consistency = (
            sum(record.score for record in valid_records) / len(valid_records)
            if valid_records
            else 0.0
        )
        return UtilityScore(
            mcq_accuracy=accuracy,
            answer_consistency=consistency,
            num_questions=len(records),
            num_valid=len(valid_records),
            is_proxy=False,
            records=records,
        )


class ProxyUtilityEvaluator:
    """Explicit smoke-test-only utility proxy; never valid as paper evidence."""

    LEVEL_SCORES = {
        "L0": 1.0,
        "L1": 0.92,
        "L2": 0.88,
        "L3": 0.82,
        "L4": 0.60,
        "L5": 0.35,
    }

    def evaluate_candidate(
        self,
        candidate: AbstractionCandidate,
        questions: Sequence[Mapping[str, Any]],
    ) -> UtilityScore:
        score = self.LEVEL_SCORES[candidate.level]
        return UtilityScore(
            mcq_accuracy=score,
            answer_consistency=score,
            num_questions=len(questions),
            num_valid=len(questions),
            is_proxy=True,
        )

    def evaluate(
        self,
        public_memory: str,
        questions: Sequence[Mapping[str, Any]],
    ) -> UtilityScore:
        raise RuntimeError("ProxyUtilityEvaluator requires the candidate level")


class PrivacyUtilityCritic:
    def __init__(
        self,
        auditor: AdversarialMemoryAuditor | None = None,
        utility_threshold: float = 0.85,
        utility_evaluator: UtilityEvaluator | None = None,
        *,
        allow_proxy: bool = False,
    ):
        if not 0.0 <= utility_threshold <= 1.0:
            raise ValueError("utility_threshold must be between 0 and 1")
        self.auditor = auditor or AdversarialMemoryAuditor()
        self.utility_threshold = utility_threshold
        self.utility_evaluator = utility_evaluator
        self.proxy_evaluator = ProxyUtilityEvaluator() if allow_proxy else None

    def score_candidate(
        self,
        candidate: AbstractionCandidate,
        privacy_items: Sequence[Mapping[str, Any]],
        questions: Sequence[Mapping[str, Any]] | None = None,
        auxiliary_context: Mapping[str, Any] | None = None,
        attack_types: tuple[str, ...] = ("exact_reconstruction",),
    ) -> CandidateScore:
        attacks = self.auditor.audit_candidate(
            candidate,
            privacy_items,
            auxiliary_context=auxiliary_context,
            attack_types=attack_types,
        )
        return CandidateScore(
            candidate_id=candidate.candidate_id,
            source_id=candidate.source_id,
            utility=self.score_utility(candidate, questions or []),
            privacy=self.score_privacy(attacks),
            attacks=attacks,
        )

    def score_utility(
        self,
        candidate: AbstractionCandidate,
        questions: Sequence[Mapping[str, Any]],
    ) -> UtilityScore:
        if self.utility_evaluator is not None:
            return self.utility_evaluator.evaluate(
                candidate.public_memory,
                questions,
            )
        if self.proxy_evaluator is not None:
            return self.proxy_evaluator.evaluate_candidate(candidate, questions)
        raise RuntimeError(
            "utility scoring requires a UtilityEvaluator; use allow_proxy=True "
            "only for smoke tests, never for paper evidence"
        )

    @staticmethod
    def score_privacy(attacks: Sequence[AttackResult]) -> PrivacyScore:
        exact = [
            attack for attack in attacks if attack.attack_type == "exact_reconstruction"
        ]
        attributes = [
            attack for attack in attacks if attack.attack_type == "attribute_inference"
        ]
        exact_rate = (
            sum(attack.success for attack in exact) / len(exact) if exact else 0.0
        )
        attribute_rate = (
            sum(attack.success for attack in attributes) / len(attributes)
            if attributes
            else None
        )
        semantic_score = (
            sum(attack.match_score for attack in attacks) / len(attacks)
            if attacks
            else 0.0
        )
        return PrivacyScore(
            exact_reconstruction_rate=exact_rate,
            attribute_inference_rate=attribute_rate,
            semantic_leakage_score=semantic_score,
            per_type=aggregate_attacks_by_type(attacks),
        )

    def select_candidate(
        self,
        candidates: Sequence[AbstractionCandidate],
        scores: Sequence[CandidateScore],
        utility_threshold: float | None = None,
    ) -> AbstractionCandidate:
        if not candidates:
            raise ValueError("cannot select from empty candidates")
        threshold = (
            self.utility_threshold if utility_threshold is None else utility_threshold
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("utility threshold must be between 0 and 1")
        score_by_id = {score.candidate_id: score for score in scores}
        missing = [
            candidate.candidate_id
            for candidate in candidates
            if candidate.candidate_id not in score_by_id
        ]
        if missing:
            raise ValueError(f"missing scores for candidates: {missing}")
        pairs = [
            (candidate, score_by_id[candidate.candidate_id]) for candidate in candidates
        ]
        valid = [pair for pair in pairs if pair[1].utility.mcq_accuracy >= threshold]
        if not valid:
            return (
                self._find_fallback(candidates, "L4")
                or self._find_fallback(candidates, "L5")
                or candidates[0]
            )
        selected, _ = min(
            valid,
            key=lambda pair: (
                pair[1].privacy.exact_reconstruction_rate,
                (
                    pair[1].privacy.attribute_inference_rate
                    if pair[1].privacy.attribute_inference_rate is not None
                    else 0.0
                ),
                self._exposure_rank(pair[0]),
                len(pair[0].public_memory),
                pair[0].level,
                pair[0].candidate_id,
            ),
        )
        return selected

    def build_training_records(
        self,
        grouped_candidates: Mapping[str, Sequence[AbstractionCandidate]],
        grouped_scores: Mapping[str, Sequence[CandidateScore]],
        source_inputs: Mapping[str, Mapping[str, Any]],
        utility_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for source_id in sorted(grouped_candidates):
            candidates = list(grouped_candidates[source_id])
            scores = list(grouped_scores.get(source_id, []))
            if not scores:
                continue
            chosen = self.select_candidate(
                candidates,
                scores,
                utility_threshold,
            )
            score_by_id = {score.candidate_id: score for score in scores}
            chosen_score = score_by_id[chosen.candidate_id]
            rejected = [
                candidate
                for candidate in candidates
                if candidate.candidate_id != chosen.candidate_id
                and self._is_valid_rejection(
                    chosen_score,
                    score_by_id[candidate.candidate_id],
                )
            ]
            records.append(
                {
                    "example_id": source_id,
                    "input": dict(source_inputs.get(source_id, {})),
                    "chosen": chosen.to_dict(),
                    "rejected": [
                        {
                            "candidate": candidate.to_dict(),
                            "score": score_by_id[candidate.candidate_id].to_dict(),
                        }
                        for candidate in rejected
                    ],
                    "scores": chosen_score.to_dict(),
                }
            )
        return records

    @staticmethod
    def _is_valid_rejection(
        chosen: CandidateScore,
        rejected: CandidateScore,
    ) -> bool:
        return (
            rejected.privacy.exact_reconstruction_rate
            > chosen.privacy.exact_reconstruction_rate
            or rejected.utility.mcq_accuracy < chosen.utility.mcq_accuracy
            or (
                rejected.privacy.exact_reconstruction_rate
                == chosen.privacy.exact_reconstruction_rate
                and rejected.utility.mcq_accuracy == chosen.utility.mcq_accuracy
            )
        )

    @staticmethod
    def _find_fallback(
        candidates: Sequence[AbstractionCandidate],
        preferred_level: str,
    ) -> AbstractionCandidate | None:
        return next(
            (
                candidate
                for candidate in candidates
                if candidate.level == preferred_level
            ),
            None,
        )

    @staticmethod
    def _exposure_rank(candidate: AbstractionCandidate) -> int:
        level_rank = {
            level: index
            for index, level in enumerate(["L5", "L4", "L3", "L2", "L1", "L0"])
        }
        effective = candidate.metadata.get("effective_levels", [])
        if effective:
            return sum(
                level_rank.get(str(item.get("level", "L0")), 5)
                for item in effective
                if isinstance(item, Mapping)
            )
        return level_rank.get(candidate.level, 5)


def group_candidates(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[AbstractionCandidate]]:
    grouped: dict[str, list[AbstractionCandidate]] = defaultdict(list)
    for record in records:
        for item in record.get("candidates", []):
            candidate = AbstractionCandidate.from_dict(item)
            grouped[candidate.source_id].append(candidate)
    return dict(grouped)


def group_scores(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[CandidateScore]]:
    grouped: dict[str, list[CandidateScore]] = defaultdict(list)
    for record in records:
        score = CandidateScore.from_dict(record)
        grouped[score.source_id].append(score)
    return dict(grouped)
