import unittest

from src.privacy_critic import PrivacyUtilityCritic
from src.privacy_schema import (
    AbstractionCandidate,
    CandidateScore,
    PrivacyScore,
    UtilityScore,
)


def candidate(
    candidate_id: str, level: str, public_memory: str
) -> AbstractionCandidate:
    return AbstractionCandidate(
        candidate_id=candidate_id,
        source_id="s1",
        level=level,  # type: ignore[arg-type]
        public_memory=public_memory,
        private_residue=[],
        abstraction_trace=[],
    )


def score(candidate_id: str, utility: float, leakage: float) -> CandidateScore:
    return CandidateScore(
        candidate_id=candidate_id,
        source_id="s1",
        utility=UtilityScore(mcq_accuracy=utility),
        privacy=PrivacyScore(exact_reconstruction_rate=leakage),
        attacks=[],
    )


class PMASelectionTest(unittest.TestCase):
    def test_selects_lower_leakage_when_utility_passes(self):
        critic = PrivacyUtilityCritic(utility_threshold=0.8)
        candidates = [
            candidate("a", "L1", "more specific"),
            candidate("b", "L2", "less specific"),
        ]
        selected = critic.select_candidate(
            candidates,
            [
                score("a", utility=0.9, leakage=0.5),
                score("b", utility=0.85, leakage=0.0),
            ],
        )
        self.assertEqual(selected.candidate_id, "b")

    def test_falls_back_to_typed_placeholder(self):
        critic = PrivacyUtilityCritic(utility_threshold=0.95)
        candidates = [
            candidate("a", "L2", "abstract"),
            candidate("b", "L4", "<Medical_Health_1>"),
        ]
        selected = critic.select_candidate(
            candidates,
            [
                score("a", utility=0.8, leakage=0.0),
                score("b", utility=0.6, leakage=0.0),
            ],
        )
        self.assertEqual(selected.level, "L4")


if __name__ == "__main__":
    unittest.main()
