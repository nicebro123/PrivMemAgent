from src.sufficiency_selector import (
    RepresentationCandidate,
    SelectorConfig,
    SufficiencySelector,
)


def candidate(text, specificity, utility, leakage, tokens):
    return RepresentationCandidate(
        text=text,
        specificity=specificity,
        utility_score=utility,
        leakage_score=leakage,
        token_count=tokens,
        representation_type=text,
    )


def test_selector_chooses_least_specific_feasible_representation():
    selector = SufficiencySelector(
        SelectorConfig(
            utility_floor=0.75,
            max_leakage=0.35,
            max_public_tokens=10,
        )
    )
    result = selector.select(
        [
            candidate("generic", 0, 0.6, 0.05, 1),
            candidate("category", 1, 0.8, 0.2, 2),
            candidate("typed", 2, 0.95, 0.5, 2),
        ]
    )

    assert result.selected.text == "category"
    assert result.feasible_count == 1


def test_selector_rejects_candidates_outside_hard_budgets():
    selector = SufficiencySelector(
        SelectorConfig(utility_floor=0.9, max_leakage=0.1, max_public_tokens=2)
    )
    result = selector.select([candidate("candidate", 1, 0.8, 0.2, 3)])
    assert result.selected is None
