from tools.public_memory_budget_selector import recommend_budget


def _run(
    budget,
    utility,
    tokens,
    *,
    audit=True,
    adversarial=True,
    exact=0.0,
    pl4=0.0,
    local=1.0,
):
    return {
        "max_public_tokens": budget,
        "audit_passed": audit,
        "cloud_audit_passed": True,
        "adversarial_passed": adversarial,
        "adversarial_failure_count": 0 if adversarial else 1,
        "exact_recovery_rate": exact,
        "pl4_public_retention_rate": pl4,
        "non_private_answer_token_recall": utility,
        "oracle_type_local_recoverability": local,
        "public_token_count": tokens,
    }


def test_recommend_budget_prefers_smallest_safe_budget_and_reports_best_utility():
    recommendation = recommend_budget(
        {
            "dataset": "toy.jsonl",
            "annotation_source": "oracle",
            "runs": [
                _run(8, 0.40, 30),
                _run(16, 0.82, 50),
                _run(32, 0.91, 80),
            ],
        },
        min_utility=0.8,
        min_local_recoverability=0.95,
    )

    assert recommendation["eligible"] is True
    assert recommendation["recommendation"]["smallest_safe_budget"] == 16
    assert recommendation["recommendation"]["best_utility_budget"] == 32
    assert recommendation["recommendation"]["recommended_budget"] == 16
    assert recommendation["evaluated_runs"][0]["eligible"] is False
    assert "utility_below_floor" in recommendation["evaluated_runs"][0]["reasons"]
    assert [run["max_public_tokens"] for run in recommendation["pareto_frontier"]] == [16, 32]


def test_recommend_budget_rejects_when_privacy_gate_fails():
    recommendation = recommend_budget(
        {
            "runs": [
                _run(16, 0.95, 50, exact=0.2),
                _run(32, 0.96, 70, adversarial=False),
            ]
        },
        min_utility=0.8,
    )

    assert recommendation["eligible"] is False
    assert recommendation["recommendation"] is None
    reasons = {reason for run in recommendation["evaluated_runs"] for reason in run["reasons"]}
    assert "exact_recovery_over_budget" in reasons
    assert "adversarial_audit_failed" in reasons


def test_pareto_frontier_drops_dominated_budget():
    recommendation = recommend_budget(
        {
            "runs": [
                _run(16, 0.90, 50),
                _run(32, 0.90, 70),
                _run(64, 0.95, 90),
            ]
        },
        min_utility=0.8,
    )

    assert [run["max_public_tokens"] for run in recommendation["pareto_frontier"]] == [16, 64]
