from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

UTILITY_FIELD = "non_private_answer_token_recall"
LOCAL_RECOVERABILITY_FIELD = "oracle_type_local_recoverability"
TOKEN_FIELD = "public_token_count"


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _gate_failed(
    run: dict[str, Any], *, max_exact_recovery: float, max_pl4_retention: float
) -> list[str]:
    failed: list[str] = []
    if not run.get("audit_passed", False):
        failed.append("builtin_audit_failed")
    if run.get("cloud_audit_passed") is False:
        failed.append("cloud_audit_failed")
    if run.get("adversarial_passed") is False:
        failed.append("adversarial_audit_failed")
    if _as_float(run.get("exact_recovery_rate"), 1.0) > max_exact_recovery:
        failed.append("exact_recovery_over_budget")
    if _as_float(run.get("pl4_public_retention_rate"), 1.0) > max_pl4_retention:
        failed.append("pl4_retention_over_budget")
    if _as_float(run.get("adversarial_failure_count"), 0.0) > 0:
        failed.append("adversarial_failures_present")
    return failed


def _score(run: dict[str, Any]) -> tuple[float, float, float, int]:
    utility = _as_float(run.get(UTILITY_FIELD), 0.0)
    local_recoverability = _as_float(run.get(LOCAL_RECOVERABILITY_FIELD), 0.0)
    public_tokens = _as_float(run.get(TOKEN_FIELD), 0.0)
    budget = int(run["max_public_tokens"])
    return (utility, local_recoverability, -public_tokens, -budget)


def _is_dominated(candidate: dict[str, Any], other: dict[str, Any]) -> bool:
    candidate_utility = _as_float(candidate.get(UTILITY_FIELD), 0.0)
    candidate_local = _as_float(candidate.get(LOCAL_RECOVERABILITY_FIELD), 0.0)
    candidate_tokens = _as_float(candidate.get(TOKEN_FIELD), 0.0)
    other_utility = _as_float(other.get(UTILITY_FIELD), 0.0)
    other_local = _as_float(other.get(LOCAL_RECOVERABILITY_FIELD), 0.0)
    other_tokens = _as_float(other.get(TOKEN_FIELD), 0.0)

    at_least_as_good = (
        other_utility >= candidate_utility
        and other_local >= candidate_local
        and other_tokens <= candidate_tokens
    )
    strictly_better = (
        other_utility > candidate_utility
        or other_local > candidate_local
        or other_tokens < candidate_tokens
    )
    return at_least_as_good and strictly_better


def _pareto_frontier(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = list(runs)
    frontier = [
        run
        for run in candidates
        if not any(_is_dominated(run, other) for other in candidates if other is not run)
    ]
    return sorted(
        frontier,
        key=lambda run: (int(run["max_public_tokens"]), _as_float(run.get(TOKEN_FIELD), 0.0)),
    )


def recommend_budget(
    sweep_summary: dict[str, Any],
    *,
    min_utility: float = 0.0,
    min_local_recoverability: float = 0.0,
    max_exact_recovery: float = 0.0,
    max_pl4_retention: float = 0.0,
) -> dict[str, Any]:
    runs = sorted(sweep_summary.get("runs", []), key=lambda run: int(run["max_public_tokens"]))
    evaluated: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []

    for run in runs:
        reasons = _gate_failed(
            run,
            max_exact_recovery=max_exact_recovery,
            max_pl4_retention=max_pl4_retention,
        )
        utility = _as_float(run.get(UTILITY_FIELD), 0.0)
        local_recoverability = _as_float(run.get(LOCAL_RECOVERABILITY_FIELD), 0.0)
        if utility < min_utility:
            reasons.append("utility_below_floor")
        if local_recoverability < min_local_recoverability:
            reasons.append("local_recoverability_below_floor")

        decorated = {
            "max_public_tokens": run["max_public_tokens"],
            "eligible": not reasons,
            "reasons": reasons,
            "utility": utility,
            "local_recoverability": local_recoverability,
            "public_token_count": run.get(TOKEN_FIELD),
            "exact_recovery_rate": run.get("exact_recovery_rate"),
            "pl4_public_retention_rate": run.get("pl4_public_retention_rate"),
            "adversarial_failure_count": run.get("adversarial_failure_count"),
            "source_run": run,
        }
        evaluated.append(decorated)
        if not reasons:
            eligible.append(run)

    thresholds = {
        "min_utility": min_utility,
        "min_local_recoverability": min_local_recoverability,
        "max_exact_recovery": max_exact_recovery,
        "max_pl4_retention": max_pl4_retention,
    }
    if not eligible:
        return {
            "dataset": sweep_summary.get("dataset"),
            "annotation_source": sweep_summary.get("annotation_source"),
            "eligible": False,
            "recommendation": None,
            "reason": "no_budget_satisfies_all_gates",
            "thresholds": thresholds,
            "evaluated_runs": evaluated,
            "pareto_frontier": [],
        }

    smallest = min(eligible, key=lambda run: int(run["max_public_tokens"]))
    best_utility = max(eligible, key=_score)
    frontier = _pareto_frontier(eligible)
    return {
        "dataset": sweep_summary.get("dataset"),
        "annotation_source": sweep_summary.get("annotation_source"),
        "eligible": True,
        "recommendation": {
            "smallest_safe_budget": smallest["max_public_tokens"],
            "best_utility_budget": best_utility["max_public_tokens"],
            "recommended_budget": smallest["max_public_tokens"],
            "rationale": (
                "Choose the smallest budget that satisfies privacy, cloud-safety, "
                "adversarial, utility, and local-recoverability gates. Use the "
                "best_utility_budget when extra public-memory size is acceptable."
            ),
        },
        "thresholds": thresholds,
        "evaluated_runs": evaluated,
        "pareto_frontier": [
            {
                "max_public_tokens": run["max_public_tokens"],
                "utility": _as_float(run.get(UTILITY_FIELD), 0.0),
                "local_recoverability": _as_float(run.get(LOCAL_RECOVERABILITY_FIELD), 0.0),
                "public_token_count": run.get(TOKEN_FIELD),
            }
            for run in frontier
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a recommended public-memory budget from a budget-sweep "
            "summary using privacy, utility, and adversarial gates."
        )
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-utility", type=float, default=0.0)
    parser.add_argument("--min-local-recoverability", type=float, default=0.0)
    parser.add_argument("--max-exact-recovery", type=float, default=0.0)
    parser.add_argument("--max-pl4-retention", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    recommendation = recommend_budget(
        summary,
        min_utility=args.min_utility,
        min_local_recoverability=args.min_local_recoverability,
        max_exact_recovery=args.max_exact_recovery,
        max_pl4_retention=args.max_pl4_retention,
    )
    rendered = json.dumps(recommendation, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
