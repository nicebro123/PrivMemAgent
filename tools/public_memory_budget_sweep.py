from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from evaluation.eval_public_memory import _load_public_memory_config, compile_dataset
from tools.adversarial_audit import audit_artifacts


def _set_budget(config: dict, max_public_tokens: int) -> dict:
    candidate = deepcopy(config)
    public_memory = candidate.setdefault("public_memory", {})
    selector = public_memory.setdefault("selector", {})
    selector["max_public_tokens"] = max_public_tokens
    return candidate


def _summarize_run(
    budget: int,
    metrics: dict,
    adversarial: dict | None,
    public_output: Path,
    cloud_safe_output: Path | None,
) -> dict:
    audit = metrics["audit"]
    utility = metrics["utility_proxy"]
    cloud_audit = (metrics.get("cloud_safe_dataset") or {}).get("audit")
    summary = {
        "max_public_tokens": budget,
        "public_output": str(public_output),
        "cloud_safe_output": str(cloud_safe_output) if cloud_safe_output else None,
        "record_count": metrics["record_count"],
        "invalid_message_count": metrics["invalid_message_count"],
        "invalid_annotation_count": metrics["invalid_annotation_count"],
        "route_counts": metrics["route_counts"],
        "representation_counts": metrics["representation_counts"],
        "audit_passed": audit["passed"],
        "exact_recovery_rate": audit["exact_recovery_rate"],
        "pl4_public_retention_rate": audit["pl4_public_retention_rate"],
        "corpus_token_reduction": audit["corpus_token_reduction"],
        "average_token_reduction": audit["average_token_reduction"],
        "source_token_count": audit["source_token_count"],
        "public_token_count": audit["public_token_count"],
        "oracle_type_local_recoverability": utility["oracle_type_local_recoverability"],
        "local_recoverability_applicable": utility["local_recoverability_applicable"],
        "non_private_answer_token_recall": utility["non_private_answer_token_recall"],
        "pl4_local_retention_rate": utility["pl4_local_retention_rate"],
        "cloud_audit_passed": cloud_audit["passed"] if cloud_audit else None,
    }
    if adversarial is not None:
        summary.update(
            {
                "adversarial_passed": adversarial["passed"],
                "adversarial_failure_count": adversarial["failure_count"],
                "adversarial_warning_count": adversarial["warning_count"],
                "adversarial_exact_exposure_count": adversarial[
                    "exact_exposure_count"
                ],
                "adversarial_pl4_exposure_count": adversarial["pl4_exposure_count"],
                "adversarial_membership_exposure_count": adversarial[
                    "membership_exposure_count"
                ],
                "adversarial_attribute_exposure_count": adversarial[
                    "attribute_exposure_count"
                ],
            }
        )
    return summary


def run_budget_sweep(
    input_path: Path,
    output_dir: Path,
    config: dict,
    budgets: Sequence[int],
    annotation_source: str,
    user_limit: int | None = None,
    minimum_token_reduction: float | None = None,
    turns_per_session: int = 20,
    turns_per_task: int = 5,
    exact_required_types: Iterable[str] = (),
    consented_reversible_types: Iterable[str] = (),
    include_cloud_safe_dataset: bool = True,
    run_adversarial_audit: bool = True,
) -> dict:
    if not budgets:
        raise ValueError("at least one budget is required")
    if any(budget < 1 for budget in budgets):
        raise ValueError("budgets must be positive integers")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for budget in budgets:
        budget_config = _set_budget(config, budget)
        if minimum_token_reduction is not None:
            budget_config.setdefault("public_memory", {}).setdefault(
                "leakage_budget", {}
            )["minimum_token_reduction"] = minimum_token_reduction

        run_dir = output_dir / f"budget_{budget}"
        run_dir.mkdir(parents=True, exist_ok=True)
        public_output = run_dir / "public_memory.jsonl"
        metrics_output = run_dir / "metrics.json"
        state_dir = run_dir / "state"
        cloud_safe_output = run_dir / "cloud_safe_dataset.jsonl" if include_cloud_safe_dataset else None

        metrics = compile_dataset(
            input_path=input_path,
            output_path=public_output,
            metrics_path=metrics_output,
            state_dir=state_dir,
            config=budget_config,
            annotation_source=annotation_source,
            user_limit=user_limit,
            turns_per_session=turns_per_session,
            turns_per_task=turns_per_task,
            exact_required_types=exact_required_types,
            consented_reversible_types=consented_reversible_types,
            cloud_safe_dataset_path=cloud_safe_output,
        )
        adversarial = None
        if run_adversarial_audit:
            artifact_paths = [public_output]
            if cloud_safe_output is not None:
                artifact_paths.append(cloud_safe_output)
            adversarial = asdict(
                audit_artifacts(
                    source_dataset=input_path,
                    artifacts=artifact_paths,
                    source_user_limit=user_limit,
                )
            )
            (run_dir / "adversarial_audit.json").write_text(
                json.dumps(adversarial, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        results.append(
            _summarize_run(
                budget=budget,
                metrics=metrics,
                adversarial=adversarial,
                public_output=public_output,
                cloud_safe_output=cloud_safe_output,
            )
        )

    return {
        "dataset": str(input_path),
        "annotation_source": annotation_source,
        "user_limit": user_limit,
        "budgets": list(budgets),
        "runs": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep public-memory max token budgets and report utility, leakage, "
            "minimality, and adversarial audit metrics for Pareto analysis."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/public_memory_config.yaml"),
    )
    parser.add_argument(
        "--budget",
        type=int,
        action="append",
        required=True,
        help="max_public_tokens value to evaluate; repeat for multiple budgets",
    )
    parser.add_argument(
        "--annotation-source",
        choices=("model", "oracle"),
        default="model",
    )
    parser.add_argument("--user-limit", type=int)
    parser.add_argument("--minimum-token-reduction", type=float)
    parser.add_argument("--turns-per-session", type=int, default=20)
    parser.add_argument("--turns-per-task", type=int, default=5)
    parser.add_argument("--exact-required-type", action="append", default=[])
    parser.add_argument("--consented-reversible-type", action="append", default=[])
    parser.add_argument(
        "--no-cloud-safe-dataset",
        action="store_true",
        help="skip cloud-safe same-schema artifact generation",
    )
    parser.add_argument(
        "--no-adversarial-audit",
        action="store_true",
        help="skip deterministic adversarial audit for faster exploratory sweeps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_public_memory_config(args.config.expanduser().resolve())
    summary = run_budget_sweep(
        input_path=args.input.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        config=config,
        budgets=args.budget,
        annotation_source=args.annotation_source,
        user_limit=args.user_limit,
        minimum_token_reduction=args.minimum_token_reduction,
        turns_per_session=args.turns_per_session,
        turns_per_task=args.turns_per_task,
        exact_required_types=args.exact_required_type,
        consented_reversible_types=args.consented_reversible_type,
        include_cloud_safe_dataset=not args.no_cloud_safe_dataset,
        run_adversarial_audit=not args.no_adversarial_audit,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    args.summary_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
