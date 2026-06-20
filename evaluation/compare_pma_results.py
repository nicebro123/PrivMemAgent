from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_result(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def result_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, values in result.get("methods", {}).items():
        utility = values.get("utility", {})
        privacy = values.get("privacy", {})
        rows.append(
            {
                "method": method,
                "utility": float(utility.get("mcq_accuracy", 0.0)),
                "exact_reconstruction": float(
                    privacy.get("exact_reconstruction_rate", 0.0)
                ),
                "attribute_inference": privacy.get("attribute_inference_rate"),
                "questions": int(utility.get("total_num", 0)),
                "attacks": int(privacy.get("num_attacks", 0)),
            }
        )
    return rows


def format_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Utility | Exact reconstruction | Attribute inference "
        "| Questions | Attacks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        attribute = row["attribute_inference"]
        attribute_text = "n/a" if attribute is None else f"{float(attribute):.4f}"
        lines.append(
            f"| {row['method']} | {row['utility']:.4f} | "
            f"{row['exact_reconstruction']:.4f} | {attribute_text} | "
            f"{row['questions']} | {row['attacks']} |"
        )
    return "\n".join(lines)


def validate_result_contract(
    result: Mapping[str, Any],
    required_methods: set[str],
    *,
    require_paper_evidence: bool,
) -> list[str]:
    errors: list[str] = []
    run = result.get("run", {})
    methods = result.get("methods", {})
    missing = required_methods - set(methods)
    if missing:
        errors.append(f"missing required methods: {sorted(missing)}")
    if require_paper_evidence and not run.get("paper_evidence", False):
        errors.append("run is marked as non-paper evidence")
    if require_paper_evidence and run.get("memory_system") != "mem0":
        errors.append("paper evidence must use the real mem0 backend")
    for method, values in methods.items():
        utility = values.get("utility", {})
        privacy = values.get("privacy", {})
        if utility.get("proxy"):
            errors.append(f"{method}: utility is a proxy")
        if privacy.get("proxy"):
            errors.append(f"{method}: privacy is a proxy")
        if not privacy.get("per_type"):
            errors.append(f"{method}: missing per-type privacy metrics")
        if int(utility.get("total_num", 0)) <= 0:
            errors.append(f"{method}: no evaluated questions")
        if int(privacy.get("num_attacks", 0)) <= 0:
            errors.append(f"{method}: no privacy attacks")

    questions_by_method: dict[str, set[tuple[str, str]]] = {}
    for record in result.get("records", []):
        method = str(record.get("method", ""))
        questions_by_method.setdefault(method, set()).add(
            (str(record.get("user_id", "")), str(record.get("question", "")))
        )
        if "selected_abstractions" not in record:
            errors.append(f"{method}: record missing selected_abstractions")
        if "attacks" not in record:
            errors.append(f"{method}: record missing attacks")
    comparable = [questions_by_method.get(method, set()) for method in required_methods]
    if comparable and any(values != comparable[0] for values in comparable[1:]):
        errors.append("methods were not evaluated on identical user/question pairs")
    return sorted(set(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and render a PMA result JSON."
    )
    parser.add_argument("result")
    parser.add_argument(
        "--required-methods",
        nargs="+",
        default=["raw", "complete", "type_specific", "pma_sft"],
    )
    parser.add_argument("--allow-non-paper-evidence", action="store_true")
    parser.add_argument("--markdown-output", default="")
    parser.add_argument("--csv-output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = load_result(args.result)
    errors = validate_result_contract(
        result,
        set(args.required_methods),
        require_paper_evidence=not args.allow_non_paper_evidence,
    )
    rows = result_rows(result)
    markdown = format_markdown(rows)
    print(f"dataset: {result.get('run', {}).get('dataset')}")
    print(f"memory_system: {result.get('run', {}).get('memory_system')}")
    print(markdown)
    if args.markdown_output:
        Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    if args.csv_output:
        with open(args.csv_output, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
    if errors:
        raise SystemExit("Result contract failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
