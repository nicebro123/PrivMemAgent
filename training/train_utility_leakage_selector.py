from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def fit_simple_weights(examples_path: Path) -> dict[str, float]:
    positives = []
    negatives = []
    for row in _iter_jsonl(examples_path):
        target = positives if row.get("label_positive") else negatives
        target.append(row)

    weights = {
        "bias": 0.0,
        "utility_score": 1.0,
        "leakage_score": -1.0,
        "token_count": -0.01,
        "abstraction_level": -0.05,
        "attribute_risk_score": -0.30,
        "exact_leak_flag": -100.0,
        "pl4_flag": -100.0,
    }
    if positives and negatives:
        pos_utility = mean(float(row.get("utility_score", 0.0)) for row in positives)
        neg_utility = mean(float(row.get("utility_score", 0.0)) for row in negatives)
        pos_leakage = mean(float(row.get("leakage_score", 1.0)) for row in positives)
        neg_leakage = mean(float(row.get("leakage_score", 1.0)) for row in negatives)
        weights["utility_score"] = max(0.5, 1.0 + pos_utility - neg_utility)
        weights["leakage_score"] = min(-0.5, -1.0 - max(0.0, neg_leakage - pos_leakage))
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a lightweight utility-leakage selector artifact.")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = fit_simple_weights(args.examples.expanduser().resolve())
    artifact = {
        "artifact_type": "linear_utility_leakage_selector",
        "version": 1,
        "weights": weights,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"weight_count": len(weights), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
