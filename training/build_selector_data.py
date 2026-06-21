from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _label(candidate: dict) -> dict:
    exact_leak = bool(candidate.get("exact_leak_flag", False))
    utility = float(candidate.get("utility_score", 0.0))
    leakage = float(candidate.get("leakage_score", 1.0))
    token_count = max(1, len(str(candidate.get("text", "")).split()))
    score = utility - leakage - 0.01 * token_count - (100.0 if exact_leak else 0.0)
    return {
        "score": score,
        "positive": bool(not exact_leak and utility >= 0.75 and leakage <= 0.35),
    }


def build_selector_examples(abstraction_examples: Path) -> list[dict]:
    rows = []
    for example in _iter_jsonl(abstraction_examples):
        item = example.get("privacy_item", {})
        for candidate in example.get("candidates", []):
            label = _label(candidate)
            rows.append(
                {
                    "user_id": example.get("user_id"),
                    "message_id": example.get("message_id"),
                    "privacy_type": item.get("privacy_type"),
                    "privacy_level": item.get("privacy_level"),
                    "candidate_text": candidate.get("text", ""),
                    "representation_type": candidate.get("representation_type"),
                    "abstraction_level": candidate.get("level"),
                    "utility_score": candidate.get("utility_score", 0.0),
                    "leakage_score": candidate.get("leakage_score", 1.0),
                    "token_count": max(1, len(str(candidate.get("text", "")).split())),
                    "label_score": label["score"],
                    "label_positive": label["positive"],
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build utility-leakage selector examples.")
    parser.add_argument("--abstraction-examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_selector_examples(args.abstraction_examples.expanduser().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"example_count": len(rows), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
