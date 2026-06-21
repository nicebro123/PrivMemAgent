from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def distill_templates(examples_path: Path) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for example in _iter_jsonl(examples_path):
        item = example.get("privacy_item", {})
        privacy_type = str(item.get("privacy_type", "*"))
        privacy_level = str(item.get("privacy_level", "*"))
        for candidate in example.get("candidates", []):
            text = str(candidate.get("text", ""))
            if not text:
                continue
            grouped[(privacy_type, privacy_level, text)].append(candidate)

    templates = []
    for (privacy_type, privacy_level, text), candidates in sorted(grouped.items()):
        utility = sum(float(c.get("utility_score", 0.0)) for c in candidates) / len(candidates)
        leakage = sum(float(c.get("leakage_score", 1.0)) for c in candidates) / len(candidates)
        level = round(sum(float(c.get("level", 2)) for c in candidates) / len(candidates))
        representation_counts = defaultdict(int)
        for candidate in candidates:
            representation_counts[str(candidate.get("representation_type", "learned_abstract"))] += 1
        representation_type = max(representation_counts.items(), key=lambda item: (item[1], item[0]))[0]
        templates.append(
            {
                "privacy_type": privacy_type,
                "privacy_level": privacy_level,
                "text": text,
                "abstraction_level": int(level),
                "utility_score": utility,
                "leakage_score": leakage,
                "representation_type": f"distilled_{representation_type}",
                "support": len(candidates),
            }
        )
    return templates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill abstraction examples into a JSON template artifact.")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    templates = [t for t in distill_templates(args.examples.expanduser().resolve()) if t["support"] >= args.min_support]
    artifact = {
        "artifact_type": "abstraction_generator_templates",
        "version": 1,
        "templates": templates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"template_count": len(templates), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
