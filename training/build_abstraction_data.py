from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

from src.abstraction_generator import AbstractionInput, RuleBasedAbstractionGenerator


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _privacy_items(message: Mapping, annotation_source: str) -> list[dict]:
    key = "privacy_info_llm" if annotation_source == "model" else "privacy_info"
    return [dict(item) for item in message.get(key, [])]


def build_examples(input_path: Path, annotation_source: str = "model") -> list[dict]:
    generator = RuleBasedAbstractionGenerator()
    examples = []
    for user in _iter_jsonl(input_path):
        user_id = str(user.get("uuid", ""))
        dialogues = list(user.get("dialogues", []))
        questions = [str(q.get("question", "")) for q in user.get("questions", []) if q.get("question")]
        for message_index, message in enumerate(dialogues):
            message_id = f"{user_id}:{message_index}"
            for item_index, item in enumerate(_privacy_items(message, annotation_source)):
                abstraction_input = AbstractionInput(
                    user_id=user_id,
                    message_id=message_id,
                    role=str(message.get("role", "user")),
                    message_text=str(message.get("content", "")),
                    privacy_item=item,
                    neighboring_context=tuple(dialogues[max(0, message_index - 1): message_index + 2]),
                    question_hints=tuple(questions[:8]),
                )
                candidates = generator.generate(abstraction_input)
                examples.append(
                    {
                        "user_id": user_id,
                        "message_id": message_id,
                        "source_item_index": item_index,
                        "message_text": message.get("content", ""),
                        "privacy_item": item,
                        "question_hints": questions[:8],
                        "candidates": [
                            {
                                "text": candidate.text,
                                "level": candidate.abstraction_level,
                                "representation_type": candidate.representation_type,
                                "utility_score": candidate.utility_score,
                                "leakage_score": candidate.leakage_score,
                                "contains_alias": candidate.contains_alias,
                            }
                            for candidate in candidates
                        ],
                    }
                )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build abstraction-generator training examples.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotation-source", choices=("model", "oracle"), default="model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = build_examples(args.input.expanduser().resolve(), args.annotation_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for example in examples:
            output.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(json.dumps({"example_count": len(examples), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
