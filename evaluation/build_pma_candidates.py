import argparse
import json
import os
from typing import Any

import yaml
from tqdm import tqdm

from src.llm_clients import build_completion_from_config
from src.privacy_abstraction import PrivacyMemoryAbstractor, load_abstraction_policy
from src.privacy_schema import stable_id


def iter_user_records(path: str, user_limit: int | None = None):
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if user_limit is not None and idx >= user_limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_source_id(user_id: str, turn_index: int, role: str) -> str:
    return stable_id(user_id, turn_index, role)


def build_candidate_records(
    input_path: str,
    users: int | None,
    max_turns: int | None,
    task_family: str,
    backend: str,
    policy: dict[str, Any],
    include_non_private: bool = False,
    completion_fn=None,
    model_name_or_path: str = "",
    model_revision: str = "main",
    fallback_on_error: bool = True,
) -> list[dict[str, Any]]:
    abstractor = PrivacyMemoryAbstractor(
        policy=policy,
        backend=backend,
        completion_fn=completion_fn,
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        fallback_on_error=fallback_on_error,
    )
    output: list[dict[str, Any]] = []
    for user_data in tqdm(
        list(iter_user_records(input_path, users)), desc="Building PMA candidates"
    ):
        user_id = user_data.get("uuid", "")
        dialogues = user_data.get("dialogues", [])
        if max_turns is not None:
            dialogues = dialogues[:max_turns]
        for turn_index, dialogue in enumerate(dialogues):
            privacy_items = dialogue.get("privacy_info_llm")
            if privacy_items is None:
                privacy_items = dialogue.get("privacy_info", [])
            if not privacy_items and not include_non_private:
                continue
            content = dialogue.get("content", "")
            if not content:
                continue
            source_id = build_source_id(
                user_id, turn_index, dialogue.get("role", "unknown")
            )
            candidates = abstractor.generate_candidates(
                dialogue_text=content,
                privacy_items=privacy_items,
                task_family=task_family,
                source_id=source_id,
            )
            output.append(
                {
                    "source": {
                        "source_id": source_id,
                        "user_id": user_id,
                        "turn_index": turn_index,
                        "role": dialogue.get("role", ""),
                    },
                    "input": {
                        "dialogue": content,
                        "privacy_items": privacy_items,
                        "task_family": task_family,
                        "policy": policy,
                    },
                    "metadata": user_data.get("metadata", {}),
                    "questions": user_data.get("questions", []),
                    "candidates": [candidate.to_dict() for candidate in candidates],
                    "backend_error": abstractor.last_backend_error,
                }
            )
    return output


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PMA abstraction candidates from privacy-annotated dialogues."
    )
    parser.add_argument("--input", required=True, help="Input dataset JSONL path.")
    parser.add_argument("--output", required=True, help="Output candidate JSONL path.")
    parser.add_argument(
        "--users", type=int, default=None, help="Maximum number of users to process."
    )
    parser.add_argument(
        "--max-turns", type=int, default=None, help="Maximum dialogue turns per user."
    )
    parser.add_argument(
        "--task-family", default="general", help="Task family conditioning string."
    )
    parser.add_argument(
        "--backend",
        default="oracle_prompt",
        choices=[
            "heuristic",
            "oracle_prompt",
            "trained_model",
            "typed_placeholder",
            "redaction",
        ],
    )
    parser.add_argument(
        "--policy-config", default=None, help="Optional PMA policy YAML."
    )
    parser.add_argument(
        "--config",
        default="evaluation/eval_config.yaml",
        help="Evaluation YAML containing the privacy_llm section.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Local PMA checkpoint for the trained_model backend.",
    )
    parser.add_argument(
        "--model-revision",
        default="main",
        help="Prefer an immutable Hugging Face commit hash.",
    )
    parser.add_argument(
        "--strict-backend",
        action="store_true",
        help="Fail instead of emitting only safe L4/L5 fallbacks on backend errors.",
    )
    parser.add_argument(
        "--include-non-private",
        action="store_true",
        help="Also emit pass-through candidates for non-private turns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_abstraction_policy(args.policy_config)
    completion_fn = None
    if args.backend == "oracle_prompt":
        with open(args.config, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        completion_fn = build_completion_from_config(config, "privacy_llm")
    if args.backend == "trained_model" and not args.model_path:
        raise ValueError("--model-path is required for trained_model backend")
    records = build_candidate_records(
        input_path=args.input,
        users=args.users,
        max_turns=args.max_turns,
        task_family=args.task_family,
        backend=args.backend,
        policy=policy,
        include_non_private=args.include_non_private,
        completion_fn=completion_fn,
        model_name_or_path=args.model_path,
        model_revision=args.model_revision,
        fallback_on_error=not args.strict_backend,
    )
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} candidate records to {args.output}")


if __name__ == "__main__":
    main()
