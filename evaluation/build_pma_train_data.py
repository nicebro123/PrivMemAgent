import argparse
import json
import os
from typing import Any

from src.privacy_critic import PrivacyUtilityCritic, group_candidates, group_scores

SYSTEM_PROMPT = (
    "You are a local Privacy Memory Abstractor. Convert raw user memory into "
    "a task-sufficient public abstraction and keep sensitive private residues local. "
    "Return strict JSON with public_memory, private_residue, and abstraction_trace."
)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_source_inputs(
    candidate_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    source_inputs: dict[str, dict[str, Any]] = {}
    for record in candidate_records:
        source_id = record.get("source", {}).get("source_id")
        if not source_id:
            continue
        source_inputs[source_id] = record.get("input", {})
    return source_inputs


def build_sft_records(training_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in training_records:
        output.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            record["input"], ensure_ascii=False, indent=2
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "public_memory": record["chosen"]["public_memory"],
                                "private_residue": record["chosen"]["private_residue"],
                                "abstraction_trace": record["chosen"][
                                    "abstraction_trace"
                                ],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    },
                ],
                "metadata": {
                    "example_id": record["example_id"],
                    "chosen_candidate_id": record["chosen"]["candidate_id"],
                    "scores": record.get("scores", {}),
                    "source_id": record["chosen"]["source_id"],
                },
            }
        )
    return output


def build_preference_records(
    training_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in training_records:
        prompt = json.dumps(record["input"], ensure_ascii=False, indent=2)
        chosen = json.dumps(
            {
                "public_memory": record["chosen"]["public_memory"],
                "private_residue": record["chosen"]["private_residue"],
                "abstraction_trace": record["chosen"]["abstraction_trace"],
            },
            ensure_ascii=False,
            indent=2,
        )
        seen_rejected: set[str] = set()
        for rejected in record.get("rejected", []):
            candidate = rejected.get("candidate")
            if not candidate:
                continue
            chosen_score = record.get("scores", {})
            rejected_score = rejected.get("score") or {}
            rejected_output = json.dumps(
                {
                    "public_memory": candidate["public_memory"],
                    "private_residue": candidate["private_residue"],
                    "abstraction_trace": candidate["abstraction_trace"],
                },
                ensure_ascii=False,
                indent=2,
            )
            if rejected_output == chosen or rejected_output in seen_rejected:
                continue
            seen_rejected.add(rejected_output)
            output.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected_output,
                    "metadata": {
                        "example_id": record["example_id"],
                        "chosen_candidate_id": record["chosen"]["candidate_id"],
                        "rejected_candidate_id": candidate["candidate_id"],
                        "source_id": record["chosen"]["source_id"],
                        "chosen_utility": chosen_score.get("utility", {}).get(
                            "mcq_accuracy"
                        ),
                        "chosen_leakage": chosen_score.get("privacy", {}).get(
                            "exact_reconstruction_rate"
                        ),
                        "rejected_utility": rejected_score.get("utility", {}).get(
                            "mcq_accuracy"
                        ),
                        "rejected_leakage": rejected_score.get("privacy", {}).get(
                            "exact_reconstruction_rate"
                        ),
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                    },
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PMA SFT and preference data from candidates and scores."
    )
    parser.add_argument("--candidates", required=True, help="Candidate JSONL path.")
    parser.add_argument("--scores", required=True, help="Candidate score JSONL path.")
    parser.add_argument("--sft-output", required=True, help="Output SFT JSONL.")
    parser.add_argument(
        "--preference-output", required=True, help="Output preference JSONL."
    )
    parser.add_argument("--utility-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_records = read_jsonl(args.candidates)
    score_records = read_jsonl(args.scores)
    critic = PrivacyUtilityCritic(utility_threshold=args.utility_threshold)
    training_records = critic.build_training_records(
        grouped_candidates=group_candidates(candidate_records),
        grouped_scores=group_scores(score_records),
        source_inputs=build_source_inputs(candidate_records),
        utility_threshold=args.utility_threshold,
    )
    sft_records = build_sft_records(training_records)
    preference_records = build_preference_records(training_records)
    write_jsonl(args.sft_output, sft_records)
    write_jsonl(args.preference_output, preference_records)
    print(f"Wrote {len(sft_records)} SFT records to {args.sft_output}")
    print(
        f"Wrote {len(preference_records)} preference records "
        f"to {args.preference_output}"
    )


if __name__ == "__main__":
    main()
