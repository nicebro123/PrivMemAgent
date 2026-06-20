from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from typing import Any

import yaml
from tqdm import tqdm

from src.llm_clients import build_completion_from_config
from src.memory_backends import Mem0Backend
from src.privacy_auditor import AdversarialMemoryAuditor
from src.privacy_critic import (
    PrivacyUtilityCritic,
    QuestionAnswerUtilityEvaluator,
)
from src.privacy_schema import AbstractionCandidate
from src.qa_evaluation import build_open_qa_judge, build_question_answerer


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_attack_types(value: str) -> tuple[str, ...]:
    if value == "all":
        return ("exact_reconstruction", "attribute_inference")
    aliases = {
        "exact": "exact_reconstruction",
        "attribute": "attribute_inference",
    }
    attack_types = tuple(
        aliases.get(part.strip(), part.strip())
        for part in value.split(",")
        if part.strip()
    )
    supported = {"exact_reconstruction", "attribute_inference"}
    unknown = set(attack_types) - supported
    if unknown:
        raise ValueError(f"unsupported attack types: {sorted(unknown)}")
    return attack_types or ("exact_reconstruction",)


def _build_direct_utility_evaluator(config: Mapping[str, Any]):
    answer_completion = build_completion_from_config(config, "answer_llm")
    judge_completion = build_completion_from_config(config, "judgment_llm")
    return QuestionAnswerUtilityEvaluator(
        build_question_answerer(answer_completion),
        build_open_qa_judge(judge_completion),
    )


def _score_with_mem0(
    candidate: AbstractionCandidate,
    questions: list[dict[str, Any]],
    config: Mapping[str, Any],
    source: Mapping[str, Any],
):
    answer_completion = build_completion_from_config(config, "answer_llm")
    judge_completion = build_completion_from_config(config, "judgment_llm")
    direct_answer = build_question_answerer(answer_completion)
    temp_dir = tempfile.mkdtemp(prefix="pma-candidate-mem0-")
    try:
        user_id = str(source.get("user_id") or candidate.source_id)
        backend = Mem0Backend(user_id, temp_dir, config)
        backend.add(
            [{"role": source.get("role", "user"), "content": candidate.public_memory}],
            user_id,
        )

        def answer_from_memory(_public_memory: str, question: Mapping[str, Any]) -> str:
            retrieved = backend.search(str(question.get("question", "")), user_id)
            return direct_answer(retrieved, question)

        evaluator = QuestionAnswerUtilityEvaluator(
            answer_from_memory,
            build_open_qa_judge(judge_completion),
        )
        return evaluator.evaluate(candidate.public_memory, questions)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def score_candidate_records(
    candidate_records: list[dict[str, Any]],
    attack_types: tuple[str, ...],
    utility_threshold: float,
    *,
    config: Mapping[str, Any] | None = None,
    memory_system: str = "none",
    auditor_backend: str = "heuristic",
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    if memory_system not in {"none", "mem0"}:
        raise ValueError("memory_system must be `none` or `mem0`")
    if not dry_run and config is None:
        raise ValueError("real utility scoring requires evaluation config")

    audit_completion = None
    if auditor_backend == "llm":
        if config is None:
            raise ValueError("LLM auditor requires evaluation config")
        section = "attack_llm" if "attack_llm" in config else "privacy_llm"
        audit_completion = build_completion_from_config(config, section)
    auditor = AdversarialMemoryAuditor(
        backend=auditor_backend,
        completion_fn=audit_completion,
    )
    direct_evaluator = (
        _build_direct_utility_evaluator(config or {}) if not dry_run else None
    )
    critic = PrivacyUtilityCritic(
        auditor=auditor,
        utility_threshold=utility_threshold,
        utility_evaluator=direct_evaluator,
        allow_proxy=dry_run,
    )

    output: list[dict[str, Any]] = []
    for record in tqdm(candidate_records, desc="Scoring PMA candidates"):
        input_record = record.get("input", {})
        privacy_items = input_record.get("privacy_items", [])
        questions = record.get("questions", [])
        source = record.get("source", {})
        for candidate_data in record.get("candidates", []):
            candidate = AbstractionCandidate.from_dict(candidate_data)
            if memory_system == "mem0" and not dry_run:
                utility = _score_with_mem0(
                    candidate,
                    questions,
                    config or {},
                    source,
                )
                attacks = auditor.audit_candidate(
                    candidate,
                    privacy_items,
                    attack_types=attack_types,
                )
                score = critic.score_privacy(attacks)
                score_dict = {
                    "candidate_id": candidate.candidate_id,
                    "source_id": candidate.source_id,
                    "utility": utility.to_dict(),
                    "privacy": score.to_dict(),
                    "attacks": [attack.to_dict() for attack in attacks],
                }
            else:
                candidate_score = critic.score_candidate(
                    candidate=candidate,
                    privacy_items=privacy_items,
                    questions=questions,
                    attack_types=attack_types,
                )
                score_dict = candidate_score.to_dict()
            score_dict.update(
                {
                    "source": source,
                    "memory_system": memory_system,
                    "paper_evidence": not dry_run,
                    "dry_run": dry_run,
                }
            )
            output.append(score_dict)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score PMA candidates with measured QA utility and privacy attacks."
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Accepted for the documented CLI contract; source questions are "
            "embedded in candidates."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--memory-system",
        choices=["none", "mem0"],
        default="none",
    )
    parser.add_argument(
        "--attack",
        default="exact",
        help="exact, attribute, all, or comma-separated canonical names",
    )
    parser.add_argument(
        "--auditor-backend",
        choices=["heuristic", "llm"],
        default="llm",
    )
    parser.add_argument("--utility-threshold", type=float, default=0.85)
    parser.add_argument(
        "--config",
        default="evaluation/eval_config.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use an explicitly marked level proxy; output is not paper evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auditor_backend = "heuristic" if args.dry_run else args.auditor_backend
    config = None
    if not args.dry_run:
        with open(args.config, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    records = read_jsonl(args.candidates)
    scores = score_candidate_records(
        candidate_records=records,
        attack_types=parse_attack_types(args.attack),
        utility_threshold=args.utility_threshold,
        config=config,
        memory_system=args.memory_system,
        auditor_backend=auditor_backend,
        dry_run=args.dry_run,
    )
    write_jsonl(args.output, scores)
    print(f"Wrote {len(scores)} candidate scores to {args.output}")


if __name__ == "__main__":
    main()
