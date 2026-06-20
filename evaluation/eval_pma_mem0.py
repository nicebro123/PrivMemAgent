from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from src.llm_clients import build_completion_from_config
from src.memory_backends import InMemoryBackend, Mem0Backend, MemoryBackend
from src.privacy_abstraction import (
    PrivacyMemoryAbstractor,
    build_private_residue,
    load_abstraction_policy,
    redacted_text,
    replace_privacy_spans,
)
from src.privacy_auditor import AdversarialMemoryAuditor, aggregate_attacks_by_type
from src.privacy_critic import (
    PrivacyUtilityCritic,
    QuestionAnswerUtilityEvaluator,
)
from src.privacy_schema import (
    AbstractionCandidate,
    AbstractionTrace,
    AttackResult,
    PrivacyItem,
    stable_id,
)
from src.qa_evaluation import (
    build_open_qa_judge,
    build_question_answerer,
    heuristic_answer_from_evidence,
)

SUPPORTED_METHODS = {
    "raw",
    "complete",
    "generic",
    "type_specific",
    "pma_oracle",
    "pma_sft",
}


class PlaceholderState:
    def __init__(self, generic: bool = False):
        self.generic = generic
        self.mapping: dict[tuple[str, str], str] = {}
        self.counts: dict[str, int] = defaultdict(int)

    def placeholder(self, item: PrivacyItem) -> str:
        key = (item.original_text, item.privacy_type)
        if key in self.mapping:
            return self.mapping[key]
        prefix = (
            "MASK"
            if self.generic
            else item.privacy_type.replace(" ", "_").replace("/", "_")
        )
        self.counts[prefix] += 1
        value = f"<{prefix}_{self.counts[prefix]}>"
        self.mapping[key] = value
        return value


def read_users(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    users: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if limit is not None and len(users) >= limit:
                break
            if not line.strip():
                continue
            try:
                users.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return users


def parse_methods(values: Sequence[str]) -> list[str]:
    methods: list[str] = []
    for value in values:
        methods.extend(
            part.strip() for part in value.replace(",", " ").split() if part.strip()
        )
    unknown = set(methods) - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"unsupported methods: {sorted(unknown)}")
    if not methods:
        raise ValueError("at least one method is required")
    return list(dict.fromkeys(methods))


def parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def privacy_items_for_dialogue(
    dialogue: Mapping[str, Any],
    mask_levels: set[str],
) -> list[dict[str, Any]]:
    values = dialogue.get("privacy_info_llm")
    if values is None:
        values = dialogue.get("privacy_info", [])
    return [
        dict(item)
        for item in values or []
        if item.get("original_text") and item.get("privacy_level") in mask_levels
    ]


def candidate_for_baseline(
    method: str,
    dialogue_text: str,
    privacy_items: Sequence[Mapping[str, Any]],
    source_id: str,
    policy: Mapping[str, Any],
    placeholder_state: PlaceholderState | None = None,
) -> AbstractionCandidate:
    parsed = [PrivacyItem.from_dict(item) for item in privacy_items]
    if method == "raw" or not parsed:
        return AbstractionCandidate(
            candidate_id=stable_id(source_id, method, dialogue_text),
            source_id=source_id,
            level="L0",
            public_memory=dialogue_text,
            private_residue=build_private_residue(parsed, policy) if parsed else [],
            metadata={"method": method},
        )
    if method == "complete":
        public_memory, traces = redacted_text(dialogue_text, parsed)
        level = "L5"
    elif method in {"generic", "type_specific"}:
        state = placeholder_state or PlaceholderState(method == "generic")
        replacements = [
            (item.original_text, state.placeholder(item)) for item in parsed
        ]
        public_memory = replace_privacy_spans(dialogue_text, replacements)
        traces = [
            AbstractionTrace(
                raw=item.original_text,
                public_abstraction=replacement,
                reason=(
                    "Generic placeholder hides the value."
                    if method == "generic"
                    else "Typed placeholder hides the value and preserves its type."
                ),
            )
            for item, (_, replacement) in zip(
                parsed,
                replacements,
                strict=True,
            )
        ]
        level = "L4"
    else:
        raise ValueError(f"unsupported baseline method: {method}")
    return AbstractionCandidate(
        candidate_id=stable_id(source_id, method, public_memory),
        source_id=source_id,
        level=level,  # type: ignore[arg-type]
        public_memory=public_memory,
        private_residue=build_private_residue(parsed, policy),
        abstraction_trace=traces,
        metadata={"method": method},
    )


def transform_dialogue(
    method: str,
    dialogue_text: str,
    privacy_items: Sequence[Mapping[str, Any]],
    source_id: str,
    policy: Mapping[str, Any],
    task_family: str,
    questions: Sequence[Mapping[str, Any]],
    *,
    placeholder_state: PlaceholderState | None,
    oracle_abstractor: PrivacyMemoryAbstractor | None,
    sft_abstractor: PrivacyMemoryAbstractor | None,
    oracle_critic: PrivacyUtilityCritic | None,
    utility_threshold: float,
) -> AbstractionCandidate:
    if method in {"raw", "complete", "generic", "type_specific"}:
        return candidate_for_baseline(
            method,
            dialogue_text,
            privacy_items,
            source_id,
            policy,
            placeholder_state,
        )
    if not privacy_items:
        return candidate_for_baseline(
            "raw",
            dialogue_text,
            privacy_items,
            source_id,
            policy,
        )
    if method == "pma_oracle":
        if oracle_abstractor is None or oracle_critic is None:
            raise RuntimeError("pma_oracle requires oracle abstractor and critic")
        candidates = oracle_abstractor.generate_candidates(
            dialogue_text,
            privacy_items,
            task_family,
            source_id=source_id,
        )
        scores = [
            oracle_critic.score_candidate(
                candidate,
                privacy_items,
                questions,
                attack_types=("exact_reconstruction", "attribute_inference"),
            )
            for candidate in candidates
        ]
        return oracle_critic.select_candidate(
            candidates,
            scores,
            utility_threshold,
        )
    if method == "pma_sft":
        if sft_abstractor is None:
            raise RuntimeError("pma_sft requires a trained-model abstractor")
        candidates = sft_abstractor.generate_candidates(
            dialogue_text,
            privacy_items,
            task_family,
            source_id=source_id,
        )
        return next(
            (
                candidate
                for candidate in candidates
                if candidate.metadata.get("generator") == "trained_model"
            ),
            candidates[0],
        )
    raise ValueError(f"unsupported method: {method}")


def _chunks(values: Sequence[dict[str, Any]], size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def evaluate_memory_system(
    *,
    input_path: str,
    users: int | None,
    max_turns: int | None,
    methods: Sequence[str],
    task_family: str,
    policy: Mapping[str, Any],
    utility_threshold: float,
    mask_levels: set[str],
    turns_per_chunk: int,
    memory_backend_factory: Callable[[str, str], MemoryBackend],
    answer_fn: Callable[[str, Mapping[str, Any]], str],
    judge_fn: Callable[[Mapping[str, Any], str, str], float | tuple[float, bool]]
    | None,
    auditor: AdversarialMemoryAuditor,
    oracle_abstractor: PrivacyMemoryAbstractor | None = None,
    sft_abstractor: PrivacyMemoryAbstractor | None = None,
    oracle_critic: PrivacyUtilityCritic | None = None,
    memory_system_name: str = "mem0",
    paper_evidence: bool = True,
    is_mcq: bool = True,
) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "score": 0.0,
            "valid": 0,
            "questions": 0,
            "attacks": [],
            "records": [],
        }
    )
    loaded_users = read_users(input_path, users)
    for user in tqdm(loaded_users, desc="Evaluating privacy-memory methods"):
        user_id = str(user.get("uuid", ""))
        questions = list(user.get("questions", []))
        dialogues = list(user.get("dialogues", []))
        if max_turns is not None:
            dialogues = dialogues[: max_turns * 2]

        for method in methods:
            backend = memory_backend_factory(user_id, method)
            placeholder_state = (
                PlaceholderState(method == "generic")
                if method in {"generic", "type_specific"}
                else None
            )
            transformed_messages: list[dict[str, Any]] = []
            selected_abstractions: list[dict[str, Any]] = []
            attacks: list[AttackResult] = []
            for turn_index, dialogue in enumerate(dialogues):
                content = str(dialogue.get("content", ""))
                if not content:
                    continue
                privacy_items = privacy_items_for_dialogue(dialogue, mask_levels)
                source_id = stable_id(
                    user_id,
                    turn_index,
                    dialogue.get("role", "unknown"),
                )
                candidate = transform_dialogue(
                    method,
                    content,
                    privacy_items,
                    source_id,
                    policy,
                    task_family,
                    questions,
                    placeholder_state=placeholder_state,
                    oracle_abstractor=oracle_abstractor,
                    sft_abstractor=sft_abstractor,
                    oracle_critic=oracle_critic,
                    utility_threshold=utility_threshold,
                )
                transformed_messages.append(
                    {
                        "role": dialogue.get("role", "user"),
                        "content": candidate.public_memory,
                        "timestamp": dialogue.get("date"),
                    }
                )
                selected_abstractions.append(
                    {
                        "turn_index": turn_index,
                        "role": dialogue.get("role", ""),
                        **candidate.to_dict(),
                    }
                )
                if privacy_items:
                    attacks.extend(
                        auditor.audit_candidate(
                            candidate,
                            privacy_items,
                            attack_types=(
                                "exact_reconstruction",
                                "attribute_inference",
                            ),
                        )
                    )

            for chunk in _chunks(
                transformed_messages,
                max(1, turns_per_chunk * 2),
            ):
                timestamp = next(
                    (
                        message.get("timestamp")
                        for message in chunk
                        if message.get("timestamp")
                    ),
                    None,
                )
                backend.add(
                    [
                        {
                            "role": message["role"],
                            "content": message["content"],
                        }
                        for message in chunk
                    ],
                    user_id,
                    timestamp,
                )

            def answer_from_memory(
                _public_memory: str,
                question: Mapping[str, Any],
                current_backend: MemoryBackend = backend,
                current_user_id: str = user_id,
            ) -> str:
                retrieved = current_backend.search(
                    str(question.get("question", "")),
                    current_user_id,
                )
                return answer_fn(retrieved, question)

            utility = QuestionAnswerUtilityEvaluator(
                answer_from_memory,
                judge_fn,
            ).evaluate("", questions)
            method_values = aggregate[method]
            method_values["score"] += sum(record.score for record in utility.records)
            method_values["valid"] += utility.num_valid
            method_values["questions"] += utility.num_questions
            method_values["attacks"].extend(attacks)
            for record in utility.records:
                method_values["records"].append(
                    {
                        "user_id": user_id,
                        "question": record.question,
                        "answer": record.reference_answer,
                        "method": method,
                        "response": record.predicted_answer,
                        "score": record.score,
                        "is_valid": record.valid,
                        "selected_abstractions": selected_abstractions,
                        "attacks": [attack.to_dict() for attack in attacks],
                    }
                )

    methods_output: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for method in methods:
        values = aggregate[method]
        attacks = values["attacks"]
        exact = [
            attack for attack in attacks if attack.attack_type == "exact_reconstruction"
        ]
        attributes = [
            attack for attack in attacks if attack.attack_type == "attribute_inference"
        ]
        methods_output[method] = {
            "utility": {
                "mcq_accuracy": (
                    values["score"] / values["questions"]
                    if values["questions"]
                    else 0.0
                ),
                "total_score": values["score"],
                "total_valid": values["valid"],
                "total_num": values["questions"],
                "proxy": False,
            },
            "privacy": {
                "exact_reconstruction_rate": (
                    sum(attack.success for attack in exact) / len(exact)
                    if exact
                    else 0.0
                ),
                "attribute_inference_rate": (
                    sum(attack.success for attack in attributes) / len(attributes)
                    if attributes
                    else None
                ),
                "num_attacks": len(attacks),
                "per_type": aggregate_attacks_by_type(attacks),
                "proxy": False,
            },
        }
        all_records.extend(values["records"])

    return {
        "run": {
            "dataset": input_path,
            "memory_system": memory_system_name,
            "num_users": len(loaded_users),
            "is_mcq": is_mcq,
            "task_family": task_family,
            "mask_levels": sorted(mask_levels),
            "methods": list(methods),
            "paper_evidence": paper_evidence,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "methods": methods_output,
        "records": all_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PMA and masking baselines through a real memory loop."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--users", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw", "complete", "type_specific", "pma_oracle"],
    )
    parser.add_argument("--task-family", default="general")
    parser.add_argument("--policy-config", default=None)
    parser.add_argument("--utility-threshold", type=float, default=0.85)
    parser.add_argument(
        "--mask-levels",
        nargs="+",
        default=["PL2", "PL3", "PL4"],
    )
    parser.add_argument("--turns-per-chunk", type=int, default=5)
    parser.add_argument("--is-mcq", type=parse_bool, default=True)
    parser.add_argument(
        "--memory-backend",
        choices=["mem0", "in_memory"],
        default="mem0",
    )
    parser.add_argument(
        "--config",
        default="evaluation/eval_config.yaml",
    )
    parser.add_argument("--pma-model-path", default="")
    parser.add_argument(
        "--pma-model-revision",
        default="main",
        help="Prefer an immutable Hugging Face commit hash.",
    )
    parser.add_argument(
        "--auditor-backend",
        choices=["heuristic", "llm"],
        default="llm",
    )
    parser.add_argument(
        "--ablation",
        choices=["none", "no_task_family", "no_policy"],
        default="none",
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Use deterministic local helpers and mark output as non-paper evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    if "pma_sft" in methods and not args.pma_model_path:
        raise ValueError("--pma-model-path is required when evaluating pma_sft")
    if args.ci_mode and args.memory_backend != "in_memory":
        raise ValueError("--ci-mode requires --memory-backend in_memory")
    if args.ci_mode and "pma_sft" in methods:
        raise ValueError("--ci-mode cannot evaluate a real pma_sft checkpoint")
    config: dict[str, Any] = {}
    if not args.ci_mode:
        with open(args.config, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    policy = load_abstraction_policy(args.policy_config)
    task_family = "general" if args.ablation == "no_task_family" else args.task_family
    if args.ablation == "no_policy":
        policy["type_overrides"] = {}

    if args.ci_mode:
        answer_fn = heuristic_answer_from_evidence
        judge_fn = None
        auditor_backend = "heuristic"
        auditor_completion = None
    else:
        answer_completion = build_completion_from_config(config, "answer_llm")
        judge_completion = build_completion_from_config(config, "judgment_llm")
        answer_fn = build_question_answerer(answer_completion)
        judge_fn = build_open_qa_judge(judge_completion)
        auditor_backend = args.auditor_backend
        auditor_completion = None
        if auditor_backend == "llm":
            section = "attack_llm" if "attack_llm" in config else "privacy_llm"
            auditor_completion = build_completion_from_config(config, section)
    auditor = AdversarialMemoryAuditor(
        backend=auditor_backend,
        completion_fn=auditor_completion,
    )
    oracle_completion = None
    if "pma_oracle" in methods and not args.ci_mode:
        oracle_completion = build_completion_from_config(config, "privacy_llm")
    oracle_abstractor = (
        PrivacyMemoryAbstractor(
            policy,
            "heuristic" if args.ci_mode else "oracle_prompt",
            completion_fn=oracle_completion,
            fallback_on_error=False,
        )
        if "pma_oracle" in methods
        else None
    )
    oracle_critic = (
        PrivacyUtilityCritic(
            auditor=auditor,
            utility_threshold=args.utility_threshold,
            utility_evaluator=QuestionAnswerUtilityEvaluator(
                answer_fn,
                judge_fn,
            ),
        )
        if "pma_oracle" in methods
        else None
    )
    sft_abstractor = (
        PrivacyMemoryAbstractor(
            policy,
            "trained_model",
            model_name_or_path=args.pma_model_path,
            model_revision=args.pma_model_revision,
            fallback_on_error=False,
        )
        if "pma_sft" in methods
        else None
    )

    output_dir = Path(args.output).resolve().parent
    memory_root = output_dir / "mem0_runtime"
    if args.memory_backend == "mem0":
        memory_root.mkdir(parents=True, exist_ok=True)

        def backend_factory(user_id: str, method: str) -> MemoryBackend:
            path = memory_root / method / user_id
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            return Mem0Backend(user_id, str(path), config)

        memory_system_name = "mem0"
    else:

        def backend_factory(_user_id: str, _method: str) -> MemoryBackend:
            return InMemoryBackend()

        memory_system_name = "in_memory_ci"

    result = evaluate_memory_system(
        input_path=args.input,
        users=args.users,
        max_turns=args.max_turns,
        methods=methods,
        task_family=task_family,
        policy=policy,
        utility_threshold=args.utility_threshold,
        mask_levels=set(args.mask_levels),
        turns_per_chunk=args.turns_per_chunk,
        memory_backend_factory=backend_factory,
        answer_fn=answer_fn,
        judge_fn=judge_fn,
        auditor=auditor,
        oracle_abstractor=oracle_abstractor,
        sft_abstractor=sft_abstractor,
        oracle_critic=oracle_critic,
        memory_system_name=memory_system_name,
        paper_evidence=(
            args.memory_backend == "mem0"
            and not args.ci_mode
            and auditor_backend == "llm"
        ),
        is_mcq=args.is_mcq,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Wrote PMA evaluation to {args.output}")


if __name__ == "__main__":
    main()
