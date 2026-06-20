from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import yaml
from tqdm import tqdm

from src.alias_router import ScopedAliasRouter
from src.context_minimizer import ContextMinimizer, ContextMinimizerConfig
from src.leakage_auditor import AuditThresholds, LeakageAuditor
from src.policy import PrivacyPolicy, RoutingContext
from src.privacy_masking import validate_privacy_items
from src.provenance import ProvenanceStore
from src.public_memory_compiler import CompiledMemory, PublicMemoryCompiler
from src.sufficiency_selector import SelectorConfig, SufficiencySelector
from src.utility_auditor import UtilityProxyAuditor


def _annotation_items(message: Mapping, source: str) -> List[dict]:
    key = "privacy_info_llm" if source == "model" else "privacy_info"
    if key not in message:
        raise KeyError(
            f"Missing {key}. Generate model annotations first or use "
            "--annotation-source oracle explicitly."
        )
    return message[key]


def _load_public_memory_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source) or {}


def _build_compiler(config: dict, state_dir: Path) -> PublicMemoryCompiler:
    public_config = config["public_memory"]
    selector_config = public_config.get("selector", {})
    minimizer_config = public_config.get("context_minimizer", {})
    return PublicMemoryCompiler(
        policy=PrivacyPolicy.from_dict(config),
        alias_router=ScopedAliasRouter(
            str(state_dir / "aliases.db"),
            key_path=str(state_dir / "aliases.key"),
        ),
        provenance_store=ProvenanceStore(str(state_dir / "provenance.db")),
        selector=SufficiencySelector(
            SelectorConfig(
                utility_floor=float(selector_config.get("utility_floor", 0.75)),
                max_leakage=float(selector_config.get("max_leakage", 0.35)),
                max_public_tokens=int(selector_config.get("max_public_tokens", 128)),
            )
        ),
        context_minimizer=ContextMinimizer(
            ContextMinimizerConfig(
                enabled=bool(minimizer_config.get("enabled", True)),
                target_public_ratio=float(minimizer_config.get("target_public_ratio", 0.65)),
                max_public_tokens=int(minimizer_config.get("max_public_tokens", 128)),
                min_public_tokens=int(minimizer_config.get("min_public_tokens", 8)),
            )
        ),
    )


def _build_auditor(config: dict, enforce_memory_reduction: bool = True) -> LeakageAuditor:
    budget = config["public_memory"].get("leakage_budget", {})
    return LeakageAuditor(
        AuditThresholds(
            exact_recovery=float(budget.get("exact_recovery", 0.01)),
            cross_scope_linkability=float(budget.get("cross_scope_linkability", 0.01)),
            pl4_public_retention=float(budget.get("pl4_public_retention", 0.0)),
            minimum_token_reduction=(
                float(budget.get("minimum_token_reduction", 0.30))
                if enforce_memory_reduction
                else -1.0
            ),
        )
    )


def _serialize_record(record: CompiledMemory) -> dict:
    return {
        "user_id": record.user_id,
        "message_id": record.message_id,
        "source_fingerprint": record.source_fingerprint,
        "public_text": record.public_text,
        "policy_version": record.policy_version,
        "source_tokens": record.source_tokens,
        "public_tokens": record.public_tokens,
        "token_reduction": record.token_reduction,
        "items": [
            {
                "source_item_index": item.source_item_index,
                "source_fingerprint": item.source_fingerprint,
                "privacy_level": item.privacy_level,
                "privacy_type": item.privacy_type,
                "route_action": item.decision.action.value,
                "rule_id": item.decision.rule_id,
                "reason": item.decision.reason,
                "representation_type": item.representation_type,
                "public_value": item.public_value,
                "utility_score": item.utility_score,
                "leakage_score": item.leakage_score,
                "alias_scope": (item.alias_scope.value if item.alias_scope else None),
                "scope_id": item.scope_id,
                "provenance_id": item.provenance_id,
            }
            for item in record.items
        ],
    }


def _compile_cloud_field(
    text: str,
    privacy_items: Iterable[Mapping[str, str]],
    compiler: PublicMemoryCompiler,
    context: RoutingContext,
) -> tuple[str, CompiledMemory, List[dict]]:
    applicable = [
        dict(item)
        for item in privacy_items
        if item.get("original_text") and item["original_text"] in text
    ]
    compiled = compiler.compile(
        message_text=text,
        privacy_items=applicable,
        context=context,
        strict=True,
    )
    return compiled.public_text, compiled, applicable


def _sanitize_questions(
    questions: Iterable[Mapping],
    privacy_items: Iterable[Mapping[str, str]],
    compiler: PublicMemoryCompiler,
    user_id: str,
    audit_records: List[CompiledMemory],
    audit_source_items: Dict[str, List[dict]],
) -> List[dict]:
    sanitized_questions = []
    fields = ("question", "answer", "evidence")
    for question_index, question in enumerate(questions):
        sanitized = {
            key: value
            for key, value in question.items()
            if key not in fields and key != "all_options"
        }
        for field_name in fields:
            if field_name not in question:
                continue
            field_id = f"{user_id}:question:{question_index}:{field_name}"
            public_text, record, source = _compile_cloud_field(
                str(question[field_name]),
                privacy_items,
                compiler,
                RoutingContext(
                    user_id=user_id,
                    message_id=field_id,
                    message_role="user",
                    turn_id=field_id,
                    session_id=f"{user_id}:evaluation",
                    task_id=f"{user_id}:question:{question_index}",
                ),
            )
            sanitized[field_name] = public_text
            audit_records.append(record)
            audit_source_items[field_id] = source

        sanitized_options = []
        for option_index, option in enumerate(question.get("all_options", [])):
            field_id = f"{user_id}:question:{question_index}:option:{option_index}"
            public_text, record, source = _compile_cloud_field(
                str(option),
                privacy_items,
                compiler,
                RoutingContext(
                    user_id=user_id,
                    message_id=field_id,
                    message_role="user",
                    turn_id=field_id,
                    session_id=f"{user_id}:evaluation",
                    task_id=f"{user_id}:question:{question_index}",
                ),
            )
            sanitized_options.append(public_text)
            audit_records.append(record)
            audit_source_items[field_id] = source
        if "all_options" in question:
            sanitized["all_options"] = sanitized_options
        sanitized_questions.append(sanitized)
    return sanitized_questions


def compile_dataset(
    input_path: Path,
    output_path: Path,
    metrics_path: Path,
    state_dir: Path,
    config: dict,
    annotation_source: str,
    user_limit: Optional[int] = None,
    turns_per_session: int = 20,
    turns_per_task: int = 5,
    exact_required_types: Iterable[str] = (),
    consented_reversible_types: Iterable[str] = (),
    cloud_safe_dataset_path: Optional[Path] = None,
) -> dict:
    state_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if cloud_safe_dataset_path:
        cloud_safe_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[CompiledMemory] = []
    source_items: Dict[str, List[dict]] = {}
    route_counts = Counter()
    representation_counts = Counter()
    invalid_messages = []
    invalid_annotations = []
    questions_by_user = {}
    privacy_items_by_user: Dict[str, List[dict]] = {}
    cloud_artifact_records: List[CompiledMemory] = []
    cloud_artifact_source_items: Dict[str, List[dict]] = {}

    with _build_compiler(config, state_dir) as compiler:
        with (
            input_path.open(encoding="utf-8") as source,
            output_path.open("w", encoding="utf-8") as destination,
            (
                cloud_safe_dataset_path.open("w", encoding="utf-8")
                if cloud_safe_dataset_path
                else nullcontext()
            ) as cloud_safe_destination,
        ):
            for user_index, line in enumerate(tqdm(source, desc="Users")):
                if user_limit is not None and user_index >= user_limit:
                    break
                if not line.strip():
                    continue
                user = json.loads(line)
                user_id = user["uuid"]
                questions_by_user[user_id] = user.get("questions", [])
                dialogues = user.get("dialogues", [])
                compilable_message_indexes = set()
                known_items_by_key = {}
                for message_index, message in enumerate(dialogues):
                    message_id = f"{user_id}:message:{message_index}"
                    try:
                        raw_items = _annotation_items(message, annotation_source)
                        message_items = validate_privacy_items(
                            raw_items,
                            dialogue_text=message.get("content", ""),
                            strict=False,
                        )
                    except Exception as exc:
                        invalid_messages.append(
                            {
                                "message_id": message_id,
                                "error": str(exc),
                            }
                        )
                        continue
                    compilable_message_indexes.add(message_index)
                    if len(message_items) != len(raw_items):
                        invalid_annotations.append(
                            {
                                "message_id": message_id,
                                "invalid_item_count": len(raw_items) - len(message_items),
                            }
                        )
                    for item in message_items:
                        key = (
                            item["original_text"],
                            item["privacy_type"],
                            item["privacy_level"],
                        )
                        known_items_by_key[key] = item

                user_name = str(user.get("metadata", {}).get("user_name", "")).strip()
                if user_name:
                    known_items_by_key.setdefault(
                        (user_name, "Real Name", "PL2"),
                        {
                            "original_text": user_name,
                            "privacy_type": "Real Name",
                            "privacy_level": "PL2",
                        },
                    )
                known_items = list(known_items_by_key.values())
                privacy_items_by_user[user_id] = known_items
                sanitized_dialogues = []
                for message_index, message in enumerate(dialogues):
                    if message_index not in compilable_message_indexes:
                        continue
                    message_id = f"{user_id}:message:{message_index}"
                    items = [
                        item
                        for item in known_items
                        if item["original_text"] in message.get("content", "")
                    ]
                    try:
                        compiled = compiler.compile(
                            message_text=message.get("content", ""),
                            privacy_items=items,
                            context=RoutingContext(
                                user_id=user_id,
                                message_id=message_id,
                                message_role=message.get("role", "user"),
                                turn_id=f"{user_id}:turn:{message_index}",
                                session_id=(
                                    f"{user_id}:session:{message_index // turns_per_session}"
                                ),
                                task_id=(f"{user_id}:task:{message_index // turns_per_task}"),
                                exact_required_types=set(exact_required_types),
                                consented_reversible_types=set(consented_reversible_types),
                            ),
                            strict=True,
                        )
                    except Exception as exc:
                        invalid_messages.append(
                            {
                                "message_id": message_id,
                                "error": str(exc),
                            }
                        )
                        continue

                    records.append(compiled)
                    source_items[message_id] = items
                    route_counts.update(item.decision.action.value for item in compiled.items)
                    representation_counts.update(
                        item.representation_type for item in compiled.items
                    )
                    destination.write(
                        json.dumps(_serialize_record(compiled), ensure_ascii=False) + "\n"
                    )
                    sanitized_message = {
                        key: value
                        for key, value in message.items()
                        if key not in {"content", "privacy_info", "privacy_info_llm"}
                    }
                    sanitized_message["content"] = compiled.public_text
                    sanitized_dialogues.append(sanitized_message)

                if cloud_safe_dataset_path:
                    user_alias = f"User-{compiler.alias_router.fingerprint(user_id)[:12]}"
                    sanitized_questions = _sanitize_questions(
                        user.get("questions", []),
                        privacy_items_by_user[user_id],
                        compiler,
                        user_id,
                        cloud_artifact_records,
                        cloud_artifact_source_items,
                    )
                    sanitized_user = {
                        "uuid": user_alias,
                        "metadata": {
                            "user_name": user_alias,
                            "language": user.get("metadata", {}).get("language"),
                        },
                        "dialogues": sanitized_dialogues,
                        "questions": sanitized_questions,
                    }
                    cloud_safe_destination.write(
                        json.dumps(sanitized_user, ensure_ascii=False) + "\n"
                    )

        report = _build_auditor(config).audit(records, source_items)
        utility_proxy = UtilityProxyAuditor().audit(
            records,
            questions_by_user,
            privacy_items_by_user,
            compiler.alias_router,
        )
        cloud_artifact_report = (
            _build_auditor(config, enforce_memory_reduction=False).audit(
                cloud_artifact_records,
                cloud_artifact_source_items,
            )
            if cloud_safe_dataset_path
            else None
        )
    result = {
        "dataset": str(input_path.resolve()),
        "annotation_source": annotation_source,
        "record_count": len(records),
        "invalid_message_count": len(invalid_messages),
        "invalid_messages": invalid_messages,
        "invalid_annotation_count": sum(
            item["invalid_item_count"] for item in invalid_annotations
        ),
        "invalid_annotations": invalid_annotations,
        "route_counts": dict(route_counts),
        "representation_counts": dict(representation_counts),
        "audit": asdict(report),
        "utility_proxy": UtilityProxyAuditor.to_dict(utility_proxy),
        "cloud_safe_dataset": (
            {
                "path": str(cloud_safe_dataset_path.resolve()),
                "query_privacy_source": "known-dialogue-annotation-match",
                "audit": asdict(cloud_artifact_report),
            }
            if cloud_safe_dataset_path and cloud_artifact_report
            else None
        ),
    }
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile and audit minimal public memory")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/public_memory_config.yaml"),
    )
    parser.add_argument(
        "--annotation-source",
        choices=("model", "oracle"),
        default="model",
    )
    parser.add_argument("--user-limit", type=int)
    parser.add_argument("--turns-per-session", type=int, default=20)
    parser.add_argument("--turns-per-task", type=int, default=5)
    parser.add_argument("--exact-required-type", action="append", default=[])
    parser.add_argument("--consented-reversible-type", action="append", default=[])
    parser.add_argument(
        "--cloud-safe-dataset-output",
        type=Path,
        help="optional same-schema JSONL for public-only memory-system QA",
    )
    parser.add_argument(
        "--minimum-token-reduction",
        type=float,
        help="override the configured long-term-memory reduction gate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_public_memory_config(args.config.expanduser().resolve())
    if args.minimum_token_reduction is not None:
        config.setdefault("public_memory", {}).setdefault("leakage_budget", {})[
            "minimum_token_reduction"
        ] = args.minimum_token_reduction
    result = compile_dataset(
        input_path=args.input.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        metrics_path=args.metrics_output.expanduser().resolve(),
        state_dir=args.state_dir.expanduser().resolve(),
        config=config,
        annotation_source=args.annotation_source,
        user_limit=args.user_limit,
        turns_per_session=args.turns_per_session,
        turns_per_task=args.turns_per_task,
        exact_required_types=args.exact_required_type,
        consented_reversible_types=args.consented_reversible_type,
        cloud_safe_dataset_path=(
            args.cloud_safe_dataset_output.expanduser().resolve()
            if args.cloud_safe_dataset_output
            else None
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
