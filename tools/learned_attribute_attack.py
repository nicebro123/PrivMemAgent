from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_SEMANTIC_KEYS = {
    "public_text",
    "content",
    "question",
    "answer",
    "evidence",
    "all_options",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{1,}|\d{2,}|[\u4e00-\u9fff]{1,}")


@dataclass(frozen=True)
class LearnedAttackReport:
    target: str
    artifact_count: int
    user_count: int
    train_user_count: int
    test_user_count: int
    positive_user_count: int
    negative_user_count: int
    applicable: bool
    accuracy: float | None
    auc: float | None
    baseline_accuracy: float | None
    top_positive_tokens: tuple[str, ...]
    predictions: tuple[dict, ...]


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    return records


def _iter_semantic_strings(value: object, key: str | None = None) -> Iterable[str]:
    if isinstance(value, str):
        if key in _SEMANTIC_KEYS:
            yield value
    elif isinstance(value, Mapping):
        for child_key, child in value.items():
            yield from _iter_semantic_strings(child, str(child_key))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _iter_semantic_strings(child, key)


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _source_labels(
    source_records: Sequence[Mapping],
    target_terms: Sequence[str],
    privacy_levels: set[str],
) -> list[int]:
    normalized_terms = [term.casefold() for term in target_terms]
    labels = []
    for user in source_records:
        matched = False
        for message in user.get("dialogues", []) or []:
            for item in message.get("privacy_info", []) or []:
                level = str(item.get("privacy_level", ""))
                if privacy_levels and level not in privacy_levels:
                    continue
                haystack = " ".join(
                    [
                        str(item.get("privacy_type", "")),
                        str(item.get("original_text", "")),
                    ]
                ).casefold()
                if any(term in haystack for term in normalized_terms):
                    matched = True
                    break
            if matched:
                break
        labels.append(1 if matched else 0)
    return labels


def _record_identity(record: Mapping) -> str:
    return str(record.get("user_id") or record.get("uuid") or "").strip()


def _artifact_documents(source_count: int, artifacts: Sequence[Path]) -> list[str]:
    docs = [""] * source_count
    for artifact in artifacts:
        identity_order: list[str] = []
        identity_to_source_index: dict[str, int] = {}
        for record in _load_jsonl(artifact):
            identity = _record_identity(record)
            if identity:
                if identity not in identity_to_source_index:
                    identity_to_source_index[identity] = len(identity_order)
                    identity_order.append(identity)
                source_index = identity_to_source_index[identity]
            else:
                source_index = 0
            if source_index >= source_count:
                continue
            docs[source_index] += "\n" + "\n".join(_iter_semantic_strings(record))
    return docs


def _split_indices(labels: Sequence[int], test_ratio: float) -> tuple[list[int], list[int]]:
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    test_pos = max(1, round(len(positives) * test_ratio)) if positives else 0
    test_neg = max(1, round(len(negatives) * test_ratio)) if negatives else 0
    test = positives[-test_pos:] + negatives[-test_neg:]
    train = [index for index in range(len(labels)) if index not in set(test)]
    return train, test


def _train_multinomial_nb(
    docs: Sequence[str],
    labels: Sequence[int],
    train_indices: Sequence[int],
) -> dict[str, object]:
    class_token_counts = {0: Counter(), 1: Counter()}
    class_doc_counts = {0: 0, 1: 0}
    vocabulary = set()
    for index in train_indices:
        label = labels[index]
        class_doc_counts[label] += 1
        counts = Counter(_tokens(docs[index]))
        class_token_counts[label].update(counts)
        vocabulary.update(counts)
    return {
        "class_token_counts": class_token_counts,
        "class_doc_counts": class_doc_counts,
        "vocabulary": vocabulary,
    }


def _predict_positive_probability(model: Mapping[str, object], doc: str) -> float:
    class_token_counts = model["class_token_counts"]
    class_doc_counts = model["class_doc_counts"]
    vocabulary = model["vocabulary"]
    vocab_size = max(1, len(vocabulary))
    total_docs = sum(class_doc_counts.values())
    counts = Counter(_tokens(doc))
    log_scores = {}
    for label in (0, 1):
        prior = (class_doc_counts[label] + 1) / (total_docs + 2)
        score = math.log(prior)
        total_tokens = sum(class_token_counts[label].values())
        for token, count in counts.items():
            token_count = class_token_counts[label][token]
            score += count * math.log((token_count + 1) / (total_tokens + vocab_size))
        log_scores[label] = score
    max_score = max(log_scores.values())
    exp_pos = math.exp(log_scores[1] - max_score)
    exp_neg = math.exp(log_scores[0] - max_score)
    return exp_pos / (exp_pos + exp_neg)


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def _top_positive_tokens(model: Mapping[str, object], limit: int = 20) -> tuple[str, ...]:
    class_token_counts = model["class_token_counts"]
    vocabulary = model["vocabulary"]
    vocab_size = max(1, len(vocabulary))
    pos_total = sum(class_token_counts[1].values())
    neg_total = sum(class_token_counts[0].values())
    scored = []
    for token in vocabulary:
        pos = (class_token_counts[1][token] + 1) / (pos_total + vocab_size)
        neg = (class_token_counts[0][token] + 1) / (neg_total + vocab_size)
        scored.append((math.log(pos / neg), token))
    return tuple(token for _, token in sorted(scored, reverse=True)[:limit])


def run_learned_attribute_attack(
    source_dataset: Path,
    artifacts: Sequence[Path],
    target: str,
    target_terms: Sequence[str],
    privacy_levels: set[str] | None = None,
    source_user_limit: int | None = None,
    test_ratio: float = 0.4,
) -> LearnedAttackReport:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    source_records = _load_jsonl(source_dataset)
    if source_user_limit is not None:
        source_records = source_records[:source_user_limit]
    levels = privacy_levels if privacy_levels is not None else {"PL3", "PL4"}
    labels = _source_labels(source_records, target_terms, levels)
    docs = _artifact_documents(len(source_records), artifacts)
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    train_indices, test_indices = _split_indices(labels, test_ratio)
    applicable = (
        positive_count >= 2
        and negative_count >= 2
        and any(labels[index] == 1 for index in train_indices)
        and any(labels[index] == 0 for index in train_indices)
        and any(labels[index] == 1 for index in test_indices)
        and any(labels[index] == 0 for index in test_indices)
    )
    if not applicable:
        return LearnedAttackReport(
            target=target,
            artifact_count=len(artifacts),
            user_count=len(source_records),
            train_user_count=len(train_indices),
            test_user_count=len(test_indices),
            positive_user_count=positive_count,
            negative_user_count=negative_count,
            applicable=False,
            accuracy=None,
            auc=None,
            baseline_accuracy=None,
            top_positive_tokens=(),
            predictions=(),
        )
    model = _train_multinomial_nb(docs, labels, train_indices)
    predictions = []
    scores = []
    test_labels = []
    correct = 0
    majority_label = 1 if sum(labels[index] for index in train_indices) >= len(train_indices) / 2 else 0
    baseline_correct = 0
    for index in test_indices:
        score = _predict_positive_probability(model, docs[index])
        predicted = 1 if score >= 0.5 else 0
        label = labels[index]
        scores.append(score)
        test_labels.append(label)
        correct += int(predicted == label)
        baseline_correct += int(majority_label == label)
        predictions.append(
            {
                "source_user_index": index,
                "label": label,
                "score": score,
                "predicted": predicted,
            }
        )
    return LearnedAttackReport(
        target=target,
        artifact_count=len(artifacts),
        user_count=len(source_records),
        train_user_count=len(train_indices),
        test_user_count=len(test_indices),
        positive_user_count=positive_count,
        negative_user_count=negative_count,
        applicable=True,
        accuracy=correct / len(test_indices),
        auc=_auc(test_labels, scores),
        baseline_accuracy=baseline_correct / len(test_indices),
        top_positive_tokens=_top_positive_tokens(model),
        predictions=tuple(predictions),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight held-out attribute attacker against public memory."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--target", required=True, help="label name, e.g. medical")
    parser.add_argument(
        "--target-term",
        action="append",
        required=True,
        help="privacy_type/original_text substring that defines a positive source user",
    )
    parser.add_argument(
        "--privacy-level",
        action="append",
        default=["PL3", "PL4"],
        help="privacy levels used for labels; repeatable",
    )
    parser.add_argument("--source-user-limit", type=int)
    parser.add_argument("--test-ratio", type=float, default=0.4)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_learned_attribute_attack(
        source_dataset=args.source.expanduser().resolve(),
        artifacts=[path.expanduser().resolve() for path in args.artifact],
        target=args.target,
        target_terms=args.target_term,
        privacy_levels=set(args.privacy_level),
        source_user_limit=args.source_user_limit,
        test_ratio=args.test_ratio,
    )
    payload = asdict(report)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
