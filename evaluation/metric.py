import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


def mixed_tokenize(text: str) -> List[str]:
    cj_range = r"\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u30fb"
    pattern = re.compile(rf"[{cj_range}]|[^{cj_range}\s]+")
    return [match.group() for match in pattern.finditer(text)]


def longest_common_contiguous(tokens1: Sequence[str], tokens2: Sequence[str]) -> int:
    previous = [0] * (len(tokens2) + 1)
    max_len = 0
    for token1 in tokens1:
        current = [0] * (len(tokens2) + 1)
        for j, token2 in enumerate(tokens2, 1):
            if token1 == token2:
                current[j] = previous[j - 1] + 1
                max_len = max(max_len, current[j])
        previous = current
    return max_len


def normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return list(vector)
    return [value / norm for value in vector]


def build_embedding_cache(
    pred_list: List[Dict[str, Any]],
    ref_list: List[Dict[str, Any]],
    client: Optional[OpenAI] = None,
    model: str = EMBEDDING_MODEL,
) -> Dict[str, List[float]]:
    """Build type embeddings only when an explicit client is supplied."""
    if client is None:
        return {}
    all_types = sorted(
        {item["privacy_type"] for item in [*pred_list, *ref_list] if item.get("privacy_type")}
    )
    if not all_types:
        return {}
    response = client.embeddings.create(model=model, input=all_types)
    if len(all_types) != len(response.data):
        raise ValueError("Embedding response size does not match the request")
    return {text: normalize(response.data[index].embedding) for index, text in enumerate(all_types)}


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Embedding vectors must have the same dimensions")
    dot = sum(v1[index] * v2[index] for index in range(len(v1)))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


NONE_STRICT_TYPES = {
    "Communication Content",
    "Accommodation Record",
    "Accommodation Records",
    "Company/Organization Info",
    "Credit Report",
    "Credit Reports",
    "Debt/Loan",
    "Debt/Loan Info",
    "Document Info",
    "Document Scans",
    "Financial",
    "Itinerary/Trajectory",
    "Job Intent/Status",
    "Judicial Record",
    "Medical Health",
    "Medical Record",
    "Minor Information",
    "Minor Info",
    "Political Views/Stance",
    "Relationship Info",
    "Sensitive Identity",
    "Transaction Record",
    "Vulnerability Details",
}


def score_original_text(pred_text: str, ref_text: str, full_dialogue: str, ref_type: str) -> float:
    if not pred_text or pred_text not in full_dialogue:
        return 0.0
    if pred_text == ref_text:
        return 1.0
    if ref_type not in NONE_STRICT_TYPES:
        return 0.0
    pred_tokens = mixed_tokenize(pred_text)
    ref_tokens = mixed_tokenize(ref_text)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = longest_common_contiguous(pred_tokens, ref_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_privacy_level(pred_level: str, ref_level: str) -> float:
    return 1.0 if pred_level == ref_level else 0.0


def score_privacy_type(
    pred_type: str,
    ref_type: str,
    embedding_cache: Dict[str, List[float]],
) -> float:
    if pred_type == ref_type:
        return 1.0
    if pred_type not in embedding_cache or ref_type not in embedding_cache:
        return 0.0
    return cosine_similarity(
        embedding_cache[pred_type],
        embedding_cache[ref_type],
    )


def _maximum_weight_assignment(weights: List[List[float]]) -> List[Tuple[int, int]]:
    """Return a maximum-weight one-to-one assignment using Hungarian matching."""
    if not weights or not weights[0]:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    size = max(row_count, column_count)
    max_weight = max(max(row) for row in weights)
    cost = [
        [
            max_weight - (weights[i][j] if i < row_count and j < column_count else 0.0)
            for j in range(size)
        ]
        for i in range(size)
    ]

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        min_values = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < min_values[column]:
                    min_values[column] = current
                    way[column] = column0
                if min_values[column] < delta:
                    delta = min_values[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = []
    for column in range(1, size + 1):
        row = p[column] - 1
        target_column = column - 1
        if row < row_count and target_column < column_count:
            assignment.append((row, target_column))
    return assignment


def match_items(
    pred_list: List[Dict[str, Any]],
    ref_list: List[Dict[str, Any]],
    full_dialogue: str,
    embedding_cache: Dict[str, List[float]],
    mode: str = "product",
):
    if mode not in {"product", "mean"}:
        raise ValueError("mode must be 'product' or 'mean'")
    if not pred_list:
        return [], {i: 0.0 for i in range(len(ref_list))}, []
    if not ref_list:
        return [0.0] * len(pred_list), {}, [(0.0, 0.0, 0.0)] * len(pred_list)

    score_matrix: List[List[float]] = []
    subscore_matrix = []
    for prediction in pred_list:
        score_row = []
        subscore_row = []
        for reference in ref_list:
            text_score = score_original_text(
                prediction.get("original_text", ""),
                reference.get("original_text", ""),
                full_dialogue,
                reference.get("privacy_type", ""),
            )
            level_score = score_privacy_level(
                prediction.get("privacy_level", ""),
                reference.get("privacy_level", ""),
            )
            type_score = score_privacy_type(
                prediction.get("privacy_type", ""),
                reference.get("privacy_type", ""),
                embedding_cache,
            )
            if text_score == 0:
                total = 0.0
            elif mode == "product":
                total = text_score * level_score * type_score
            else:
                total = (text_score + level_score + type_score) / 3
            score_row.append(total)
            subscore_row.append((text_score, level_score, type_score))
        score_matrix.append(score_row)
        subscore_matrix.append(subscore_row)

    pred_scores = [0.0] * len(pred_list)
    ref_scores = {i: 0.0 for i in range(len(ref_list))}
    subscore_records = []
    for pred_index, ref_index in _maximum_weight_assignment(score_matrix):
        total = score_matrix[pred_index][ref_index]
        pred_scores[pred_index] = total
        ref_scores[ref_index] = total
        subscore_records.append(subscore_matrix[pred_index][ref_index])

    target_size = max(len(pred_list), len(ref_list))
    subscore_records.extend([(0.0, 0.0, 0.0)] * (target_size - len(subscore_records)))
    return pred_scores, ref_scores, subscore_records


def compute_metrics(pred_scores, ref_scores):
    precision = sum(pred_scores) / len(pred_scores) if pred_scores else 0.0
    recall = sum(ref_scores.values()) / len(ref_scores) if ref_scores else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_privacy(
    messages: List[Dict[str, Any]],
    pred_list: List[Dict[str, Any]],
    ref_list: List[Dict[str, Any]],
    mode: str = "product",
    embedding_client: Optional[OpenAI] = None,
    embedding_model: str = EMBEDDING_MODEL,
):
    if not pred_list and not ref_list:
        perfect = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        return {
            "overall": perfect.copy(),
            "per_level_metrics": {level: perfect.copy() for level in ("PL2", "PL3", "PL4")},
            "mean_subscores": {
                "original_text_mean": 1.0,
                "privacy_level_mean": 1.0,
                "privacy_type_mean": 1.0,
            },
        }

    full_dialogue = "\n".join(message.get("content", "") for message in messages)
    embedding_cache = build_embedding_cache(
        pred_list,
        ref_list,
        client=embedding_client,
        model=embedding_model,
    )
    pred_scores, ref_scores, subscore_records = match_items(
        pred_list,
        ref_list,
        full_dialogue,
        embedding_cache,
        mode,
    )
    precision, recall, f1 = compute_metrics(pred_scores, ref_scores)

    per_level_metrics = {}
    for level in ("PL2", "PL3", "PL4"):
        level_predictions = [
            prediction for prediction in pred_list if prediction.get("privacy_level") == level
        ]
        level_references = [
            reference for reference in ref_list if reference.get("privacy_level") == level
        ]
        if not level_predictions and not level_references:
            per_level_metrics[level] = {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
            continue
        level_pred_scores, level_ref_scores, _ = match_items(
            level_predictions,
            level_references,
            full_dialogue,
            embedding_cache,
            mode,
        )
        level_precision, level_recall, level_f1 = compute_metrics(
            level_pred_scores, level_ref_scores
        )
        per_level_metrics[level] = {
            "precision": level_precision,
            "recall": level_recall,
            "f1": level_f1,
        }

    denominator = len(subscore_records)
    if denominator:
        mean_text = sum(record[0] for record in subscore_records) / denominator
        mean_level = sum(record[1] for record in subscore_records) / denominator
        mean_type = sum(record[2] for record in subscore_records) / denominator
    else:
        mean_text = mean_level = mean_type = 0.0

    return {
        "overall": {"precision": precision, "recall": recall, "f1": f1},
        "per_level_metrics": per_level_metrics,
        "mean_subscores": {
            "original_text_mean": mean_text,
            "privacy_level_mean": mean_level,
            "privacy_type_mean": mean_type,
        },
    }
