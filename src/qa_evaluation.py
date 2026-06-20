from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from .llm_clients import parse_json_content


def build_question_answerer(
    completion_fn: Callable[[str], str],
) -> Callable[[str, Mapping[str, Any]], str]:
    def answer(public_memory: str, question: Mapping[str, Any]) -> str:
        options = question.get("all_options") or []
        if options:
            prompt = (
                "Answer the multiple-choice question using only the supplied "
                'public memory. Return strict JSON: {"answer": "A|B|C|D", '
                '"reason": "..."}.\n\n'
                f"Public memory:\n{public_memory}\n\n"
                f"Question:\n{question.get('question', '')}\n\n"
                "Options:\n" + "\n".join(str(option) for option in options)
            )
            parsed = parse_json_content(completion_fn(prompt))
            if isinstance(parsed, dict):
                return str(parsed.get("answer", ""))
            return str(parsed)
        prompt = (
            "Answer the question using only the supplied public memory. If the "
            "memory is insufficient, say so explicitly.\n\n"
            f"Public memory:\n{public_memory}\n\n"
            f"Question:\n{question.get('question', '')}"
        )
        return completion_fn(prompt).strip()

    return answer


def build_open_qa_judge(
    completion_fn: Callable[[str], str],
) -> Callable[[Mapping[str, Any], str, str], tuple[float, bool]]:
    def judge(
        question: Mapping[str, Any],
        prediction: str,
        reference: str,
    ) -> tuple[float, bool]:
        prompt = (
            "Evaluate an answer against the reference. Return strict JSON with "
            "judgment equal to correct, partially_correct, or incorrect and a "
            "short reason.\n\n"
            f"Question: {question.get('question', '')}\n"
            f"Reference: {reference}\n"
            f"Answer: {prediction}\n"
        )
        parsed = parse_json_content(completion_fn(prompt))
        if not isinstance(parsed, dict):
            return 0.0, False
        label = str(parsed.get("judgment", "")).casefold()
        if label == "correct":
            return 1.0, True
        if label == "partially_correct":
            return 0.5, True
        if label == "incorrect":
            return 0.0, True
        return 0.0, False

    return judge


def heuristic_answer_from_evidence(
    public_memory: str,
    question: Mapping[str, Any],
) -> str:
    """Deterministic CI helper, not a paper-evidence answer model."""
    answer = str(question.get("answer", ""))
    evidence = question.get("evidence")
    if evidence:
        evidence_text = json.dumps(evidence, ensure_ascii=False)
        if evidence_text and evidence_text.casefold() in public_memory.casefold():
            return answer
    options = question.get("all_options") or []
    for index, option in enumerate(options):
        option_text = re.sub(r"^[A-Da-d][\).:\s-]+", "", str(option)).strip()
        if option_text and option_text.casefold() in public_memory.casefold():
            return "ABCD"[index]
    return ""
