import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

import json_repair
from openai import OpenAI
from tqdm import tqdm

from evaluation.metric import evaluate_privacy
from evaluation.utils import _load_prompt
from src.privacy_masking import validate_privacy_items

PRIVACY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "original_text": {"type": "string"},
            "privacy_type": {"type": "string"},
            "privacy_level": {
                "type": "string",
                "enum": ["PL2", "PL3", "PL4"],
            },
        },
        "required": ["original_text", "privacy_type", "privacy_level"],
        "additionalProperties": False,
    },
}


def build_vllm_writer(model_path: str, revision: Optional[str] = None) -> Callable[[str, str], str]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError:  # vLLM >= 0.10 uses guided decoding APIs instead.
        StructuredOutputsParams = None

    is_local_model = Path(model_path).expanduser().exists()
    if not is_local_model and not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
        raise ValueError(
            "--revision must be an immutable 40-character commit SHA "
            "when --model is a Hugging Face repository ID"
        )
    # Remote model identifiers are accepted only with an immutable commit SHA.
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        model_path,
        revision=revision,
        local_files_only=is_local_model,
    )
    sampling_kwargs = {
        "temperature": 0.1,
        "top_p": 0.1,
        "repetition_penalty": 1.05,
        "max_tokens": 6144,
    }
    if StructuredOutputsParams is not None:
        sampling_kwargs["structured_outputs"] = StructuredOutputsParams(json=PRIVACY_SCHEMA)
    sampling_params = SamplingParams(**sampling_kwargs)
    model = LLM(
        model=model_path,
        revision=revision,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=0.9,
    )

    def writer(system_prompt: str, query: str) -> str:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": system_prompt + query}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        output = model.generate([text], sampling_params)[0]
        return output.outputs[0].text.strip()

    return writer


def build_openai_writer(model: str, base_url: str, api_key: str) -> Callable[[str, str], str]:
    client = OpenAI(base_url=base_url, api_key=api_key or "local")

    def writer(system_prompt: str, query: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": system_prompt + query}],
            temperature=0.1,
            top_p=0.1,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Privacy model returned empty content")
        return content.strip()

    return writer


def run_evaluation(
    input_path: Path,
    output_path: Path,
    metrics_path: Path,
    writer: Callable[[str, str], str],
    embedding_client: Optional[OpenAI] = None,
    embedding_model: str = "text-embedding-3-small",
) -> None:
    system_prompt = _load_prompt("prompts/extract_privacy.txt")
    records = []
    product_metrics = []
    mean_metrics = []
    failures = []

    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(tqdm(source, desc="Users"), 1):
            if not line.strip():
                continue
            data = json.loads(line)
            saved = {
                "uuid": data.get("uuid"),
                "metadata": data.get("metadata"),
                "dialogues": [],
                "questions": data.get("questions", []),
            }
            real_name = data.get("metadata", {}).get("user_name", "")
            for message_index, dialogue in enumerate(data.get("dialogues", [])):
                annotated = copy.deepcopy(dialogue)
                current_input = {
                    "role": dialogue.get("role", "user"),
                    "content": dialogue.get("content", ""),
                }
                reference = validate_privacy_items(
                    dialogue.get("privacy_info", []),
                    dialogue_text=current_input["content"],
                    strict=False,
                )
                try:
                    raw_prediction = writer(
                        system_prompt.format(real_name=real_name),
                        json.dumps(current_input, ensure_ascii=False),
                    )
                    prediction = validate_privacy_items(
                        json_repair.loads(raw_prediction),
                        dialogue_text=current_input["content"],
                        strict=True,
                    )
                except Exception as exc:
                    prediction = []
                    failures.append(
                        {
                            "line": line_number,
                            "message_index": message_index,
                            "error": str(exc),
                        }
                    )
                annotated["privacy_info_llm"] = prediction
                product_metrics.append(
                    evaluate_privacy(
                        [current_input],
                        prediction,
                        reference,
                        mode="product",
                        embedding_client=embedding_client,
                        embedding_model=embedding_model,
                    )
                )
                mean_metrics.append(
                    evaluate_privacy(
                        [current_input],
                        prediction,
                        reference,
                        mode="mean",
                        embedding_client=embedding_client,
                        embedding_model=embedding_model,
                    )
                )
                saved["dialogues"].append(annotated)
            records.append(saved)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "false_prediction_count": len(failures),
                "type_matching": "embedding" if embedding_client else "exact",
                "embedding_model": embedding_model if embedding_client else None,
                "failures": failures,
                "product": product_metrics,
                "mean": mean_metrics,
                "product_macro": _macro_average(product_metrics),
                "mean_macro": _macro_average(mean_metrics),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _macro_average(records: list[dict]) -> dict:
    if not records:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        metric: sum(record["overall"][metric] for record in records) / len(records)
        for metric in ("precision", "recall", "f1")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a privacy extractor")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--backend", choices=("vllm", "openai"), default="vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--revision",
        help="Immutable Hugging Face commit SHA; required for remote model IDs",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-api-key", default="")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    writer = (
        build_vllm_writer(args.model, args.revision)
        if args.backend == "vllm"
        else build_openai_writer(args.model, args.base_url, args.api_key)
    )
    embedding_client = (
        OpenAI(
            base_url=args.embedding_base_url,
            api_key=args.embedding_api_key or "local",
        )
        if args.embedding_base_url
        else None
    )
    start = time.time()
    run_evaluation(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.metrics_output.expanduser().resolve(),
        writer,
        embedding_client=embedding_client,
        embedding_model=args.embedding_model,
    )
    print(f"Completed in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
