import argparse
import json
import logging
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path, PurePath
from typing import List, Optional

from mem0 import Memory
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

from evaluation.cloud_safe_guard import enforce_no_mask_input_safety
from evaluation.privacy_masking import (
    PrivacyStore,
    collect_user_privacy_items,
    protect_known_values,
    unmask_dialogue,
)
from evaluation.utils import (
    _load_config,
    _load_prompt,
    call_llm,
    file_sha256,
    summarize_scores_by_question_type,
    verify_mcq_answer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load config and set up environment
# ---------------------------------------------------------------------------
_config = _load_config()
os.environ["OPENAI_API_KEY"] = _config["openai_api_key"]
os.environ["OPENAI_BASE_URL"] = _config["openai_base_url"]

# ---------------------------------------------------------------------------
# Build mem0 config and initialise Memory instance
# ---------------------------------------------------------------------------

_mem0_cfg = _config["mem0_config"]


def _configured_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(_config["_config_dir"]) / candidate).resolve())


_mem0_cfg["db_path"] = _configured_path(_mem0_cfg["db_path"])
_config["privacy"]["db_path"] = _configured_path(_config["privacy"]["db_path"])


def _annotation_source() -> str:
    return os.getenv("MEMPRIVACY_ANNOTATION_SOURCE", "model")


def _build_embedder_config() -> dict:
    embedding_config = _config.get("embedding_model", {})
    provider = embedding_config.get("provider", "openai")
    if provider == "huggingface":
        return {
            "provider": "huggingface",
            "config": {
                "model": embedding_config["model"],
                "embedding_dims": embedding_config.get("dimensions"),
                "model_kwargs": {"device": embedding_config.get("device", "cpu")},
            },
        }
    return {
        "provider": provider,
        "config": {
            "model": embedding_config["model"],
        },
    }


def load_mem0_config(user_id: str, db_path: str):

    mem0_config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": f"{user_id}",
                "path": db_path,
            },
        },
        "embedder": _build_embedder_config(),
        "llm": {
            "provider": "openai",
            "config": {
                "model": _config["memory_llm"]["model"],
                "temperature": _config["memory_llm"]["temperature"],
                "max_tokens": _config["memory_llm"]["max_tokens"],
            },
        },
    }

    return mem0_config


# ---------------------------------------------------------------------------
# Shared retry decorator built from mem0 retry settings
# ---------------------------------------------------------------------------
_mem0_retry = retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(
        min=_mem0_cfg["wait_time_lower"],
        max=_mem0_cfg["wait_time_upper"],
    ),
    stop=stop_after_attempt(_mem0_cfg["retry_times"]),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@_mem0_retry
def add_message(
    memory: Memory,
    messages: List[dict],
    user_id: str,
    timestamp: Optional[str] = None,
) -> None:
    """Add messages to mem0 memory for the given user."""
    memory.add(
        messages=messages,
        user_id=user_id,
        metadata={"timestamp": timestamp} if timestamp else None,
    )


@_mem0_retry
def search_memory(
    memory: Memory,
    query: str,
    user_id: str,
    limit: int = 20,
) -> str:
    """Search mem0 memory and return matching entries as a newline-joined string."""
    search_results = memory.search(
        query=query,
        user_id=user_id,
        limit=limit,
    )

    memories = [f"  - {item['memory']}" for item in search_results["results"]]

    return "\n".join(memories)


def chunk_dialogues(dialogues: List[dict], turns_per_chunk: int = 1):
    """
    Split dialogues into chunks, each containing `turns_per_chunk` dialogue turns.
    One turn = user + assistant.

    Handles the case where the last turn only has a user message.
    """

    chunks = []
    i = 0
    n = len(dialogues)

    while i < n:
        turns = []
        turn_count = 0

        while i < n and turn_count < turns_per_chunk:
            user_msg = dialogues[i]
            assistant_msg = dialogues[i + 1] if i + 1 < n else None

            turns.append((user_msg, assistant_msg))

            i += 2
            turn_count += 1

        chunks.append(turns)

    return chunks


def process_single_user_data_add_memory(
    user_data: dict,
    is_mask: bool,
    mask_level: list[str],
    is_mcq: bool,  # are Multiple-choice questions
    turns_per_chunk: int = 1,
    mask_mode: str = "type_specific",  # "type_specific" or "generic" or "complete"
):

    user_id = user_data["uuid"]
    sub_path = (
        f"Mem0_{_annotation_source()}_{'mask' if is_mask else 'unmask'}_"
        f"{''.join(mask_level)}_{mask_mode}_{'mcq' if is_mcq else 'qa'}"
    )
    os.makedirs(os.path.join(_mem0_cfg["db_path"], sub_path), exist_ok=True)
    mem0_db_path = os.path.join(_mem0_cfg["db_path"], sub_path, user_id)
    os.makedirs(mem0_db_path, exist_ok=True)
    mem0_config = load_mem0_config(user_id, mem0_db_path)
    memory = Memory.from_config(mem0_config)

    if is_mask and mask_mode != "complete":
        privacy_sub_path = os.path.join(
            _config["privacy"]["db_path"],
            f"Mem0_{_annotation_source()}_{'mask' if is_mask else 'unmask'}_"
            f"{''.join(mask_level)}_{mask_mode}_{'mcq' if is_mcq else 'qa'}",
        )
        os.makedirs(privacy_sub_path, exist_ok=True)
        privacy_db_path = os.path.join(privacy_sub_path, f"{user_id}.db")
        privacy_store = PrivacyStore(
            db_path=os.path.abspath(privacy_db_path),
            mask_mode=mask_mode,
            namespace=user_id,
        )

    known_privacy_items = collect_user_privacy_items(user_data) if is_mask else []
    dialogues = user_data["dialogues"]
    dialogue_chunks = chunk_dialogues(dialogues, turns_per_chunk)

    for chunk in tqdm(dialogue_chunks, desc="Processing message chunks"):
        pure_messages = []
        timestamp = None

        for user_msg, assistant_msg in chunk:
            if timestamp is None:
                timestamp = user_msg.get("date", None)

            user_content = user_msg["content"]

            if is_mask:
                user_content = protect_known_values(
                    user_content,
                    known_privacy_items,
                    mask_level,
                    mask_mode,
                    privacy_store if mask_mode != "complete" else None,
                )

            pure_messages.append({"role": user_msg["role"], "content": user_content})

            if assistant_msg:
                assistant_content = assistant_msg["content"]

                if is_mask:
                    assistant_content = protect_known_values(
                        assistant_content,
                        known_privacy_items,
                        mask_level,
                        mask_mode,
                        privacy_store if mask_mode != "complete" else None,
                    )

                pure_messages.append({"role": assistant_msg["role"], "content": assistant_content})

        add_message(memory, pure_messages, user_id, timestamp)

    print(f"Memory added to {mem0_db_path} successfully")
    if is_mask and mask_mode != "complete":
        print(f"Privacy store added to {privacy_db_path} successfully")
        privacy_store.close()


def process_single_user_data_qa(
    user_data: dict,
    is_mask: bool,
    mask_level: list[str],
    is_mcq: bool,  # are Multiple-choice questions
    turns_per_chunk: int = 1,
    mask_mode: str = "type_specific",  # "type_specific" or "generic" or "complete"
):

    user_id = user_data["uuid"]
    user_name = user_data["metadata"]["user_name"]
    sub_path = (
        f"Mem0_{_annotation_source()}_{'mask' if is_mask else 'unmask'}_"
        f"{''.join(mask_level)}_{mask_mode}_{'mcq' if is_mcq else 'qa'}"
    )
    os.makedirs(os.path.join(_mem0_cfg["db_path"], sub_path), exist_ok=True)
    mem0_db_path = os.path.join(_mem0_cfg["db_path"], sub_path, user_id)
    mem0_config = load_mem0_config(user_id, mem0_db_path)
    memory = Memory.from_config(mem0_config)

    if is_mask and mask_mode != "complete":
        privacy_sub_path = os.path.join(
            _config["privacy"]["db_path"],
            f"Mem0_{_annotation_source()}_{'mask' if is_mask else 'unmask'}_"
            f"{''.join(mask_level)}_{mask_mode}_{'mcq' if is_mcq else 'qa'}",
        )
        os.makedirs(privacy_sub_path, exist_ok=True)
        privacy_db_path = os.path.join(privacy_sub_path, f"{user_id}.db")
        privacy_store = PrivacyStore(
            db_path=os.path.abspath(privacy_db_path),
            mask_mode=mask_mode,
            namespace=user_id,
        )

    results = []
    known_privacy_items = collect_user_privacy_items(user_data) if is_mask else []
    cloud_user_name = (
        protect_known_values(
            user_name,
            known_privacy_items,
            mask_level,
            mask_mode,
            privacy_store if mask_mode != "complete" else None,
        )
        if is_mask
        else user_name
    )
    total_score = 0
    total_valid = 0
    total_num = len(user_data["questions"])
    for question in tqdm(user_data["questions"], desc="Processing questions"):
        query = question["question"]
        answer = question["answer"]
        cloud_query = (
            protect_known_values(
                query,
                known_privacy_items,
                mask_level,
                mask_mode,
                privacy_store if mask_mode != "complete" else None,
            )
            if is_mask
            else query
        )

        memories_text = search_memory(
            memory=memory,
            query=cloud_query,
            user_id=user_id,
            limit=20,
        )

        is_valid = True
        if is_mcq:
            prompt_path = _config["prompts"]["answer_prompt_2"]
            prompt = _load_prompt(prompt_path)
            cloud_options = [
                protect_known_values(
                    option,
                    known_privacy_items,
                    mask_level,
                    mask_mode,
                    privacy_store if mask_mode != "complete" else None,
                )
                if is_mask
                else option
                for option in question["all_options"]
            ]
            query_prompt = prompt.format(
                question=cloud_query,
                user_memories=memories_text,
                options_text="\n".join(cloud_options),
            )

            response = call_llm(
                query=query_prompt,
                llm_type="answer_llm",
                extract_json=True,
                return_parsed_json=True,
            )

            if isinstance(response, dict):
                is_correct, is_valid = verify_mcq_answer(response.get("answer", ""), answer)
                response_content = json.dumps(response, ensure_ascii=False)
            else:
                is_correct = False
                response_content = str(response)
                is_valid = False

            score = 1 if is_correct else 0

        else:
            answer_prompt_path = _config["prompts"]["answer_prompt_1"]
            judge_prompt_path = _config["prompts"]["judge_prompt"]
            answer_prompt = _load_prompt(answer_prompt_path)
            judge_prompt = _load_prompt(judge_prompt_path)

            query_prompt = answer_prompt.format(
                user_name=cloud_user_name,
                user_memories=memories_text,
                question=cloud_query,
            )

            response_content = call_llm(
                query=query_prompt,
                llm_type="answer_llm",
                extract_json=False,
                return_parsed_json=False,
            )

            cloud_reference_answer = (
                protect_known_values(
                    answer,
                    known_privacy_items,
                    mask_level,
                    mask_mode,
                    privacy_store if mask_mode != "complete" else None,
                )
                if is_mask
                else answer
            )
            judge_prompt = judge_prompt.format(
                question=cloud_query,
                reference_answer=cloud_reference_answer,
                response=response_content,
            )

            judgment_response = call_llm(
                query=judge_prompt,
                llm_type="judgment_llm",
                extract_json=True,
                return_parsed_json=True,
            )

            if isinstance(judgment_response, dict):
                result_label = judgment_response.get("judgment", "unknown")
                is_correct = result_label == "correct"
                is_valid = result_label in {"correct", "partially_correct", "incorrect"}
                score = 1 if is_correct else (0.5 if result_label == "partially_correct" else 0)
            else:
                is_correct = False
                score = 0
                is_valid = False

            if is_mask and mask_mode != "complete" and response_content.strip():
                response_content = unmask_dialogue(response_content, privacy_store)

        results.append(
            {
                "user_id": user_id,
                "user_name": user_name,
                "question": query,
                "question_type": question.get("question_type", "Unknown"),
                "answer": answer,
                "response": response_content,
                "score": score,
                "is_valid": is_valid,
            }
        )
        total_score += score
        total_valid += is_valid

    if is_mask and mask_mode != "complete":
        privacy_store.close()
    return results, total_score, total_valid, total_num


def _worker_add_memory(args):
    user_data, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode = args
    return process_single_user_data_add_memory(
        user_data, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode
    )


def process_all_users_data_add_memory(
    user_data_path: str,
    is_mask: bool,
    mask_level: list[str],
    is_mcq: bool,
    turns_per_chunk: int = 1,
    mask_mode: str = "type_specific",
    user_num: int = None,
    num_workers: int = 10,
):

    with open(user_data_path, "r", encoding="utf-8") as f:
        user_data_list = [json.loads(line) for line in f]

    if user_num is not None:
        user_data_list = user_data_list[:user_num]

    sub_path = (
        f"Mem0_{_annotation_source()}_{'mask' if is_mask else 'unmask'}_"
        f"{''.join(mask_level)}_{mask_mode}_{'mcq' if is_mcq else 'qa'}"
    )
    shutil.rmtree(os.path.join(_mem0_cfg["db_path"], sub_path), ignore_errors=True)
    if is_mask and mask_mode != "complete":
        privacy_sub_path = os.path.join(
            _config["privacy"]["db_path"],
            f"Mem0_{_annotation_source()}_mask_{''.join(mask_level)}_{mask_mode}_"
            f"{'mcq' if is_mcq else 'qa'}",
        )
        shutil.rmtree(privacy_sub_path, ignore_errors=True)

    tasks = [(u, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode) for u in user_data_list]

    with Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_worker_add_memory, tasks), total=len(tasks), desc="Adding memory"
        ):
            pass

    print("Finished adding memory!")


def _worker_qa(args):
    user_data, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode = args
    return process_single_user_data_qa(
        user_data, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode
    )


def process_all_users_data_qa(
    user_data_path: str,
    is_mask: bool,
    mask_level: list[str],
    is_mcq: bool,
    turns_per_chunk: int = 1,
    mask_mode: str = "type_specific",
    user_num: int = None,
    num_workers: int = 10,
):

    with open(user_data_path, "r", encoding="utf-8") as f:
        user_data_list = [json.loads(line) for line in f]

    if user_num is not None:
        user_data_list = user_data_list[:user_num]

    tasks = [(u, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode) for u in user_data_list]

    all_results = []
    all_total_score = 0
    all_total_valid = 0
    all_total_num = 0

    with Pool(num_workers) as pool:
        for user_results, total_score, total_valid, total_num in tqdm(
            pool.imap_unordered(_worker_qa, tasks), total=len(tasks), desc="Processing questions"
        ):
            all_results.extend(user_results)
            all_total_score += total_score
            all_total_valid += total_valid
            all_total_num += total_num

    all_accuracy = all_total_score / all_total_num if all_total_num > 0 else 0
    all_accuracy_valid = all_total_score / all_total_valid if all_total_valid > 0 else 0

    output_dict = {
        "run_config": {
            "dataset": str(Path(user_data_path).resolve()),
            "dataset_sha256": file_sha256(user_data_path),
            "annotation_source": _annotation_source(),
            "is_mask": is_mask,
            "mask_level": mask_level,
            "mask_mode": mask_mode,
            "is_mcq": is_mcq,
            "turns_per_chunk": turns_per_chunk,
            "user_num": user_num,
            "num_workers": num_workers,
            "memory_model": _config["memory_llm"]["model"],
        },
        "total_score": all_total_score,
        "total_valid": all_total_valid,
        "total_num": all_total_num,
        "accuracy": all_accuracy,
        "accuracy_valid": all_accuracy_valid,
        "metrics_by_question_type": summarize_scores_by_question_type(all_results),
        "records": all_results,
    }

    user_data_name = PurePath(user_data_path).stem
    output_root = Path(_config["_config_dir"]) / _config["output_path"]
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(
        output_root,
        f"Mem0_{user_data_name}_{_annotation_source()}_"
        f"{'mask' if is_mask else 'unmask'}_{''.join(mask_level)}_{mask_mode}_"
        f"{time.strftime('%Y%m%d%H%M%S')}.json",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)

    print("accuracy:", all_accuracy)
    print(f"Output saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mem0 with privacy masking")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-level", nargs="+", default=["PL2", "PL3", "PL4"])
    parser.add_argument("--mcq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--turns-per-chunk", type=int, default=5)
    parser.add_argument("--user-num", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--annotation-source",
        choices=("model", "oracle"),
        default="model",
        help="model uses privacy_info_llm; oracle uses ground-truth privacy_info",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("generic", "type_specific", "complete"),
        default="type_specific",
    )
    parser.add_argument(
        "--allow-unsafe-no-mask",
        action="store_true",
        help=(
            "allow --no-mask with a raw/non-cloud-safe input. Use only for trusted "
            "local debugging; cloud memory evaluations should use eval_public_memory "
            "--cloud-safe-dataset-output instead."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    args = parse_args()
    os.environ["MEMPRIVACY_ANNOTATION_SOURCE"] = args.annotation_source
    enforce_no_mask_input_safety(
        input_path=args.input,
        is_mask=args.mask,
        allow_unsafe_no_mask=args.allow_unsafe_no_mask,
        user_limit=args.user_num,
    )

    start_time = time.time()

    process_all_users_data_add_memory(
        user_data_path=args.input,
        is_mask=args.mask,
        mask_level=args.mask_level,
        is_mcq=args.mcq,
        turns_per_chunk=args.turns_per_chunk,
        mask_mode=args.mask_mode,
        user_num=args.user_num,
        num_workers=args.num_workers,
    )

    process_all_users_data_qa(
        user_data_path=args.input,
        is_mask=args.mask,
        mask_level=args.mask_level,
        is_mcq=args.mcq,
        turns_per_chunk=args.turns_per_chunk,
        mask_mode=args.mask_mode,
        user_num=args.user_num,
        num_workers=args.num_workers,
    )

    end_time = time.time()
    elapsed = end_time - start_time

    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Time taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
