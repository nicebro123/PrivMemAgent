import argparse
import json
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path, PurePath
from typing import Callable, List, Optional

from langchain_core.messages.utils import count_tokens_approximately, trim_messages

# LangGraph & LangMem imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store
from langmem import create_manage_memory_tool, create_search_memory_tool
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

# Custom utility imports (ensure these exist in your environment)
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

os.environ["OPENAI_BASE_URL"] = _config["openai_base_url"]
os.environ["OPENAI_API_KEY"] = _config["openai_api_key"]


def _configured_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(_config["_config_dir"]) / candidate).resolve())


_config["privacy"]["db_path"] = _configured_path(_config["privacy"]["db_path"])


def _annotation_source() -> str:
    return os.getenv("MEMPRIVACY_ANNOTATION_SOURCE", "model")


MODEL = _config["memory_llm"]["model"]
EMBEDDING_MODEL = _config["embedding_model"]

RETRY_TIMES = _config["memory_llm"].get("retry_times", 3)
WAIT_TIME_LOWER = _config["memory_llm"].get("wait_time_lower", 1)
WAIT_TIME_UPPER = _config["memory_llm"].get("wait_time_upper", 10)
MAX_ALLOWED_TOKENS_FOR_INPUT = _config["memory_llm"].get("input_max_tokens", 2000)


# ---------------------------------------------------------------------------
# LangMem Setup
# ---------------------------------------------------------------------------
def prompt(state):
    """Prepare the messages for the LLM."""
    store = get_store()
    memories = store.search(
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful assistant.

## Memories
<memories>
{memories}
</memories>
"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]


def pre_model_hook(state: dict) -> dict:
    """Trim messages to prevent context window overflow."""
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_ALLOWED_TOKENS_FOR_INPUT,
    )
    return {"llm_input_messages": trimmed}


def _build_embedding_function() -> Callable[[list[str]], list[list[float]]] | str:
    provider = EMBEDDING_MODEL.get("provider", "openai")
    if provider != "huggingface":
        return f"{provider}:{EMBEDDING_MODEL['model']}"

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMBEDDING_MODEL["model"],
        device=EMBEDDING_MODEL.get("device", "cpu"),
    )

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, convert_to_numpy=True).tolist()

    return embed


class LangMem:
    def __init__(self):
        self.store = InMemoryStore(
            index={
                "dims": EMBEDDING_MODEL["dimensions"],
                "embed": _build_embedding_function(),
            }
        )
        self.checkpointer = MemorySaver()  # Checkpoint graph state

        self.agent = create_react_agent(
            f"openai:{MODEL}",
            prompt=prompt,
            tools=[
                create_manage_memory_tool(namespace=("memories",)),
                create_search_memory_tool(namespace=("memories",)),
            ],
            pre_model_hook=pre_model_hook,
            store=self.store,
            checkpointer=self.checkpointer,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
        stop=stop_after_attempt(RETRY_TIMES),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def add_memory(self, message: str, config: dict):
        return self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]}, config=config
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
        stop=stop_after_attempt(RETRY_TIMES),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def search_memory(self, query: str, config: dict):
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": query}]}, config=config
        )
        return response["messages"][-1].content


# ---------------------------------------------------------------------------
# Public API Wrappers for Evaluation logic
# ---------------------------------------------------------------------------
def add_message(
    memory_agent: LangMem,
    messages: List[dict],
    user_id: str,
    timestamp: Optional[str] = None,
) -> None:
    """Format and add messages to LangMem for the given user."""
    if timestamp is None:
        formatted_messages = [f"{msg['role']}: {msg['content']}" for msg in messages]
    else:
        formatted_messages = [f"{timestamp} | {msg['role']}: {msg['content']}" for msg in messages]

    message_str = "\n".join(formatted_messages)

    # Use user_id as the thread_id for state isolation
    config = {"configurable": {"thread_id": user_id}}
    memory_agent.add_memory(message=message_str, config=config)


def search_memory(
    memory_agent: LangMem,
    query: str,
    user_id: str,
) -> str:
    """Search LangMem and return the synthesized response."""
    config = {"configurable": {"thread_id": user_id}}
    return memory_agent.search_memory(query=query, config=config)


# ---------------------------------------------------------------------------
# Core Evaluation Logic
# ---------------------------------------------------------------------------
def chunk_dialogues(dialogues: List[dict], turns_per_chunk: int = 1):
    """
    Split dialogues into chunks, each containing `turns_per_chunk` dialogue turns.
    One turn = user + assistant.
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


def process_single_user_data(
    user_data: dict,
    is_mask: bool,
    mask_level: list[str],
    is_mcq: bool,  # Flag for Multiple-choice questions
    turns_per_chunk: int = 1,
    mask_mode: str = "type_specific",  # "type_specific" or "generic" or "complete"
):
    user_id = user_data["uuid"]
    user_name = user_data["metadata"]["user_name"]

    # Initialize LangMem instance for the current user
    memory = LangMem()

    if is_mask and mask_mode != "complete":
        privacy_sub_path = os.path.join(
            _config["privacy"]["db_path"],
            f"Langmem_{_annotation_source()}_"
            f"{'mask' if is_mask else 'unmask'}_{''.join(mask_level)}_{mask_mode}_"
            f"{'mcq' if is_mcq else 'qa'}",
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

    # 1. Store memories
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

    # 2. Evaluate questions against the stored memories
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

        # Retrieve relevant memories from LangMem
        memories_text = search_memory(
            memory_agent=memory,
            query=cloud_query,
            user_id=user_id,
        )

        is_valid = True
        if is_mcq:
            prompt_path = _config["prompts"]["answer_prompt_2"]
            prompt_template = _load_prompt(prompt_path)
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
            query_prompt = prompt_template.format(
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
            judge_prompt_text = judge_prompt.format(
                question=cloud_query,
                reference_answer=cloud_reference_answer,
                response=response_content,
            )

            judgment_response = call_llm(
                query=judge_prompt_text,
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


def _worker(args):
    """Wrapper function to unpack arguments for pool map."""
    user_data, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode = args
    return process_single_user_data(
        user_data, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode
    )


def process_all_users_data(
    user_data_path: str,
    is_mask: bool,
    mask_level: list[str],
    is_mcq: bool,
    turns_per_chunk: int = 1,
    mask_mode: str = "type_specific",
    user_num: int = None,
    num_workers: int = 10,
):
    """Main function to load dataset and process users in parallel."""
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
            pool.imap_unordered(_worker, tasks), total=len(tasks), desc="Processing users"
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
            "memory_model": MODEL,
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
        f"LangMem_{user_data_name}_{_annotation_source()}_"
        f"{'mask' if is_mask else 'unmask'}_{''.join(mask_level)}_{mask_mode}_"
        f"{time.strftime('%Y%m%d%H%M%S')}.json",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)

    print("accuracy:", all_accuracy)
    print(f"Output saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LangMem with privacy masking")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-level", nargs="+", default=["PL2", "PL3", "PL4"])
    parser.add_argument("--mcq", action=argparse.BooleanOptionalAction, default=False)
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

    process_all_users_data(
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
