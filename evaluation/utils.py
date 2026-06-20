import hashlib
import logging
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import json_repair
import yaml
from openai import OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)
_EVALUATION_DIR = Path(__file__).resolve().parent
_ENVIRONMENT_OVERRIDES = {
    "openai_base_url": "OPENAI_BASE_URL",
    "openai_api_key": "OPENAI_API_KEY",
}
_SECTION_ENVIRONMENT_OVERRIDES = {
    ("memobase", "project_url"): "MEMOBASE_PROJECT_URL",
    ("memobase", "api_key"): "MEMOBASE_API_KEY",
}
_ENVIRONMENT_REFERENCE = re.compile(r"^\$(?:\{(?P<braced>[A-Z_][A-Z0-9_]*)\}|(?P<plain>[A-Z_][A-Z0-9_]*))$")


def _resolve_evaluation_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (_EVALUATION_DIR / candidate).resolve()


def _resolve_environment_reference(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = _ENVIRONMENT_REFERENCE.fullmatch(value.strip())
    if not match:
        return value
    return os.getenv(match.group("braced") or match.group("plain"), "")


def _expand_environment_references(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _expand_environment_references(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_expand_environment_references(item) for item in value]
    return _resolve_environment_reference(value)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(config_path: str = "eval_config.yaml") -> Dict[str, Any]:
    resolved = _resolve_evaluation_path(config_path)
    with resolved.open("r", encoding="utf-8") as f:
        config = _expand_environment_references(yaml.safe_load(f) or {})
    for config_key, environment_key in _ENVIRONMENT_OVERRIDES.items():
        environment_value = os.getenv(environment_key)
        if environment_value:
            config[config_key] = environment_value
    for (section, config_key), environment_key in _SECTION_ENVIRONMENT_OVERRIDES.items():
        environment_value = os.getenv(environment_key)
        if environment_value:
            config.setdefault(section, {})[config_key] = environment_value
    config["_config_dir"] = str(resolved.parent)
    return config


def _load_prompt(prompt_path: str) -> str:
    with _resolve_evaluation_path(prompt_path).open("r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Module-level cache: config and OpenAI client instances
# ---------------------------------------------------------------------------
_CONFIGS: Dict[str, Dict[str, Any]] = {}
_CLIENTS: Dict[Tuple[str, str], OpenAI] = {}


def _get_config(config_path: str = "eval_config.yaml") -> Dict[str, Any]:
    resolved = str(_resolve_evaluation_path(config_path))
    if resolved not in _CONFIGS:
        _CONFIGS[resolved] = _load_config(resolved)
    return _CONFIGS[resolved]


def _get_client(base_url: str, api_key: str) -> OpenAI:
    """Cache client by (base_url, api_key) to avoid repeated creation."""
    key = (base_url, api_key)
    if key not in _CLIENTS:
        _CLIENTS[key] = OpenAI(base_url=base_url, api_key=api_key)
    return _CLIENTS[key]


def _resolve_llm_params(
    llm_type: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve the full set of request parameters and client info from config
    based on the given llm_type.
    If a specific llm section has its own non-empty base_url / api_key,
    those values are used; otherwise fall back to the top-level
    openai_base_url / openai_api_key.
    """
    if llm_type not in config:
        raise ValueError(
            f"Unknown llm_type '{llm_type}'. "
            f"Available types: {[k for k in config if k.endswith('_llm')]}"
        )

    llm_cfg: Dict[str, Any] = config[llm_type]

    # base_url / api_key: prefer the llm section's own config
    base_url = llm_cfg.get("base_url") or config.get("openai_base_url", "")
    api_key = llm_cfg.get("api_key") or config.get("openai_api_key", "")

    if not base_url or not api_key:
        raise ValueError(
            f"base_url or api_key is missing for llm_type '{llm_type}'. "
            "Please set them in the llm section or at the top level of the config."
        )

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": llm_cfg["model"],
        "max_tokens": llm_cfg.get("max_tokens"),
        "temperature": llm_cfg.get("temperature"),
        "timeout": llm_cfg.get("timeout"),
        "retry_times": llm_cfg.get("retry_times", 3),
        "wait_time_lower": llm_cfg.get("wait_time_lower", 10),
        "wait_time_upper": llm_cfg.get("wait_time_upper", 30),
    }


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def call_llm(
    query: str,
    llm_type: str = "answer_llm",
    config_path: str = "eval_config.yaml",
    return_parsed_json: bool = False,
    extract_json: bool = False,
) -> str:
    """
    Send a request to the LLM and return the result.

    Parameters
    ----------
    query : str
        The full content to send to the LLM (used as the user message).
    llm_type : str
        The LLM type key in the config, e.g. "memory_llm", "answer_llm",
        "judgment_llm", "privacy_llm", etc. New types can be added to
        eval_config.yaml and will be supported automatically.
    config_path : str
        Path to the config file. Defaults to "eval_config.yaml".
    return_parsed_json : bool
        Whether to attempt parsing the response content as JSON (dict / list).
    extract_json : bool
        If False, return raw text without JSON parsing;
        if True, behaviour depends on return_parsed_json.

    Returns
    -------
    str | dict | list
        The LLM response content. The exact type depends on extract_json
        and return_parsed_json parameters.
    """
    config = _get_config(config_path)
    params = _resolve_llm_params(llm_type, config)

    # Build a retry-wrapped inner request function using per-llm retry params
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(
            min=params["wait_time_lower"],
            max=params["wait_time_upper"],
        ),
        stop=stop_after_attempt(params["retry_times"]),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _request():
        client = _get_client(params["base_url"], params["api_key"])

        messages = [{"role": "user", "content": query}]

        request_kwargs: Dict[str, Any] = {
            "model": params["model"],
            "messages": messages,
        }
        if params.get("max_tokens") is not None:
            request_kwargs["max_tokens"] = params["max_tokens"]
        if params.get("temperature") is not None:
            request_kwargs["temperature"] = params["temperature"]
        if params.get("timeout") is not None:
            request_kwargs["timeout"] = params["timeout"]

        response = client.chat.completions.create(**request_kwargs)
        raw_content = response.choices[0].message.content
        if not raw_content:
            raise ValueError(f"{llm_type} returned empty content")
        content: str = raw_content.strip()
        logger.debug("[%s] API response received (%s chars)", llm_type, len(content))

        # Strip possible <think>...</think> blocks
        content = re.sub(
            r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE
        ).strip()

        return content

    content = _request()

    # ---- Post-processing ----
    if not extract_json:
        return content

    if return_parsed_json:
        try:
            repaired = json_repair.loads(content)
            if not isinstance(repaired, (dict, list)):
                raise ValueError("Parsed JSON must be a dict or list")
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}, {traceback.format_exc()}")
            raise ValueError(f"Failed to parse JSON from content: {content[:200]}...") from e
        return repaired

    return content


def verify_mcq_answer(response: str, answer: str) -> Tuple[bool, bool]:
    pattern = re.compile(r"^[\(<{]?([abcd])[\)>}]?$", re.IGNORECASE)

    r_match = pattern.match(response.strip())
    a_match = pattern.match(answer.strip())

    if not r_match or not a_match:
        return False, False

    return r_match.group(1).lower() == a_match.group(1).lower(), True


def summarize_scores_by_question_type(records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict]:
    grouped: Dict[str, Dict[str, float]] = {}
    for record in records:
        question_type = str(record.get("question_type") or "Unknown")
        summary = grouped.setdefault(
            question_type,
            {
                "total_score": 0.0,
                "total_valid": 0,
                "total_num": 0,
            },
        )
        summary["total_score"] += float(record.get("score", 0.0))
        summary["total_valid"] += int(bool(record.get("is_valid", False)))
        summary["total_num"] += 1

    results = {}
    for question_type, summary in sorted(grouped.items()):
        total_num = int(summary["total_num"])
        total_valid = int(summary["total_valid"])
        total_score = summary["total_score"]
        results[question_type] = {
            "total_score": total_score,
            "total_valid": total_valid,
            "total_num": total_num,
            "accuracy": total_score / total_num if total_num else 0.0,
            "accuracy_valid": total_score / total_valid if total_valid else 0.0,
        }
    return results


if __name__ == "__main__":
    # 1. Simplest call — plain text response
    result = call_llm(
        query="who are you?",
        llm_type="answer_llm",
    )
    print(result)

    # 2. Request JSON response
    result = call_llm(
        query='Return your answer in JSON format: {"score": ..., "reason": ...}',
        llm_type="judgment_llm",
        extract_json=True,
        return_parsed_json=True,
    )
    print(result)  # dict
