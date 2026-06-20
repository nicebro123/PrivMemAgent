from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class CompletionFunction(Protocol):
    def __call__(self, prompt: str) -> str: ...


def strip_thinking_blocks(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def resolve_secret(value: str | None) -> str:
    """Resolve `$ENV_NAME` values without ever printing the secret."""
    if not value:
        return ""
    if value.startswith("$"):
        return os.environ.get(value[1:], "")
    return value


@dataclass
class OpenAIChatCompletion:
    model: str
    base_url: str = ""
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: int = 120

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        section: str,
    ) -> OpenAIChatCompletion:
        if section not in config:
            raise ValueError(f"missing LLM config section: {section}")
        values = dict(config[section] or {})
        model = str(values.get("model", "")).strip()
        if not model:
            raise ValueError(f"missing model for LLM config section: {section}")
        base_url = resolve_secret(
            str(values.get("base_url") or config.get("openai_base_url") or "")
        )
        api_key = resolve_secret(
            str(values.get("api_key") or config.get("openai_api_key") or "")
        )
        if not api_key:
            raise ValueError(
                f"missing API key for {section}; use an environment reference such as "
                "`$OPENAI_API_KEY` in the YAML config"
            )
        return cls(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=int(values.get("max_tokens", 4096)),
            temperature=float(values.get("temperature", 0.0)),
            timeout=int(values.get("timeout", 120)),
        )

    def __call__(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "OpenAI-compatible generation requires the `openai` package"
            ) from exc

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty content")
        return strip_thinking_blocks(content)


class TransformersTextGenerator:
    """Lazy local-model generator used by the `trained_model` PMA backend."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        device_map: str = "auto",
        trust_remote_code: bool = False,
        revision: str = "main",
    ):
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required")
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Local PMA inference requires `transformers` and a supported "
                "PyTorch installation"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        self._model.eval()

    def __call__(self, prompt: str) -> str:
        self._load()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("local model failed to initialize")
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:  # pragma: no cover - old tokenizer fallback
            rendered = prompt
        inputs = self._tokenizer(rendered, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        generated = self._model.generate(**inputs, **generation_kwargs)
        completion = generated[0, inputs["input_ids"].shape[1] :]
        return strip_thinking_blocks(
            self._tokenizer.decode(completion, skip_special_tokens=True)
        )


def parse_json_content(content: str) -> Any:
    try:
        import json_repair
    except ImportError:
        return json.loads(content)
    return json_repair.loads(content)


def build_completion_from_config(
    config: Mapping[str, Any],
    section: str,
) -> Callable[[str], str]:
    return OpenAIChatCompletion.from_config(config, section)
