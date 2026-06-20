from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class MemoryBackend(Protocol):
    def add(
        self,
        messages: Sequence[Mapping[str, Any]],
        user_id: str,
        timestamp: str | None = None,
    ) -> None: ...

    def search(self, query: str, user_id: str, limit: int = 20) -> str: ...


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fff]|[a-z0-9_@.-]+", text.casefold()))


class InMemoryBackend:
    """Deterministic lexical backend for integration tests and CI."""

    def __init__(self):
        self._records: dict[str, list[str]] = defaultdict(list)

    def add(
        self,
        messages: Sequence[Mapping[str, Any]],
        user_id: str,
        timestamp: str | None = None,
    ) -> None:
        for message in messages:
            prefix = f"{timestamp} | " if timestamp else ""
            self._records[user_id].append(
                f"{prefix}{message.get('role', 'unknown')}: "
                f"{message.get('content', '')}"
            )

    def search(self, query: str, user_id: str, limit: int = 20) -> str:
        query_tokens = _tokens(query)
        ranked = sorted(
            self._records.get(user_id, []),
            key=lambda record: (
                len(query_tokens & _tokens(record)),
                len(record),
            ),
            reverse=True,
        )
        return "\n".join(ranked[:limit])


class Mem0Backend:
    """Thin real Mem0 adapter; imports heavy dependencies only when selected."""

    def __init__(
        self,
        user_id: str,
        db_path: str,
        config: Mapping[str, Any],
    ):
        try:
            from mem0 import Memory
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Mem0 evaluation requires `mem0ai` and `chromadb`"
            ) from exc

        embedding = config.get("embedding_model", {})
        memory_llm = config.get("memory_llm", {})
        openai_key = str(config.get("openai_api_key", ""))
        openai_url = str(config.get("openai_base_url", ""))
        if openai_key.startswith("$"):
            openai_key = os.environ.get(openai_key[1:], "")
        if openai_url.startswith("$"):
            openai_url = os.environ.get(openai_url[1:], "")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if openai_url:
            os.environ["OPENAI_BASE_URL"] = openai_url

        mem0_config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": user_id,
                    "path": db_path,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": embedding.get("model", "text-embedding-3-small")},
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": memory_llm.get("model"),
                    "temperature": memory_llm.get("temperature", 0),
                    "max_tokens": memory_llm.get("max_tokens", 4096),
                },
            },
        }
        self.memory = Memory.from_config(mem0_config)

    def add(
        self,
        messages: Sequence[Mapping[str, Any]],
        user_id: str,
        timestamp: str | None = None,
    ) -> None:
        metadata = {"timestamp": timestamp} if timestamp else None
        self.memory.add(
            messages=list(messages),
            user_id=user_id,
            metadata=metadata,
        )

    def search(self, query: str, user_id: str, limit: int = 20) -> str:
        response = self.memory.search(query=query, user_id=user_id, limit=limit)
        return "\n".join(f"- {item['memory']}" for item in response.get("results", []))
