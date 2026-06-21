from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+|[^\x00-\x7F]", text))


@dataclass(frozen=True)
class ContextMinimizerConfig:
    enabled: bool = True
    target_public_ratio: float = 0.65
    max_public_tokens: int = 128
    min_public_tokens: int = 8

    def __post_init__(self) -> None:
        if not 0.0 < self.target_public_ratio <= 1.0:
            raise ValueError("target_public_ratio must be in (0, 1]")
        if self.max_public_tokens < 1:
            raise ValueError("max_public_tokens must be positive")
        if self.min_public_tokens < 1:
            raise ValueError("min_public_tokens must be positive")
        if self.min_public_tokens > self.max_public_tokens:
            raise ValueError("min_public_tokens cannot exceed max_public_tokens")


@dataclass(frozen=True)
class TextUnit:
    index: int
    text: str
    token_count: int
    score: float
    protected: bool
    must_keep: bool


@dataclass(frozen=True)
class MinimizationResult:
    text: str
    source_tokens: int
    public_tokens: int
    selected_units: int
    total_units: int


class ContextMinimizer:
    """Rule baseline for selecting memory-worthy dialogue units."""

    FACT_PATTERNS = (
        r"\b(?:i am|i'm|i have|i live|i work|i prefer|i like|i dislike|my |we |our )",
        r"\b(?:now|currently|changed|updated|no longer|instead|remember)\b",
        r"\b(?:born|diagnosed|purchased|paid|booked|scheduled|located|address)\b",
        r"(?:我|我的|我们|目前|现在|已经|改为|不再|喜欢|偏好|住在|工作|购买|支付|预约)",
        r"(?:[$€£¥]\s?\d|\b\d{2,}\b|\d{4}-\d{1,2}-\d{1,2})",
    )
    MUST_KEEP_FACT_PATTERNS = (
        r"\b(?:i prefer|i like|i dislike|my preferred|my favou?rite)\b",
        r"\b(?:now|currently|changed|updated|no longer|instead|remember)\b",
        r"(?:偏好|喜欢|不喜欢|目前|现在|已经|改为|不再|记住|忘记)",
    )
    QUESTION_PATTERNS = (
        r"\?$",
        r"？$",
        r"\b(?:could you|would you|can you|what are|how can|do you want)\b",
        r"(?:可以吗|能否|你能|要不要|是否)",
    )
    MEMORY_CONTROL_PATTERNS = (
        r"\b(?:please\s+)?forget\s+(?:that|my|this|the|about)\b",
        r"\b(?:remove|delete|erase)\s+(?:that|my|this|the|from memory|from your memory)\b",
        r"\bdo not remember\b",
        r"\bdon't remember\b",
        r"(?:请忘记|忘掉|不要记住|从记忆中删除|删除这段记忆)",
    )
    EPHEMERAL_REQUEST_PATTERNS = (
        r"\b(?:could you|can you|would you|please)\s+(?:help|suggest|explain|describe|introduce|provide|tell|compile|refine|improve|make|give)\b",
        r"\b(?:what are|what is|why|how can|how do|when did|is there|does the same|do you think)\b",
        r"\b(?:help me|suggest|explain|describe|introduce|provide|tell me|refine|improve)\b",
        r"(?:你能|能否|可以|请帮|解释|介绍|建议|为什么|如何|是否)",
    )
    BOILERPLATE_PATTERNS = (
        r"\b(?:happy to help|glad to hear|you are welcome|would you like me to)\b",
        r"\b(?:here are|here is|certainly|of course|let me know)\b",
        r"\b(?:i will remember|i will not retain|keep .* private|not retain .* code)\b",
        r"(?:很高兴|不客气|当然可以|以下是|希望这些|如果你愿意|请告诉我)",
    )

    def __init__(self, config: Optional[ContextMinimizerConfig] = None):
        self.config = config or ContextMinimizerConfig()

    @staticmethod
    def _split_units(text: str) -> List[str]:
        units = []
        for line in text.splitlines():
            units.extend(
                match.group(0).strip()
                for match in re.finditer(
                    r".+?(?:[.!?。！？；;]+[\"'”’]*|$)(?:\s+|$)",
                    line,
                )
                if match.group(0).strip()
            )
        return units

    def _score(
        self,
        text: str,
        role: str,
        protected_fragments: Sequence[str],
    ) -> tuple[float, bool, bool]:
        normalized = text.lower()
        protected = any(fragment and fragment in text for fragment in protected_fragments)
        score = 10.0 if protected else 0.0
        if role == "user":
            score += 1.0
        score += 2.0 * sum(
            bool(re.search(pattern, normalized, flags=re.IGNORECASE))
            for pattern in self.FACT_PATTERNS
        )
        score -= 2.0 * sum(
            bool(re.search(pattern, normalized, flags=re.IGNORECASE))
            for pattern in self.QUESTION_PATTERNS
        )
        is_memory_control = any(
            re.search(pattern, normalized, flags=re.IGNORECASE)
            for pattern in self.MEMORY_CONTROL_PATTERNS
        )
        is_ephemeral_request = any(
            re.search(pattern, normalized, flags=re.IGNORECASE)
            for pattern in self.EPHEMERAL_REQUEST_PATTERNS
        )
        if is_memory_control:
            score -= 8.0
        if is_ephemeral_request:
            score -= 5.0
        score -= 3.0 * sum(
            bool(re.search(pattern, normalized, flags=re.IGNORECASE))
            for pattern in self.BOILERPLATE_PATTERNS
        )
        if len(text) < 12:
            score -= 0.5
        must_keep = (
            role == "user"
            and not is_memory_control
            and not is_ephemeral_request
            and any(
                re.search(pattern, normalized, flags=re.IGNORECASE)
                for pattern in self.MUST_KEEP_FACT_PATTERNS
            )
        )
        return score, protected, must_keep

    def minimize(
        self,
        text: str,
        role: str = "user",
        protected_fragments: Iterable[str] = (),
    ) -> MinimizationResult:
        source_tokens = estimate_tokens(text)
        units_text = self._split_units(text)
        protected_list = [fragment for fragment in protected_fragments if fragment]
        if not self.config.enabled:
            return MinimizationResult(
                text=text,
                source_tokens=source_tokens,
                public_tokens=source_tokens,
                selected_units=len(units_text),
                total_units=len(units_text),
            )
        if len(units_text) <= 1:
            score, protected, must_keep = self._score(
                text,
                role,
                protected_list,
            )
            if not protected and not must_keep and score <= 0:
                return MinimizationResult(
                    text="",
                    source_tokens=source_tokens,
                    public_tokens=0,
                    selected_units=0,
                    total_units=len(units_text),
                )
            return MinimizationResult(
                text=text,
                source_tokens=source_tokens,
                public_tokens=source_tokens,
                selected_units=len(units_text),
                total_units=len(units_text),
            )

        units = []
        for index, unit_text in enumerate(units_text):
            score, protected, must_keep = self._score(
                unit_text,
                role,
                protected_list,
            )
            units.append(
                TextUnit(
                    index=index,
                    text=unit_text,
                    token_count=max(1, estimate_tokens(unit_text)),
                    score=score,
                    protected=protected,
                    must_keep=must_keep,
                )
            )

        target_tokens = min(
            self.config.max_public_tokens,
            max(
                self.config.min_public_tokens,
                math.floor(source_tokens * self.config.target_public_ratio),
            ),
        )
        selected = [unit for unit in units if unit.protected]
        selected_ids = {unit.index for unit in selected}
        used_tokens = sum(unit.token_count for unit in selected)

        factual = sorted(
            (
                unit
                for unit in units
                if unit.index not in selected_ids and unit.must_keep
            ),
            key=lambda unit: unit.index,
        )
        for unit in factual:
            if used_tokens + unit.token_count > self.config.max_public_tokens and selected:
                continue
            selected.append(unit)
            selected_ids.add(unit.index)
            used_tokens += unit.token_count

        ranked = sorted(
            (unit for unit in units if unit.index not in selected_ids),
            key=lambda unit: (-unit.score, unit.token_count, unit.index),
        )
        for unit in ranked:
            if unit.score <= 0 and selected:
                continue
            if used_tokens + unit.token_count > target_tokens and selected:
                continue
            selected.append(unit)
            selected_ids.add(unit.index)
            used_tokens += unit.token_count
            if used_tokens >= target_tokens:
                break

        if not selected:
            best_unit = max(units, key=lambda unit: (unit.score, -unit.token_count))
            if best_unit.score <= 0 and not best_unit.protected and not best_unit.must_keep:
                return MinimizationResult(
                    text="",
                    source_tokens=source_tokens,
                    public_tokens=0,
                    selected_units=0,
                    total_units=len(units),
                )
            selected = [best_unit]

        selected.sort(key=lambda unit: unit.index)
        minimized = " ".join(unit.text for unit in selected)
        return MinimizationResult(
            text=minimized,
            source_tokens=source_tokens,
            public_tokens=estimate_tokens(minimized),
            selected_units=len(selected),
            total_units=len(units),
        )
