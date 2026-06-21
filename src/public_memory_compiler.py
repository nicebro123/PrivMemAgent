from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from src.alias_router import ScopedAliasRouter
from src.context_minimizer import ContextMinimizer
from src.policy import (
    AliasScope,
    PrivacyPolicy,
    RouteAction,
    RouteDecision,
    RoutingContext,
)
from src.privacy_masking import validate_privacy_items
from src.provenance import ProvenanceStore, build_provenance_record
from src.sufficiency_selector import (
    RepresentationCandidate,
    SelectionResult,
    SelectorConfig,
    SufficiencySelector,
)
from src.utility_leakage_selector import LearnedUtilityLeakageSelector


def _estimate_tokens(text: str) -> int:
    latin_tokens = re.findall(r"[A-Za-z0-9_]+|[^\x00-\x7F]", text)
    return max(1, len(latin_tokens))


def _normalize_public_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = start
    while left > 0 and text[left - 1] not in ".!?。！？;；\n":
        left -= 1
    while left < start and text[left].isspace():
        left += 1

    right = end
    while right < len(text) and text[right] not in ".!?。！？;；\n":
        right += 1
    if right < len(text):
        right += 1
    while right < len(text) and text[right].isspace():
        right += 1
    return left, right


@dataclass(frozen=True)
class CompiledItem:
    source_item_index: int
    source_fingerprint: str
    privacy_level: str
    privacy_type: str
    decision: RouteDecision
    representation_type: str
    public_value: Optional[str]
    utility_score: float
    leakage_score: float
    alias_scope: Optional[AliasScope]
    scope_id: Optional[str]
    provenance_id: str


@dataclass(frozen=True)
class CompiledMemory:
    user_id: str
    message_id: str
    source_fingerprint: str
    public_text: str
    items: tuple[CompiledItem, ...]
    policy_version: str
    source_tokens: int
    public_tokens: int

    @property
    def token_reduction(self) -> float:
        if self.source_tokens == 0:
            return 0.0
        return 1.0 - self.public_tokens / self.source_tokens


class RuleBasedAbstractor:
    """Generates ordered public abstractions without external model calls."""

    CATEGORY_RULES = (
        (
            ("email", "phone", "address", "contact", "zip"),
            "contact information",
        ),
        (
            ("medical", "health", "diagnosis", "prescription", "biometric"),
            "health constraint relevant to future assistance",
        ),
        (
            ("financial", "transaction", "income", "asset", "debt", "loan", "credit"),
            "financial constraint relevant to future assistance",
        ),
        (
            ("job", "employer", "company", "school", "identity background"),
            "professional or educational context",
        ),
        (
            ("location", "itinerary", "trajectory", "accommodation"),
            "location or itinerary constraint",
        ),
        (
            ("account", "username", "network", "device", "ip"),
            "account or device reference",
        ),
        (
            ("relationship", "third-party", "real name", "name"),
            "person relevant to the user",
        ),
    )

    def category_abstraction(self, privacy_type: str) -> str:
        normalized = privacy_type.lower()
        for keywords, abstraction in self.CATEGORY_RULES:
            if any(keyword in normalized for keyword in keywords):
                return abstraction
        return "private contextual detail"

    def candidates(
        self,
        privacy_item: Mapping[str, str],
        required_types: Iterable[str],
    ) -> List[RepresentationCandidate]:
        privacy_type = privacy_item["privacy_type"]
        is_required = privacy_type in set(required_types)
        category_text = self.category_abstraction(privacy_type)
        return [
            RepresentationCandidate(
                text="private detail",
                specificity=0,
                utility_score=0.40 if is_required else 0.68,
                leakage_score=0.05,
                token_count=_estimate_tokens("private detail"),
                representation_type="generic_abstract",
            ),
            RepresentationCandidate(
                text=category_text,
                specificity=1,
                utility_score=0.92 if is_required else 0.82,
                leakage_score=0.20,
                token_count=_estimate_tokens(category_text),
                representation_type="category_abstract",
            ),
            RepresentationCandidate(
                text=f"private {privacy_type.lower()} value",
                specificity=2,
                utility_score=0.96,
                leakage_score=0.45,
                token_count=_estimate_tokens(privacy_type) + 2,
                representation_type="typed_abstract",
            ),
        ]


class PublicMemoryCompiler:
    """Compiles raw dialogue into a policy-constrained public representation."""

    def __init__(
        self,
        policy: PrivacyPolicy,
        alias_router: ScopedAliasRouter,
        provenance_store: ProvenanceStore,
        selector: Optional[SufficiencySelector] = None,
        abstractor: Optional[RuleBasedAbstractor] = None,
        learned_selector: Optional[LearnedUtilityLeakageSelector] = None,
        context_minimizer: Optional[ContextMinimizer] = None,
    ):
        self.policy = policy
        self.alias_router = alias_router
        self.provenance_store = provenance_store
        self.selector = selector or SufficiencySelector()
        self.abstractor = abstractor or RuleBasedAbstractor()
        self.learned_selector = learned_selector
        self.context_minimizer = context_minimizer or ContextMinimizer()

    @classmethod
    def with_defaults(
        cls,
        alias_db_path: str,
        provenance_db_path: str,
        selector_config: Optional[SelectorConfig] = None,
        encryption_key: Optional[str] = None,
        key_path: Optional[str] = None,
    ) -> "PublicMemoryCompiler":
        return cls(
            policy=PrivacyPolicy.default(),
            alias_router=ScopedAliasRouter(
                alias_db_path,
                encryption_key=encryption_key,
                key_path=key_path,
            ),
            provenance_store=ProvenanceStore(provenance_db_path),
            selector=SufficiencySelector(selector_config),
        )

    def _abstraction_candidates(
        self,
        privacy_item: Mapping[str, str],
        context: RoutingContext,
        message_text: str,
    ) -> list[RepresentationCandidate]:
        try:
            return self.abstractor.candidates(
                privacy_item,
                required_types=context.exact_required_types,
                user_id=context.user_id,
                message_id=context.message_id,
                role=context.message_role,
                message_text=message_text,
            )
        except TypeError:
            return self.abstractor.candidates(
                privacy_item,
                required_types=context.exact_required_types,
            )

    def _select_abstraction(
        self,
        candidates: list[RepresentationCandidate],
        privacy_item: Mapping[str, str],
    ) -> SelectionResult:
        if self.learned_selector is not None:
            return self.learned_selector.select(candidates, privacy_item=privacy_item)
        return self.selector.select(candidates)

    @staticmethod
    def _selected_items(items: Iterable[Mapping[str, str]]) -> List[Tuple[int, Mapping[str, str]]]:
        selected: Dict[str, Tuple[int, Mapping[str, str]]] = {}
        for index, item in enumerate(items):
            current = selected.get(item["original_text"])
            candidate_rank = (-int(item["privacy_level"][-1]), item["privacy_type"])
            current_rank = (
                (-int(current[1]["privacy_level"][-1]), current[1]["privacy_type"])
                if current
                else None
            )
            if current is None or candidate_rank < current_rank:
                selected[item["original_text"]] = (index, item)
        return list(selected.values())

    def _compile_item(
        self,
        item_index: int,
        privacy_item: Mapping[str, str],
        context: RoutingContext,
        message_text: str,
    ) -> Tuple[CompiledItem, str]:
        decision = self.policy.route(privacy_item, context)
        public_value: Optional[str] = None
        representation_type = decision.action.value
        utility_score = 0.0
        leakage_score = 0.0
        scope_id = None

        if decision.action == RouteAction.PUBLIC_ABSTRACT:
            self.alias_router.store_local(
                privacy_item["original_text"],
                privacy_item["privacy_type"],
                privacy_item["privacy_level"],
                context,
            )
            selection = self._select_abstraction(
                self._abstraction_candidates(privacy_item, context, message_text),
                privacy_item,
            )
            if selection.selected:
                public_value = selection.selected.text
                representation_type = selection.selected.representation_type
                utility_score = selection.selected.utility_score
                leakage_score = selection.selected.leakage_score
            else:
                representation_type = "budget_rejected"
        elif decision.action == RouteAction.PUBLIC_REVERSIBLE:
            self.alias_router.store_local(
                privacy_item["original_text"],
                privacy_item["privacy_type"],
                privacy_item["privacy_level"],
                context,
            )
            public_value, scope_id = self.alias_router.get_alias(
                privacy_item["original_text"],
                privacy_item["privacy_type"],
                privacy_item["privacy_level"],
                decision.alias_scope,
                context,
            )
            representation_type = "scoped_reversible_alias"
            utility_score = 1.0
            leakage_score = 0.30
        elif decision.action == RouteAction.LOCAL_ONLY:
            self.alias_router.store_local(
                privacy_item["original_text"],
                privacy_item["privacy_type"],
                privacy_item["privacy_level"],
                context,
            )
            representation_type = "encrypted_local_only"
        else:
            representation_type = "dropped"

        provenance_id = uuid.uuid4().hex
        provenance = build_provenance_record(
            record_id=provenance_id,
            context_user_id=context.user_id,
            source_message_id=context.message_id,
            source_item_index=item_index,
            decision=decision,
            representation_type=representation_type,
            public_text=public_value,
            scope_id=scope_id,
        )
        self.provenance_store.add(provenance)

        compiled = CompiledItem(
            source_item_index=item_index,
            source_fingerprint=self.alias_router.fingerprint(privacy_item["original_text"]),
            privacy_level=privacy_item["privacy_level"],
            privacy_type=privacy_item["privacy_type"],
            decision=decision,
            representation_type=representation_type,
            public_value=public_value,
            utility_score=utility_score,
            leakage_score=leakage_score,
            alias_scope=decision.alias_scope,
            scope_id=scope_id,
            provenance_id=provenance_id,
        )
        return compiled, public_value or ""

    def compile(
        self,
        message_text: str,
        privacy_items: List[Mapping[str, str]],
        context: RoutingContext,
        strict: bool = True,
    ) -> CompiledMemory:
        validated = validate_privacy_items(
            privacy_items,
            dialogue_text=message_text,
            strict=strict,
        )
        compiled_items = []
        replacements = []
        for item_index, privacy_item in self._selected_items(validated):
            compiled, replacement = self._compile_item(
                item_index,
                privacy_item,
                context,
                message_text,
            )
            compiled_items.append(compiled)
            original = privacy_item["original_text"]
            start = 0
            while True:
                start = message_text.find(original, start)
                if start < 0:
                    break
                end = start + len(original)
                if compiled.decision.action in {RouteAction.DROP, RouteAction.LOCAL_ONLY}:
                    sentence_start, sentence_end = _sentence_bounds(message_text, start, end)
                    replacements.append(
                        (
                            sentence_start,
                            sentence_end,
                            "",
                            max(len(original), sentence_end - sentence_start),
                        )
                    )
                else:
                    replacements.append((start, end, replacement, len(original)))
                start += len(original)

        replacements.sort(key=lambda item: (-item[3], item[0]))
        selected_replacements = []
        for start, end, replacement, _ in replacements:
            overlapping = [
                index
                for index, (selected_start, selected_end, selected_replacement) in enumerate(
                    selected_replacements
                )
                if start < selected_end and end > selected_start
            ]
            if not overlapping:
                selected_replacements.append((start, end, replacement))
                continue

            if replacement == "" and all(
                selected_replacements[index][2] == "" for index in overlapping
            ):
                merged_start = min(
                    [start] + [selected_replacements[index][0] for index in overlapping]
                )
                merged_end = max(
                    [end] + [selected_replacements[index][1] for index in overlapping]
                )
                for index in sorted(overlapping, reverse=True):
                    selected_replacements.pop(index)
                selected_replacements.append((merged_start, merged_end, ""))
            elif replacement == "":
                for index in sorted(overlapping, reverse=True):
                    selected_start, selected_end, selected_replacement = (
                        selected_replacements[index]
                    )
                    if selected_replacement == "":
                        selected_replacements.pop(index)
                selected_replacements.append((start, end, replacement))

        public_text = message_text
        for start, end, replacement in sorted(selected_replacements, reverse=True):
            public_text = public_text[:start] + replacement + public_text[end:]
        public_text = _normalize_public_text(public_text)
        minimization = self.context_minimizer.minimize(
            public_text,
            role=context.message_role,
            protected_fragments=[item.public_value for item in compiled_items if item.public_value],
        )
        public_text = _normalize_public_text(minimization.text)

        return CompiledMemory(
            user_id=context.user_id,
            message_id=context.message_id,
            source_fingerprint=self.alias_router.fingerprint(message_text),
            public_text=public_text,
            items=tuple(compiled_items),
            policy_version=self.policy.version,
            source_tokens=_estimate_tokens(message_text),
            public_tokens=_estimate_tokens(public_text) if public_text else 0,
        )

    def revoke_messages(
        self,
        user_id: str,
        message_ids: Iterable[str],
        reason: str = "user-requested deletion",
    ) -> List[str]:
        return self.provenance_store.revoke_by_source(
            user_id,
            message_ids,
            reason,
        )

    def close(self) -> None:
        self.provenance_store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
