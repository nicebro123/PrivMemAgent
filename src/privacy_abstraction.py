from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import yaml

from .llm_clients import TransformersTextGenerator, parse_json_content
from .privacy_schema import (
    VALID_ABSTRACTION_LEVELS,
    AbstractionCandidate,
    AbstractionTrace,
    PolicyValidationError,
    PrivacyItem,
    PrivateResidue,
    stable_id,
    validate_candidate_against_policy,
)

DEFAULT_POLICY: dict[str, Any] = {
    "default_task_family": "general",
    "utility_threshold": 0.85,
    "max_candidates_per_turn": 6,
    "allowed_levels": {
        "PL1": ["L0", "L1", "L2", "L3"],
        "PL2": ["L1", "L2", "L3", "L4", "L5"],
        "PL3": ["L2", "L3", "L4", "L5"],
        "PL4": ["L4", "L5"],
    },
    "type_overrides": {
        "Verification Code": {
            "allowed_levels": ["L5"],
            "retention": "no_retention",
        },
        "Password": {"allowed_levels": ["L5"], "retention": "no_retention"},
        "API Key": {"allowed_levels": ["L5"], "retention": "no_retention"},
        "Recovery Code": {
            "allowed_levels": ["L5"],
            "retention": "no_retention",
        },
        "Detailed Address": {
            "allowed_levels": ["L2", "L3", "L4", "L5"],
            "retention": "local_only",
        },
    },
}

SUPPORTED_BACKENDS = {
    "heuristic",
    "oracle_prompt",
    "trained_model",
    "typed_placeholder",
    "redaction",
}
LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5"]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in output
            and isinstance(output[key], dict)
            and isinstance(value, Mapping)
        ):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = deepcopy(value)
    return output


def load_abstraction_policy(config_path: str | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "privacy_abstraction_config.yaml",
        )
    if not os.path.exists(config_path):
        return deepcopy(DEFAULT_POLICY)
    with open(config_path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    values = loaded.get("privacy_abstraction", loaded)
    policy = _deep_merge(DEFAULT_POLICY, values)
    validate_policy(policy)
    return policy


def validate_policy(policy: Mapping[str, Any]) -> None:
    max_candidates = int(policy.get("max_candidates_per_turn", 0))
    if max_candidates < 2:
        raise ValueError(
            "max_candidates_per_turn must be at least 2 for L4/L5 fallbacks"
        )
    threshold = float(policy.get("utility_threshold", -1))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("utility_threshold must be between 0 and 1")
    allowed_by_level = policy.get("allowed_levels", {})
    for privacy_level in ("PL1", "PL2", "PL3", "PL4"):
        allowed = set(allowed_by_level.get(privacy_level, []))
        if not allowed or not allowed <= VALID_ABSTRACTION_LEVELS:
            raise ValueError(f"invalid allowed levels for {privacy_level}: {allowed}")
    for privacy_type, values in policy.get("type_overrides", {}).items():
        allowed = set(values.get("allowed_levels", []))
        if allowed and not allowed <= VALID_ABSTRACTION_LEVELS:
            raise ValueError(f"invalid allowed levels for {privacy_type}: {allowed}")
        retention = values.get("retention", "local_only")
        if retention not in {"local_only", "session_only", "no_retention"}:
            raise ValueError(f"invalid retention for {privacy_type}: {retention}")


def privacy_items_from_dicts(items: Sequence[Mapping[str, Any]]) -> list[PrivacyItem]:
    return [PrivacyItem.from_dict(item) for item in items if item.get("original_text")]


def type_to_mask_prefix(privacy_type: str) -> str:
    return privacy_type.replace(" ", "_").replace("/", "_")


def build_private_residue(
    privacy_items: Sequence[PrivacyItem],
    policy: Mapping[str, Any],
) -> list[PrivateResidue]:
    overrides = policy.get("type_overrides", {})
    return [
        PrivateResidue(
            raw=item.original_text,
            privacy_type=item.privacy_type,
            privacy_level=item.privacy_level,
            retention=str(
                overrides.get(item.privacy_type, {}).get(
                    "retention",
                    "local_only",
                )
            ),  # type: ignore[arg-type]
        )
        for item in privacy_items
    ]


def allowed_levels_for_item(
    item: PrivacyItem,
    policy: Mapping[str, Any],
) -> list[str]:
    return list(
        policy.get("type_overrides", {})
        .get(item.privacy_type, {})
        .get(
            "allowed_levels",
            policy.get("allowed_levels", {}).get(item.privacy_level, ["L5"]),
        )
    )


def allowed_levels_for_items(
    privacy_items: Sequence[PrivacyItem],
    policy: Mapping[str, Any],
) -> set[str]:
    allowed: set[str] | None = None
    for item in privacy_items:
        current = set(allowed_levels_for_item(item, policy))
        allowed = current if allowed is None else allowed & current
    return allowed or {"L5"}


def effective_level_for_item(
    target_level: str,
    item: PrivacyItem,
    policy: Mapping[str, Any],
) -> str:
    if target_level not in LEVEL_ORDER:
        raise ValueError(f"invalid target abstraction level: {target_level}")
    allowed = set(allowed_levels_for_item(item, policy))
    if target_level in allowed:
        return target_level
    target_index = LEVEL_ORDER.index(target_level)
    for level in LEVEL_ORDER[target_index + 1 :]:
        if level in allowed:
            return level
    return "L5"


def replace_privacy_spans(
    text: str,
    replacements: Sequence[tuple[str, str]],
) -> str:
    ordered = sorted(replacements, key=lambda item: len(item[0]), reverse=True)
    output = text
    for original, replacement in ordered:
        output = output.replace(original, replacement)
    return output


def _effective_level_records(
    privacy_items: Sequence[PrivacyItem],
    target_level: str,
    policy: Mapping[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "raw": item.original_text,
            "privacy_type": item.privacy_type,
            "level": effective_level_for_item(target_level, item, policy),
        }
        for item in privacy_items
    ]


def typed_placeholder_text(
    dialogue_text: str,
    privacy_items: Sequence[PrivacyItem],
) -> tuple[str, list[AbstractionTrace]]:
    counts: dict[str, int] = defaultdict(int)
    replacements: list[tuple[str, str]] = []
    traces: list[AbstractionTrace] = []
    for item in privacy_items:
        counts[item.privacy_type] += 1
        placeholder = (
            f"<{type_to_mask_prefix(item.privacy_type)}_{counts[item.privacy_type]}>"
        )
        replacements.append((item.original_text, placeholder))
        traces.append(
            AbstractionTrace(
                raw=item.original_text,
                public_abstraction=placeholder,
                reason=(
                    "Typed placeholder preserves the privacy type while hiding "
                    "the raw value."
                ),
            )
        )
    return replace_privacy_spans(dialogue_text, replacements), traces


def redacted_text(
    dialogue_text: str,
    privacy_items: Sequence[PrivacyItem],
) -> tuple[str, list[AbstractionTrace]]:
    replacements = [(item.original_text, "***") for item in privacy_items]
    traces = [
        AbstractionTrace(
            raw=item.original_text,
            public_abstraction="***",
            reason="Full redaction removes the public sensitive value.",
        )
        for item in privacy_items
    ]
    return replace_privacy_spans(dialogue_text, replacements), traces


def heuristic_abstraction_for_item(
    item: PrivacyItem,
    level: str,
    task_family: str,
) -> str:
    mappings: dict[str, dict[str, str]] = {
        "Detailed Address": {
            "L1": "a specific private location",
            "L2": "a coarse nearby area relevant for low-commute planning",
            "L3": "a preference for nearby, low-commute options",
        },
        "Location": {
            "L1": "a private location",
            "L2": "a broad location constraint",
            "L3": "a preference for nearby options",
        },
        "Precise Location": {
            "L1": "a private location",
            "L2": "a broad location constraint",
            "L3": "a preference for nearby options",
        },
        "Medical Health": {
            "L1": "a health-related condition",
            "L2": (
                "a functional health constraint relevant to comfort and accessibility"
            ),
            "L3": "a preference for plans respecting comfort and accessibility",
        },
        "Medical Record": {
            "L1": "a private medical detail",
            "L2": "a functional health constraint",
            "L3": "a preference for health-compatible plans",
        },
        "Itinerary/Trajectory": {
            "L1": "a private travel detail",
            "L2": "a broad travel-planning constraint",
            "L3": "a preference for plans that reduce travel friction",
        },
        "Relationship Info": {
            "L1": "a private relationship detail",
            "L2": "a social context relevant to planning",
            "L3": "a preference to account for close companions",
        },
        "Religious Beliefs": {
            "L1": "a sensitive personal belief",
            "L2": "a content-sensitivity constraint",
            "L3": "a preference for recommendations compatible with personal values",
        },
        "Political Views/Stance": {
            "L1": "a sensitive political attribute",
            "L2": "a content-sensitivity constraint",
            "L3": "a preference to avoid conflicting political content",
        },
    }
    if item.privacy_type in mappings:
        return mappings[item.privacy_type].get(level, "a private user constraint")
    level_name = {
        "L1": "category-level",
        "L2": "functional",
        "L3": "task-preference",
    }.get(level, "abstract")
    task_suffix = f" for {task_family}" if task_family != "general" else ""
    return f"a {level_name} {item.privacy_type.lower()} signal{task_suffix}"


def abstract_text_heuristically(
    dialogue_text: str,
    privacy_items: Sequence[PrivacyItem],
    level: str,
    task_family: str,
    policy: Mapping[str, Any] | None = None,
) -> tuple[str, list[AbstractionTrace]]:
    active_policy = policy or DEFAULT_POLICY
    replacements: list[tuple[str, str]] = []
    traces: list[AbstractionTrace] = []
    placeholder_counts: dict[str, int] = defaultdict(int)
    for item in privacy_items:
        effective = effective_level_for_item(level, item, active_policy)
        if effective == "L4":
            placeholder_counts[item.privacy_type] += 1
            abstraction = (
                f"<{type_to_mask_prefix(item.privacy_type)}_"
                f"{placeholder_counts[item.privacy_type]}>"
            )
            reason = "Policy escalated this item to a typed placeholder."
        elif effective == "L5":
            abstraction = "***"
            reason = "Policy escalated this item to full redaction."
        else:
            abstraction = heuristic_abstraction_for_item(
                item,
                effective,
                task_family,
            )
            reason = (
                f"{effective} abstraction removes the raw value while retaining "
                "task-relevant semantics."
            )
        replacements.append((item.original_text, abstraction))
        traces.append(
            AbstractionTrace(
                raw=item.original_text,
                public_abstraction=abstraction,
                reason=reason,
            )
        )
    return replace_privacy_spans(dialogue_text, replacements), traces


class PrivacyMemoryAbstractor:
    def __init__(
        self,
        policy: Mapping[str, Any] | None = None,
        backend: str = "heuristic",
        *,
        completion_fn: Callable[[str], str] | None = None,
        model_name_or_path: str = "",
        model_revision: str = "main",
        prompt_path: str | None = None,
        fallback_on_error: bool = True,
    ):
        self.policy = _deep_merge(DEFAULT_POLICY, policy or {})
        validate_policy(self.policy)
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"unknown PMA backend: {backend}; "
                f"supported={sorted(SUPPORTED_BACKENDS)}"
            )
        self.backend = backend
        self.completion_fn = completion_fn
        self.model_name_or_path = model_name_or_path
        self.model_revision = model_revision
        self.prompt_path = prompt_path or os.path.join(
            os.path.dirname(__file__),
            "..",
            "evaluation",
            "prompts",
            "pma_generate_candidates.txt",
        )
        self.fallback_on_error = fallback_on_error
        self.last_backend_error: str | None = None

    def generate_candidates(
        self,
        dialogue_text: str,
        privacy_items: Sequence[Mapping[str, Any]],
        task_family: str = "general",
        policy: Mapping[str, Any] | None = None,
        source_id: str = "",
    ) -> list[AbstractionCandidate]:
        active_policy = _deep_merge(self.policy, policy or {})
        validate_policy(active_policy)
        parsed_items = privacy_items_from_dicts(privacy_items)
        source_id = source_id or stable_id(
            dialogue_text,
            [item.to_dict() for item in parsed_items],
            task_family,
        )
        if not parsed_items:
            return [
                AbstractionCandidate(
                    candidate_id=stable_id(source_id, "L0", dialogue_text),
                    source_id=source_id,
                    level="L0",
                    public_memory=dialogue_text,
                    metadata=self._metadata(task_family, "passthrough"),
                )
            ]

        self.last_backend_error = None
        generated: list[AbstractionCandidate] = []
        try:
            if self.backend == "oracle_prompt":
                generated = self._generate_oracle_candidates(
                    dialogue_text,
                    parsed_items,
                    task_family,
                    active_policy,
                    source_id,
                )
            elif self.backend == "trained_model":
                generated = self._generate_trained_candidate(
                    dialogue_text,
                    parsed_items,
                    task_family,
                    active_policy,
                    source_id,
                )
            elif self.backend == "typed_placeholder":
                generated = [
                    self._fallback_candidate(
                        "L4",
                        dialogue_text,
                        parsed_items,
                        task_family,
                        active_policy,
                        source_id,
                    )
                ]
            elif self.backend == "redaction":
                generated = [
                    self._fallback_candidate(
                        "L5",
                        dialogue_text,
                        parsed_items,
                        task_family,
                        active_policy,
                        source_id,
                    )
                ]
            else:
                generated = self._generate_heuristic_candidates(
                    dialogue_text,
                    parsed_items,
                    task_family,
                    active_policy,
                    source_id,
                )
        except Exception as exc:
            self.last_backend_error = f"{type(exc).__name__}: {exc}"
            if not self.fallback_on_error:
                raise

        candidates = list(generated)
        for level in ("L4", "L5"):
            if not any(candidate.level == level for candidate in candidates):
                candidates.append(
                    self._fallback_candidate(
                        level,
                        dialogue_text,
                        parsed_items,
                        task_family,
                        active_policy,
                        source_id,
                    )
                )
        return self._dedupe_validate_and_limit(
            candidates,
            parsed_items,
            active_policy,
        )

    def abstract(
        self,
        dialogue_text: str,
        privacy_items: Sequence[Mapping[str, Any]],
        task_family: str = "general",
        policy: Mapping[str, Any] | None = None,
        source_id: str = "",
    ) -> AbstractionCandidate:
        return self.generate_candidates(
            dialogue_text,
            privacy_items,
            task_family,
            policy,
            source_id,
        )[0]

    def _generate_heuristic_candidates(
        self,
        dialogue_text: str,
        privacy_items: Sequence[PrivacyItem],
        task_family: str,
        policy: Mapping[str, Any],
        source_id: str,
    ) -> list[AbstractionCandidate]:
        output: list[AbstractionCandidate] = []
        for level in ("L1", "L2", "L3"):
            public_memory, traces = abstract_text_heuristically(
                dialogue_text,
                privacy_items,
                level,
                task_family,
                policy,
            )
            output.append(
                self._build_candidate(
                    level,
                    public_memory,
                    traces,
                    privacy_items,
                    task_family,
                    policy,
                    source_id,
                    "heuristic",
                )
            )
        return output

    def _generate_oracle_candidates(
        self,
        dialogue_text: str,
        privacy_items: Sequence[PrivacyItem],
        task_family: str,
        policy: Mapping[str, Any],
        source_id: str,
    ) -> list[AbstractionCandidate]:
        if self.completion_fn is None:
            raise RuntimeError("oracle_prompt backend requires completion_fn")
        with open(self.prompt_path, encoding="utf-8") as handle:
            template = handle.read()
        input_payload = {
            "dialogue": dialogue_text,
            "privacy_items": [item.to_dict() for item in privacy_items],
            "task_family": task_family,
            "policy": policy,
        }
        prompt = template.format(
            input_json=json.dumps(input_payload, ensure_ascii=False, indent=2)
        )
        content = self.completion_fn(prompt)
        return parse_candidates_json(
            content,
            source_id=source_id,
            metadata=self._metadata(task_family, "oracle_prompt"),
            privacy_items=privacy_items,
            policy=policy,
        )

    def _generate_trained_candidate(
        self,
        dialogue_text: str,
        privacy_items: Sequence[PrivacyItem],
        task_family: str,
        policy: Mapping[str, Any],
        source_id: str,
    ) -> list[AbstractionCandidate]:
        completion = self.completion_fn
        if completion is None:
            completion = TransformersTextGenerator(
                self.model_name_or_path,
                revision=self.model_revision,
            )
        payload = {
            "dialogue": dialogue_text,
            "privacy_items": [item.to_dict() for item in privacy_items],
            "task_family": task_family,
            "policy": policy,
        }
        prompt = (
            "You are a local Privacy Memory Abstractor. Return strict JSON with "
            "keys level, public_memory, private_residue, and abstraction_trace. "
            "Never place raw private values in public_memory.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        parsed = parse_json_content(completion(prompt))
        if isinstance(parsed, dict) and "candidates" in parsed:
            parsed = parsed["candidates"]
        if isinstance(parsed, list):
            if not parsed:
                raise ValueError("trained model returned no candidates")
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            raise ValueError("trained model output must be a JSON object")
        level = str(parsed.get("level", "L2"))
        return [
            candidate_from_model_json(
                parsed,
                source_id,
                level,
                self._metadata(task_family, "trained_model"),
                privacy_items,
                policy,
            )
        ]

    def _fallback_candidate(
        self,
        level: str,
        dialogue_text: str,
        privacy_items: Sequence[PrivacyItem],
        task_family: str,
        policy: Mapping[str, Any],
        source_id: str,
    ) -> AbstractionCandidate:
        if level == "L4":
            public_memory, traces = abstract_text_heuristically(
                dialogue_text,
                privacy_items,
                "L4",
                task_family,
                policy,
            )
            generator = "typed_placeholder"
        elif level == "L5":
            public_memory, traces = redacted_text(dialogue_text, privacy_items)
            generator = "redaction"
        else:
            raise ValueError(f"unsupported fallback level: {level}")
        return self._build_candidate(
            level,
            public_memory,
            traces,
            privacy_items,
            task_family,
            policy,
            source_id,
            generator,
        )

    def _build_candidate(
        self,
        level: str,
        public_memory: str,
        traces: Sequence[AbstractionTrace],
        privacy_items: Sequence[PrivacyItem],
        task_family: str,
        policy: Mapping[str, Any],
        source_id: str,
        generator: str,
    ) -> AbstractionCandidate:
        metadata = self._metadata(task_family, generator)
        metadata["effective_levels"] = _effective_level_records(
            privacy_items,
            level,
            policy,
        )
        if self.last_backend_error:
            metadata["backend_error"] = self.last_backend_error
        return AbstractionCandidate(
            candidate_id=stable_id(source_id, level, public_memory),
            source_id=source_id,
            level=level,  # type: ignore[arg-type]
            public_memory=public_memory,
            private_residue=build_private_residue(privacy_items, policy),
            abstraction_trace=list(traces),
            metadata=metadata,
        )

    def _dedupe_validate_and_limit(
        self,
        candidates: Sequence[AbstractionCandidate],
        privacy_items: Sequence[PrivacyItem],
        policy: Mapping[str, Any],
    ) -> list[AbstractionCandidate]:
        seen: set[tuple[str, str]] = set()
        valid: list[AbstractionCandidate] = []
        for candidate in candidates:
            key = (candidate.level, candidate.public_memory)
            if key in seen:
                continue
            seen.add(key)
            try:
                validate_candidate_against_policy(
                    candidate,
                    privacy_items,
                    policy,
                )
            except PolicyValidationError:
                continue
            valid.append(candidate)

        for required in ("L4", "L5"):
            if not any(candidate.level == required for candidate in valid):
                raise RuntimeError(
                    f"internal error: required {required} fallback failed validation"
                )

        max_candidates = int(policy.get("max_candidates_per_turn", 6))
        semantic = [
            candidate for candidate in valid if candidate.level not in {"L4", "L5"}
        ]
        fallbacks = [
            next(candidate for candidate in valid if candidate.level == "L4"),
            next(candidate for candidate in valid if candidate.level == "L5"),
        ]
        return semantic[: max_candidates - 2] + fallbacks

    @staticmethod
    def _metadata(task_family: str, generator: str) -> dict[str, Any]:
        return {
            "task_family": task_family,
            "generator": generator,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def candidate_from_model_json(
    data: Mapping[str, Any],
    source_id: str,
    level: str,
    metadata: Mapping[str, Any],
    privacy_items: Sequence[PrivacyItem],
    policy: Mapping[str, Any],
) -> AbstractionCandidate:
    traces = [
        AbstractionTrace.from_dict(item) for item in data.get("abstraction_trace", [])
    ]
    candidate_metadata = dict(metadata)
    candidate_metadata["effective_levels"] = _effective_level_records(
        privacy_items,
        level,
        policy,
    )
    return AbstractionCandidate(
        candidate_id=stable_id(source_id, level, data.get("public_memory", "")),
        source_id=source_id,
        level=level,  # type: ignore[arg-type]
        public_memory=str(data.get("public_memory", "")),
        private_residue=build_private_residue(privacy_items, policy),
        abstraction_trace=traces,
        metadata=candidate_metadata,
    )


def parse_candidates_json(
    content: str,
    source_id: str,
    metadata: Mapping[str, Any],
    privacy_items: Sequence[PrivacyItem],
    policy: Mapping[str, Any],
) -> list[AbstractionCandidate]:
    parsed = parse_json_content(content)
    if isinstance(parsed, dict):
        parsed = parsed.get("candidates", [])
    if not isinstance(parsed, list):
        raise ValueError(
            "candidate model output must be a list or an object with candidates"
        )
    candidates: list[AbstractionCandidate] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        level = str(item.get("level", f"L{index + 1}"))
        if level not in VALID_ABSTRACTION_LEVELS:
            continue
        candidate = candidate_from_model_json(
            item,
            source_id,
            level,
            metadata,
            privacy_items,
            policy,
        )
        try:
            validate_candidate_against_policy(candidate, privacy_items, policy)
        except PolicyValidationError:
            continue
        candidates.append(candidate)
    return candidates
