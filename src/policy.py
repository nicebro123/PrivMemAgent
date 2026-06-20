from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Set

import yaml


class RouteAction(str, Enum):
    DROP = "drop"
    LOCAL_ONLY = "local_only"
    PUBLIC_ABSTRACT = "public_abstract"
    PUBLIC_REVERSIBLE = "public_reversible"


class AliasScope(str, Enum):
    TURN = "turn"
    SESSION = "session"
    TASK = "task"
    PERSISTENT = "persistent"


@dataclass(frozen=True)
class RoutingContext:
    user_id: str
    message_id: str
    message_role: str = "user"
    turn_id: Optional[str] = None
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    exact_required_types: Set[str] = field(default_factory=set)
    consented_reversible_types: Set[str] = field(default_factory=set)

    def scope_identifier(self, scope: AliasScope) -> str:
        identifiers = {
            AliasScope.TURN: self.turn_id,
            AliasScope.SESSION: self.session_id,
            AliasScope.TASK: self.task_id,
            AliasScope.PERSISTENT: "persistent",
        }
        identifier = identifiers[scope]
        if not identifier:
            raise ValueError(f"{scope.value}_id is required for {scope.value} alias scope")
        return identifier


@dataclass(frozen=True)
class RouteDecision:
    action: RouteAction
    privacy_level: str
    privacy_type: str
    policy_version: str
    rule_id: str
    reason: str
    confidence: float
    alias_scope: Optional[AliasScope] = None


@dataclass(frozen=True)
class PolicyRule:
    action: RouteAction
    alias_scope: Optional[AliasScope] = None
    reason: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.action == RouteAction.PUBLIC_REVERSIBLE and self.alias_scope is None:
            raise ValueError("public_reversible rules require alias_scope")
        if self.action != RouteAction.PUBLIC_REVERSIBLE and self.alias_scope is not None:
            raise ValueError("alias_scope is valid only for public_reversible rules")


class PrivacyPolicy:
    """Deterministic policy router used as the minimal-public-memory baseline."""

    def __init__(
        self,
        version: str,
        level_rules: Mapping[str, PolicyRule],
        type_rules: Optional[Mapping[str, PolicyRule]] = None,
        alias_scopes: Optional[Mapping[str, AliasScope]] = None,
    ):
        if not version.strip():
            raise ValueError("policy version must be non-empty")
        required_levels = {"PL2", "PL3", "PL4"}
        missing = required_levels - set(level_rules)
        if missing:
            raise ValueError(f"missing policy rules for levels: {sorted(missing)}")
        self.version = version
        self.level_rules = dict(level_rules)
        self.type_rules = dict(type_rules or {})
        self.alias_scopes = {
            "PL2": AliasScope.TASK,
            "PL3": AliasScope.SESSION,
            "PL4": AliasScope.TURN,
        }
        self.alias_scopes.update(alias_scopes or {})

    @classmethod
    def default(cls) -> "PrivacyPolicy":
        return cls(
            version="minimal-public-v1",
            level_rules={
                "PL2": PolicyRule(
                    action=RouteAction.PUBLIC_ABSTRACT,
                    reason="identifiable data is abstracted before cloud retention",
                ),
                "PL3": PolicyRule(
                    action=RouteAction.LOCAL_ONLY,
                    reason="highly sensitive data remains on the trusted edge",
                ),
                "PL4": PolicyRule(
                    action=RouteAction.DROP,
                    reason="actionable secrets must not enter long-term memory",
                ),
            },
        )

    @classmethod
    def from_dict(cls, config: Mapping) -> "PrivacyPolicy":
        policy_config = config.get("public_memory", config)
        version = str(policy_config.get("policy_version", "minimal-public-v1"))
        raw_policy = policy_config.get("policy", {})
        raw_scopes = policy_config.get("alias_scope", {})
        level_rules: Dict[str, PolicyRule] = {}
        for level in ("PL2", "PL3", "PL4"):
            action_value = raw_policy.get(level.lower(), raw_policy.get(level))
            if action_value is None:
                raise ValueError(f"missing public_memory.policy.{level.lower()}")
            action = RouteAction(action_value)
            scope = (
                AliasScope(raw_scopes.get(level.lower(), raw_scopes.get(level)))
                if action == RouteAction.PUBLIC_REVERSIBLE
                else None
            )
            level_rules[level] = PolicyRule(
                action=action,
                alias_scope=scope,
                reason=f"configured {level} policy",
            )

        type_rules = {}
        for privacy_type, raw_rule in policy_config.get("type_overrides", {}).items():
            if isinstance(raw_rule, str):
                action = RouteAction(raw_rule)
                scope = None
            else:
                action = RouteAction(raw_rule["action"])
                scope = AliasScope(raw_rule["alias_scope"]) if raw_rule.get("alias_scope") else None
            type_rules[privacy_type] = PolicyRule(
                action=action,
                alias_scope=scope,
                reason=f"configured override for {privacy_type}",
            )

        alias_scopes = {
            level: AliasScope(raw_scopes[level.lower()])
            for level in ("PL2", "PL3", "PL4")
            if level.lower() in raw_scopes
        }
        return cls(version, level_rules, type_rules, alias_scopes or None)

    @classmethod
    def from_yaml(cls, path: str) -> "PrivacyPolicy":
        with open(path, encoding="utf-8") as source:
            return cls.from_dict(yaml.safe_load(source) or {})

    def route(self, privacy_item: Mapping[str, str], context: RoutingContext) -> RouteDecision:
        privacy_level = privacy_item["privacy_level"]
        privacy_type = privacy_item["privacy_type"]
        if privacy_level not in self.level_rules:
            raise ValueError(f"unsupported privacy level: {privacy_level}")

        rule = self.type_rules.get(privacy_type, self.level_rules[privacy_level])
        rule_id = (
            f"type:{privacy_type}" if privacy_type in self.type_rules else f"level:{privacy_level}"
        )

        if privacy_level == "PL4":
            return RouteDecision(
                action=RouteAction.DROP,
                privacy_level=privacy_level,
                privacy_type=privacy_type,
                policy_version=self.version,
                rule_id="mandatory:PL4-drop",
                reason="actionable secrets are never retained or made reversible",
                confidence=1.0,
            )

        if privacy_type in context.exact_required_types:
            if privacy_type in context.consented_reversible_types:
                scope = rule.alias_scope or self.alias_scopes[privacy_level]
                return RouteDecision(
                    action=RouteAction.PUBLIC_REVERSIBLE,
                    alias_scope=scope,
                    privacy_level=privacy_level,
                    privacy_type=privacy_type,
                    policy_version=self.version,
                    rule_id="exact-required-with-consent",
                    reason="future task requires exact value and the user permitted reversible aliasing",
                    confidence=1.0,
                )
            return RouteDecision(
                action=RouteAction.LOCAL_ONLY,
                privacy_level=privacy_level,
                privacy_type=privacy_type,
                policy_version=self.version,
                rule_id="exact-required-without-consent",
                reason="exact value is needed but reversible cloud exposure lacks consent",
                confidence=1.0,
            )

        return RouteDecision(
            action=rule.action,
            alias_scope=rule.alias_scope,
            privacy_level=privacy_level,
            privacy_type=privacy_type,
            policy_version=self.version,
            rule_id=rule_id,
            reason=rule.reason,
            confidence=rule.confidence,
        )

    def route_all(
        self, privacy_items: Iterable[Mapping[str, str]], context: RoutingContext
    ) -> list[RouteDecision]:
        return [self.route(item, context) for item in privacy_items]
