from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

from src.alias_router import ScopedAliasRouter
from src.policy import AliasScope, RoutingContext


@dataclass(frozen=True)
class QueryAlias:
    privacy_type: str
    privacy_level: str
    alias: str
    scope: AliasScope
    scope_id: str


@dataclass(frozen=True)
class PreparedQuery:
    original_query_fingerprint: str
    cloud_query: str
    aliases: tuple[QueryAlias, ...]
    missing_types: tuple[str, ...]


class EdgeQueryBroker:
    """Hydrates exact private facts as task-scoped aliases at query time."""

    def __init__(self, alias_router: ScopedAliasRouter):
        self.alias_router = alias_router

    def prepare(
        self,
        query_text: str,
        required_privacy_types: Iterable[str],
        context: RoutingContext,
        maximum_values_per_type: int = 5,
    ) -> PreparedQuery:
        if not context.task_id:
            raise ValueError("task_id is required for query-time hydration")
        required_types = list(dict.fromkeys(required_privacy_types))
        unauthorized_types = [
            privacy_type
            for privacy_type in required_types
            if privacy_type not in context.exact_required_types
            or privacy_type not in context.consented_reversible_types
        ]
        if unauthorized_types:
            raise PermissionError(
                "query-time private hydration requires both an explicit task need "
                f"and reversible-use consent for: {sorted(unauthorized_types)}"
            )
        alias_records: List[QueryAlias] = []
        missing = []

        for privacy_type in required_types:
            records = self.alias_router.query_local(context.user_id, privacy_type)
            records = [record for record in records if record["privacy_level"] != "PL4"]
            if not records:
                missing.append(privacy_type)
                continue
            records = sorted(
                records,
                key=lambda record: (
                    -int(record["privacy_level"][-1]),
                    record["id"],
                ),
            )[:maximum_values_per_type]
            for record in records:
                alias, scope_id = self.alias_router.get_alias(
                    record["original_text"],
                    record["privacy_type"],
                    record["privacy_level"],
                    AliasScope.TASK,
                    context,
                )
                alias_records.append(
                    QueryAlias(
                        privacy_type=record["privacy_type"],
                        privacy_level=record["privacy_level"],
                        alias=alias,
                        scope=AliasScope.TASK,
                        scope_id=scope_id,
                    )
                )

        context_lines = [f"- {record.privacy_type}: {record.alias}" for record in alias_records]
        cloud_query = query_text
        if context_lines:
            cloud_query += (
                "\n\nEdge-provided private references follow. "
                "Use the aliases verbatim; do not infer their hidden values.\n"
                + "\n".join(context_lines)
            )
        return PreparedQuery(
            original_query_fingerprint=self.alias_router.fingerprint(query_text),
            cloud_query=cloud_query,
            aliases=tuple(alias_records),
            missing_types=tuple(missing),
        )

    def restore_response(
        self,
        cloud_response: str,
        context: RoutingContext,
        prepared_query: Optional[PreparedQuery] = None,
    ) -> str:
        additional_scopes = []
        if prepared_query:
            additional_scopes = [(alias.scope, alias.scope_id) for alias in prepared_query.aliases]
        return self.alias_router.restore(
            cloud_response,
            context,
            additional_scopes=additional_scopes,
        )

    @staticmethod
    def alias_manifest(prepared_query: PreparedQuery) -> List[Mapping[str, str]]:
        return [
            {
                "privacy_type": alias.privacy_type,
                "privacy_level": alias.privacy_level,
                "alias": alias.alias,
                "scope": alias.scope.value,
                "scope_id": alias.scope_id,
            }
            for alias in prepared_query.aliases
        ]
