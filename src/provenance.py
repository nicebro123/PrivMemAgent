from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from src.policy import AliasScope, RouteAction, RouteDecision


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProvenanceRecord:
    record_id: str
    user_id: str
    source_message_id: str
    source_item_index: int
    policy_version: str
    route_action: RouteAction
    rule_id: str
    privacy_level: str
    privacy_type: str
    representation_type: str
    public_text: Optional[str]
    alias_scope: Optional[AliasScope]
    scope_id: Optional[str]
    created_at: str
    expires_at: Optional[str] = None
    cloud_memory_ids: tuple[str, ...] = ()
    revoked_at: Optional[str] = None
    revocation_reason: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


class ProvenanceStore:
    """Stores public-memory lineage without storing raw sensitive values."""

    def __init__(self, db_path: str):
        path = Path(db_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            pass
        else:
            os.close(file_descriptor)
        path.chmod(0o600)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS public_memory_provenance (
                record_id          TEXT PRIMARY KEY,
                user_id            TEXT NOT NULL,
                source_message_id   TEXT NOT NULL,
                source_item_index   INTEGER NOT NULL,
                policy_version      TEXT NOT NULL,
                route_action       TEXT NOT NULL,
                rule_id            TEXT NOT NULL,
                privacy_level       TEXT NOT NULL,
                privacy_type        TEXT NOT NULL,
                representation_type TEXT NOT NULL,
                public_text         TEXT,
                alias_scope         TEXT,
                scope_id            TEXT,
                created_at          TEXT NOT NULL,
                expires_at          TEXT,
                cloud_memory_ids    TEXT NOT NULL,
                revoked_at          TEXT,
                revocation_reason   TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_provenance_source "
            "ON public_memory_provenance(user_id, source_message_id)"
        )
        self._conn.commit()

    def add(self, record: ProvenanceRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO public_memory_provenance (
                    record_id, user_id, source_message_id, source_item_index,
                    policy_version, route_action, rule_id, privacy_level,
                    privacy_type, representation_type, public_text, alias_scope,
                    scope_id, created_at, expires_at, cloud_memory_ids,
                    revoked_at, revocation_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.user_id,
                    record.source_message_id,
                    record.source_item_index,
                    record.policy_version,
                    record.route_action.value,
                    record.rule_id,
                    record.privacy_level,
                    record.privacy_type,
                    record.representation_type,
                    record.public_text,
                    record.alias_scope.value if record.alias_scope else None,
                    record.scope_id,
                    record.created_at,
                    record.expires_at,
                    json.dumps(record.cloud_memory_ids),
                    record.revoked_at,
                    record.revocation_reason,
                ),
            )

    def attach_cloud_memory_id(self, record_id: str, cloud_memory_id: str) -> None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT cloud_memory_ids FROM public_memory_provenance WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown provenance record: {record_id}")
            identifiers = list(json.loads(row["cloud_memory_ids"]))
            if cloud_memory_id not in identifiers:
                identifiers.append(cloud_memory_id)
            self._conn.execute(
                "UPDATE public_memory_provenance SET cloud_memory_ids = ? WHERE record_id = ?",
                (json.dumps(identifiers), record_id),
            )

    def revoke_by_source(
        self,
        user_id: str,
        source_message_ids: Iterable[str],
        reason: str,
    ) -> List[str]:
        source_ids = list(source_message_ids)
        if not source_ids:
            return []
        rows = []
        with self._lock, self._conn:
            now = _utc_now()
            for source_message_id in source_ids:
                rows.extend(
                    self._conn.execute(
                        "SELECT record_id, cloud_memory_ids "
                        "FROM public_memory_provenance "
                        "WHERE user_id = ? AND source_message_id = ? "
                        "AND revoked_at IS NULL",
                        (user_id, source_message_id),
                    ).fetchall()
                )
                self._conn.execute(
                    "UPDATE public_memory_provenance "
                    "SET revoked_at = ?, revocation_reason = ? "
                    "WHERE user_id = ? AND source_message_id = ? "
                    "AND revoked_at IS NULL",
                    (now, reason, user_id, source_message_id),
                )
        return sorted(
            {cloud_id for row in rows for cloud_id in json.loads(row["cloud_memory_ids"])}
        )

    def get(self, record_id: str) -> Optional[ProvenanceRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM public_memory_provenance WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_active(self, user_id: str) -> List[ProvenanceRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM public_memory_provenance "
                "WHERE user_id = ? AND revoked_at IS NULL ORDER BY created_at",
                (user_id,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ProvenanceRecord:
        data = dict(row)
        data["route_action"] = RouteAction(data["route_action"])
        data["alias_scope"] = AliasScope(data["alias_scope"]) if data["alias_scope"] else None
        data["cloud_memory_ids"] = tuple(json.loads(data["cloud_memory_ids"]))
        return ProvenanceRecord(**data)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def build_provenance_record(
    record_id: str,
    context_user_id: str,
    source_message_id: str,
    source_item_index: int,
    decision: RouteDecision,
    representation_type: str,
    public_text: Optional[str],
    scope_id: Optional[str],
    expires_at: Optional[str] = None,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        record_id=record_id,
        user_id=context_user_id,
        source_message_id=source_message_id,
        source_item_index=source_item_index,
        policy_version=decision.policy_version,
        route_action=decision.action,
        rule_id=decision.rule_id,
        privacy_level=decision.privacy_level,
        privacy_type=decision.privacy_type,
        representation_type=representation_type,
        public_text=public_text,
        alias_scope=decision.alias_scope,
        scope_id=scope_id,
        created_at=_utc_now(),
        expires_at=expires_at,
    )
