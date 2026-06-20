from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from cryptography.fernet import Fernet

from src.policy import AliasScope, RoutingContext
from src.privacy_masking import PrivacyStore, unmask_dialogue


class ScopedAliasRouter:
    """Creates stable aliases inside a scope and rotates them across scopes."""

    def __init__(
        self,
        db_path: str,
        encryption_key: Optional[str] = None,
        key_path: Optional[str] = None,
        mask_mode: str = "type_specific",
    ):
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.mask_mode = mask_mode
        self._key_path = key_path
        if encryption_key:
            Fernet(encryption_key.encode("ascii"))
            self._key = encryption_key
        else:
            self._key = self._load_or_create_shared_key(key_path)

    def _load_or_create_shared_key(self, key_path: Optional[str]) -> str:
        if key_path:
            path = Path(key_path).expanduser().resolve()
        else:
            identity = self.db_path.encode("utf-8")
            key_name = hashlib.sha256(identity).hexdigest() + ".key"
            path = Path.home() / ".config" / "memprivacy" / "keys" / key_name
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
            with os.fdopen(file_descriptor, "wb") as key_file:
                key_file.write(Fernet.generate_key() + b"\n")
        path.chmod(0o600)
        key = path.read_text(encoding="ascii").strip()
        Fernet(key.encode("ascii"))
        return key

    @property
    def fingerprint_key(self) -> bytes:
        return self._key.encode("ascii")

    def fingerprint(self, value: str) -> str:
        return hmac.new(
            self.fingerprint_key,
            value.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _namespace(self, context: RoutingContext, scope: AliasScope) -> str:
        scope_id = context.scope_identifier(scope)
        return f"{context.user_id}:{scope.value}:{scope_id}"

    def _store(self, namespace: str) -> PrivacyStore:
        return PrivacyStore(
            db_path=self.db_path,
            mask_mode=self.mask_mode,
            namespace=namespace,
            encryption_key=self._key,
        )

    def get_alias(
        self,
        original_text: str,
        privacy_type: str,
        privacy_level: str,
        scope: AliasScope,
        context: RoutingContext,
    ) -> Tuple[str, str]:
        namespace = self._namespace(context, scope)
        with self._store(namespace) as store:
            alias = store.get_or_create(original_text, privacy_type, privacy_level)
        return alias, context.scope_identifier(scope)

    def store_local(
        self,
        original_text: str,
        privacy_type: str,
        privacy_level: str,
        context: RoutingContext,
    ) -> None:
        namespace = f"{context.user_id}:local-only"
        with self._store(namespace) as store:
            store.get_or_create(original_text, privacy_type, privacy_level)

    def query_local(
        self,
        user_id: str,
        privacy_type: Optional[str] = None,
    ) -> List[Dict]:
        namespace = f"{user_id}:local-only"
        with self._store(namespace) as store:
            if privacy_type is None:
                return store.get_all()
            return store.query_by_privacy_type(privacy_type)

    def restore(
        self,
        text: str,
        context: RoutingContext,
        additional_scopes: Optional[Iterable[Tuple[AliasScope, str]]] = None,
    ) -> str:
        scope_pairs = [
            (scope, context.scope_identifier(scope))
            for scope in AliasScope
            if scope == AliasScope.PERSISTENT
            or {
                AliasScope.TURN: context.turn_id,
                AliasScope.SESSION: context.session_id,
                AliasScope.TASK: context.task_id,
            }[scope]
        ]
        scope_pairs.extend(additional_scopes or [])

        restored = text
        seen = set()
        for scope, scope_id in scope_pairs:
            key = (scope, scope_id)
            if key in seen:
                continue
            seen.add(key)
            namespace = f"{context.user_id}:{scope.value}:{scope_id}"
            with self._store(namespace) as store:
                restored = unmask_dialogue(restored, store)
        return restored
