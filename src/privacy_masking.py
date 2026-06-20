import hashlib
import hmac
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import json_repair
import yaml
from cryptography.fernet import Fernet, InvalidToken
from openai import OpenAI

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent
_ALLOWED_PRIVACY_LEVELS = {"PL2", "PL3", "PL4"}


def load_yaml_config(config_path: Optional[str] = None) -> dict:
    path = Path(config_path) if config_path else _MODULE_DIR / "privacy_config.yaml"
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["_config_dir"] = str(path.parent)
    return config


def _is_local_endpoint(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _validate_llm_endpoint(config: dict) -> None:
    llm_config = config.get("llm", {})
    base_url = str(llm_config.get("base_url", "")).strip()
    if not base_url:
        raise ValueError("llm.base_url is required and must point to the local detector")
    if _is_local_endpoint(base_url):
        return
    if not llm_config.get("allow_remote", False):
        raise ValueError(
            "Refusing to send raw privacy-bearing text to a remote detector. "
            "Use a localhost endpoint or explicitly set llm.allow_remote=true."
        )
    logger.warning(
        "Remote privacy detection is enabled. Raw dialogue leaves the device, "
        "so the architecture-level isolation guarantee does not apply."
    )


class PrivacyStore:
    """Encrypted, namespace-isolated storage for reversible placeholders."""

    _schema_locks_guard = threading.Lock()
    _schema_locks: Dict[str, threading.Lock] = {}

    def __init__(
        self,
        db_path: str = "privacy_store.db",
        mask_mode: str = "type_specific",
        namespace: str = "default",
        encryption_key: Optional[str] = None,
        key_path: Optional[str] = None,
    ):
        if mask_mode not in {"type_specific", "generic"}:
            raise ValueError("mask_mode must be 'type_specific' or 'generic'")
        if not namespace or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")

        self.db_path = db_path
        self.mask_mode = mask_mode
        self.namespace = namespace.strip()
        self._lock = threading.RLock()
        self._key = self._load_or_create_key(encryption_key, key_path)
        self._fernet = Fernet(self._key)
        self._hmac_key = self._key
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=30000")
        for attempt in range(10):
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._schema_lock():
            self._create_or_migrate_table()
        self._secure_file_permissions()

    def _schema_lock(self) -> threading.Lock:
        if self.db_path == ":memory:":
            return threading.Lock()
        identity = str(Path(self.db_path).expanduser().resolve())
        with self._schema_locks_guard:
            return self._schema_locks.setdefault(identity, threading.Lock())

    def _load_or_create_key(self, encryption_key: Optional[str], key_path: Optional[str]) -> bytes:
        configured = encryption_key or os.getenv("MEMPRIVACY_STORE_KEY")
        if configured:
            key = configured.encode("ascii")
            Fernet(key)
            return key

        if self.db_path == ":memory:":
            return Fernet.generate_key()

        if key_path:
            path = Path(key_path).expanduser().resolve()
        else:
            db_identity = str(Path(self.db_path).expanduser().resolve()).encode("utf-8")
            key_name = hashlib.sha256(db_identity).hexdigest() + ".key"
            path = Path.home() / ".config" / "memprivacy" / "keys" / key_name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            key = path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            with os.fdopen(fd, "wb") as f:
                f.write(key + b"\n")
        os.chmod(path, 0o600)
        Fernet(key)
        return key

    def _secure_file_permissions(self) -> None:
        if self.db_path == ":memory:":
            return
        path = Path(self.db_path)
        if path.exists():
            os.chmod(path, 0o600)

    def _create_table(self, commit: bool = True) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS privacy_items (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace           TEXT NOT NULL,
                original_hash       TEXT NOT NULL,
                original_ciphertext BLOB NOT NULL,
                privacy_type        TEXT NOT NULL,
                privacy_level       TEXT NOT NULL,
                mask                TEXT NOT NULL,
                UNIQUE(namespace, original_hash, privacy_type),
                UNIQUE(namespace, mask)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_privacy_type ON privacy_items(namespace, privacy_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_privacy_level "
            "ON privacy_items(namespace, privacy_level)"
        )
        self._conn.execute("PRAGMA user_version=2")
        if commit:
            self._conn.commit()

    def _create_or_migrate_table(self) -> None:
        existing = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='privacy_items'"
        ).fetchone()
        if not existing:
            self._create_table()
            return

        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(privacy_items)").fetchall()
        }
        if {"namespace", "original_hash", "original_ciphertext"} <= columns:
            schema_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if schema_version < 2:
                self._migrate_secure_v1_table()
                return
            self._create_table()
            return

        legacy_rows = self._conn.execute("SELECT * FROM privacy_items").fetchall()
        with self._conn:
            self._conn.execute("ALTER TABLE privacy_items RENAME TO privacy_items_legacy")
            self._create_table(commit=False)
            for row in legacy_rows:
                original = row["original_text"]
                self._conn.execute(
                    """
                    INSERT INTO privacy_items (
                        namespace, original_hash, original_ciphertext,
                        privacy_type, privacy_level, mask
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.namespace,
                        self._hash_original(original),
                        self._encrypt(original),
                        row["privacy_type"],
                        row["privacy_level"],
                        row["mask"],
                    ),
                )
            self._conn.execute("DROP TABLE privacy_items_legacy")
        self._create_table()
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._conn.execute("VACUUM")

    def _migrate_secure_v1_table(self) -> None:
        rows = self._conn.execute("SELECT * FROM privacy_items").fetchall()
        with self._conn:
            self._conn.execute("ALTER TABLE privacy_items RENAME TO privacy_items_secure_v1")
            self._create_table(commit=False)
            for row in rows:
                self._conn.execute(
                    """
                    INSERT INTO privacy_items (
                        namespace, original_hash, original_ciphertext,
                        privacy_type, privacy_level, mask
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["namespace"],
                        row["original_hash"],
                        row["original_ciphertext"],
                        row["privacy_type"],
                        row["privacy_level"],
                        row["mask"],
                    ),
                )
            self._conn.execute("DROP TABLE privacy_items_secure_v1")
        self._create_table()
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._conn.execute("VACUUM")

    def _hash_original(self, original_text: str) -> str:
        payload = f"{self.namespace}\0{original_text}".encode("utf-8")
        return hmac.new(self._hmac_key, payload, hashlib.sha256).hexdigest()

    def _encrypt(self, value: str) -> bytes:
        return self._fernet.encrypt(value.encode("utf-8"))

    def _decrypt(self, value: bytes) -> str:
        try:
            return self._fernet.decrypt(value).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Privacy store key does not match the database") from exc

    @staticmethod
    def _type_to_mask_prefix(privacy_type: str) -> str:
        prefix = re.sub(r"[^A-Za-z0-9]+", "_", privacy_type).strip("_")
        return prefix or "PRIVATE"

    def _next_mask(self, privacy_type: str) -> str:
        prefix = "MASK" if self.mask_mode == "generic" else self._type_to_mask_prefix(privacy_type)
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM privacy_items WHERE namespace = ? AND privacy_type = ?",
            (self.namespace, privacy_type),
        ).fetchone()
        sequence = (row["cnt"] if row else 0) + 1
        token = secrets.token_hex(6)
        return f"<MPM_{prefix}_{sequence}_{token}>"

    @staticmethod
    def _validate_record(original_text: str, privacy_type: str, privacy_level: str) -> None:
        if not isinstance(original_text, str) or not original_text.strip():
            raise ValueError("original_text must be a non-empty string")
        if not isinstance(privacy_type, str) or not privacy_type.strip():
            raise ValueError("privacy_type must be a non-empty string")
        if privacy_level not in _ALLOWED_PRIVACY_LEVELS:
            raise ValueError("privacy_level must be one of PL2, PL3, or PL4")

    def get_or_create(self, original_text: str, privacy_type: str, privacy_level: str) -> str:
        self._validate_record(original_text, privacy_type, privacy_level)
        original_hash = self._hash_original(original_text)

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT mask, privacy_type, privacy_level FROM privacy_items "
                    "WHERE namespace = ? AND original_hash = ? AND privacy_type = ?",
                    (self.namespace, original_hash, privacy_type),
                ).fetchone()
                if row:
                    if int(privacy_level[-1]) > int(row["privacy_level"][-1]):
                        self._conn.execute(
                            "UPDATE privacy_items SET privacy_level = ? "
                            "WHERE namespace = ? AND original_hash = ? AND privacy_type = ?",
                            (
                                privacy_level,
                                self.namespace,
                                original_hash,
                                privacy_type,
                            ),
                        )
                    self._conn.commit()
                    return row["mask"]

                mask = self._next_mask(privacy_type)
                self._conn.execute(
                    """
                    INSERT INTO privacy_items (
                        namespace, original_hash, original_ciphertext,
                        privacy_type, privacy_level, mask
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.namespace,
                        original_hash,
                        self._encrypt(original_text),
                        privacy_type,
                        privacy_level,
                        mask,
                    ),
                )
                self._conn.commit()
                return mask
            except Exception:
                self._conn.rollback()
                raise

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        result = dict(row)
        result["original_text"] = self._decrypt(result.pop("original_ciphertext"))
        result.pop("original_hash", None)
        return result

    def query_by_mask(self, mask: str) -> Optional[Dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM privacy_items WHERE namespace = ? AND mask = ?",
                (self.namespace, mask),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def query_by_original_text(self, original_text: str) -> Optional[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM privacy_items WHERE namespace = ? AND original_hash = ?",
                (self.namespace, self._hash_original(original_text)),
            ).fetchall()
            if not rows:
                return None
            row = max(rows, key=lambda item: int(item["privacy_level"][-1]))
            return self._row_to_dict(row)

    def query_by_privacy_type(self, privacy_type: str) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM privacy_items WHERE namespace = ? AND privacy_type = ?",
                (self.namespace, privacy_type),
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def query_by_privacy_level(self, privacy_level: str) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM privacy_items WHERE namespace = ? AND privacy_level = ?",
                (self.namespace, privacy_level),
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_all(self) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM privacy_items WHERE namespace = ?", (self.namespace,)
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def validate_privacy_items(
    privacy_items: Iterable[Dict],
    dialogue_text: Optional[str] = None,
    strict: bool = True,
) -> List[Dict]:
    if not isinstance(privacy_items, list):
        raise ValueError("privacy_items must be a JSON list")

    validated = []
    seen = set()
    for index, item in enumerate(privacy_items):
        try:
            if not isinstance(item, dict):
                raise ValueError("item must be an object")
            original = item["original_text"]
            privacy_type = item["privacy_type"]
            privacy_level = item["privacy_level"]
            PrivacyStore._validate_record(original, privacy_type, privacy_level)
            if dialogue_text is not None and original not in dialogue_text:
                raise ValueError("original_text does not occur in the dialogue")
        except (KeyError, ValueError, TypeError) as exc:
            if strict:
                raise ValueError(f"Invalid privacy item at index {index}: {exc}") from exc
            logger.warning("Ignoring invalid privacy item %s: %s", index, exc)
            continue

        key = (original, privacy_type, privacy_level)
        if key not in seen:
            validated.append(
                {
                    "original_text": original,
                    "privacy_type": privacy_type,
                    "privacy_level": privacy_level,
                }
            )
            seen.add(key)
    return validated


def _find_non_overlapping_replacements(
    dialogue_text: str, replacements: List[Tuple[str, str]]
) -> List[Tuple[int, int, str]]:
    candidates = []
    for original, mask in replacements:
        start = 0
        while True:
            start = dialogue_text.find(original, start)
            if start < 0:
                break
            candidates.append((start, start + len(original), mask, len(original)))
            start += max(1, len(original))

    candidates.sort(key=lambda item: (-item[3], item[0]))
    selected = []
    for start, end, mask, _ in candidates:
        if any(start < other_end and end > other_start for other_start, other_end, _ in selected):
            continue
        selected.append((start, end, mask))
    return sorted(selected, reverse=True)


def mask_dialogue(
    dialogue_text: str,
    privacy_items: List[Dict],
    store: PrivacyStore,
    mask_levels: Optional[List[str]] = None,
    strict: bool = True,
) -> str:
    if not isinstance(dialogue_text, str):
        raise TypeError("dialogue_text must be a string")
    levels = set(mask_levels or ["PL3", "PL4"])
    invalid_levels = levels - _ALLOWED_PRIVACY_LEVELS
    if invalid_levels:
        raise ValueError(f"Unsupported mask levels: {sorted(invalid_levels)}")

    selected_items = {}
    for item in validate_privacy_items(privacy_items, dialogue_text, strict=strict):
        if item["privacy_level"] not in levels:
            continue
        current = selected_items.get(item["original_text"])
        candidate_rank = (-int(item["privacy_level"][-1]), item["privacy_type"])
        current_rank = (
            (-int(current["privacy_level"][-1]), current["privacy_type"]) if current else None
        )
        if current is None or candidate_rank < current_rank:
            selected_items[item["original_text"]] = item

    replacements = []
    for item in selected_items.values():
        mask = store.get_or_create(
            item["original_text"], item["privacy_type"], item["privacy_level"]
        )
        if mask in dialogue_text:
            raise ValueError(f"Generated mask already exists in dialogue: {mask}")
        replacements.append((item["original_text"], mask))

    masked_text = dialogue_text
    for start, end, mask in _find_non_overlapping_replacements(dialogue_text, replacements):
        masked_text = masked_text[:start] + mask + masked_text[end:]
    return masked_text


def complete_mask_dialogue(
    dialogue_text: str,
    privacy_items: List[Dict],
    mask_levels: Optional[List[str]] = None,
    strict: bool = True,
) -> str:
    levels = set(mask_levels or ["PL3", "PL4"])
    invalid_levels = levels - _ALLOWED_PRIVACY_LEVELS
    if invalid_levels:
        raise ValueError(f"Unsupported mask levels: {sorted(invalid_levels)}")
    replacements = [
        (item["original_text"], "***")
        for item in validate_privacy_items(privacy_items, dialogue_text, strict=strict)
        if item["privacy_level"] in levels
    ]
    masked_text = dialogue_text
    for start, end, mask in _find_non_overlapping_replacements(dialogue_text, replacements):
        masked_text = masked_text[:start] + mask + masked_text[end:]
    return masked_text


def unmask_dialogue(masked_text: str, store: PrivacyStore) -> str:
    restored = masked_text
    records = sorted(store.get_all(), key=lambda record: len(record["mask"]), reverse=True)
    for record in records:
        restored = restored.replace(record["mask"], record["original_text"])
    return restored


def _build_llm_client(config: dict) -> OpenAI:
    _validate_llm_endpoint(config)
    llm_config = config["llm"]
    return OpenAI(
        base_url=llm_config["base_url"],
        api_key=llm_config.get("api_key") or "local",
    )


def _load_prompt_template(config: dict) -> str:
    prompt_path = Path(config["privacy"]["prompt_path"]).expanduser()
    if not prompt_path.is_absolute():
        config_dir = Path(config.get("_config_dir", _MODULE_DIR))
        prompt_path = config_dir / prompt_path
    return prompt_path.resolve().read_text(encoding="utf-8")


def detect_and_mask_dialogue(
    message_text: str,
    config: dict,
    store: PrivacyStore,
    mask_levels: Optional[List[str]] = None,
    real_name: str = "",
) -> Tuple[str, List[Dict]]:
    if mask_levels is None:
        mask_levels = config.get("privacy", {}).get("mask_levels", ["PL3", "PL4"])

    client = _build_llm_client(config)
    prompt_template = _load_prompt_template(config)
    user_prompt = prompt_template.format(
        real_name=real_name,
        input_dialogue_content=message_text,
    )
    llm_config = config["llm"]
    request_params = {
        "model": llm_config["model"],
        "messages": [{"role": "user", "content": user_prompt}],
    }
    for key, converter in (
        ("max_tokens", int),
        ("temperature", float),
        ("timeout", float),
    ):
        if llm_config.get(key) is not None:
            request_params[key] = converter(llm_config[key])

    attempts = int(llm_config.get("retry_times", 3))
    wait_lower = float(llm_config.get("wait_time_lower", 1))
    wait_upper = float(llm_config.get("wait_time_upper", 10))
    last_error = None
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(**request_params)
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Privacy detector returned empty content")
            content = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE
            ).strip()
            parsed = json_repair.loads(content)
            privacy_items = validate_privacy_items(parsed, dialogue_text=message_text, strict=True)
            masked_text = mask_dialogue(
                message_text, privacy_items, store, mask_levels, strict=True
            )
            return masked_text, privacy_items
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            delay = min(wait_upper, wait_lower * (2**attempt))
            logger.warning(
                "Privacy detector attempt %s/%s failed: %s",
                attempt + 1,
                attempts,
                exc,
            )
            time.sleep(delay)
    raise RuntimeError("Privacy detection failed after all retries") from last_error
