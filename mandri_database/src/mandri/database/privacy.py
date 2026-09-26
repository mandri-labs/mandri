import asyncio
import hashlib
import json
import secrets
from dataclasses import dataclass, field
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from mandri.core.clock import system_now_ms
from mandri.core.ports.database import DatabasePort, SqlParams
from mandri.core.ports.privacy_keys import PrivacyKeyProvider
from mandri.core.types.execution import ProtectionError
from mandri.database.errors import DatabaseError

_VERSION = 1


class PrivacyRevisionConflict(ProtectionError):
    def __init__(self) -> None:
        super().__init__("privacy_revision_conflict", "Privacy state changed concurrently")


@dataclass(frozen=True)
class StoredPrivacyScope:
    scope_id: str
    revision: int
    payload: dict[str, Any] = field(repr=False)
    key: bytes = field(repr=False)
    fingerprint: bytes = field(default=b"", repr=False)


def _seal(key: bytes, data: bytes, associated: bytes) -> bytes:
    nonce = secrets.token_bytes(12)
    return nonce + AESGCM(key).encrypt(nonce, data, associated)


def _open(key: bytes, data: bytes, associated: bytes) -> bytes:
    try:
        return AESGCM(key).decrypt(data[:12], data[12:], associated)
    except (InvalidTag, ValueError):
        raise ProtectionError(
            "privacy_state_unavailable", "Privacy state authentication failed"
        ) from None


def _associated(scope_id: str, purpose: str) -> bytes:
    return f"mandri-privacy:{_VERSION}:{scope_id}:{purpose}".encode()


def _serialize(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _fingerprint(key: bytes, encrypted: bytes) -> bytes:
    return hashlib.sha256(key + encrypted).digest()


class PrivacyRepository:
    def __init__(self, db: DatabasePort, keys: PrivacyKeyProvider) -> None:
        self._db = db
        self._keys = keys

    async def _fetch_one(self, sql: str, params: SqlParams = ()) -> dict[str, Any] | None:
        try:
            return await self._db.fetch_one(sql, params)
        except DatabaseError:
            raise ProtectionError(
                "privacy_state_unavailable", "Privacy storage is unavailable"
            ) from None

    async def _execute(self, sql: str, params: SqlParams = ()) -> None:
        try:
            await self._db.execute(sql, params)
        except DatabaseError:
            raise ProtectionError(
                "privacy_state_unavailable", "Privacy storage is unavailable"
            ) from None

    async def readiness(self) -> None:
        await self._wrapping_key()

    async def _wrapping_key(self) -> bytes:
        existing = await self._fetch_one("SELECT id FROM privacy_scope LIMIT 1")
        return await asyncio.to_thread(self._keys.load, create=existing is None)

    async def create(self, scope_id: str, payload: dict[str, Any]) -> StoredPrivacyScope:
        wrapping_key = await self._wrapping_key()
        key = secrets.token_bytes(32)
        now = int(system_now_ms())
        wrapped = _seal(wrapping_key, key, _associated(scope_id, "key"))
        encrypted = _seal(key, _serialize(payload), _associated(scope_id, "payload:0"))
        await self._execute(
            "INSERT INTO privacy_scope"
            " (id, version, revision, wrapped_key, payload, created_at, updated_at)"
            " VALUES (?, ?, 0, ?, ?, ?, ?)",
            (scope_id, _VERSION, wrapped, encrypted, now, now),
        )
        return StoredPrivacyScope(scope_id, 0, payload, key, _fingerprint(key, encrypted))

    async def revision(self, scope_id: str) -> tuple[int, bytes]:
        row = await self._fetch_one(
            "SELECT version, revision, wrapped_key, payload FROM privacy_scope WHERE id = ?",
            (scope_id,),
        )
        if row is None or row["version"] != _VERSION:
            raise ProtectionError("privacy_state_unavailable", "Privacy scope is unavailable")
        wrapping_key = await asyncio.to_thread(self._keys.load)
        key = _open(wrapping_key, row["wrapped_key"], _associated(scope_id, "key"))
        fingerprint = await asyncio.to_thread(_fingerprint, key, row["payload"])
        return int(row["revision"]), fingerprint

    async def load(self, scope_id: str) -> StoredPrivacyScope:
        row = await self._fetch_one("SELECT * FROM privacy_scope WHERE id = ?", (scope_id,))
        if row is None or row["version"] != _VERSION:
            raise ProtectionError("privacy_state_unavailable", "Privacy scope is unavailable")
        revision = int(row["revision"])
        wrapping_key = await asyncio.to_thread(self._keys.load)
        key = _open(wrapping_key, row["wrapped_key"], _associated(scope_id, "key"))
        raw = _open(key, row["payload"], _associated(scope_id, f"payload:{revision}"))
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise ProtectionError("privacy_state_unavailable", "Privacy state is invalid") from None
        if not isinstance(payload, dict):
            raise ProtectionError("privacy_state_unavailable", "Privacy state is invalid")
        return StoredPrivacyScope(
            scope_id, revision, payload, key, _fingerprint(key, row["payload"])
        )

    async def save(self, scope: StoredPrivacyScope, payload: dict[str, Any]) -> StoredPrivacyScope:
        revision = scope.revision + 1
        encrypted = _seal(
            scope.key, _serialize(payload), _associated(scope.scope_id, f"payload:{revision}")
        )
        row = await self._fetch_one(
            "UPDATE privacy_scope SET payload = ?, revision = ?, updated_at = ?"
            " WHERE id = ? AND revision = ? AND version = ? RETURNING revision",
            (
                encrypted,
                revision,
                int(system_now_ms()),
                scope.scope_id,
                scope.revision,
                _VERSION,
            ),
        )
        if row is None:
            raise PrivacyRevisionConflict()
        return StoredPrivacyScope(
            scope.scope_id, revision, payload, scope.key, _fingerprint(scope.key, encrypted)
        )

    async def delete(self, scope_id: str) -> None:
        await self._execute(
            "DELETE FROM privacy_scope WHERE id = ?"
            " AND NOT EXISTS (SELECT 1 FROM session WHERE privacy_scope_id = ?)"
            " AND NOT EXISTS (SELECT 1 FROM gateway_route WHERE privacy_scope_id = ?)",
            (scope_id, scope_id, scope_id),
        )
        row = await self._fetch_one(
            "SELECT id FROM session WHERE privacy_scope_id = ?"
            " UNION ALL SELECT id FROM gateway_route WHERE privacy_scope_id = ? LIMIT 1",
            (scope_id, scope_id),
        )
        if row is not None:
            raise ProtectionError("privacy_scope_in_use", "Privacy scope is still referenced")
