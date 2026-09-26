from dataclasses import replace
from typing import Any

from mandri.core.ids import HarnessSessionId, SessionId
from mandri.core.ports.database import DatabasePort
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.sessions import Session

MAX_LINEAGE_DEPTH = 128
_POLICY_FIELDS = (
    "execution_backend",
    "privacy_mode",
    "privacy_scope_id",
    "model_source",
    "model",
    "parent_native_id",
    "parent_session_id",
)


def ordered_sessions(rows: list[Session]) -> list[Session]:
    native = {str(row.native_id): row for row in rows if row.native_id is not None}
    if len(native) != sum(row.native_id is not None for row in rows):
        raise ProtectionError("session_lineage_invalid", "Native session identifiers are ambiguous")
    result: list[Session] = []
    done: set[str] = set()
    for row in rows:
        chain: list[Session] = []
        visiting: set[str] = set()
        current: Session | None = row
        while current is not None and str(current.native_id) not in done:
            identity = str(current.native_id)
            if identity in visiting or len(chain) >= MAX_LINEAGE_DEPTH:
                raise ProtectionError(
                    "session_lineage_invalid", "Native session lineage is invalid"
                )
            visiting.add(identity)
            chain.append(current)
            current = (
                native.get(str(current.parent_native_id)) if current.parent_native_id else None
            )
        for item in reversed(chain):
            done.add(str(item.native_id))
            result.append(item)
    return result


def _validate(record: dict[str, Any]) -> None:
    if record.get("privacy_mode") == PrivacyMode.SURROGATE.value:
        if not record.get("privacy_scope_id"):
            raise ProtectionError(
                "privacy_state_unavailable", "Inherited privacy state is unavailable"
            )
        if record.get("model_source") != ModelSource.GATEWAY.value:
            raise ProtectionError(
                "privacy_native_unsupported", "Protected lineage requires gateway models"
            )


def _inherit(record: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    _validate(parent)
    _validate(record)
    result = dict(record)
    result["parent_session_id"] = parent["id"]
    if (
        parent.get("execution_backend") == ExecutionBackend.DOCKER.value
        and record.get("execution_backend") != ExecutionBackend.DOCKER.value
    ):
        raise ProtectionError(
            "session_lineage_invalid", "Container lineage requires owned native state"
        )
    if parent.get("privacy_mode") == PrivacyMode.SURROGATE.value:
        scope = record.get("privacy_scope_id")
        if scope and scope != parent["privacy_scope_id"]:
            raise ProtectionError("session_lineage_invalid", "Inherited privacy scopes disagree")
        result.update(
            privacy_mode=PrivacyMode.SURROGATE.value,
            privacy_scope_id=parent["privacy_scope_id"],
            model_source=ModelSource.GATEWAY.value,
            model=record.get("model") if scope else parent.get("model"),
        )
    return result


class SessionLineage:
    def __init__(self, database: DatabasePort) -> None:
        self._db = database

    async def _parent(self, record: dict[str, Any]) -> dict[str, Any] | None:
        parent_id = record.get("parent_session_id")
        parent_native = record.get("parent_native_id")
        if parent_id:
            parent = await self._db.fetch_one("SELECT * FROM session WHERE id = ?", (parent_id,))
            if parent is None or parent["harness"] != record["harness"]:
                raise ProtectionError(
                    "session_lineage_unavailable", "The parent session is unavailable"
                )
            return parent
        if not parent_native:
            return None
        parents = await self._db.fetch_all(
            "SELECT * FROM session WHERE harness = ? AND native_id = ? ORDER BY deleted ASC",
            (record["harness"], parent_native),
        )
        live = [parent for parent in parents if not parent["deleted"]]
        if len(live) == 1:
            return live[0]
        if len(parents) == 1:
            return parents[0]
        if parents:
            raise ProtectionError("session_lineage_invalid", "The parent session is ambiguous")
        return None

    async def prepare(self, row: Session, existing: dict[str, Any] | None) -> Session:
        parent_native = row.parent_native_id
        if existing and existing.get("parent_native_id"):
            if parent_native and parent_native != existing["parent_native_id"]:
                raise ProtectionError("session_lineage_invalid", "Native session parent changed")
            parent_native = HarnessSessionId(str(existing["parent_native_id"]))
        if not parent_native:
            return row
        record: dict[str, Any] = (
            dict(existing)
            if existing
            else {
                "id": str(row.id),
                "harness": row.harness.value,
                "execution_backend": row.execution_backend.value,
                "privacy_mode": row.privacy_mode.value,
                "privacy_scope_id": row.privacy_scope_id,
                "model_source": row.model_source.value,
                "model": row.model,
            }
        )
        record["parent_native_id"] = str(parent_native)
        parent = await self._parent(record)
        if parent is not None:
            parent = await self.ensure(str(parent["id"]))
            record = _inherit(record, parent)
        return replace(
            row,
            parent_native_id=parent_native,
            parent_session_id=SessionId(str(record["parent_session_id"]))
            if record.get("parent_session_id")
            else None,
            execution_backend=ExecutionBackend(record.get("execution_backend", "host")),
            privacy_mode=PrivacyMode(record.get("privacy_mode", "none")),
            privacy_scope_id=record.get("privacy_scope_id"),
            model_source=ModelSource(record.get("model_source", "gateway")),
            model=record.get("model"),
        )

    async def persist(self, existing: dict[str, Any], row: Session) -> None:
        if row.parent_native_id is None:
            return
        updated = dict(existing)
        updated.update({key: getattr(row, key) for key in _POLICY_FIELDS})
        await self._save(existing, updated)

    async def ensure(self, session_id: str) -> dict[str, Any]:
        record = await self._db.fetch_one("SELECT * FROM session WHERE id = ?", (session_id,))
        if record is None:
            raise ProtectionError(
                "session_lineage_unavailable", "The session lineage is unavailable"
            )
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        while True:
            identity = str(record["id"])
            if identity in seen or len(chain) >= MAX_LINEAGE_DEPTH:
                raise ProtectionError("session_lineage_invalid", "The session lineage is invalid")
            seen.add(identity)
            _validate(record)
            chain.append(record)
            parent = await self._parent(record)
            if parent is None:
                if record.get("parent_native_id") or record.get("parent_session_id"):
                    raise ProtectionError(
                        "session_lineage_unavailable", "The parent session is unavailable"
                    )
                break
            record = parent
        parent = chain.pop()
        for child in reversed(chain):
            parent = await self._save(child, _inherit(child, parent))
        return parent

    async def _save(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        if all(before.get(key) == after.get(key) for key in _POLICY_FIELDS):
            return after
        assignments = ", ".join(f"{key} = ?" for key in _POLICY_FIELDS)
        updated = await self._db.fetch_one(
            f"UPDATE session SET {assignments}, policy_revision = policy_revision + 1"
            " WHERE id = ? AND policy_revision = ? RETURNING *",
            (*[after.get(key) for key in _POLICY_FIELDS], before["id"], before["policy_revision"]),
        )
        if updated is None:
            raise ProtectionError(
                "session_lineage_changed", "Session protection changed concurrently"
            )
        return updated
