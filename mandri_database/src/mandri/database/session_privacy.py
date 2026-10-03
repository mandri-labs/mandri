import sqlite3

from mandri.core.ports.session_privacy import SessionPrivacyPort
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.core.types.sessions import Session
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage_transactions import transaction


class SessionPrivacyRepository(SessionPrivacyPort):
    def __init__(self, database: AiosqliteDatabase) -> None:
        self._database = database

    async def set_privacy(
        self, expected: Session, mode: PrivacyMode, scope_id: str | None
    ) -> None:
        def save(db: sqlite3.Connection) -> None:
            row = db.execute(
                "UPDATE session SET privacy_mode=?, privacy_scope_id=?, privacy_override=1,"
                " policy_revision=policy_revision+1 WHERE id=? AND deleted=0"
                " AND policy_revision=? AND model_source='gateway'"
                " AND gateway_route_id IS ? RETURNING id",
                (mode.value, scope_id, str(expected.id), expected.policy_revision,
                 expected.gateway_route_id),
            ).fetchone()
            if row is None:
                raise ProtectionError("session_policy_conflict", "Session policy changed")
            if expected.gateway_route_id is not None:
                route = db.execute(
                    "UPDATE gateway_route SET privacy_mode=?, privacy_scope_id=?"
                    " WHERE id=? AND execution_backend=? AND privacy_mode=?"
                    " AND privacy_scope_id IS ? RETURNING id",
                    (mode.value, scope_id, expected.gateway_route_id,
                     expected.execution_backend.value, expected.privacy_mode.value,
                     expected.privacy_scope_id),
                ).fetchone()
                if route is None:
                    raise ProtectionError("privacy_route_mismatch", "Session route changed")
                shared = db.execute(
                    "SELECT id FROM session WHERE gateway_route_id=? AND deleted=0 AND id!=?",
                    (expected.gateway_route_id, str(expected.id)),
                ).fetchone()
                if shared is not None:
                    raise ProtectionError("privacy_route_mismatch", "Session route is shared")

        await transaction(self._database, save, write=True)
