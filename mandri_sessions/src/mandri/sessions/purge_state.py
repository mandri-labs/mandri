from dataclasses import replace

from mandri.core.ids import HarnessSessionId, RouteId
from mandri.core.ports.database import DatabasePort
from mandri.core.types.sessions import Session


async def retain_purge_state(db: DatabasePort, session: Session) -> Session:
    await db.execute(
        "INSERT INTO session_purge (session_id, native_id, gateway_route_id,"
        " privacy_scope_id, execution_context) VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(session_id) DO NOTHING",
        (
            str(session.id),
            session.native_id,
            session.gateway_route_id,
            session.privacy_scope_id,
            session.execution_context,
        ),
    )
    row = await db.fetch_one("SELECT * FROM session_purge WHERE session_id = ?", (str(session.id),))
    if row is None:
        raise RuntimeError("Session purge state was not persisted")
    return replace(
        session,
        native_id=HarnessSessionId(str(row["native_id"])) if row["native_id"] else None,
        gateway_route_id=RouteId(str(row["gateway_route_id"])) if row["gateway_route_id"] else None,
        privacy_scope_id=str(row["privacy_scope_id"]) if row["privacy_scope_id"] else None,
        execution_context=str(row["execution_context"]) if row["execution_context"] else None,
    )
