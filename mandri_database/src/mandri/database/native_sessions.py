import sqlite3

from mandri.core.fs.paths import normalize_fs_path
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage_transactions import transaction


class NativePiSessionIdentities:
    def __init__(self, database: AiosqliteDatabase) -> None:
        self._database = database

    async def adopt(self, session_id: str, native_id: str) -> bool:
        return await transaction(
            self._database,
            lambda connection: _adopt(connection, session_id, native_id),
            write=True,
        )


def _adopt(connection: sqlite3.Connection, session_id: str, native_id: str) -> bool:
    current = connection.execute(
        "SELECT * FROM session WHERE id=? AND deleted=0 AND harness='pi'", (session_id,)
    ).fetchone()
    if current is None:
        return False
    if current["native_id"] == native_id:
        return True
    claimed = connection.execute(
        "SELECT * FROM session WHERE native_id=? AND id!=?", (native_id, session_id)
    ).fetchall()
    if len(claimed) > 1:
        return False
    title = None
    if claimed:
        target = claimed[0]
        if (
            target["harness"] != "pi"
            or target["state"] != "discovered"
            or target["deleted"]
            or target["gateway_route_id"] is not None
            or target["worktree"] is not None
            or connection.execute(
                "SELECT 1 FROM execution_generation WHERE session_id=? LIMIT 1", (target["id"],)
            ).fetchone()
            is not None
            or target["execution_backend"] != current["execution_backend"]
            or target["privacy_mode"] != current["privacy_mode"]
            or target["privacy_scope_id"] != current["privacy_scope_id"]
            or normalize_fs_path(target["project_path"])
            != normalize_fs_path(current["project_path"])
            or (
                current["execution_backend"] == "docker"
                and target["execution_context"] != current["execution_context"]
            )
        ):
            return False
        title = target["native_title"]
        connection.execute(
            "UPDATE session SET native_id=NULL,deleted=1 WHERE id=?", (target["id"],)
        )
    connection.execute(
        "UPDATE session SET native_id=?,native_title=? WHERE id=?", (native_id, title, session_id)
    )
    return True
