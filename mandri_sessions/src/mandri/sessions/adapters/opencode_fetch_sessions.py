"""Opencode session fetch adapter reading the opencode SQLite store read-only."""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from mandri.core.ids import (
    EpochMs,
    HarnessKind,
    HarnessSessionId,
    ProjectPath,
    SessionId,
    SessionState,
    SessionTitle,
)
from mandri.core.ports.sessions import FetchSessionsPort
from mandri.core.types.sessions import Session
from mandri.sessions.errors import (
    DatabaseAccessError,
    SchemaDriftError,
    SessionParseError,
)

REQUIRED_SESSION_COLUMNS = frozenset(
    {"id", "title", "directory", "time_created", "time_updated", "time_archived"}
)
SQLITE_TIMEOUT_S = 2.0


class OpencodeSqliteFetchSessionsAdapter(FetchSessionsPort):
    """Read-only listing of non-archived opencode sessions from the state DB."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    def fetch(self) -> list[Session]:
        uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_S)) as connection:
                self._check_schema(connection)
                columns = {row[1] for row in connection.execute("PRAGMA table_info(session)")}
                parent = "parent_id" if "parent_id" in columns else "NULL"
                rows = connection.execute(
                    f"SELECT id, title, directory, time_created, time_updated, {parent}"
                    " FROM session"
                    " WHERE time_archived IS NULL"
                    " ORDER BY time_updated DESC"
                ).fetchall()
        except sqlite3.Error as error:
            raise DatabaseAccessError(f"cannot read opencode store: {error}") from error
        return [self._row_to_session(row) for row in rows]

    @staticmethod
    def _check_schema(connection: sqlite3.Connection) -> None:
        columns = connection.execute("PRAGMA table_info(session)").fetchall()
        names = {str(row[1]) for row in columns}
        missing = REQUIRED_SESSION_COLUMNS - names
        if missing:
            raise SchemaDriftError(f"opencode store missing columns: {sorted(missing)}")

    @staticmethod
    def _row_to_session(row: tuple[Any, ...]) -> Session:
        try:
            native_id, title, directory, created_ms, updated_ms, parent_id = row
            return Session(
                id=SessionId(str(native_id)),
                harness=HarnessKind.OPENCODE,
                native_id=HarnessSessionId(str(native_id)),
                native_title=SessionTitle(str(title)) if title else None,
                title_overlay=None,
                project_path=ProjectPath(str(directory or "")),
                created_at=EpochMs(int(created_ms or 0)),
                updated_at=EpochMs(int(updated_ms or 0)),
                state=SessionState.DISCOVERED,
                model=None,
                gateway_route_id=None,
                deleted=False,
                last_synced_at=EpochMs(0),
                parent_native_id=HarnessSessionId(str(parent_id)) if parent_id else None,
            )
        except (TypeError, ValueError) as error:
            raise SessionParseError(f"malformed opencode session row: {error}") from error
