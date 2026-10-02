import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.core.ports.agents import AgentDiscoveryResult
from mandri.core.types.agents import NativeAgent
from mandri.core.types.sessions import Session
from mandri.sessions.errors import DatabaseAccessError
from mandri.sessions.opencode_store import session_tables


class OpencodeAgentDiscovery:
    def __init__(self, database: Path) -> None:
        self._database = database

    def discover(self, sessions: list[Session]) -> AgentDiscoveryResult:
        if not self._database.is_file():
            return AgentDiscoveryResult(HarnessKind.OPENCODE, [])
        try:
            uri = f"{self._database.resolve().as_uri()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=2)) as db:
                records: dict[str, tuple[Any, ...]] = {}
                tables = session_tables(db)
                for table in tables:
                    columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                    if "parent_id" not in columns:
                        continue
                    migrated = (
                        " AND id NOT IN (SELECT id FROM session_v2)"
                        if table == "session" and "session_v2" in tables
                        else ""
                    )
                    rows = db.execute(
                        f"SELECT id, parent_id, title, time_created, time_updated FROM {table}"
                        f" WHERE parent_id IS NOT NULL AND time_archived IS NULL{migrated}"
                    ).fetchall()
                    records.update((row[0], row) for row in rows)
        except sqlite3.Error as error:
            raise DatabaseAccessError("Cannot read OpenCode agent relationships") from error
        agents = [
            NativeAgent(
                HarnessKind.OPENCODE, row[0], row[1], row[2] or "OpenCode agent", row[3], row[4]
            )
            for row in records.values()
            if row[0] != row[1]
        ]

        return AgentDiscoveryResult(HarnessKind.OPENCODE, agents)
