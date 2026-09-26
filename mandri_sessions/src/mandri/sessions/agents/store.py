from __future__ import annotations

import builtins
from dataclasses import asdict
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.core.ports.database import DatabasePort
from mandri.core.types.agents import Agent, AgentState
from mandri.core.types.sessions import Session
from mandri.sessions.errors import SessionNotFoundError

_COLUMNS = (
    "id, parent_session_id, parent_agent_id, session_id, harness, native_id, title, state,"
    " created_at, updated_at, delegation_id, task_id, transcript_path"
)


class AgentStore:
    def __init__(self, database: DatabasePort) -> None:
        self._db = database

    async def upsert(self, agent: Agent) -> None:
        values = asdict(agent)
        columns = [column.strip() for column in _COLUMNS.split(",")]
        updates = ", ".join(f"{column} = excluded.{column}" for column in columns[1:])
        await self._db.execute(
            f"INSERT INTO agent ({_COLUMNS}) VALUES ({', '.join('?' for _ in columns)})"
            f" ON CONFLICT(id) DO UPDATE SET {updates}",
            tuple(values[column] for column in columns),
        )

    async def get(self, agent_id: str) -> Agent:
        row = await self._db.fetch_one(f"SELECT {_COLUMNS} FROM agent WHERE id = ?", (agent_id,))
        if row is None:
            raise SessionNotFoundError("Unknown agent")
        return _agent(row)

    async def list(self, parent_session_id: str | None = None) -> list[Agent]:
        query = f"SELECT {_COLUMNS} FROM agent"
        parameters: tuple[str, ...] = ()
        if parent_session_id is not None:
            query += " WHERE parent_session_id = ?"
            parameters = (parent_session_id,)
        rows = await self._db.fetch_all(query + " ORDER BY created_at, id", parameters)
        return [_agent(row) for row in rows]

    async def remove(self, agent_id: str) -> None:
        await self._db.execute("DELETE FROM agent WHERE id = ?", (agent_id,))

    async def classify(self, sessions: builtins.list[Session]) -> None:
        for session in sessions:
            await self._db.execute(
                "INSERT INTO agent_classification VALUES (?, ?, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET"
                " harness = excluded.harness, native_id = excluded.native_id"
                " WHERE harness IS NOT excluded.harness OR native_id IS NOT excluded.native_id",
                (str(session.id), session.harness.value, session.native_id),
            )

    async def classified(self) -> builtins.list[str]:
        rows = await self._db.fetch_all(
            "SELECT s.id AS session_id FROM session s LEFT JOIN agent_classification c"
            " ON s.id = c.session_id AND s.harness = c.harness"
            " AND s.native_id IS c.native_id WHERE s.deleted = 0"
            " AND (s.native_id IS NULL OR c.session_id IS NOT NULL) ORDER BY s.id"
        )
        return [row["session_id"] for row in rows]

    async def set_revision(self, session_id: str, revision: str) -> None:
        await self._db.execute(
            "INSERT INTO agent_cache(session_id, revision) VALUES (?, ?)"
            " ON CONFLICT(session_id) DO UPDATE SET revision = excluded.revision"
            " WHERE revision IS NOT excluded.revision",
            (session_id, revision),
        )

    async def pending(self) -> dict[str, str]:
        rows = await self._db.fetch_all(
            "SELECT session_id, revision FROM agent_cache WHERE completed_revision IS NOT revision"
        )
        return {row["session_id"]: row["revision"] for row in rows}

    async def complete(self, session_id: str, revision: str) -> None:
        await self._db.execute(
            "UPDATE agent_cache SET completed_revision = ? WHERE session_id = ? AND revision = ?",
            (revision, session_id, revision),
        )

    async def invalidate(self, session_id: str) -> None:
        await self._db.execute(
            "UPDATE agent_cache SET completed_revision = NULL WHERE session_id = ?",
            (session_id,),
        )


def _agent(row: dict[str, Any]) -> Agent:
    return Agent(
        id=row["id"],
        parent_session_id=row["parent_session_id"],
        parent_agent_id=row["parent_agent_id"],
        session_id=row["session_id"],
        harness=HarnessKind(row["harness"]),
        native_id=row["native_id"],
        title=row["title"],
        state=AgentState(row["state"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        delegation_id=row["delegation_id"],
        task_id=row["task_id"],
        transcript_path=row["transcript_path"],
    )
