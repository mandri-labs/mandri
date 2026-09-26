"""Substitute adapters and builders for sessions tests."""

from pathlib import Path
from typing import Any

import aiosqlite
from mandri.core.ids import (
    EpochMs,
    HarnessKind,
    HarnessSessionId,
    PageToken,
    ProjectPath,
    RawEvent,
    SessionId,
    SessionState,
    SessionTitle,
)
from mandri.core.ports.database import DatabasePort, SqlParams
from mandri.core.ports.transcripts import SessionRef, TranscriptPage
from mandri.core.types.sessions import Session
from mandri.sessions.activity import SessionActivity
from mandri.sessions.sync import HarnessState, SessionsBackend

SCHEMA = (
    """
    CREATE TABLE session (
      id TEXT PRIMARY KEY,
      harness TEXT NOT NULL,
      native_id TEXT,
      native_title TEXT,
      title_overlay TEXT,
      project_path TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      state TEXT NOT NULL,
      model TEXT,
      gateway_route_id TEXT,
      deleted INTEGER NOT NULL DEFAULT 0,
      last_synced_at INTEGER NOT NULL,
      interaction_mode TEXT,
      reasoning_effort TEXT,
      model_source TEXT NOT NULL DEFAULT 'gateway',
      execution_backend TEXT NOT NULL DEFAULT 'host',
      privacy_mode TEXT NOT NULL DEFAULT 'none',
      privacy_scope_id TEXT,
      execution_context TEXT,
      policy_revision INTEGER NOT NULL DEFAULT 1,
      parent_native_id TEXT,
      parent_session_id TEXT,
      worktree TEXT
    )
    """,
)


class FakeDatabase(DatabasePort):
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    @classmethod
    async def create(cls) -> "FakeDatabase":
        connection = await aiosqlite.connect(":memory:", isolation_level=None)
        connection.row_factory = aiosqlite.Row
        for statement in SCHEMA:
            await connection.execute(statement)
        return cls(connection)

    async def connect(self, db_path: Path | str) -> None:
        connection = await aiosqlite.connect(db_path, isolation_level=None)
        connection.row_factory = aiosqlite.Row
        self._connection = connection

    async def close(self) -> None:
        await self._connection.close()

    async def migrate(self) -> None:
        return None

    async def execute(self, sql: str, params: SqlParams = ()) -> None:
        await self._connection.execute(sql, params)

    async def fetch_all(self, sql: str, params: SqlParams = ()) -> list[dict[str, Any]]:
        cursor = await self._connection.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def fetch_one(self, sql: str, params: SqlParams = ()) -> dict[str, Any] | None:
        cursor = await self._connection.execute(sql, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    return {row.keys()[index]: row[index] for index in range(len(row))}


async def insert_session_row(
    db: DatabasePort,
    session_id: str,
    harness: str = "claude",
    native_id: str | None = None,
    native_title: str | None = None,
    title_overlay: str | None = None,
    project_path: str = "C:/work/proj",
    created_at: int = 1_000,
    updated_at: int = 2_000,
    state: str = "discovered",
    deleted: int = 0,
) -> None:
    await db.execute(
        "INSERT INTO session (id, harness, native_id, native_title, title_overlay,"
        " project_path, created_at, updated_at, state, deleted, last_synced_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            harness,
            native_id,
            native_title,
            title_overlay,
            project_path,
            created_at,
            updated_at,
            state,
            deleted,
            updated_at,
        ),
    )


def make_session(
    native_id: str,
    title: str = "native title",
    project_path: str = "C:/work/proj",
    created_at: int = 1_000,
    updated_at: int = 2_000,
    harness: HarnessKind = HarnessKind.CLAUDE,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> Session:
    return Session(
        id=SessionId(native_id),
        harness=harness,
        native_id=HarnessSessionId(native_id),
        native_title=SessionTitle(title),
        title_overlay=None,
        project_path=ProjectPath(project_path),
        created_at=EpochMs(created_at),
        updated_at=EpochMs(updated_at),
        state=SessionState.DISCOVERED,
        model=model,
        gateway_route_id=None,
        deleted=False,
        last_synced_at=EpochMs(updated_at),
        reasoning_effort=reasoning_effort,
    )


class FakeBackend(SessionsBackend):
    def __init__(self, sessions: list[Session] | None = None) -> None:
        self.sessions: list[Session] = sessions if sessions is not None else []
        self.deleted: list[SessionId] = []
        self.fail = False

    def fetch(self) -> list[Session]:
        if self.fail:
            raise RuntimeError("store unavailable")
        return list(self.sessions)

    def rename(self, session_id: SessionId, title: SessionTitle) -> None:
        raise NotImplementedError

    def delete(self, session_id: SessionId) -> None:
        self.deleted.append(session_id)

    def exists(self, session_id: SessionId) -> bool:
        return any(str(session.native_id) == str(session_id) for session in self.sessions)


class FakeEngine:
    def __init__(self) -> None:
        self.sync_calls = 0
        self._backends: dict[HarnessKind, FakeBackend] = {}
        self._activities: dict[SessionId, SessionActivity] = {}

    def attach(self, kind: HarnessKind, backend: FakeBackend) -> None:
        self._backends[kind] = backend

    def set_activity(self, session_id: SessionId, activity: SessionActivity) -> None:
        self._activities[session_id] = activity

    async def sync(self) -> None:
        self.sync_calls += 1

    def harness_state(self, kind: HarnessKind) -> HarnessState:
        return HarnessState()

    def backend(self, kind: HarnessKind) -> SessionsBackend | None:
        return self._backends.get(kind)

    def activity_of(self, session_id: SessionId) -> SessionActivity | None:
        return self._activities.get(session_id)


class FakeReader:
    def __init__(self) -> None:
        self.calls: list[tuple[SessionRef, PageToken | None, int]] = []

    def page(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        self.calls.append((session, cursor, limit))
        return TranscriptPage(entries=[RawEvent('{"t": 1}')], next_token=None, has_more=False)
