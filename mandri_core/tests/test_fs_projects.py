from pathlib import Path

import aiosqlite
import pytest
from mandri.core.fs.service import list_project_paths
from mandri.core.ports.database import DatabasePort

SESSION_SCHEMA = """
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
  interaction_mode TEXT
)
"""


class SqliteDb(DatabasePort):
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        cursor = await self._connection.execute(sql, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return [dict(row) for row in rows]


def _insert_sql(session_id: str, project_path: str, updated_at: int) -> str:
    return (
        f"INSERT INTO session (id, harness, project_path, created_at, updated_at,"
        f" state, deleted, last_synced_at)"
        f" VALUES ('{session_id}', 'claude', '{project_path}', {updated_at},"
        f" {updated_at}, 'stopped', 0, {updated_at})"
    )


async def _seed(connection: aiosqlite.Connection) -> None:
    await connection.execute(SESSION_SCHEMA)
    await connection.execute(_insert_sql("s1", "D:/Dev/example-project", 500))
    await connection.execute(_insert_sql("s2", "D:/Dev/example-project", 900))
    await connection.execute(_insert_sql("s3", "C:/Users/test-user", 700))
    await connection.execute(_insert_sql("s4", "D:/other", 100))
    await connection.execute(_insert_sql("s5", "D:/deleted", 999))
    await connection.execute("UPDATE session SET deleted = 1 WHERE id = 's5'")
    await connection.execute(_insert_sql("s6", "", 1))


@pytest.mark.asyncio
async def test_projects_ordered_by_most_recent_session():
    async with aiosqlite.connect(":memory:") as connection:
        connection.row_factory = aiosqlite.Row
        await _seed(connection)
        paths = await list_project_paths(SqliteDb(connection))
    assert [Path(path).as_posix() for path in paths] == [
        "D:/Dev/example-project", "C:/Users/test-user", "D:/other",
    ]


@pytest.mark.asyncio
async def test_projects_tie_break_alphabetically():
    async with aiosqlite.connect(":memory:") as connection:
        connection.row_factory = aiosqlite.Row
        await connection.execute(SESSION_SCHEMA)
        await connection.execute(_insert_sql("s1", "D:/b", 100))
        await connection.execute(_insert_sql("s2", "D:/a", 100))
        paths = await list_project_paths(SqliteDb(connection))
    assert [Path(path).as_posix() for path in paths] == ["D:/a", "D:/b"]


@pytest.mark.asyncio
async def test_normalization_deduplicates_paths():
    async with aiosqlite.connect(":memory:") as connection:
        connection.row_factory = aiosqlite.Row
        await connection.execute(SESSION_SCHEMA)
        await connection.execute(_insert_sql("s1", "D:/Dev/example-project", 100))
        await connection.execute(_insert_sql("s2", "D:/Dev/./example-project", 200))
        paths = await list_project_paths(SqliteDb(connection))
    assert len(paths) == 1
