"""Tests for the SQLite database adapter."""

import pytest
from mandri.core.ports.database import DatabasePort
from mandri.database.errors import (
    ConstraintError,
    DatabaseConnectionError,
    DatabaseError,
)
from mandri.database.sqlite_adapter import AiosqliteDatabase

_INSERT = (
    "INSERT INTO session (id, harness, project_path, created_at, updated_at,"
    " state, deleted, last_synced_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)"
)


async def _connected(tmp_path: object) -> AiosqliteDatabase:
    adapter = AiosqliteDatabase()
    await adapter.connect(tmp_path / "test.db")
    return adapter


async def test_conforms_to_database_port(tmp_path: object) -> None:
    adapter = await _connected(tmp_path)
    try:
        assert isinstance(adapter, DatabasePort)
    finally:
        await adapter.close()


async def test_connect_applies_sqlite_pragmas(tmp_path: object) -> None:
    adapter = await _connected(tmp_path)
    try:
        assert await adapter.fetch_one("PRAGMA journal_mode") == {"journal_mode": "wal"}
        assert await adapter.fetch_one("PRAGMA busy_timeout") == {"timeout": 5000}
        assert await adapter.fetch_one("PRAGMA foreign_keys") == {"foreign_keys": 1}
    finally:
        await adapter.close()


async def test_migrate_and_fetch_round_trip(tmp_path: object) -> None:
    adapter = await _connected(tmp_path)
    try:
        await adapter.migrate()
        await adapter.execute(_INSERT, ("s1", "claude", "C:/work/proj", 1, 2, "discovered", 3))
        row = await adapter.fetch_one("SELECT * FROM session WHERE id = ?", ("s1",))
        assert row is not None
        assert row["harness"] == "claude"
        assert await adapter.fetch_all("SELECT * FROM session") == [row]
    finally:
        await adapter.close()


async def test_operational_error_becomes_connection_error(tmp_path: object) -> None:
    adapter = await _connected(tmp_path)
    try:
        with pytest.raises(DatabaseConnectionError):
            await adapter.fetch_all("SELECT * FROM missing_table")
    finally:
        await adapter.close()


async def test_integrity_error_becomes_constraint_error(tmp_path: object) -> None:
    adapter = await _connected(tmp_path)
    try:
        await adapter.migrate()
        params = ("s1", "claude", "C:/work/proj", 1, 2, "discovered", 3)
        await adapter.execute(_INSERT, params)
        with pytest.raises(ConstraintError):
            await adapter.execute(_INSERT, params)
    finally:
        await adapter.close()


async def test_operations_without_connection_raise_database_error() -> None:
    adapter = AiosqliteDatabase()
    with pytest.raises(DatabaseError):
        await adapter.fetch_one("SELECT 1")


async def test_close_is_idempotent(tmp_path: object) -> None:
    adapter = await _connected(tmp_path)
    await adapter.close()
    await adapter.close()
