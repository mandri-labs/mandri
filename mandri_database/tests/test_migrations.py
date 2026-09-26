"""Tests for the SQLite user_version migration chain."""

import aiosqlite
import mandri.database.migrations as migrations_module
import pytest
from mandri.database.errors import MigrationError
from mandri.database.migrations import LATEST_VERSION, MIGRATIONS, Migration, apply_migrations


async def _connect(tmp_path: object) -> aiosqlite.Connection:
    return await aiosqlite.connect(tmp_path / "db.sqlite", isolation_level=None)


async def _version(connection: aiosqlite.Connection) -> int:
    cursor = await connection.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def _table_exists(connection: aiosqlite.Connection, name: str) -> bool:
    cursor = await connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    )
    return await cursor.fetchone() is not None


async def _columns(connection: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await connection.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return {str(row[1]) for row in rows}


def test_migration_versions_are_sequential() -> None:
    assert [migration.version for migration in MIGRATIONS] == list(range(1, LATEST_VERSION + 1))


async def test_apply_migrations_builds_latest_schema(tmp_path: object) -> None:
    connection = await _connect(tmp_path)
    try:
        await apply_migrations(connection)
        assert await _version(connection) == LATEST_VERSION
        assert await _table_exists(connection, "session")
        assert await _table_exists(connection, "gateway_route")
        assert await _table_exists(connection, "agent")
        assert not await _table_exists(connection, "model")
    finally:
        await connection.close()


async def test_apply_migrations_is_idempotent(tmp_path: object) -> None:
    connection = await _connect(tmp_path)
    try:
        await apply_migrations(connection)
        await apply_migrations(connection)
        assert await _version(connection) == LATEST_VERSION
    finally:
        await connection.close()


async def test_native_source_migration_preserves_existing_sessions(tmp_path, monkeypatch):
    connection = await _connect(tmp_path)
    try:
        monkeypatch.setattr(migrations_module, "MIGRATIONS", MIGRATIONS[:4])
        await apply_migrations(connection)
        await connection.execute(
            "INSERT INTO session (id, harness, project_path, created_at, updated_at, "
            "state, model, gateway_route_id, last_synced_at) "
            "VALUES ('old', 'codex', '/project', 1, 2, 'stopped', 'provider/model', 'route', 2)"
        )
        monkeypatch.setattr(migrations_module, "MIGRATIONS", MIGRATIONS)
        await apply_migrations(connection)
        cursor = await connection.execute(
            "SELECT model, gateway_route_id, model_source FROM session WHERE id = 'old'"
        )
        assert await cursor.fetchone() == ("provider/model", "route", "gateway")
    finally:
        await connection.close()


async def test_latest_schema_has_reasoning_effort_columns(tmp_path: object) -> None:
    connection = await _connect(tmp_path)
    try:
        await apply_migrations(connection)
        assert "reasoning_effort" in await _columns(connection, "gateway_route")
        assert "reasoning_effort" in await _columns(connection, "session")
    finally:
        await connection.close()


@pytest.mark.parametrize(
    "harness,model,route,native_id,expected",
    [
        ("codex", "gpt-6-astra", None, "thread", "native"),
        ("claude", None, None, "thread", "native"),
        ("codex", "provider/model", None, "thread", "gateway"),
        ("codex", "gpt-6-astra", "route", "thread", "gateway"),
        ("codex", "gpt-6-astra", None, None, "gateway"),
        ("opencode", "model", None, "thread", "gateway"),
    ],
)
async def test_repair_imported_native_source(
    tmp_path, monkeypatch, harness, model, route, native_id, expected
):
    connection = await _connect(tmp_path)
    try:
        monkeypatch.setattr(migrations_module, "MIGRATIONS", MIGRATIONS[:5])
        await apply_migrations(connection)
        await connection.execute(
            "INSERT INTO session (id, harness, native_id, project_path, created_at, updated_at, "
            "state, model, gateway_route_id, last_synced_at) "
            "VALUES ('imported', ?, ?, '/project', 1, 2, 'discovered', ?, ?, 2)",
            (harness, native_id, model, route),
        )
        monkeypatch.setattr(migrations_module, "MIGRATIONS", MIGRATIONS)
        await apply_migrations(connection)
        cursor = await connection.execute(
            "SELECT model, gateway_route_id, model_source FROM session"
        )
        assert await cursor.fetchone() == (model, route, expected)
    finally:
        await connection.close()


async def test_apply_migrations_resume_from_applied_prefix(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tuple(
        Migration(version=version, statements=(f"CREATE TABLE t{version} (id TEXT)",))
        for version in (1, 2, 3)
    )
    monkeypatch.setattr(migrations_module, "MIGRATIONS", fake)
    connection = await _connect(tmp_path)
    try:
        await connection.execute("CREATE TABLE t1 (id TEXT)")
        await connection.execute("PRAGMA user_version = 1")
        await apply_migrations(connection)
        assert await _version(connection) == 3
        assert await _table_exists(connection, "t1")
        assert await _table_exists(connection, "t2")
        assert await _table_exists(connection, "t3")
    finally:
        await connection.close()


async def test_failed_migration_encapsulates_error(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = (Migration(version=1, statements=("CREATE TABLE",)),)
    monkeypatch.setattr(migrations_module, "MIGRATIONS", broken)
    connection = await _connect(tmp_path)
    try:
        with pytest.raises(MigrationError):
            await apply_migrations(connection)
        assert await _version(connection) == 0
    finally:
        await connection.close()


async def test_parent_lineage_migration_preserves_protected_sessions(tmp_path, monkeypatch):
    connection = await _connect(tmp_path)
    try:
        monkeypatch.setattr(migrations_module, "MIGRATIONS", MIGRATIONS[:10])
        await apply_migrations(connection)
        await connection.execute(
            "INSERT INTO session (id, harness, project_path, created_at, updated_at,"
            " state, model, last_synced_at, privacy_mode, privacy_scope_id)"
            " VALUES ('protected', 'codex', '/project', 1, 2, 'stopped',"
            " 'provider/model', 2, 'surrogate', 'local-scope')"
        )
        monkeypatch.setattr(migrations_module, "MIGRATIONS", MIGRATIONS)
        await apply_migrations(connection)
        cursor = await connection.execute(
            "SELECT privacy_mode, privacy_scope_id, parent_native_id, parent_session_id"
            " FROM session WHERE id='protected'"
        )
        assert await cursor.fetchone() == ("surrogate", "local-scope", None, None)
        assert await _version(connection) == LATEST_VERSION
    finally:
        await connection.close()
