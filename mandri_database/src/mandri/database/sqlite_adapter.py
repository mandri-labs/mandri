import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite
from mandri.core.ports.database import DatabasePort, SqlParams
from mandri.database.errors import (
    ConstraintError,
    DatabaseConnectionError,
    DatabaseError,
)
from mandri.database.migrations import apply_migrations

_PRAGMAS: tuple[tuple[str, Any], ...] = (
    ("journal_mode", "WAL"),
    ("busy_timeout", 5000),
    ("synchronous", "NORMAL"),
    ("foreign_keys", "ON"),
)


def _translate(error: sqlite3.Error) -> DatabaseError:
    if isinstance(error, sqlite3.IntegrityError):
        return ConstraintError(str(error))
    if isinstance(error, sqlite3.OperationalError):
        return DatabaseConnectionError(str(error))
    return DatabaseError(str(error))


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    return {row.keys()[index]: row[index] for index in range(len(row))}


class AiosqliteDatabase(DatabasePort):
    def __init__(self) -> None:
        self._connection: aiosqlite.Connection | None = None

    async def connect(self, db_path: Path | str) -> None:
        connection: aiosqlite.Connection | None = None
        try:
            connection = await aiosqlite.connect(db_path, isolation_level=None)
            connection.row_factory = aiosqlite.Row
            for name, value in _PRAGMAS:
                cursor = await connection.execute(f"PRAGMA {name} = {value}")
                await cursor.fetchall()
        except sqlite3.Error as error:
            if connection is not None:
                await connection.close()
            raise _translate(error) from error
        self._connection = connection

    async def close(self) -> None:
        if self._connection is None:
            return
        try:
            await self._connection.close()
        except sqlite3.Error as error:
            raise _translate(error) from error
        finally:
            self._connection = None

    async def migrate(self) -> None:
        connection = self._require_connection()
        try:
            await apply_migrations(connection)
        except sqlite3.Error as error:
            raise _translate(error) from error

    async def execute(self, sql: str, params: SqlParams = ()) -> None:
        connection = self._require_connection()
        try:
            await connection.execute(sql, params)
        except sqlite3.Error as error:
            raise _translate(error) from error

    async def fetch_all(self, sql: str, params: SqlParams = ()) -> list[dict[str, Any]]:
        connection = self._require_connection()
        try:
            cursor = await connection.execute(sql, params)
            rows = await cursor.fetchall()
        except sqlite3.Error as error:
            raise _translate(error) from error
        return [_row_to_dict(row) for row in rows]

    async def fetch_one(self, sql: str, params: SqlParams = ()) -> dict[str, Any] | None:
        connection = self._require_connection()
        try:
            cursor = await connection.execute(sql, params)
            row = await cursor.fetchone()
        except sqlite3.Error as error:
            raise _translate(error) from error
        if row is None:
            return None
        return _row_to_dict(row)

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise DatabaseError("database is not connected")
        return self._connection
