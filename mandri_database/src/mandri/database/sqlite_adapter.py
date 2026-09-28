import asyncio
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiosqlite
from mandri.core.ports.database import DatabasePort, SqlParams
from mandri.database.errors import ConstraintError, DatabaseConnectionError, DatabaseError
from mandri.database.migrations import apply_migrations
from mandri.database.sqlite_pool import ConnectionPool


def _translate(error: sqlite3.Error) -> DatabaseError:
    if isinstance(error, sqlite3.IntegrityError):
        return ConstraintError(str(error))
    if isinstance(error, sqlite3.OperationalError):
        return DatabaseConnectionError(str(error))
    return DatabaseError(str(error))


def _read_only(sql: str) -> bool:
    statement = sql.lstrip().upper()
    return statement.startswith(("SELECT ", "EXPLAIN ")) or (
        statement.startswith("PRAGMA ") and "=" not in statement
    )


class AiosqliteDatabase(DatabasePort):
    def __init__(self) -> None:
        self._connection: aiosqlite.Connection | None = None
        self._writer: ConnectionPool | None = None
        self._readers: ConnectionPool | None = None

    async def connect(self, db_path: Path | str) -> None:
        if self._connection is not None:
            raise DatabaseError("database is already connected")
        opened: list[aiosqlite.Connection] = []
        try:
            writer = await self._open(db_path)
            opened.append(writer)
            async with writer.execute("PRAGMA journal_mode=WAL") as cursor:
                await cursor.fetchone()
            readers = []
            if str(db_path) != ":memory:":
                for _ in range(2):
                    reader = await self._open(db_path)
                    opened.append(reader)
                    await self._configure_reader(reader)
                    readers.append(reader)
            self._connection = writer
            self._writer = ConnectionPool([writer])
            self._readers = ConnectionPool(readers) if readers else self._writer
        except BaseException:
            await asyncio.gather(*(connection.close() for connection in opened))
            raise

    async def _open(self, path: Path | str) -> aiosqlite.Connection:
        connection = await aiosqlite.connect(path, isolation_level=None, uri=True)
        try:
            connection.row_factory = sqlite3.Row
            for statement in (
                "PRAGMA busy_timeout=5000",
                "PRAGMA synchronous=FULL",
                "PRAGMA foreign_keys=ON",
            ):
                async with connection.execute(statement):
                    pass
            return connection
        except BaseException:
            await connection.close()
            raise

    async def _configure_reader(self, connection: aiosqlite.Connection) -> None:
        async with connection.execute("PRAGMA query_only=ON"):
            pass

    async def close(self) -> None:
        pools = {pool for pool in (self._writer, self._readers) if pool is not None}
        self._connection = None
        self._writer = self._readers = None
        await asyncio.gather(*(pool.close() for pool in pools))

    async def migrate(self) -> None:
        await apply_migrations(self._require_connection())

    async def run[T](self, operation: Callable[[sqlite3.Connection], T], *, write: bool) -> T:
        pool = self._writer if write else self._readers
        if pool is None:
            raise DatabaseError("database is not connected")
        try:
            return await pool.run(operation)
        except sqlite3.Error as error:
            raise _translate(error) from error

    async def execute(self, sql: str, params: SqlParams = ()) -> None:
        def execute(connection: sqlite3.Connection) -> None:
            connection.execute(sql, params).close()

        await self.run(execute, write=True)

    async def fetch_all(self, sql: str, params: SqlParams = ()) -> list[dict[str, Any]]:
        def fetch(connection: sqlite3.Connection) -> list[dict[str, Any]]:
            cursor = connection.execute(sql, params)
            try:
                return [dict(row) for row in cursor]
            finally:
                cursor.close()

        return await self.run(fetch, write=not _read_only(sql))

    async def fetch_one(self, sql: str, params: SqlParams = ()) -> dict[str, Any] | None:
        def fetch(connection: sqlite3.Connection) -> dict[str, Any] | None:
            cursor = connection.execute(sql, params)
            try:
                row = cursor.fetchone()
                return dict(row) if row is not None else None
            finally:
                cursor.close()

        return await self.run(fetch, write=not _read_only(sql))

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise DatabaseError("database is not connected")
        return self._connection
