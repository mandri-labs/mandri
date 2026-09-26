import sqlite3
from collections.abc import Awaitable, Callable
from typing import cast

from mandri.database.sqlite_adapter import AiosqliteDatabase


async def transaction[T](
    database: AiosqliteDatabase, operation: Callable[[sqlite3.Connection], T], *, write: bool
) -> T:
    connection = database._require_connection()

    def run() -> T:
        raw = connection._conn
        raw.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            result = operation(raw)
            raw.commit()
            return result
        except BaseException:
            raw.rollback()
            raise

    execute = cast(Callable[[Callable[[], T]], Awaitable[T]], connection._execute)
    return await execute(run)
