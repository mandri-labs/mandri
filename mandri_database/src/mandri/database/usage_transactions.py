import sqlite3
from collections.abc import Callable

from mandri.database.sqlite_adapter import AiosqliteDatabase


async def transaction[T](
    database: AiosqliteDatabase, operation: Callable[[sqlite3.Connection], T], *, write: bool
) -> T:
    def run(raw: sqlite3.Connection) -> T:
        raw.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            result = operation(raw)
            raw.commit()
            return result
        except BaseException:
            raw.rollback()
            raise

    run.__qualname__ = operation.__qualname__
    return await database.run(run, write=write)
