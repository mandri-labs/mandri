import asyncio
import logging
import sqlite3
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

import aiosqlite
from mandri.database.errors import DatabaseError

logger = logging.getLogger(__name__)


class ConnectionPool:
    def __init__(self, connections: list[aiosqlite.Connection]) -> None:
        self._connections = connections
        self._available: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        for connection in connections:
            self._available.put_nowait(connection)

    async def run[T](self, operation: Callable[[sqlite3.Connection], T]) -> T:
        if self._closed:
            raise DatabaseError("database is closed")
        queued_at = time.monotonic()
        connection = await self._available.get()
        if self._closed:
            self._available.put_nowait(connection)
            raise DatabaseError("database is closed")

        async def execute() -> T:
            try:

                def measured(raw: sqlite3.Connection) -> T:
                    started = time.monotonic()
                    try:
                        return operation(raw)
                    finally:
                        elapsed = time.monotonic() - started
                        waited = started - queued_at
                        if elapsed >= 0.25 or waited >= 0.25:
                            logger.warning(
                                "Slow SQLite operation=%s queue_ms=%.1f execute_ms=%.1f",
                                operation.__qualname__,
                                waited * 1000,
                                elapsed * 1000,
                            )

                enqueue = cast(Callable[..., Awaitable[T]], connection._execute)
                return await enqueue(measured, connection._conn)
            finally:
                self._available.put_nowait(connection)

        task = asyncio.create_task(execute())
        self._tasks.add(task)
        task.add_done_callback(self._completed)
        return await asyncio.shield(task)

    def _completed(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def close(self) -> None:
        self._closed = True
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await asyncio.gather(*(connection.close() for connection in self._connections))
