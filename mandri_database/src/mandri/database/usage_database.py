from pathlib import Path

import aiosqlite
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage_migrations import migrate_usage


class UsageDatabase(AiosqliteDatabase):
    def __init__(self, sessions_path: Path) -> None:
        super().__init__()
        self._sessions_path = sessions_path

    async def _configure_reader(self, connection: aiosqlite.Connection) -> None:
        uri = self._sessions_path.resolve().as_uri() + "?mode=ro"
        async with connection.execute("ATTACH DATABASE ? AS control", (uri,)):
            pass
        await super()._configure_reader(connection)

    async def migrate(self) -> None:
        await migrate_usage(self._require_connection())
