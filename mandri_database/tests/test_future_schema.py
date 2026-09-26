import aiosqlite
import pytest
from mandri.database.errors import MigrationError
from mandri.database.migrations import LATEST_VERSION, apply_migrations


async def test_newer_database_is_rejected_without_changes(tmp_path):
    async with aiosqlite.connect(tmp_path / "future.db") as connection:
        await connection.execute(f"PRAGMA user_version = {LATEST_VERSION + 1}")
        with pytest.raises(MigrationError, match="newer than supported"):
            await apply_migrations(connection)
        cursor = await connection.execute("PRAGMA user_version")
        assert await cursor.fetchone() == (LATEST_VERSION + 1,)
