from collections.abc import AsyncIterator

import pytest

from mandri_sessions.tests.substitutes import FakeDatabase


@pytest.fixture(autouse=True)
async def close_fake_databases(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    databases: list[FakeDatabase] = []
    create = FakeDatabase.create

    async def tracked_create(cls: type[FakeDatabase]) -> FakeDatabase:
        database = await create()
        databases.append(database)
        return database

    monkeypatch.setattr(FakeDatabase, "create", classmethod(tracked_create))
    try:
        yield
    finally:
        for database in reversed(databases):
            await database.close()
