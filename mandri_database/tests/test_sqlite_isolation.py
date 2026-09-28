import asyncio
import sqlite3
import threading
from pathlib import Path

import pytest
from mandri.database.errors import DatabaseConnectionError
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.database.usage_database import UsageDatabase
from mandri.database.usage_transactions import transaction


@pytest.fixture
async def databases(tmp_path: Path):
    control = AiosqliteDatabase()
    await control.connect(tmp_path / "mandri.db")
    await control.migrate()
    usage = UsageDatabase(tmp_path / "mandri.db")
    await usage.connect(tmp_path / "usage.db")
    await usage.migrate()
    try:
        yield control, usage
    finally:
        await usage.close()
        await control.close()


async def entered(event: threading.Event) -> None:
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.001)


async def test_usage_writer_cannot_block_control_writes_or_committed_reads(databases):
    control, usage = databases
    started, release = threading.Event(), threading.Event()

    def hold(connection: sqlite3.Connection) -> None:
        connection.execute("UPDATE usage_revision SET revision=1")
        started.set()
        assert release.wait(3)

    task = asyncio.create_task(transaction(usage, hold, write=True))
    try:
        await entered(started)
        async with asyncio.timeout(1):
            assert await UsageRepository(usage).revision() == 0
            await control.execute("CREATE TABLE independent (id INTEGER)")
            await control.execute("INSERT INTO independent VALUES (1)")
            assert await control.fetch_one("SELECT id FROM independent") == {"id": 1}
    finally:
        release.set()
        await task
    assert await UsageRepository(usage).revision() == 1


async def test_reader_snapshot_and_other_readers_survive_concurrent_write(databases):
    _, usage = databases
    started, release = threading.Event(), threading.Event()

    def read(connection: sqlite3.Connection) -> tuple[int, int]:
        before = connection.execute("SELECT revision FROM usage_revision").fetchone()[0]
        started.set()
        assert release.wait(3)
        after = connection.execute("SELECT revision FROM usage_revision").fetchone()[0]
        return before, after

    task = asyncio.create_task(transaction(usage, read, write=False))
    try:
        await entered(started)
        async with asyncio.timeout(1):
            await usage.execute("UPDATE usage_revision SET revision=7")
            assert await UsageRepository(usage).revision() == 7
    finally:
        release.set()
    assert await task == (0, 0)


async def test_cancelled_caller_does_not_release_unfinished_writer(databases):
    _, usage = databases
    started, release = threading.Event(), threading.Event()

    def hold(connection: sqlite3.Connection) -> None:
        connection.execute("UPDATE usage_revision SET revision=1")
        started.set()
        assert release.wait(3)

    task = asyncio.create_task(transaction(usage, hold, write=True))
    await entered(started)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    next_write = asyncio.create_task(usage.execute("UPDATE usage_revision SET revision=revision+1"))
    try:
        await asyncio.sleep(0.02)
        assert not next_write.done()
        assert await UsageRepository(usage).revision() == 0
    finally:
        release.set()
    await asyncio.wait_for(next_write, 1)
    assert await UsageRepository(usage).revision() == 2


async def test_usage_read_attachment_cannot_write_control_database(databases):
    control, usage = databases
    await control.execute("CREATE TABLE protected (value INTEGER)")
    with pytest.raises(DatabaseConnectionError):
        await usage.run(
            lambda connection: connection.execute("INSERT INTO control.protected VALUES (1)"),
            write=False,
        )
    assert await control.fetch_all("SELECT * FROM protected") == []
    assert (await UsageRepository(usage).overview())["summary"]["fact_count"] == 0
    assert await control.fetch_all("SELECT name FROM sqlite_master WHERE name LIKE 'usage_%'") == []
