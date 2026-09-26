import asyncio

import pytest
from mandri.core.types.execution import ExecutionPhase, ProtectionError
from mandri.database.executions import ExecutionRepository
from mandri.database.sqlite_adapter import AiosqliteDatabase


async def test_concurrent_generation_allocation_is_atomic_across_connections(tmp_path):
    first = AiosqliteDatabase()
    second = AiosqliteDatabase()
    await first.connect(tmp_path / "db.sqlite")
    await first.migrate()
    await second.connect(tmp_path / "db.sqlite")
    try:
        stores = (ExecutionRepository(first), ExecutionRepository(second))
        records = await asyncio.gather(
            *[stores[index % 2].create("session", "owner", "{}") for index in range(24)]
        )
        assert sorted(record.generation for record in records) == list(range(1, 25))
        latest = await stores[1].latest("session")
        assert latest.generation == 24
        assert latest.revision == 1
    finally:
        await first.close()
        await second.close()


async def test_transition_compare_and_swap_rejects_stale_state_and_invalid_revival(tmp_path):
    database = AiosqliteDatabase()
    await database.connect(tmp_path / "db.sqlite")
    await database.migrate()
    repository = ExecutionRepository(database)
    try:
        original = await repository.create("session", "owner", "{}")
        starting = await repository.update(original, ExecutionPhase.STARTING)
        with pytest.raises(ProtectionError, match="concurrently"):
            await repository.update(original, ExecutionPhase.FAILED)
        ready = await repository.update(starting, ExecutionPhase.READY, container_id="container-id")
        stopping = await repository.update(ready, ExecutionPhase.STOPPING)
        stopped = await repository.update(
            stopping, ExecutionPhase.STOPPED, context='{"exit_code":137,"oom_killed":true}'
        )
        assert stopped.container_id == "container-id" and stopped.revision == 5
        with pytest.raises(ProtectionError, match="transition"):
            await repository.update(stopped, ExecutionPhase.READY)
        assert await repository.active("owner") == []
        assert (await repository.latest("session")).context == '{"exit_code":137,"oom_killed":true}'
    finally:
        await database.close()


async def test_execution_reconciliation_is_owner_scoped_and_durable(tmp_path):
    database = AiosqliteDatabase()
    await database.connect(tmp_path / "db.sqlite")
    await database.migrate()
    repository = ExecutionRepository(database)
    try:
        ours = await repository.create("ours", "owner-a", "{}")
        theirs = await repository.create("theirs", "owner-b", "{}")
        assert await repository.active("owner-a") == [ours]
        await repository.update(ours, ExecutionPhase.FAILED, reason="execution_interrupted")
        assert await repository.active("owner-b") == [theirs]
    finally:
        await database.close()
    await database.connect(tmp_path / "db.sqlite")
    try:
        assert (await repository.latest("ours")).reason == "execution_interrupted"
        assert (await repository.latest("theirs")).phase is ExecutionPhase.CHECKING
    finally:
        await database.close()
