import asyncio

import pytest
from mandri.core.types.conversation_status import StatusRevisionError, WorkObservation
from mandri.database.conversation_status import ConversationStatusRepository
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.sessions.conversation_status import ConversationStatuses


@pytest.fixture
async def database(tmp_path):
    database = AiosqliteDatabase()
    await database.connect(tmp_path / "status.db")
    await database.migrate()
    yield database
    await database.close()


def cycle(start="start", end="end"):
    return [
        WorkObservation(state="working", progress=True, key=start),
        WorkObservation(state="idle", outcome="completed", key=end),
    ]


async def test_history_and_process_lifecycle_do_not_create_unread(database):
    repository = ConversationStatusRepository(database)
    status = await repository.observe("session:old", [
        WorkObservation(state="idle", outcome="completed", key="old-result"),
        WorkObservation(state="idle", outcome="interrupted", key="process-stop"),
    ], "native", {"offset": 200})
    assert status.completion_revision == status.read_revision == 0
    assert await repository.checkpoint("session:old", "native") == {"offset": 200}


async def test_replay_is_idempotent_and_read_does_not_acknowledge_next_result(database):
    repository = ConversationStatusRepository(database)
    first = await repository.observe("session:s", cycle())
    assert first.completion_revision == 1
    replayed = await repository.observe("session:s", cycle())
    assert replayed.completion_revision == 1
    await repository.observe("session:s", cycle("next-start", "next-end"))
    read = await repository.acknowledge("session:s", 1, first.completion_key)
    assert read.read_revision == 1
    assert read.completion_revision == 2
    assert read.revision > replayed.revision
    with pytest.raises(StatusRevisionError):
        await repository.acknowledge("session:s", 2, first.completion_key)
    with pytest.raises(StatusRevisionError):
        await repository.acknowledge("session:s", 3)
    assert (await repository.acknowledge("session:s", 0)).read_revision == 1


async def test_restart_preserves_unread_and_reconciles_running_to_unknown(database):
    repository = ConversationStatusRepository(database)
    await repository.observe("session:done", cycle())
    await repository.observe("agent:running", [cycle()[0]])
    statuses = ConversationStatuses(ConversationStatusRepository(database))
    await statuses.start()
    try:
        assert statuses.get("session:done").completion_revision == 1
        assert statuses.get("session:done").read_revision == 0
        assert statuses.get("agent:running").work_state == "unknown"
        assert statuses.get("agent:running").cycle_active
        await statuses.observe("agent:running", [cycle()[1]])
        assert statuses.get("agent:running").completion_revision == 1
    finally:
        await statuses.close()


async def test_parent_read_is_independent_and_link_moves_pending_work(database):
    repository = ConversationStatusRepository(database)
    statuses = ConversationStatuses(repository)
    await statuses.start()
    try:
        await statuses.observe("session:parent", cycle())
        await statuses.observe("agent:child", cycle("child-start", "child-end"))
        await statuses.acknowledge("session:parent", 1)
        await statuses.link("agent:child", "session:child")
        assert statuses.get("agent:child").target == "session:child"
        assert statuses.get("agent:child").completion_revision == 1
        assert statuses.get("agent:child").read_revision == 0
        await statuses.link("agent:child", "session:child")
        assert statuses.get("session:child").completion_revision == 1
        await statuses.observe("agent:pending", [cycle("pending-start")[0]])
        await statuses.link("agent:pending", "session:pending")
        await statuses.observe("agent:pending", [cycle(end="pending-end")[1]])
        assert statuses.get("session:pending").completion_revision == 1
    finally:
        await statuses.close()


async def test_worker_counts_distinct_cycles_and_coalesces_one_native_frame(database):
    statuses = ConversationStatuses(ConversationStatusRepository(database))
    await statuses.start()
    try:
        for observation in cycle():
            statuses.enqueue("session:s", observation)
        for observation in cycle("start2", "end2"):
            statuses.enqueue("session:s", observation)
        await statuses.flush()
        assert statuses.get("session:s").completion_revision == 2
        statuses.enqueue("session:s", WorkObservation(state="working", progress=True, key="s3"))
        statuses.enqueue("session:s", WorkObservation(state="idle", outcome="completed", key="e3"))
        statuses.enqueue("session:s", WorkObservation(state="working", key="e3"))
        await statuses.flush()
        assert statuses.get("session:s").completion_revision == 2
        statuses.enqueue("session:s", WorkObservation(state="idle", key="bg-end"))
        await statuses.flush()
        assert statuses.get("session:s").completion_revision == 3
    finally:
        await statuses.close()


async def test_concurrent_reads_are_monotone(database):
    repository = ConversationStatusRepository(database)
    await repository.observe("session:s", cycle() + cycle("s2", "e2"))
    await asyncio.gather(
        repository.acknowledge("session:s", 2), repository.acknowledge("session:s", 1)
    )
    assert (await repository.all())[0].read_revision == 2
