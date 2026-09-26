import pytest
from mandri.core.types.execution import ExecutionPhase
from mandri.database.executions import ExecutionRepository
from mandri.database.sqlite_adapter import AiosqliteDatabase


@pytest.fixture
async def store(tmp_path):
    database = AiosqliteDatabase()
    await database.connect(tmp_path / "sessions.sqlite")
    await database.migrate()
    try:
        yield database, ExecutionRepository(database)
    finally:
        await database.close()


async def session(database, identity, *, backend="docker", deleted=False):
    await database.execute(
        "INSERT INTO session"
        " (id,harness,project_path,created_at,updated_at,state,last_synced_at,"
        "execution_backend,deleted)"
        " VALUES (?,'codex','/workspace',1,1,'live',1,?,?)",
        (identity, backend, int(deleted)),
    )


@pytest.mark.parametrize(
    "phase", [ExecutionPhase.FAILED, ExecutionPhase.BLOCKED, ExecutionPhase.STOPPED]
)
async def test_terminal_execution_repairs_persisted_live_session_without_a_runtime_process(
    store, phase
):
    database, repository = store
    await session(database, "orphan")
    record = await repository.create("orphan", "daemon", "{}")
    if phase is ExecutionPhase.STOPPED:
        record = await repository.update(record, ExecutionPhase.STOPPING)
    terminal = await repository.update(record, phase)
    assert await repository.reconcile_stopped_sessions("daemon", frozenset()) == ["orphan"]
    row = await database.fetch_one("SELECT state,updated_at FROM session WHERE id='orphan'")
    assert row["state"] == "stopped"
    assert row["updated_at"] > 1
    assert await repository.latest("orphan") == terminal
    assert await repository.reconcile_stopped_sessions("daemon", frozenset()) == []


@pytest.mark.parametrize(
    "active_phase", [ExecutionPhase.CHECKING, ExecutionPhase.STARTING, ExecutionPhase.READY]
)
async def test_newer_active_generation_prevents_stopping_a_resumed_session(store, active_phase):
    database, repository = store
    await session(database, "resumed")
    old = await repository.create("resumed", "daemon", "{}")
    await repository.update(old, ExecutionPhase.FAILED)
    current = await repository.create("resumed", "daemon", "{}")
    if active_phase is not ExecutionPhase.CHECKING:
        current = await repository.update(current, ExecutionPhase.STARTING)
    if active_phase is ExecutionPhase.READY:
        await repository.update(current, ExecutionPhase.READY)
    assert await repository.reconcile_stopped_sessions("daemon", frozenset()) == []
    assert (await database.fetch_one("SELECT state FROM session WHERE id='resumed'"))[
        "state"
    ] == "live"


async def test_reconciliation_preserves_other_owners_host_deleted_and_transitioning_sessions(store):
    database, repository = store
    for identity in ("foreign", "host", "deleted", "transitioning", "tracked", "unknown"):
        await session(
            database,
            identity,
            backend="host" if identity == "host" else "docker",
            deleted=identity == "deleted",
        )
        if identity != "unknown":
            record = await repository.create(
                identity, "another-daemon" if identity == "foreign" else "daemon", "{}"
            )
            await repository.update(record, ExecutionPhase.FAILED)
    assert (
        await repository.reconcile_stopped_sessions(
            "daemon", frozenset({"transitioning", "tracked"})
        )
        == []
    )
    assert {row["state"] for row in await database.fetch_all("SELECT state FROM session")} == {
        "live"
    }
