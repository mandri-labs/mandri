from types import SimpleNamespace

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, SessionId
from mandri.database.errors import ConstraintError
from mandri.database.executions import ExecutionRepository
from mandri.database.native_sessions import NativePiSessionIdentities
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.sessions.errors import SessionConflictError
from mandri.sessions.ownership import processes
from mandri.sessions.service import SessionsService
from mandri.sessions.sync import SyncEngine

from .substitutes import FakeBackend, insert_session_row, make_session


@pytest.fixture
async def identities(tmp_path, monkeypatch):
    monkeypatch.setattr(processes.psutil, "process_iter", lambda *args, **kwargs: [])
    database = AiosqliteDatabase()
    await database.connect(tmp_path / "sessions.db")
    await database.migrate()
    backend = FakeBackend()
    engine = SyncEngine(database, {HarnessKind.PI: backend})
    service = SessionsService(database, engine, pi_identities=NativePiSessionIdentities(database))
    await insert_session_row(
        database, "managed", harness="pi", native_id="previous", state="live"
    )
    try:
        yield service, database, backend, engine
    finally:
        await database.close()


async def test_pi_can_rotate_before_native_discovery(identities):
    service, database, _, _ = identities
    updated = await service.adopt_pi_native_id(SessionId("managed"), HarnessSessionId("new"))
    assert updated.native_id == "new"
    assert len(await database.fetch_all("SELECT * FROM session")) == 1


@pytest.mark.parametrize("other_writer", [False, True])
async def test_pi_initial_binding_adopts_passive_row_but_never_another_writer(
    identities, monkeypatch, other_writer
):
    service, database, _, _ = identities
    await database.execute("UPDATE session SET native_id=NULL WHERE id='managed'")
    await insert_session_row(database, "passive", harness="pi", native_id="new")
    writers = [
        SimpleNamespace(
            pid=pid,
            info={
                "name": "pi",
                "cmdline": ["pi", "--session", "new"],
                "cwd": "C:/work/proj",
                "create_time": 1.0,
            },
        )
        for pid in ([12, 13] if other_writer else [12])
    ]
    monkeypatch.setattr(processes.psutil, "process_iter", lambda *args, **kwargs: writers)
    if other_writer:
        with pytest.raises(SessionConflictError, match="external or unconfirmed writer"):
            await service.adopt_pi_native_id(
                SessionId("managed"), HarnessSessionId("new"), ignored_pid=12
            )
        assert (await service.get_session(SessionId("managed"))).native_id is None
        assert (await service.get_session(SessionId("passive"))).native_id == "new"
    else:
        updated = await service.adopt_pi_native_id(
            SessionId("managed"), HarnessSessionId("new"), ignored_pid=12
        )
        assert updated.native_id == "new"


async def test_pi_repeated_identity_binding_does_not_rescan_its_own_process(
    identities, monkeypatch
):
    service, _, _, _ = identities

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("No ownership scan for unchanged native identity")

    monkeypatch.setattr(processes.psutil, "process_iter", unexpected_scan)
    updated = await service.adopt_pi_native_id(SessionId("managed"), HarnessSessionId("previous"))
    assert updated.native_id == "previous"


@pytest.mark.parametrize("other_writer", [False, True])
async def test_pi_adoption_checks_external_writer_with_managed_identities(
    identities, monkeypatch, other_writer
):
    service, _, _, _ = identities
    writers = [
        SimpleNamespace(
            pid=pid,
            info={
                "name": "pi",
                "cmdline": ["pi", "--mode", "rpc"],
                "cwd": "C:/work/proj",
                "create_time": 1.0,
            },
        )
        for pid in ([12, 13] if other_writer else [12])
    ]
    monkeypatch.setattr(processes.psutil, "process_iter", lambda *args, **kwargs: writers)
    known = {12: "previous"}
    service.set_pi_processes(lambda: known)
    if other_writer:
        with pytest.raises(SessionConflictError, match="external or unconfirmed writer"):
            await service.adopt_pi_native_id(SessionId("managed"), HarnessSessionId("new"))
        assert (await service.get_session(SessionId("managed"))).native_id == "previous"
    else:
        updated = await service.adopt_pi_native_id(SessionId("managed"), HarnessSessionId("new"))
        assert updated.native_id == "new"


async def test_pi_adopts_passive_identity_discovered_before_callback(identities):
    service, database, backend, engine = identities
    backend.sessions = [
        make_session("previous", harness=HarnessKind.PI),
        make_session("new", title="Native fork", harness=HarnessKind.PI),
    ]
    await engine.sync()
    duplicate = await database.fetch_one("SELECT id FROM session WHERE native_id='new'")
    assert duplicate is not None
    updated = await service.adopt_pi_native_id(SessionId("managed"), HarnessSessionId("new"))
    assert updated.native_title == "Native fork"
    assert updated.native_id == "new"
    await engine.sync()
    hidden = await database.fetch_one("SELECT * FROM session WHERE id=?", (duplicate["id"],))
    assert hidden is not None and hidden["deleted"] and hidden["native_id"] is None
    owners = await database.fetch_all("SELECT id FROM session WHERE native_id='new'")
    assert owners == [{"id": "managed"}]
    assert not engine.harness_state(HarnessKind.PI).degraded
    assert backend.deleted == []


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("state", "stopped"),
        ("state", "live"),
        ("gateway_route_id", "owned-route"),
        ("worktree", '{"id":"owned-worktree"}'),
        ("harness", "claude"),
        ("execution_backend", "docker"),
        ("privacy_mode", "surrogate"),
        ("privacy_scope_id", "other-scope"),
        ("project_path", "/other-project"),
        ("deleted", 1),
    ],
)
async def test_pi_does_not_adopt_owned_or_incompatible_identity(identities, column, value):
    service, database, _, _ = identities
    await insert_session_row(database, "target", harness="pi", native_id="taken")
    await database.execute(f"UPDATE session SET {column}=? WHERE id='target'", (value,))
    before = await database.fetch_all("SELECT * FROM session ORDER BY id")
    with pytest.raises(SessionConflictError, match="already claimed"):
        await service.adopt_pi_native_id(SessionId("managed"), HarnessSessionId("taken"))
    assert await database.fetch_all("SELECT * FROM session ORDER BY id") == before


async def test_pi_adoption_compares_normalized_project_paths(identities):
    service, database, _, _ = identities
    await insert_session_row(database, "target", harness="pi", native_id="taken")
    await database.execute("UPDATE session SET project_path='C:/work/proj/.' WHERE id='target'")
    updated = await service.adopt_pi_native_id(SessionId("managed"), HarnessSessionId("taken"))
    assert updated.native_id == "taken"


async def test_pi_does_not_adopt_a_managed_session_during_startup(identities):
    service, database, _, _ = identities
    await insert_session_row(database, "target", harness="pi", native_id="taken")
    await ExecutionRepository(database).create("target", "daemon", "{}")
    with pytest.raises(SessionConflictError, match="already claimed"):
        await service.adopt_pi_native_id(SessionId("managed"), HarnessSessionId("taken"))
    assert (await service.get_session(SessionId("target"))).native_id == "taken"


async def test_pi_adoption_does_not_cross_container_state(identities):
    service, database, _, _ = identities
    await insert_session_row(database, "target", harness="pi", native_id="taken")
    await database.execute("UPDATE session SET execution_backend='docker'")
    await database.execute("UPDATE session SET execution_context='state-a' WHERE id='managed'")
    await database.execute("UPDATE session SET execution_context='state-b' WHERE id='target'")
    with pytest.raises(SessionConflictError, match="already claimed"):
        await service.adopt_pi_native_id(SessionId("managed"), HarnessSessionId("taken"))


async def test_pi_identity_adoption_rolls_back_both_rows_on_failure(identities):
    service, database, _, _ = identities
    await insert_session_row(database, "target", harness="pi", native_id="taken")
    before = await database.fetch_all("SELECT * FROM session ORDER BY id")
    await database.execute(
        "CREATE TRIGGER reject_identity BEFORE UPDATE OF native_id ON session"
        " WHEN NEW.id='managed' BEGIN SELECT RAISE(ABORT,'synthetic failure'); END"
    )
    with pytest.raises(ConstraintError, match="synthetic failure"):
        await service.adopt_pi_native_id(SessionId("managed"), HarnessSessionId("taken"))
    assert await database.fetch_all("SELECT * FROM session ORDER BY id") == before
