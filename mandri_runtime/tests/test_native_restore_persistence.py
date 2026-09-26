import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, ProjectPath, RouteId
from mandri.core.types.availability import SessionOwner
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from mandri.core.types.model_selection import ModelSource
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.service import RuntimeService
from mandri.sessions.errors import SessionConflictError
from mandri.sessions.ownership.service import NativeOwnership
from mandri.sessions.service import SessionsService


@pytest.fixture
async def store(tmp_path):
    path = tmp_path / "mandri.sqlite"
    database = AiosqliteDatabase()
    await database.connect(path)
    await database.migrate()
    service = SessionsService(database, Mock())
    service.native_ownership = AsyncMock(return_value=NativeOwnership(SessionOwner.UNOWNED))
    try:
        yield database, service, path
    finally:
        await database.close()


async def gateway_session(service, harness=HarnessKind.CODEX):
    session = await service.create_session(
        harness,
        "gateway/previous-model",
        RouteId("previous-route"),
        ProjectPath("/workspace"),
        reasoning_effort="high",
    )
    await service.reveal_native_id(session.id, HarnessSessionId("native-conversation"))
    return await service.get_session(session.id)


@pytest.mark.parametrize(
    "harness,model", [(HarnessKind.CODEX, "gpt-5.6-luna"), (HarnessKind.CLAUDE, "default")]
)
async def test_manual_restore_persists_model_source_route_and_effort_after_restart(
    store, monkeypatch, harness, model
):
    database, service, path = store
    record = await gateway_session(service, harness)
    restore = AsyncMock()
    monkeypatch.setattr("mandri.runtime.service.restore_codex_model", restore)
    runtime = RuntimeService({}, sessions=service)
    runtime._spawn_harness = AsyncMock(side_effect=AssertionError("unexpected process"))

    result = await runtime.restore_native_model(str(record.id))

    assert result.can_resume
    assert restore.await_count == (1 if harness is HarnessKind.CODEX else 0)
    runtime._spawn_harness.assert_not_awaited()
    await database.close()
    await database.connect(path)
    restarted = SessionsService(database, Mock())
    updated = await restarted.get_session(record.id)
    assert (
        updated.model,
        updated.model_source,
        updated.gateway_route_id,
        updated.reasoning_effort,
    ) == (
        model,
        ModelSource.NATIVE,
        None,
        None,
    )
    assert updated.native_id == record.native_id
    assert not runtime._session_state(str(record.id)).native_restore_failed
    assert not runtime._session_state(str(record.id)).resuming


async def test_native_failure_does_not_change_mandri_selection(store, monkeypatch):
    _, service, _ = store
    record = await gateway_session(service)
    monkeypatch.setattr(
        "mandri.runtime.service.restore_codex_model",
        AsyncMock(side_effect=ControlTransportError("settings update rejected")),
    )
    runtime = RuntimeService({}, sessions=service)

    with pytest.raises(ControlTransportError):
        await runtime.restore_native_model(str(record.id))

    assert await service.get_session(record.id) == record
    assert runtime._session_state(str(record.id)).native_restore_failed
    assert not runtime._session_state(str(record.id)).resuming


async def test_restore_preserves_model_selected_while_native_update_is_running(store, monkeypatch):
    _, service, _ = store
    record = await gateway_session(service)
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def native_restore(*_args):
        entered.set()
        await finish.wait()

    monkeypatch.setattr("mandri.runtime.service.restore_codex_model", native_restore)
    runtime = RuntimeService({}, sessions=service)
    restore = asyncio.create_task(runtime.restore_native_model(str(record.id)))
    await entered.wait()
    selected = await service.set_session_model(record.id, "gateway/new-model")
    finish.set()

    with pytest.raises(SessionConflictError, match="selection changed"):
        await restore

    assert await service.get_session(record.id) == selected
    assert runtime._session_state(str(record.id)).native_restore_failed
    assert not runtime._session_state(str(record.id)).resuming


async def test_automatic_release_restores_codex_without_replacing_mandri_selection(
    store, monkeypatch
):
    _, service, _ = store
    record = await gateway_session(service)
    restore = AsyncMock()
    monkeypatch.setattr("mandri.runtime.service.restore_codex_model", restore)
    runtime = RuntimeService({}, sessions=service)
    runtime.registry.mark_live(str(record.id), harness="codex")
    runtime.registry.mark_stopped(str(record.id))
    runtime._session_state(str(record.id)).launched_model = (
        ModelSource.GATEWAY,
        record.model,
        record.reasoning_effort,
    )

    await runtime._restore_after_release(str(record.id))

    restore.assert_awaited_once()
    assert await service.get_session(record.id) == record


async def test_database_failure_does_not_report_successful_restoration(monkeypatch):
    record = SimpleNamespace(
        id="session",
        harness=HarnessKind.CLAUDE,
        native_id="native",
        model="gateway/model",
        model_source=ModelSource.GATEWAY,
        execution_backend=ExecutionBackend.HOST,
        privacy_mode=PrivacyMode.NONE,
        gateway_route_id="route",
    )
    service = SimpleNamespace(
        get_session=AsyncMock(return_value=record),
        native_ownership=AsyncMock(return_value=NativeOwnership(SessionOwner.UNOWNED)),
        set_session_model=AsyncMock(side_effect=RuntimeError("database unavailable")),
    )
    runtime = RuntimeService({}, sessions=service)

    with pytest.raises(RuntimeError, match="database unavailable"):
        await runtime.restore_native_model("session")

    assert runtime._session_state("session").native_restore_failed
    assert not runtime._session_state("session").resuming
