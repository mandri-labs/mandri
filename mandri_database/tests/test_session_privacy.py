import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import RouteId, SessionId
from mandri.core.types.availability import SessionOwner
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.core.types.model_selection import ModelSource
from mandri.database.session_privacy import SessionPrivacyRepository
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.gateway.route_registry import RouteRegistry
from mandri.runtime.registry import LIVE
from mandri.runtime.service import RuntimeService
from mandri.sessions.errors import SessionRunningError
from mandri.sessions.lineage import SessionLineage
from mandri.sessions.service import SessionsService


@pytest.fixture
async def store(tmp_path):
    database = AiosqliteDatabase()
    await database.connect(tmp_path / "sessions.sqlite")
    await database.migrate()
    await database.execute(
        "INSERT INTO gateway_route (id,provider_name,model_ref,formats,created_at)"
        " VALUES ('route','synthetic','synthetic/model','[\"openai\"]',0)"
    )
    await database.execute(
        "INSERT INTO session (id,harness,native_id,project_path,created_at,updated_at,"
        " state,last_synced_at,model,model_source,gateway_route_id)"
        " VALUES ('session','codex','native-session','/workspace',0,0,'live',0,"
        " 'synthetic/model','gateway','route')"
    )
    scopes = SimpleNamespace(create=AsyncMock(return_value="scope"), validate=AsyncMock(),
                             delete=AsyncMock())
    sessions = SessionsService(
        database, Mock(), privacy_scopes=scopes,
        session_privacy=SessionPrivacyRepository(database),
    )
    try:
        yield database, sessions, scopes
    finally:
        await database.close()


@pytest.mark.parametrize("backend", ["host", "docker"])
async def test_live_toggle_preserves_execution_and_reuses_scope(store, backend):
    database, sessions, scopes = store
    await database.execute("UPDATE session SET execution_backend=?", (backend,))
    await database.execute("UPDATE gateway_route SET execution_backend=?", (backend,))
    runtime = RuntimeService({}, sessions=sessions)
    runtime._access.availability = AsyncMock(
        return_value=SimpleNamespace(owner=SessionOwner.MANDRI)
    )
    runtime._events.publish_both = Mock()
    runtime._registry.status = Mock(return_value=LIVE)
    runtime.stop_session = AsyncMock()
    runtime._resume_session = AsyncMock()
    state = runtime._session_state("session")
    state.launched_model = (ModelSource.GATEWAY, "synthetic/model", None)
    control = state.control = Mock()
    routes = RouteRegistry(database, Mock())
    in_flight = await routes.get(RouteId("route"))
    original = await sessions.get_session(SessionId("session"))
    enabled = await runtime.set_session_privacy("session", PrivacyMode.SURROGATE)
    route = await routes.get(RouteId("route"))
    assert await routes._bound_conversation(route) == "session"
    assert route.privacy_mode is PrivacyMode.SURROGATE
    assert in_flight.privacy_mode is PrivacyMode.NONE
    assert route.privacy_scope_id == enabled.privacy_scope_id == "scope"
    assert enabled.policy_revision == 2
    assert replace(enabled, privacy_mode=original.privacy_mode,
                   privacy_scope_id=original.privacy_scope_id,
                   policy_revision=original.policy_revision) == original
    disabled = await runtime.set_session_privacy("session", PrivacyMode.NONE)
    assert disabled.privacy_scope_id == "scope"
    assert (await routes.get(RouteId("route"))).privacy_mode is PrivacyMode.NONE
    reenabled = await runtime.set_session_privacy("session", PrivacyMode.SURROGATE)
    assert reenabled.privacy_scope_id == "scope" and reenabled.policy_revision == 4
    scopes.create.assert_awaited_once_with(str(Path("/workspace")))
    scopes.validate.assert_awaited_once_with("scope")
    scopes.delete.assert_not_called()
    assert runtime._session_state("session").policy == reenabled.policy
    assert not runtime._session_state("session").resuming
    assert runtime._events.publish_both.call_count == 3
    assert (await sessions.ensure_session_policy(reenabled.id)) == reenabled
    assert state.control is control
    assert state.launched_model == (ModelSource.GATEWAY, "synthetic/model", None)
    runtime.stop_session.assert_not_awaited()
    runtime._resume_session.assert_not_awaited()


async def test_native_model_is_rejected_without_scope_or_route_mutation(store):
    database, sessions, scopes = store
    await database.execute("UPDATE session SET model_source='native',gateway_route_id=NULL")
    record = await sessions.get_session(SessionId("session"))
    with pytest.raises(ProtectionError) as caught:
        await sessions.set_session_privacy(record, PrivacyMode.SURROGATE)
    assert caught.value.code == "privacy_native_unsupported"
    assert await sessions.get_session(record.id) == record
    scopes.create.assert_not_called()


async def test_route_mismatch_rolls_back_and_deletes_unused_new_scope(store):
    database, sessions, scopes = store
    await database.execute("UPDATE gateway_route SET execution_backend='docker'")
    record = await sessions.get_session(SessionId("session"))
    with pytest.raises(ProtectionError, match="route changed"):
        await sessions.set_session_privacy(record, PrivacyMode.SURROGATE)
    assert await sessions.get_session(record.id) == record
    scopes.delete.assert_awaited_once_with("scope")


async def test_concurrent_policy_changes_have_one_winner(store):
    database, sessions, _scopes = store
    record = await sessions.get_session(SessionId("session"))
    repository = SessionPrivacyRepository(database)
    results = await asyncio.gather(
        repository.set_privacy(record, PrivacyMode.SURROGATE, "scope-a"),
        repository.set_privacy(record, PrivacyMode.SURROGATE, "scope-b"),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ProtectionError) for result in results) == 1
    updated = await sessions.get_session(record.id)
    route = await database.fetch_one("SELECT * FROM gateway_route WHERE id='route'")
    assert updated.policy_revision == 2
    assert updated.privacy_scope_id == route["privacy_scope_id"]


async def test_explicit_child_toggle_survives_lineage_refresh(store):
    database, sessions, _scopes = store
    await database.execute(
        "INSERT INTO session (id,harness,native_id,project_path,created_at,updated_at,"
        " state,last_synced_at,model_source,privacy_mode,privacy_scope_id)"
        " VALUES ('parent','codex','native-parent','/workspace',0,0,'stopped',0,"
        " 'gateway','surrogate','inherited-scope')"
    )
    await database.execute(
        "UPDATE session SET parent_session_id='parent',parent_native_id='native-parent',"
        " privacy_mode='surrogate',privacy_scope_id='inherited-scope' WHERE id='session'"
    )
    await database.execute(
        "UPDATE gateway_route SET privacy_mode='surrogate',privacy_scope_id='inherited-scope'"
    )
    record = await sessions.ensure_session_policy(SessionId("session"))
    disabled = await sessions.set_session_privacy(record, PrivacyMode.NONE)
    refreshed = await SessionLineage(database).ensure("session")
    assert refreshed["privacy_mode"] == "none"
    assert refreshed["privacy_scope_id"] == disabled.privacy_scope_id == "inherited-scope"
    assert refreshed["parent_session_id"] == "parent"


async def test_external_writer_cannot_change_privacy(store):
    _database, sessions, scopes = store
    runtime = RuntimeService({}, sessions=sessions)
    runtime._access.availability = AsyncMock(
        return_value=SimpleNamespace(owner=SessionOwner.EXTERNAL)
    )
    with pytest.raises(ProtectionError, match="external writer"):
        await runtime.set_session_privacy("session", PrivacyMode.SURROGATE)
    scopes.create.assert_not_called()
    assert not runtime._session_state("session").resuming


@pytest.mark.parametrize("transition", ["resuming", "stopping"])
async def test_privacy_change_rejects_session_transition(store, transition):
    _database, sessions, scopes = store
    runtime = RuntimeService({}, sessions=sessions)
    setattr(runtime._session_state("session"), transition, True)
    with pytest.raises(SessionRunningError):
        await runtime.set_session_privacy("session", PrivacyMode.SURROGATE)
    scopes.create.assert_not_called()


async def test_running_native_harness_rejects_pending_gateway_selection(store):
    _database, sessions, scopes = store
    runtime = RuntimeService({}, sessions=sessions)
    runtime._access.availability = AsyncMock(
        return_value=SimpleNamespace(owner=SessionOwner.MANDRI)
    )
    runtime._registry.status = Mock(return_value=LIVE)
    runtime._session_state("session").launched_model = (ModelSource.NATIVE, "default", None)
    with pytest.raises(ProtectionError, match="native model"):
        await runtime.set_session_privacy("session", PrivacyMode.SURROGATE)
    scopes.create.assert_not_called()


async def test_shared_route_rejects_toggle_and_rolls_back(store):
    database, sessions, scopes = store
    await database.execute(
        "INSERT INTO session (id,harness,project_path,created_at,updated_at,state,"
        "last_synced_at,model_source,gateway_route_id)"
        " VALUES ('other','codex','/workspace',0,0,'stopped',0,'gateway','route')"
    )
    record = await sessions.get_session(SessionId("session"))
    with pytest.raises(ProtectionError, match="shared"):
        await sessions.set_session_privacy(record, PrivacyMode.SURROGATE)
    assert await sessions.get_session(record.id) == record
    route = await database.fetch_one("SELECT privacy_mode FROM gateway_route WHERE id='route'")
    assert route["privacy_mode"] == "none"
    scopes.delete.assert_awaited_once_with("scope")


async def test_cancelled_request_finishes_policy_update_before_releasing_transition(store):
    _database, sessions, scopes = store
    runtime = RuntimeService({}, sessions=sessions)
    runtime._access.availability = AsyncMock(
        return_value=SimpleNamespace(owner=SessionOwner.MANDRI)
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def create_scope(_path):
        entered.set()
        await release.wait()
        return "scope"

    scopes.create.side_effect = create_scope
    request = asyncio.create_task(runtime.set_session_privacy("session", PrivacyMode.SURROGATE))
    await entered.wait()
    request.cancel()
    await asyncio.sleep(0)
    assert runtime._session_state("session").resuming
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await request
    updated = await sessions.get_session(SessionId("session"))
    assert updated.privacy_mode is PrivacyMode.SURROGATE
    assert runtime._session_state("session").policy == updated.policy
    assert not runtime._session_state("session").resuming
    scopes.delete.assert_not_called()
