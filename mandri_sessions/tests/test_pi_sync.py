import dataclasses

import pytest
from mandri.core.ids import HarnessKind
from mandri.core.types.model_selection import ModelSource
from mandri.sessions.service import SessionsService
from mandri.sessions.sync import SyncEngine

from mandri_sessions.tests.substitutes import FakeBackend, FakeDatabase, make_session


async def test_pi_sync_tracks_native_model_and_thinking_changes():
    db = await FakeDatabase.create()
    original = dataclasses.replace(
        make_session("pi-session", harness=HarnessKind.PI, model="custom/org/model"),
        model_source=ModelSource.NATIVE,
        reasoning_effort="high",
    )
    backend = FakeBackend([original])
    engine = SyncEngine(db, {HarnessKind.PI: backend})
    try:
        await engine.sync()
        first = await db.fetch_one("SELECT * FROM session WHERE native_id = 'pi-session'")
        assert first is not None
        assert (first["model_source"], first["model"], first["reasoning_effort"]) == (
            "native",
            "custom/org/model",
            "high",
        )
        backend.sessions = [
            dataclasses.replace(original, model="custom/other", reasoning_effort="low")
        ]
        await engine.sync()
        changed = await db.fetch_one("SELECT * FROM session WHERE native_id = 'pi-session'")
        assert changed is not None
        assert (changed["model_source"], changed["model"], changed["reasoning_effort"]) == (
            "native",
            "custom/other",
            "low",
        )
    finally:
        await db.close()


async def test_pi_session_model_can_switch_between_gateway_and_native():
    db = await FakeDatabase.create()
    engine = SyncEngine(db, {})
    service = SessionsService(db, engine)
    try:
        session = await service.create_session(
            HarnessKind.PI, model="provider/model", project_path="/workspace"
        )
        native = await service.set_session_model(session.id, "custom/model", ModelSource.NATIVE)
        assert native.model_source is ModelSource.NATIVE
        assert native.model == "custom/model"
        assert native.gateway_route_id is None
        assert await service.observe_native_selection(native, "custom/command-model", "high")
        changed = await service.get_session(session.id)
        assert (changed.model, changed.reasoning_effort) == ("custom/command-model", "high")
        assert not await service.observe_native_selection(native, "custom/stale", "low")
        gateway = await service.set_session_model(session.id, "provider/model", ModelSource.GATEWAY)
        assert gateway.model_source is ModelSource.GATEWAY
    finally:
        await db.close()


@pytest.mark.parametrize(
    "state,route", [("live", None), ("stopped", None), ("discovered", "route")]
)
async def test_pi_sync_preserves_managed_model_selection(state, route):
    db = await FakeDatabase.create()
    native = dataclasses.replace(
        make_session("pi-session", harness=HarnessKind.PI, model="mandri/gateway"),
        model_source=ModelSource.NATIVE,
        reasoning_effort="low",
    )
    engine = SyncEngine(db, {HarnessKind.PI: FakeBackend([native])})
    try:
        await engine.sync()
        await db.execute(
            "UPDATE session SET state = ?, gateway_route_id = ?, model_source = 'gateway',"
            " model = 'provider/model', reasoning_effort = 'high' WHERE native_id = 'pi-session'",
            (state, route),
        )
        await engine.sync()
        row = await db.fetch_one("SELECT * FROM session WHERE native_id = 'pi-session'")
        assert row is not None
        assert (row["model_source"], row["model"], row["reasoning_effort"]) == (
            "gateway",
            "provider/model",
            "high",
        )
    finally:
        await db.close()
