from dataclasses import replace

import pytest
from mandri.core.ids import HarnessKind
from mandri.core.types.model_selection import ModelSource
from mandri.sessions.errors import SessionConflictError
from mandri.sessions.service import SessionsService
from mandri.sessions.sync import SyncEngine

from mandri_sessions.tests.substitutes import FakeBackend, FakeDatabase, FakeEngine, make_session


@pytest.mark.parametrize(
    "kind,model,source",
    [
        (HarnessKind.CODEX, "gpt-6-astra", ModelSource.NATIVE),
        (HarnessKind.CLAUDE, None, ModelSource.NATIVE),
        (HarnessKind.CLAUDE, "fable", ModelSource.NATIVE),
        (HarnessKind.CODEX, "provider/model", ModelSource.GATEWAY),
        (HarnessKind.OPENCODE, "provider/model", ModelSource.GATEWAY),
    ],
)
async def test_import_preserves_model_and_identifies_source(kind, model, source):
    db = await FakeDatabase.create()
    try:
        engine = SyncEngine(
            db, {kind: FakeBackend([make_session("native-1", harness=kind, model=model)])}
        )
        await engine.sync()
        row = await db.fetch_one("SELECT * FROM session WHERE native_id = 'native-1'")
        assert row["model"] == model
        assert row["model_source"] == source.value
        await engine.sync()
        updated = await db.fetch_one("SELECT * FROM session WHERE native_id = 'native-1'")
        assert updated["model_source"] == source.value
    finally:
        await db.close()


async def test_native_source_survives_read_and_model_change_without_route():
    db = await FakeDatabase.create()
    try:
        service = SessionsService(db, FakeEngine())
        session = await service.create_session(
            HarnessKind.CLAUDE,
            "fable",
            model_source=ModelSource.NATIVE,
            reasoning_effort="max",
        )
        fetched = await service.get_session(session.id)
        assert fetched.model_source is ModelSource.NATIVE
        assert fetched.gateway_route_id is None
        assert fetched.reasoning_effort == "max"
        updated = await service.set_session_model(session.id, "opus")
        assert updated.model_source is ModelSource.NATIVE
        assert updated.model == "opus"
        assert updated.reasoning_effort is None
    finally:
        await db.close()


@pytest.mark.parametrize(
    "harness", [HarnessKind.CODEX, HarnessKind.CLAUDE, HarnessKind.AGY, HarnessKind.PI]
)
async def test_switch_from_gateway_clears_route_and_incompatible_effort(harness):
    db = await FakeDatabase.create()
    try:
        service = SessionsService(db, FakeEngine())
        session = await service.create_session(
            harness,
            "provider/model",
            "route-1",
            reasoning_effort="high",
        )
        updated = await service.set_session_model(session.id, "gpt-native", ModelSource.NATIVE)
        assert updated.gateway_route_id is None
        assert updated.model_source is ModelSource.NATIVE
        assert updated.reasoning_effort is None
        gateway = await service.set_session_model(session.id, "provider/model", ModelSource.GATEWAY)
        assert gateway.model_source is ModelSource.GATEWAY
    finally:
        await db.close()


async def test_native_selection_rejects_opencode():
    db = await FakeDatabase.create()
    try:
        service = SessionsService(db, FakeEngine())
        session = await service.create_session(HarnessKind.OPENCODE, "provider/model")
        with pytest.raises(SessionConflictError):
            await service.set_session_model(session.id, "default", ModelSource.NATIVE)
    finally:
        await db.close()


async def test_agy_observational_binding_updates_discovered_model_source():
    db = await FakeDatabase.create()
    session = replace(
        make_session("agy-native", harness=HarnessKind.AGY), model_source=ModelSource.NATIVE
    )
    backend = FakeBackend([session])
    engine = SyncEngine(db, {HarnessKind.AGY: backend})
    await engine.sync()
    row = await db.fetch_one("SELECT model_source FROM session WHERE native_id = 'agy-native'")
    assert row["model_source"] == "native"
    backend.sessions = [replace(session, model="synthetic/model", model_source=ModelSource.GATEWAY)]
    await engine.sync()
    row = await db.fetch_one(
        "SELECT model, model_source FROM session WHERE native_id = 'agy-native'"
    )
    assert row["model"] == "synthetic/model"
    assert row["model_source"] == "gateway"
