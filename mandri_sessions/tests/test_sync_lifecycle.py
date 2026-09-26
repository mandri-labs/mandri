import asyncio
from threading import Event

import pytest
from mandri.core.ids import HarnessKind
from mandri.sessions.sync import SyncEngine

from mandri_sessions.tests.substitutes import (
    FakeBackend,
    FakeDatabase,
    insert_session_row,
    make_session,
)


@pytest.mark.parametrize("kind", list(HarnessKind))
@pytest.mark.parametrize("privacy_mode", ["none", "surrogate"])
async def test_missing_live_session_survives_until_native_history_is_written(kind, privacy_mode):
    db = await FakeDatabase.create()
    await insert_session_row(db, "managed", harness=kind.value, native_id="native", state="live")
    await db.execute(
        "UPDATE session SET privacy_mode = ?, privacy_scope_id = 'scope',"
        " gateway_route_id = 'route' WHERE id = 'managed'",
        (privacy_mode,),
    )
    backend = FakeBackend()
    engine = SyncEngine(db, {kind: backend})

    for _ in range(2):
        await engine.sync()
        row = await db.fetch_one("SELECT * FROM session WHERE id = 'managed'")
        assert row["deleted"] == 0
        assert row["state"] == "live"

    backend.sessions = [make_session("native", harness=kind)]
    await engine.sync()
    row = await db.fetch_one("SELECT * FROM session WHERE id = 'managed'")
    assert row["deleted"] == 0
    assert row["privacy_mode"] == privacy_mode
    assert row["privacy_scope_id"] == "scope"
    assert row["gateway_route_id"] == "route"


@pytest.mark.parametrize("state", ["discovered", "stopped"])
async def test_missing_inactive_session_is_still_removed(state):
    db = await FakeDatabase.create()
    await insert_session_row(db, "managed", harness="pi", native_id="native", state=state)
    await SyncEngine(db, {HarnessKind.PI: FakeBackend()}).sync()
    row = await db.fetch_one("SELECT deleted FROM session WHERE id = 'managed'")
    assert row["deleted"] == 1


@pytest.mark.parametrize("change", ["insert", "reveal", "resume", "rotate"])
async def test_inventory_cannot_remove_a_session_changed_during_fetch(change, monkeypatch):
    db = await FakeDatabase.create()
    if change != "insert":
        await insert_session_row(
            db,
            "managed",
            harness="pi",
            native_id=None if change == "reveal" else "previous",
            state="stopped",
        )
    started, release = Event(), Event()
    backend = FakeBackend()

    def fetch():
        started.set()
        if not release.wait(5):
            raise TimeoutError("Native inventory was not released")
        return []

    monkeypatch.setattr(backend, "fetch", fetch)
    task = asyncio.create_task(SyncEngine(db, {HarnessKind.PI: backend}).sync())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        if change == "insert":
            await insert_session_row(
                db, "managed", harness="pi", native_id="native", state="stopped"
            )
        elif change == "resume":
            await db.execute("UPDATE session SET state = 'live' WHERE id = 'managed'")
        else:
            await db.execute("UPDATE session SET native_id = 'native' WHERE id = 'managed'")
    finally:
        release.set()
        await asyncio.wait_for(task, 5)
    row = await db.fetch_one("SELECT deleted FROM session WHERE id = 'managed'")
    assert row["deleted"] == 0


async def test_sync_does_not_restore_explicitly_deleted_sessions():
    db = await FakeDatabase.create()
    await insert_session_row(
        db, "managed", harness="pi", native_id="native", state="stopped", deleted=1
    )
    backend = FakeBackend([make_session("native", harness=HarnessKind.PI)])
    await SyncEngine(db, {HarnessKind.PI: backend}).sync()
    row = await db.fetch_one("SELECT deleted FROM session WHERE id = 'managed'")
    assert row["deleted"] == 1
