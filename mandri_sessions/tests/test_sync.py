"""Tests for the sync engine reconciliation."""

import asyncio

import pytest
from mandri.core.fs.paths import normalize_fs_path
from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind, SessionId
from mandri.sessions.sync import SyncEngine

from mandri_sessions.tests.substitutes import FakeBackend, FakeDatabase, make_session


async def _engine(db: FakeDatabase, backend: FakeBackend, hub: Hub | None = None) -> SyncEngine:
    return SyncEngine(db, {HarnessKind.CLAUDE: backend}, hub=hub)


async def test_sync_inserts_new_sessions() -> None:
    db = await FakeDatabase.create()
    backend = FakeBackend([make_session("n1")])
    engine = await _engine(db, backend)
    await engine.sync()
    row = await db.fetch_one("SELECT * FROM session WHERE native_id = 'n1'")
    assert row is not None
    assert row["native_title"] == "native title"
    assert row["state"] == "discovered"
    assert row["deleted"] == 0


async def test_sync_updates_existing_sessions() -> None:
    db = await FakeDatabase.create()
    backend = FakeBackend([make_session("n1", title="new", updated_at=200)])
    engine = await _engine(db, backend)
    await engine.sync()
    await engine.sync()
    row = await db.fetch_one("SELECT * FROM session WHERE native_id = 'n1'")
    assert row is not None
    assert row["native_title"] == "new"
    assert row["updated_at"] == 200


@pytest.mark.parametrize(
    "kind", [HarnessKind.CLAUDE, HarnessKind.CODEX, HarnessKind.OPENCODE, HarnessKind.AGY]
)
@pytest.mark.parametrize("state", ["discovered", "stopped", "live"])
async def test_sync_preserves_known_workspace_when_native_metadata_is_incomplete(
    kind: HarnessKind, state: str, tmp_path
) -> None:
    workspace = str(tmp_path / "workspace")
    relocated = str(tmp_path / "workspace" / "moved")
    db = await FakeDatabase.create()
    backend = FakeBackend([make_session("n1", project_path=workspace, harness=kind)])
    engine = SyncEngine(db, {kind: backend})
    try:
        await engine.sync()
        await db.execute("UPDATE session SET state = ? WHERE native_id = 'n1'", (state,))
        backend.sessions = [make_session("n1", title="updated", project_path="", harness=kind)]
        await engine.sync()
        row = await db.fetch_one("SELECT * FROM session WHERE native_id = 'n1'")
        assert row is not None
        assert row["project_path"] == workspace
        assert row["native_title"] == "updated"
        assert row["state"] == state
        backend.sessions = [make_session("n1", project_path=relocated, harness=kind)]
        await engine.sync()
        moved = await db.fetch_one("SELECT * FROM session WHERE native_id = 'n1'")
        assert moved is not None
        assert moved["project_path"] == relocated
        assert moved["id"] == row["id"]
    finally:
        await db.close()


async def test_sync_tombstones_missing_sessions() -> None:
    db = await FakeDatabase.create()
    backend = FakeBackend([])
    engine = await _engine(db, backend)
    await db.execute(
        "INSERT INTO session (id, harness, native_id, project_path, created_at,"
        " updated_at, state, deleted, last_synced_at)"
        " VALUES ('s1', 'claude', 'gone', 'C:/work/proj', 1, 2, 'discovered', 0, 2)"
    )
    await engine.sync()
    row = await db.fetch_one("SELECT deleted FROM session WHERE id = 's1'")
    assert row is not None
    assert row["deleted"] == 1


async def test_sync_claims_pending_session_by_evidence() -> None:
    db = await FakeDatabase.create()
    backend = FakeBackend([make_session("n9", project_path="C:/work/proj", created_at=2_000)])
    engine = await _engine(db, backend)
    await db.execute(
        "INSERT INTO session (id, harness, native_id, project_path, created_at,"
        " updated_at, state, deleted, last_synced_at)"
        " VALUES ('p1', 'claude', NULL, 'C:/work/proj', 1000, 1000, 'discovered', 0, 1000)"
    )
    await engine.sync()
    row = await db.fetch_one("SELECT * FROM session WHERE id = 'p1'")
    assert row is not None
    assert row["native_id"] == "n9"
    assert await db.fetch_one("SELECT * FROM session WHERE native_id = 'n9' AND id != 'p1'") is None


async def test_sync_does_not_claim_outside_window() -> None:
    db = await FakeDatabase.create()
    backend = FakeBackend([make_session("n9", project_path="C:/work/proj", created_at=100_000)])
    engine = await _engine(db, backend)
    await db.execute(
        "INSERT INTO session (id, harness, native_id, project_path, created_at,"
        " updated_at, state, deleted, last_synced_at)"
        " VALUES ('p1', 'claude', NULL, 'C:/work/proj', 1000, 1000, 'discovered', 0, 1000)"
    )
    await engine.sync()
    row = await db.fetch_one("SELECT * FROM session WHERE id = 'p1'")
    assert row is not None
    assert row["native_id"] is None


async def test_sync_marks_degraded_and_publishes_state() -> None:
    db = await FakeDatabase.create()
    hub = Hub()
    handle = hub.subscribe(Topic("runtimes"))
    backend = FakeBackend()
    backend.fail = True
    engine = SyncEngine(db, {HarnessKind.CLAUDE: backend}, hub=hub)
    await engine.sync()
    state = engine.harness_state(HarnessKind.CLAUDE)
    assert state.degraded is True
    assert state.last_sync_error is not None
    frame = await asyncio.wait_for(handle.queue.get(), timeout=1)
    assert frame["payload"]["degraded"] is True
    assert frame["payload"]["harness"] == "claude"


async def test_stale_before_and_after_sync() -> None:
    db = await FakeDatabase.create()
    backend = FakeBackend([make_session("n1")])
    engine = await _engine(db, backend)
    assert engine.stale(HarnessKind.CLAUDE) is True
    await engine.sync()
    assert engine.stale(HarnessKind.CLAUDE) is False


async def test_activity_of_is_none_before_observation() -> None:
    db = await FakeDatabase.create()
    backend = FakeBackend([make_session("n1")])
    engine = await _engine(db, backend)
    assert engine.activity_of(SessionId("n1")) is None
    await engine.sync()
    row = await db.fetch_one("SELECT id FROM session WHERE native_id = 'n1'")
    assert row is not None
    assert engine.activity_of(SessionId(str(row["id"]))) is not None


async def test_sync_keeps_managed_worktree_visible_when_native_history_disappears() -> None:
    db = await FakeDatabase.create()
    backend = FakeBackend([make_session("n1", project_path="/managed/worktree")])
    engine = await _engine(db, backend)
    await engine.sync()
    await db.execute("UPDATE session SET worktree = '{}' WHERE native_id = 'n1'")
    backend.sessions = [make_session("n1", title="updated", project_path="/original/project")]
    await engine.sync()
    row = await db.fetch_one("SELECT * FROM session WHERE native_id = 'n1'")
    assert row["project_path"] == normalize_fs_path("/managed/worktree")
    backend.sessions = []
    await engine.sync()
    row = await db.fetch_one("SELECT * FROM session WHERE native_id = 'n1'")
    assert row["deleted"] == 0
    assert row["worktree"] == "{}"
