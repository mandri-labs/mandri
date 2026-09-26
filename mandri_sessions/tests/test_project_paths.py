import ntpath
import posixpath

import pytest
from mandri.core.fs import paths
from mandri.core.ids import HarnessKind, ProjectPath
from mandri.sessions.service import SessionsService
from mandri.sessions.sync import SyncEngine

from mandri_sessions.tests.substitutes import FakeBackend, FakeDatabase, FakeEngine, make_session


@pytest.mark.parametrize("kind", list(HarnessKind))
@pytest.mark.parametrize("path_module", [ntpath, posixpath])
async def test_sync_normalizes_existing_and_new_sessions(monkeypatch, kind, path_module):
    monkeypatch.setattr(paths, "path", path_module)
    db = await FakeDatabase.create()
    raw = "D:/work/project" if path_module is ntpath else r"/home/a\b"
    expected = path_module.normpath(raw)
    backend = FakeBackend([make_session("n1", project_path=raw)])
    engine = SyncEngine(db, {kind: backend})
    await engine.sync()
    row = await db.fetch_one("SELECT * FROM session WHERE native_id = 'n1'")
    assert row is not None
    assert row["project_path"] == expected
    await db.execute("UPDATE session SET project_path = ? WHERE id = ?", (raw, row["id"]))
    await engine.sync()
    updated = await db.fetch_one("SELECT * FROM session WHERE native_id = 'n1'")
    assert updated is not None
    assert updated["id"] == row["id"]
    assert updated["project_path"] == expected


async def test_windows_claim_matches_mixed_separators(monkeypatch):
    monkeypatch.setattr(paths, "path", ntpath)
    db = await FakeDatabase.create()
    await db.execute(
        "INSERT INTO session (id, harness, project_path, created_at, updated_at,"
        " state, deleted, last_synced_at) VALUES ('pending', 'opencode', ?, 1000,"
        " 1000, 'discovered', 0, 1000)",
        (r"D:\work\project",),
    )
    backend = FakeBackend([make_session("n1", project_path="D:/work/project", created_at=1000)])
    engine = SyncEngine(db, {HarnessKind.OPENCODE: backend})
    await engine.sync()
    rows = await db.fetch_all("SELECT id, native_id, project_path FROM session")
    assert rows == [{"id": "pending", "native_id": "n1", "project_path": r"D:\work\project"}]


@pytest.mark.parametrize("path_module", [ntpath, posixpath])
@pytest.mark.parametrize("raw", ["D:/work/project", r"/home/a\b", ""])
async def test_service_persists_and_filters_normalized_paths(monkeypatch, path_module, raw):
    monkeypatch.setattr(paths, "path", path_module)
    db = await FakeDatabase.create()
    service = SessionsService(db, FakeEngine())
    session = await service.create_session(HarnessKind.OPENCODE, project_path=ProjectPath(raw))
    expected = path_module.normpath(raw) if raw else ""
    assert session.project_path == expected
    row = await db.fetch_one("SELECT project_path FROM session WHERE id = ?", (str(session.id),))
    assert row == {"project_path": expected}
    assert [item.id for item in await service.list_sessions(project_path=raw)] == [session.id]
    await db.execute("UPDATE session SET project_path = ? WHERE id = ?", (raw, str(session.id)))
    assert (await service.get_session(session.id)).project_path == expected
