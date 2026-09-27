import json
from pathlib import Path

import pytest
from mandri.core.ids import HarnessKind
from mandri.sessions.adapters.pi_sessions import PiSessionsAdapter
from mandri.sessions.pi_store import PiSessionStore
from mandri.sessions.sync import SyncEngine

from mandri_sessions.tests.substitutes import FakeDatabase
from mandri_sessions.tests.test_pi_sessions import branched_session, write_leaf, write_session


@pytest.mark.parametrize("partial", [b"", b'{"type":"ses', b'{"type":"session"}\n'])
async def test_partial_native_rewrite_keeps_external_session_visible(tmp_path, partial):
    path = tmp_path / "workspace" / "native.jsonl"
    write_session(path)
    original = path.read_bytes()
    backend = PiSessionsAdapter(tmp_path)
    database = await FakeDatabase.create()
    engine = SyncEngine(database, {HarnessKind.PI: backend})
    try:
        await engine.sync()
        initial = await database.fetch_one("SELECT * FROM session WHERE native_id = 'native'")
        path.write_bytes(partial)
        for _ in range(2):
            await engine.sync()
            row = await database.fetch_one("SELECT * FROM session WHERE id = ?", (initial["id"],))
            assert row["deleted"] == 0
            assert row["native_title"] == initial["native_title"]
        path.write_bytes(original)
        await engine.sync()
        restored = await database.fetch_one("SELECT * FROM session WHERE native_id = 'native'")
        assert restored["id"] == initial["id"]
        assert restored["deleted"] == 0
    finally:
        await database.close()


def test_temporary_read_error_retains_known_inventory_and_retries(tmp_path, monkeypatch):
    path = tmp_path / "workspace" / "native.jsonl"
    write_session(path)
    store = PiSessionStore(tmp_path)
    original = store.fetch()
    path.write_text(
        path.read_text() + json.dumps({"type": "session_info", "name": "Updated"}) + "\n"
    )
    opening = Path.open

    def unavailable(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError("native transcript is temporarily unavailable")
        return opening(candidate, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", unavailable)
        assert store.fetch() == original
        assert store.fetch() == original
    assert store.fetch()[0].name == "Updated"


def test_complete_identity_replacement_does_not_retain_previous_session(tmp_path):
    path = tmp_path / "workspace" / "native.jsonl"
    write_session(path)
    store = PiSessionStore(tmp_path)
    assert store.fetch()[0].native_id == "native"
    write_session(path, "replacement")
    assert [row.native_id for row in store.fetch()] == ["replacement"]
    path.unlink()
    assert store.fetch() == []


@pytest.mark.parametrize("leaf", ["branch-a", "branch-b", None])
def test_partial_rewrite_retains_selected_branch_metadata(tmp_path, leaf):
    path = tmp_path / "native.jsonl"
    branched_session(path)
    store = PiSessionStore(tmp_path)
    write_leaf(path, "branch-a")
    assert store.fetch()[0].thinking_level == "high"
    write_leaf(path, leaf)
    selected = store.fetch()
    path.write_bytes(b'{"type":"ses')
    assert store.fetch() == selected
    assert store.fetch() == selected
