import json
import sqlite3
from pathlib import Path

import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind
from mandri.sessions.adapters.codex_fetch_sessions import CodexFetchSessions
from mandri.sessions.docker_titles import docker_title
from mandri.sessions.sync import SyncEngine

from mandri_sessions.tests.substitutes import FakeDatabase, insert_session_row

NATIVE_ID = "11111111-1111-4111-8111-111111111111"


def write_title(state: Path, kind: HarnessKind, title: str) -> None:
    if kind is HarnessKind.CODEX:
        database = state / ".codex/state_5.sqlite"
        database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS threads (id TEXT, cwd TEXT, model TEXT,"
                " title TEXT, name TEXT, first_user_message TEXT, archived INT,"
                " archived_at INT, created_at_ms INT, updated_at_ms INT)"
            )
            db.execute("DELETE FROM threads")
            db.execute(
                "INSERT INTO threads VALUES (?, '/workspace', 'model', ?, NULL, ?, 0, NULL, 1, 2)",
                (NATIVE_ID, title, title),
            )
    elif kind is HarnessKind.OPENCODE:
        database = state / ".local/share/opencode/opencode.db"
        database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS session (id TEXT, title TEXT, directory TEXT,"
                " time_created INT, time_updated INT, time_archived INT)"
            )
            db.execute("DELETE FROM session")
            db.execute(
                "INSERT INTO session VALUES (?, ?, '/workspace', 1, 2, NULL)", (NATIVE_ID, title)
            )
    elif kind is HarnessKind.CLAUDE:
        path = state / ".claude/projects/-workspace" / f"{NATIVE_ID}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "type": "user",
                    "sessionId": NATIVE_ID,
                    "message": {"role": "user", "content": "First prompt"},
                }
            )
            + "\n"
            + json.dumps({"type": "custom-title", "customTitle": title})
            + "\n"
        )
    elif kind is HarnessKind.PI:
        path = state / ".pi/agent/sessions/--workspace--" / f"{NATIVE_ID}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"type": "session", "id": NATIVE_ID, "cwd": "/workspace", "version": 3})
            + "\n"
            + json.dumps({"type": "session_info", "name": title, "id": "title", "parentId": None})
            + "\n"
        )
    else:
        root = state / ".gemini/antigravity-cli"
        (root / "conversations").mkdir(parents=True, exist_ok=True)
        (root / "conversations" / f"{NATIVE_ID}.db").touch()
        cache = root / "cache/conversation_metadata.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({NATIVE_ID: {"title": title}}))


async def stored_session(db: FakeDatabase, root: Path, kind: HarnessKind, sid: str) -> Path:
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    state = root / "state" / sid
    state.mkdir(parents=True)
    await insert_session_row(
        db, sid, native_id=NATIVE_ID, harness=kind.value, project_path=str(workspace)
    )
    context = json.dumps(
        {
            "version": "1",
            "native_state_root": str(state),
            "workspace_root": str(workspace),
            "native_home": "/home/worker",
            "container_root": "/workspace",
        }
    )
    await db.execute(
        "UPDATE session SET execution_backend = 'docker', privacy_mode = 'surrogate',"
        " privacy_scope_id = 'scope', execution_context = ?, title_overlay = 'Custom title'"
        " WHERE id = ?",
        (context, sid),
    )
    return state


@pytest.mark.parametrize("kind", list(HarnessKind))
async def test_docker_titles_sync_and_refresh_without_changing_policy(tmp_path: Path, kind) -> None:
    db = await FakeDatabase.create()
    try:
        state = await stored_session(db, tmp_path, kind, "session")
        hub = Hub()
        feed = hub.subscribe(Topic("sessions.all"))
        engine = SyncEngine(db, {}, hub=hub, docker_title_reader=docker_title)
        for title in ("First title", "Resolved title"):
            write_title(state, kind, title)
            await engine.sync()
            row = await db.fetch_one("SELECT * FROM session WHERE id = 'session'")
            assert row["native_title"] == title
            assert row["title_overlay"] == "Custom title"
            assert row["project_path"] == str(tmp_path / "workspace")
            assert row["privacy_mode"] == "surrogate"
            assert row["privacy_scope_id"] == "scope"
            assert row["updated_at"] == 2000
            assert row["deleted"] == 0
            assert feed.queue.get_nowait()["payload"]["raw"]["type"] == "sessions_changed"
            await engine.sync()
            assert feed.queue.empty()
        assert len(await db.fetch_all("SELECT * FROM session")) == 1
    finally:
        await db.close()


async def test_unavailable_or_escaping_store_does_not_block_other_sessions(tmp_path: Path) -> None:
    db = await FakeDatabase.create()
    try:
        bad = await stored_session(db, tmp_path, HarnessKind.CODEX, "bad")
        good = await stored_session(db, tmp_path, HarnessKind.CODEX, "good")
        write_title(good, HarnessKind.CODEX, "Good title")
        await db.execute("UPDATE session SET native_title = 'Previous title' WHERE id = 'bad'")
        engine = SyncEngine(db, {}, docker_title_reader=docker_title)
        await engine.sync()
        (bad / ".codex").symlink_to(good / ".codex", target_is_directory=True)
        await engine.sync()
        rows = await db.fetch_all("SELECT id, native_title, deleted FROM session ORDER BY id")
        assert rows == [
            {"id": "bad", "native_title": "Previous title", "deleted": 0},
            {"id": "good", "native_title": "Good title", "deleted": 0},
        ]
    finally:
        await db.close()


async def test_native_codex_name_arriving_later_is_resynchronized(tmp_path: Path) -> None:
    db = await FakeDatabase.create()
    try:
        write_title(tmp_path, HarnessKind.CODEX, "First prompt")
        database = tmp_path / ".codex/state_5.sqlite"
        backend = CodexFetchSessions(database, tmp_path / ".codex/sessions")
        engine = SyncEngine(db, {HarnessKind.CODEX: backend})
        await engine.sync()
        before = await db.fetch_one("SELECT * FROM session")
        assert before["native_title"] == "First prompt"
        with sqlite3.connect(database) as native:
            native.execute("UPDATE threads SET name = 'Resolved name'")
        await engine.sync()
        after = await db.fetch_one("SELECT * FROM session")
        assert after["id"] == before["id"]
        assert after["native_title"] == "Resolved name"
    finally:
        await db.close()
