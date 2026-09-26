import json
import sqlite3

import pytest
from mandri.core.ids import HarnessKind
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.sessions.adapters.codex_fetch_sessions import (
    CodexRolloutScanAdapter,
    CodexStateDbFetchAdapter,
)
from mandri.sessions.agents.store import AgentStore
from mandri.sessions.codex_source import is_internal_source
from mandri.sessions.sync import SyncEngine

from mandri_sessions.tests.substitutes import FakeDatabase, insert_session_row


def seed_native_db(path, variant):
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE threads (id TEXT, cwd TEXT, model TEXT, title TEXT,"
            " first_user_message TEXT, archived INT, archived_at INT,"
            " created_at_ms INT, updated_at_ms INT, source TEXT)"
        )
        for identity, source in (
            ("guardian", {variant: {"other": "guardian"}}),
            ("root", "cli"),
            ("child", {variant: {"thread_spawn": {"parent_thread_id": "root"}}}),
        ):
            db.execute(
                "INSERT INTO threads VALUES (?, '/project', NULL, ?, NULL, 0, NULL, 1, 2, ?)",
                (identity, "The following is the Codex agent history", json.dumps(source)),
            )


@pytest.mark.parametrize("variant", ["subagent", "sub_agent", "subAgent"])
def test_guardian_sqlite_filter_preserves_user_and_delegated_sessions(tmp_path, variant):
    path = tmp_path / "state_5.sqlite"
    seed_native_db(path, variant)
    assert {str(row.native_id) for row in CodexStateDbFetchAdapter(path).fetch()} == {
        "root",
        "child",
    }
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM threads").fetchone()[0] == 3


@pytest.mark.parametrize("variant", ["subagent", "sub_agent", "subAgent"])
def test_guardian_rollout_filter_preserves_user_and_delegated_sessions(tmp_path, variant):
    identities = [f"10000000-0000-4000-8000-00000000000{index}" for index in range(3)]
    sources = [
        {variant: {"other": "guardian"}},
        "cli",
        {variant: {"thread_spawn": {"parent_thread_id": identities[1]}}},
    ]
    for identity, source in zip(identities, sources, strict=True):
        path = tmp_path / f"rollout-2026-09-11T12-00-00-{identity}.jsonl"
        path.write_text(
            json.dumps({"type": "session_meta", "payload": {"id": identity, "source": source}})
        )
    assert {str(row.native_id) for row in CodexRolloutScanAdapter(tmp_path).fetch()} == set(
        identities[1:]
    )
    assert len(list(tmp_path.glob("*.jsonl"))) == 3


@pytest.mark.parametrize(
    "source",
    [None, "cli", "not json", {}, {"subagent": "guardian"}, {"subagent": {"other": "review"}}],
)
def test_only_explicit_guardian_metadata_is_internal(source):
    assert not is_internal_source(source)


async def test_previously_imported_guardian_disappears_on_sync_without_native_deletion(tmp_path):
    path = tmp_path / "state_5.sqlite"
    seed_native_db(path, "subagent")
    db = await FakeDatabase.create()
    await insert_session_row(db, "old-import", harness="codex", native_id="guardian")
    engine = SyncEngine(db, {HarnessKind.CODEX: CodexStateDbFetchAdapter(path)})
    await engine.sync()
    assert (await db.fetch_one("SELECT deleted FROM session WHERE id='old-import'"))["deleted"] == 1
    assert len(await db.fetch_all("SELECT id FROM session WHERE deleted=0")) == 2
    with sqlite3.connect(path) as native:
        assert native.execute("SELECT count(*) FROM threads").fetchone()[0] == 3


async def test_mandri_only_sessions_are_roots_before_native_discovery(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "mandri.db")
    try:
        await db.migrate()
        await insert_session_row(db, "stopped-launch", harness="codex", state="stopped")
        await insert_session_row(db, "starting-launch", harness="codex", state="live")
        store = AgentStore(db)
        assert await store.classified() == ["starting-launch", "stopped-launch"]
        await db.execute("UPDATE session SET native_id='assigned' WHERE id='starting-launch'")
        assert await store.classified() == ["stopped-launch"]
    finally:
        await db.close()
