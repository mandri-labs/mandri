import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessKind, SessionId
from mandri.core.types.availability import SessionOwner
from mandri.core.types.execution import ExecutionBackend, ProtectionError
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.sessions.execution_context import DockerSessionContext, transcript_reference
from mandri.sessions.ownership.docker import docker_ownership
from mandri.sessions.ownership.file_lock import try_lock, unlock
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts.resolver import TranscriptResolver

from .substitutes import FakeReader, make_session


@pytest.fixture
def docker_session(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "native" / "session-1"
    for directory in (".codex/sessions", ".claude/projects", ".gemini"):
        (state / directory).mkdir(parents=True)
    session = replace(
        make_session("native-1", harness=HarnessKind.CODEX, project_path=str(workspace)),
        id=SessionId("session-1"),
        execution_backend=ExecutionBackend.DOCKER,
        execution_context=json.dumps(
            {
                "version": "1",
                "workspace_root": str(workspace),
                "container_root": "/workspace",
                "native_state_root": str(state),
                "native_home": "/home/worker",
                "image": "sha256:synthetic",
            }
        ),
    )
    return session, state, workspace


def test_codex_rollout_database_paths_resolve_inside_session_state(docker_session):
    session, state, _ = docker_session
    path = state / ".codex/sessions/rollout-native-1.jsonl"
    event = {"type": "event_msg", "payload": {"type": "user_message", "message": "synthetic"}}
    path.write_text(json.dumps(event) + "\n")
    with sqlite3.connect(state / ".codex/state_5.sqlite") as db:
        db.execute("CREATE TABLE threads (id TEXT, rollout_path TEXT)")
        db.execute(
            "INSERT INTO threads VALUES (?, ?)",
            ("native-1", "/home/worker/.codex/sessions/rollout-native-1.jsonl"),
        )
    host = FakeReader()
    reader = TranscriptResolver({HarnessKind.CODEX: host}).for_session(session)
    page = reader.page(transcript_reference(session), None, 10)
    assert [json.loads(entry) for entry in page.entries] == [event]
    assert host.calls == []


def test_claude_nested_project_uses_container_path(docker_session):
    session, state, workspace = docker_session
    (workspace / "api/src").mkdir(parents=True)
    session = replace(session, harness=HarnessKind.CLAUDE, project_path=str(workspace / "api/src"))
    path = state / ".claude/projects/-workspace-api-src/native-1.jsonl"
    path.parent.mkdir(parents=True)
    event = {"type": "user", "message": {"role": "user", "content": "synthetic"}}
    path.write_text(json.dumps(event) + "\n")
    reference = transcript_reference(session)
    assert reference.project_path == "/workspace/api/src"
    reader = TranscriptResolver({}).for_session(session)
    assert [json.loads(entry) for entry in reader.page(reference, None, 10).entries] == [event]


def test_forged_rollout_path_cannot_read_host_file(docker_session, tmp_path):
    session, state, _ = docker_session
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"private":"synthetic"}\n')
    with sqlite3.connect(state / ".codex/state_5.sqlite") as db:
        db.execute("CREATE TABLE threads (id TEXT, rollout_path TEXT)")
        db.execute("INSERT INTO threads VALUES (?, ?)", ("native-1", str(outside)))
    reader = TranscriptResolver({}).for_session(session)
    with pytest.raises(ProtectionError, match="external path"):
        reader.page(transcript_reference(session), None, 10)


def test_docker_owner_tracks_durable_state_lease(docker_session):
    session, state, _ = docker_session
    assert docker_ownership(session).owner is SessionOwner.UNOWNED
    lock = state.parent / ".session-1.lock"
    with lock.open("w+b") as stream:
        assert try_lock(stream)
        try:
            assert docker_ownership(session).owner is SessionOwner.MANDRI
        finally:
            unlock(stream)
    assert docker_ownership(session).owner is SessionOwner.UNOWNED


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_state_root", "/"),
        ("version", "other"),
        ("native_home", "/home/another"),
        ("container_root", "/"),
    ],
)
def test_invalid_execution_context_never_falls_back_to_host(docker_session, field, value):
    session, _, _ = docker_session
    context = json.loads(session.execution_context)
    context[field] = value
    session = replace(session, execution_context=json.dumps(context))
    with pytest.raises(ProtectionError):
        DockerSessionContext.from_session(session)
    assert docker_ownership(session).owner is SessionOwner.UNKNOWN


@pytest.mark.parametrize("purge", [False, True])
async def test_docker_deletion_uses_owned_state_without_global_harness_backend(
    docker_session, tmp_path, purge
):
    session, state, workspace = docker_session
    source = workspace / "source.py"
    source.write_text("synthetic = True\n")
    other = state.parent / "other-session"
    other.mkdir()
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "test.sqlite")
    await db.migrate()
    engine = Mock()
    try:
        await db.execute(
            "INSERT INTO session (id, harness, native_id, project_path, created_at, updated_at,"
            " state, last_synced_at, execution_backend, execution_context)"
            " VALUES (?, 'codex', 'native-1', ?, 0, 0, 'stopped', 0, 'docker', ?)",
            (str(session.id), str(workspace), session.execution_context),
        )
        service = SessionsService(db, engine)
        await service.delete_session(session.id, purge=purge)
        record = await db.fetch_one("SELECT * FROM session WHERE id = ?", (str(session.id),))
        assert record["deleted"] == 1
        assert state.exists() is not purge
        assert source.read_text() == "synthetic = True\n"
        assert other.is_dir()
        engine.backend.assert_not_called()
        assert (record["execution_context"] is None) is purge
    finally:
        await db.close()


async def test_failed_scope_purge_can_resume_after_native_state_was_removed(
    docker_session, tmp_path
):
    session, state, workspace = docker_session
    source = workspace / "source.py"
    source.write_text("synthetic = True\n")
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "test.sqlite")
    await db.migrate()
    scopes = Mock()
    scopes.delete = AsyncMock(
        side_effect=[
            ProtectionError("privacy_state_unavailable", "Synthetic storage failure"),
            None,
        ]
    )
    try:
        await db.execute(
            "INSERT INTO session (id, harness, native_id, project_path, created_at, updated_at,"
            " state, last_synced_at, execution_backend, execution_context, privacy_mode,"
            " privacy_scope_id) VALUES (?, 'codex', 'native-1', ?, 0, 0, 'stopped', 0,"
            " 'docker', ?, 'surrogate', 'scope-one')",
            (str(session.id), str(workspace), session.execution_context),
        )
        service = SessionsService(db, Mock(), privacy_scopes=scopes)
        with pytest.raises(ProtectionError, match="storage failure"):
            await service.delete_session(session.id, purge=True)
        assert not state.exists()
        journal = await db.fetch_one(
            "SELECT * FROM session_purge WHERE session_id = ?", (session.id,)
        )
        assert journal["privacy_scope_id"] == "scope-one"
        with pytest.raises(ProtectionError, match="purge must finish"):
            await service.ensure_no_pending_purge(session.id)
        restarted = SessionsService(db, Mock(), privacy_scopes=scopes)
        await restarted.delete_session(session.id, purge=True)
        await restarted.ensure_no_pending_purge(session.id)
        assert [call.args for call in scopes.delete.call_args_list] == [
            ("scope-one",),
            ("scope-one",),
        ]
        assert source.read_text() == "synthetic = True\n"
        assert await db.fetch_one("SELECT * FROM session_purge") is None
    finally:
        await db.close()


async def test_explicit_purge_after_tombstone_keeps_workspace(docker_session, tmp_path):
    session, state, workspace = docker_session
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "test.sqlite")
    await db.migrate()
    try:
        await db.execute(
            "INSERT INTO session (id, harness, native_id, project_path, created_at, updated_at,"
            " state, last_synced_at, execution_backend, execution_context)"
            " VALUES (?, 'codex', 'native-1', ?, 0, 0, 'stopped', 0, 'docker', ?)",
            (str(session.id), str(workspace), session.execution_context),
        )
        service = SessionsService(db, Mock())
        await service.delete_session(session.id)
        assert state.exists()
        await service.delete_session(session.id, purge=True)
        assert not state.exists()
        assert workspace.exists()
        await service.delete_session(session.id, purge=True)
        assert workspace.exists()
    finally:
        await db.close()
