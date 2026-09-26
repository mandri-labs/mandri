import asyncio
import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessKind, SessionId, SessionState
from mandri.core.types.availability import SessionOwner
from mandri.core.types.execution import ExecutionBackend, ProtectionError
from mandri.core.types.model_selection import ModelSource
from mandri.sessions import service as sessions_service
from mandri.sessions.fork_source import selected_codex_source
from mandri.sessions.ownership.docker import docker_ownership
from mandri.sessions.ownership.file_lock import try_lock, unlock
from mandri.sessions.ownership.service import NativeOwnership
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts.codex_transcripts import CodexTranscriptReader
from mandri.sessions.transcripts.resolver import TranscriptResolver

from .substitutes import make_session


@pytest.fixture(params=["0.154.0", "0.157.0"])
def native_source(tmp_path: Path, request):
    workspace = tmp_path / "workspace"
    (workspace / "api/src").mkdir(parents=True)
    state = tmp_path / "native" / "session-1"
    sessions = state / ".codex/sessions"
    sessions.mkdir(parents=True)
    session = replace(
        make_session(
            "native-1", harness=HarnessKind.CODEX, project_path=str(workspace / "api/src")
        ),
        id=SessionId("session-1"),
        state=SessionState.STOPPED,
        execution_backend=ExecutionBackend.DOCKER,
        execution_context=json.dumps(
            {
                "version": "1",
                "workspace_root": str(workspace),
                "workspace_device": str(workspace.stat().st_dev),
                "workspace_inode": str(workspace.stat().st_ino),
                "container_root": "/workspace",
                "native_state_root": str(state),
                "native_home": "/home/worker",
            }
        ),
    )
    path = sessions / "rollout-native-1.jsonl"
    metadata = {
        "type": "session_meta",
        "payload": {"id": "native-1", "cli_version": request.param, "cwd": "/workspace/api/src"},
    }
    turn = {"type": "response_item", "payload": {"type": "message", "content": "synthetic"}}
    path.write_text(json.dumps(metadata) + "\n" + json.dumps(turn) + "\n")
    (state / ".codex/auth.json").write_text('{"secret":"must never be copied"}')
    return session, path


def test_selected_native_rollout_copies_exactly_with_source_lease(native_source, tmp_path):
    session, path = native_source
    reader = TranscriptResolver({}).for_session(session)
    before = path.read_bytes()
    with selected_codex_source(session, ExecutionBackend.DOCKER, reader) as source:
        assert docker_ownership(session).owner is SessionOwner.MANDRI
        assert source.workspace_root == Path(session.project_path).parents[1]
        destination = tmp_path / "selected.jsonl"
        source.copy_rollout(destination)
        assert destination.read_bytes() == before
        assert b"must never be copied" not in destination.read_bytes()
    assert docker_ownership(session).owner is SessionOwner.UNOWNED
    assert path.read_bytes() == before
    with pytest.raises(ProtectionError):
        source.verify()


@pytest.mark.parametrize(
    "change",
    [
        {"state": SessionState.LIVE},
        {"native_id": None},
        {"harness": HarnessKind.CLAUDE},
        {"model_source": ModelSource.NATIVE},
        {"deleted": True},
    ],
)
def test_unqualified_sources_fail_before_mutation(native_source, change):
    session, path = native_source
    reader = TranscriptResolver({}).for_session(session)
    before = path.read_bytes()
    with (
        pytest.raises(ProtectionError),
        selected_codex_source(replace(session, **change), ExecutionBackend.DOCKER, reader),
    ):
        pytest.fail("unqualified source was accepted")
    assert path.read_bytes() == before


def test_cross_backend_history_is_not_rewritten_implicitly(native_source):
    session, _ = native_source
    reader = TranscriptResolver({}).for_session(session)
    with (
        pytest.raises(ProtectionError),
        selected_codex_source(session, ExecutionBackend.HOST, reader),
    ):
        pytest.fail("unqualified path transition was accepted")


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "another-native-id"),
        ("cwd", "/workspace/other"),
        ("cli_version", "0.155.0"),
        ("cli_version", None),
        ("cli_version", []),
    ],
)
def test_native_identity_directory_and_version_are_checked(native_source, field, value):
    session, path = native_source
    rows = path.read_text().splitlines()
    metadata = json.loads(rows[0])
    metadata["payload"][field] = value
    path.write_text(json.dumps(metadata) + "\n" + "\n".join(rows[1:]) + "\n")
    reader = TranscriptResolver({}).for_session(session)
    with (
        pytest.raises(ProtectionError),
        selected_codex_source(session, ExecutionBackend.DOCKER, reader),
    ):
        pytest.fail("incompatible native metadata was accepted")
    assert docker_ownership(session).owner is SessionOwner.UNOWNED


def test_existing_native_writer_prevents_fork(native_source):
    session, path = native_source
    reader = TranscriptResolver({}).for_session(session)
    lock = path.parents[3] / ".session-1.lock"
    with lock.open("a+b") as handle:
        assert try_lock(handle)
        try:
            with (
                pytest.raises(ProtectionError, match="writer"),
                selected_codex_source(session, ExecutionBackend.DOCKER, reader),
            ):
                pytest.fail("native writer lease was bypassed")
        finally:
            unlock(handle)


def test_source_change_prevents_copy(native_source, tmp_path):
    session, path = native_source
    reader = TranscriptResolver({}).for_session(session)
    destination = tmp_path / "selected.jsonl"
    with (
        pytest.raises(ProtectionError),
        selected_codex_source(session, ExecutionBackend.DOCKER, reader) as source,
    ):
        original = path.read_bytes()
        path.write_bytes(original.replace(b"synthetic", b"different"))
        source.copy_rollout(destination)
    assert not destination.exists()
    assert docker_ownership(session).owner is SessionOwner.UNOWNED


def test_existing_destination_is_preserved(native_source, tmp_path):
    session, _ = native_source
    reader = TranscriptResolver({}).for_session(session)
    destination = tmp_path / "selected.jsonl"
    destination.write_bytes(b"preexisting")
    with (
        selected_codex_source(session, ExecutionBackend.DOCKER, reader) as source,
        pytest.raises(FileExistsError),
    ):
        source.copy_rollout(destination)
    assert destination.read_bytes() == b"preexisting"


@pytest.mark.parametrize("during_lease", [False, True])
def test_replaced_workspace_cannot_be_forked(native_source, tmp_path, during_lease):
    session, path = native_source
    reader = TranscriptResolver({}).for_session(session)
    root = Path(session.project_path).parents[1]
    destination = tmp_path / "selected.jsonl"
    original = path.read_bytes()

    def replace_workspace():
        root.rename(root.with_name("retained-original"))
        (root / "api/src").mkdir(parents=True)

    if not during_lease:
        replace_workspace()
    with (
        pytest.raises(ProtectionError, match="workspace identity changed"),
        selected_codex_source(session, ExecutionBackend.DOCKER, reader) as source,
    ):
        if during_lease:
            replace_workspace()
        source.copy_rollout(destination)
    assert not destination.exists()
    assert path.read_bytes() == original
    assert docker_ownership(session).owner is SessionOwner.UNOWNED


def test_legacy_context_without_workspace_identity_cannot_fork(native_source):
    session, path = native_source
    context = json.loads(session.execution_context)
    del context["workspace_inode"]
    session = replace(session, execution_context=json.dumps(context))
    reader = TranscriptResolver({}).for_session(session)
    original = path.read_bytes()
    with (
        pytest.raises(ProtectionError, match="workspace identity changed"),
        selected_codex_source(session, ExecutionBackend.DOCKER, reader),
    ):
        pytest.fail("Missing workspace provenance was accepted")
    assert path.read_bytes() == original


def test_host_fork_uses_only_selected_native_store(native_source, tmp_path):
    session, path = native_source
    session = replace(session, execution_backend=ExecutionBackend.HOST, execution_context=None)
    rows = path.read_text().splitlines()
    metadata = json.loads(rows[0])
    metadata["payload"]["cwd"] = str(session.project_path)
    path.write_text(json.dumps(metadata) + "\n" + "\n".join(rows[1:]) + "\n")
    reader = CodexTranscriptReader(path.parent)
    with selected_codex_source(session, ExecutionBackend.HOST, reader) as source:
        assert source.rollout_path == path
        source.copy_rollout(tmp_path / "selected.jsonl")


async def test_cancelled_admission_releases_source_lease(native_source, monkeypatch):
    session, _ = native_source
    acquired, release = threading.Event(), threading.Event()

    @contextmanager
    def delayed_source(*args):
        with selected_codex_source(*args) as source:
            acquired.set()
            assert release.wait(5)
            yield source

    monkeypatch.setattr(sessions_service, "selected_codex_source", delayed_source)
    service = SessionsService(Mock(), Mock(), TranscriptResolver({}))
    monkeypatch.setattr(service, "ensure_no_pending_purge", AsyncMock())
    monkeypatch.setattr(service, "get_session", AsyncMock(return_value=session))
    monkeypatch.setattr(
        service, "native_ownership", AsyncMock(return_value=NativeOwnership(SessionOwner.UNOWNED))
    )

    async def enter():
        async with service.codex_fork_source(session.id, ExecutionBackend.DOCKER):
            pytest.fail("cancelled fork entered the caller")

    task = asyncio.create_task(enter())
    assert await asyncio.to_thread(acquired.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert docker_ownership(session).owner is SessionOwner.UNOWNED
