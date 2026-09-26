import asyncio
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, SessionId, SessionState
from mandri.core.types.availability import SessionOwner
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError
from mandri.core.types.model_selection import ModelSource
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.runtime.service import RuntimeService
from mandri.sessions.errors import SessionRunningError
from mandri.sessions.ownership.service import NativeOwnership
from mandri.sessions.service import SessionsService


@pytest.fixture
async def sessions(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "state.sqlite")
    await db.migrate()
    try:
        yield SessionsService(
            db,
            SimpleNamespace(activity_of=lambda _: None, backend=lambda _: None),
            worktrees_dir=tmp_path / "worktrees",
        )
    finally:
        await db.close()


@pytest.fixture
def repository(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    commands = [
        ["init", "-b", "main"],
        ["config", "user.email", "test@example.test"],
        ["config", "user.name", "Test"],
        ["add", "file.txt"],
        ["commit", "-m", "Initial"],
    ]
    (path / "file.txt").write_text("initial\n")
    for command in commands:
        subprocess.run(["git", "-C", str(path), *command], check=True, capture_output=True)
    return path


def runtime_for(sessions, scopes=None):
    runtime = RuntimeService(
        {"claude": ["claude"]},
        sessions=sessions,
        privacy_scopes=scopes,
        gateway_port=9876,
        token_issuer=lambda _: "synthetic-token",
    )
    runtime._spawn_harness = AsyncMock(return_value=SimpleNamespace(returncode=None))
    runtime._resolve_metadata = AsyncMock(return_value=None)
    return runtime


async def test_native_worktree_uses_isolated_cwd_and_persists_identity(
    tmp_path, repository, sessions
):
    runtime = runtime_for(sessions)
    result = await runtime.start_session(
        "claude",
        "default",
        str(repository),
        model_source=ModelSource.NATIVE,
        worktree=True,
        worktree_id="feature/search",
    )
    assert result.worktree.id == "feature/search"
    assert result.project_path == result.worktree.path
    assert runtime._spawn_harness.call_args.kwargs["cwd"] == result.worktree.path
    persisted = await sessions.get_session(SessionId(result.id))
    assert persisted.worktree == result.worktree
    assert persisted.project_path == result.worktree.path
    await sessions.set_session_state(persisted.id, SessionState.STOPPED)
    await sessions.delete_session(persisted.id)


async def test_worktree_surrogate_seeds_both_source_and_execution_paths(
    tmp_path, repository, sessions
):
    scopes = SimpleNamespace(
        readiness=AsyncMock(), create=AsyncMock(return_value="scope"), add_workspace=AsyncMock()
    )
    runtime = runtime_for(sessions, scopes)
    result = await runtime.start_session(
        "claude",
        "provider/model",
        str(repository),
        privacy_mode=PrivacyMode.SURROGATE,
        worktree=True,
    )
    scopes.create.assert_awaited_once_with(str(repository))
    scopes.add_workspace.assert_awaited_once_with("scope", result.worktree.path)
    assert result.privacy_mode is PrivacyMode.SURROGATE
    await sessions.set_session_state(SessionId(result.id), SessionState.STOPPED)
    await sessions.delete_session(SessionId(result.id))


async def test_failed_start_removes_checkout_branch_and_session(tmp_path, repository, sessions):
    runtime = runtime_for(sessions)
    runtime._spawn_harness.side_effect = RuntimeError("synthetic spawn failure")
    with pytest.raises(RuntimeError, match="synthetic spawn failure"):
        await runtime.start_session(
            "claude", "default", str(repository), model_source=ModelSource.NATIVE, worktree=True
        )
    assert await sessions.list_sessions() == []
    assert list((tmp_path / "worktrees").iterdir()) == []


async def test_cancelled_start_waits_for_cleanup(tmp_path, repository, sessions):
    runtime = runtime_for(sessions)
    spawning = asyncio.Event()

    async def block(*args, **kwargs):
        spawning.set()
        await asyncio.Future()

    runtime._spawn_harness.side_effect = block
    operation = str(uuid.uuid4())
    pending = asyncio.create_task(
        runtime.start_session(
            "claude",
            "default",
            str(repository),
            model_source=ModelSource.NATIVE,
            worktree=True,
            operation_id=operation,
        )
    )
    await spawning.wait()
    await runtime.cancel_start(operation)
    with pytest.raises(ProtectionError) as error:
        await pending
    assert error.value.code == "operation_cancelled"
    assert await sessions.list_sessions() == []
    assert list((tmp_path / "worktrees").iterdir()) == []


async def test_docker_and_worktree_rejected_before_side_effects(tmp_path, repository, sessions):
    runtime = runtime_for(sessions)
    with pytest.raises(ProtectionError) as error:
        await runtime.start_session(
            "claude",
            "provider/model",
            str(repository),
            execution_backend=ExecutionBackend.DOCKER,
            worktree_id="feature",
        )
    assert error.value.code == "worktree_docker_incompatible"
    runtime._spawn_harness.assert_not_called()
    assert await sessions.list_sessions() == []
    assert not (tmp_path / "worktrees").exists()


async def test_cancellation_during_git_preparation_waits_then_removes_owned_files(
    tmp_path, repository, monkeypatch, sessions
):
    runtime = runtime_for(sessions)
    entered = asyncio.Event()
    release = asyncio.Event()
    prepare = sessions.worktrees._prepare

    async def gated(*args):
        value = await prepare(*args)
        entered.set()
        await release.wait()
        return value

    monkeypatch.setattr(sessions.worktrees, "_prepare", gated)
    operation = str(uuid.uuid4())
    pending = asyncio.create_task(
        runtime.start_session(
            "claude",
            "default",
            str(repository),
            model_source=ModelSource.NATIVE,
            worktree=True,
            operation_id=operation,
        )
    )
    await entered.wait()
    cancelling = asyncio.create_task(runtime.cancel_start(operation))
    await asyncio.sleep(0)
    assert not cancelling.done()
    release.set()
    await cancelling
    with pytest.raises(ProtectionError) as error:
        await pending
    assert error.value.code == "operation_cancelled"
    runtime._spawn_harness.assert_not_called()
    assert await sessions.list_sessions() == []
    assert list((tmp_path / "worktrees").iterdir()) == []


@pytest.mark.parametrize("entrypoint", ["resume_session", "_resume_session"])
async def test_delete_waits_for_resume_and_then_refuses_live_session(
    tmp_path, repository, sessions, entrypoint
):
    runtime = runtime_for(sessions)
    session = await sessions.create_session(HarnessKind.CLAUDE)
    await sessions.worktrees.prepare(session.id, str(repository), "feature")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def resume(*args, **kwargs):
        entered.set()
        await release.wait()
        await sessions.set_session_state(session.id, SessionState.LIVE)

    runtime._resume_in_workspace = AsyncMock(side_effect=resume)
    pending = asyncio.create_task(getattr(runtime, entrypoint)(str(session.id)))
    await entered.wait()
    deleting = asyncio.create_task(sessions.delete_session(session.id))
    await asyncio.sleep(0)
    assert not deleting.done()
    release.set()
    await pending
    with pytest.raises(SessionRunningError):
        await deleting
    await sessions.set_session_state(session.id, SessionState.STOPPED)
    await sessions.delete_session(session.id)


async def test_integration_blocks_resume_and_closed_worktree_is_read_only(
    repository, sessions, monkeypatch
):
    runtime = runtime_for(sessions)
    session = await sessions.create_session(HarnessKind.CLAUDE)
    worktree = await sessions.worktrees.prepare(session.id, str(repository), "feature")
    from_path = repository.parent / "worktrees" / worktree.path.split("/")[-1]
    (from_path / "file.txt").write_text("changed\n")
    plan = await runtime.preview_worktree(str(session.id), "main", "squash")
    entered = asyncio.Event()
    proceed = asyncio.Event()
    original = sessions.worktrees.integrate

    async def blocked(*args):
        entered.set()
        await proceed.wait()
        return await original(*args)

    monkeypatch.setattr(sessions.worktrees, "integrate", blocked)
    pending = asyncio.create_task(
        runtime.integrate_worktree(str(session.id), "main", "squash", plan.token, "Feature")
    )
    await entered.wait()
    with pytest.raises(SessionRunningError):
        await runtime.resume_session(str(session.id))
    proceed.set()
    await pending
    await runtime.finish_worktree(str(session.id))
    availability = await runtime.session_availability(str(session.id))
    assert not availability.can_resume
    assert not availability.can_restore
    assert availability.reason == "worktree_closed"
    with pytest.raises(ProtectionError) as error:
        await runtime.resume_session(str(session.id))
    assert error.value.code == "worktree_closed"
    assert not runtime._spawn_harness.called


async def test_starting_worktree_cannot_be_integrated_before_process_registration(
    repository, sessions
):
    session = await sessions.create_session(HarnessKind.CLAUDE, initial_state=SessionState.LIVE)
    await sessions.worktrees.prepare(session.id, str(repository), "feature")
    runtime = runtime_for(sessions)
    with pytest.raises(SessionRunningError):
        await runtime.preview_worktree(str(session.id), "main", "squash")


@pytest.mark.parametrize("strategy", ["squash", "merge"])
@pytest.mark.parametrize("stored_state", [SessionState.STOPPED, SessionState.LIVE])
async def test_unowned_session_integrates_with_a_fresh_runtime(
    repository, sessions, strategy, stored_state
):
    session = await sessions.create_session(HarnessKind.CLAUDE, initial_state=stored_state)
    worktree = await sessions.worktrees.prepare(session.id, str(repository), "feature")
    await sessions.reveal_native_id(session.id, HarnessSessionId("synthetic-native-session"))
    sessions.native_ownership = AsyncMock(return_value=NativeOwnership(SessionOwner.UNOWNED))
    (Path(worktree.path) / "file.txt").write_text("changed\n")
    runtime = runtime_for(sessions)
    assert runtime.registry.status(str(session.id)) is None

    plan = await runtime.preview_worktree(str(session.id), "main", strategy)
    await runtime.integrate_worktree(str(session.id), "main", strategy, plan.token, "Feature")
    assert (repository / "file.txt").read_text() == "changed\n"
    await runtime.finish_worktree(str(session.id))

    record = await sessions.get_session(session.id)
    assert record.state is SessionState.STOPPED
    assert record.worktree.state == "closed"
    assert runtime.registry.status(str(session.id)) is None
    runtime._spawn_harness.assert_not_called()


@pytest.mark.parametrize("owner", [SessionOwner.EXTERNAL, SessionOwner.UNKNOWN])
async def test_stale_live_worktree_still_requires_proven_absence_of_a_writer(
    repository, sessions, owner
):
    session = await sessions.create_session(HarnessKind.CLAUDE, initial_state=SessionState.LIVE)
    await sessions.worktrees.prepare(session.id, str(repository), "feature")
    await sessions.reveal_native_id(session.id, HarnessSessionId("synthetic-native-session"))
    sessions.native_ownership = AsyncMock(return_value=NativeOwnership(owner))
    runtime = runtime_for(sessions)
    with pytest.raises(SessionRunningError):
        await runtime.preview_worktree(str(session.id), "main", "squash")
    assert (await sessions.get_session(session.id)).state is SessionState.LIVE
    runtime._spawn_harness.assert_not_called()
