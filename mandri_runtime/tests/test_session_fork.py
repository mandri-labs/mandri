from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import HarnessKind
from mandri.core.types.execution import (
    ExecutionBackend,
    PrivacyMode,
    ProtectionError,
    SessionPolicy,
)
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.service import RuntimeService


def source_record():
    return SimpleNamespace(
        session=SimpleNamespace(
            id="source",
            harness=HarnessKind.CODEX,
            model="provider/model",
            model_source=ModelSource.GATEWAY,
            interaction_mode=SimpleNamespace(mode="ask"),
            reasoning_effort="medium",
            privacy_mode=PrivacyMode.SURROGATE,
            privacy_scope_id="source-map",
        ),
        workspace_root=Path("/workspace"),
        native_id="native-source",
    )


@pytest.mark.parametrize("version", ["0.154.0", "0.157.0", "0.159.0", "0.161.0"])
async def test_explicit_fork_uses_source_context_and_selected_target_policy(version):
    source = source_record()
    held = False

    @asynccontextmanager
    async def selected(session_id, target):
        nonlocal held
        assert session_id == "source" and target is ExecutionBackend.DOCKER
        held = True
        try:
            yield source
        finally:
            held = False

    runtime = RuntimeService(
        {},
        sessions=SimpleNamespace(codex_fork_source=selected, ensure_no_pending_purge=AsyncMock()),
    )

    async def start(*args, **kwargs):
        assert held
        assert args == ("codex", "provider/model", str(Path("/workspace")))
        assert kwargs["mode"] == "ask"
        assert kwargs["_fork_source"] is source
        assert kwargs["execution_backend"] is ExecutionBackend.DOCKER
        assert kwargs["privacy_mode"] is PrivacyMode.NONE
        return SimpleNamespace(id="new")

    runtime.start_session = AsyncMock(side_effect=start)
    runtime.docker_harness_version = AsyncMock(return_value=version)
    assert (
        await runtime.fork_session("source", ExecutionBackend.DOCKER, PrivacyMode.NONE)
    ).id == "new"
    assert not held


async def test_unqualified_docker_version_does_not_open_source_or_launch_fork():
    sessions = SimpleNamespace(codex_fork_source=AsyncMock(), ensure_no_pending_purge=AsyncMock())
    runtime = RuntimeService({}, sessions=sessions)
    runtime.docker_harness_version = AsyncMock(return_value="0.158.0")
    runtime.start_session = AsyncMock()
    with pytest.raises(ProtectionError, match="not qualified"):
        await runtime.fork_session("source", ExecutionBackend.DOCKER, PrivacyMode.NONE)
    sessions.codex_fork_source.assert_not_called()
    runtime.start_session.assert_not_called()


async def test_source_changes_after_launch_roll_back_only_new_fork():
    @asynccontextmanager
    async def changed(*args):
        yield source_record()
        raise ProtectionError("session_transition_unsupported", "Source changed")

    runtime = RuntimeService(
        {}, sessions=SimpleNamespace(codex_fork_source=changed, ensure_no_pending_purge=AsyncMock())
    )
    process = SimpleNamespace()
    runtime.start_session = AsyncMock(return_value=SimpleNamespace(id="new", process=process))
    runtime.docker_harness_version = AsyncMock(return_value="0.157.0")
    runtime._rollback_process = AsyncMock()
    runtime._rollback_session = AsyncMock()
    with pytest.raises(ProtectionError, match="Source changed"):
        await runtime.fork_session("source", ExecutionBackend.DOCKER, PrivacyMode.SURROGATE)
    runtime._rollback_process.assert_awaited_once_with(process)
    runtime._rollback_session.assert_awaited_once_with("new")


async def test_protected_fork_preserves_alias_scope_without_reusing_remote_session():
    scopes = SimpleNamespace(fork=AsyncMock(return_value="new-map"), create=AsyncMock())
    runtime = RuntimeService({}, privacy_scopes=scopes)
    source = source_record()
    result = await runtime._create_privacy_scope(
        SessionPolicy(ExecutionBackend.DOCKER, PrivacyMode.SURROGATE), "/workspace", source
    )
    assert result == "new-map"
    scopes.fork.assert_awaited_once_with("source-map")
    scopes.create.assert_not_awaited()
    source.session.privacy_scope_id = None
    with pytest.raises(ProtectionError, match="source privacy scope"):
        await runtime._create_privacy_scope(
            SessionPolicy(ExecutionBackend.DOCKER, PrivacyMode.SURROGATE), "/workspace", source
        )


async def test_source_failure_after_worktree_launch_cleans_new_worktree():
    @asynccontextmanager
    async def changed(*args):
        yield source_record()
        raise ProtectionError("session_transition_unsupported", "Source changed")

    runtime = RuntimeService(
        {}, sessions=SimpleNamespace(codex_fork_source=changed, ensure_no_pending_purge=AsyncMock())
    )
    runtime.start_session = AsyncMock(
        return_value=SimpleNamespace(
            id="new", process=SimpleNamespace(), worktree=SimpleNamespace(id="feature")
        )
    )
    runtime._rollback_process = AsyncMock()
    runtime._rollback_session = AsyncMock()
    runtime._rollback_worktree = AsyncMock()
    with pytest.raises(ProtectionError, match="Source changed"):
        await runtime.fork_session(
            "source", ExecutionBackend.HOST, PrivacyMode.SURROGATE, worktree=True
        )
    runtime._rollback_worktree.assert_awaited_once_with("new")
