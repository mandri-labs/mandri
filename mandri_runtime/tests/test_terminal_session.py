import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.terminal import TerminalStart
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.runtime.service import RuntimeService


@pytest.fixture
def terminal_runtime(tmp_path):
    sessions = SimpleNamespace(
        create_session=AsyncMock(return_value=SimpleNamespace(id="terminal-session")),
        set_session_state=AsyncMock(),
    )
    routes = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(id="terminal-route")),
        delete=AsyncMock(),
    )
    scopes = SimpleNamespace(create=AsyncMock(return_value="terminal-scope"), delete=AsyncMock())
    runtime = RuntimeService(
        {"claude": ["claude", "--print", "--input-format", "stream-json"]},
        sessions=sessions,
        routes=routes,
        privacy_scopes=scopes,
        gateway_port=8000,
        token_issuer=lambda route: f"token-for-{route}",
    )
    runtime._resolve_metadata = AsyncMock(return_value=None)
    runtime._spawn_execution = AsyncMock(return_value=SimpleNamespace())
    runtime.stop_session = AsyncMock()
    spec = TerminalStart(
        harness="claude", model="provider/model", cwd=str(tmp_path), privacy_mode="surrogate"
    )
    return runtime, sessions, routes, scopes, spec


async def test_terminal_binds_privacy_scope_and_launches_native_interface(terminal_runtime):
    runtime, sessions, routes, scopes, spec = terminal_runtime
    async with runtime.terminal_session(spec) as session:
        assert session.privacy_mode is PrivacyMode.SURROGATE
        assert session.privacy_scope_id == "terminal-scope"
        assert routes.create.call_args.kwargs["privacy_scope_id"] == "terminal-scope"
        assert routes.create.call_args.kwargs["privacy_mode"] is PrivacyMode.SURROGATE
        assert sessions.create_session.call_args.kwargs["privacy_scope_id"] == "terminal-scope"
        prepared = runtime._spawn_execution.call_args.args[2]
        assert prepared.argv == ["claude"]
        assert prepared.env["ANTHROPIC_AUTH_TOKEN"] == "token-for-terminal-route"
        assert prepared.env["CLAUDE_CODE_SUBAGENT_MODEL"] == "provider/model"
        assert runtime._spawn_execution.call_args.kwargs["terminal"] is spec
    runtime.stop_session.assert_awaited_once_with("terminal-session", restore_native=False)
    scopes.delete.assert_not_awaited()


async def test_terminal_failed_launch_retains_scope_bound_to_failed_record(terminal_runtime):
    runtime, _, routes, scopes, spec = terminal_runtime
    runtime._spawn_execution.side_effect = RuntimeError("launch failure")
    with pytest.raises(RuntimeError, match="launch failure"):
        async with runtime.terminal_session(spec):
            pytest.fail("Failed launch became usable")
    routes.delete.assert_awaited_once_with("terminal-route")
    scopes.delete.assert_not_awaited()
    assert runtime.registry.live_ids() == []


async def test_terminal_failed_record_creation_discards_unbound_scope(terminal_runtime):
    runtime, sessions, routes, scopes, spec = terminal_runtime
    sessions.create_session.side_effect = RuntimeError("record failure")
    with pytest.raises(RuntimeError, match="record failure"):
        async with runtime.terminal_session(spec):
            pytest.fail("Failed record became usable")
    routes.delete.assert_awaited_once_with("terminal-route")
    scopes.delete.assert_awaited_once_with("terminal-scope")
    runtime._spawn_execution.assert_not_awaited()


async def test_unavailable_docker_terminal_cannot_fall_back_to_host(terminal_runtime):
    runtime, _, routes, scopes, spec = terminal_runtime
    spec.execution_backend = ExecutionBackend.DOCKER
    with pytest.raises(DockerExecutionError, match="Docker execution is not configured"):
        async with runtime.terminal_session(spec):
            pytest.fail("Docker absence was ignored")
    runtime._spawn_execution.assert_not_awaited()
    scopes.create.assert_not_awaited()
    routes.create.assert_not_awaited()


async def test_cancelled_terminal_waits_for_cleanup_to_finish(terminal_runtime):
    runtime, _, _, _, spec = terminal_runtime
    stopping = asyncio.Event()
    release = asyncio.Event()
    stopped = asyncio.Event()

    async def stop(*args, **kwargs):
        stopping.set()
        await release.wait()
        stopped.set()

    runtime.stop_session = stop

    async def run():
        async with runtime.terminal_session(spec):
            pass

    task = asyncio.create_task(run())
    await asyncio.wait_for(stopping.wait(), 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert stopped.is_set()


async def test_missing_privacy_storage_fails_before_any_harness_launch(terminal_runtime):
    runtime, _, routes, _, spec = terminal_runtime
    runtime._privacy_scopes = None
    with pytest.raises(ProtectionError, match="Privacy storage"):
        async with runtime.terminal_session(spec):
            pytest.fail("Privacy absence was ignored")
    runtime._spawn_execution.assert_not_awaited()
    routes.create.assert_not_awaited()
