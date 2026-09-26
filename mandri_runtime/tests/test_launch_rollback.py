import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import SessionState
from mandri.runtime.service import RuntimeService


@pytest.fixture
def launch():
    sessions = SimpleNamespace(
        create_session=AsyncMock(return_value=SimpleNamespace(id="session")),
        set_session_state=AsyncMock(),
        set_session_interaction_mode=AsyncMock(),
    )
    routes = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(id="route")),
        delete=AsyncMock(),
    )
    process = Mock(returncode=None, stop=AsyncMock(return_value=0))
    runtime = RuntimeService(
        {"codex": ["codex"], "opencode": ["opencode", "{listen_port}"]},
        sessions=sessions,
        routes=routes,
    )
    runtime._spawn_harness = AsyncMock(return_value=process)
    runtime._resolve_metadata = AsyncMock(return_value=None)
    return runtime, sessions, routes, process


@pytest.mark.parametrize("stage", ["metadata", "spawn", "create", "state", "attach", "opencode"])
async def test_initial_launch_failure_releases_acquired_resources(launch, stage):
    runtime, sessions, routes, process = launch
    failure = RuntimeError(stage)
    harness = "opencode" if stage == "opencode" else "codex"
    if stage == "metadata":
        runtime._resolve_metadata.side_effect = failure
    elif stage == "spawn":
        runtime._spawn_harness.side_effect = failure
    elif stage == "create":
        sessions.create_session.side_effect = failure
    elif stage == "state":
        sessions.set_session_state.side_effect = [failure, None]
    elif stage == "attach":
        runtime._attach_control = Mock(side_effect=failure)
    else:
        runtime._attach_opencode = AsyncMock(side_effect=failure)
    with pytest.raises(RuntimeError, match=stage):
        await runtime.start_session(harness, "provider/model", "/workspace")
    routes.delete.assert_awaited_once_with("route")
    if stage not in {"metadata", "spawn", "create"}:
        process.stop.assert_awaited_once_with(2.0)
    else:
        process.stop.assert_not_awaited()
    if stage == "create":
        runtime._spawn_harness.assert_not_awaited()
    assert runtime.registry.live_ids() == []
    assert runtime.registry.process("session") is None
    if stage in {"state", "attach", "opencode"}:
        sessions.set_session_state.assert_any_await("session", SessionState.STOPPED)
        assert runtime._session_state("session").launched_model is None


async def test_rollback_continues_after_resource_failure_and_cancels_tasks(launch):
    runtime, _sessions, routes, process = launch
    feed = SimpleNamespace(stop=AsyncMock())
    watcher = SimpleNamespace(stop=AsyncMock(side_effect=RuntimeError("watcher failed")))
    control = SimpleNamespace(aclose=AsyncMock())
    tasks = []

    def attach(*_args, **_kwargs):
        state = runtime._session_state("session")
        state.feed, state.watcher, state.control = feed, watcher, control
        state.delivery = Mock()
        task = asyncio.create_task(asyncio.Event().wait())
        tasks.append(task)
        runtime._track_control_task("session", task)
        raise RuntimeError("initialization failed")

    runtime._attach_control = attach
    with pytest.raises(RuntimeError, match="initialization failed"):
        await runtime.start_session("codex", "provider/model", "/workspace")
    assert all(task.cancelled() for task in tasks)
    assert runtime._session_state("session").delivery is None
    control.aclose.assert_awaited_once()
    feed.stop.assert_awaited_once()
    process.stop.assert_awaited_once()
    routes.delete.assert_awaited_once()


async def test_cancelled_initialization_releases_process_and_route(launch):
    runtime, sessions, routes, process = launch
    entered = asyncio.Event()

    async def initialize(*_args):
        entered.set()
        await asyncio.Event().wait()

    runtime._attach_opencode = initialize
    task = asyncio.create_task(runtime.start_session("opencode", "provider/model", "/workspace"))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    process.stop.assert_awaited_once_with(2.0)
    routes.delete.assert_awaited_once_with("route")
    sessions.set_session_state.assert_any_await("session", SessionState.STOPPED)
    assert runtime.registry.live_ids() == []


async def test_successful_launch_retains_resources(launch):
    runtime, sessions, routes, process = launch
    session = await runtime.start_session("codex", "provider/model", "/workspace")
    assert session.id == "session"
    assert runtime.registry.process("session") is process
    process.stop.assert_not_awaited()
    routes.delete.assert_not_awaited()
    sessions.set_session_state.assert_any_await("session", SessionState.LIVE)
    runtime.viewer_joined("session")
