import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.hub import Hub
from mandri.core.ids import SessionState
from mandri.core.types.execution import ExecutionPhase
from mandri.runtime.service import RuntimeService
from mandri.runtime.session_feed import session_topic


@pytest.mark.parametrize("returncode", [None, 0])
async def test_unexpected_stdout_eof_releases_writer_before_publishing_stopped(returncode):
    hub = Hub()
    runtime = RuntimeService({}, hub=hub)
    process = SimpleNamespace(returncode=returncode, stop=AsyncMock(return_value=0))
    runtime.registry.mark_live("session", process, "codex")
    runtime._persist_execution_exit = AsyncMock()
    runtime._mark_db_state = AsyncMock()
    runtime._restore_after_release = AsyncMock()
    handle = hub.subscribe(session_topic("session"))
    completed = asyncio.create_task(asyncio.sleep(0))
    await runtime._close_control_after_feed_eof("session", completed)
    process.stop.assert_awaited_once_with(1.0)
    runtime._mark_db_state.assert_awaited_once_with("session", SessionState.STOPPED)
    assert runtime._persist_execution_exit.await_args_list[-1].args == (
        "session",
        ExecutionPhase.FAILED,
    )
    assert runtime.registry.status("session") == "stopped"
    assert runtime.registry.process("session") is None
    runtime._restore_after_release.assert_not_awaited()
    frames = []
    while not handle.queue.empty():
        frames.append(handle.queue.get_nowait()["payload"])
    assert frames[-1]["type"] == "session_stopped"
    assert frames[-1]["raw"]["state"] == "stopped"
    assert frames[-1]["raw"]["cause"] == "crash"
    hub.unsubscribe(handle)


@pytest.mark.parametrize("transition", ["stopping", "resuming"])
async def test_expected_eof_during_transition_does_not_stop_replacement(transition):
    runtime = RuntimeService({})
    process = SimpleNamespace(returncode=None, stop=AsyncMock())
    runtime.registry.mark_live("session", process, "codex")
    setattr(runtime._session_state("session"), transition, True)
    completed = asyncio.create_task(asyncio.sleep(0))
    await runtime._close_control_after_feed_eof("session", completed)
    process.stop.assert_not_awaited()
    assert runtime.registry.status("session") == "live"
