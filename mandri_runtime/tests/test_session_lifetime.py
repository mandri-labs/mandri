import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import SessionStopCause
from mandri.core.types.execution import ExecutionPhase
from mandri.runtime.service import RuntimeService
from mandri.sessions.errors import SessionRunningError


async def test_resume_reserves_session_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    service = RuntimeService({})
    entered = asyncio.Event()
    release = asyncio.Event()

    async def load(session_id: str) -> None:
        entered.set()
        await release.wait()
        raise ValueError("load failed")

    loader = AsyncMock(side_effect=load)
    monkeypatch.setattr(service, "_load_resumable_record", loader)
    first = asyncio.create_task(service.resume_session("session"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(SessionRunningError):
            await service.resume_session("session")
        assert loader.await_count == 1
    finally:
        release.set()
        with pytest.raises(ValueError, match="load failed"):
            await first
    with pytest.raises(ValueError, match="load failed"):
        await service.resume_session("session")
    assert loader.await_count == 2


async def test_viewer_join_prevents_already_scheduled_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RuntimeService({})
    stop = AsyncMock()
    monkeypatch.setattr(service, "stop_session", stop)
    monkeypatch.setattr(service, "is_busy", lambda session_id: False)
    service.registry.mark_live("session")
    service.viewer_joined("session")
    assert await service._lifetime.execute_zero_viewer_stop("session")
    stop.assert_not_awaited()


async def test_resume_timer_is_not_armed_with_existing_viewer() -> None:
    service = RuntimeService({})
    service.viewer_joined("session")
    service._lifetime.arm_zero_viewer_decision("session")
    assert service._session_state("session").lifetime_task is None
    service.viewers_zero("session")
    task = service._session_state("session").lifetime_task
    service.viewer_joined("session")
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_zero_viewer_release_uses_configured_continuous_idle_period(monkeypatch):
    runtime = RuntimeService({}, idle_release_seconds=60)
    runtime.registry.mark_live("session", SimpleNamespace(returncode=None))
    now = 0.0
    busy = False
    runtime.is_busy = lambda _: busy
    monkeypatch.setattr("mandri.runtime.session_lifetime.time.monotonic", lambda: now)
    stop = AsyncMock()
    runtime.stop_session = stop
    runtime._lifetime.arm_zero_viewer_decision("session")
    try:
        now = 4.0
        await runtime._lifetime.release_idle_processes()
        stop.assert_not_awaited()
        busy = True
        now = 59.0
        await runtime._lifetime.release_idle_processes()
        busy = False
        now = 60.0
        await runtime._lifetime.release_idle_processes()
        now = 119.0
        await runtime._lifetime.release_idle_processes()
        stop.assert_not_awaited()
        now = 120.0
        await runtime._lifetime.release_idle_processes()
        stop.assert_awaited_once_with("session", cause=SessionStopCause.IDLE_TIMEOUT)
    finally:
        runtime.viewer_joined("session")


@pytest.mark.parametrize("activity", ["viewer", "resuming", "prompt"])
async def test_idle_release_rechecks_activity_after_persisting_stop(activity):
    runtime = RuntimeService({})
    process = SimpleNamespace(returncode=None, stop=AsyncMock(return_value=0))
    runtime.registry.mark_live("session", process)
    entered = asyncio.Event()
    proceed = asyncio.Event()
    busy = False
    runtime.is_busy = lambda _: busy

    async def persist(*args, **kwargs):
        entered.set()
        await proceed.wait()

    runtime._persist_execution_exit = AsyncMock(side_effect=persist)
    runtime._executions.phase = AsyncMock()
    task = asyncio.create_task(runtime.stop_session("session", cause=SessionStopCause.IDLE_TIMEOUT))
    await asyncio.wait_for(entered.wait(), 1)
    if activity == "viewer":
        runtime.viewer_joined("session")
    elif activity == "resuming":
        runtime._session_state("session").resuming = True
    else:
        busy = True
    proceed.set()
    assert await task == 0
    process.stop.assert_not_awaited()
    assert runtime.registry.status("session") == "live"
    assert not runtime._session_state("session").stopping
    runtime._executions.phase.assert_awaited_once_with("session", ExecutionPhase.READY)
