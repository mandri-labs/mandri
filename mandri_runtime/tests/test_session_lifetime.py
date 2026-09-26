import asyncio
from unittest.mock import AsyncMock

import pytest
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
