import asyncio
import sys
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import SessionStopCause
from mandri.runtime.process import spawn
from mandri.runtime.registry import SessionRegistry
from mandri.runtime.service import RuntimeService


async def test_natural_exit_releases_process_guard_and_reference() -> None:
    process = Mock(returncode=0, stop=AsyncMock(return_value=0))
    registry = SessionRegistry()
    registry.mark_live("one", process)
    assert await registry.reconcile() == ["one"]
    process.stop.assert_awaited_once()
    assert registry.process("one") is None
    assert await registry.reconcile() == []


async def test_idle_release_preserves_viewed_and_resuming_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RuntimeService({}, idle_release_seconds=5)
    service.registry.mark_live("one", Mock(returncode=None))
    service.viewer_joined("one")
    busy = False
    now = 10.0
    monkeypatch.setattr(service, "is_busy", lambda _: busy)
    monkeypatch.setattr("mandri.runtime.session_lifetime.time.monotonic", lambda: now)
    stop = AsyncMock()
    monkeypatch.setattr(service, "stop_session", stop)
    await service._lifetime.release_idle_processes()
    busy = True
    now = 20.0
    await service._lifetime.release_idle_processes()
    stop.assert_not_awaited()
    busy = False
    await service._lifetime.release_idle_processes()
    now = 26.0
    await service._lifetime.release_idle_processes()
    stop.assert_not_awaited()
    service._session_state("one").viewed = False
    service._session_state("one").resuming = True
    now = 100.0
    await service._lifetime.release_idle_processes()
    stop.assert_not_awaited()
    service._session_state("one").resuming = False
    await service._lifetime.release_idle_processes()
    now = 106.0
    await service._lifetime.release_idle_processes()
    stop.assert_awaited_once_with("one", cause=SessionStopCause.IDLE_TIMEOUT)


def test_force_kill_continues_after_one_process_error():
    service = RuntimeService({})
    broken = Mock(returncode=None)
    broken.kill_now.side_effect = OSError("gone")
    healthy = Mock(returncode=None)
    service.registry.mark_live("broken", broken)
    service.registry.mark_live("healthy", healthy)
    service.kill_all_now()
    broken.kill_now.assert_called_once()
    healthy.kill_now.assert_called_once()


async def test_force_kill_terminates_a_real_managed_process():
    process = await spawn(
        [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(60)"]
    )
    try:
        assert (await asyncio.wait_for(process.read_stdout_line(), 5)).strip() == "ready"
        process.kill_now()
        assert await asyncio.wait_for(process.wait(), 2) != 0
    finally:
        await process.stop(grace=1)
