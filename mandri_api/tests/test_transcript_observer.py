import asyncio
from unittest.mock import AsyncMock

from mandri.api.transcript_observer import TranscriptObserver
from mandri.core.hub import Hub, Topic


async def test_observer_is_shared_passive_and_released() -> None:
    sessions = AsyncMock()
    sessions.transcript_revision.return_value = ("file", 1)
    sessions.external_status.return_value = (False, "custom/model")
    managed = False
    hub = Hub()
    handle = hub.subscribe(Topic("session.one"))
    observer = TranscriptObserver(sessions, hub, lambda _: managed, interval=0.01)
    observer.join("one")
    observer.join("one")
    try:
        first = await asyncio.wait_for(handle.queue.get(), 1)
        assert first["payload"]["raw"]["ownership"] == "external"
        assert first["payload"]["raw"]["external_busy"] is False
        assert first["payload"]["raw"]["external_model"] == "custom/model"
        await asyncio.sleep(0.03)
        assert handle.queue.empty()
        sessions.transcript_revision.return_value = ("file", 2)
        await asyncio.wait_for(handle.queue.get(), 1)
        managed = True
        await asyncio.sleep(0.03)
        calls = sessions.transcript_revision.await_count
        await asyncio.sleep(0.03)
        assert sessions.transcript_revision.await_count == calls
    finally:
        await observer.leave("one")
        hub.unsubscribe(handle)
    calls = sessions.transcript_revision.await_count
    await asyncio.sleep(0.03)
    assert sessions.transcript_revision.await_count == calls
    assert not observer._tasks
