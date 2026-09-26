import asyncio
from unittest.mock import AsyncMock

import pytest
from mandri.api.agent_observer import AgentObserver
from mandri.core.errors import MandriError
from mandri.core.hub import Hub, Topic
from mandri.core.ports.transcripts import TranscriptPage
from mandri.core.types.sessions import SessionError


async def test_child_history_observation_is_shared_private_and_released():
    service = AsyncMock()
    service.history.return_value = TranscriptPage(
        ['{"text":"synthetic private history"}'], None, False
    )
    hub = Hub()
    handle = hub.subscribe(Topic("agent.child"))
    observer = AgentObserver(service, hub, interval=0.01)
    observer.join("agent.child")
    observer.join("agent.child")
    try:
        first = await asyncio.wait_for(handle.queue.get(), 1)
        assert first["payload"]["raw"] == {"type": "history_changed"}
        assert "synthetic private" not in str(first)
        await observer.leave("agent.child")
        service.history.return_value = TranscriptPage(['{"text":"new"}'], None, False)
        await asyncio.wait_for(handle.queue.get(), 1)
        await observer.leave("agent.child")
        calls = service.history.await_count
        await asyncio.sleep(0.03)
        assert service.history.await_count == calls
    finally:
        await observer.close()
        hub.unsubscribe(handle)
    assert not observer._tasks


@pytest.mark.parametrize(
    "error", [SessionError("missing transcript"), MandriError("database unavailable")]
)
async def test_observer_reports_unavailable_once_and_recovers(error):
    service = AsyncMock()
    service.history.side_effect = error
    hub = Hub()
    handle = hub.subscribe(Topic("agent.child"))
    observer = AgentObserver(service, hub, interval=0.01)
    observer.join("agent.child")
    try:
        first = await asyncio.wait_for(handle.queue.get(), 1)
        assert first["payload"]["raw"]["unavailable"]
        await asyncio.sleep(0.04)
        assert handle.queue.empty()
        service.history.side_effect = None
        service.history.return_value = TranscriptPage([], None, False)
        recovered = await asyncio.wait_for(handle.queue.get(), 1)
        assert recovered["payload"]["raw"] == {"type": "history_changed"}
    finally:
        await observer.close()
        hub.unsubscribe(handle)


async def test_global_observer_notices_parent_capability_changes():
    service = AsyncMock()
    service.list.return_value = ([], {"parent": {"create": False}})
    hub = Hub()
    handle = hub.subscribe(Topic("agents.all"))
    observer = AgentObserver(service, hub, interval=0.01)
    observer.join("agents.all")
    try:
        await asyncio.wait_for(handle.queue.get(), 1)
        service.list.return_value = ([], {"parent": {"create": True}})
        changed = await asyncio.wait_for(handle.queue.get(), 1)
        assert changed["payload"]["raw"] == {"type": "agents_changed"}
    finally:
        await observer.close()
        hub.unsubscribe(handle)
