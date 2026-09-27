import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.hub import Hub
from mandri.core.ids import HarnessKind
from mandri.runtime.service import RuntimeService
from mandri.runtime.session_feed import SessionFeed, session_topic


@pytest.mark.parametrize("persist", [False, True])
async def test_reconciliation_delivers_buffered_result_before_stopped_event(persist):
    hub = Hub()
    runtime = RuntimeService({}, hub=hub)
    runtime.registry.mark_live(
        "session", SimpleNamespace(returncode=0, stop=AsyncMock(return_value=0)), "agy"
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def publish(topic, payload):
        entered.set()
        await release.wait()
        hub.publish(topic, payload)

    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    stdout.feed_data(b'{"event":"result","result":{"status":"ERROR"}}\n')
    stdout.feed_eof()
    stderr.feed_eof()
    handle = hub.subscribe(session_topic("session"))
    feed = SessionFeed(
        hub,
        "session",
        HarnessKind.AGY,
        SimpleNamespace(stdout=stdout, stderr=stderr),
        publisher=publish,
    )
    runtime._session_state("session").feed = feed
    feed.start()
    await entered.wait()
    reconcile = runtime.reconcile_and_persist if persist else runtime.reconcile
    task = asyncio.create_task(reconcile())
    try:
        await asyncio.sleep(0)
        release.set()
        assert await task == ["session"]
        frames = []
        while not handle.queue.empty():
            frames.append(handle.queue.get_nowait())
        assert frames[0] is not None, frames
        assert frames[0]["payload"]["raw"]["result"]["status"] == "ERROR"
        assert frames[-1]["payload"]["type"] == "session_stopped"
        assert frames[-1]["seq"] == frames[0]["seq"] + 1
        assert not handle.closed
    finally:
        release.set()
        await feed.stop()
        hub.unsubscribe(handle)


async def test_reconciliation_bounds_drain_when_descendant_keeps_pipes_open(monkeypatch):
    monkeypatch.setattr("mandri.runtime.runtime_events._FEED_DRAIN_TIMEOUT_SECONDS", 0.01)
    hub = Hub()
    runtime = RuntimeService({}, hub=hub)
    runtime.registry.mark_live(
        "session", SimpleNamespace(returncode=0, stop=AsyncMock(return_value=0)), "agy"
    )
    feed = SessionFeed(
        hub,
        "session",
        HarnessKind.AGY,
        SimpleNamespace(stdout=asyncio.StreamReader(), stderr=asyncio.StreamReader()),
    )
    runtime._session_state("session").feed = feed
    feed.start()
    try:
        assert await asyncio.wait_for(runtime.reconcile(), timeout=1) == ["session"]
        assert all(task.done() for task in feed.tasks)
        assert runtime._session_state("session").feed is None
    finally:
        await feed.stop()
