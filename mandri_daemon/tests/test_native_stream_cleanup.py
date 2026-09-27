import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind
from mandri.daemon.serve import build_harness_adapters
from mandri.runtime.adapters import AdapterContext


@pytest.mark.parametrize(
    "kind", [HarnessKind.CODEX, HarnessKind.PI, HarnessKind.CLAUDE, HarnessKind.AGY]
)
@pytest.mark.parametrize("start_reader", [False, True])
async def test_native_control_closes_eager_subscription_even_without_reader(kind, start_reader):
    hub = Hub()
    topic = Topic("session.test")
    adapters = build_harness_adapters(
        AdapterContext(
            kind=kind,
            process=Mock(write_stdin=AsyncMock()),
            hub=hub,
            topic=topic,
            agy_bridge=Mock(aclose=AsyncMock()),
        )
    )
    assert adapters is not None
    handle = next(iter(hub._topics[topic].subscribers.values()))
    task = None
    if start_reader:
        task = asyncio.create_task(adapters.control.capture_identity())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    try:
        await adapters.control.aclose()
        assert handle.closed
        assert hub._topics[topic].subscribers == {}
        queued = handle.queue.qsize()
        hub.publish(topic, {"source": kind.value, "raw": {"id": "later-process"}})
        assert handle.queue.qsize() == queued
        await adapters.control.aclose()
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await hub.close_all()
