import asyncio

import pytest
from mandri.core.hub import Hub, Topic


async def test_internal_subscriber_keeps_rpc_response_across_event_burst():
    hub = Hub(queue_size=2)
    topic = Topic("session.native")
    control = hub.subscribe(topic, internal=True)
    viewer = hub.subscribe(topic)
    response = {"source": "codex", "raw": {"id": 1, "result": {"thread": {"id": "native"}}}}
    hub.publish(topic, response)
    for index in range(1024):
        hub.publish(topic, {"source": "codex", "raw": {"method": "item/delta", "index": index}})
    hub.publish(topic, {"source": "codex", "raw": {"method": "turn/completed"}})

    assert control.queue.qsize() == 1026
    frames = [await asyncio.wait_for(control.queue.get(), 1) for _ in range(1026)]
    assert frames[0]["payload"] == response
    assert [frame["seq"] for frame in frames] == list(range(1, 1027))
    assert frames[-1]["payload"]["raw"]["method"] == "turn/completed"
    assert viewer.queue.qsize() == 2
    gap = await viewer.queue.get()
    assert gap["reason"] == "slow_consumer"
    assert (gap["from_seq"], gap["seq"]) == (1, 1026)
    hub.unsubscribe(control)
    assert await control.queue.get() is None


@pytest.mark.parametrize("count", [3, 4, 5, 10])
async def test_repeated_viewer_overflow_preserves_the_entire_missing_range(count):
    hub = Hub(queue_size=2)
    topic = Topic("session.viewer")
    viewer = hub.subscribe(topic)
    for index in range(count):
        hub.publish(topic, {"index": index})
    gap, latest = await viewer.queue.get(), await viewer.queue.get()
    assert gap["type"] == "gap"
    assert (gap["from_seq"], gap["seq"]) == (1, count)
    assert latest["seq"] == count
    assert latest["payload"] == {"index": count - 1}
