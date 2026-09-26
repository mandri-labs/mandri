"""Tests for the neutral topic event bus: publish/subscribe, replay, gaps."""

import asyncio

import pytest
from mandri.core.hub import Hub, SubscriberHandle, SubscriberId, Topic


async def _drain(handle: SubscriberHandle, count: int) -> list[dict[str, object]]:
    return [await asyncio.wait_for(handle.queue.get(), timeout=1) for _ in range(count)]


async def test_publish_delivers_to_subscribers_with_monotonic_seq() -> None:
    hub = Hub()
    handle = hub.subscribe(Topic("t"))
    seq1 = hub.publish(Topic("t"), {"value": 1})
    seq2 = hub.publish(Topic("t"), {"value": 2})
    assert seq1 == 1
    assert seq2 == 2
    first = await handle.queue.get()
    second = await handle.queue.get()
    assert first == {"topic": "t", "seq": 1, "payload": {"value": 1}}
    assert second == {"topic": "t", "seq": 2, "payload": {"value": 2}}


async def test_topics_have_independent_sequences() -> None:
    hub = Hub()
    handle_a = hub.subscribe(Topic("a"))
    handle_b = hub.subscribe(Topic("b"))
    hub.publish(Topic("a"), {"x": 1})
    hub.publish(Topic("b"), {"y": 1})
    frame_a = await handle_a.queue.get()
    frame_b = await handle_b.queue.get()
    assert frame_a["seq"] == 1
    assert frame_b["seq"] == 1
    assert frame_b["topic"] == "b"


async def test_replay_after_subscribe_returns_retained_frames() -> None:
    hub = Hub()
    hub.publish(Topic("t"), {"n": 1})
    hub.publish(Topic("t"), {"n": 2})
    hub.publish(Topic("t"), {"n": 3})
    handle = hub.subscribe(Topic("t"), since=1)
    assert handle.from_seq == 2
    assert handle.replay == [
        {"topic": "t", "seq": 2, "payload": {"n": 2}},
        {"topic": "t", "seq": 3, "payload": {"n": 3}},
    ]


async def test_replay_with_retention_gap_emits_gap_frame() -> None:
    hub = Hub(ring_size=2)
    for index in range(5):
        hub.publish(Topic("t"), {"n": index})
    handle = hub.subscribe(Topic("t"), since=0)
    assert handle.replay[0]["type"] == "gap"
    assert handle.replay[0]["reason"] == "retention_exceeded"
    assert handle.replay[0]["from_seq"] == 1
    assert [frame["seq"] for frame in handle.replay[1:]] == [4, 5]


async def test_resume_beyond_head_reports_history_lost() -> None:
    hub = Hub()
    hub.publish(Topic("t"), {"n": 1})
    handle = hub.subscribe(Topic("t"), since=10)
    assert handle.replay[0]["type"] == "gap"
    assert handle.replay[0]["reason"] == "history_lost"
    assert handle.from_seq == 2


async def test_fresh_subscription_gets_no_replay() -> None:
    hub = Hub()
    handle = hub.subscribe(Topic("t"))
    assert handle.replay == []
    assert handle.from_seq == 1


async def test_unsubscribe_closes_queue_with_none_sentinel() -> None:
    hub = Hub()
    handle = hub.subscribe(Topic("t"))
    hub.unsubscribe(handle)
    assert handle.closed is True
    assert await handle.queue.get() is None
    hub.publish(Topic("t"), {"n": 1})
    assert handle.queue.empty()


async def test_collect_topic_disconnects_all_subscribers() -> None:
    hub = Hub()
    handle = hub.subscribe(Topic("t"))
    hub.collect_topic(Topic("t"))
    assert handle.closed is True
    assert await handle.queue.get() is None


async def test_slow_consumer_drops_oldest_and_inserts_gap() -> None:
    hub = Hub(queue_size=2)
    handle = hub.subscribe(Topic("t"))
    for index in range(3):
        hub.publish(Topic("t"), {"n": index})
    frames = await _drain(handle, 2)
    assert frames[0]["type"] == "gap"
    assert frames[0]["reason"] == "slow_consumer"
    assert frames[0]["from_seq"] == 1
    assert frames[0]["seq"] == 3
    assert frames[1]["seq"] == 3
    assert frames[1]["payload"] == {"n": 2}


def test_subscriber_handle_defaults_to_viewer() -> None:
    handle = SubscriberHandle(SubscriberId(1), Topic("t"), asyncio.Queue(maxsize=2))
    assert handle.internal is False


async def test_internal_subscriber_skips_heartbeat_cleanup() -> None:
    hub = Hub()
    hub.set_heartbeat_policy(idle_seconds=0.05, max_misses=3)
    internal = hub.subscribe(Topic("t"), internal=True)
    viewer = hub.subscribe(Topic("t"))
    await asyncio.sleep(0.08)
    assert internal.closed is False
    assert internal.misses == 0
    assert internal.queue.empty()
    assert viewer.misses > 0
    assert viewer.closed is False


async def test_internal_subscriber_never_receives_ping_frames() -> None:
    hub = Hub()
    hub.set_heartbeat_policy(idle_seconds=0.05, max_misses=5)
    internal = hub.subscribe(Topic("t"), internal=True)
    viewer = hub.subscribe(Topic("t"))
    frame = await asyncio.wait_for(viewer.queue.get(), timeout=1)
    assert frame == {"type": "ping"}
    assert internal.queue.empty()


async def test_pong_resets_misses() -> None:
    hub = Hub()
    handle = hub.subscribe(Topic("t"))
    handle.misses = 2
    hub.pong(handle)
    assert handle.misses == 0


def test_constructor_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        Hub(ring_size=0)
    with pytest.raises(ValueError):
        Hub(queue_size=1)


def test_heartbeat_policy_rejects_invalid_values() -> None:
    hub = Hub()
    with pytest.raises(ValueError):
        hub.set_heartbeat_policy(idle_seconds=0)
    with pytest.raises(ValueError):
        hub.set_heartbeat_policy(max_misses=0)


async def test_close_all_disconnects_subscribers() -> None:
    hub = Hub()
    handle = hub.subscribe(Topic("t"))
    await hub.close_all()
    assert handle.closed is True
