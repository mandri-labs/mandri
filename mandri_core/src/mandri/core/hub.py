"""Neutral topic event bus with per-topic replay rings, gap frames, and heartbeats."""

import asyncio
import contextlib
import time
import typing
from collections import deque
from typing import Any

Topic = typing.NewType("Topic", str)
SubscriberId = typing.NewType("SubscriberId", int)

Frame = dict[str, Any]

_DEFAULT_RING_SIZE = 1024
_DEFAULT_QUEUE_SIZE = 512
_DEFAULT_IDLE_SECONDS = 15.0
_DEFAULT_MAX_MISSES = 2


class SubscriberHandle:
    """Delivery handle for one topic subscription."""

    def __init__(
        self,
        subscriber_id: SubscriberId,
        topic: Topic,
        queue: asyncio.Queue[Frame | None],
        *,
        internal: bool = False,
    ) -> None:
        self.id = subscriber_id
        self.topic = topic
        self.queue = queue
        self.internal = internal
        self.replay: list[Frame] = []
        self.from_seq = 0
        self.misses = 0
        self.closed = False
        self.pending_gap: tuple[int, int] | None = None


class _TopicState:
    def __init__(self, ring_size: int) -> None:
        self.seq = 0
        self.ring: deque[tuple[int, dict[str, Any]]] = deque(maxlen=ring_size)
        self.subscribers: dict[SubscriberId, SubscriberHandle] = {}
        self.last_activity = time.monotonic()


class Hub:
    """Per-topic sequence counter, retention ring, and subscriber queues."""

    def __init__(
        self,
        ring_size: int = _DEFAULT_RING_SIZE,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
    ) -> None:
        if ring_size < 1:
            raise ValueError("ring_size must be at least 1")
        if queue_size < 2:
            raise ValueError("queue_size must be at least 2")
        self._ring_size = ring_size
        self._queue_size = queue_size
        self._topics: dict[Topic, _TopicState] = {}
        self._next_subscriber_id = 0
        self._idle_seconds = _DEFAULT_IDLE_SECONDS
        self._max_misses = _DEFAULT_MAX_MISSES
        self._heartbeat_task: asyncio.Task[None] | None = None

    def publish(self, topic: Topic, payload: dict[str, Any]) -> int:
        state = self._topic_state(topic)
        state.seq += 1
        seq = state.seq
        state.ring.append((seq, payload))
        state.last_activity = time.monotonic()
        frame: Frame = {"topic": str(topic), "seq": seq, "payload": payload}
        for handle in list(state.subscribers.values()):
            self._offer(handle, frame)
        return seq

    def subscribe(
        self, topic: Topic, since: int | None = None, *, internal: bool = False
    ) -> SubscriberHandle:
        state = self._topic_state(topic)
        self._next_subscriber_id += 1
        handle = SubscriberHandle(
            SubscriberId(self._next_subscriber_id),
            topic,
            asyncio.Queue(maxsize=0 if internal else self._queue_size),
            internal=internal,
        )
        state.subscribers[handle.id] = handle
        self._prepare_resume(state, handle, since)
        return handle

    def unsubscribe(self, handle: SubscriberHandle) -> None:
        state = self._topics.get(handle.topic)
        if state is not None:
            state.subscribers.pop(handle.id, None)
        self._disconnect(handle)

    def sequence(self, topic: Topic) -> int:
        state = self._topics.get(topic)
        return state.seq if state is not None else 0

    def collect_topic(self, topic: Topic) -> None:
        state = self._topics.pop(topic, None)
        if state is None:
            return
        for handle in list(state.subscribers.values()):
            self._disconnect(handle)

    def set_heartbeat_policy(
        self,
        idle_seconds: float = _DEFAULT_IDLE_SECONDS,
        max_misses: int = _DEFAULT_MAX_MISSES,
    ) -> None:
        if idle_seconds <= 0:
            raise ValueError("idle_seconds must be positive")
        if max_misses < 1:
            raise ValueError("max_misses must be at least 1")
        self._idle_seconds = idle_seconds
        self._max_misses = max_misses
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def pong(self, handle: SubscriberHandle) -> None:
        handle.misses = 0

    async def close_all(self) -> None:
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for state in self._topics.values():
            for handle in list(state.subscribers.values()):
                self._disconnect(handle)

    def _topic_state(self, topic: Topic) -> _TopicState:
        state = self._topics.get(topic)
        if state is None:
            state = _TopicState(self._ring_size)
            self._topics[topic] = state
        return state

    def _prepare_resume(
        self, state: _TopicState, handle: SubscriberHandle, since: int | None
    ) -> None:
        if since is None:
            handle.from_seq = state.seq + 1
            return
        retained = [(seq, payload) for seq, payload in state.ring if seq > since]
        frames: list[Frame] = []
        if retained:
            oldest = retained[0][0]
            if since + 1 < oldest:
                frames.append(
                    self._gap_frame(handle.topic, since + 1, oldest, "retention_exceeded")
                )
                handle.from_seq = since + 1
            else:
                handle.from_seq = oldest
            frames.extend(
                self._event_frame(handle.topic, seq, payload) for seq, payload in retained
            )
        elif since < state.seq:
            frames.append(
                self._gap_frame(handle.topic, since + 1, state.seq + 1, "retention_exceeded")
            )
            handle.from_seq = since + 1
        elif since > state.seq:
            frames.append(self._gap_frame(handle.topic, since + 1, state.seq + 1, "history_lost"))
            handle.from_seq = state.seq + 1
        else:
            handle.from_seq = state.seq + 1
        handle.replay = frames

    def _offer(self, handle: SubscriberHandle, frame: Frame) -> None:
        if handle.closed:
            return
        if handle.internal:
            handle.queue.put_nowait(frame)
            return
        slots = 2 if handle.pending_gap is not None else 1
        while handle.queue.qsize() + slots > handle.queue.maxsize:
            if handle.queue.empty():
                break
            self._drop_oldest(handle)
            slots = 2 if handle.pending_gap is not None else 1
        if handle.pending_gap is not None:
            gap = self._gap_frame(
                handle.topic, handle.pending_gap[0], handle.pending_gap[1], "slow_consumer"
            )
            handle.pending_gap = None
            handle.queue.put_nowait(gap)
        handle.queue.put_nowait(frame)

    def _drop_oldest(self, handle: SubscriberHandle) -> None:
        dropped = handle.queue.get_nowait()
        if dropped is None:
            return
        if "payload" in dropped:
            missing = (dropped["seq"], dropped["seq"] + 1)
        elif dropped.get("type") == "gap":
            missing = (dropped["from_seq"], dropped["seq"])
        else:
            return
        if handle.pending_gap is None:
            handle.pending_gap = missing
        else:
            start, end = handle.pending_gap
            handle.pending_gap = (min(start, missing[0]), max(end, missing[1]))

    def _disconnect(self, handle: SubscriberHandle) -> None:
        if handle.closed:
            return
        handle.closed = True
        while True:
            try:
                handle.queue.put_nowait(None)
                return
            except asyncio.QueueFull:
                handle.queue.get_nowait()

    def _event_frame(self, topic: Topic, seq: int, payload: dict[str, Any]) -> Frame:
        return {"topic": str(topic), "seq": seq, "payload": payload}

    def _gap_frame(self, topic: Topic, from_seq: int, seq: int, reason: str) -> Frame:
        return {
            "topic": str(topic),
            "type": "gap",
            "from_seq": from_seq,
            "seq": seq,
            "reason": reason,
        }

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._idle_seconds)
            now = time.monotonic()
            for state in self._topics.values():
                if now - state.last_activity < self._idle_seconds:
                    continue
                for handle in list(state.subscribers.values()):
                    if handle.internal:
                        continue
                    handle.misses += 1
                    if handle.misses > self._max_misses:
                        self._disconnect(handle)
                    else:
                        self._offer(handle, {"type": "ping"})
