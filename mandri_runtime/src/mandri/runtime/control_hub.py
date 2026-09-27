"""Hub session frames relayed as newline-delimited harness lines for control adapters."""

import json
from collections.abc import AsyncIterator
from typing import Any

from mandri.core.hub import Hub, SubscriberHandle, Topic


class HubEventLines:
    """Streams harness-sourced frames of one session topic as JSON byte lines."""

    def __init__(self, hub: Hub, topic: Topic, source: str, *, since: int = 0) -> None:
        self._hub = hub
        self._topic = topic
        self._source = source
        self._handle: SubscriberHandle | None = hub.subscribe(topic, since=since, internal=True)
        self._last_seq = since
        self._closed = False

    async def chunks(self) -> AsyncIterator[bytes]:
        if self._closed:
            return
        handle = self._handle
        if handle is None:
            handle = self._hub.subscribe(self._topic, since=self._last_seq, internal=True)
        self._handle = None
        try:
            for frame in handle.replay:
                if _is_gap(frame):
                    self._skip_lost(frame)
                    continue
                self._last_seq = _frame_seq(frame)
                chunk = _harness_line(frame, self._source)
                if chunk is not None:
                    yield chunk
            while True:
                delivered: dict[str, Any] | None = await handle.queue.get()
                if delivered is None:
                    return
                if _is_gap(delivered):
                    if delivered.get("reason") != "slow_consumer":
                        self._skip_lost(delivered)
                        continue
                    return
                self._last_seq = _frame_seq(delivered)
                chunk = _harness_line(delivered, self._source)
                if chunk is not None:
                    yield chunk
        finally:
            self._hub.unsubscribe(handle)

    def _skip_lost(self, frame: dict[str, Any]) -> None:
        end = _frame_seq(frame)
        if end - 1 > self._last_seq:
            self._last_seq = end - 1

    def close(self) -> None:
        self._closed = True
        handle = self._handle
        self._handle = None
        if handle is not None:
            self._hub.unsubscribe(handle)


def _is_gap(frame: dict[str, Any]) -> bool:
    return frame.get("type") == "gap"


def _frame_seq(frame: dict[str, Any]) -> int:
    seq = frame.get("seq")
    return seq if isinstance(seq, int) else 0


def _harness_line(frame: dict[str, Any], source: str) -> bytes | None:
    payload = frame.get("payload")
    if not isinstance(payload, dict) or "type" in payload:
        return None
    if payload.get("source") != source:
        return None
    raw = payload.get("raw")
    if not isinstance(raw, dict):
        return None
    return (json.dumps(raw) + "\n").encode("utf-8")
