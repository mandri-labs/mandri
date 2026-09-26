"""Common pump that forwards raw harness events from a source onto one hub topic."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from mandri.core.clock import system_now_ms
from mandri.core.hub import Hub, Topic

NowMs = Callable[[], int]
EventPublisher = Callable[[Topic, dict[str, Any]], Awaitable[None]]


class EventPipe:
    """Reads events from an injected iterator and publishes raw frames to the hub."""

    source: str = ""

    def __init__(
        self,
        hub: Hub,
        topic: Topic,
        source_iter: AsyncIterator[Any],
        clock: NowMs = system_now_ms,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._hub = hub
        self._topic = topic
        self._source_iter = source_iter
        self._clock = clock
        self._publisher = publisher
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def run(self) -> None:
        async for event in self._source_iter:
            if self.closed or event is None:
                return
            payload = self._wrap(event)
            if payload is not None:
                if self._publisher is None:
                    self._hub.publish(self._topic, payload)
                else:
                    await self._publisher(self._topic, payload)

    def _frame(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {"source": self.source, "raw": raw, "ts": self._clock()}

    def _wrap(self, event: Any) -> dict[str, Any] | None:
        raise NotImplementedError
