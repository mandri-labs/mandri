import asyncio
import contextlib
import hashlib
from typing import Any

from mandri.core.clock import system_now_ms
from mandri.core.errors import MandriError
from mandri.core.hub import Hub, Topic
from mandri.core.protocol.agents import AgentView
from mandri.runtime.agents import AgentService


class AgentObserver:
    def __init__(self, agents: AgentService, hub: Hub, interval: float = 1.0) -> None:
        self._agents = agents
        self._hub = hub
        self._interval = interval
        self._viewers: dict[str, int] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def join(self, topic: str) -> None:
        self._viewers[topic] = self._viewers.get(topic, 0) + 1
        if topic not in self._tasks:
            self._tasks[topic] = asyncio.create_task(self._watch(topic))

    async def leave(self, topic: str) -> None:
        count = self._viewers.get(topic, 0)
        if count > 1:
            self._viewers[topic] = count - 1
            return
        self._viewers.pop(topic, None)
        task = self._tasks.pop(topic, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def close(self) -> None:
        for topic in list(self._tasks):
            self._viewers[topic] = 1
            await self.leave(topic)

    async def _revision(self, topic: str) -> object:
        if topic == "agents.all":
            agents, capabilities = await self._agents.list()
            return (
                tuple(AgentView.model_validate(agent).model_dump_json() for agent in agents),
                capabilities,
                await self._agents.history_store.classified(),
            )
        page = await self._agents.history(topic.removeprefix("agent."), None, 100)
        digest = hashlib.sha256()
        for entry in page.entries:
            digest.update(entry.encode("utf-8"))
            digest.update(b"\x00")
        return digest.digest(), page.next_token, page.has_more

    async def _watch(self, topic: str) -> None:
        previous: object = None
        unavailable = False
        delay = self._interval
        while True:
            try:
                revision = await self._revision(topic)
                if revision != previous or unavailable:
                    self._publish(
                        topic,
                        {"type": "agents_changed" if topic == "agents.all" else "history_changed"},
                    )
                    previous = revision
                unavailable = False
                delay = self._interval
            except (MandriError, OSError, ValueError):
                if not unavailable:
                    self._publish(topic, {"type": "history_changed", "unavailable": True})
                unavailable = True
                delay = min(max(delay * 2, self._interval), 5.0)
            await asyncio.sleep(delay)

    def _publish(self, topic: str, raw: dict[str, Any]) -> None:
        self._hub.publish(Topic(topic), {"source": "mandri", "ts": system_now_ms(), "raw": raw})
