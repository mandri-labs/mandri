import asyncio
import contextlib
import logging
from collections.abc import Sequence
from typing import Any

from mandri.core.hub import Hub, Topic
from mandri.core.ports.conversation_status import ConversationStatusRepositoryPort
from mandri.core.types.conversation_status import ConversationStatus, WorkObservation

logger = logging.getLogger(__name__)


class ConversationStatuses:
    def __init__(
        self, repository: ConversationStatusRepositoryPort, hub: Hub | None = None
    ) -> None:
        self.repository = repository
        self._hub = hub
        self._summaries: dict[str, ConversationStatus] = {}
        self._aliases: dict[str, str] = {}
        self._queue: asyncio.Queue[tuple[str, WorkObservation]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._summaries = {row.target: row for row in await self.repository.all()}
        for row in list(self._summaries.values()):
            if row.work_state in {"working", "waiting"}:
                await self.observe(row.target, [WorkObservation(state="unknown")])
        self._task = asyncio.create_task(self._consume(), name="conversation-status")

    def target(self, key: str) -> str:
        return self._aliases.get(key, key)

    def all(self) -> list[ConversationStatus]:
        return list(self._summaries.values())

    def get(self, key: str) -> ConversationStatus:
        target = self.target(key)
        return self._summaries.get(target, ConversationStatus(target=target))

    def enqueue(self, target: str, observation: WorkObservation) -> None:
        self._queue.put_nowait((target, observation))

    async def flush(self) -> None:
        if self._task is not None:
            if self._task.done():
                self._task.result()
            joining = asyncio.create_task(self._queue.join())
            done, _ = await asyncio.wait((joining, self._task), return_when=asyncio.FIRST_COMPLETED)
            if self._task in done:
                joining.cancel()
                await asyncio.gather(joining, return_exceptions=True)
                self._task.result()
            else:
                await joining

    async def close(self) -> None:
        await self.flush()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def observe(
        self,
        target: str,
        observations: Sequence[WorkObservation],
        source: str | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> ConversationStatus:
        return self._publish(
            await self.repository.observe(self.target(target), observations, source, checkpoint)
        )

    async def acknowledge(
        self, target: str, through_revision: int, completion_key: str | None = None
    ) -> ConversationStatus:
        await self.flush()
        return self._publish(
            await self.repository.acknowledge(self.target(target), through_revision, completion_key)
        )

    async def link(self, source: str, target: str) -> None:
        await self.flush()
        if self.target(source) == target:
            return
        if source in self._summaries:
            self._publish(await self.repository.link(source, target))
            self._summaries.pop(source, None)
        self._aliases[source] = target

    def _publish(self, summary: ConversationStatus) -> ConversationStatus:
        previous = self._summaries.get(summary.target)
        if previous is not None and previous.revision > summary.revision:
            return previous
        self._summaries[summary.target] = summary
        if self._hub is not None and summary != previous:
            self._hub.publish(
                Topic("conversations.all"),
                {
                    "type": "conversation_status",
                    "status": summary.model_dump(),
                },
            )
        return summary

    async def _consume(self) -> None:
        while True:
            target, first = await self._queue.get()
            batch = [(target, first)]
            while not self._queue.empty():
                batch.append(self._queue.get_nowait())
            grouped: dict[str, list[WorkObservation]] = {}
            for key, observation in batch:
                grouped.setdefault(key, []).append(observation)
            try:
                for key, observations in grouped.items():
                    work: list[WorkObservation] = []
                    frame_key = observations[0].key
                    final_state = None
                    for item in observations:
                        if item.key != frame_key:
                            work.append(WorkObservation(state=final_state))
                            frame_key = item.key
                        final_state = item.state
                        work.append(
                            WorkObservation(
                                source=item.source,
                                content_key=item.content_key,
                                progress=item.progress,
                                outcome=item.outcome,
                                key=item.key,
                            )
                        )
                    work.append(WorkObservation(state=final_state))
                    await self.observe(key, work)
            except Exception:
                logger.exception("Conversation status update failed")
                raise
            finally:
                for _ in batch:
                    self._queue.task_done()
