"""Per-session stdout/stderr pumps feeding harness event pipes on the hub."""

import asyncio
import contextlib
import enum
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, final

from mandri.core.clock import system_now_ms
from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind
from mandri.runtime.errors import ProcessIOError
from mandri.runtime.pump import (
    LineEvent,
    LineEventKind,
    LinePump,
    ProcessStreams,
    pump_process_streams,
)
from mandri.runtime.translators.agy import AgyEventPipe
from mandri.runtime.translators.base import EventPipe, EventPublisher
from mandri.runtime.translators.claude import ClaudeEventPipe
from mandri.runtime.translators.codex import CodexEventPipe
from mandri.runtime.translators.opencode import OpencodeEventPipe
from mandri.runtime.translators.pi import PiEventPipe

logger = logging.getLogger(__name__)

_JSON_LINE_KINDS = frozenset(
    {HarnessKind.CLAUDE, HarnessKind.CODEX, HarnessKind.AGY, HarnessKind.PI}
)

DEGRADED_SOURCE = "mandri"


class DegradationKind(enum.StrEnum):
    OVERSIZE = "oversize"
    INCOMPLETE = "incomplete"
    DECODE_ERROR = "decode_error"
    PARSE_ERROR = "parse_error"


def session_topic(session_id: str) -> Topic:
    return Topic(f"session.{session_id}")


@final
class SessionFeed:
    """Pumps one child process's streams onto its session hub topic."""

    def __init__(
        self,
        hub: Hub,
        session_id: str,
        kind: HarnessKind | None,
        streams: ProcessStreams,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._session_id = session_id
        self._hub = hub
        stdout, stderr = pump_process_streams(streams)
        pipe: EventPipe | None = None
        if kind is not None:
            source_iter = self._stdout_events(stdout, kind)
            pipe = _event_pipe(kind, hub, session_topic(session_id), source_iter, publisher)
        self._pipe = pipe
        self._stderr = stderr
        self._tasks: tuple[asyncio.Task[None], ...] = ()

    @property
    def tasks(self) -> tuple[asyncio.Task[None], ...]:
        return self._tasks

    def start(self) -> None:
        tasks: list[asyncio.Task[None]] = []
        if self._pipe is not None:
            tasks.append(
                asyncio.create_task(self._consume_stdout(), name=f"session-feed:{self._session_id}")
            )
        tasks.append(
            asyncio.create_task(self._consume_stderr(), name=f"session-stderr:{self._session_id}")
        )
        self._tasks = tuple(tasks)

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _consume_stdout(self) -> None:
        if self._pipe is None:
            return
        try:
            await self._pipe.run()
        except ProcessIOError as error:
            logger.warning("session %s stdout unreadable: %s", self._session_id, error)

    async def _consume_stderr(self) -> None:
        try:
            async for event in self._stderr.lines():
                logger.warning("session %s stderr: %s", self._session_id, event.text)
        except ProcessIOError as error:
            logger.warning("session %s stderr unreadable: %s", self._session_id, error)

    async def _stdout_events(self, pump: LinePump, kind: HarnessKind) -> AsyncIterator[Any]:
        async for event in pump.lines():
            match event.kind:
                case LineEventKind.LINE:
                    value = self._event_from_line(event, kind)
                    if value is not None:
                        yield value
                case LineEventKind.INCOMPLETE:
                    self._publish_degradation(DegradationKind.INCOMPLETE, event.size)
                    value = self._event_from_line(event, kind)
                    if value is not None:
                        yield value
                case LineEventKind.OVERSIZE:
                    logger.warning(
                        "session %s dropped oversize stdout line of %d bytes",
                        self._session_id,
                        event.size,
                    )
                    self._publish_degradation(DegradationKind.OVERSIZE, event.size)
                case LineEventKind.DECODE_ERROR:
                    logger.warning(
                        "session %s dropped undecodable stdout line: %s",
                        self._session_id,
                        event.text,
                    )
                    self._publish_degradation(DegradationKind.DECODE_ERROR, event.size)

    def _event_from_line(self, event: LineEvent, kind: HarnessKind) -> Any | None:
        if kind not in _JSON_LINE_KINDS:
            return event.text
        try:
            return json.loads(event.text)
        except json.JSONDecodeError:
            logger.debug("session %s ignored non-JSON stdout line", self._session_id)
            self._publish_degradation(DegradationKind.PARSE_ERROR, event.size)
            return None

    def _publish_degradation(self, degradation: DegradationKind, size: int) -> None:
        self._hub.publish(
            session_topic(self._session_id),
            {
                "source": DEGRADED_SOURCE,
                "raw": {"error": degradation.value, "size": size},
                "ts": system_now_ms(),
            },
        )


def _event_pipe(
    kind: HarnessKind,
    hub: Hub,
    topic: Topic,
    source_iter: AsyncIterator[Any],
    publisher: EventPublisher | None = None,
) -> EventPipe:
    if kind is HarnessKind.PI:
        return PiEventPipe(hub, topic, source_iter, publisher=publisher)
    if kind is HarnessKind.AGY:
        return AgyEventPipe(hub, topic, source_iter, publisher=publisher)
    if kind == HarnessKind.CLAUDE:
        return ClaudeEventPipe(hub, topic, source_iter, publisher=publisher)
    if kind == HarnessKind.CODEX:
        return CodexEventPipe(hub, topic, source_iter, publisher=publisher)
    return OpencodeEventPipe(hub, topic, source_iter, publisher=publisher)
