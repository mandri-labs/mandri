"""Shared passive transcript observers for websocket viewers."""

import asyncio
import contextlib
from collections.abc import Callable

from mandri.core.clock import system_now_ms
from mandri.core.hub import Hub, Topic
from mandri.core.ids import SessionId
from mandri.core.types.sessions import SessionError
from mandri.sessions.service import SessionsService


class TranscriptObserver:
    def __init__(
        self,
        sessions: SessionsService,
        hub: Hub,
        managed: Callable[[str], bool],
        interval: float = 1.0,
    ) -> None:
        self._sessions = sessions
        self._hub = hub
        self._managed = managed
        self._interval = interval
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def join(self, session_id: str) -> None:
        if session_id not in self._tasks:
            self._tasks[session_id] = asyncio.create_task(self._watch(session_id))

    async def leave(self, session_id: str) -> None:
        task = self._tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _watch(self, session_id: str) -> None:
        previous: object = None
        unavailable = False
        delay = self._interval
        while True:
            try:
                if self._managed(session_id):
                    previous = None
                else:
                    revision = await self._sessions.transcript_revision(SessionId(session_id))
                    if revision != previous:
                        busy, model = await self._sessions.external_status(SessionId(session_id))
                        if self._managed(session_id):
                            previous = None
                            continue
                        self._hub.publish(
                            Topic(f"session.{session_id}"),
                            {
                                "source": "mandri",
                                "ts": system_now_ms(),
                                "raw": {
                                    "type": "history_changed",
                                    "ownership": "external",
                                    "external_busy": busy,
                                    "external_model": model,
                                },
                            },
                        )
                        previous = revision
                delay = self._interval
                unavailable = False
            except (SessionError, OSError, ValueError, TypeError):
                if not unavailable:
                    self._hub.publish(
                        Topic(f"session.{session_id}"),
                        {
                            "source": "mandri",
                            "ts": system_now_ms(),
                            "raw": {
                                "type": "history_changed",
                                "unavailable": True,
                            },
                        },
                    )
                unavailable = True
                previous = None
                delay = min(max(delay * 2, self._interval), 5.0)
            await asyncio.sleep(delay)
