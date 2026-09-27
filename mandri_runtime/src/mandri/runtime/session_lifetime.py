import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable

from mandri.core.ids import SessionStopCause
from mandri.runtime.errors import SessionNotRunningError
from mandri.runtime.registry import LIVE, SessionRegistry
from mandri.runtime.session_state import RuntimeStates

_BUSY_POLL_SECONDS = 1.0
logger = logging.getLogger(__name__)


class SessionLifetime:
    def __init__(
        self,
        states: RuntimeStates,
        registry: SessionRegistry,
        is_busy: Callable[[str], bool],
        stop: Callable[[str, SessionStopCause], Awaitable[int]],
        idle_release_seconds: float,
    ) -> None:
        self._states = states
        self._session_state = states.session
        self._registry = registry
        self._is_busy = is_busy
        self._stop = stop
        self._idle_release_seconds = idle_release_seconds

    def viewer_joined(self, session_id: str) -> None:
        self._session_state(session_id).viewed = True
        self._session_state(session_id).idle_since = None
        self.cancel_lifetime_task(session_id)

    def viewers_zero(self, session_id: str) -> None:
        self._session_state(session_id).viewed = False
        self.arm_zero_viewer_decision(session_id)

    def arm_zero_viewer_decision(self, session_id: str) -> None:
        self.cancel_lifetime_task(session_id)
        if self._session_state(session_id).viewed:
            return
        self._session_state(session_id).idle_since = time.monotonic()
        task = asyncio.create_task(
            self._stop_when_zero_viewers_and_idle(session_id),
            name=f"lifetime:{session_id}",
        )
        self._session_state(session_id).lifetime_task = task
        task.add_done_callback(lambda done: self._forget_lifetime_task(session_id, done))

    def cancel_lifetime_task(self, session_id: str) -> None:
        task = self._session_state(session_id).lifetime_task
        self._session_state(session_id).lifetime_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    def _forget_lifetime_task(self, session_id: str, done: asyncio.Task[None]) -> None:
        if self._session_state(session_id).lifetime_task is done:
            self._session_state(session_id).lifetime_task = None

    async def _stop_when_zero_viewers_and_idle(self, session_id: str) -> None:
        while self._registry.status(session_id) == LIVE:
            await asyncio.sleep(min(_BUSY_POLL_SECONDS, self._idle_release_seconds))
            if not self._idle_release_due(session_id):
                continue
            if not await asyncio.shield(self.execute_zero_viewer_stop(session_id)):
                continue
            return

    async def execute_zero_viewer_stop(self, session_id: str) -> bool:
        if self._session_state(session_id).viewed:
            return True
        if self._registry.status(session_id) != LIVE or self.release_blocked(session_id):
            return False
        logger.info("session %s releasing idle process without viewers", session_id)
        with contextlib.suppress(SessionNotRunningError):
            await self._stop(session_id, SessionStopCause.IDLE_TIMEOUT)
        return True

    def release_blocked(self, session_id: str) -> bool:
        state = self._session_state(session_id)
        return self._is_busy(session_id) or state.resuming or state.viewed

    def _idle_release_due(self, session_id: str) -> bool:
        state = self._session_state(session_id)
        if state.stopping or self.release_blocked(session_id):
            state.idle_since = None
            return False
        now = time.monotonic()
        if state.idle_since is None:
            state.idle_since = now
        return now - state.idle_since >= self._idle_release_seconds

    async def release_idle_processes(self) -> None:
        live = set(self._registry.live_ids())
        for session_id, state in self._states.items():
            if session_id not in live:
                state.idle_since = None
        for session_id in live:
            if self._idle_release_due(session_id):
                await self.execute_zero_viewer_stop(session_id)
