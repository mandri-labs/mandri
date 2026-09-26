import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable

from mandri.core.ids import SessionStopCause
from mandri.runtime.errors import SessionNotRunningError
from mandri.runtime.registry import LIVE, SessionRegistry
from mandri.runtime.session_state import RuntimeStates

_ZERO_VIEWER_SETTLE_SECONDS = 2.0
_BUSY_POLL_SECONDS = 1.0


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
        self.cancel_lifetime_task(session_id)

    def viewers_zero(self, session_id: str) -> None:
        self._session_state(session_id).viewed = False
        self.arm_zero_viewer_decision(session_id)

    def arm_zero_viewer_decision(self, session_id: str) -> None:
        self.cancel_lifetime_task(session_id)
        if self._session_state(session_id).viewed:
            return
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
        await asyncio.sleep(_ZERO_VIEWER_SETTLE_SECONDS)
        while self._registry.status(session_id) == LIVE:
            if self._is_busy(session_id):
                await asyncio.sleep(_BUSY_POLL_SECONDS)
                continue
            if not await asyncio.shield(self.execute_zero_viewer_stop(session_id)):
                continue
            return

    async def execute_zero_viewer_stop(self, session_id: str) -> bool:
        if self._session_state(session_id).viewed:
            return True
        if self._registry.status(session_id) != LIVE or self._is_busy(session_id):
            return False
        with contextlib.suppress(SessionNotRunningError):
            await self._stop(session_id, SessionStopCause.VIEWER_STOP)
        return True

    async def release_idle_processes(self) -> None:
        live = set(self._registry.live_ids())
        for session_id, state in self._states.items():
            if session_id not in live:
                state.idle_since = None
        for session_id in live:
            if (
                self._is_busy(session_id)
                or self._session_state(session_id).stopping
                or self._session_state(session_id).resuming
                or self._session_state(session_id).viewed
            ):
                self._session_state(session_id).idle_since = None
                continue
            now = time.monotonic()
            state = self._session_state(session_id)
            if state.idle_since is None:
                state.idle_since = now
            since = state.idle_since
            if now - since >= self._idle_release_seconds:
                with contextlib.suppress(SessionNotRunningError):
                    await self._stop(session_id, SessionStopCause.IDLE_TIMEOUT)
