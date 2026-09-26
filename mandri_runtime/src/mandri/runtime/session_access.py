from collections.abc import Awaitable, Callable

from mandri.core.ids import HarnessKind, SessionId
from mandri.core.types.availability import SessionActivityState, SessionAvailability, SessionOwner
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.errors import SessionNotResumableError
from mandri.runtime.registry import LIVE, SessionRegistry
from mandri.sessions.errors import SessionConflictError
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts import TranscriptError


class SessionAccess:
    def __init__(
        self,
        sessions: SessionsService | None,
        registry: SessionRegistry,
        is_busy: Callable[[str], bool],
    ) -> None:
        self._sessions = sessions
        self._registry = registry
        self._is_busy = is_busy

    async def availability(self, session_id: str) -> SessionAvailability:
        if self._sessions is None:
            raise SessionNotResumableError("No session store is configured")
        record = await self._sessions.get_session(SessionId(session_id))
        worktree = getattr(record, "worktree", None)
        if worktree is not None and worktree.state == "closed":
            return SessionAvailability(
                SessionOwner.UNOWNED, SessionActivityState.IDLE, reason="worktree_closed"
            )
        if self._registry.status(session_id) == LIVE:
            activity = (
                SessionActivityState.BUSY
                if self._is_busy(session_id)
                else SessionActivityState.IDLE
            )
            return SessionAvailability(SessionOwner.MANDRI, activity, can_release=True)
        owner = await self._sessions.native_ownership(SessionId(session_id))
        if owner.owner is SessionOwner.UNOWNED:
            activity = SessionActivityState.IDLE
        else:
            try:
                busy, _ = await self._sessions.external_status(SessionId(session_id))
            except TranscriptError:
                busy = None
            activity = (
                SessionActivityState.UNKNOWN
                if busy is None
                else SessionActivityState.BUSY
                if busy
                else SessionActivityState.IDLE
            )
        free = owner.owner is SessionOwner.UNOWNED and record.native_id is not None
        resumable = free and (
            record.model_source is ModelSource.NATIVE
            or record.gateway_route_id is not None
            or record.model is not None
        )
        return SessionAvailability(
            owner.owner,
            activity,
            can_resume=resumable,
            can_release=owner.can_release,
            can_restore=free
            and record.harness
            in (HarnessKind.CODEX, HarnessKind.CLAUDE, HarnessKind.AGY, HarnessKind.PI),
            reason=owner.reason,
        )

    async def release(
        self, session_id: str, confirmed: bool, stop: Callable[[str], Awaitable[int]]
    ) -> SessionAvailability:
        if not confirmed:
            raise SessionConflictError("Explicit confirmation is required to release a writer")
        availability = await self.availability(session_id)
        if availability.owner is SessionOwner.UNOWNED:
            return availability
        if not availability.can_release or self._sessions is None:
            raise SessionConflictError("This native writer cannot be released safely")
        if availability.owner is SessionOwner.MANDRI:
            await stop(session_id)
        else:
            await self._sessions.release_native_writer(SessionId(session_id))
        return await self.availability(session_id)
