"""Liveness session-lifecycle errors."""

from mandri.core.ids import SessionId
from mandri.runtime.liveness.types import LivenessError


class UnknownSessionError(LivenessError):
    """Raised when liveness is queried or fed evidence for an unregistered session."""

    def __init__(self, session_id: SessionId) -> None:
        super().__init__(f"no registered liveness session {session_id}")
