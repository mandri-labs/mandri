"""Session mutation and lifecycle errors."""

from mandri.core.types.sessions import SessionError


class SessionRenameError(SessionError):
    """Raised when a session rename fails."""


class SessionDeleteError(SessionError):
    """Raised when a session deletion fails."""


class SessionNotFoundError(SessionError):
    """Raised when the target session does not exist."""


class SessionRunningError(SessionError):
    """Raised when a mutation is rejected because the session is live."""


class SessionConflictError(SessionError):
    """Raised on concurrent conflicting session operations."""
