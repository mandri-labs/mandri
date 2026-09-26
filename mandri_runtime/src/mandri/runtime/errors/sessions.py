"""Session-level runtime errors."""

from mandri.runtime.errors.base import RuntimeDomainError


class SessionNotRunningError(RuntimeDomainError):
    """No live session exists under the requested id."""


class SessionNotResumableError(RuntimeDomainError):
    """The session has no harness-native conversation to resume."""
