"""Harness-level runtime errors."""

from mandri.runtime.errors.base import RuntimeDomainError


class HarnessNotInstalledError(RuntimeDomainError):
    """The requested harness has no command configured or is not installed."""


class OpencodeSessionMissingError(RuntimeDomainError):
    """The opencode-native conversation for a session id no longer exists."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"opencode session {session_id} is gone and cannot be resumed")
