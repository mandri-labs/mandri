"""Session retrieval and fetch errors."""

from mandri.core.types.sessions import SessionError


class SessionFetchError(SessionError):
    """Base error for session retrieval failures."""


class AgentBinaryNotFoundError(SessionFetchError):
    """Raised when the agent CLI binary is not installed or found in PATH."""


class ServerTimeoutError(SessionFetchError):
    """Raised when the agent server fails to respond within the configured timeout."""


class ServerCommunicationError(SessionFetchError):
    """Raised on HTTP, network, or process communication failures."""


class SessionParseError(SessionFetchError):
    """Raised when session data returned by the agent cannot be parsed."""


class DatabaseAccessError(SessionFetchError):
    """Raised when the session storage database cannot be accessed."""


class SchemaDriftError(SessionFetchError):
    """Raised when harness storage schema changed and a fallback is needed."""
