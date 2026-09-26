"""Transcript module error types."""

from mandri.core.types.sessions import SessionError


class TranscriptError(SessionError):
    """Base class for transcript reader failures."""


class PageTokenInvalidError(TranscriptError):
    """A page token is unreadable, tampered, or shaped for another store."""


class PageTokenStaleError(PageTokenInvalidError):
    """A page token references a truncated or shrunken transcript store."""


class TranscriptNotFoundError(TranscriptError):
    """No transcript store exists for the referenced session."""


class TranscriptStoreError(TranscriptError):
    """The transcript store could not be read."""


class HarnessStoreUnavailableError(TranscriptError):
    """No local harness transcript store is reachable for the session."""
