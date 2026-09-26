"""Session sync errors."""

from mandri.core.types.sessions import SessionError


class StorageReadError(SessionError):
    """Raised when reading from an underlying storage fails."""


class SyncError(StorageReadError):
    """Raised when a harness sync round fails."""
