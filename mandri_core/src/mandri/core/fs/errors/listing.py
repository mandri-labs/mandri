"""Filesystem browsing errors."""

from mandri.core.fs.errors.base import FsError


class FsNotFoundError(FsError):
    """Raised when a filesystem path does not exist."""


class FsNotADirectoryError(FsError):
    """Raised when a listing target is not a directory."""


class FsReadError(FsError):
    """Raised when a directory listing fails."""
