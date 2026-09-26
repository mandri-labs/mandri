"""Filesystem module error root."""

from mandri.core.errors import MandriError


class FsError(MandriError):
    """Base class for all filesystem browsing errors."""
