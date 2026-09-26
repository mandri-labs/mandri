"""Runtime module error root."""

from mandri.core.errors import MandriError


class RuntimeDomainError(MandriError):
    """Base class for all runtime subprocess management errors."""
