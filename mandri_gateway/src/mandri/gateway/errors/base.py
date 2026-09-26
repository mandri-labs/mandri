"""Gateway module error root."""

from mandri.core.errors import MandriError


class GatewayError(MandriError):
    """Base class for all gateway errors."""
