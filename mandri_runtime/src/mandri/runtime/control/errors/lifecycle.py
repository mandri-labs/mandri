"""Control transport and mode errors."""

from mandri.runtime.control.errors.base import ControlError


class SteerUnsupportedError(ControlError):
    """Raised when the harness does not support mid-turn steering."""


class SteerNoActiveTurnError(ControlError):
    """Raised when steering a session with no active turn."""


class ModeRequiresRestartError(ControlError):
    """Raised when a mode change can only apply at next launch."""


class ModeRejectedError(ControlError):
    """Raised when the harness rejects a mode change."""


class PromptDeliveryFailedError(ControlError):
    """Raised when a prompt could not be delivered to the harness."""


class ControlTransportError(ControlError):
    """Raised when the control channel fails or disconnects."""


class HarnessNotInitializedError(ControlError):
    """Raised when control is used before the harness adapter is initialized."""


class ThreadOwnershipError(ControlError):
    """Raised when another process owns the harness thread being attached."""
