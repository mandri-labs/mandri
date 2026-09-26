"""Run command error types."""


class RunError(Exception):
    """Base class for run command failures."""


class DaemonUnreachableError(RunError):
    """Raised when the daemon cannot be reached."""


class ReadinessTimeoutError(RunError):
    """Raised when the daemon does not become ready in time."""


class ForeignServiceError(RunError):
    """Raised when the target service is not managed by this daemon."""


class ModelResolutionError(RunError):
    """Raised when a model reference cannot be resolved."""


class HarnessBinaryNotFoundError(RunError):
    """Raised when the harness binary is not found."""
