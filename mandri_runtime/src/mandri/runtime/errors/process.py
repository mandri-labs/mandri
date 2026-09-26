"""Process-level runtime errors."""

from mandri.runtime.errors.base import RuntimeDomainError


class ProcessSpawnError(RuntimeDomainError):
    """The child process could not be spawned."""


class ProcessTerminationError(RuntimeDomainError):
    """The termination ladder failed to stop the child process."""


class ProcessJobError(RuntimeDomainError):
    """The Windows job object guarding the process tree failed."""


class ProcessIOError(RuntimeDomainError):
    """Reading or writing a child process stream failed."""
