from mandri.runtime.errors.base import RuntimeDomainError


class DockerExecutionError(RuntimeDomainError):
    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)
