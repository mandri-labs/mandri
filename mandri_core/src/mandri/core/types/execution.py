from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mandri.core.errors import MandriError
from mandri.core.types.model_selection import ModelSource


class ExecutionBackend(StrEnum):
    HOST = "host"
    DOCKER = "docker"


class PrivacyMode(StrEnum):
    NONE = "none"
    SURROGATE = "surrogate"


class ExecutionPhase(StrEnum):
    CHECKING = "checking"
    PREPARING_IMAGE = "preparing_image"
    PREPARING_STATE = "preparing_state"
    STARTING = "starting"
    READY = "ready"
    BLOCKED = "blocked"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ProtectionError(MandriError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SessionPolicy:
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE

    def validate(self, model_source: ModelSource = ModelSource.GATEWAY) -> None:
        if self.privacy_mode is PrivacyMode.SURROGATE and model_source is ModelSource.NATIVE:
            raise ProtectionError(
                "privacy_native_unsupported", "Native models are unavailable in protected sessions"
            )


@dataclass(frozen=True)
class ExecutionStatus:
    phase: ExecutionPhase
    generation: int = 0
    revision: int = 0
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
