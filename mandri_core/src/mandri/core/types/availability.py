import enum
from dataclasses import dataclass


class SessionOwner(enum.StrEnum):
    MANDRI = "mandri"
    EXTERNAL = "external"
    UNOWNED = "unowned"
    UNKNOWN = "unknown"


class SessionActivityState(enum.StrEnum):
    IDLE = "idle"
    BUSY = "busy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SessionAvailability:
    owner: SessionOwner
    activity: SessionActivityState
    can_resume: bool = False
    can_release: bool = False
    can_restore: bool = False
    reason: str | None = None
