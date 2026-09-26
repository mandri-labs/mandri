"""Working-state domain types for the liveness gate."""

import dataclasses
import enum

from mandri.core.errors import MandriError


class LivenessError(MandriError):
    """Base class for all liveness domain errors."""


class BusyReason(enum.StrEnum):
    TURN_ACTIVE = "turn_active"
    BACKGROUND_WORK = "background_work"
    APPROVAL_PENDING = "approval_pending"
    STATE_UNCERTAIN = "state_uncertain"


@dataclasses.dataclass(frozen=True)
class WorkingState:
    busy: bool = False
    reasons: frozenset[BusyReason] = dataclasses.field(default_factory=frozenset)
    uncertain: bool = False

    def __post_init__(self) -> None:
        reasons = frozenset(self.reasons)
        if BusyReason.STATE_UNCERTAIN in reasons:
            object.__setattr__(self, "uncertain", True)
        if self.uncertain:
            reasons = reasons | {BusyReason.STATE_UNCERTAIN}
            object.__setattr__(self, "busy", True)
        object.__setattr__(self, "reasons", reasons)
        if self.busy != bool(reasons):
            raise LivenessError("busy verdict must match the present reasons")
