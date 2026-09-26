"""Approval request domain model and lifecycle."""

import dataclasses
from typing import Any

from mandri.core.errors import MandriError
from mandri.core.ids import (
    ApprovalDecision,
    ApprovalId,
    ApprovalKind,
    ApprovalStatus,
    EpochMs,
    HarnessKind,
    RawEvent,
    SessionId,
)


class ApprovalError(MandriError):
    """Base class for all approval domain errors."""


class ApprovalExpiredError(ApprovalError):
    """Raised when acting on an approval whose deadline has passed."""


class ApprovalNotPendingError(ApprovalError):
    """Raised when the approval is no longer pending."""


class ApprovalAlreadyAnsweredError(ApprovalError):
    """Raised when transitioning an approval that is already answered."""


class DuplicateApprovalAnswerError(ApprovalError):
    """Raised when a second correlated answer arrives for the same approval."""


_LEGAL_TRANSITIONS: dict[ApprovalStatus, frozenset[ApprovalStatus]] = {
    ApprovalStatus.PENDING: frozenset(
        {ApprovalStatus.ANSWERED, ApprovalStatus.EXPIRED, ApprovalStatus.CANCELLED}
    ),
    ApprovalStatus.ANSWERED: frozenset(),
    ApprovalStatus.EXPIRED: frozenset(),
    ApprovalStatus.CANCELLED: frozenset(),
}


@dataclasses.dataclass(frozen=True)
class ApprovalRequest:
    id: ApprovalId
    session_id: SessionId
    harness: HarnessKind
    native_request: RawEvent
    native_request_ref: str
    kind: ApprovalKind
    created_at: EpochMs
    deadline: EpochMs
    status: ApprovalStatus
    decision: ApprovalDecision | None
    answers: list[dict[str, Any]] | None = None

    def transition(
        self,
        new_status: ApprovalStatus,
        decision: ApprovalDecision | None = None,
    ) -> "ApprovalRequest":
        if self.status is ApprovalStatus.ANSWERED:
            if new_status is ApprovalStatus.ANSWERED:
                raise DuplicateApprovalAnswerError(
                    f"approval {self.id} already answered by {self.decision}"
                )
            raise ApprovalAlreadyAnsweredError(f"approval {self.id} already answered")
        if self.status is ApprovalStatus.EXPIRED:
            raise ApprovalExpiredError(f"approval {self.id} expired")
        if self.status is ApprovalStatus.CANCELLED:
            raise ApprovalNotPendingError(f"approval {self.id} cancelled")
        if new_status not in _LEGAL_TRANSITIONS[self.status]:
            raise ApprovalNotPendingError(
                f"illegal transition {self.status.value} -> {new_status.value}"
            )
        if new_status is ApprovalStatus.ANSWERED and decision is None:
            raise ApprovalNotPendingError("answered transition requires a decision")
        return dataclasses.replace(self, status=new_status, decision=decision)
