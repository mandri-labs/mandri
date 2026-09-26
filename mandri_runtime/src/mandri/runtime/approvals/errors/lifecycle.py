"""Approval lifecycle errors."""

from mandri.core.types.approvals import ApprovalError


class DuplicateApprovalRequestError(ApprovalError):
    """Raised when a harness request reference is already pending for a session."""
