"""In-memory storage of approval requests with pending-reference uniqueness."""

from mandri.core.ids import ApprovalId, ApprovalStatus, SessionId
from mandri.core.types.approvals import ApprovalRequest
from mandri.runtime.approvals.errors import DuplicateApprovalRequestError


class ApprovalRegistry:
    def __init__(self) -> None:
        self._requests: dict[ApprovalId, ApprovalRequest] = {}

    def add(self, request: ApprovalRequest) -> None:
        if self.has_pending_ref(request.session_id, request.native_request_ref):
            raise DuplicateApprovalRequestError(
                f"request {request.native_request_ref} already pending "
                f"for session {request.session_id}"
            )
        self._requests[request.id] = request

    def get(self, approval_id: ApprovalId) -> ApprovalRequest | None:
        return self._requests.get(approval_id)

    def pending(self) -> list[ApprovalRequest]:
        return [
            request
            for request in self._requests.values()
            if request.status is ApprovalStatus.PENDING
        ]

    def pending_for_session(self, session_id: SessionId) -> list[ApprovalRequest]:
        return [request for request in self.pending() if request.session_id == session_id]

    def has_pending_ref(self, session_id: SessionId, native_request_ref: str) -> bool:
        return any(
            request.native_request_ref == native_request_ref
            for request in self.pending_for_session(session_id)
        )
