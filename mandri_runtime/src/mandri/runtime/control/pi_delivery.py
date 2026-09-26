import json
from typing import Any

from mandri.core.ids import ApprovalStatus
from mandri.core.types.approvals import ApprovalRequest
from mandri.runtime.control.errors import ControlError
from mandri.runtime.control.pi import ALLOW_DECISIONS, PiControlAdapter
from mandri.runtime.question_answers import valid_question_answers


class PiApprovalDelivery:
    def __init__(self, control: PiControlAdapter) -> None:
        self._control = control

    async def deliver(self, request: ApprovalRequest) -> bool:
        try:
            raw = json.loads(request.native_request)
        except (TypeError, ValueError) as error:
            raise ControlError("Pi dialog has invalid metadata") from error
        if not isinstance(raw, dict):
            raise ControlError("Pi dialog has invalid metadata")
        allowed = request.status is ApprovalStatus.ANSWERED and request.decision in ALLOW_DECISIONS
        if raw.get("method") == "confirm":
            result: dict[str, Any] = (
                {"confirmed": allowed}
                if request.status is ApprovalStatus.ANSWERED
                else {"cancelled": True}
            )
        elif not allowed:
            result = {"cancelled": True}
        else:
            answers = request.answers
            if (
                not valid_question_answers(answers)
                or len(answers or []) != 1
                or answers is None
                or answers[0]["question"] != request.native_request_ref
                or len(answers[0]["answers"]) != 1
            ):
                raise ControlError("Pi dialog requires exactly one matching answer")
            value = answers[0]["answers"][0]
            if raw.get("method") == "select" and value not in raw.get("options", []):
                raise ControlError("Pi dialog answer is not an available option")
            result = {"value": value}
        delivered = await self._control.answer_native_request(request.native_request_ref, result)
        if not delivered and request.status is ApprovalStatus.ANSWERED:
            raise ControlError("Pi dialog is no longer pending")
        return delivered
