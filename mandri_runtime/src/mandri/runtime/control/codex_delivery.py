import json
from typing import Any

from mandri.core.ids import ApprovalDecision, ApprovalKind, ApprovalStatus
from mandri.core.types.approvals import ApprovalRequest
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.control.codex_structured import elicitation_response, permission_grant
from mandri.runtime.control.errors import ControlError
from mandri.runtime.question_answers import valid_question_answers

_ALLOW = frozenset(
    {
        ApprovalDecision.ALLOW,
        ApprovalDecision.ONCE,
        ApprovalDecision.ALWAYS,
        ApprovalDecision.ACCEPT,
        ApprovalDecision.ACCEPT_FOR_SESSION,
    }
)


class CodexApprovalDelivery:
    def __init__(self, control: CodexControlAdapter) -> None:
        self._control = control

    async def deliver(self, request: ApprovalRequest) -> bool:
        allowed = request.status is ApprovalStatus.ANSWERED and request.decision in _ALLOW
        if request.kind is ApprovalKind.USER_INPUT:
            result = {"answers": self._answers(request) if allowed else {}}
            delivered = await self._control.answer_native_request(
                request.native_request_ref, result
            )
        elif request.kind is ApprovalKind.ELICITATION:
            delivered = await self._control.answer_native_request(
                request.native_request_ref,
                elicitation_response(request)
                if allowed
                else {
                    "action": "decline"
                    if request.status is ApprovalStatus.ANSWERED
                    and request.decision is not ApprovalDecision.CANCEL
                    else "cancel"
                },
            )
        elif request.kind is ApprovalKind.PERMISSION_SCOPE:
            delivered = await self._control.answer_native_request(
                request.native_request_ref,
                permission_grant(request) if allowed else {"permissions": {}, "scope": "turn"},
            )
        else:
            decision = request.decision if request.status is ApprovalStatus.ANSWERED else None
            delivered = await self._control.answer_approval(
                request.native_request_ref, decision or ApprovalDecision.CANCEL
            )
        if not delivered:
            raise ControlError("Native approval is no longer pending")
        return True

    @staticmethod
    def _answers(request: ApprovalRequest) -> dict[str, Any]:
        if not valid_question_answers(request.answers):
            raise ControlError("Native question requires non-empty answers")
        try:
            raw = json.loads(request.native_request)
        except (TypeError, ValueError) as error:
            raise ControlError("Native question has invalid metadata") from error
        params = raw.get("params") if isinstance(raw, dict) else None
        questions = params.get("questions") if isinstance(params, dict) else None
        if not isinstance(questions, list) or not questions:
            raise ControlError("Native question has invalid metadata")
        identifiers = [
            question.get("id") if isinstance(question, dict) else None for question in questions
        ]
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ControlError("Native question has invalid identifiers")
        if len(set(identifiers)) != len(identifiers):
            raise ControlError("Native question has duplicate identifiers")
        result: dict[str, Any] = {}
        for answer in request.answers or []:
            identifier = answer["question"]
            if identifier not in identifiers or identifier in result:
                raise ControlError("Native question answer does not identify a question")
            if any(not value.strip() for value in answer["answers"]):
                raise ControlError("Native question requires non-empty answers")
            result[identifier] = {"answers": answer["answers"]}
        if set(result) != set(identifiers):
            raise ControlError("Every native question requires an answer")
        return result
