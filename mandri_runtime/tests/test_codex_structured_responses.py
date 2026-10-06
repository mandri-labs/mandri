import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
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
from mandri.core.types.approvals import ApprovalRequest
from mandri.runtime.control.codex_delivery import CodexApprovalDelivery
from mandri.runtime.control.errors import ControlError


def request(kind, params, result, decision=ApprovalDecision.ACCEPT):
    return ApprovalRequest(
        ApprovalId("approval"),
        SessionId("s"),
        HarnessKind.CODEX,
        RawEvent(json.dumps({"params": params})),
        "native",
        kind,
        EpochMs(1),
        EpochMs(2),
        ApprovalStatus.ANSWERED,
        decision,
        updated_input=RawEvent(json.dumps(result)),
    )


@pytest.mark.parametrize("scope", ["turn", "session"])
async def test_explicit_permission_grants_reach_the_native_request(scope):
    control = AsyncMock()
    permissions = {"fileSystem": {"write": ["/synthetic/output"]}}
    result = {"permissions": permissions, "scope": scope}
    await CodexApprovalDelivery(control).deliver(
        request(ApprovalKind.PERMISSION_SCOPE, {"permissions": permissions}, result)
    )
    control.answer_native_request.assert_awaited_once_with("native", result)


async def test_permissions_cannot_be_widened_beyond_the_displayed_request():
    control = AsyncMock()
    native = {"permissions": {"fileSystem": {"write": ["/synthetic/output"]}}}
    with pytest.raises(ControlError, match="requested"):
        await CodexApprovalDelivery(control).deliver(
            request(
                ApprovalKind.PERMISSION_SCOPE,
                native,
                {"permissions": {"fileSystem": {"write": ["/"]}}, "scope": "session"},
            )
        )
    control.answer_native_request.assert_not_awaited()


async def test_elicitation_content_is_validated_before_native_delivery():
    control = AsyncMock()
    params = {
        "requestedSchema": {
            "type": "object",
            "required": ["enabled"],
            "properties": {"enabled": {"type": "boolean"}},
            "additionalProperties": False,
        }
    }
    result = {"action": "accept", "content": {"enabled": False}}
    await CodexApprovalDelivery(control).deliver(request(ApprovalKind.ELICITATION, params, result))
    control.answer_native_request.assert_awaited_once_with("native", result)
    control.reset_mock()
    with pytest.raises(ControlError, match="requested form"):
        await CodexApprovalDelivery(control).deliver(
            request(
                ApprovalKind.ELICITATION,
                params,
                {"action": "accept", "content": {"enabled": "false"}},
            )
        )
    control.answer_native_request.assert_not_awaited()


async def test_remote_schema_references_are_not_retrieved():
    control = AsyncMock()
    with pytest.raises(ControlError, match="requested form"):
        await CodexApprovalDelivery(control).deliver(
            request(
                ApprovalKind.ELICITATION,
                {"requestedSchema": {"$ref": "https://unused.invalid/schema"}},
                {"action": "accept", "content": {}},
            )
        )
    control.answer_native_request.assert_not_awaited()


async def test_elicitation_cancel_decision_reaches_the_native_request_as_cancel():
    control = AsyncMock()
    await CodexApprovalDelivery(control).deliver(
        request(ApprovalKind.ELICITATION, {}, {}, decision=ApprovalDecision.CANCEL)
    )
    control.answer_native_request.assert_awaited_once_with("native", {"action": "cancel"})


@pytest.mark.parametrize("cancelled,action", [(False, "decline"), (True, "cancel")])
async def test_elicitation_rejection_and_cancellation_are_distinct(cancelled, action):
    control = AsyncMock()
    rejected = request(ApprovalKind.ELICITATION, {}, {}, ApprovalDecision.DECLINE)
    if cancelled:
        rejected = replace(rejected, status=ApprovalStatus.CANCELLED, decision=None)
    await CodexApprovalDelivery(control).deliver(rejected)
    control.answer_native_request.assert_awaited_once_with("native", {"action": action})
