import dataclasses
import json
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import (
    ApprovalDecision,
    ApprovalKind,
    ApprovalStatus,
    EpochMs,
    HarnessKind,
    RawEvent,
    SessionId,
)
from mandri.runtime.approvals.recognition import detect
from mandri.runtime.approvals.registry import ApprovalRegistry
from mandri.runtime.approvals.service import ApprovalService
from mandri.runtime.control.errors import ControlError
from mandri.runtime.control.pi_delivery import PiApprovalDelivery


async def request(method="input", **kwargs):
    service = ApprovalService(ApprovalRegistry(), lambda: EpochMs(1000), lambda _: None)
    raw = RawEvent(
        json.dumps({"type": "extension_ui_request", "id": "dialog", "method": method, **kwargs})
    )
    descriptor = detect(HarnessKind.PI, raw)
    assert descriptor is not None
    pending = await service.register(
        SessionId("session"), HarnessKind.PI, raw, "dialog", descriptor.kind, 300
    )
    return dataclasses.replace(
        pending, status=ApprovalStatus.ANSWERED, decision=ApprovalDecision.ACCEPT
    )


@pytest.mark.parametrize("method", ["select", "confirm", "input", "editor"])
async def test_only_native_dialogs_are_recognized(method):
    value = await request(method)
    assert value.kind is (
        ApprovalKind.PERMISSION_SCOPE if method == "confirm" else ApprovalKind.USER_INPUT
    )
    assert (
        detect(
            HarnessKind.PI,
            json.dumps({"type": "extension_ui_request", "id": "notice", "method": "notify"}),
        )
        is None
    )


async def test_native_timeout_shortens_mandri_deadline():
    value = await request(timeout=1250)
    assert value.deadline == 2250


@pytest.mark.parametrize("method", ["input", "editor"])
@pytest.mark.parametrize("text", ["  leading\ntrailing  ", ""])
async def test_text_dialogs_preserve_whitespace_and_allow_explicit_empty_text(method, text):
    value = dataclasses.replace(
        await request(method), answers=[{"question": "dialog", "answers": [text]}]
    )
    control = AsyncMock()
    control.answer_native_request.return_value = True
    assert await PiApprovalDelivery(control).deliver(value)
    control.answer_native_request.assert_awaited_once_with("dialog", {"value": text})


async def test_selection_validates_option_and_question_id():
    control = AsyncMock()
    delivery = PiApprovalDelivery(control)
    value = dataclasses.replace(
        await request("select", options=["first", "second"]),
        answers=[{"question": "dialog", "answers": ["other"]}],
    )
    with pytest.raises(ControlError, match="available option"):
        await delivery.deliver(value)
    control.answer_native_request.assert_not_awaited()
    value = dataclasses.replace(value, answers=[{"question": "wrong", "answers": ["first"]}])
    with pytest.raises(ControlError, match="matching answer"):
        await delivery.deliver(value)


async def test_confirmation_maps_decision_and_expiry_cancels():
    value = await request("confirm")
    control = AsyncMock()
    control.answer_native_request.return_value = True
    delivery = PiApprovalDelivery(control)
    await delivery.deliver(value)
    control.answer_native_request.assert_awaited_with("dialog", {"confirmed": True})
    await delivery.deliver(dataclasses.replace(value, decision=ApprovalDecision.DENY))
    control.answer_native_request.assert_awaited_with("dialog", {"confirmed": False})
    await delivery.deliver(dataclasses.replace(value, status=ApprovalStatus.EXPIRED, decision=None))
    control.answer_native_request.assert_awaited_with("dialog", {"cancelled": True})
