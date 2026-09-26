import json
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
from mandri.runtime.approvals.recognition import detect
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.control.codex_delivery import CodexApprovalDelivery
from mandri.runtime.control.errors import ControlError, ControlTransportError


@pytest.mark.parametrize(
    "method,kind",
    [
        ("item/commandExecution/requestApproval", ApprovalKind.COMMAND_EXECUTION),
        ("item/fileChange/requestApproval", ApprovalKind.FILE_CHANGE),
        ("item/tool/requestUserInput", ApprovalKind.USER_INPUT),
    ],
)
async def test_integer_and_string_native_ids_remain_distinct_until_wire_delivery(method, kind):
    control = CodexControlAdapter(None, None)
    control._thread_id = "thread"
    control._send = AsyncMock()
    requests = []
    for identifier in (0, "0"):
        raw = {
            "id": identifier,
            "method": method,
            "params": {"questions": [{"id": "choice", "question": "Choose"}]},
        }
        control._dispatch(raw)
        surfaced = await control.next_native_request()
        descriptor = detect(HarnessKind.CODEX, json.dumps(raw))
        assert descriptor.native_request_ref == surfaced.native_request_ref
        requests.append(
            ApprovalRequest(
                ApprovalId(str(identifier)),
                SessionId("session"),
                HarnessKind.CODEX,
                RawEvent(json.dumps(raw)),
                surfaced.native_request_ref,
                kind,
                EpochMs(1),
                EpochMs(2),
                ApprovalStatus.ANSWERED,
                ApprovalDecision.ACCEPT,
                [{"question": "choice", "answers": ["first"]}],
            )
        )
    assert requests[0].native_request_ref != requests[1].native_request_ref
    delivery = CodexApprovalDelivery(control)
    for request, identifier in zip(reversed(requests), ("0", 0), strict=True):
        assert await delivery.deliver(request)
        sent = control._send.call_args.args[0]
        assert sent["id"] == identifier
        assert type(sent["id"]) is type(identifier)
        expected = (
            {"answers": {"choice": {"answers": ["first"]}}}
            if kind is ApprovalKind.USER_INPUT
            else {"decision": "accept"}
        )
        assert sent["result"] == expected
    with pytest.raises(ControlError, match="no longer pending"):
        await delivery.deliver(requests[0])


async def test_failed_native_write_preserves_original_identifier_for_retry():
    control = CodexControlAdapter(None, None)
    control._thread_id = "thread"
    control._dispatch({"id": 0, "method": "item/commandExecution/requestApproval"})
    request = await control.next_native_request()
    control._send = AsyncMock(side_effect=ControlTransportError("write failed"))
    with pytest.raises(ControlTransportError):
        await control.answer_approval(request.native_request_ref, ApprovalDecision.ACCEPT)
    control._send = AsyncMock()
    assert await control.answer_approval(request.native_request_ref, ApprovalDecision.DECLINE)
    assert control._send.call_args.args[0]["id"] == 0
    assert not await control.answer_approval('"0"', ApprovalDecision.ACCEPT)


def question_request(answers, decision=ApprovalDecision.ACCEPT):
    return ApprovalRequest(
        ApprovalId("approval"),
        SessionId("session"),
        HarnessKind.CODEX,
        RawEvent(
            json.dumps(
                {
                    "params": {
                        "questions": [
                            {"id": "first", "question": "Same label"},
                            {"id": "second", "question": "Same label"},
                        ]
                    }
                }
            )
        ),
        "request",
        ApprovalKind.USER_INPUT,
        EpochMs(1),
        EpochMs(2),
        ApprovalStatus.ANSWERED,
        decision,
        answers,
    )


async def test_multiple_questions_use_native_ids_not_ambiguous_labels():
    control = AsyncMock()
    control.answer_native_request.return_value = True
    request = question_request(
        [
            {"question": "second", "answers": ["B", "C"]},
            {"question": "first", "answers": ["A"]},
        ]
    )
    assert await CodexApprovalDelivery(control).deliver(request)
    control.answer_native_request.assert_awaited_once_with(
        "request",
        {
            "answers": {
                "second": {"answers": ["B", "C"]},
                "first": {"answers": ["A"]},
            }
        },
    )


@pytest.mark.parametrize(
    "answers",
    [
        None,
        [{"question": "Same label", "answers": ["A"]}],
        [{"question": "first", "answers": ["A"]}],
        [{"question": "first", "answers": ["A"]}, {"question": "first", "answers": ["B"]}],
        [{"question": "first", "answers": [" "]}, {"question": "second", "answers": ["B"]}],
    ],
)
async def test_invalid_answers_never_send_generic_accept(answers):
    control = AsyncMock()
    with pytest.raises(ControlError):
        await CodexApprovalDelivery(control).deliver(question_request(answers))
    control.answer_native_request.assert_not_awaited()
    control.answer_approval.assert_not_awaited()


async def test_question_cancellation_sends_empty_native_answers():
    control = AsyncMock()
    control.answer_native_request.return_value = True
    await CodexApprovalDelivery(control).deliver(question_request(None, ApprovalDecision.CANCEL))
    control.answer_native_request.assert_awaited_once_with("request", {"answers": {}})


async def test_server_resolution_prevents_stale_answer_delivery():
    control = CodexControlAdapter(None, None)
    control._thread_id = "thread"
    control._send = AsyncMock()
    control._dispatch({"id": 7, "method": "item/tool/requestUserInput"})
    request = await control.next_native_request()
    control._dispatch(
        {
            "method": "serverRequest/resolved",
            "params": {
                "threadId": "thread",
                "requestId": 7,
            },
        }
    )
    assert not await control.answer_native_request(request.native_request_ref, {"answers": {}})
    control._send.assert_not_awaited()
