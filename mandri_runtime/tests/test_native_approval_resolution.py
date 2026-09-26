import json
from unittest.mock import AsyncMock

import pytest
from mandri.core.hub import Hub
from mandri.core.ids import (
    ApprovalDecision,
    ApprovalKind,
    ApprovalStatus,
    EpochMs,
    HarnessKind,
    RawEvent,
    SessionId,
)
from mandri.runtime.approvals.registry import ApprovalRegistry
from mandri.runtime.approvals.service import ApprovalService
from mandri.runtime.approvals.watcher import ApprovalWatcher
from mandri.runtime.control.errors import ControlTransportError


def service():
    return ApprovalService(ApprovalRegistry(), lambda: EpochMs(100), lambda _: None)


async def register(approvals, harness, raw, reference, session="session"):
    return await approvals.register(
        SessionId(session),
        harness,
        RawEvent(json.dumps(raw)),
        reference,
        ApprovalKind.USER_INPUT,
        60,
    )


@pytest.mark.parametrize(
    "harness,original,resolution,reference",
    [
        (
            HarnessKind.CODEX,
            {"id": 7, "params": {"threadId": "native"}},
            {"method": "serverRequest/resolved", "params": {"threadId": "native", "requestId": 7}},
            "7",
        ),
        (
            HarnessKind.OPENCODE,
            {"type": "question.asked", "properties": {"id": "q", "sessionID": "native"}},
            {"type": "question.replied", "properties": {"requestID": "q", "sessionID": "native"}},
            "q",
        ),
        (
            HarnessKind.OPENCODE,
            {"type": "question.asked", "properties": {"id": "q", "sessionID": "native"}},
            {"type": "question.rejected", "properties": {"requestID": "q", "sessionID": "native"}},
            "q",
        ),
    ],
)
async def test_native_resolution_removes_pending_card_once(
    harness, original, resolution, reference
):
    approvals = service()
    request = await register(approvals, harness, original, reference)
    other = await register(approvals, harness, original, reference, "other-session")
    publish = AsyncMock()
    watcher = ApprovalWatcher(
        Hub(), approvals, SessionId("session"), harness, 60, publisher=publish
    )
    await watcher._process({"source": harness.value, "raw": resolution})
    assert request.status is ApprovalStatus.CANCELLED
    assert request.decision is None
    assert other.status is ApprovalStatus.PENDING
    assert not approvals.pending_for_session(SessionId("session"))
    assert publish.call_args.args[1]["raw"] == {
        "approval_id": str(request.id),
        "outcome": "cancelled",
    }
    await watcher._process({"source": harness.value, "raw": resolution})
    assert publish.await_count == 1


@pytest.mark.parametrize(
    "params",
    [
        {"threadId": "other", "requestId": 7},
        {"threadId": "native", "requestId": "7"},
        {"requestId": 7},
    ],
)
async def test_resolution_requires_native_owner_and_exact_typed_id(params):
    approvals = service()
    request = await register(approvals, HarnessKind.CODEX, {"params": {"threadId": "native"}}, "7")
    watcher = ApprovalWatcher(Hub(), approvals, SessionId("session"), HarnessKind.CODEX, 60)
    await watcher._process(
        {"source": "codex", "raw": {"method": "serverRequest/resolved", "params": params}}
    )
    assert request.status is ApprovalStatus.PENDING


async def test_delivery_failure_leaves_request_pending_and_preserves_answers():
    approvals = service()
    request = await register(approvals, HarnessKind.CODEX, {}, "1")
    answers = [{"question": "choice", "answers": ["A"]}]
    deliver = AsyncMock(side_effect=ControlTransportError("native delivery failed"))
    with pytest.raises(ControlTransportError):
        await approvals.answer(
            request.id, ApprovalDecision.ACCEPT, answers=answers, deliver=deliver
        )
    assert request.status is ApprovalStatus.PENDING
    assert request.decision is None
    assert request.answers is None
    deliver.side_effect = None
    await approvals.answer(request.id, ApprovalDecision.ACCEPT, answers=answers, deliver=deliver)
    assert request.status is ApprovalStatus.ANSWERED
    assert request.answers == answers
