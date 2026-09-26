"""Tests for websocket wire frame validation and discriminated unions."""

import pytest
from mandri.core.protocol.errors import ProtocolErrorCode
from mandri.core.protocol.frames import (
    CLIENT_ADAPTER,
    SERVER_ADAPTER,
    ErrorFrame,
    EventFrame,
    GapFrame,
    PingFrame,
    PongFrame,
    RequestFrame,
    ResponseFrame,
    SnapshotFrame,
    SubscribeFrame,
    UnsubscribeFrame,
)
from pydantic import ValidationError


def test_client_adapter_discriminates_on_op() -> None:
    subscribe = CLIENT_ADAPTER.validate_python({"op": "subscribe", "topic": "sessions.all"})
    assert isinstance(subscribe, SubscribeFrame)
    assert subscribe.topic == "sessions.all"
    assert subscribe.since is None

    unsubscribe = CLIENT_ADAPTER.validate_python({"op": "unsubscribe", "topic": "t"})
    assert isinstance(unsubscribe, UnsubscribeFrame)


def test_client_adapter_rejects_unknown_op() -> None:
    with pytest.raises(ValidationError):
        CLIENT_ADAPTER.validate_python({"op": "explode", "topic": "t"})


def test_subscribe_frame_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CLIENT_ADAPTER.validate_python({"op": "subscribe", "topic": "t", "extra": 1})


def test_request_frame_requires_non_empty_action() -> None:
    frame = CLIENT_ADAPTER.validate_python(
        {"type": "request", "op_id": "a" * 36, "action": "session.prompt"}
    )
    assert isinstance(frame, RequestFrame)
    assert frame.params == {}
    with pytest.raises(ValidationError):
        CLIENT_ADAPTER.validate_python({"type": "request", "op_id": "a" * 36, "action": ""})


def test_response_frame_shape_validator() -> None:
    ok = ResponseFrame(type="response", op_id="a" * 36, ok=True, result={"x": 1})
    assert ok.error is None
    with pytest.raises(ValidationError):
        ResponseFrame(type="response", op_id="a" * 36, ok=True)
    with pytest.raises(ValidationError):
        ResponseFrame(
            type="response",
            op_id="a" * 36,
            ok=False,
            error={"code": ProtocolErrorCode.UNKNOWN_ACTION, "message": "m"},
            result={"x": 1},
        )


def test_server_adapter_dispatches_event_frame() -> None:
    frame = SERVER_ADAPTER.validate_python(
        {"topic": "session.1", "seq": 1, "source": "claude", "raw": {"x": 1}, "ts": 5}
    )
    assert isinstance(frame, EventFrame)
    assert frame.raw == {"x": 1}


def test_server_adapter_dispatches_gap_ping_pong() -> None:
    gap = SERVER_ADAPTER.validate_python(
        {"type": "gap", "topic": "t", "from_seq": 1, "seq": 2, "reason": "slow_consumer"}
    )
    assert isinstance(gap, GapFrame)
    assert SERVER_ADAPTER.validate_python({"type": "ping"}) == PingFrame(type="ping")
    assert SERVER_ADAPTER.validate_python({"type": "pong"}) == PongFrame(type="pong")


def test_server_adapter_snapshot_and_error_frames() -> None:
    snapshot = SERVER_ADAPTER.validate_python(
        {"type": "snapshot", "topic": "sessions.all", "sessions": [], "runtimes": []}
    )
    assert isinstance(snapshot, SnapshotFrame)
    error = SERVER_ADAPTER.validate_python({"type": "error", "detail": "d"})
    assert isinstance(error, ErrorFrame)
    with pytest.raises(ValidationError):
        SERVER_ADAPTER.validate_python({"type": "error", "detail": "d", "extra": 1})


def test_approval_frames_carry_lifecycle_fields() -> None:
    from mandri.core.protocol.frames import ApprovalPendingFrame, ApprovalResolvedFrame

    pending = ApprovalPendingFrame(
        type="approval.pending",
        topic="session.1",
        seq=1,
        source="claude",
        raw={},
        ts=1,
        approval_id="a",
        deadline=10,
        status="pending",
    )
    assert pending.approval_id == "a"
    resolved = ApprovalResolvedFrame(
        type="approval.resolved",
        topic="session.1",
        seq=2,
        source="claude",
        raw={},
        ts=2,
        approval_id="a",
        outcome="answered",
    )
    assert resolved.decision is None
