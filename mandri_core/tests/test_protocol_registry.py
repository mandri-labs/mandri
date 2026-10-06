"""Tests for topic/action registries and the correlated request dispatcher."""

import asyncio

import pytest
from mandri.core.ids import SessionStopCause
from mandri.core.protocol.errors import ProtocolError, ProtocolErrorCode
from mandri.core.protocol.frames import RequestFrame, ResponseFrame
from mandri.core.protocol.registry import (
    ACTIONS,
    TOPICS,
    ActionRegistry,
    SessionLifecyclePayload,
    SessionPromptParams,
    parse_params,
    resolve,
    topic_slug,
)
from mandri.core.types.execution import ProtectionError
from pydantic import BaseModel, ConfigDict, ValidationError

SESSION_ID = "0123456789abcdef0123456789abcdef0123"


def test_lifecycle_payload_accepts_session_stopped_with_cause() -> None:
    payload = SessionLifecyclePayload.model_validate(
        {"type": "session_stopped", "session_id": SESSION_ID, "harness": "codex", "cause": "crash"}
    )
    assert payload.cause is SessionStopCause.CRASH


def test_lifecycle_payload_rejects_unknown_cause() -> None:
    with pytest.raises(ValidationError):
        SessionLifecyclePayload.model_validate(
            {
                "type": "session_stopped",
                "session_id": SESSION_ID,
                "harness": "codex",
                "cause": "alien",
            }
        )


def test_lifecycle_payload_stopped_without_cause_stays_valid() -> None:
    payload = SessionLifecyclePayload.model_validate(
        {"type": "session_stopped", "session_id": SESSION_ID, "harness": "claude"}
    )
    assert payload.cause is None


def test_lifecycle_payload_accepts_control_lost() -> None:
    payload = SessionLifecyclePayload.model_validate(
        {"type": "control_lost", "session_id": SESSION_ID, "harness": "opencode"}
    )
    assert payload.cause is None
    assert payload.state is None
    assert payload.activity is None


def test_lifecycle_payload_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        SessionLifecyclePayload.model_validate(
            {"type": "session_boomed", "session_id": SESSION_ID, "harness": "codex"}
        )


def test_topics_registry_contents() -> None:
    assert set(TOPICS) == {
        "conversations.all",
        "executions.all",
        "execution.{id}",
        "sessions.all",
        "session.{id}",
        "agents.all",
        "agent.{id}",
        "runtimes",
        "gateway.events",
        "usage.changed",
    }
    for spec in TOPICS.values():
        assert spec.direction == "send"


def test_resolve_exact_and_template_topics() -> None:
    assert resolve("sessions.all") is TOPICS["sessions.all"]
    spec = resolve("session.abc-123")
    assert spec.name == "session.{id}"


def test_resolve_unknown_topic_raises_key_error() -> None:
    with pytest.raises(KeyError):
        resolve("nope")


def test_parse_params_validates_known_action() -> None:
    params = parse_params(
        "session.prompt",
        {"session_id": "0123456789abcdef0123456789abcdef0123", "content": "hi"},
    )
    assert isinstance(params, SessionPromptParams)
    assert params.content == "hi"


def test_parse_params_unknown_action_raises_protocol_error() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        parse_params("nope", {})
    assert excinfo.value.code is ProtocolErrorCode.UNKNOWN_ACTION


def test_parse_params_invalid_params_raises_protocol_error() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        parse_params("session.prompt", {"session_id": "0123456789abcdef0123456789abcdef0123"})
    assert excinfo.value.code is ProtocolErrorCode.INVALID_PARAMS


def test_actions_registry_covers_session_and_approval_actions() -> None:
    assert set(ACTIONS) == {
        "conversation.read",
        "command.catalogs",
        "command.catalog",
        "session.commands",
        "command.invoke",
        "command.get",
        "command.list",
        "command.cancel",
        "session.history",
        "session.list",
        "approval.answer",
        "approval.cancel",
        "session.mode",
        "session.prompt",
        "session.interrupt",
        "agent.list",
        "agent.history",
        "agent.create",
        "agent.message",
        "agent.stop",
    }


def test_topic_slug_strips_templates() -> None:
    assert topic_slug("session.{id}") == "session"
    assert topic_slug("sessions.all") == "sessionsAll"
    assert topic_slug("gateway.events") == "gatewayEvents"


class _Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


async def test_registry_dispatches_to_handler() -> None:
    registry = ActionRegistry()

    async def handler(params: BaseModel) -> dict[str, object]:
        return {"echo": params.model_dump(mode="json")}

    registry.register("session.interrupt", handler)
    frame = RequestFrame(
        type="request",
        op_id="a" * 36,
        action="session.interrupt",
        params={"session_id": "0123456789abcdef0123456789abcdef0123"},
    )
    response = await registry.handle(frame)
    assert response.ok is True
    assert response.error is None


@pytest.mark.parametrize(
    "code", ["privacy_state_unavailable", "privacy_key_unavailable", "workspace_identity_changed"]
)
async def test_registry_preserves_actionable_protection_error_codes(code: str) -> None:
    registry = ActionRegistry()

    async def handler(params: BaseModel) -> dict[str, object]:
        raise ProtectionError(code, "The selected protection prerequisite changed")

    registry.register("session.prompt", handler)
    frame = RequestFrame(
        type="request",
        op_id="a" * 36,
        action="session.prompt",
        params={"session_id": SESSION_ID, "content": "Continue the selected task"},
    )
    response = await registry.handle(frame)
    assert response.ok is False and response.error is not None
    assert response.error.code == code
    assert response.error.message == "The selected protection prerequisite changed"
    assert ResponseFrame.model_validate_json(response.model_dump_json()) == response


async def test_registry_rejects_unknown_action_registration() -> None:
    registry = ActionRegistry()

    async def handler(params: BaseModel) -> dict[str, object]:
        return {}

    with pytest.raises(ProtocolError):
        registry.register("nope", handler)


async def test_registry_returns_error_response_for_unknown_action() -> None:
    registry = ActionRegistry()
    frame = RequestFrame(type="request", op_id="a" * 36, action="nope")
    response = await registry.handle(frame)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code is ProtocolErrorCode.UNKNOWN_ACTION


async def test_registry_rejects_duplicate_in_flight_op_id() -> None:
    registry = ActionRegistry()
    started = asyncio.Event()

    async def handler(params: BaseModel) -> dict[str, object]:
        started.set()
        await asyncio.sleep(0.05)
        return {}

    registry.register("session.interrupt", handler)
    frame = RequestFrame(
        type="request",
        op_id="a" * 36,
        action="session.interrupt",
        params={"session_id": "0123456789abcdef0123456789abcdef0123"},
    )
    first, second = await asyncio.gather(registry.handle(frame), registry.handle(frame))
    responses = [first, second]
    assert any(r.ok for r in responses)
    assert any(
        not r.ok and r.error is not None and r.error.code is ProtocolErrorCode.DUPLICATE_OP_ID
        for r in responses
    )


async def test_registry_converts_handler_protocol_error() -> None:
    registry = ActionRegistry()

    async def handler(params: BaseModel) -> dict[str, object]:
        raise ProtocolError(ProtocolErrorCode.SESSION_NOT_RUNNING, "not running")

    registry.register("session.interrupt", handler)
    frame = RequestFrame(
        type="request",
        op_id="a" * 36,
        action="session.interrupt",
        params={"session_id": "0123456789abcdef0123456789abcdef0123"},
    )
    response = await registry.handle(frame)
    assert isinstance(response, ResponseFrame)
    assert response.ok is False
    assert response.error is not None
    assert response.error.code is ProtocolErrorCode.SESSION_NOT_RUNNING
