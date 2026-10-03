"""Topic and action registries for the live feed: payload models, specs, and resolution."""

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from mandri.core.ids import (
    ActivityState,
    ApprovalDecision,
    ApprovalId,
    ApprovalStatus,
    CorrelationId,
    EpochMs,
    HarnessKind,
    SessionState,
    SessionStopCause,
)
from mandri.core.protocol.agents import (
    AgentCreateParams,
    AgentCreateResult,
    AgentHistoryParams,
    AgentHistoryResult,
    AgentListParams,
    AgentListResult,
    AgentMessageParams,
    AgentMessageResult,
    AgentStopParams,
    AgentStopResult,
)
from mandri.core.protocol.commands import (
    CommandCatalog,
    CommandCatalogParams,
    CommandCatalogsParams,
    CommandCatalogsSnapshot,
    CommandGetParams,
    CommandInvocation,
    CommandInvokeParams,
    CommandSessionParams,
    CommandSnapshot,
    ScopedCommandCatalog,
)
from mandri.core.protocol.errors import ProtocolError, ProtocolErrorCode
from mandri.core.protocol.frames import RequestFrame, ResponseError, ResponseFrame
from mandri.core.protocol.types import RouteId, SessionId
from mandri.core.types.conversation_status import ConversationStatus
from mandri.core.types.execution import (
    ExecutionBackend,
    ExecutionPhase,
    PrivacyMode,
    ProtectionError,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Direction = Literal["send", "receive"]
logger = logging.getLogger(__name__)


class ConversationStatusPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["conversation_status"] = "conversation_status"
    status: ConversationStatus


class ConversationReadParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str = Field(pattern=r"^(session|agent):.+$")
    through_revision: int = Field(ge=1, strict=True)
    completion_key: str = Field(min_length=1)


class SessionLifecyclePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["session_started", "session_stopped", "session_state", "activity", "control_lost"]
    session_id: SessionId
    harness: HarnessKind
    state: SessionState | None = None
    activity: ActivityState | None = None
    last_activity_at: int | None = None
    cause: SessionStopCause | None = None
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE
    policy_revision: int = 1


class RuntimePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    harness: HarnessKind
    installed: bool
    degraded: bool


class ExecutionEventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["execution.updated"] = "execution.updated"
    session_id: SessionId
    generation: int
    revision: int
    phase: ExecutionPhase
    reason: str | None = None
    operation_id: str | None = None


class GatewayEventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: Literal["route_created", "route_updated", "route_deleted"]
    route_id: RouteId | None = None
    provider_name: str | None = None


class UsageChangedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int = Field(ge=0)
    session_id: str | None = None


class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: HarnessKind | Literal["mandri"]
    raw: dict[str, Any]
    ts: int
    type: str | None = None
    approval_id: ApprovalId | None = None
    deadline: EpochMs | None = None
    status: ApprovalStatus | None = None
    outcome: ApprovalStatus | None = None
    decision: ApprovalDecision | None = None


@dataclass(frozen=True)
class TopicSpec:
    name: str
    payload: type[BaseModel]
    direction: Direction
    description: str


TOPICS: dict[str, TopicSpec] = {
    "conversations.all": TopicSpec(
        "conversations.all",
        ConversationStatusPayload,
        "send",
        "Committed conversation work and read revisions.",
    ),
    "usage.changed": TopicSpec(
        name="usage.changed",
        payload=UsageChangedPayload,
        direction="send",
        description="Committed usage statistics invalidation.",
    ),
    "executions.all": TopicSpec(
        name="executions.all",
        payload=ExecutionEventPayload,
        direction="send",
        description="Session execution preparation and lifecycle changes.",
    ),
    "execution.{id}": TopicSpec(
        name="execution.{id}",
        payload=ExecutionEventPayload,
        direction="send",
        description="Execution generation and readiness for one session.",
    ),
    "agents.all": TopicSpec(
        name="agents.all",
        payload=EventEnvelope,
        direction="send",
        description="Delegated agent discovery and status changes.",
    ),
    "agent.{id}": TopicSpec(
        name="agent.{id}",
        payload=EventEnvelope,
        direction="send",
        description="Events and transcript updates for a delegated agent.",
    ),
    "sessions.all": TopicSpec(
        name="sessions.all",
        payload=SessionLifecyclePayload,
        direction="send",
        description="Session lifecycle snapshots (started, state changed, stopped).",
    ),
    "session.{id}": TopicSpec(
        name="session.{id}",
        payload=EventEnvelope,
        direction="send",
        description="Raw harness events for one daemon session id, verbatim.",
    ),
    "runtimes": TopicSpec(
        name="runtimes",
        payload=RuntimePayload,
        direction="send",
        description="Runtime availability changes (installed, degraded).",
    ),
    "gateway.events": TopicSpec(
        name="gateway.events",
        payload=GatewayEventPayload,
        direction="send",
        description="Route lifecycle and model swap notifications.",
    ),
}


def resolve(name: str) -> TopicSpec:
    spec = TOPICS.get(name)
    if spec is not None:
        return spec
    for candidate in TOPICS.values():
        if _matches(candidate.name, name):
            return candidate
    raise KeyError(name)


class ApprovalAnswerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: ApprovalId
    decision: ApprovalDecision
    updated_input: str | None = None
    answers: list[dict[str, Any]] | None = None


class ApprovalCancelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: ApprovalId


class SessionModeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: SessionId
    mode: str = Field(min_length=1)


class SessionPromptParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: SessionId
    content: str = ""
    attachments: list[str] = Field(default_factory=list, max_length=4)
    agent: str | None = None
    model: str | None = None

    @model_validator(mode="after")
    def require_content(self) -> "SessionPromptParams":
        if not self.content.strip() and not self.attachments:
            raise ValueError("A message requires text or attachments")
        return self


class SessionInterruptParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: SessionId


class SessionHistoryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: SessionId
    cursor: str | None = None
    limit: int = Field(default=100, ge=1, le=500)


class SessionListParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class ActionSpec:
    name: str
    params: type[BaseModel]
    description: str
    result: type[BaseModel] | None = None


ACTIONS: dict[str, ActionSpec] = {
    "conversation.read": ActionSpec(
        "conversation.read",
        ConversationReadParams,
        "Acknowledge a specific completion revision.",
        ConversationStatus,
    ),
    "command.catalogs": ActionSpec(
        "command.catalogs",
        CommandCatalogsParams,
        "Read startup native command catalogs.",
        CommandCatalogsSnapshot,
    ),
    "command.catalog": ActionSpec(
        "command.catalog",
        CommandCatalogParams,
        "Read a cached native command catalog for an execution scope.",
        ScopedCommandCatalog,
    ),
    "session.commands": ActionSpec(
        "session.commands",
        CommandSessionParams,
        "Discover native commands from the live harness.",
        CommandCatalog,
    ),
    "command.invoke": ActionSpec(
        "command.invoke",
        CommandInvokeParams,
        "Start a native command with an idempotent invocation identity.",
        CommandInvocation,
    ),
    "command.get": ActionSpec(
        "command.get",
        CommandGetParams,
        "Read authoritative command state without replaying it.",
        CommandInvocation,
    ),
    "command.list": ActionSpec(
        "command.list",
        CommandSessionParams,
        "Recover command invocations for this daemon session.",
        CommandSnapshot,
    ),
    "command.cancel": ActionSpec(
        "command.cancel",
        CommandGetParams,
        "Interrupt a running native command when supported.",
        CommandInvocation,
    ),
    "agent.list": ActionSpec(
        "agent.list",
        AgentListParams,
        "List delegated agents and their capabilities.",
        AgentListResult,
    ),
    "agent.history": ActionSpec(
        "agent.history",
        AgentHistoryParams,
        "Read a delegated agent transcript.",
        AgentHistoryResult,
    ),
    "agent.create": ActionSpec(
        "agent.create",
        AgentCreateParams,
        "Create an agent when supported by the parent harness.",
        AgentCreateResult,
    ),
    "agent.message": ActionSpec(
        "agent.message",
        AgentMessageParams,
        "Send input to an agent when its harness supports it.",
        AgentMessageResult,
    ),
    "agent.stop": ActionSpec(
        "agent.stop",
        AgentStopParams,
        "Stop a delegated agent using its native control channel.",
        AgentStopResult,
    ),
    "session.list": ActionSpec(
        name="session.list",
        params=SessionListParams,
        description="Read session metadata without starting a harness.",
    ),
    "session.history": ActionSpec(
        name="session.history",
        params=SessionHistoryParams,
        description="Read recent stored transcript pages without resuming a session.",
    ),
    "approval.answer": ActionSpec(
        name="approval.answer",
        params=ApprovalAnswerParams,
        description="Answer a pending approval with a harness-native decision.",
    ),
    "approval.cancel": ActionSpec(
        name="approval.cancel",
        params=ApprovalCancelParams,
        description="Cancel a pending approval; the timeout default applies.",
    ),
    "session.mode": ActionSpec(
        name="session.mode",
        params=SessionModeParams,
        description="Change the interaction mode of a running session mid-flight.",
    ),
    "session.prompt": ActionSpec(
        name="session.prompt",
        params=SessionPromptParams,
        description="Deliver a prompt or steering input to a running session.",
    ),
    "session.interrupt": ActionSpec(
        name="session.interrupt",
        params=SessionInterruptParams,
        description="Interrupt the active turn of a running session.",
    ),
}


def parse_params(action: str, params: Mapping[str, Any]) -> BaseModel:
    spec = ACTIONS.get(action)
    if spec is None:
        raise ProtocolError(ProtocolErrorCode.UNKNOWN_ACTION, f"unknown action: {action}")
    try:
        return spec.params.model_validate(dict(params))
    except ValidationError as exc:
        raise ProtocolError(
            ProtocolErrorCode.INVALID_PARAMS, f"invalid params for action: {action}"
        ) from exc


ActionHandler = Callable[[BaseModel], Awaitable[dict[str, Any]]]


def _error_response(
    op_id: CorrelationId, code: ProtocolErrorCode | str, message: str
) -> ResponseFrame:
    return ResponseFrame(
        type="response",
        op_id=op_id,
        ok=False,
        error=ResponseError(code=code, message=message),
    )


class ActionRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ActionHandler] = {}
        self._in_flight: set[str] = set()

    def register(self, action: str, handler: ActionHandler) -> None:
        if action not in ACTIONS:
            raise ProtocolError(ProtocolErrorCode.UNKNOWN_ACTION, f"unknown action: {action}")
        self._handlers[action] = handler

    async def handle(self, frame: RequestFrame) -> ResponseFrame:
        if frame.op_id in self._in_flight:
            return _error_response(
                frame.op_id,
                ProtocolErrorCode.DUPLICATE_OP_ID,
                f"op_id already in flight: {frame.op_id}",
            )
        self._in_flight.add(frame.op_id)
        try:
            return await self._dispatch(frame)
        finally:
            self._in_flight.discard(frame.op_id)

    async def _dispatch(self, frame: RequestFrame) -> ResponseFrame:
        try:
            params = parse_params(frame.action, frame.params)
        except ProtocolError as exc:
            return _error_response(frame.op_id, exc.code, exc.message)
        handler = self._handlers.get(frame.action)
        if handler is None:
            return _error_response(
                frame.op_id,
                ProtocolErrorCode.UNKNOWN_ACTION,
                f"no handler registered for action: {frame.action}",
            )
        try:
            result = await handler(params)
            result_model = ACTIONS[frame.action].result
            if result_model is not None:
                result = result_model.model_validate(result).model_dump(mode="json")
        except ProtocolError as exc:
            return _error_response(frame.op_id, exc.code, exc.message)
        except ProtectionError as exc:
            return _error_response(frame.op_id, exc.code, str(exc))
        except Exception:
            logger.exception("Request action failed: %s", frame.action)
            return _error_response(
                frame.op_id, ProtocolErrorCode.INTERNAL_ERROR, "The operation failed unexpectedly"
            )
        return ResponseFrame(type="response", op_id=frame.op_id, ok=True, result=result)


def topic_slug(name: str) -> str:
    parts = [part for part in name.split(".") if not part.startswith("{")]
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _matches(template: str, name: str) -> bool:
    parts = template.split(".")
    values = name.split(".")
    if len(parts) != len(values):
        return False
    return all(
        part == value or (part.startswith("{") and value)
        for part, value in zip(parts, values, strict=True)
    )
