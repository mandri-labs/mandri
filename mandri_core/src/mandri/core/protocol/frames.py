"""Websocket feed frame DTOs and unions for the /v1/ws protocol."""

from typing import Annotated, Any, Literal

from mandri.core.ids import ApprovalKind, CorrelationId
from mandri.core.protocol.errors import ProtocolErrorCode
from mandri.core.types.conversation_status import ConversationStatus
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class SubscribeFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["subscribe"]
    topic: str
    since: int | None = None


class UnsubscribeFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["unsubscribe"]
    topic: str


class SubscribedAck(BaseModel):
    op: Literal["subscribed"]
    topic: str
    from_seq: int


class UnsubscribedAck(BaseModel):
    op: Literal["unsubscribed"]
    topic: str


class EventFrame(BaseModel):
    topic: str
    seq: int
    source: str
    raw: dict[str, Any]
    ts: int
    agent_id: str | None = None


class ApprovalPendingFrame(BaseModel):
    type: Literal["approval.pending"]
    topic: str
    seq: int
    source: str
    raw: dict[str, Any]
    ts: int
    approval_id: str
    deadline: int
    status: str
    kind: ApprovalKind | None = None
    permission_modes: list[str] | None = None
    agent_id: str | None = None


class ApprovalResolvedFrame(BaseModel):
    type: Literal["approval.resolved"]
    topic: str
    seq: int
    source: str
    raw: dict[str, Any]
    ts: int
    approval_id: str
    outcome: str
    decision: str | None = None


class InteractionModeFrame(EventFrame):
    type: Literal["interaction_mode"]


class SessionStoppedFrame(BaseModel):
    type: Literal["session_stopped"]
    topic: str
    seq: int
    source: str
    raw: dict[str, Any]
    ts: int


class ControlLostFrame(BaseModel):
    type: Literal["control_lost"]
    topic: str
    seq: int
    source: str
    raw: dict[str, Any]
    ts: int


class GapFrame(BaseModel):
    type: Literal["gap"]
    topic: str
    from_seq: int
    seq: int
    reason: str


class PingFrame(BaseModel):
    type: Literal["ping"]


class PongFrame(BaseModel):
    type: Literal["pong"]


class ErrorFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["error"]
    detail: str


class SnapshotFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["snapshot"]
    topic: str
    sessions: list[dict[str, Any]]
    statuses: list[ConversationStatus] = Field(default_factory=list)
    runtimes: list[dict[str, Any]]


class RequestFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["request"]
    op_id: CorrelationId
    action: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class ResponseError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ProtocolErrorCode | str
    message: str


class ResponseFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["response"]
    op_id: CorrelationId
    ok: bool
    result: dict[str, Any] | None = None
    error: ResponseError | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> "ResponseFrame":
        if self.ok:
            if self.result is None or self.error is not None:
                raise ValueError("ok response requires result and no error")
        elif self.error is None or self.result is not None:
            raise ValueError("failed response requires error and no result")
        return self


ClientFrame = Annotated[SubscribeFrame | UnsubscribeFrame, Field(discriminator="op")]
AckFrame = Annotated[SubscribedAck | UnsubscribedAck, Field(discriminator="op")]
ServerFrame = (
    AckFrame
    | ApprovalPendingFrame
    | ApprovalResolvedFrame
    | SessionStoppedFrame
    | ControlLostFrame
    | InteractionModeFrame
    | EventFrame
    | GapFrame
    | SnapshotFrame
    | ErrorFrame
    | PingFrame
    | PongFrame
    | ResponseFrame
)

CLIENT_ADAPTER: TypeAdapter[ClientFrame | RequestFrame] = TypeAdapter(ClientFrame | RequestFrame)
SERVER_ADAPTER: TypeAdapter[ServerFrame] = TypeAdapter(ServerFrame)
