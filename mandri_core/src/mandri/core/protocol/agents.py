from mandri.core.ids import HarnessKind
from mandri.core.protocol.types import SessionId
from mandri.core.types.agents import AgentCapabilities, AgentState
from mandri.core.types.conversation_status import ConversationStatus
from pydantic import BaseModel, ConfigDict, Field


class AgentListParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: SessionId | None = None


class AgentHistoryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(min_length=1)
    cursor: str | None = None
    limit: int = Field(default=100, ge=1, le=500)


class AgentCreateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: SessionId
    content: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1, max_length=200)


class AgentMessageParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class AgentStopParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(min_length=1)


class AgentView(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)
    id: str
    parent_session_id: str
    parent_agent_id: str | None
    session_id: str | None
    native_id: str
    harness: HarnessKind
    title: str
    state: AgentState
    status: ConversationStatus | None = None
    delegation_id: str | None
    task_id: str | None = None
    capabilities: AgentCapabilities
    created_at: int
    updated_at: int


class ParentAgentCapabilities(BaseModel):
    create: bool


class AgentListResult(BaseModel):
    agents: list[AgentView]
    classified_session_ids: list[str] = Field(default_factory=list)
    parent_capabilities: dict[str, ParentAgentCapabilities]


class AgentHistoryResult(BaseModel):
    completion_revision: int | None = None
    completion_target: str | None = None
    entries: list[str]
    next_cursor: str | None
    has_more: bool


class AgentCreateResult(BaseModel):
    agent: AgentView


class AgentMessageResult(BaseModel):
    agent_id: str
    accepted: bool


class AgentStopResult(BaseModel):
    agent_id: str
    stopped: bool
