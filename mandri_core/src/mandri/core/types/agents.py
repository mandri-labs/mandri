from dataclasses import dataclass, field
from enum import StrEnum

from mandri.core.ids import HarnessKind


class AgentState(StrEnum):
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AgentCapabilities:
    message: bool = False
    stop: bool = False


@dataclass(frozen=True)
class NativeAgent:
    harness: HarnessKind
    native_id: str
    parent_native_id: str
    title: str
    created_at: int = 0
    updated_at: int = 0
    parent_agent_native_id: str | None = None
    delegation_id: str | None = None
    task_id: str | None = None
    transcript_path: str | None = None
    state: AgentState = AgentState.UNKNOWN


@dataclass(frozen=True)
class Agent:
    id: str
    parent_session_id: str
    harness: HarnessKind
    native_id: str
    title: str
    state: AgentState
    created_at: int
    updated_at: int
    parent_agent_id: str | None = None
    session_id: str | None = None
    delegation_id: str | None = None
    task_id: str | None = None
    transcript_path: str | None = field(default=None, repr=False)
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
