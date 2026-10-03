import dataclasses
from typing import Any, Literal

from pydantic import BaseModel

type WorkState = Literal["idle", "working", "waiting", "unknown"]
type WorkOutcome = Literal["completed", "failed", "interrupted"]


class ConversationStatus(BaseModel):
    target: str
    revision: int = 0
    work_state: WorkState = "idle"
    completion_revision: int = 0
    read_revision: int = 0
    outcome: WorkOutcome | None = None
    completion_key: str | None = None
    completion_content_key: str | None = None
    observation_source: Literal["live", "native"] | None = None
    cycle_active: bool = False
    pending_outcome: WorkOutcome | None = None


@dataclasses.dataclass(frozen=True)
class WorkObservation:
    source: Literal["live", "native"] | None = None
    content_key: str | None = None
    state: WorkState | None = None
    progress: bool = False
    outcome: WorkOutcome | None = None
    key: str | None = None


@dataclasses.dataclass(frozen=True)
class WorkDelta:
    checkpoint: dict[str, Any]
    observations: tuple[WorkObservation, ...] = ()
    baseline: bool = False


class StatusRevisionError(ValueError):
    pass
