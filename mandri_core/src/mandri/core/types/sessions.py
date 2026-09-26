"""Session domain model and state machine."""

import dataclasses

from mandri.core.errors import MandriError
from mandri.core.ids import (
    EpochMs,
    HarnessKind,
    HarnessSessionId,
    ProjectPath,
    RouteId,
    SessionId,
    SessionState,
    SessionTitle,
)
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, SessionPolicy
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.worktrees import Worktree


class SessionError(MandriError):
    """Base class for all session domain errors."""


class SessionStateError(SessionError):
    """Raised on invalid session state transitions or invariants."""


_LEGAL_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.DISCOVERED: frozenset({SessionState.LIVE, SessionState.STOPPED}),
    SessionState.LIVE: frozenset({SessionState.STOPPED, SessionState.DISCOVERED}),
    SessionState.STOPPED: frozenset({SessionState.DISCOVERED}),
}


def can_transition(old: SessionState, new: SessionState) -> bool:
    """Return True when the state machine allows old -> new."""
    return new in _LEGAL_TRANSITIONS[old]


@dataclasses.dataclass(frozen=True)
class InteractionMode:
    mode: str
    applied: str


@dataclasses.dataclass(frozen=True)
class Session:
    id: SessionId
    harness: HarnessKind
    native_id: HarnessSessionId | None
    native_title: SessionTitle | None
    title_overlay: SessionTitle | None
    project_path: ProjectPath
    created_at: EpochMs
    updated_at: EpochMs
    state: SessionState
    model: str | None
    gateway_route_id: RouteId | None
    deleted: bool
    last_synced_at: EpochMs
    interaction_mode: InteractionMode | None = None
    reasoning_effort: str | None = None
    model_source: ModelSource = ModelSource.GATEWAY
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE
    privacy_scope_id: str | None = None
    execution_context: str | None = None
    worktree: Worktree | None = None
    policy_revision: int = 1
    parent_native_id: HarnessSessionId | None = None
    parent_session_id: SessionId | None = None

    @property
    def policy(self) -> SessionPolicy:
        return SessionPolicy(self.execution_backend, self.privacy_mode)

    @property
    def effective_title(self) -> str:
        return self.title_overlay or self.native_title or "untitled"

    def transition(self, new_state: SessionState) -> "Session":
        if not can_transition(self.state, new_state):
            raise SessionStateError(f"illegal transition {self.state.value} -> {new_state.value}")
        return dataclasses.replace(self, state=new_state)

    def validate(self) -> None:
        self.policy.validate(self.model_source)
        if self.updated_at < self.created_at:
            raise SessionStateError("updated_at must be >= created_at")
