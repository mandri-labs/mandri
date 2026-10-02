"""In-memory derivation of WorkingState from neutral liveness evidence."""

import typing
from dataclasses import dataclass

from mandri.core.ids import SessionId
from mandri.runtime.liveness.errors import UnknownSessionError
from mandri.runtime.liveness.evidence import LivenessEvidence, LivenessEvidenceKind
from mandri.runtime.liveness.types import BusyReason, WorkingState

_REASON_ADDED_BY_KIND: dict[LivenessEvidenceKind, BusyReason] = {
    LivenessEvidenceKind.TURN_STARTED: BusyReason.TURN_ACTIVE,
    LivenessEvidenceKind.BACKGROUND_STARTED: BusyReason.BACKGROUND_WORK,
    LivenessEvidenceKind.APPROVAL_OPENED: BusyReason.APPROVAL_PENDING,
}

_REASON_REMOVED_BY_KIND: dict[LivenessEvidenceKind, BusyReason] = {
    LivenessEvidenceKind.TURN_ENDED: BusyReason.TURN_ACTIVE,
    LivenessEvidenceKind.BACKGROUND_ENDED: BusyReason.BACKGROUND_WORK,
    LivenessEvidenceKind.APPROVAL_CLOSED: BusyReason.APPROVAL_PENDING,
}


@dataclass(frozen=True)
class _SessionEvidence:
    reasons: frozenset[BusyReason] = frozenset()
    uncertain: bool = False
    prompts: frozenset[str] = frozenset()


@typing.final
class WorkingStateTracker:
    """In-memory liveness port deriving working states with fail-toward-busy."""

    def __init__(self) -> None:
        self._sessions: dict[SessionId, _SessionEvidence] = {}

    def register(self, session_id: SessionId) -> None:
        self._sessions.setdefault(session_id, _SessionEvidence())

    def forget(self, session_id: SessionId) -> None:
        self._sessions.pop(session_id, None)

    def observe(self, evidence: LivenessEvidence) -> None:
        state = self._sessions.get(evidence.session_id)
        if state is None:
            raise UnknownSessionError(evidence.session_id)
        kind = evidence.kind
        reasons = state.reasons
        uncertain = state.uncertain
        prompts = state.prompts
        if kind is LivenessEvidenceKind.PROMPT_STARTED and evidence.prompt_id is not None:
            prompts = prompts | {evidence.prompt_id}
        elif kind is LivenessEvidenceKind.PROMPT_REJECTED and evidence.prompt_id is not None:
            prompts = prompts - {evidence.prompt_id}
        elif kind is LivenessEvidenceKind.STATE_UNCERTAIN:
            uncertain = True
        elif kind is LivenessEvidenceKind.STATE_SYNCED:
            uncertain = False
        else:
            if kind is LivenessEvidenceKind.TURN_ENDED:
                prompts = frozenset()
            added = _REASON_ADDED_BY_KIND.get(kind)
            removed = _REASON_REMOVED_BY_KIND.get(kind)
            if added is not None:
                reasons = reasons | {added}
            if removed is not None:
                reasons = reasons - {removed}
        self._sessions[evidence.session_id] = _SessionEvidence(
            reasons=frozenset(reasons), uncertain=uncertain, prompts=prompts
        )

    def working_state(self, session_id: SessionId) -> WorkingState:
        state = self._sessions.get(session_id)
        if state is None:
            raise UnknownSessionError(session_id)
        reasons = state.reasons | ({BusyReason.TURN_ACTIVE} if state.prompts else set())
        return WorkingState(
            busy=bool(reasons),
            reasons=frozenset(reasons),
            uncertain=state.uncertain,
        )
