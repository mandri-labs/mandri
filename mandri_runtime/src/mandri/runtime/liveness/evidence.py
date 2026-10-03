"""Harness-agnostic liveness evidence supplied by per-harness adapters."""

import dataclasses
import enum

from mandri.core.ids import SessionId
from mandri.core.types.conversation_status import WorkOutcome


class LivenessEvidenceKind(enum.StrEnum):
    PROMPT_STARTED = "prompt_started"
    PROMPT_REJECTED = "prompt_rejected"
    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"
    BACKGROUND_STARTED = "background_started"
    BACKGROUND_ENDED = "background_ended"
    APPROVAL_OPENED = "approval_opened"
    APPROVAL_CLOSED = "approval_closed"
    STATE_UNCERTAIN = "state_uncertain"
    STATE_SYNCED = "state_synced"


@dataclasses.dataclass(frozen=True)
class LivenessEvidence:
    session_id: SessionId
    kind: LivenessEvidenceKind
    prompt_id: str | None = None
    event_key: str | None = None
    content_key: str | None = None
    outcome: WorkOutcome | None = None
