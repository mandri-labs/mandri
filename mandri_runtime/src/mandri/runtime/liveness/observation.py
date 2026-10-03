from typing import Any

from mandri.core.ids import HarnessKind, SessionId
from mandri.core.types.conversation_status import WorkOutcome
from mandri.core.work_content import work_content_key
from mandri.core.work_outcomes import terminal_outcome, work_event_key
from mandri.runtime.liveness.evidence import LivenessEvidence, LivenessEvidenceKind


class WorkEventContext:
    def __init__(self, harness: HarnessKind) -> None:
        self._harness = harness
        self.key: str | None = None
        self.outcome: WorkOutcome | None = None
        self.content_key: str | None = None
        self._settles = False

    def observe(self, raw: dict[str, Any], timestamp: object = None, *, root: bool = True) -> None:
        self.key = work_event_key(raw, timestamp)
        if root:
            content_key = work_content_key(self._harness, raw)
            if content_key is not None:
                self.content_key = content_key
            if raw.get("type") in {"user", "agent_start"} or raw.get("method") == "turn/started":
                self.content_key = None
        outcome = terminal_outcome(self._harness, raw) if root else None
        if outcome is not None and (
            outcome != "completed"
            or self.outcome not in {"failed", "interrupted"}
            or (
                raw.get("type") not in {"session.idle", "session.status"}
                and raw.get("event") not in {"hook", "Stop"}
            )
        ):
            self.outcome = outcome
        self._settles = root and self._can_settle(raw)

    def evidence(self, session: SessionId, kind: LivenessEvidenceKind) -> LivenessEvidence:
        outcome = None
        if kind is LivenessEvidenceKind.TURN_ENDED and self._settles:
            outcome = self.outcome or "completed"
        elif kind is LivenessEvidenceKind.TURN_STARTED:
            self.outcome = None
        return LivenessEvidence(
            session,
            kind,
            event_key=self.key,
            outcome=outcome,
            content_key=self.content_key if outcome is not None else None,
        )

    def _can_settle(self, raw: dict[str, Any]) -> bool:
        if self._harness is HarnessKind.PI:
            return raw.get("type") == "agent_settled"
        if self._harness is HarnessKind.CODEX:
            return raw.get("method") == "turn/completed" and self.outcome is not None
        if self._harness is HarnessKind.CLAUDE:
            return raw.get("type") == "result"
        if self._harness is HarnessKind.AGY:
            data = raw.get("data") if raw.get("event") == "hook" else raw
            hook = raw.get("hook") if raw.get("event") == "hook" else raw.get("event")
            return hook == "Stop" and isinstance(data, dict) and data.get("fullyIdle") is True
        return raw.get("type") in {"session.idle", "session.status", "session.error"} and (
            terminal_outcome(self._harness, raw) is not None
        )
