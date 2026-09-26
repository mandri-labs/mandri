"""Derived session activity states and tiered metadata backoff."""

import dataclasses

from mandri.core.ids import ActivityState, EpochMs, SessionId
from mandri.core.types.config import SyncConfig


@dataclasses.dataclass(frozen=True)
class ActivityTransition:
    session_id: SessionId
    state: ActivityState


@dataclasses.dataclass(frozen=True)
class SessionActivity:
    session_id: SessionId
    state: ActivityState
    last_activity_at: EpochMs
    backoff_level: int


def derive_activity(observed_at: EpochMs, now: EpochMs, quiet_period_seconds: int) -> ActivityState:
    if now - observed_at < quiet_period_seconds * 1000:
        return ActivityState.ACTIVE
    return ActivityState.IDLE


def backoff_interval(config: SyncConfig, level: int) -> int:
    if level < 0:
        raise ValueError("backoff level must be non-negative")
    tier = config.mtime_poll_seconds * int(2**level)
    return min(tier, config.backoff_max_seconds)


class ActivityTracker:
    """Tracks per-session store metadata observations with edge-triggered states."""

    def __init__(self) -> None:
        self._metadata: dict[SessionId, EpochMs] = {}
        self._states: dict[SessionId, ActivityState] = {}
        self._levels: dict[SessionId, int] = {}

    def observe(
        self,
        session_id: SessionId,
        metadata_at: EpochMs,
        now: EpochMs,
        quiet_period_seconds: int,
    ) -> ActivityTransition | None:
        previous = self._metadata.get(session_id)
        if previous is not None and previous == metadata_at:
            self._levels[session_id] = self._levels.get(session_id, 0) + 1
        else:
            self._levels[session_id] = 0
        self._metadata[session_id] = metadata_at
        state = derive_activity(metadata_at, now, quiet_period_seconds)
        if self._states.get(session_id) is state:
            return None
        self._states[session_id] = state
        return ActivityTransition(session_id=session_id, state=state)

    def backoff_level(self, session_id: SessionId) -> int:
        return self._levels.get(session_id, 0)

    def state_of(self, session_id: SessionId) -> ActivityState | None:
        return self._states.get(session_id)

    def snapshot(self, session_id: SessionId) -> SessionActivity | None:
        state = self._states.get(session_id)
        metadata_at = self._metadata.get(session_id)
        if state is None or metadata_at is None:
            return None
        return SessionActivity(
            session_id=session_id,
            state=state,
            last_activity_at=metadata_at,
            backoff_level=self._levels.get(session_id, 0),
        )
