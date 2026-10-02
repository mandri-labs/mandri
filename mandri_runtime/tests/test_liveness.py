"""Tests for the liveness domain port and the working-state tracker."""

import pytest
from mandri.core.ids import SessionId
from mandri.runtime.liveness import (
    BusyReason,
    LivenessError,
    LivenessEvidence,
    LivenessEvidenceKind,
    LivenessPort,
    UnknownSessionError,
    WorkingState,
    WorkingStateTracker,
)


def test_working_state_idle_is_consistent() -> None:
    state = WorkingState()
    assert not state.busy
    assert state.reasons == frozenset()
    assert not state.uncertain


def test_working_state_uncertain_forces_busy() -> None:
    state = WorkingState(busy=False, reasons=frozenset(), uncertain=True)
    assert state.busy
    assert state.reasons == frozenset({BusyReason.STATE_UNCERTAIN})


def test_working_state_uncertain_reason_implies_uncertain() -> None:
    state = WorkingState(busy=False, reasons=frozenset({BusyReason.STATE_UNCERTAIN}))
    assert state.uncertain
    assert state.busy


def test_working_state_rejects_busy_without_reasons() -> None:
    with pytest.raises(LivenessError):
        WorkingState(busy=True, reasons=frozenset())


def test_tracker_satisfies_the_liveness_port() -> None:
    port: LivenessPort = WorkingStateTracker()
    assert isinstance(port, WorkingStateTracker)


async def test_unknown_session_state_raises() -> None:
    tracker = WorkingStateTracker()
    with pytest.raises(UnknownSessionError):
        tracker.working_state(SessionId("s1"))


async def test_unknown_session_evidence_raises() -> None:
    tracker = WorkingStateTracker()
    evidence = LivenessEvidence(
        session_id=SessionId("s1"),
        kind=LivenessEvidenceKind.TURN_STARTED,
    )
    with pytest.raises(UnknownSessionError):
        tracker.observe(evidence)


def test_registered_session_starts_idle() -> None:
    tracker = WorkingStateTracker()
    sid = SessionId("s1")
    tracker.register(sid)
    assert tracker.working_state(sid) == WorkingState()


def test_turn_evidence_drives_busy_then_idle() -> None:
    tracker = WorkingStateTracker()
    sid = SessionId("s1")
    tracker.register(sid)
    tracker.observe(LivenessEvidence(session_id=sid, kind=LivenessEvidenceKind.TURN_STARTED))
    state = tracker.working_state(sid)
    assert state.busy
    assert state.reasons == frozenset({BusyReason.TURN_ACTIVE})
    assert not state.uncertain
    tracker.observe(LivenessEvidence(session_id=sid, kind=LivenessEvidenceKind.TURN_ENDED))
    assert tracker.working_state(sid) == WorkingState()


def test_reasons_accumulate_and_clear_independently() -> None:
    tracker = WorkingStateTracker()
    sid = SessionId("s1")
    tracker.register(sid)
    for kind in (
        LivenessEvidenceKind.TURN_STARTED,
        LivenessEvidenceKind.BACKGROUND_STARTED,
        LivenessEvidenceKind.APPROVAL_OPENED,
    ):
        tracker.observe(LivenessEvidence(session_id=sid, kind=kind))
    state = tracker.working_state(sid)
    assert state.busy
    assert state.reasons == frozenset(
        {
            BusyReason.TURN_ACTIVE,
            BusyReason.BACKGROUND_WORK,
            BusyReason.APPROVAL_PENDING,
        }
    )
    tracker.observe(LivenessEvidence(session_id=sid, kind=LivenessEvidenceKind.APPROVAL_CLOSED))
    assert tracker.working_state(sid).reasons == frozenset(
        {BusyReason.TURN_ACTIVE, BusyReason.BACKGROUND_WORK}
    )
    tracker.observe(LivenessEvidence(session_id=sid, kind=LivenessEvidenceKind.TURN_ENDED))
    assert tracker.working_state(sid).reasons == frozenset({BusyReason.BACKGROUND_WORK})
    tracker.observe(LivenessEvidence(session_id=sid, kind=LivenessEvidenceKind.BACKGROUND_ENDED))
    assert tracker.working_state(sid) == WorkingState()


def test_uncertainty_forces_busy_without_other_evidence() -> None:
    tracker = WorkingStateTracker()
    sid = SessionId("s1")
    tracker.register(sid)
    tracker.observe(LivenessEvidence(session_id=sid, kind=LivenessEvidenceKind.STATE_UNCERTAIN))
    state = tracker.working_state(sid)
    assert state.busy
    assert state.uncertain
    assert state.reasons == frozenset({BusyReason.STATE_UNCERTAIN})


def test_resynced_evidence_resolves_uncertainty() -> None:
    tracker = WorkingStateTracker()
    sid = SessionId("s1")
    tracker.register(sid)
    tracker.observe(LivenessEvidence(session_id=sid, kind=LivenessEvidenceKind.STATE_UNCERTAIN))
    tracker.observe(LivenessEvidence(session_id=sid, kind=LivenessEvidenceKind.TURN_STARTED))
    state = tracker.working_state(sid)
    assert state.busy
    assert state.uncertain
    tracker.observe(LivenessEvidence(session_id=sid, kind=LivenessEvidenceKind.STATE_SYNCED))
    assert not tracker.working_state(sid).uncertain
    tracker.observe(LivenessEvidence(session_id=sid, kind=LivenessEvidenceKind.TURN_ENDED))
    assert tracker.working_state(sid) == WorkingState()


def test_partial_approval_resolution_does_not_clear_gap_uncertainty():
    tracker = WorkingStateTracker()
    sid = SessionId("session")
    tracker.register(sid)
    for kind in (
        LivenessEvidenceKind.STATE_UNCERTAIN,
        LivenessEvidenceKind.APPROVAL_OPENED,
        LivenessEvidenceKind.APPROVAL_CLOSED,
        LivenessEvidenceKind.TURN_ENDED,
    ):
        tracker.observe(LivenessEvidence(sid, kind))
    assert tracker.working_state(sid).busy
    assert tracker.working_state(sid).uncertain


def test_forget_drops_state_until_reregistered() -> None:
    tracker = WorkingStateTracker()
    sid = SessionId("s1")
    tracker.register(sid)
    tracker.observe(LivenessEvidence(session_id=sid, kind=LivenessEvidenceKind.TURN_STARTED))
    tracker.forget(sid)
    with pytest.raises(UnknownSessionError):
        tracker.working_state(sid)
    tracker.register(sid)
    assert tracker.working_state(sid) == WorkingState()
