from typing import Any

import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.ids import SessionId
from mandri.runtime.liveness.agy import AgyLivenessAdapter
from mandri.runtime.liveness.evidence import LivenessEvidence, LivenessEvidenceKind
from mandri.runtime.liveness.tracker import WorkingStateTracker
from mandri.runtime.liveness.types import BusyReason


@pytest.fixture
def monitor() -> tuple[AgyLivenessAdapter, WorkingStateTracker, SessionId]:
    tracker = WorkingStateTracker()
    sid = SessionId("synthetic-session")
    tracker.register(sid)
    adapter = AgyLivenessAdapter(Hub(), Topic("session.synthetic-session"), tracker, sid)
    emit(adapter, {"event": "init", "conversation_id": "root"})
    return adapter, tracker, sid


def emit(adapter: AgyLivenessAdapter, raw: dict[str, Any]) -> None:
    adapter._absorb({"payload": {"source": "agy", "raw": raw}})


def stop(adapter: AgyLivenessAdapter, native_id: str, idle: bool) -> None:
    emit(
        adapter,
        {"event": "hook", "hook": "Stop", "data": {"conversationId": native_id, "fullyIdle": idle}},
    )


def step(adapter: AgyLivenessAdapter, native_id: str, state: str) -> None:
    emit(
        adapter,
        {
            "event": "step_update",
            "step_update": {
                "conversation_id": native_id,
                "step_type": "tool",
                "state": state,
            },
        },
    )


def test_background_completion_after_result_releases_turn(monitor: Any) -> None:
    adapter, tracker, sid = monitor
    step(adapter, "root", "ACTIVE")
    stop(adapter, "root", False)
    emit(adapter, {"event": "result", "result": {"status": "SUCCESS"}})
    step(adapter, "root", "DONE")
    assert tracker.working_state(sid).busy
    stop(adapter, "root", True)
    assert not tracker.working_state(sid).busy


def test_result_without_stop_keeps_uncertain_session_busy(monitor: Any) -> None:
    adapter, tracker, sid = monitor
    step(adapter, "root", "ACTIVE")
    emit(adapter, {"event": "result", "result": {"status": "SUCCESS"}})
    assert tracker.working_state(sid).uncertain
    stop(adapter, "root", True)
    assert not tracker.working_state(sid).busy


@pytest.mark.parametrize("background", [False, True])
def test_cli_command_rejection_releases_prompt_without_erasing_background(monitor, background):
    adapter, tracker, sid = monitor
    if background:
        stop(adapter, "root", False)
    tracker.observe(LivenessEvidence(sid, LivenessEvidenceKind.PROMPT_STARTED, "slash-command"))
    emit(
        adapter,
        {
            "event": "result",
            "result": {
                "conversation_id": "root",
                "status": "ERROR",
                "num_turns": 0,
                "error": "/model is answered by the CLI itself and is unavailable with "
                "--input-format stream-json; run it as its own --print /model invocation",
            },
        },
    )
    assert tracker.working_state(sid).busy is background


def test_child_stop_never_ends_parent_turn(monitor: Any) -> None:
    adapter, tracker, sid = monitor
    step(adapter, "root", "ACTIVE")
    step(adapter, "child", "ACTIVE")
    stop(adapter, "child", True)
    assert BusyReason.TURN_ACTIVE in tracker.working_state(sid).reasons
    stop(adapter, "root", True)
    assert not tracker.working_state(sid).busy


def test_parent_idle_waits_for_known_children(monitor: Any) -> None:
    adapter, tracker, sid = monitor
    step(adapter, "root", "ACTIVE")
    step(adapter, "child", "ACTIVE")
    stop(adapter, "root", True)
    assert BusyReason.BACKGROUND_WORK in tracker.working_state(sid).reasons
    stop(adapter, "child", True)
    assert not tracker.working_state(sid).busy


def test_child_result_does_not_end_parent_turn(monitor: Any) -> None:
    adapter, tracker, sid = monitor
    step(adapter, "root", "ACTIVE")
    emit(adapter, {"event": "result", "conversation_id": "child"})
    assert BusyReason.TURN_ACTIVE in tracker.working_state(sid).reasons
    emit(adapter, {"event": "result", "result": {"conversation_id": "child", "status": "SUCCESS"}})
    assert BusyReason.TURN_ACTIVE in tracker.working_state(sid).reasons


def test_root_result_after_authoritative_stop_stays_idle(monitor: Any) -> None:
    adapter, tracker, sid = monitor
    step(adapter, "root", "ACTIVE")
    stop(adapter, "root", True)
    emit(adapter, {"event": "result", "result": {"conversation_id": "root", "status": "SUCCESS"}})
    assert not tracker.working_state(sid).busy


def test_gap_remains_uncertain_until_authoritative_idle(monitor: Any) -> None:
    adapter, tracker, sid = monitor
    adapter._absorb({"type": "gap"})
    emit(adapter, {"event": "approval_request", "request_id": "root:1"})
    emit(adapter, {"event": "approval_response", "request_id": "root:1"})
    assert tracker.working_state(sid).uncertain
    stop(adapter, "root", True)
    assert not tracker.working_state(sid).busy


def test_multiple_approvals_close_independently(monitor: Any) -> None:
    adapter, tracker, sid = monitor
    emit(adapter, {"event": "approval_request", "request_id": "root:1"})
    emit(adapter, {"event": "approval_request", "request_id": "child:2"})
    emit(adapter, {"event": "approval_response", "request_id": "root:1"})
    assert BusyReason.APPROVAL_PENDING in tracker.working_state(sid).reasons
    emit(adapter, {"event": "approval_response", "request_id": "child:2"})
    assert not tracker.working_state(sid).busy


def test_late_subagent_metadata_does_not_resurrect_finished_child(monitor: Any) -> None:
    adapter, tracker, sid = monitor
    stop(adapter, "child", True)
    emit(
        adapter,
        {
            "event": "step_update",
            "step_update": {
                "conversation_id": "root",
                "state": "DONE",
                "step_type": "tool",
                "subagent_info": {"subagents": [{"conversation_id": "child"}]},
            },
        },
    )
    stop(adapter, "root", True)
    assert not tracker.working_state(sid).busy
