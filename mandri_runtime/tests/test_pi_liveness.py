from mandri.core.hub import Hub, Topic
from mandri.core.ids import SessionId
from mandri.runtime.liveness.pi import PiLivenessAdapter
from mandri.runtime.liveness.tracker import WorkingStateTracker


def test_agent_end_keeps_session_busy_until_retries_and_compaction_settle():
    tracker = WorkingStateTracker()
    session = SessionId("managed")
    tracker.register(session)
    adapter = PiLivenessAdapter(Hub(), Topic("pi-test"), tracker, session)
    adapter._observe({"type": "agent_start"})
    adapter._observe({"type": "agent_end"})
    assert tracker.working_state(session).busy
    adapter._observe({"type": "auto_compaction_start"})
    adapter._observe({"type": "auto_compaction_end"})
    assert tracker.working_state(session).busy
    adapter._observe({"type": "agent_settled"})
    assert not tracker.working_state(session).busy


def test_handled_extension_command_does_not_wait_for_nonexistent_agent_run():
    tracker = WorkingStateTracker()
    session = SessionId("managed")
    tracker.register(session)
    adapter = PiLivenessAdapter(Hub(), Topic("pi-test"), tracker, session)
    adapter._observe(
        {
            "type": "response",
            "command": "prompt",
            "success": True,
            "data": {"disposition": "handled"},
        }
    )
    assert not tracker.working_state(session).busy


def test_root_state_does_not_clear_uncertainty_about_missing_dialog_events():
    tracker = WorkingStateTracker()
    session = SessionId("managed")
    tracker.register(session)
    adapter = PiLivenessAdapter(Hub(), Topic("pi-test"), tracker, session)
    adapter._absorb({"type": "gap"})
    adapter._observe(
        {
            "type": "response",
            "command": "get_state",
            "success": True,
            "data": {"isStreaming": False, "pendingMessageCount": 0},
        }
    )
    assert tracker.working_state(session).uncertain
