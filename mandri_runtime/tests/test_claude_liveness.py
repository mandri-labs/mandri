from mandri.core.hub import Hub, Topic
from mandri.core.ids import SessionId
from mandri.runtime.liveness import WorkingStateTracker
from mandri.runtime.liveness.claude import ClaudeLivenessAdapter


def test_multiple_background_tasks_stay_busy_until_last_completion():
    tracker = WorkingStateTracker()
    session = SessionId("session")
    tracker.register(session)
    adapter = ClaudeLivenessAdapter(Hub(), Topic("session.session"), tracker, session)
    for task_id in ("first", "second"):
        adapter._translate({"type": "system", "subtype": "task_started", "task_id": task_id})
    adapter._translate({"type": "result"})
    for task_id in ("unrelated", "first", "second"):
        adapter._translate(
            {
                "type": "system",
                "subtype": "task_notification",
                "task_id": task_id,
                "status": "completed",
            }
        )
        assert tracker.working_state(session).busy is (task_id != "second")


def test_degradation_and_partial_events_keep_activity_uncertain():
    tracker = WorkingStateTracker()
    session = SessionId("session")
    tracker.register(session)
    adapter = ClaudeLivenessAdapter(Hub(), Topic("session.session"), tracker, session)
    adapter._absorb({"payload": {"source": "mandri", "raw": {"error": "parse_error"}}})
    adapter._translate({"type": "control_response"})
    adapter._translate({"type": "result"})
    assert tracker.working_state(session).uncertain
    assert tracker.working_state(session).busy


def test_native_and_local_approval_resolution_keep_other_requests_busy():
    tracker = WorkingStateTracker()
    session = SessionId("session")
    tracker.register(session)
    adapter = ClaudeLivenessAdapter(Hub(), Topic("session.session"), tracker, session)
    for request_id in ("first", "second"):
        adapter._translate(
            {
                "type": "control_request",
                "request_id": request_id,
                "request": {"subtype": "can_use_tool"},
            }
        )
    adapter._translate({"type": "control_response", "response": {"request_id": "first"}})
    assert tracker.working_state(session).busy
    adapter._absorb(
        {
            "payload": {
                "type": "approval.resolved",
                "source": "mandri",
                "raw": {"native_request_ref": "second"},
            }
        }
    )
    assert not tracker.working_state(session).busy
