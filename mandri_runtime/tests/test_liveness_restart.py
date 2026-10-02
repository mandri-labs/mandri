import pytest
from mandri.core.hub import Hub
from mandri.core.ids import SessionId
from mandri.runtime.liveness import UnknownSessionError, WorkingStateTracker
from mandri.runtime.service import RuntimeService


@pytest.mark.parametrize("harness", ["claude", "opencode", "codex", "agy", "pi"])
async def test_detach_and_reattach_drop_uncertain_generation_state(harness):
    hub = Hub()
    tracker = WorkingStateTracker()
    runtime = RuntimeService({}, hub=hub, liveness=tracker)
    session = SessionId("session")
    runtime._events.attach_liveness("session", harness)
    adapter = runtime._session_state("session").liveness_adapter
    adapter._absorb({"type": "gap"})
    assert tracker.working_state(session).uncertain
    await runtime._events.forget_liveness("session")
    with pytest.raises(UnknownSessionError):
        tracker.working_state(session)
    runtime._events.attach_liveness("session", harness)
    assert not tracker.working_state(session).busy
    await runtime._events.forget_liveness("session")


@pytest.mark.parametrize("harness", ["claude", "opencode", "codex", "agy", "pi"])
@pytest.mark.parametrize(
    "error", ["usage_collection_failed", "incomplete", "oversize", "decode_error", "parse_error"]
)
async def test_only_lost_native_stream_data_marks_activity_uncertain(harness, error):
    hub = Hub()
    tracker = WorkingStateTracker()
    runtime = RuntimeService({}, hub=hub, liveness=tracker)
    session = SessionId("session")
    runtime._events.attach_liveness("session", harness)
    adapter = runtime._session_state("session").liveness_adapter
    adapter._absorb({"payload": {"source": "mandri", "raw": {"error": error}}})
    state = tracker.working_state(session)
    assert state.uncertain is (error != "usage_collection_failed")
    assert state.busy is (error != "usage_collection_failed")
    await runtime._events.forget_liveness("session")
