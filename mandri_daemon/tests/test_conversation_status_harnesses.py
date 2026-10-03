import asyncio

import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind, SessionId
from mandri.daemon.conversation_status import observe_liveness
from mandri.database.conversation_status import ConversationStatusRepository
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.runtime.liveness.factory import liveness_adapter
from mandri.runtime.liveness.tracker import WorkingStateTracker
from mandri.sessions.conversation_status import ConversationStatuses


@pytest.fixture
async def observed(tmp_path):
    database = AiosqliteDatabase()
    await database.connect(tmp_path / "test.db")
    await database.migrate()
    statuses = ConversationStatuses(ConversationStatusRepository(database))
    await statuses.start()
    yield statuses
    await statuses.close()
    await database.close()


async def deliver(hub, statuses, harness, *raw):
    for record in raw:
        hub.publish(Topic("session.s"), {"source": harness, "raw": record, "ts": 1})
    await asyncio.sleep(0)
    await statuses.flush()
    return statuses.get("session:s")


@pytest.mark.parametrize("harness", list(HarnessKind))
async def test_managed_native_settlement_is_specific_to_each_harness(observed, harness):
    hub = Hub()
    tracker = WorkingStateTracker(
        lambda evidence, state: observe_liveness(observed, evidence, state)
    )
    tracker.register(SessionId("s"))
    adapter = liveness_adapter(harness, hub, Topic("session.s"), tracker, SessionId("s"),
                               native_identity=lambda: "native")
    adapter.start()
    try:
        if harness is HarnessKind.CODEX:
            start = {"method": "turn/started", "params": {"threadId": "native"}}
            interim = {"method": "item/started", "params": {"threadId": "native", "item": {
                "type": "subAgentActivity", "agentThreadId": "child", "kind": "started"}}}
            end = {"method": "turn/completed", "params": {"threadId": "native", "turn": {
                "id": "turn", "status": "completed"}}}
            final = {"method": "item/completed", "params": {"threadId": "native", "item": {
                "type": "subAgentActivity", "agentThreadId": "child", "kind": "completed"}}}
        elif harness is HarnessKind.CLAUDE:
            start = {"type": "user", "message": {"role": "user"}}
            interim = {"type": "system", "subtype": "task_started", "task_id": "child"}
            end = {"type": "result", "subtype": "success", "queued_turn_count": 0}
            final = {"type": "system", "subtype": "task_notification", "task_id": "child",
                     "status": "completed"}
        elif harness is HarnessKind.OPENCODE:
            start = {"type": "session.status", "properties": {"sessionID": "native",
                     "status": {"type": "busy"}}}
            interim = {"type": "session.status", "properties": {"sessionID": "native",
                       "status": {"type": "retry"}}}
            end = {"type": "message.updated", "properties": {"info": {"sessionID": "native",
                   "role": "assistant", "finish": "stop", "time": {"completed": 2}}}}
            final = {"type": "session.idle", "properties": {"sessionID": "native"}}
        elif harness is HarnessKind.PI:
            start = {"type": "agent_start"}
            interim = {"type": "turn_end"}
            end = {"type": "agent_end", "willRetry": False}
            final = {"type": "agent_settled"}
        else:
            start = {"event": "hook", "hook": "PreInvocation", "data": {"conversationId": "native"}}
            interim = {"event": "step_update", "step_update": {"conversation_id": "native",
                       "step_type": "planner", "state": "DONE"}}
            end = {"event": "result", "result": {"status": "SUCCESS"}}
            final = {"event": "hook", "hook": "Stop", "data": {
                     "conversationId": "native", "fullyIdle": True}}
        status = await deliver(hub, observed, harness.value, start, interim, end)
        assert status.completion_revision == 0
        assert status.cycle_active
        status = await deliver(hub, observed, harness.value, final)
        assert status.work_state == "idle"
        assert status.completion_revision == 1
        assert status.read_revision == 0
    finally:
        await adapter.stop()
        await hub.close_all()


async def test_claude_queued_result_is_not_global_completion(observed):
    hub = Hub()
    tracker = WorkingStateTracker(
        lambda evidence, state: observe_liveness(observed, evidence, state)
    )
    tracker.register(SessionId("s"))
    adapter = liveness_adapter(HarnessKind.CLAUDE, hub, Topic("session.s"), tracker, SessionId("s"))
    adapter.start()
    try:
        status = await deliver(hub, observed, "claude", {"type": "user"},
                               {"type": "result", "queued_turn_count": 1})
        assert status.work_state == "working"
        assert status.completion_revision == 0
        status = await deliver(hub, observed, "claude", {"type": "result", "queued_turn_count": 0})
        assert status.completion_revision == 1
    finally:
        await adapter.stop()


async def test_pi_retry_updates_outcome_and_waits_for_settlement(observed):
    hub = Hub()
    tracker = WorkingStateTracker(
        lambda evidence, state: observe_liveness(observed, evidence, state)
    )
    tracker.register(SessionId("s"))
    adapter = liveness_adapter(HarnessKind.PI, hub, Topic("session.s"), tracker, SessionId("s"))
    adapter.start()
    try:
        status = await deliver(hub, observed, "pi", {"type": "agent_start"},
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "error"}},
            {"type": "auto_retry_end", "willRetry": False}, {"type": "agent_end"})
        assert status.completion_revision == 0
        status = await deliver(hub, observed, "pi",
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}},
            {"type": "agent_settled"})
        assert status.completion_revision == 1
        assert status.outcome == "completed"
    finally:
        await adapter.stop()
