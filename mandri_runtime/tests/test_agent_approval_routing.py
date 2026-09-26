import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mandri.core.hub import Hub, Topic
from mandri.core.ids import ApprovalKind, ApprovalStatus, HarnessKind, RawEvent, SessionId
from mandri.core.types.agents import Agent, AgentState
from mandri.runtime.agent_events import AgentEventRouter
from mandri.runtime.approvals.watcher import ApprovalWatcher
from mandri.runtime.service import RuntimeService


async def test_child_approval_pending_resolved_and_replay_remain_scoped():
    hub = Hub()
    runtime = RuntimeService(harness_commands={}, hub=hub)
    history = AsyncMock()
    history.list.return_value = [
        Agent(
            "child", "parent", HarnessKind.CODEX, "child-native", "Child", AgentState.RUNNING, 1, 2
        )
    ]
    sessions = AsyncMock()
    sessions.get_session.return_value = SimpleNamespace(
        id="parent", native_id="parent-native", harness=HarnessKind.CODEX
    )
    router = AgentEventRouter(hub, history, sessions)
    runtime.set_event_router(router.publish, router.approval_topic)
    child = hub.subscribe(Topic("agent.child"))
    other = hub.subscribe(Topic("agent.other"))
    parent = hub.subscribe(Topic("session.parent"))
    raw = {
        "id": 42,
        "method": "item/commandExecution/requestApproval",
        "params": {"threadId": "child-native"},
    }
    payload = {"source": "codex", "ts": 1, "raw": raw}
    await router.publish(Topic("session.parent"), payload)
    parent.queue.get_nowait()
    child.queue.get_nowait()
    watcher = ApprovalWatcher(
        hub,
        runtime._approvals,
        SessionId("parent"),
        HarnessKind.CODEX,
        60,
        publisher=router.publish,
    )
    await watcher._process(payload)
    pending = child.queue.get_nowait()["payload"]
    assert pending["type"] == "approval.pending"
    assert parent.queue.get_nowait()["payload"]["agent_id"] == "child"
    await runtime._approvals.register(
        session_id=SessionId("parent"),
        harness=HarnessKind.CODEX,
        native_request=RawEvent(
            json.dumps({**raw, "id": 43, "params": {"threadId": "parent-native"}})
        ),
        native_request_ref="43",
        kind=ApprovalKind.COMMAND_EXECUTION,
        timeout_seconds=60,
    )
    runtime.replay_pending_approvals("parent", Topic("agent.child"))
    assert child.queue.get_nowait()["payload"]["approval_id"] == pending["approval_id"]
    assert child.queue.empty()
    runtime.replay_pending_approvals("parent", Topic("agent.other"))
    assert other.queue.empty()
    request = runtime._approvals.pending_for_session(SessionId("parent"))[0]
    runtime._events.publish_resolved(request.transition(ApprovalStatus.CANCELLED))
    assert child.queue.get_nowait()["payload"]["type"] == "approval.resolved"
    assert parent.queue.get_nowait()["payload"]["type"] == "approval.resolved"
    assert other.queue.empty()
    for handle in (child, other, parent):
        hub.unsubscribe(handle)
