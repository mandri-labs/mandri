import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind
from mandri.core.types.agents import Agent, AgentState
from mandri.runtime.agent_events import AgentEventRouter, event_native_id


def setup_router(state=AgentState.UNKNOWN, *, parent_native="root", known=True):
    hub = Hub()
    parent = hub.subscribe(Topic("session.parent"))
    child = hub.subscribe(Topic("agent.child"))
    history = AsyncMock()
    history.list.return_value = (
        [Agent("child", "parent", HarnessKind.AGY, "native-child", "Child", state, 1, 2)]
        if known
        else []
    )
    sessions = AsyncMock()
    sessions.get_session.return_value = SimpleNamespace(
        id="parent", native_id=parent_native, harness=HarnessKind.AGY
    )
    return AgentEventRouter(hub, history, sessions), history, parent, child


async def publish(router, raw):
    await router.publish(Topic("session.parent"), {"source": "agy", "raw": raw, "ts": 1})


def hook(name, **data):
    return {"event": "hook", "hook": name, "data": {"conversationId": "native-child", **data}}


@pytest.mark.parametrize(
    "raw",
    [
        {"event": "init", "conversation_id": "native-child"},
        {
            "event": "step_update",
            "conversation_id": "root",
            "step_update": {"conversation_id": "native-child"},
        },
        {"event": "result", "result": {"conversation_id": "native-child"}},
        hook("PreInvocation"),
        {"event": "approval_request", "data": {"conversationId": "native-child"}},
        {"event": "approval_response", "data": {"conversationId": "native-child"}},
    ],
)
def test_agy_event_identity_uses_native_conversation_fields(raw):
    assert event_native_id(HarnessKind.AGY, raw) == "native-child"


def test_agy_step_or_request_identifier_does_not_invent_conversation_identity():
    assert (
        event_native_id(HarnessKind.AGY, {"event": "step_update", "step_update": {"step_index": 2}})
        is None
    )
    assert (
        event_native_id(HarnessKind.AGY, {"event": "approval_request", "request_id": "unknown:2"})
        is None
    )


async def test_agy_child_steps_route_only_to_child_and_refresh_native_relationships():
    router, history, parent, child = setup_router()
    raw = {
        "event": "step_update",
        "step_update": {"conversation_id": "native-child", "state": "ACTIVE", "step_index": 2},
    }
    await publish(router, raw)
    assert child.queue.get_nowait()["payload"]["raw"] == raw
    assert parent.queue.empty()
    history.refresh.assert_awaited_once_with(force=True)
    assert history.save.await_args.args[0].state is AgentState.RUNNING


@pytest.mark.parametrize(
    "raw,state",
    [
        (hook("PreInvocation"), AgentState.RUNNING),
        (hook("Stop", fullyIdle=False), AgentState.RUNNING),
        (hook("Stop", fullyIdle=True), AgentState.COMPLETED),
    ],
)
async def test_agy_child_lifetime_hook_keeps_parent_copy(raw, state):
    router, history, parent, child = setup_router()
    await publish(router, raw)
    assert child.queue.get_nowait()["payload"]["raw"] == raw
    copied = parent.queue.get_nowait()["payload"]
    assert copied["agent_id"] == "child"
    assert copied["raw"] == raw
    assert history.save.await_args.args[0].state is state


@pytest.mark.parametrize("name", ["PreToolUse", "PostToolUse"])
async def test_agy_child_tool_hook_does_not_duplicate_tool_in_parent(name):
    router, _history, parent, child = setup_router(AgentState.RUNNING)
    raw = hook(
        name, stepIdx=2, toolCall={"name": "run_command", "args": {"CommandLine": "echo test"}}
    )
    await publish(router, raw)
    assert child.queue.get_nowait()["payload"]["raw"] == raw
    assert parent.queue.empty()


async def test_agy_child_approval_reaches_registry_child_and_response_topic():
    router, history, parent, child = setup_router(AgentState.RUNNING)
    request = {
        "event": "approval_request",
        "request_id": "native-child:2",
        "tool_name": "run_command",
        "input": {"CommandLine": "echo test"},
        "data": {"conversationId": "native-child", "stepIdx": 2},
    }
    await publish(router, request)
    assert parent.queue.get_nowait()["payload"]["agent_id"] == "child"
    assert child.queue.get_nowait()["payload"]["raw"] == request
    assert history.save.await_args.args[0].state is AgentState.WAITING
    stored = SimpleNamespace(
        native_request=json.dumps(request), session_id="parent", harness=HarnessKind.AGY
    )
    assert router.approval_topic(stored) == Topic("agent.child")
    reply = {
        "event": "approval_response",
        "request_id": "native-child:2",
        "data": request["data"],
        "response": {"decision": "allow"},
    }
    await publish(router, reply)
    assert parent.queue.get_nowait()["payload"]["agent_id"] == "child"
    assert child.queue.get_nowait()["payload"]["raw"] == reply
    assert history.save.await_args.args[0].state is AgentState.RUNNING


async def test_agy_child_success_result_alone_does_not_mark_completed_or_pollute_parent():
    router, history, parent, child = setup_router(AgentState.RUNNING)
    raw = {"event": "result", "result": {"conversation_id": "native-child", "status": "SUCCESS"}}
    await publish(router, raw)
    assert child.queue.get_nowait()["payload"]["raw"] == raw
    assert parent.queue.empty()
    history.save.assert_not_awaited()


async def test_agy_initial_root_init_is_delivered_before_database_identity_binding():
    router, history, parent, child = setup_router(parent_native=None)
    raw = {"event": "init", "conversation_id": "new-root"}
    await publish(router, raw)
    assert parent.queue.get_nowait()["payload"]["raw"] == raw
    assert child.queue.empty()
    history.refresh.assert_not_awaited()
    history.save.assert_not_awaited()


async def test_agy_slash_command_result_without_conversation_stays_on_parent():
    router, history, parent, child = setup_router()
    raw = {"event": "result", "result": {"conversation_id": "", "status": "SUCCESS"}}
    await publish(router, raw)
    assert event_native_id(HarnessKind.AGY, raw) is None
    assert parent.queue.get_nowait()["payload"]["raw"] == raw
    assert child.queue.empty()
    history.refresh.assert_not_awaited()


async def test_agy_child_init_refreshes_cached_unbound_parent_identity():
    router, history, parent, child = setup_router(parent_native=None)
    await publish(router, {"event": "init", "conversation_id": "root"})
    parent.queue.get_nowait()
    router._sessions.get_session.return_value = SimpleNamespace(
        id="parent", native_id="root", harness=HarnessKind.AGY
    )
    raw = {"event": "init", "conversation_id": "native-child"}
    await publish(router, raw)
    assert child.queue.get_nowait()["payload"]["raw"] == raw
    assert parent.queue.empty()
    history.refresh.assert_awaited_once_with(force=True)


async def test_agy_unknown_relationship_is_not_inferred_from_child_hook():
    router, history, parent, child = setup_router(known=False)
    await publish(router, hook("PreInvocation"))
    assert parent.queue.empty()
    assert child.queue.empty()
    history.refresh.assert_awaited_once_with(force=True)
    history.save.assert_not_awaited()
