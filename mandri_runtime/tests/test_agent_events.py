from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.errors import MandriError
from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind
from mandri.core.types.agents import Agent, AgentState
from mandri.runtime.agent_events import AgentEventRouter


@pytest.mark.parametrize("status", ["starting", "ready", "failed"])
async def test_owned_mcp_initialization_reaches_control_before_identity_is_persisted(status):
    hub = Hub()
    handle = hub.subscribe(Topic("session.parent"))
    history, sessions = AsyncMock(), AsyncMock()
    sessions.get_session.return_value = SimpleNamespace(
        id="parent", native_id=None, harness=HarnessKind.CODEX
    )
    router = AgentEventRouter(hub, history, sessions)
    raw = {
        "method": "mcpServer/startupStatus/updated",
        "params": {"threadId": "starting-thread", "name": "qualified", "status": status},
    }
    payload = {"source": "codex", "raw": raw, "ts": 1}
    await router.publish(Topic("session.parent"), payload)
    assert handle.queue.get_nowait()["payload"] == payload
    history.refresh.assert_not_awaited()
    history.save.assert_not_awaited()
    assert sessions.get_session.await_count == 2
    hub.unsubscribe(handle)


async def test_mcp_initialization_does_not_admit_foreign_events_after_identity_refresh():
    hub = Hub()
    handle = hub.subscribe(Topic("session.parent"))
    history, sessions = AsyncMock(), AsyncMock()
    history.list.return_value = []
    sessions.get_session.side_effect = [
        SimpleNamespace(id="parent", native_id=None, harness=HarnessKind.CODEX),
        SimpleNamespace(id="parent", native_id="owned", harness=HarnessKind.CODEX),
        SimpleNamespace(id="parent", native_id="owned", harness=HarnessKind.CODEX),
    ]
    router = AgentEventRouter(hub, history, sessions)
    raw = {
        "method": "mcpServer/startupStatus/updated",
        "params": {"threadId": "foreign", "name": "qualified", "status": "ready"},
    }
    await router.publish(Topic("session.parent"), {"source": "codex", "raw": raw})
    assert handle.queue.empty()
    history.save.assert_not_awaited()
    hub.unsubscribe(handle)


@pytest.mark.parametrize(
    "harness,raw",
    [
        (HarnessKind.CODEX, {"method": "turn/started", "params": {"threadId": "child-native"}}),
        (
            HarnessKind.OPENCODE,
            {"type": "message.part.updated", "properties": {"part": {"sessionID": "child-native"}}},
        ),
        (
            HarnessKind.CLAUDE,
            {"type": "assistant", "parent_tool_use_id": "delegation", "session_id": "root-native"},
        ),
    ],
)
async def test_native_child_events_do_not_pollute_parent(harness, raw):
    hub = Hub()
    parent_handle = hub.subscribe(Topic("session.parent"))
    child_handle = hub.subscribe(Topic("agent.child"))
    history = AsyncMock()
    history.list.return_value = [
        Agent(
            "child",
            "parent",
            harness,
            "child-native",
            "Child",
            AgentState.UNKNOWN,
            1,
            2,
            delegation_id="delegation",
        )
    ]
    sessions = AsyncMock()
    sessions.get_session.return_value = SimpleNamespace(
        id="parent", native_id="root-native", harness=harness
    )
    router = AgentEventRouter(hub, history, sessions)
    payload = {"source": harness.value, "raw": raw, "ts": 10}
    await router.publish(Topic("session.parent"), payload)
    child = child_handle.queue.get_nowait()
    assert child["payload"]["raw"] is raw
    assert parent_handle.queue.empty()
    assert history.save.await_args.args[0].state is AgentState.RUNNING
    hub.unsubscribe(parent_handle)
    hub.unsubscribe(child_handle)


async def test_unrelated_session_announcement_does_not_invent_child():
    hub = Hub()
    history = AsyncMock()
    history.list.return_value = []
    sessions = AsyncMock()
    sessions.get_session.return_value = SimpleNamespace(
        id="parent", native_id="root-native", harness=HarnessKind.OPENCODE
    )
    router = AgentEventRouter(hub, history, sessions)
    await router.publish(
        Topic("session.parent"),
        {
            "source": "opencode",
            "ts": 1,
            "raw": {
                "type": "session.created",
                "properties": {"info": {"id": "other-child", "parentID": "unrelated-root"}},
            },
        },
    )
    history.save.assert_not_awaited()


async def test_child_approval_reaches_parent_registry_and_child_view():
    hub = Hub()
    parent_handle = hub.subscribe(Topic("session.parent"))
    child_handle = hub.subscribe(Topic("agent.child"))
    history = AsyncMock()
    history.list.return_value = [
        Agent(
            "child", "parent", HarnessKind.CODEX, "child-native", "Child", AgentState.RUNNING, 1, 2
        )
    ]
    sessions = AsyncMock()
    sessions.get_session.return_value = SimpleNamespace(
        id="parent", native_id="root-native", harness=HarnessKind.CODEX
    )
    router = AgentEventRouter(hub, history, sessions)
    raw = {
        "id": 42,
        "method": "item/commandExecution/requestApproval",
        "params": {"threadId": "child-native"},
    }
    await router.publish(Topic("session.parent"), {"source": "codex", "ts": 1, "raw": raw})
    assert parent_handle.queue.get_nowait()["payload"]["agent_id"] == "child"
    assert child_handle.queue.get_nowait()["payload"]["raw"] is raw
    assert history.save.await_args.args[0].state is AgentState.WAITING
    hub.unsubscribe(parent_handle)
    hub.unsubscribe(child_handle)


@pytest.mark.parametrize("unavailable", [False, True])
async def test_unrelated_native_session_never_reaches_parent(unavailable):
    hub = Hub()
    handle = hub.subscribe(Topic("session.parent"))
    history = AsyncMock()
    history.list.return_value = []
    if unavailable:
        history.refresh.side_effect = MandriError("unavailable")
    sessions = AsyncMock()
    sessions.get_session.return_value = SimpleNamespace(
        id="parent", native_id="root-native", harness=HarnessKind.OPENCODE
    )
    router = AgentEventRouter(hub, history, sessions)
    await router.publish(
        Topic("session.parent"),
        {
            "source": "opencode",
            "raw": {"type": "session.idle", "properties": {"sessionID": "unrelated"}},
        },
    )
    assert handle.queue.empty()
    history.save.assert_not_awaited()
    hub.unsubscribe(handle)


async def test_parent_cache_avoids_query_per_token_but_refreshes_rotated_identity():
    hub = Hub()
    handle = hub.subscribe(Topic("session.parent"))
    history = AsyncMock()
    sessions = AsyncMock()
    sessions.get_session.return_value = SimpleNamespace(
        id="parent", native_id="original", harness=HarnessKind.CODEX
    )
    router = AgentEventRouter(hub, history, sessions)
    for _ in range(3):
        await router.publish(
            Topic("session.parent"),
            {"raw": {"method": "item/agentMessage/delta", "params": {"threadId": "original"}}},
        )
    assert sessions.get_session.await_count == 1
    sessions.get_session.return_value = SimpleNamespace(
        id="parent", native_id="rotated", harness=HarnessKind.CODEX
    )
    await router.publish(
        Topic("session.parent"),
        {"raw": {"method": "turn/started", "params": {"threadId": "rotated"}}},
    )
    assert sessions.get_session.await_count == 2
    assert handle.queue.qsize() == 4
    history.refresh.assert_not_awaited()
    hub.unsubscribe(handle)


async def test_parent_claude_approval_tool_identity_is_not_mistaken_for_child():
    hub = Hub()
    handle = hub.subscribe(Topic("session.parent"))
    history = AsyncMock()
    history.list.return_value = []
    sessions = AsyncMock()
    sessions.get_session.return_value = SimpleNamespace(
        id="parent", native_id="root-native", harness=HarnessKind.CLAUDE
    )
    router = AgentEventRouter(hub, history, sessions)
    raw = {
        "type": "control_request",
        "tool_use_id": "parent-tool",
        "request": {"subtype": "can_use_tool"},
    }
    await router.publish(Topic("session.parent"), {"source": "claude", "raw": raw})
    assert handle.queue.get_nowait()["payload"]["raw"] == raw
    hub.unsubscribe(handle)


def test_malformed_approval_native_payload_has_no_child_topic():
    router = AgentEventRouter(Hub(), AsyncMock(), AsyncMock())
    assert router.approval_topic(SimpleNamespace(native_request="invalid")) is None
