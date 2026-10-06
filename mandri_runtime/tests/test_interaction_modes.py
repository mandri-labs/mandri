import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.ids import ApprovalDecision, ApprovalKind, HarnessKind, RawEvent, SessionId
from mandri.runtime.control.claude import ClaudeApprovalMessenger
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.interaction_modes import InteractionModes
from mandri.runtime.service import RuntimeService


def tool_call(name, identifier="tool"):
    return {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "id": identifier, "name": name, "input": {}}]},
    }


def tool_result(identifier="tool", failed=False):
    return {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "tool_use_id": identifier, "is_error": failed}]
        },
    }


def runtime():
    service = RuntimeService(harness_commands={})
    service._registry.mark_live("s", harness="claude")
    service._sessions = SimpleNamespace(set_session_interaction_mode=AsyncMock())
    service._events.publish_both = Mock()
    return service


def test_claude_plan_mode_requires_success_and_excludes_subagents():
    modes = InteractionModes()
    modes.confirm("bypassPermissions")
    assert modes.observe("claude", tool_call("EnterPlanMode")) is None
    assert modes.observe("claude", tool_result(failed=True)) is None
    assert modes.mode == "bypassPermissions"
    assert (
        modes.observe("claude", {**tool_call("EnterPlanMode"), "parent_tool_use_id": "child"})
        is None
    )
    assert modes.observe("claude", tool_result()) is None
    modes.observe("claude", tool_call("EnterPlanMode"))
    assert modes.observe("claude", tool_result()) == "plan"
    assert "bypassPermissions" in modes.plan_modes()


@pytest.mark.parametrize("subtype", ["init", "status"])
async def test_native_state_is_persisted_and_published_before_the_native_frame(subtype):
    service = runtime()
    publisher = AsyncMock()
    service._events.publisher = publisher
    frame = {
        "source": "claude",
        "raw": {"type": "system", "subtype": subtype, "permissionMode": "plan"},
        "ts": 10,
    }
    await service._events.publish_event(Topic("session.s"), frame)
    service._sessions.set_session_interaction_mode.assert_awaited_once_with(
        SessionId("s"), "plan", "mid_session_applied"
    )
    assert service._events.publish_both.call_args.args[1]["mode"] == "plan"
    publisher.assert_awaited_once_with(Topic("session.s"), frame)
    await service._events.publish_event(Topic("session.s"), frame)
    assert service._sessions.set_session_interaction_mode.await_count == 1


@pytest.mark.parametrize("failed", [False, True])
async def test_plan_approval_sends_native_mode_and_waits_for_confirmation(failed):
    service = runtime()
    service._session_state("s").interaction_modes.confirm("bypassPermissions")
    await service._events.publish_event(
        Topic("session.s"),
        {
            "source": "claude",
            "raw": {"type": "system", "subtype": "status", "permissionMode": "plan"},
        },
    )
    raw = {
        "type": "control_request",
        "request_id": "native",
        "request": {
            "subtype": "can_use_tool",
            "tool_name": "ExitPlanMode",
            "tool_use_id": "exit",
            "input": {"plan": "Synthetic plan"},
        },
    }
    request = await service._approvals.register(
        SessionId("s"),
        HarnessKind.CLAUDE,
        RawEvent(json.dumps(raw)),
        "native",
        ApprovalKind.PERMISSION_SCOPE,
        120,
    )
    frames = []
    service._session_state("s").delivery = ClaudeApprovalMessenger(
        stdin=SimpleNamespace(write=frames.append, drain=AsyncMock())
    )
    await service.answer_approval(
        str(request.id), ApprovalDecision.ALLOW, permission_mode="bypassPermissions"
    )
    assert json.loads(frames[0])["response"]["response"] == {
        "behavior": "allow",
        "updatedInput": {"plan": "Synthetic plan"},
        "updatedPermissions": [
            {"type": "setMode", "mode": "bypassPermissions", "destination": "session"}
        ],
    }
    assert service._session_state("s").interaction_modes.mode == "plan"
    await service._events.publish_event(
        Topic("session.s"), {"source": "claude", "raw": tool_result("exit", failed)}
    )
    assert service._session_state("s").interaction_modes.mode == (
        "plan" if failed else "bypassPermissions"
    )


async def test_unavailable_bypass_does_not_resolve_or_deliver_the_plan_request():
    service = runtime()
    raw = {"request": {"tool_name": "ExitPlanMode", "tool_use_id": "exit", "input": {}}}
    request = await service._approvals.register(
        SessionId("s"),
        HarnessKind.CLAUDE,
        RawEvent(json.dumps(raw)),
        "native",
        ApprovalKind.PERMISSION_SCOPE,
        120,
    )
    service._session_state("s").delivery = AsyncMock()
    with pytest.raises(ControlTransportError, match="unavailable"):
        await service.answer_approval(
            str(request.id), ApprovalDecision.ALLOW, permission_mode="bypassPermissions"
        )
    assert request.status.value == "pending"
    service._session_state("s").delivery.deliver.assert_not_awaited()


@pytest.mark.parametrize(
    "harness,raw,expected",
    [
        (
            "pi",
            {
                "type": "extension_ui_request",
                "method": "setStatus",
                "statusKey": "_mandri_permissions",
                "statusText": "token:plan",
            },
            "plan",
        ),
        (
            "opencode",
            {"type": "session.updated", "properties": {"info": {"permission": {"*": "allow"}}}},
            "auto",
        ),
        (
            "codex",
            {
                "result": {
                    "thread": {"id": "thread"},
                    "approvalPolicy": "never",
                    "sandbox": {"type": "dangerFullAccess"},
                }
            },
            "full-access",
        ),
    ],
)
def test_native_permission_events_are_recognized_without_conflating_agent_modes(
    harness, raw, expected
):
    assert InteractionModes().observe(harness, raw) == expected
    assert (
        InteractionModes().observe(
            "opencode", {"type": "session.updated", "properties": {"info": {"agent": "plan"}}}
        )
        is None
    )


async def test_launch_persistence_cannot_overwrite_a_newer_native_mode():
    service = runtime()
    service._session_state("s").interaction_modes.confirm("plan")
    await service._persist_current_launch_mode("s", "bypassPermissions")
    service._sessions.set_session_interaction_mode.assert_not_awaited()


async def test_plan_request_and_replay_publish_available_execution_modes():
    hub = Hub()
    service = RuntimeService(harness_commands={}, hub=hub)
    service._registry.mark_live("s", harness="claude")
    service._session_state("s").interaction_modes.confirm("bypassPermissions")
    service._session_state("s").interaction_modes.confirm("plan")
    service._attach_approvals("s", HarnessKind.CLAUDE)
    handle = hub.subscribe(Topic("session.s"))
    raw = {
        "type": "control_request",
        "request_id": "native",
        "request": {
            "subtype": "can_use_tool",
            "tool_name": "ExitPlanMode",
            "tool_use_id": "exit",
            "input": {"plan": "Synthetic plan"},
        },
    }
    try:
        await service._events.publish_event(Topic("session.s"), {"source": "claude", "raw": raw})
        await asyncio.wait_for(handle.queue.get(), 1)
        frame = await asyncio.wait_for(handle.queue.get(), 1)
        payload = frame["payload"]
        assert payload["type"] == "approval.pending"
        assert payload["kind"] == "permission_scope"
        assert payload["permission_modes"] == ["default", "acceptEdits", "bypassPermissions"]
        service.replay_pending_approvals("s")
        replay = await asyncio.wait_for(handle.queue.get(), 1)
        assert replay["payload"]["permission_modes"] == payload["permission_modes"]
    finally:
        await service._session_state("s").watcher.stop()
        hub.unsubscribe(handle)


@pytest.mark.parametrize("harness", ["opencode", "codex"])
def test_native_modes_from_other_sessions_do_not_change_the_parent(harness):
    modes = InteractionModes()
    modes.confirm("default")
    raw = (
        {
            "type": "session.updated",
            "properties": {"info": {"id": "child", "permission": {"*": "allow"}}},
        }
        if harness == "opencode"
        else {
            "result": {
                "thread": {"id": "child"},
                "approvalPolicy": "never",
                "sandbox": {"type": "dangerFullAccess"},
            }
        }
    )
    assert modes.observe(harness, raw, "parent") is None
    assert modes.mode == "default"
