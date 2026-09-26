from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from mandri.core.ids import HarnessKind
from mandri.core.types.agents import Agent, AgentState
from mandri.runtime.control.agents.base import UnsupportedAgentOperation
from mandri.runtime.control.agents.claude import ClaudeAgentControl
from mandri.runtime.control.agents.codex import CodexAgentControl
from mandri.runtime.control.agents.opencode import OpencodeAgentControl
from mandri.runtime.control.errors import ControlTransportError


def child(harness=HarnessKind.CODEX):
    return Agent("agent", "parent", harness, "native-child", "Child", AgentState.RUNNING, 1, 2)


@pytest.mark.parametrize("active,method", [(True, "turn/steer"), (False, "turn/start")])
async def test_codex_messages_only_explicit_child(active, method):
    turns = [{"id": "child-turn", "status": "inProgress"}] if active else []
    call = AsyncMock(
        side_effect=[{"result": {"thread": {"id": "native-child", "turns": turns}}}, {"result": {}}]
    )
    await CodexAgentControl(call).message(child(), "hello")
    assert call.await_args_list[1].args[0] == method
    parameters = call.await_args_list[1].args[1]
    assert parameters["threadId"] == "native-child"
    assert parameters.get("expectedTurnId") == ("child-turn" if active else None)
    assert "model" not in parameters and "approvalPolicy" not in parameters


@pytest.mark.parametrize("operation", ["message", "stop"])
async def test_codex_wrong_thread_read_never_mutates(operation):
    call = AsyncMock(return_value={"result": {"thread": {"id": "parent", "turns": []}}})
    control = CodexAgentControl(call)
    with pytest.raises(ControlTransportError):
        if operation == "message":
            await control.message(child(), "hello")
        else:
            await control.stop(child())
    assert call.await_count == 1


async def test_codex_stop_interrupts_child_active_turn_only():
    call = AsyncMock(
        side_effect=[
            {
                "result": {
                    "thread": {
                        "id": "native-child",
                        "turns": [
                            {"id": "done", "status": "completed"},
                            {"id": "active", "status": "inProgress"},
                        ],
                    }
                }
            },
            {"result": {}},
        ]
    )
    assert await CodexAgentControl(call).stop(child())
    assert call.await_args.args == (
        "turn/interrupt",
        {"threadId": "native-child", "turnId": "active"},
    )


async def test_native_unsupported_actions_do_not_call_transport():
    call = AsyncMock()
    for control in (CodexAgentControl(call), ClaudeAgentControl(call)):
        with pytest.raises(UnsupportedAgentOperation):
            await control.create("hello", None)
    claude = ClaudeAgentControl(call)
    assert not claude.capabilities(child(HarnessKind.CLAUDE)).stop
    with pytest.raises(UnsupportedAgentOperation):
        await claude.stop(child(HarnessKind.CLAUDE))
    with pytest.raises(UnsupportedAgentOperation):
        await claude.message(child(HarnessKind.CLAUDE), "hello")
    call.assert_not_awaited()


async def test_claude_stops_verified_task_identity_not_agent_identity():
    call = AsyncMock(return_value=True)
    assert await ClaudeAgentControl(call).stop(
        replace(child(HarnessKind.CLAUDE), task_id="native-task")
    )
    call.assert_awaited_once_with("native-task")


@pytest.mark.parametrize("payload", [[], None, {}, {"id": "child", "parentID": "unrelated"}])
async def test_opencode_rejects_invalid_child_before_prompt(payload):
    call = AsyncMock(
        side_effect=[httpx.Response(200, json={"id": "parent"}), httpx.Response(200, json=payload)]
    )
    with pytest.raises(ControlTransportError):
        await OpencodeAgentControl(call, "parent").create("hello", None)
    assert call.await_count == 2


async def test_opencode_child_inherits_permissions_and_agent_but_uses_gateway_alias():
    settings = {
        "permission": [{"permission": "bash", "pattern": "*", "action": "deny"}],
        "agent": "plan",
        "model": {"id": "selected-model", "providerID": "selected-provider"},
    }
    call = AsyncMock(
        side_effect=[
            httpx.Response(200, json={"id": "parent", **settings}),
            httpx.Response(200, json={"id": "child", "parentID": "parent"}),
            httpx.Response(204),
        ]
    )
    await OpencodeAgentControl(call, "parent").create("hello", "Child")
    assert call.await_args_list[1].args == (
        "POST",
        "/session",
        {
            "parentID": "parent",
            "title": "Child",
            **settings,
            "model": {"providerID": "mandri", "id": "mandri_gateway"},
        },
    )
    assert call.await_args_list[2].args[1] == "/session/child/prompt_async"
    assert call.await_args_list[2].args[2]["model"] == {
        "providerID": "mandri",
        "modelID": "mandri_gateway",
    }


@pytest.mark.parametrize("stopped", [True, False])
async def test_opencode_stop_preserves_native_result(stopped):
    call = AsyncMock(return_value=httpx.Response(200, json=stopped))
    assert await OpencodeAgentControl(call, "parent").stop(child(HarnessKind.OPENCODE)) is stopped


async def test_opencode_native_identity_is_one_url_segment():
    call = AsyncMock(return_value=httpx.Response(204))
    await OpencodeAgentControl(call, "parent").message(
        replace(child(HarnessKind.OPENCODE), native_id="child/../parent"), "hello"
    )
    assert call.await_args.args[1] == "/session/child%2F..%2Fparent/prompt_async"
