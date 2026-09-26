import asyncio
from unittest.mock import AsyncMock

import pytest
from mandri.runtime.control.errors import ControlError, ControlTransportError
from mandri.runtime.control.pi_commands import PiCommands


async def test_dynamic_catalog_preserves_all_resources_and_extension_builtin_precedence():
    call = AsyncMock(
        return_value={
            "commands": [
                {"name": "compact", "description": "Custom compact", "source": "extension"},
                {"name": "review", "source": "prompt"},
                {"name": "skill:analyze", "source": "skill"},
            ]
        }
    )
    rows = await PiCommands(call).list_commands()
    assert [(row["id"], row["kind"]) for row in rows[:3]] == [
        ("compact", "extension"),
        ("review", "prompt"),
        ("skill:analyze", "skill"),
    ]
    assert sum(row["id"] == "compact" for row in rows) == 1
    assert next(row for row in rows if row["id"] == "settings")["available"] is False


async def test_command_revalidates_native_catalog_before_exact_invocation():
    call = AsyncMock(
        side_effect=[
            {"commands": [{"name": "custom", "source": "extension"}]},
            {"disposition": "handled"},
        ]
    )
    result = await PiCommands(call).execute_command("custom", "  keep spacing\n")
    assert result["message"] == "Command handled"
    call.assert_awaited_with(
        "prompt", {"message": "/custom   keep spacing\n", "streamingBehavior": "followUp"}
    )


@pytest.mark.parametrize("identifier", ["removed", "settings"])
async def test_unknown_and_terminal_only_commands_never_become_model_prompts(identifier):
    call = AsyncMock(return_value={"commands": []})
    with pytest.raises(ControlError):
        await PiCommands(call).execute_command(identifier, "")
    assert call.await_count == 1


async def test_native_builtin_uses_rpc_and_honors_extension_cancellation():
    call = AsyncMock(side_effect=[{"commands": []}, {"cancelled": True}])
    with pytest.raises(ControlError, match="cancelled"):
        await PiCommands(call).execute_command("new", "")
    call.assert_awaited_with("new_session", {})


async def test_gateway_model_selection_uses_generic_session_control():
    call = AsyncMock(return_value={"commands": []})
    with pytest.raises(ControlError, match="session model selector"):
        await PiCommands(call, gateway_mode=True).execute_command("model", "other/model")
    assert call.await_count == 1


async def test_invalid_catalog_is_not_empty_success():
    call = AsyncMock(return_value={"commands": [{"name": "broken", "source": "unknown"}]})
    with pytest.raises(ControlTransportError):
        await PiCommands(call).list_commands()


@pytest.mark.parametrize("early", [False, True])
async def test_model_command_waits_for_settled_not_agent_end(early):
    started = asyncio.Event()
    commands = None

    async def call(command, params):
        if command == "get_commands":
            return {"commands": [{"name": "review", "source": "prompt"}]}
        started.set()
        commands.observe({"type": "agent_start"})
        if early:
            commands.observe({"type": "agent_settled"})
        return {"disposition": "started"}

    commands = PiCommands(call)
    task = asyncio.create_task(commands.execute_command("review", ""))
    await started.wait()
    if not early:
        commands.observe({"type": "agent_end"})
        assert not task.done()
        commands.observe({"type": "agent_settled"})
    assert (await task)["message"] == "Command completed"


async def test_model_failure_after_acceptance_is_not_command_success():
    started = asyncio.Event()

    async def call(command, params):
        if command == "get_commands":
            return {"commands": [{"name": "review", "source": "prompt"}]}
        started.set()
        return {"disposition": "started"}

    commands = PiCommands(call)
    task = asyncio.create_task(commands.execute_command("review", ""))
    await started.wait()
    commands.observe(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "error",
                "errorMessage": "Provider unavailable",
            },
        }
    )
    commands.observe({"type": "agent_settled"})
    with pytest.raises(ControlError, match="Provider unavailable"):
        await task


async def test_successful_native_retry_clears_intermediate_provider_failure():
    started = asyncio.Event()

    async def call(command, params):
        if command == "get_commands":
            return {"commands": [{"name": "review", "source": "prompt"}]}
        started.set()
        return {"disposition": "started"}

    commands = PiCommands(call)
    task = asyncio.create_task(commands.execute_command("review", ""))
    await started.wait()
    commands.observe(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "error",
                "errorMessage": "Transient overload",
            },
        }
    )
    commands.observe({"type": "agent_end"})
    assert not task.done()
    commands.observe({"type": "agent_start"})
    commands.observe(
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}}
    )
    commands.observe({"type": "agent_settled"})
    assert (await task)["message"] == "Command completed"


async def test_pi_087_ack_without_disposition_queries_state_and_waits_for_run():
    checked = asyncio.Event()

    async def call(command, params):
        if command == "get_commands":
            return {"commands": [{"name": "review", "source": "prompt"}]}
        if command == "get_state":
            checked.set()
            return {"isStreaming": True, "pendingMessageCount": 0}
        return {}

    commands = PiCommands(call)
    task = asyncio.create_task(commands.execute_command("review", ""))
    await checked.wait()
    assert not task.done()
    commands.observe({"type": "agent_end"})
    assert not task.done()
    commands.observe({"type": "agent_settled"})
    assert (await task)["message"] == "Command completed"


async def test_gateway_thinking_selection_uses_generic_session_control():
    call = AsyncMock(return_value={"commands": []})
    with pytest.raises(ControlError, match="session reasoning selector"):
        await PiCommands(call, gateway_mode=True).execute_command("thinking", "high")
    assert call.await_count == 1
