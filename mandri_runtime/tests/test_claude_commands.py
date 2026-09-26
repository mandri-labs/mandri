import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from mandri.core.ids import HarnessSessionId
from mandri.runtime.control.claude import ClaudeControlAdapter
from mandri.runtime.control.claude_commands import ClaudeCommands
from mandri.runtime.control.errors import ControlError, ControlTransportError
from mandri.runtime.pump import LinePump


class Wire:
    def __init__(self) -> None:
        self.input: asyncio.Queue[bytes] = asyncio.Queue()
        self.sent: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def write(self, data: bytes) -> None:
        self.sent.put_nowait(json.loads(data))

    async def drain(self) -> None:
        pass

    async def lines(self) -> AsyncIterator[bytes]:
        while True:
            yield await self.input.get()

    def emit(self, frame: dict[str, Any]) -> None:
        self.input.put_nowait((json.dumps(frame) + "\n").encode())


async def test_live_discovery_execution_waits_for_native_result() -> None:
    wire, stderr = Wire(), Wire()
    adapter = ClaudeControlAdapter(
        stdout_pump=LinePump(wire.lines),
        stderr_pump=LinePump(stderr.lines),
        stdin=wire,
    )
    try:
        listing = asyncio.create_task(adapter.list_commands())
        request = await wire.sent.get()
        assert request["request"] == {"subtype": "initialize"}
        wire.emit(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": request["request_id"],
                    "response": {
                        "commands": [
                            {
                                "name": "custom",
                                "description": "My command",
                                "argumentHint": "<path>",
                            }
                        ]
                    },
                },
            }
        )
        catalog = await listing
        assert catalog[0]["argument_hint"] == "<path>"
        invocation = asyncio.create_task(adapter.execute_command("custom", "a  b"))
        sent = await wire.sent.get()
        assert sent["message"]["content"] == "/custom a  b"
        assert sent["uuid"]
        assert not invocation.done()
        wire.emit(
            {"type": "system", "subtype": "local_command_output", "content": "Readable result"}
        )
        wire.emit({"type": "result", "subtype": "success", "is_error": False})
        assert (await invocation)["text"] == "Readable result"
    finally:
        await adapter.aclose()


async def test_replacement_rejects_stale_selection_without_prompt_fallback() -> None:
    commands = ClaudeCommands()
    commands.replace([{"name": "old"}])
    commands.observe(
        {"type": "system", "subtype": "commands_changed", "commands": [{"name": "new"}]}
    )
    assert [item["name"] for item in commands.descriptors()] == ["new"]
    with pytest.raises(ControlError, match="catalog changed"):
        commands.begin("old", "")


async def test_terminal_commands_visible_but_not_sent() -> None:
    commands = ClaudeCommands()
    commands.observe(
        {
            "type": "system",
            "subtype": "init",
            "slash_commands": ["local"],
            "terminal_slash_commands": ["local"],
        }
    )
    assert commands.descriptors()[0]["available"] is False
    with pytest.raises(ControlError, match="terminal"):
        commands.begin("local", "")


async def test_timeout_keeps_slot_until_native_result() -> None:
    commands = ClaudeCommands()
    commands.replace(["custom"])
    _, _, result = commands.begin("custom", "")
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(result), timeout=0.001)
    with pytest.raises(ControlError, match="in progress"):
        commands.begin("custom", "")
    commands.observe({"type": "result", "subtype": "success"})
    assert (await result)["kind"] == "notice"


async def test_native_error_is_not_completed() -> None:
    commands = ClaudeCommands()
    commands.replace(["custom"])
    _, _, result = commands.begin("custom", "")
    commands.observe(
        {"type": "result", "subtype": "error_during_execution", "errors": ["Unavailable here"]}
    )
    with pytest.raises(ControlError, match="Unavailable here"):
        await result


async def test_close_fails_pending_command_and_reset_updates_identity() -> None:
    wire, stderr = Wire(), Wire()
    identities: list[HarnessSessionId] = []
    adapter = ClaudeControlAdapter(
        stdout_pump=LinePump(wire.lines),
        stderr_pump=LinePump(stderr.lines),
        stdin=wire,
        on_identity=identities.append,
        on_conversation_reset=identities.append,
    )
    adapter._dispatch(
        {"type": "system", "subtype": "init", "session_id": "old", "slash_commands": ["custom"]}
    )
    invocation = asyncio.create_task(adapter.execute_command("custom", ""))
    await wire.sent.get()
    adapter._dispatch({"type": "conversation_reset", "new_conversation_id": "new"})
    assert identities == ["old", "new"]
    await adapter.aclose()
    with pytest.raises(ControlTransportError, match="unknown"):
        await invocation


async def test_structured_report_survives_native_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized = {
        "kind": "fields",
        "title": "Context usage",
        "fields": [
            {"label": "Model", "value": "native-model"},
            {"label": "Tokens used", "value": 123},
        ],
        "items": [{"title": "Messages", "description": "100 tokens"}],
        "text": "Native report",
    }
    monkeypatch.setattr(
        "mandri.runtime.control.claude_commands.command_result",
        lambda frame: normalized if "context_usage" in frame else None,
    )
    commands = ClaudeCommands()
    commands.replace(["custom"])
    _, _, result = commands.begin("custom", "")
    commands.observe({"type": "assistant", "context_usage": {"total_tokens": 123}})
    assert not result.done()
    commands.observe({"type": "result", "subtype": "success", "result": "Text report"})
    assert await result == normalized


async def test_native_result_text_is_preserved_without_assistant_event() -> None:
    commands = ClaudeCommands()
    commands.replace(["custom"])
    _, _, result = commands.begin("custom", "")
    commands.observe({"type": "result", "subtype": "success", "result": "Native final answer"})
    assert (await result)["text"] == "Native final answer"


async def test_synthetic_assistant_command_text_fills_empty_terminal_result() -> None:
    commands = ClaudeCommands()
    commands.replace(["custom"])
    _, _, result = commands.begin("custom", "")
    commands.observe(
        {
            "type": "assistant",
            "message": {
                "model": "<synthetic>",
                "content": [{"type": "text", "text": "Native command response"}],
            },
        }
    )
    commands.observe({"type": "result", "subtype": "success", "result": ""})
    assert (await result)["text"] == "Native command response"


async def test_assistant_fallback_excludes_thinking_tools_subagents_and_meta() -> None:
    commands = ClaudeCommands()
    commands.replace(["custom"])
    _, _, result = commands.begin("custom", "")
    commands.observe(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "Private reasoning"},
                    {"type": "tool_use", "name": "Read", "input": {"path": "file"}},
                    {"type": "text", "text": "Main answer"},
                ]
            },
        }
    )
    commands.observe(
        {
            "type": "assistant",
            "parent_tool_use_id": "agent-1",
            "message": {"content": "Child answer"},
        }
    )
    commands.observe(
        {"type": "assistant", "isMeta": True, "message": {"content": "Injected instructions"}}
    )
    commands.observe({"type": "result", "subtype": "success"})
    assert (await result)["text"] == "Main answer"


async def test_explicit_native_result_takes_precedence_over_assistant_fallback() -> None:
    commands = ClaudeCommands()
    commands.replace(["custom"])
    _, _, result = commands.begin("custom", "")
    commands.observe({"type": "assistant", "message": {"content": "Intermediate explanation"}})
    commands.observe({"type": "result", "subtype": "success", "result": "Final native output"})
    assert (await result)["text"] == "Final native output"


async def test_empty_final_assistant_does_not_present_intermediate_text_as_answer() -> None:
    commands = ClaudeCommands()
    commands.replace(["custom"])
    _, _, result = commands.begin("custom", "")
    commands.observe(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Let me inspect the project"}]},
        }
    )
    commands.observe({"type": "assistant", "message": {"content": [{"type": "text", "text": ""}]}})
    commands.observe({"type": "result", "subtype": "success", "result": ""})
    output = await result
    assert output["kind"] == "notice"
    assert "without a textual response" in output["text"]
    assert "inspect" not in output["text"]


async def test_visible_assistant_fallback_is_classified_as_transcript() -> None:
    commands = ClaudeCommands()
    commands.replace(["custom"])
    _, _, result = commands.begin("custom", "")
    commands.observe(
        {"type": "assistant", "message": {"model": "native-model", "content": "Final answer"}}
    )
    commands.observe({"type": "result", "subtype": "success"})
    output = await result
    assert output["kind"] == "transcript"
    assert output["text"] == "Final answer"


async def test_native_result_repeating_visible_assistant_points_to_transcript() -> None:
    commands = ClaudeCommands()
    commands.replace(["custom"])
    _, _, result = commands.begin("custom", "")
    commands.observe(
        {"type": "assistant", "message": {"model": "native-model", "content": "Final answer"}}
    )
    commands.observe({"type": "result", "subtype": "success", "result": "Final answer"})
    output = await result
    assert output["kind"] == "transcript"
    assert output["text"] == "Final answer"


async def test_local_command_output_is_not_discarded_when_result_matches_assistant() -> None:
    commands = ClaudeCommands()
    commands.replace(["custom"])
    _, _, result = commands.begin("custom", "")
    commands.observe(
        {"type": "system", "subtype": "local_command_output", "content": "Local output"}
    )
    commands.observe(
        {"type": "assistant", "message": {"model": "native-model", "content": "Final answer"}}
    )
    commands.observe({"type": "result", "subtype": "success", "result": "Final answer"})
    output = await result
    assert output["kind"] == "text"
    assert output["text"] == "Local output\n\nFinal answer"
