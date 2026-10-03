import asyncio

import pytest
from mandri.core.ports.control import PromptState
from mandri.core.types.prompt import UserPrompt
from mandri.runtime.control.claude import ClaudeControlAdapter
from mandri.runtime.pump import LinePump

from mandri_runtime.tests.test_claude_commands import Wire


def adapter_for(wire, stderr, **kwargs):
    return ClaudeControlAdapter(
        stdout_pump=LinePump(wire.lines),
        stderr_pump=LinePump(stderr.lines),
        stdin=wire,
        **kwargs,
    )


def init(wire, sid="session"):
    wire.emit({"type": "system", "subtype": "init", "session_id": sid})


def reply(wire, request, subtype="success"):
    wire.emit(
        {
            "type": "control_response",
            "response": {
                "subtype": subtype,
                "request_id": request["request_id"],
                "response": {"title": "Session model selection"},
            },
        }
    )


@pytest.mark.parametrize(
    "content", ["Fix the session model selector", UserPrompt("Fix the session model selector")]
)
async def test_native_title_generated_once_and_persisted_without_blocking_prompt(content):
    wire, stderr = Wire(), Wire()
    adapter = adapter_for(wire, stderr)
    try:
        assert (await adapter.send_prompt(content)).state is PromptState.QUEUED
        assert (await wire.sent.get())["type"] == "user"
        assert wire.sent.empty()
        init(wire)
        request = await asyncio.wait_for(wire.sent.get(), 1)
        assert request["request"] == {
            "subtype": "generate_session_title",
            "description": "Fix the session model selector",
            "persist": True,
        }
        assert (await adapter.send_prompt("continue")).state is PromptState.QUEUED
        assert (await wire.sent.get())["type"] == "user"
        reply(wire, request)
        await adapter._title_task
        assert wire.sent.empty()
        await adapter.send_prompt("Try again")
        assert (await wire.sent.get())["type"] == "user"
        assert wire.sent.empty()
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("failure", ["rejection", "timeout"])
async def test_title_failure_does_not_break_chat_or_retry_every_message(monkeypatch, failure):
    monkeypatch.setattr("mandri.runtime.control.claude._ACK_TIMEOUT_S", 0.01)
    wire, stderr = Wire(), Wire()
    adapter = adapter_for(wire, stderr)
    try:
        init(wire)
        await adapter.send_prompt("Fix the session model selector")
        await wire.sent.get()
        request = await asyncio.wait_for(wire.sent.get(), 1)
        if failure == "rejection":
            reply(wire, request, "error")
        await adapter._title_task
        assert not adapter._acks
        assert (await adapter.send_prompt("continue")).state is PromptState.QUEUED
        assert (await wire.sent.get())["type"] == "user"
        assert wire.sent.empty()
    finally:
        await adapter.aclose()


async def test_resume_keeps_title_and_reset_enables_title_for_new_conversation():
    wire, stderr = Wire(), Wire()
    adapter = adapter_for(wire, stderr, resumed=True)
    try:
        init(wire)
        await adapter.send_prompt("continue")
        assert (await wire.sent.get())["type"] == "user"
        assert adapter._title_task is None
        await adapter.capture_identity()
        adapter._dispatch({"type": "conversation_reset", "new_conversation_id": "new-session"})
        await adapter.send_prompt("Fix the session model selector")
        await wire.sent.get()
        request = await asyncio.wait_for(wire.sent.get(), 1)
        assert request["request"]["description"] == "Fix the session model selector"
        reply(wire, request)
        await adapter._title_task
    finally:
        await adapter.aclose()


async def test_close_cancels_pending_title_request():
    wire, stderr = Wire(), Wire()
    adapter = adapter_for(wire, stderr)
    init(wire)
    await adapter.send_prompt("Fix the session model selector")
    await wire.sent.get()
    await asyncio.wait_for(wire.sent.get(), 1)
    task = adapter._title_task
    await adapter.aclose()
    assert task.cancelled()
    assert not adapter._acks


async def test_empty_prompt_and_slash_command_do_not_consume_title_attempt():
    wire, stderr = Wire(), Wire()
    adapter = adapter_for(wire, stderr)
    try:
        init(wire)
        for prompt in (" ", "/help"):
            await adapter.send_prompt(prompt)
            await wire.sent.get()
            assert adapter._title_task is None
        await adapter.send_prompt("Fix the session model selector")
        await wire.sent.get()
        request = await asyncio.wait_for(wire.sent.get(), 1)
        reply(wire, request)
        await adapter._title_task
    finally:
        await adapter.aclose()
