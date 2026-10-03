import asyncio
import json
from unittest.mock import Mock

import pytest
from mandri.core.ids import HarnessSessionId
from mandri.runtime.control.errors import (
    ControlTransportError,
    PromptDeliveryFailedError,
    PromptDeliveryUnknownError,
    ThreadOwnershipError,
)
from mandri.runtime.control.pi import PiControlAdapter
from mandri.runtime.pump import LinePump


class PiTransport:
    def __init__(self):
        self.output = asyncio.Queue()
        self.commands = asyncio.Queue()

    async def stream(self):
        while (value := await self.output.get()) is not None:
            yield value

    def write(self, payload):
        self.commands.put_nowait(json.loads(payload))

    async def drain(self):
        pass

    def emit(self, record):
        self.output.put_nowait((json.dumps(record, ensure_ascii=False) + "\n").encode())

    async def reply(self, data=None, *, success=True):
        command = await self.commands.get()
        self.emit(
            {
                "type": "response",
                "id": command["id"],
                "command": command["type"],
                "success": success,
                "data": data or {},
                "error": "rejected",
            }
        )
        return command

    def adapter(self, **kwargs):
        return PiControlAdapter(LinePump(self.stream), self, **kwargs)


async def initialize(transport, adapter, **extra):
    task = asyncio.create_task(adapter.capture_identity())
    await transport.reply({"sessionId": "pi-session", **extra})
    assert await task == "pi-session"


async def test_capture_uses_native_identity_and_rejects_wrong_resume():
    transport = PiTransport()
    control = transport.adapter(expected_session_id=HarnessSessionId("other"))
    task = asyncio.create_task(control.capture_identity())
    await transport.reply({"sessionId": "pi-session"})
    with pytest.raises(ThreadOwnershipError):
        await task
    await control.aclose()


async def test_out_of_order_responses_and_unicode_line_separator_are_preserved():
    transport = PiTransport()
    control = transport.adapter()
    first = asyncio.create_task(control._call("get_state", {}))
    second = asyncio.create_task(control._call("get_commands", {}))
    left, right = await transport.commands.get(), await transport.commands.get()
    for command in (right, left):
        transport.emit(
            {
                "id": command["id"],
                "type": "response",
                "command": command["type"],
                "success": True,
                "data": {"text": "alpha\u2028beta\u2029gamma"},
            }
        )
    assert await first == await second == {"text": "alpha\u2028beta\u2029gamma"}
    await control.aclose()


async def test_closed_stream_fails_pending_requests_and_cannot_restart_reader():
    transport = PiTransport()
    control = transport.adapter()
    request = asyncio.create_task(control._call("get_state", {}))
    await transport.commands.get()
    transport.output.put_nowait(None)
    with pytest.raises(ControlTransportError, match="closed"):
        await request
    with pytest.raises(ControlTransportError, match="closed"):
        await control._call("get_state", {})
    await control.aclose()


async def test_gateway_rejects_foreign_model_before_prompt():
    transport = PiTransport()
    control = transport.adapter(gateway_mode=True)
    await initialize(transport, control, model={"provider": "mandri", "id": "gateway"})
    request = asyncio.create_task(control.send_prompt("hello"))
    await transport.reply(
        {"sessionId": "pi-session", "model": {"provider": "external", "id": "model"}}
    )
    with pytest.raises(ControlTransportError, match="outside"):
        await request
    assert transport.commands.empty()
    await control.aclose()


async def test_dialog_response_uses_native_reference_and_preserves_text():
    transport = PiTransport()
    control = transport.adapter()
    await initialize(transport, control)
    transport.emit({"type": "extension_ui_request", "id": "dialog", "method": "input"})
    assert (await control.next_native_request()).native_request_ref == "dialog"
    assert not await control.answer_native_request("unknown", {"value": "wrong"})
    assert await control.answer_native_request("dialog", {"value": "  exact\ntext  "})
    assert await transport.commands.get() == {
        "type": "extension_ui_response",
        "id": "dialog",
        "value": "  exact\ntext  ",
    }
    assert not await control.answer_native_request("dialog", {"value": "duplicate"})
    await control.aclose()


async def test_expired_native_dialog_is_not_sent_to_pi():
    transport = PiTransport()
    control = transport.adapter()
    await initialize(transport, control)
    transport.emit(
        {"type": "extension_ui_request", "id": "expired", "method": "input", "timeout": 0}
    )
    await control.next_native_request()
    assert not await control.answer_native_request("expired", {"value": "late"})
    assert transport.commands.empty()
    await control.aclose()


async def test_interrupt_clears_queued_followups_before_abort():
    transport = PiTransport()
    control = transport.adapter()
    await initialize(transport, control)
    task = asyncio.create_task(control.interrupt())
    assert (await transport.reply())["type"] == "clear_queue"
    assert (await transport.reply())["type"] == "abort"
    assert await task
    await control.aclose()


async def test_session_change_notifies_reset_instead_of_initial_identity():
    initial, reset = Mock(), Mock()
    transport = PiTransport()
    control = transport.adapter(on_identity=initial, on_conversation_reset=reset)
    await initialize(transport, control)
    await control._update_identity({"sessionId": "new-session"})
    initial.assert_called_once_with("pi-session")
    reset.assert_called_once_with("new-session")
    await control.aclose()


async def test_concurrent_identity_callers_share_one_native_handshake():
    transport = PiTransport()
    identity = Mock()
    control = transport.adapter(on_identity=identity)
    first = asyncio.create_task(control.capture_identity())
    second = asyncio.create_task(control.capture_identity())
    await transport.reply({"sessionId": "pi-session"})
    assert await first == await second == "pi-session"
    assert transport.commands.empty()
    identity.assert_called_once_with("pi-session")
    await control.aclose()


async def test_unsolicited_startup_info_cannot_bypass_resume_identity_validation():
    transport = PiTransport()
    control = transport.adapter(expected_session_id=HarnessSessionId("expected"))
    pending = asyncio.create_task(control.capture_identity())
    await transport.commands.get()
    transport.emit({"type": "session_info_changed"})
    transport.emit(
        {
            "type": "response",
            "id": "1",
            "command": "get_state",
            "success": True,
            "data": {"sessionId": "wrong"},
        }
    )
    with pytest.raises(ThreadOwnershipError):
        await pending
    assert control._session_id is None
    assert transport.commands.empty()
    await control.aclose()


async def test_native_session_change_waits_for_persisted_identity_before_next_prompt():
    entered, release = asyncio.Event(), asyncio.Event()

    async def reset(native_id):
        assert native_id == "resumed-session"
        entered.set()
        await release.wait()

    transport = PiTransport()
    control = transport.adapter(on_conversation_reset=reset)
    await initialize(transport, control)
    prompt = asyncio.create_task(control.send_prompt("hello"))
    await transport.reply({"sessionId": "resumed-session"})
    await entered.wait()
    assert control._session_id == "pi-session"
    assert transport.commands.empty()
    assert not prompt.done()
    release.set()
    command = await transport.reply({"disposition": "started"})
    assert command["type"] == "prompt"
    await prompt
    assert control._session_id == "resumed-session"
    await control.aclose()


async def test_failed_session_reset_keeps_previous_identity_and_sends_no_prompt():
    async def reset(_native_id):
        raise ControlTransportError("Native session is already active")

    transport = PiTransport()
    control = transport.adapter(on_conversation_reset=reset)
    await initialize(transport, control)
    prompt = asyncio.create_task(control.send_prompt("hello"))
    await transport.reply({"sessionId": "claimed-session"})
    with pytest.raises(ControlTransportError, match="already active"):
        await prompt
    assert control._session_id == "pi-session"
    assert transport.commands.empty()
    await control.aclose()


async def test_gateway_session_switch_restores_mandri_model_and_effort_before_prompt():
    transport = PiTransport()
    reset = Mock()
    control = transport.adapter(gateway_mode=True, on_conversation_reset=reset)
    await initialize(transport, control, model={"provider": "mandri", "id": "gateway"})
    control._thinking_level = "medium"
    prompt = asyncio.create_task(control.send_prompt("hello"))
    await transport.reply(
        {"sessionId": "resumed-session", "model": {"provider": "native", "id": "model"}}
    )
    command = await transport.reply({"provider": "mandri", "id": "gateway"})
    assert command == {"id": "3", "type": "set_model", "provider": "mandri", "modelId": "gateway"}
    reset.assert_not_called()
    command = await transport.reply({"level": "medium"})
    assert command["type"] == "set_thinking_level" and command["level"] == "medium"
    assert (await transport.reply({"disposition": "started"}))["type"] == "prompt"
    await prompt
    reset.assert_called_once_with("resumed-session")
    await control.aclose()


async def test_native_model_callback_waits_for_persistence_and_ignores_initial_selection():
    entered, release = asyncio.Event(), asyncio.Event()
    changes = []

    async def changed(model, thinking):
        changes.append((model, thinking))
        entered.set()
        await release.wait()

    transport = PiTransport()
    control = transport.adapter(on_model_selection=changed)
    await initialize(
        transport, control, model={"provider": "native", "id": "first"}, thinkingLevel="low"
    )
    assert changes == []
    prompt = asyncio.create_task(control.send_prompt("hello"))
    await transport.reply(
        {
            "sessionId": "pi-session",
            "model": {"provider": "native", "id": "second"},
            "thinkingLevel": "high",
        }
    )
    await entered.wait()
    assert changes == [("native/second", "high")]
    assert transport.commands.empty()
    assert not prompt.done()
    release.set()
    await transport.reply({"disposition": "started"})
    await prompt
    await control._update_identity(
        {
            "sessionId": "pi-session",
            "model": {"provider": "native", "id": "second"},
            "thinkingLevel": "high",
        }
    )
    assert changes == [("native/second", "high")]
    await control.aclose()


async def test_native_thinking_only_change_notifies_selection_callback():
    changed = Mock()
    transport = PiTransport()
    control = transport.adapter(on_model_selection=changed)
    await initialize(
        transport, control, model={"provider": "native", "id": "model"}, thinkingLevel="low"
    )
    await control._update_identity(
        {
            "sessionId": "pi-session",
            "model": {"provider": "native", "id": "model"},
            "thinkingLevel": "off",
        }
    )
    changed.assert_called_once_with("native/model", "off")
    await control.aclose()


async def test_gateway_changes_never_notify_native_selection_callback():
    changed = Mock()
    transport = PiTransport()
    control = transport.adapter(gateway_mode=True, on_model_selection=changed)
    await initialize(
        transport, control, model={"provider": "mandri", "id": "gateway"}, thinkingLevel="low"
    )
    await control._update_identity(
        {
            "sessionId": "pi-session",
            "model": {"provider": "mandri", "id": "gateway"},
            "thinkingLevel": "high",
        }
    )
    changed.assert_not_called()
    await control.aclose()


async def test_permission_command_waits_for_correlated_extension_ack():
    transport = PiTransport()
    control = transport.adapter()
    await initialize(transport, control)
    change = asyncio.create_task(control.set_mode("plan"))
    await transport.reply({"commands": [{"name": "mandri-permissions", "source": "extension"}]})
    command = await transport.reply({"disposition": "handled"})
    name, mode, token = command["message"].split()
    assert (name, mode) == ("/mandri-permissions", "plan")
    control._dispatch(
        {
            "type": "extension_ui_request",
            "method": "setStatus",
            "statusKey": "_mandri_permissions",
            "statusText": "wrong:plan",
        }
    )
    assert not change.done()
    transport.emit(
        {
            "type": "extension_ui_request",
            "method": "setStatus",
            "statusKey": "_mandri_permissions",
            "statusText": f"{token}:plan",
        }
    )
    assert (await change).value == "mid_session_applied"
    await control.aclose()


async def test_missing_permission_extension_never_sends_a_prompt():
    transport = PiTransport()
    control = transport.adapter()
    await initialize(transport, control)
    change = asyncio.create_task(control.set_mode("plan"))
    await transport.reply({"commands": []})
    assert (await change).value == "requires_restart"
    assert transport.commands.empty()
    await control.aclose()


@pytest.mark.parametrize("failure", ["rejected", "closed", "identity_closed"])
async def test_prompt_response_failure_preserves_delivery_certainty(failure):
    transport = PiTransport()
    control = transport.adapter()
    await initialize(transport, control)
    task = asyncio.create_task(control.send_prompt("Synthetic message"))
    try:
        await transport.reply({"sessionId": "pi-session"})
        if failure == "rejected":
            command = await transport.reply(success=False)
        else:
            if failure == "identity_closed":
                command = await transport.reply({"disposition": "handled"})
                await transport.commands.get()
            else:
                command = await transport.commands.get()
            transport.output.put_nowait(None)
        assert command["type"] == "prompt"
        expected = (
            PromptDeliveryFailedError if failure == "rejected" else PromptDeliveryUnknownError
        )
        with pytest.raises(expected):
            await task
        assert transport.commands.empty()
    finally:
        await control.aclose()
