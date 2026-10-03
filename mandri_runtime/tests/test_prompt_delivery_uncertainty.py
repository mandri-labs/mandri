import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from mandri.runtime.control.agy import AgyControlAdapter
from mandri.runtime.control.claude import ClaudeControlAdapter
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.control.errors import (
    ControlTransportError,
    PromptDeliveryFailedError,
    PromptDeliveryUnknownError,
)
from mandri.runtime.control.opencode import OpencodeControlAdapter
from mandri.runtime.pump import LinePump


class Transport:
    def __init__(self):
        self.output = asyncio.Queue()
        self.input = asyncio.Queue()

    async def stream(self):
        while (value := await self.output.get()) is not None:
            yield value

    def write(self, payload):
        self.input.put_nowait(json.loads(payload))

    async def drain(self):
        pass


@pytest.mark.parametrize("steering", [False, True])
@pytest.mark.parametrize("outcome", ["eof", "timeout", "rejection"])
async def test_codex_distinguishes_rejection_from_lost_response(monkeypatch, steering, outcome):
    monkeypatch.setattr("mandri.runtime.control.codex.RPC_RESPONSE_TIMEOUT_SECONDS", 0.05)
    transport = Transport()
    control = CodexControlAdapter(LinePump(transport.stream), transport)
    control._thread_id = "native"
    if steering:
        control._active_turn_id = "turn"
    task = asyncio.create_task(control.send_prompt("Synthetic message"))
    try:
        frame = await asyncio.wait_for(transport.input.get(), 1)
        assert frame["method"] == ("turn/steer" if steering else "turn/start")
        assert frame["params"]["input"][0]["text"] == "Synthetic message"
        if outcome == "eof":
            transport.output.put_nowait(None)
        elif outcome == "rejection":
            transport.output.put_nowait((json.dumps({
                "id": frame["id"], "error": {"message": "Rejected by harness"}
            }) + "\n").encode())
        expected = (
            PromptDeliveryFailedError if outcome == "rejection" else PromptDeliveryUnknownError
        )
        with pytest.raises(expected):
            await task
        assert transport.input.empty()
    finally:
        await control.aclose()


async def test_claude_failed_write_is_not_a_definite_rejection():
    control = object.__new__(ClaudeControlAdapter)
    control._ensure_pump = lambda: None
    control._commands = type("Commands", (), {"pending": False})()
    control._send = AsyncMock(side_effect=ControlTransportError("drain failed"))
    with pytest.raises(PromptDeliveryUnknownError):
        await control.send_prompt("Synthetic message")
    control._send.assert_awaited_once()


async def test_opencode_preparation_and_prompt_transport_failures_are_distinct():
    control = object.__new__(OpencodeControlAdapter)
    control._session_id = "native"
    control._request = AsyncMock(side_effect=ControlTransportError("model request failed"))
    with pytest.raises(ControlTransportError):
        await control.send_prompt("Synthetic message")
    control._request = AsyncMock(side_effect=[
        type("Response", (), {"status_code": 200})(), ControlTransportError("prompt reply lost")
    ])
    with pytest.raises(PromptDeliveryUnknownError):
        await control.send_prompt("Synthetic message")
    assert control._request.await_count == 2


async def test_agy_failed_drain_preserves_uncertainty():
    control = object.__new__(AgyControlAdapter)
    control.capture_identity = AsyncMock(return_value="native")
    control._stdin = Transport()
    control._stdin.drain = AsyncMock(side_effect=OSError("closed after write"))
    with pytest.raises(PromptDeliveryUnknownError):
        await control.send_prompt("Synthetic message")
    assert (await control._stdin.input.get())["message"]["content"] == "Synthetic message"


@pytest.mark.parametrize("status", [400, 500, 502, 504])
async def test_opencode_distinguishes_rejection_from_server_failure(status):
    control = object.__new__(OpencodeControlAdapter)
    control._session_id = "native"
    control._request = AsyncMock(side_effect=[
        type("Response", (), {"status_code": 200})(),
        type("Response", (), {"status_code": status})(),
    ])
    expected = PromptDeliveryFailedError if status < 500 else PromptDeliveryUnknownError
    with pytest.raises(expected):
        await control.send_prompt("Synthetic message")
    assert control._request.await_count == 2
