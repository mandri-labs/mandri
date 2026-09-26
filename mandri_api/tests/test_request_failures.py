import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.api.actions import build_action_registry
from mandri.api.ws import _handle_request, _send
from mandri.core.protocol.errors import ProtocolErrorCode
from mandri.core.protocol.frames import RequestFrame
from mandri.core.protocol.registry import ActionRegistry
from mandri.runtime.commands import CommandService
from mandri.sessions.errors import SessionConflictError
from starlette.websockets import WebSocketState

SESSION_ID = "00000000-0000-0000-0000-000000000001"


def prompt_frame():
    return RequestFrame(
        type="request",
        op_id="request-1",
        action="session.prompt",
        params={"session_id": SESSION_ID, "content": "Hello"},
    )


async def test_native_model_conflict_returns_correlated_terminal_error():
    runtime = SimpleNamespace(
        commands=Mock(spec=CommandService),
        send_session_prompt=AsyncMock(side_effect=SessionConflictError("Wait for native identity"))
    )
    response = await build_action_registry(runtime).handle(prompt_frame())
    assert response.op_id == "request-1"
    assert not response.ok
    assert response.error.code is ProtocolErrorCode.SESSION_CONFLICT


async def test_unexpected_failure_is_correlated_sanitized_and_releases_operation():
    registry = ActionRegistry()
    handler = AsyncMock(
        side_effect=[RuntimeError("private provider detail"), {"state": "accepted"}]
    )
    registry.register("session.prompt", handler)
    failed = await registry.handle(prompt_frame())
    assert failed.op_id == "request-1"
    assert not failed.ok
    assert failed.error.code is ProtocolErrorCode.INTERNAL_ERROR
    assert "private" not in failed.model_dump_json()
    assert (await registry.handle(prompt_frame())).ok


async def test_cancellation_propagates_and_releases_operation():
    registry = ActionRegistry()
    registry.register("session.prompt", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await registry.handle(prompt_frame())
    registry.register("session.prompt", AsyncMock(return_value={}))
    assert (await registry.handle(prompt_frame())).ok


async def test_unavailable_actions_return_correlated_response():
    socket = SimpleNamespace(application_state=WebSocketState.CONNECTED, send_json=AsyncMock())
    await _handle_request(socket, SimpleNamespace(actions=None), prompt_frame())
    response = socket.send_json.call_args.args[0]
    assert response["type"] == "response"
    assert response["op_id"] == "request-1"
    assert response["ok"] is False


async def test_disconnect_during_send_is_tolerated_but_other_runtime_errors_propagate():
    socket = SimpleNamespace(application_state=WebSocketState.CONNECTED)

    async def disconnect(frame):
        socket.application_state = WebSocketState.DISCONNECTED
        raise RuntimeError("closed")

    socket.send_json = disconnect
    await _send(socket, {"type": "response", "op_id": "request-1", "ok": True, "result": {}})
    socket.application_state = WebSocketState.CONNECTED
    socket.send_json = AsyncMock(side_effect=RuntimeError("unexpected send failure"))
    with pytest.raises(RuntimeError, match="unexpected send failure"):
        await _send(socket, {"type": "response", "op_id": "request-1", "ok": True, "result": {}})
