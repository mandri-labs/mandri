import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mandri.api import ws as ws_module
from mandri.api.ws import _handle_text, _send
from mandri.core.hub import Hub, Topic
from mandri.core.protocol.frames import ResponseFrame
from starlette.websockets import WebSocket, WebSocketState


async def test_feed_does_not_read_again_after_send_disconnect(monkeypatch):
    websocket = SimpleNamespace(
        app=object(),
        scope={},
        application_state=WebSocketState.CONNECTED,
        accept=AsyncMock(),
        receive_text=AsyncMock(return_value="{}"),
    )
    state = SimpleNamespace(hub=Hub(), heartbeat_configured=True)

    async def disconnect(*args):
        websocket.application_state = WebSocketState.DISCONNECTED

    cleanup = AsyncMock()
    monkeypatch.setattr(ws_module, "app_state", lambda app: state)
    monkeypatch.setattr(ws_module, "_handle_text", disconnect)
    monkeypatch.setattr(ws_module, "_cleanup", cleanup)
    await ws_module.ws_feed(websocket)
    await asyncio.sleep(0)
    websocket.receive_text.assert_awaited_once()
    cleanup.assert_awaited_once()


async def test_long_request_does_not_block_pongs_or_other_requests():
    gate = asyncio.Event()
    started = asyncio.Event()
    finished = asyncio.Event()

    async def handle(frame):
        if frame.op_id == "slow":
            started.set()
            await gate.wait()
        return ResponseFrame(type="response", op_id=frame.op_id, ok=True, result={})

    async def send(frame):
        if frame.get("op_id") == "slow":
            finished.set()

    websocket = SimpleNamespace(
        application_state=WebSocketState.CONNECTED, send_json=AsyncMock(side_effect=send)
    )
    state = SimpleNamespace(actions=SimpleNamespace(handle=handle))
    hub = Hub()
    subscription = hub.subscribe(Topic("sessions.all"))
    subscription.misses = 1
    subscriptions = {"sessions.all": subscription}
    try:
        await _handle_text(
            websocket,
            hub,
            state,
            json.dumps(
                {
                    "type": "request",
                    "op_id": "slow",
                    "action": "session.prompt",
                    "params": {"session_id": "s1", "content": "Hello"},
                }
            ),
            subscriptions,
            {},
        )
        await asyncio.wait_for(started.wait(), 1)
        await _handle_text(websocket, hub, state, '{"type":"pong"}', subscriptions, {})
        assert subscription.misses == 0
        await _handle_text(
            websocket,
            hub,
            state,
            json.dumps(
                {
                    "type": "request",
                    "op_id": "fast",
                    "action": "session.list",
                    "params": {},
                }
            ),
            subscriptions,
            {},
        )
        await asyncio.sleep(0)
        assert websocket.send_json.call_args.args[0]["op_id"] == "fast"
        assert not finished.is_set()
    finally:
        gate.set()
        await asyncio.wait_for(finished.wait(), 1)
        await hub.close_all()


async def test_late_response_after_disconnect_is_not_sent():
    websocket = SimpleNamespace(
        application_state=WebSocketState.DISCONNECTED, send_json=AsyncMock()
    )
    await _send(websocket, {"type": "response", "op_id": "late", "ok": True, "result": {}})
    websocket.send_json.assert_not_awaited()


async def test_completed_relays_close_the_shared_socket_once(caplog):
    send = AsyncMock()
    websocket = WebSocket(
        {"type": "websocket"}, AsyncMock(return_value={"type": "websocket.connect"}), send
    )
    await websocket.accept()
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    for _ in range(3):
        ws_module._close_after_relay(websocket, task)
    await asyncio.gather(*ws_module._background_tasks)
    await asyncio.sleep(0)
    closes = [call for call in send.call_args_list if call.args[0]["type"] == "websocket.close"]
    assert len(closes) == 1
    assert not caplog.records
