"""Websocket subscription replays outstanding approvals."""

import asyncio
import contextlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mandri.api.deps import LifespanState
from mandri.api.ws import _subscribe
from mandri.core.hub import Hub
from mandri.core.ids import ApprovalKind, HarnessKind, RawEvent, SessionId
from mandri.core.protocol.frames import SubscribeFrame
from mandri.runtime.service import RuntimeService
from starlette.websockets import WebSocketState


async def test_subscribe_recovers_pending_request_without_history_cursor():
    hub = Hub()
    runtime = RuntimeService(harness_commands={}, hub=hub)
    request = await runtime._approvals.register(
        SessionId("s1"),
        HarnessKind.CLAUDE,
        RawEvent(json.dumps({"type": "control_request", "request_id": "r1"})),
        "r1",
        ApprovalKind.COMMAND_EXECUTION,
        120,
    )
    delivered = asyncio.Event()
    frames = []

    async def send(frame):
        frames.append(frame)
        if frame.get("type") == "approval.pending":
            delivered.set()

    websocket = SimpleNamespace(
        send_json=send, close=AsyncMock(), application_state=WebSocketState.CONNECTED
    )
    state = LifespanState(hub=hub, runtime=runtime)
    subscriptions, tasks = {}, {}
    try:
        await _subscribe(
            websocket,
            hub,
            state,
            SubscribeFrame(op="subscribe", topic="session.s1"),
            subscriptions,
            tasks,
        )
        await asyncio.wait_for(delivered.wait(), 1)
        assert frames[0]["op"] == "subscribed"
        assert frames[1]["approval_id"] == request.id
        assert frames[1]["deadline"] == request.deadline
    finally:
        for task in tasks.values():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for handle in subscriptions.values():
            hub.unsubscribe(handle)
