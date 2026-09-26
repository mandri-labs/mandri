import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient
from mandri.api.actions import build_action_registry
from mandri.api.app import create_app
from mandri.core.hub import Hub
from mandri.core.ports.control import PromptOutcome, PromptState
from mandri.runtime.service import RuntimeService


def test_pending_model_restart_acknowledges_prompt_without_reconnecting():
    app = create_app()
    hub = Hub()
    runtime = RuntimeService({}, hub=hub)
    session_id = "00000000-0000-0000-0000-000000000001"
    runtime.registry.mark_live(
        session_id, SimpleNamespace(returncode=None, stop=AsyncMock(return_value=0)), "agy"
    )
    runtime.is_busy = Mock(return_value=False)
    runtime._models.needs_restart = AsyncMock(return_value=True)
    runtime._session_state(session_id).control = SimpleNamespace(aclose=AsyncMock())
    control = SimpleNamespace(send_prompt=AsyncMock(return_value=PromptOutcome(PromptState.QUEUED)))

    async def resume(sid, **kwargs):
        await asyncio.sleep(0.01)
        runtime.registry.mark_live(sid, SimpleNamespace(returncode=None), "agy")
        runtime._session_state(sid).control = control

    runtime._resume_session = AsyncMock(side_effect=resume)
    with TestClient(app) as client:
        state = app.state.lifespan
        state.hub = hub
        state.runtime = runtime
        state.actions = build_action_registry(runtime)
        with client.websocket_connect("/v1/ws") as socket:
            socket.send_json({"op": "subscribe", "topic": f"session.{session_id}"})
            assert socket.receive_json()["op"] == "subscribed"
            socket.send_json(
                {
                    "type": "request",
                    "op_id": "resume-and-send",
                    "action": "session.prompt",
                    "params": {"session_id": session_id, "content": "continue"},
                }
            )
            while True:
                frame = socket.receive_json()
                if frame.get("op_id") == "resume-and-send":
                    assert frame["ok"]
                    assert frame["result"]["state"] == "queued"
                    break
            socket.send_json({"op": "subscribe", "topic": "gateway.events"})
            while True:
                frame = socket.receive_json()
                if frame.get("op") == "subscribed":
                    assert frame["topic"] == "gateway.events"
                    break
    control.send_prompt.assert_awaited_once_with("continue")
