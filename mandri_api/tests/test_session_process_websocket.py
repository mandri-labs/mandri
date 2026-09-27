from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from mandri.api.app import create_app
from mandri.core.hub import Hub, Topic
from mandri.runtime.service import RuntimeService


def test_process_stop_keeps_socket_and_session_sequence_for_resume():
    app = create_app()
    hub = Hub()
    runtime = RuntimeService({}, hub=hub)
    session_id = "00000000-0000-0000-0000-000000000001"
    topic = f"session.{session_id}"
    runtime.registry.mark_live(
        session_id, SimpleNamespace(returncode=None, stop=AsyncMock(return_value=0)), "pi"
    )
    with TestClient(app) as client:
        state = app.state.lifespan
        state.hub = hub
        state.runtime = runtime
        state.lifetime = runtime
        with client.websocket_connect("/v1/ws") as socket:
            socket.send_json({"op": "subscribe", "topic": topic})
            assert socket.receive_json()["op"] == "subscribed"
            socket.send_json({"op": "subscribe", "topic": "gateway.events"})
            assert socket.receive_json()["op"] == "subscribed"
            client.portal.call(runtime.stop_session, session_id)
            stopped = socket.receive_json()
            assert stopped["type"] == "session_stopped"
            assert stopped["raw"]["session_id"] == session_id
            client.portal.call(
                hub.publish, Topic(topic), {"source": "pi", "raw": {"type": "agent_start"}, "ts": 1}
            )
            resumed = socket.receive_json()
            assert resumed["seq"] == stopped["seq"] + 1
            assert resumed["raw"]["type"] == "agent_start"
            socket.send_json({"op": "unsubscribe", "topic": "gateway.events"})
            assert socket.receive_json() == {"op": "unsubscribed", "topic": "gateway.events"}
