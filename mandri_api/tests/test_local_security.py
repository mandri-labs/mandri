import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from mandri.api.local_security import LocalSecurity
from starlette.websockets import WebSocketDisconnect

TOKEN = "synthetic-local-access-token-for-tests"


@pytest.fixture
def client():
    app = FastAPI()
    app.add_middleware(
        LocalSecurity, hosts=["127.0.0.1"], origins=["tauri://localhost"], token=TOKEN
    )

    @app.get("/data")
    async def data():
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept(subprotocol=websocket.scope.get("mandri.subprotocol"))
        await websocket.send_json({"ok": True})

    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client


@pytest.mark.parametrize(
    "headers,status",
    [
        ({}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Authorization": f"Bearer {TOKEN}"}, 200),
        ({"Authorization": f"Bearer {TOKEN}", "Host": "attacker.example"}, 403),
        ({"Authorization": f"Bearer {TOKEN}", "Origin": "https://attacker.example"}, 403),
        ({"Authorization": f"Bearer {TOKEN}", "Origin": "null"}, 403),
        ({"Authorization": f"Bearer {TOKEN}", "Origin": "tauri://localhost"}, 200),
    ],
)
def test_http_access(client, headers, status):
    assert client.get("/data", headers=headers).status_code == status


def test_websocket_requires_token(client):
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("ws://127.0.0.1/ws"):
        pass


def test_websocket_accepts_private_subprotocol(client):
    with client.websocket_connect(
        "ws://127.0.0.1/ws", subprotocols=["mandri", f"mandri-token.{TOKEN}"]
    ) as socket:
        assert socket.receive_json() == {"ok": True}


def test_websocket_rejects_foreign_origin(client):
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(
        "ws://127.0.0.1/ws",
        subprotocols=["mandri", f"mandri-token.{TOKEN}"],
            headers={"Origin": "https://attacker.example"},
        ),
    ):
        pass
