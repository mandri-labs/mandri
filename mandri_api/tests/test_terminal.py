import asyncio
import sys
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from mandri.api.app import create_app
from mandri.api.deps import LifespanState
from mandri.core.types.execution import ProtectionError
from mandri.runtime.terminal_process import spawn_terminal

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminal required")


def client_for(context):
    app = create_app()
    app.state.lifespan = LifespanState(
        runtime=SimpleNamespace(terminal_session=context),
        gateway=SimpleNamespace(reasoning_catalog=None),
    )
    return TestClient(app)


def test_terminal_websocket_relays_bytes_resize_and_native_exit():
    stopped = threading.Event()
    observed = []

    @asynccontextmanager
    async def session(spec):
        observed.append(spec)
        process = await spawn_terminal(
            [
                sys.executable,
                "-c",
                "import os,sys; print('READY', flush=True); line=input(); "
                "print('ECHO', line, os.get_terminal_size().columns, flush=True); sys.exit(19)",
            ],
            spec,
        )
        try:
            yield SimpleNamespace(id="session", process=process)
        finally:
            await process.stop()
            stopped.set()

    with client_for(session).websocket_connect("/v1/runtime/terminal") as websocket:
        websocket.send_json(
            {
                "harness": "codex",
                "model": "p/model",
                "cwd": "/workspace",
                "execution_backend": "docker",
                "privacy_mode": "surrogate",
            }
        )
        assert websocket.receive_json() == {"type": "started", "session_id": "session"}
        assert websocket.receive_bytes() == b"READY\r\n"
        websocket.send_json({"rows": 32, "columns": 113})
        websocket.send_bytes(b"canary-input\n")
        output = b""
        while True:
            frame = websocket.receive()
            if frame.get("text"):
                assert frame["text"] == '{"type":"exit","code":19}'
                break
            output += frame["bytes"]
        assert b"ECHO canary-input 113\r\n" in output
    assert stopped.wait(5)
    assert observed[0].privacy_mode.value == "surrogate"


def test_disconnected_cli_cancels_inflight_launch():
    started = threading.Event()
    cancelled = threading.Event()

    @asynccontextmanager
    async def session(spec):
        started.set()
        try:
            await asyncio.sleep(60)
            yield
        finally:
            cancelled.set()

    with client_for(session).websocket_connect("/v1/runtime/terminal") as websocket:
        websocket.send_json({"harness": "codex", "model": "p/model", "cwd": "/workspace"})
        assert started.wait(5)
    assert cancelled.wait(5)


def test_terminal_reports_privacy_failure_without_starting_session():
    @asynccontextmanager
    async def session(spec):
        raise ProtectionError("privacy_key_unavailable", "Privacy key unavailable")
        yield

    with client_for(session).websocket_connect("/v1/runtime/terminal") as websocket:
        websocket.send_json({"harness": "codex", "model": "p/model", "cwd": "/workspace"})
        assert websocket.receive_json() == {"type": "error", "message": "Privacy key unavailable"}
