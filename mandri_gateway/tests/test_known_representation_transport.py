import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from mandri.gateway.privacy_protocol import GatewayProtocol
from mandri.gateway.privacy_response import restore_sse
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope
from mandri.gateway.surrogate.types import Mapping


@pytest.mark.parametrize("width", [1, 3, 7])
async def test_actual_sse_deltas_restore_known_encodings_and_escaped_tool_arguments(
    tmp_path, width
):
    original, alias = "alice.smith@private.test", "julia.jones@fictional.com"
    engine = SurrogateEngine(
        SurrogateScope("transport", mappings=[Mapping("email", original, alias)])
    )
    encoded = base64.b64encode(original.encode()).decode()
    protected = engine.protect_text(encoded)
    arguments = json.dumps({"email": alias, "encoded": protected, "note": 'quoted "value"'})
    frames = []
    for index, char in enumerate(arguments):
        call = {"index": 0, "function": {"arguments": char}}
        if index == 0:
            call.update({"id": "call_fixture", "type": "function"})
            call["function"]["name"] = "read_fixture"
        frames.append(
            {"id": "chunk_fixture", "choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}
        )
    frames.append(
        {
            "id": "chunk_fixture",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
    )
    wire = (
        b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)
        + b"data: [DONE]\n\n"
    )

    class FixtureHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            for offset in range(0, len(wire), width):
                self.wfile.write(wire[offset : offset + width])
                self.wfile.flush()

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        async with (
            httpx.AsyncClient(trust_env=False) as client,
            client.stream("GET", f"http://127.0.0.1:{server.server_port}") as response,
        ):
            restored = b"".join(
                [
                    chunk
                    async for chunk in restore_sse(
                        response.aiter_bytes(chunk_size=width), engine, GatewayProtocol.CHAT
                    )
                ]
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    values = [
        json.loads(frame[6:])
        for frame in restored.decode().split("\n\n")
        if frame.startswith("data: {")
    ]
    actual = "".join(
        call["function"].get("arguments", "")
        for frame in values
        for choice in frame.get("choices", [])
        for call in choice.get("delta", {}).get("tool_calls", [])
    )
    assert json.loads(actual) == {"email": original, "encoded": encoded, "note": 'quoted "value"'}
    (tmp_path / "provider.sse").write_bytes(wire)
    (tmp_path / "gateway.sse").write_bytes(restored)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "provider_sha256": hashlib.sha256(wire).hexdigest(),
                "gateway_sha256": hashlib.sha256(restored).hexdigest(),
                "delta_events": len(frames),
                "transport_chunk_bytes": width,
            }
        ),
        encoding="utf-8",
    )
