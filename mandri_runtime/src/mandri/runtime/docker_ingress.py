import asyncio
import contextlib
import hmac
import os
import socket
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import uvicorn
from mandri.runtime.errors.docker import DockerExecutionError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
_MAX_BODY = 20 * 1024 * 1024


class WorkerIngress:
    def __init__(self, gateway_port: int, route_id: str | None, token: str | None) -> None:
        self._gateway_port = gateway_port
        self._route_prefix = f"/v1/gateway/llm/{route_id}/" if route_id else None
        self._route_token = token
        self._hook_path: str | None = None
        self._hook_token: str | None = None
        self._client = httpx.AsyncClient(timeout=None, trust_env=False)
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._socket: socket.socket | None = None
        self.port: int | None = None
        self.socket_path: Path | None = None
        self.app = Starlette(routes=[Route("/{path:path}", self.forward, methods=["GET", "POST"])])

    def set_hook(self, session_id: str, token: str) -> None:
        self._hook_path = f"/v1/runtime/agy/{session_id}/hook"
        self._hook_token = token

    def authenticates(self, request: Request) -> bool:
        path = request.url.path
        if "%" in path or ".." in path or "\\" in path:
            return False
        if self._route_prefix and path.startswith(self._route_prefix):
            expected = self._route_token
        elif path == self._hook_path:
            expected = self._hook_token
        else:
            return False
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
        supplied = supplied or request.headers.get("x-api-key", "")
        supplied = supplied or request.headers.get("x-goog-api-key", "")
        return bool(expected and hmac.compare_digest(supplied, expected))

    async def forward(self, request: Request) -> Response:
        if not self.authenticates(request):
            return Response(status_code=403)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > _MAX_BODY:
                return Response(status_code=413)
        target = f"http://127.0.0.1:{self._gateway_port}{request.url.path}"
        if request.url.query:
            target += f"?{request.url.query}"
        headers = {
            key: value for key, value in request.headers.items() if key.lower() not in _HOP_HEADERS
        }
        outbound = self._client.build_request(
            request.method, target, headers=headers, content=bytes(body)
        )
        try:
            response = await self._client.send(outbound, stream=True)
        except httpx.HTTPError:
            return Response(status_code=502)

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(
            stream(),
            status_code=response.status_code,
            headers={
                key: value
                for key, value in response.headers.items()
                if key.lower() not in _HOP_HEADERS
            },
        )

    async def start(self, bind: str, *, socket_path: Path | None = None) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind, 0))
        self.port = int(sock.getsockname()[1])
        if socket_path is not None:
            sock.close()
            if sys.platform != "linux":
                raise DockerExecutionError(
                    "docker_unavailable", "Secure Unix worker sockets require Linux"
                )
            else:
                socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                directory = os.open(socket_path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    sock.bind(f"/proc/self/fd/{directory}/{socket_path.name}")
                finally:
                    os.close(directory)
                os.chmod(socket_path, 0o660)
                self.socket_path = socket_path
        sock.listen(128)
        sock.setblocking(False)
        self._socket = sock
        config = uvicorn.Config(self.app, log_level="error", access_log=False, lifespan="off")
        server = uvicorn.Server(config)
        self._server = server
        self._task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if self._task.done():
                await self._task
                raise RuntimeError("Worker ingress did not start")
            await asyncio.sleep(0.01)
        return self.port

    async def aclose(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), 5.0)
            except TimeoutError:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
        if self._socket:
            self._socket.close()
        if self.socket_path:
            self.socket_path.unlink(missing_ok=True)
        await self._client.aclose()
