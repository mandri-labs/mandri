"""Codex session mutations over the app-server JSON-RPC line protocol."""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, Protocol

from mandri.core.ids import SessionId, SessionTitle
from mandri.core.ports.sessions import (
    CheckSessionExistsPort,
    DeleteSessionPort,
    RenameSessionPort,
)
from mandri.core.version import __version__
from mandri.sessions.errors import (
    AgentBinaryNotFoundError,
    ServerCommunicationError,
    ServerTimeoutError,
    SessionDeleteError,
    SessionNotFoundError,
    SessionRenameError,
)

CLIENT_INFO = {"clientInfo": {"name": "mandri", "title": "Mandri", "version": __version__}}
DEFAULT_CODEX_BINARY = "codex"
READ_TIMEOUT_S = 15.0
STOP_TIMEOUT_S = 5.0

UNKNOWN_THREAD_MARKER = "unknown thread"

ConnectFactory = Callable[[], Awaitable[Any]]


class LineConnection(Protocol):
    async def send(self, message: dict[str, Any]) -> None: ...

    async def receive(self) -> dict[str, Any]: ...


class JsonRpcCallError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"json-rpc error {code}: {message}")
        self.code = code
        self.message = message


class JsonRpcClient:
    def __init__(self, connection: LineConnection) -> None:
        self._connection = connection
        self._request_id = 0

    async def notify(self, method: str) -> None:
        await self._send({"jsonrpc": "2.0", "method": method})

    async def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        await self._send(message)
        while True:
            reply = await self._receive()
            if "method" in reply or reply.get("id") != request_id:
                continue
            if "error" in reply:
                error = reply["error"]
                raise JsonRpcCallError(
                    int(error.get("code", 0)) if isinstance(error, dict) else 0,
                    str(error.get("message", "")) if isinstance(error, dict) else str(error),
                )
            result = reply.get("result")
            if not isinstance(result, dict):
                return {}
            return dict(result)

    async def _send(self, message: dict[str, Any]) -> None:
        try:
            await self._connection.send(message)
        except (OSError, ValueError, TypeError) as error:
            raise ServerCommunicationError(f"cannot reach codex app-server: {error}") from error

    async def _receive(self) -> dict[str, Any]:
        try:
            async with asyncio.timeout(READ_TIMEOUT_S):
                reply = await self._connection.receive()
        except TimeoutError as error:
            raise ServerTimeoutError("codex app-server reply timeout") from error
        except (OSError, ValueError) as error:
            raise ServerCommunicationError(f"bad app-server stream: {error}") from error
        if not isinstance(reply, dict):
            raise ServerCommunicationError(f"unexpected app-server message: {reply!r}")
        return reply


async def connect_codex_app_server(
    binary_path: str = DEFAULT_CODEX_BINARY,
    *,
    command: Sequence[str] | None = None,
) -> SubprocessLineConnection:
    spawn_options: dict[str, Any] = {}
    if sys.platform == "win32":
        spawn_options["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        process = await asyncio.create_subprocess_exec(
            *(command if command is not None else (binary_path, "app-server")),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **spawn_options,
        )
    except (FileNotFoundError, PermissionError, OSError) as error:
        raise AgentBinaryNotFoundError(f"cannot spawn {binary_path}: {error}") from error
    return SubprocessLineConnection(process)


class SubprocessLineConnection:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process

    async def send(self, message: dict[str, Any]) -> None:
        if self._process.stdin is None:
            raise ServerCommunicationError("app-server stdin unavailable")
        self._process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        await self._process.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise ServerCommunicationError("app-server stdout unavailable")
        line = await self._process.stdout.readline()
        if not line:
            raise ServerCommunicationError("app-server stdout closed")
        reply = json.loads(line)
        if not isinstance(reply, dict):
            raise ServerCommunicationError(f"unexpected app-server message: {reply!r}")
        return reply

    async def close(self) -> None:
        if self._process.returncode is not None:
            return
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            await asyncio.wait_for(self._process.wait(), STOP_TIMEOUT_S)
        except TimeoutError:
            self._process.terminate()
            await self._process.wait()


class CodexMutationsAdapter(RenameSessionPort, DeleteSessionPort, CheckSessionExistsPort):
    """Rename/delete/exists against the codex app-server JSON-RPC surface."""

    def __init__(self, connect: ConnectFactory) -> None:
        self._connect = connect

    def rename(self, session_id: SessionId, title: SessionTitle) -> None:
        asyncio.run(self.rename_async(session_id, title))

    def delete(self, session_id: SessionId) -> None:
        asyncio.run(self.delete_async(session_id))

    def exists(self, session_id: SessionId) -> bool:
        return asyncio.run(self.exists_async(session_id))

    async def rename_async(self, session_id: SessionId, title: SessionTitle) -> None:
        name = str(title)
        if not name.strip():
            raise SessionRenameError("title must not be empty")
        async with self._client() as client:
            try:
                await client.call("thread/name/set", {"threadId": str(session_id), "name": name})
            except JsonRpcCallError as error:
                if UNKNOWN_THREAD_MARKER in error.message:
                    raise SessionNotFoundError(f"unknown codex thread {session_id}") from error
                raise SessionRenameError(f"thread/name/set failed: {error.message}") from error

    async def delete_async(self, session_id: SessionId) -> None:
        async with self._client() as client:
            try:
                await client.call("thread/delete", {"threadId": str(session_id)})
            except JsonRpcCallError as error:
                raise SessionDeleteError(f"thread/delete failed: {error.message}") from error

    async def exists_async(self, session_id: SessionId) -> bool:
        async with self._client() as client:
            try:
                await client.call("thread/resume", {"threadId": str(session_id)})
            except JsonRpcCallError as error:
                if UNKNOWN_THREAD_MARKER in error.message:
                    return False
                raise ServerCommunicationError(
                    f"codex thread {session_id} not reachable: {error.message}"
                ) from error
        return True

    @contextlib.asynccontextmanager
    async def _client(self) -> AsyncIterator[JsonRpcClient]:
        connection = await self._open_connection()
        try:
            client = JsonRpcClient(connection)
            try:
                await client.call("initialize", CLIENT_INFO)
            except JsonRpcCallError as error:
                raise ServerCommunicationError(
                    f"codex app-server initialize failed: {error.message}"
                ) from error
            await client.notify("initialized")
            yield client
        finally:
            closer = getattr(connection, "close", None)
            if closer is not None:
                await closer()

    async def _open_connection(self) -> Any:
        try:
            return await self._connect()
        except (OSError, RuntimeError) as error:
            raise ServerCommunicationError(
                f"cannot connect to codex app-server: {error}"
            ) from error
