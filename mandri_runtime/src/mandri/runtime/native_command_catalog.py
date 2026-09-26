import asyncio
import contextlib
import re
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from mandri.core.ids import HarnessKind
from mandri.runtime.control.agy_commands import AgyCommandRunner
from mandri.runtime.control.claude_commands import ClaudeCommands
from mandri.runtime.control.codex_commands import CodexCommands
from mandri.runtime.control.errors import ControlError, ControlTransportError
from mandri.runtime.control.opencode_commands import OpencodeCommands
from mandri.runtime.control.pi_commands import PiCommands
from mandri.runtime.control.pi_rpc import PiRpcConnection
from mandri.runtime.control.stdio_rpc import StdioRpcConnection
from mandri.runtime.process import ManagedProcess

_LISTENING = re.compile(r"opencode server listening on (http://127\.0\.0\.1:([0-9]+))$")


async def _drain(read: Callable[[], Awaitable[str]]) -> None:
    while await read():
        pass


def _opencode_argv(command: list[str]) -> list[str]:
    result = []
    skip = False
    for argument in command:
        if skip:
            skip = False
            continue
        if argument in ("--port", "--hostname"):
            skip = True
        elif argument not in ("--mdns", "--no-mdns") and not argument.startswith(
            ("--port=", "--hostname=", "--mdns=")
        ):
            result.append(argument)
    return [*result, "--hostname", "127.0.0.1", "--port", "0", "--mdns=false"]


async def _opencode_catalog(
    process: ManagedProcess,
    cwd: str | Path,
    password: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict[str, Any]]:
    address = None
    while line := await process.read_stdout_line():
        match = _LISTENING.fullmatch(line.strip())
        if match and 0 < int(match[2]) < 65536:
            address = match[1]
            break
    if address is None:
        raise ControlTransportError("OpenCode did not start its command discovery server")
    stdout = asyncio.create_task(_drain(process.read_stdout_line))
    try:
        async with httpx.AsyncClient(
            base_url=address,
            auth=("opencode", password),
            params={"directory": str(cwd)},
            trust_env=False,
            timeout=15,
            transport=transport,
        ) as client:

            async def call(method: str, path: str, body: dict[str, Any] | None) -> httpx.Response:
                return await client.request(method, path, json=body)

            return await OpencodeCommands(call, "").list_commands()
    finally:
        stdout.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stdout


async def discover_commands(
    harness: HarnessKind,
    command: list[str],
    cwd: str | Path,
    env: dict[str, str],
    spawn: Callable[..., Awaitable[ManagedProcess]],
    *,
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict[str, Any]]:
    if not command:
        raise ControlError("No native harness executable is configured")
    if harness is HarnessKind.AGY:

        async def launch(argv: list[str]) -> ManagedProcess:
            return await spawn(argv, cwd=cwd, env=env)

        return await AgyCommandRunner(command, launch).list_commands()
    password = secrets.token_urlsafe(32)
    argv = _opencode_argv(command) if harness is HarnessKind.OPENCODE else command
    if harness is HarnessKind.CLAUDE and "--no-session-persistence" not in argv:
        argv = [*argv, "--no-session-persistence"]
    if harness is HarnessKind.PI and "--no-session" not in argv:
        argv = [*argv, "--no-session"]
    environment = dict(env)
    if harness is HarnessKind.OPENCODE:
        environment.update(OPENCODE_SERVER_PASSWORD=password, OPENCODE_SERVER_USERNAME="opencode")
    process = await spawn(argv, cwd=cwd, env=environment)
    stderr = asyncio.create_task(_drain(process.read_stderr_line))
    try:
        async with asyncio.timeout(20):
            if harness is HarnessKind.OPENCODE:
                return await _opencode_catalog(process, cwd, password, http_transport)
            if harness is HarnessKind.PI:
                return await PiCommands(PiRpcConnection(process).call).list_commands()
            connection = StdioRpcConnection(process)
            if harness is HarnessKind.CLAUDE:
                result = await connection.call("initialize", {}, claude=True)
                rows = result.get("commands")
                if not isinstance(rows, list):
                    raise ControlError("Claude did not return a native command catalog")
                commands = ClaudeCommands()
                commands.replace(rows)
                return commands.descriptors()
            if harness is not HarnessKind.CODEX:
                raise ControlError("Native command discovery is unavailable for this harness")
            await connection.call(
                "initialize",
                {
                    "clientInfo": {"name": "mandri", "version": "0.1.0"},
                    "capabilities": {},
                },
            )
            await connection.send({"method": "initialized"})

            async def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
                return {"result": await connection.call(method, params)}

            return await CodexCommands(call, str(cwd)).list_commands()
    finally:
        try:
            await process.stop(grace=1)
        finally:
            stderr.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr
