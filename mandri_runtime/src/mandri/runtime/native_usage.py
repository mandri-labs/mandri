import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from mandri.core.clock import system_now_ms
from mandri.core.ids import HarnessKind
from mandri.core.types.usage import UsageAccount
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.control.stdio_rpc import StdioRpcConnection
from mandri.runtime.process import ManagedProcess
from mandri.sessions.usage.account_queries import (
    agy_usage_snapshot,
    claude_usage_snapshot,
    pi_usage_snapshots,
)
from mandri.sessions.usage.accounts import codex_account_snapshot


async def _drain(process: ManagedProcess, harness: HarnessKind) -> None:
    while line := await process.read_stderr_line():
        if harness is HarnessKind.AGY and "authentication required" in line.lower():
            raise PermissionError("Antigravity CLI authentication required")


async def _result(process: ManagedProcess, *, pi: bool = False) -> dict[str, Any]:
    size = 0
    while line := await process.read_stdout_line():
        size += len(line)
        if size > 1024 * 1024:
            raise ControlTransportError("Native quota response exceeded limit")
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if pi and isinstance(payload, dict) and payload.get("type") == "extension_ui_request":
            if payload.get("method") != "notify":
                continue
            try:
                payload = json.loads(payload.get("message", ""))
            except (ValueError, TypeError):
                continue
        if isinstance(payload, dict) and (not pi or payload.get("type") == "mandri_usage"):
            return payload
    raise ControlTransportError("Native quota response unavailable")


async def _agy_result(process: ManagedProcess, stderr: asyncio.Task[None]) -> dict[str, Any]:
    stdout = asyncio.create_task(_result(process))
    try:
        await asyncio.wait((stdout, stderr), return_when=asyncio.FIRST_COMPLETED)
        if stderr.done():
            await stderr
        payload = await stdout
        code = await process.wait()
        if code != 0 or payload.get("status") == "ERROR":
            error = str(payload.get("error", "")).lower()
            if "authentication" in error or "not logged in" in error:
                raise PermissionError("Antigravity CLI authentication required")
            raise ControlTransportError("Native quota command failed")
        return payload
    finally:
        stdout.cancel()
        await asyncio.gather(stdout, return_exceptions=True)


async def read_native_usage(
    harness: HarnessKind,
    profile_id: str,
    command: list[str],
    cwd: Path,
    env: Mapping[str, str],
    spawn: Callable[..., Awaitable[ManagedProcess]],
) -> list[UsageAccount]:
    process = await spawn(command, cwd=cwd, env=env, line_limit=1024 * 1024)
    stderr = asyncio.create_task(_drain(process, harness))
    try:
        async with asyncio.timeout(30):
            connection = StdioRpcConnection(process)
            if harness is HarnessKind.CODEX:
                await connection.call(
                    "initialize",
                    {
                        "clientInfo": {"name": "mandri-quota", "version": "0.1.0"},
                        "capabilities": {},
                    },
                )
                await connection.send({"method": "initialized"})
                payload = await connection.call("account/rateLimits/read", {})
                return [codex_account_snapshot(profile_id, payload, observed_at_ms=system_now_ms())]
            if harness is HarnessKind.CLAUDE:
                await connection.call("initialize", {}, claude=True)
                payload = await connection.call("get_usage", {"skip_behaviors": True}, claude=True)
                return [claude_usage_snapshot(profile_id, payload, observed_at_ms=system_now_ms())]
            if harness is HarnessKind.PI:
                payload = await _result(process, pi=True)
                return pi_usage_snapshots(profile_id, payload, observed_at_ms=system_now_ms())
            payload = await _agy_result(process, stderr)
            return [agy_usage_snapshot(profile_id, payload, observed_at_ms=system_now_ms())]
    finally:
        try:
            await process.stop(grace=0.5)
        finally:
            stderr.cancel()
            await asyncio.gather(stderr, return_exceptions=True)
