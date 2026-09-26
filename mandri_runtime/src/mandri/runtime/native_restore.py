import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable, Mapping

from mandri.core.ids import HarnessKind
from mandri.core.types.sessions import Session
from mandri.runtime import wiring
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.control.stdio_rpc import StdioRpcConnection
from mandri.runtime.errors import HarnessNotInstalledError
from mandri.runtime.native_launch import native_launch
from mandri.runtime.process import ManagedProcess
from mandri.sessions.errors import SessionConflictError

NATIVE_RESTORE_MODEL = "gpt-5.6-luna"
NATIVE_RESTORE_PROVIDER = "openai"


async def restore_codex_model(
    session: Session,
    commands: Mapping[str, list[str]],
    spawn: Callable[..., Awaitable[ManagedProcess]],
) -> None:
    if session.harness is not HarnessKind.CODEX or session.native_id is None:
        raise SessionConflictError("Restoration requires a Codex native conversation")
    command = commands.get("codex")
    if command is None:
        raise HarnessNotInstalledError("No Codex command is configured")
    launch = native_launch(HarnessKind.CODEX, NATIVE_RESTORE_MODEL)
    process = await spawn(
        [*command, *launch.args],
        cwd=session.project_path,
        env=wiring.merged_env(os.environ, launch, None),
    )
    stderr = asyncio.create_task(_drain_stderr(process))
    try:
        async with asyncio.timeout(20):
            connection = StdioRpcConnection(process)
            await connection.call(
                "initialize",
                {
                    "clientInfo": {"name": "mandri", "version": "0.1.0"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            await connection.send({"method": "initialized"})
            result = await connection.call(
                "thread/resume",
                {
                    "threadId": str(session.native_id),
                    "model": NATIVE_RESTORE_MODEL,
                    "modelProvider": NATIVE_RESTORE_PROVIDER,
                },
            )
            thread = result.get("thread")
            if not isinstance(thread, dict) or thread.get("id") != str(session.native_id):
                raise ControlTransportError("Restoration returned a different native conversation")
            provider = result.get("modelProvider", thread.get("modelProvider"))
            if provider != NATIVE_RESTORE_PROVIDER or result.get("model") != NATIVE_RESTORE_MODEL:
                raise ControlTransportError("Native model restoration was not confirmed")
            await connection.call(
                "thread/settings/update",
                {"threadId": str(session.native_id), "model": NATIVE_RESTORE_MODEL},
            )
    finally:
        try:
            await process.stop(grace=1)
        finally:
            stderr.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr


async def _drain_stderr(process: ManagedProcess) -> None:
    while await process.read_stderr_line():
        pass
