import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from mandri.api.deps import app_state
from mandri.api.errors import ApiError
from mandri.api.routers._effort import normalize_effort, validate_effort
from mandri.core.terminal import TerminalSize, TerminalStart
from mandri.core.types.execution import ProtectionError
from mandri.providers.errors import ProviderInvalidError, ProviderNotFoundError
from mandri.runtime.docker_process import DockerProcess
from mandri.runtime.errors import HarnessNotInstalledError, ProcessIOError, ProcessSpawnError
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.runtime.service import RuntimeService
from mandri.runtime.terminal_process import TerminalProcess
from pydantic import ValidationError

router = APIRouter(tags=["runtime"])


async def relay_terminal(
    websocket: WebSocket, runtime: RuntimeService, spec: TerminalStart
) -> None:
    active: list[TerminalProcess | DockerProcess] = []

    async def output() -> None:
        async with runtime.terminal_session(spec) as session:
            process = session.process
            if not isinstance(process, (TerminalProcess, DockerProcess)):
                raise ProcessIOError("Session has no terminal")
            active.append(process)
            await websocket.send_json({"type": "started", "session_id": session.id})
            stdout = process.process.stdout
            if stdout is None:
                raise ProcessIOError("Terminal output is unavailable")
            while chunk := await stdout.read(65536):
                await websocket.send_bytes(chunk)
            code = await process.wait()
        await websocket.send_json({"type": "exit", "code": code})

    async def input_stream() -> None:
        while True:
            frame = await websocket.receive()
            if frame["type"] == "websocket.disconnect":
                return
            if not active:
                raise ValueError("Wait for the terminal to start before sending input")
            if frame.get("bytes") is not None:
                await active[0].write_stdin(frame["bytes"])
            elif frame.get("text") is not None:
                active[0].resize(TerminalSize.model_validate_json(frame["text"]))

    tasks = [asyncio.create_task(output()), asyncio.create_task(input_stream())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@router.websocket("/runtime/terminal")
async def terminal(websocket: WebSocket) -> None:
    await websocket.accept(subprotocol=websocket.scope.get("mandri.subprotocol"))
    state = app_state(websocket.app)
    try:
        if state is None or state.runtime is None or state.gateway is None:
            raise ValueError("Runtime gateway is unavailable")
        spec = TerminalStart.model_validate_json(await websocket.receive_text())
        validate_effort(state.gateway, spec.model, spec.effort)
        spec.effort = normalize_effort(spec.effort)
        await relay_terminal(websocket, state.runtime, spec)
    except WebSocketDisconnect:
        return
    except (
        ApiError,
        ProtectionError,
        DockerExecutionError,
        HarnessNotInstalledError,
        ProcessIOError,
        ProcessSpawnError,
        ValidationError,
        ValueError,
        OSError,
        ProviderInvalidError,
        ProviderNotFoundError,
    ) as error:
        with contextlib.suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "error", "message": str(error)})
    finally:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await websocket.close()
