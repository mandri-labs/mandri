import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

from mandri.core.ids import ApprovalDecision, HarnessSessionId, ModeApplication
from mandri.core.ports.control import ControlRequest, ControlSink, PromptOutcome, PromptState
from mandri.core.types.prompt import UserPrompt, prompt_text
from mandri.runtime.control.agy_bridge import AgyBridge
from mandri.runtime.control.agy_commands import AgyCommandRunner
from mandri.runtime.control.agy_policy import MODES
from mandri.runtime.control.errors import (
    ControlTransportError,
    ModeRejectedError,
    PromptDeliveryFailedError,
    PromptDeliveryUnknownError,
)
from mandri.runtime.pump import LineEventKind, LinePump


class AgyControlAdapter:
    def __init__(
        self,
        stdout: LinePump,
        stdin: ControlSink,
        bridge: AgyBridge,
        interrupt: Callable[[], Awaitable[bool]],
        expected_id: HarnessSessionId | None = None,
        on_identity: Callable[[HarnessSessionId], None] | None = None,
        commands: AgyCommandRunner | None = None,
    ) -> None:
        self._commands = commands
        self._stdout = stdout
        self._stdin = stdin
        self._bridge = bridge
        self._interrupt = interrupt
        self._expected = expected_id
        self._on_identity = on_identity
        self._identity: HarnessSessionId | None = None
        self._ready = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._error: str | None = None

    def _start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for line in self._stdout.lines():
                if line.kind is not LineEventKind.LINE:
                    continue
                try:
                    event = json.loads(line.text)
                except ValueError:
                    continue
                if not isinstance(event, dict) or event.get("event") != "init":
                    continue
                identity = event.get("conversation_id")
                if not isinstance(identity, str) or not identity:
                    self._error = "Antigravity omitted its conversation identity"
                elif self._expected is not None and identity != self._expected:
                    self._error = "Antigravity resumed a different conversation"
                else:
                    self._identity = HarnessSessionId(identity)
                    if self._on_identity:
                        self._on_identity(self._identity)
                self._ready.set()
        finally:
            self._ready.set()

    async def capture_identity(self) -> HarnessSessionId:
        self._start()
        try:
            await asyncio.wait_for(self._ready.wait(), 15)
        except TimeoutError as error:
            raise ControlTransportError("Antigravity initialization timed out") from error
        if self._error or self._identity is None:
            raise ControlTransportError(self._error or "Antigravity closed before initialization")
        return self._identity

    async def send_prompt(self, content: str | UserPrompt) -> PromptOutcome:
        try:
            await self.capture_identity()
        except ControlTransportError as error:
            raise PromptDeliveryFailedError(str(error)) from error
        frame = {"event": "user", "message": {"content": prompt_text(content)}}
        try:
            self._stdin.write((json.dumps(frame) + "\n").encode())
            await self._stdin.drain()
        except OSError as error:
            raise PromptDeliveryUnknownError("Antigravity input closed") from error
        return PromptOutcome(PromptState.QUEUED)

    async def list_commands(self) -> list[dict[str, Any]]:
        if self._commands is None:
            raise ControlTransportError("Antigravity command execution context is unavailable")
        return await self._commands.list_commands()

    async def execute_command(self, command_id: str, arguments: str) -> dict[str, Any]:
        if self._commands is None:
            raise ControlTransportError("Antigravity command execution context is unavailable")
        return await self._commands.execute_command(command_id, arguments)

    async def set_mode(self, mode: str) -> ModeApplication:
        if mode not in MODES:
            raise ModeRejectedError("Unsupported Antigravity mode")
        return ModeApplication.REQUIRES_RESTART

    async def answer_approval(self, native_request_ref: str, decision: ApprovalDecision) -> bool:
        raise ControlTransportError("Antigravity approvals require correlated delivery")

    async def next_native_request(self) -> ControlRequest:
        raise ControlTransportError("Antigravity approvals are delivered through hooks")

    async def interrupt(self) -> bool:
        return await self._interrupt()

    async def aclose(self) -> None:
        self._stdout.close()
        await self._bridge.aclose()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
