"""Claude stream-json control adapter."""

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, final

from mandri.core.ids import ApprovalDecision, ApprovalStatus, HarnessSessionId, ModeApplication
from mandri.core.ports.control import ControlRequest, ControlSink, PromptOutcome, PromptState
from mandri.core.types.approvals import ApprovalRequest
from mandri.core.types.prompt import UserPrompt, prompt_text
from mandri.runtime.control.agents.claude import ClaudeAgentControl
from mandri.runtime.control.claude_commands import ClaudeCommands
from mandri.runtime.control.errors import (
    ControlError,
    ControlTransportError,
    HarnessNotInitializedError,
    ModeRejectedError,
    PromptDeliveryUnknownError,
)
from mandri.runtime.control.prompt import native_parts
from mandri.runtime.pump import LineEventKind, LinePump

_IDENTITY_TIMEOUT_S = 5.0
_ACK_TIMEOUT_S = 5.0
_REQUEST_TIMEOUT_S = 15.0
_DENY_MESSAGE = "User denied this action in Mandri"
_MISSING_INPUT: dict[str, Any] = {}
_logger = logging.getLogger(__name__)


@final
class ClaudeControlAdapter:
    def __init__(
        self,
        *,
        stdout_pump: LinePump,
        stderr_pump: LinePump,
        stdin: ControlSink,
        on_identity: Callable[[HarnessSessionId], None] | None = None,
        on_conversation_reset: Callable[[HarnessSessionId], Awaitable[None] | None] | None = None,
        resumed: bool = False,
    ) -> None:
        self._stdout_pump = stdout_pump
        self._stderr_pump = stderr_pump
        self._stdin = stdin
        self._on_identity = on_identity
        self._on_conversation_reset = on_conversation_reset
        self._session_id: HarnessSessionId | None = None
        self._identity_set = asyncio.Event()
        self._requests: asyncio.Queue[ControlRequest] = asyncio.Queue()
        self._surfaced: dict[str, ControlRequest] = {}
        self._acks: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._pump_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self.agents = ClaudeAgentControl(self.stop_agent_task)
        self._commands = ClaudeCommands()
        self._command_discovery_lock = asyncio.Lock()
        self._turn_active = False
        self._title_requested = resumed
        self._title_task: asyncio.Task[None] | None = None

    async def list_commands(self) -> list[dict[str, Any]]:
        async with self._command_discovery_lock:
            if not self._commands.initialized:
                request_id = _new_request_id()
                response = await self._await_ack(
                    request_id,
                    {
                        "type": "control_request",
                        "request_id": request_id,
                        "request": {"subtype": "initialize"},
                    },
                )
                if response.get("subtype") != "success":
                    raise ControlTransportError(_ack_error_message(response))
                body = response.get("response")
                if isinstance(body, dict) and isinstance(body.get("commands"), list):
                    self._commands.replace(body["commands"])
                if not self._commands.initialized:
                    raise ControlTransportError("Claude did not provide a command catalog")
        return self._commands.descriptors()

    async def execute_command(self, command_id: str, arguments: str) -> dict[str, Any]:
        await self.list_commands()
        if self._turn_active:
            raise ControlError("Wait for the current Claude turn to finish")
        native_id, content, result = self._commands.begin(command_id, arguments)
        self._turn_active = True
        try:
            await self._send(
                {
                    "type": "user",
                    "uuid": native_id,
                    "message": {"role": "user", "content": content},
                }
            )
        except ControlTransportError:
            self._commands.fail("Claude command delivery is uncertain")
        return await asyncio.shield(result)

    async def stop_agent_task(self, task_id: str) -> bool:
        self._require_identity()
        request_id = _new_request_id()
        response = await self._await_ack(
            request_id,
            {
                "type": "control_request",
                "request_id": request_id,
                "request": {"subtype": "stop_task", "task_id": task_id},
            },
        )
        if response.get("subtype") != "success":
            raise ControlTransportError("Claude rejected the agent stop operation")
        return True

    async def capture_identity(self) -> HarnessSessionId | None:
        self._ensure_pump()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._identity_set.wait(), timeout=_IDENTITY_TIMEOUT_S)
        return self._session_id

    async def next_native_request(self) -> ControlRequest:
        self._ensure_pump()
        try:
            return await asyncio.wait_for(self._requests.get(), timeout=_REQUEST_TIMEOUT_S)
        except TimeoutError as exc:
            raise ControlTransportError("no approval request arrived") from exc

    async def answer_approval(self, native_request_ref: str, decision: ApprovalDecision) -> bool:
        self._require_identity()
        inner = self._approval_payload(native_request_ref, decision)
        frame = {
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": native_request_ref,
                "response": inner,
            },
        }
        await self._send(frame)
        return True

    async def set_mode(self, mode: str) -> ModeApplication:
        self._require_identity()
        request_id = _new_request_id()
        frame = {
            "type": "control_request",
            "request_id": request_id,
            "request": {"subtype": "set_permission_mode", "mode": mode},
        }
        response = await self._await_ack(request_id, frame)
        if response.get("subtype") != "success":
            raise ModeRejectedError(_ack_error_message(response))
        return ModeApplication.MID_SESSION_APPLIED

    async def send_prompt(self, content: str | UserPrompt) -> PromptOutcome:
        self._ensure_pump()
        if self._commands.pending:
            return PromptOutcome(state=PromptState.ERROR, code="command_in_progress")
        self._turn_active = True
        frame = {
            "type": "user",
            "message": {"role": "user", "content": native_parts(content, "claude")},
        }
        try:
            await self._send(frame)
        except ControlTransportError as error:
            raise PromptDeliveryUnknownError(str(error)) from error
        description = prompt_text(content).strip()
        if not self._title_requested and description and not description.startswith("/"):
            self._title_requested = True
            self._title_task = asyncio.create_task(self._generate_title(description))
        return PromptOutcome(state=PromptState.QUEUED)

    async def _generate_title(self, description: str) -> None:
        try:
            await self._identity_set.wait()
            if self._session_id is None:
                return
            request_id = _new_request_id()
            response = await self._await_ack(
                request_id,
                {
                    "type": "control_request",
                    "request_id": request_id,
                    "request": {
                        "subtype": "generate_session_title",
                        "description": description,
                        "persist": True,
                    },
                },
            )
            if response.get("subtype") != "success":
                _logger.debug("Claude session title request was rejected")
        except ControlTransportError:
            _logger.debug("Claude session title request did not complete")

    async def interrupt(self) -> bool:
        self._require_identity()
        request_id = _new_request_id()
        frame = {
            "type": "control_request",
            "request_id": request_id,
            "request": {"subtype": "interrupt"},
        }
        await self._await_ack(request_id, frame)
        return True

    async def aclose(self) -> None:
        self._stdout_pump.close()
        self._stderr_pump.close()
        self._commands.fail("Claude control stream closed; command outcome is unknown")
        tasks = [
            task for task in (self._pump_task, self._stderr_task, self._title_task)
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._pump_task = None
        self._stderr_task = None
        self._title_task = None

    def _approval_payload(
        self, native_request_ref: str, decision: ApprovalDecision
    ) -> dict[str, Any]:
        if decision is ApprovalDecision.ALLOW:
            request = self._surfaced.get(native_request_ref)
            tool_input = request.tool_input if request is not None else _MISSING_INPUT
            return {"behavior": "allow", "updatedInput": tool_input}
        return {"behavior": "deny", "message": _DENY_MESSAGE}

    def _require_identity(self) -> HarnessSessionId:
        if self._session_id is None:
            raise HarnessNotInitializedError("harness identity has not been captured")
        return self._session_id

    async def _await_ack(self, request_id: str, frame: dict[str, Any]) -> dict[str, Any]:
        self._ensure_pump()
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._acks[request_id] = future
        try:
            await self._send(frame)
            return await asyncio.wait_for(future, timeout=_ACK_TIMEOUT_S)
        except TimeoutError as exc:
            raise ControlTransportError("control ack timed out") from exc
        finally:
            self._acks.pop(request_id, None)

    async def _send(self, frame: dict[str, Any]) -> None:
        payload = json.dumps(frame) + "\n"
        try:
            self._stdin.write(payload.encode("utf-8"))
            await self._stdin.drain()
        except OSError as exc:
            raise ControlTransportError("stdin write failed") from exc

    def _ensure_pump(self) -> None:
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.get_running_loop().create_task(self._pump_stdout())
        if self._stderr_task is None or self._stderr_task.done():
            self._stderr_task = asyncio.get_running_loop().create_task(self._drain_stderr())

    async def _pump_stdout(self) -> None:
        try:
            async for event in self._stdout_pump.lines():
                if event.kind is not LineEventKind.LINE:
                    continue
                frame = _parse_frame(event.text)
                if frame is not None:
                    self._dispatch(frame)
        finally:
            self._identity_set.set()
            self._fail_pending_acks()
            self._commands.fail("Claude control stream closed; command outcome is unknown")

    async def _drain_stderr(self) -> None:
        async for _event in self._stderr_pump.lines():
            pass

    def _dispatch(self, frame: dict[str, Any]) -> None:
        self._commands.observe(frame)
        kind = frame.get("type")
        if kind == "result":
            self._turn_active = False
        if kind == "conversation_reset":
            new_id = frame.get("new_conversation_id")
            if isinstance(new_id, str) and new_id and new_id != self._session_id:
                if self._title_task is not None:
                    self._title_task.cancel()
                self._title_requested = False
                self._session_id = HarnessSessionId(new_id)
                if self._on_conversation_reset is not None:
                    self._on_conversation_reset(self._session_id)
        body = frame.get("request", frame)
        if kind == "system" and frame.get("subtype") == "init":
            self._absorb_identity(frame)
        elif (
            kind == "control_request"
            and isinstance(body, dict)
            and body.get("subtype") == "can_use_tool"
        ):
            self._surface_request(frame)
        elif kind == "control_response":
            self._resolve_ack(frame)

    def _absorb_identity(self, frame: dict[str, Any]) -> None:
        session_id = frame.get("session_id")
        if isinstance(session_id, str) and self._session_id is None:
            self._session_id = HarnessSessionId(session_id)
            self._notify_identity(self._session_id)
        self._identity_set.set()

    def _notify_identity(self, native_id: HarnessSessionId) -> None:
        if self._on_identity is not None:
            self._on_identity(native_id)

    def _surface_request(self, frame: dict[str, Any]) -> None:
        request = _tool_request(frame)
        if request is None:
            return
        self._surfaced[request.native_request_ref] = request
        self._requests.put_nowait(request)

    def _resolve_ack(self, frame: dict[str, Any]) -> None:
        response = frame.get("response")
        if not isinstance(response, dict):
            return
        request_id = response.get("request_id")
        if not isinstance(request_id, str):
            return
        future = self._acks.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(response)

    def _fail_pending_acks(self) -> None:
        pending = list(self._acks.values())
        self._acks.clear()
        for future in pending:
            if not future.done():
                future.set_exception(ControlTransportError("control stream closed"))


@final
class ClaudeApprovalMessenger:
    """Delivers approval decisions to a claude harness over stdin, natively."""

    def __init__(self, *, stdin: ControlSink) -> None:
        self._stdin = stdin

    async def deliver(self, request: ApprovalRequest) -> bool:
        frame = {
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request.native_request_ref,
                "response": self._payload(request),
            },
        }
        payload = json.dumps(frame) + "\n"
        try:
            self._stdin.write(payload.encode("utf-8"))
            await self._stdin.drain()
        except OSError as exc:
            raise ControlTransportError("stdin write failed") from exc
        return True

    def _payload(self, request: ApprovalRequest) -> dict[str, Any]:
        if request.status is ApprovalStatus.EXPIRED:
            return {
                "behavior": "deny",
                "message": "Mandri approval expired without a user response",
            }
        if request.status is ApprovalStatus.CANCELLED:
            return {"behavior": "deny", "message": "Approval cancelled in Mandri"}
        if request.decision is not ApprovalDecision.ALLOW:
            return {"behavior": "deny", "message": _DENY_MESSAGE}
        updated = _echoed_input(request.native_request)
        if request.updated_input is not None:
            replacement = _parse_frame(str(request.updated_input))
            if replacement is None:
                raise ControlError("Claude tool input must be an object")
            updated = replacement
        if request.answers:
            updated["answers"] = {
                item["question"]: ", ".join(item["answers"])
                for item in request.answers
                if isinstance(item.get("question"), str)
                and isinstance(item.get("answers"), list)
                and all(isinstance(answer, str) for answer in item["answers"])
            }
        payload = {"behavior": "allow", "updatedInput": updated}
        native = _parse_frame(str(request.native_request)) or {}
        body = native.get("request")
        if isinstance(body, dict) and body.get("tool_name") == "ExitPlanMode":
            payload["updatedPermissions"] = [{
                "type": "setMode", "mode": request.permission_mode or "default",
                "destination": "session",
            }]
        return payload


def _tool_request(frame: dict[str, Any]) -> ControlRequest | None:
    request_id = frame.get("request_id")
    body = frame.get("request", frame)
    if not isinstance(body, dict):
        return None
    tool_name = body.get("tool_name")
    tool_use_id = body.get("tool_use_id")
    tool_input = body.get("input")
    if (
        not isinstance(request_id, str)
        or not isinstance(tool_name, str)
        or not isinstance(tool_use_id, str)
        or not isinstance(tool_input, dict)
    ):
        return None
    return ControlRequest(
        native_request_ref=request_id,
        tool_name=tool_name,
        tool_input=dict(tool_input),
        tool_use_id=tool_use_id,
    )


def _echoed_input(native_request: object) -> dict[str, Any]:
    event = _parse_frame(str(native_request))
    if event is None:
        return dict(_MISSING_INPUT)
    body = event.get("request")
    nested = body if isinstance(body, dict) else {}
    tool_input = nested.get("input")
    return dict(tool_input) if isinstance(tool_input, dict) else dict(_MISSING_INPUT)


def _ack_error_message(response: dict[str, Any]) -> str:
    error = response.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    return "mode change rejected"


def _new_request_id() -> str:
    return f"mandri-{uuid.uuid4().hex}"


def _parse_frame(text: str) -> dict[str, Any] | None:
    try:
        frame = json.loads(text)
    except json.JSONDecodeError:
        return None
    return frame if isinstance(frame, dict) else None
