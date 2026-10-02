"""Codex app-server v2 control adapter over stdio JSON-RPC."""

import asyncio
import contextlib
import dataclasses
import json
import logging
from typing import Any, final

from mandri.core.ids import ApprovalDecision, ApprovalKind, HarnessSessionId, ModeApplication
from mandri.core.ports.control import ControlRequest, ControlSink, PromptOutcome, PromptState
from mandri.core.types.prompt import UserPrompt
from mandri.runtime.codex_request_ids import request_reference
from mandri.runtime.control.agents.codex import CodexAgentControl
from mandri.runtime.control.codex_commands import CodexCommands
from mandri.runtime.control.codex_liveness import CodexLivenessSnapshot, read_queue, read_snapshot
from mandri.runtime.control.errors import (
    ControlError,
    ControlTransportError,
    HarnessNotInitializedError,
    ModeRejectedError,
    PromptDeliveryFailedError,
    SteerNoActiveTurnError,
    ThreadOwnershipError,
)
from mandri.runtime.control.modes import CODEX_PROFILES
from mandri.runtime.control.prompt import native_parts
from mandri.runtime.pump import LineEventKind, LinePump

JSONRPC_VERSION = "2.0"
CLIENT_NAME = "mandri"
CLIENT_VERSION = "0.1.0"
RPC_RESPONSE_TIMEOUT_SECONDS = 60.0
logger = logging.getLogger(__name__)

INITIALIZE_METHOD = "initialize"
INITIALIZED_NOTIFICATION = "initialized"
THREAD_START_METHOD = "thread/start"
THREAD_RESUME_METHOD = "thread/resume"
THREAD_FORK_METHOD = "thread/fork"
THREAD_OWNERSHIP_ERROR_CODE = -32600
TURN_START_METHOD = "turn/start"
TURN_STEER_METHOD = "turn/steer"
TURN_INTERRUPT_METHOD = "turn/interrupt"

TURN_STARTED_NOTIFICATION = "turn/started"
TURN_COMPLETED_NOTIFICATION = "turn/completed"

_STEER_FAILURE_MARKERS = ("NoActiveTurn", "ExpectedTurnMismatch")
_APPROVAL_POLICIES = frozenset({"untrusted", "on-request", "never"})

_APPROVAL_METHOD_KINDS = {
    "item/commandExecution/requestApproval": ApprovalKind.COMMAND_EXECUTION,
    "item/fileChange/requestApproval": ApprovalKind.FILE_CHANGE,
    "item/permissions/requestApproval": ApprovalKind.PERMISSION_SCOPE,
    "item/tool/requestUserInput": ApprovalKind.USER_INPUT,
    "mcpServer/elicitation/request": ApprovalKind.ELICITATION,
}

_DECISION_VALUES = {
    ApprovalDecision.ALLOW: "accept",
    ApprovalDecision.DENY: "decline",
    ApprovalDecision.ONCE: "accept",
    ApprovalDecision.ALWAYS: "acceptForSession",
    ApprovalDecision.ACCEPT: "accept",
    ApprovalDecision.ACCEPT_FOR_SESSION: "acceptForSession",
    ApprovalDecision.DECLINE: "decline",
    ApprovalDecision.CANCEL: "cancel",
}


@final
@dataclasses.dataclass(frozen=True)
class CodexControlRequest(ControlRequest):
    kind: ApprovalKind = ApprovalKind.UNKNOWN
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)
    native_request: dict[str, Any] = dataclasses.field(default_factory=dict)


def _turn_id_of(params: dict[str, Any]) -> str | None:
    turn = params.get("turn")
    if not isinstance(turn, dict):
        return None
    turn_id = turn.get("id")
    return turn_id if isinstance(turn_id, str) else None


def _error_text(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    return message if isinstance(message, str) else json.dumps(payload)


def _request_from_message(message: dict[str, Any]) -> CodexControlRequest:
    method = message.get("method")
    method_name = method if isinstance(method, str) else ""
    params = message.get("params")
    detail = params if isinstance(params, dict) else {}
    request_id = message.get("id")
    reference = request_reference(request_id)
    if reference is None:
        raise ControlTransportError("Invalid native request identifier")
    return CodexControlRequest(
        native_request_ref=reference,
        tool_name=method_name or "request",
        tool_input=detail,
        tool_use_id=str(request_id),
        kind=_APPROVAL_METHOD_KINDS.get(method_name, ApprovalKind.UNKNOWN),
        detail=detail,
        native_request=message,
    )


@final
class CodexControlAdapter:
    """Drives one codex app-server session across its stdio JSON-RPC connection."""

    def __init__(
        self,
        stdout_pump: LinePump,
        stdin: ControlSink,
        thread_start_params: dict[str, Any] | None = None,
        resume_thread_id: HarnessSessionId | None = None,
        fork_thread_id: HarnessSessionId | None = None,
        fork_path: str | None = None,
    ) -> None:
        self._stdout_pump = stdout_pump
        self._stdin = stdin
        self._thread_start_params = dict(thread_start_params) if thread_start_params else {}
        self._resume_thread_id = resume_thread_id
        self._fork_thread_id = fork_thread_id
        self._fork_path = fork_path
        if resume_thread_id is not None and fork_thread_id is not None:
            raise ValueError("Native resume and fork are mutually exclusive")
        self._pending: dict[int | str, asyncio.Future[dict[str, Any]]] = {}
        self._server_requests: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._native_request_ids: dict[str, int | str] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._next_request_id = 0
        self._thread_id: str | None = None
        self._session_id: str | None = None
        self._active_turn_id: str | None = None
        self._approval_policy: str | None = None
        self._sandbox_policy: dict[str, Any] | None = None
        self.agents = CodexAgentControl(self._call)
        cwd = self._thread_start_params.get("cwd")
        self._commands = CodexCommands(self._call, cwd if isinstance(cwd, str) else None)

    async def capture_identity(self) -> HarnessSessionId | None:
        if self._thread_id is not None:
            return HarnessSessionId(self._thread_id)
        await self._handshake()
        resume_thread_id = self._resume_thread_id
        if self._fork_thread_id is not None:
            params = {
                **self._thread_start_params,
                "threadId": str(self._fork_thread_id),
                "deferGoalContinuation": True,
            }
            if self._fork_path is not None:
                params["path"] = self._fork_path
            message = await self._call(THREAD_FORK_METHOD, params)
            result = self._identity_result(message, resume=True)
            thread = result.get("thread")
            if isinstance(thread, dict) and thread.get("id") == str(self._fork_thread_id):
                raise ControlTransportError("Native fork returned its source conversation")
        elif resume_thread_id is None:
            message = await self._call(THREAD_START_METHOD, dict(self._thread_start_params))
            result = self._identity_result(message, resume=False)
        else:
            message = await self._call(
                THREAD_RESUME_METHOD,
                {**self._thread_start_params, "threadId": str(resume_thread_id)},
            )
            result = self._identity_result(message, resume=True)
        thread_id = self._claim_thread(result)
        return HarnessSessionId(thread_id)

    async def next_native_request(self) -> CodexControlRequest:
        self._require_thread()
        message = await self._server_requests.get()
        if message is None:
            raise ControlTransportError("codex control stream closed")
        return _request_from_message(message)

    async def answer_approval(self, native_request_ref: str, decision: ApprovalDecision) -> bool:
        self._require_thread()
        decision_value = _DECISION_VALUES.get(decision)
        if decision_value is None:
            return False
        return await self.answer_native_request(native_request_ref, {"decision": decision_value})

    async def answer_native_request(self, reference: str, result: dict[str, Any]) -> bool:
        self._require_thread()
        if reference not in self._native_request_ids:
            return False
        identifier = self._native_request_ids[reference]
        await self._send(
            {
                "jsonrpc": JSONRPC_VERSION,
                "id": identifier,
                "result": result,
            }
        )
        self._native_request_ids.pop(reference, None)
        return True

    async def set_mode(self, mode: str) -> ModeApplication:
        thread_id = self._require_thread()
        profile = CODEX_PROFILES.get(mode)
        if profile is None and mode not in _APPROVAL_POLICIES:
            raise ModeRejectedError(f"unsupported codex approval policy: {mode}")
        if self._active_turn_id is not None:
            return ModeApplication.REQUIRES_RESTART
        params: dict[str, Any] = {"threadId": thread_id}
        if profile is None:
            params["approvalPolicy"] = mode
        else:
            params.update(
                approvalPolicy=profile["approvalPolicy"],
                approvalsReviewer=profile["approvalsReviewer"],
                sandboxPolicy={
                    "type": "dangerFullAccess" if mode == "full-access" else "workspaceWrite"
                },
            )
        message = await self._call("thread/settings/update", params)
        self._result_or_failure(message, ModeRejectedError)
        self._approval_policy = params["approvalPolicy"]
        if "sandboxPolicy" in params:
            self._sandbox_policy = params["sandboxPolicy"]
        return ModeApplication.MID_SESSION_APPLIED

    async def send_prompt(self, content: str | UserPrompt) -> PromptOutcome:
        thread_id = self._require_thread()
        if self._active_turn_id is not None:
            return await self._steer(thread_id, content)
        return await self._start_turn(thread_id, content)

    async def list_commands(self) -> list[dict[str, Any]]:
        self._require_thread()
        return await self._commands.list_commands()

    async def execute_command(self, identifier: str, arguments: str) -> dict[str, Any]:
        thread_id = self._require_thread()
        if self._active_turn_id is not None:
            raise ControlError("Wait for the current Codex turn before running a skill")
        return await self._commands.execute(identifier, self._turn_params(thread_id, arguments))

    async def interrupt(self) -> bool:
        thread_id = self._require_thread()
        turn_id = self._active_turn_id
        if turn_id is None:
            return False
        message = await self._call(
            TURN_INTERRUPT_METHOD, {"threadId": thread_id, "turnId": turn_id}
        )
        if "error" in message:
            return False
        self._active_turn_id = None
        return True

    async def read_usage_limits(self) -> dict[str, Any]:
        return await self.read_account_rate_limits()

    async def read_queue(self) -> bool:
        return await read_queue(self._call, self._require_thread())

    async def read_liveness(self) -> CodexLivenessSnapshot:
        return await read_snapshot(self._call, self._require_thread())

    async def read_account_rate_limits(self) -> dict[str, Any]:
        message = await self._call("account/rateLimits/read", {})
        return self._result_or_failure(message, ControlTransportError)

    async def aclose(self) -> None:
        self._stdout_pump.close()
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None

    def _require_thread(self) -> str:
        if self._thread_id is None:
            raise HarnessNotInitializedError("codex control used before identity capture")
        return self._thread_id

    def _identity_result(self, message: dict[str, Any], resume: bool) -> dict[str, Any]:
        payload = message.get("error")
        if isinstance(payload, dict):
            if resume and payload.get("code") == THREAD_OWNERSHIP_ERROR_CODE:
                raise ThreadOwnershipError(_error_text(payload)) from None
            raise ControlTransportError(_error_text(payload)) from None
        result = message.get("result")
        return result if isinstance(result, dict) else {}

    def _claim_thread(self, result: dict[str, Any]) -> str:
        thread = result.get("thread")
        if not isinstance(thread, dict):
            raise ControlTransportError("thread/start response has no thread")
        thread_id = thread.get("id")
        if not isinstance(thread_id, str):
            raise ControlTransportError("thread/start response has no thread id")
        session_id = thread.get("sessionId")
        if isinstance(session_id, str):
            self._session_id = session_id
        approval_policy = result.get("approvalPolicy")
        if isinstance(approval_policy, str):
            self._approval_policy = approval_policy
        sandbox = result.get("sandbox")
        if isinstance(sandbox, dict):
            self._sandbox_policy = sandbox
        self._thread_id = thread_id
        return thread_id

    async def _handshake(self) -> dict[str, Any]:
        self._ensure_reader()
        message = await self._call(
            INITIALIZE_METHOD,
            {
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
                "capabilities": {"experimentalApi": True},
            },
        )
        result = self._result_or_failure(message, ControlTransportError)
        async with asyncio.timeout(RPC_RESPONSE_TIMEOUT_SECONDS):
            await self._send({"jsonrpc": JSONRPC_VERSION, "method": INITIALIZED_NOTIFICATION})
        return result

    async def _start_turn(self, thread_id: str, content: str | UserPrompt) -> PromptOutcome:
        message = await self._call(TURN_START_METHOD, self._turn_params(thread_id, content))
        result = self._result_or_failure(message, PromptDeliveryFailedError)
        turn_id = _turn_id_of(result)
        if turn_id is not None:
            self._active_turn_id = turn_id
        return PromptOutcome(state=PromptState.QUEUED)

    async def _steer(self, thread_id: str, content: str | UserPrompt) -> PromptOutcome:
        active_turn_id = self._active_turn_id
        if active_turn_id is None:
            raise SteerNoActiveTurnError("no active turn to steer")
        params = self._turn_params(thread_id, content)
        params["expectedTurnId"] = active_turn_id
        message = await self._call(TURN_STEER_METHOD, params)
        payload = message.get("error")
        if isinstance(payload, dict):
            self._handle_steer_failure(_error_text(payload))
        return PromptOutcome(state=PromptState.STEERED)

    def _handle_steer_failure(self, text: str) -> None:
        self._active_turn_id = None
        if any(marker in text for marker in _STEER_FAILURE_MARKERS):
            raise SteerNoActiveTurnError(text) from None
        raise PromptDeliveryFailedError(text) from None

    def _turn_params(self, thread_id: str, content: str | UserPrompt) -> dict[str, Any]:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": native_parts(content, "codex"),
        }
        if self._approval_policy is not None:
            params["approvalPolicy"] = self._approval_policy
        if self._sandbox_policy is not None:
            params["sandboxPolicy"] = self._sandbox_policy
        return params

    async def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_reader()
        self._next_request_id += 1
        request_id = self._next_request_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        started = asyncio.get_running_loop().time()
        logger.debug(
            "codex request started: method=%s id=%s thread=%s", method, request_id, self._thread_id
        )
        try:
            async with asyncio.timeout(RPC_RESPONSE_TIMEOUT_SECONDS):
                await self._send(
                    {
                        "jsonrpc": JSONRPC_VERSION,
                        "id": request_id,
                        "method": method,
                        "params": params,
                    }
                )
                return await future
        except TimeoutError:
            logger.warning("codex request timed out: method=%s id=%s", method, request_id)
            raise
        finally:
            self._pending.pop(request_id, None)
            logger.debug(
                "codex request finished: method=%s id=%s elapsed=%.3fs",
                method,
                request_id,
                asyncio.get_running_loop().time() - started,
            )

    async def _send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, separators=(",", ":"))
        try:
            self._stdin.write((line + "\n").encode("utf-8"))
            await self._stdin.drain()
        except (ConnectionError, OSError) as error:
            raise ControlTransportError(f"codex control write failed: {error}") from error

    def _result_or_failure(
        self, message: dict[str, Any], error_type: type[Exception]
    ) -> dict[str, Any]:
        payload = message.get("error")
        if isinstance(payload, dict):
            raise error_type(_error_text(payload)) from None
        result = message.get("result")
        return result if isinstance(result, dict) else {}

    def _ensure_reader(self) -> None:
        if self._reader_task is None or self._reader_task.done():
            self._reader_task = asyncio.get_running_loop().create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            async for event in self._stdout_pump.lines():
                if event.kind is not LineEventKind.LINE:
                    continue
                try:
                    message = json.loads(event.text)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict):
                    self._dispatch(message)
        finally:
            self._fail_pending()
            self._server_requests.put_nowait(None)

    def _dispatch(self, message: dict[str, Any]) -> None:
        if "method" in message:
            if "id" in message:
                identifier = message["id"]
                reference = request_reference(identifier)
                if reference is None:
                    return
                self._native_request_ids[reference] = identifier
                self._server_requests.put_nowait(message)
            else:
                self._observe_notification(message)
            return
        self._resolve_pending(message)

    def _resolve_pending(self, message: dict[str, Any]) -> None:
        raw_id = message.get("id")
        message_id = raw_id if isinstance(raw_id, (int, str)) else None
        if message_id is None:
            return
        future = self._pending.pop(message_id, None)
        if future is not None and not future.done():
            future.set_result(message)

    def _observe_notification(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params")
        if (
            method == "serverRequest/resolved"
            and isinstance(params, dict)
            and params.get("threadId") == self._thread_id
        ):
            reference = request_reference(params.get("requestId"))
            if reference is not None:
                self._native_request_ids.pop(reference, None)
            return
        if method not in (TURN_STARTED_NOTIFICATION, TURN_COMPLETED_NOTIFICATION):
            return
        if not isinstance(params, dict) or params.get("threadId") != self._thread_id:
            return
        turn_id = _turn_id_of(params)
        if turn_id is None:
            return
        if method == TURN_STARTED_NOTIFICATION:
            self._active_turn_id = turn_id
        else:
            turn = params.get("turn")
            if isinstance(turn, dict):
                self._commands.observe(turn)
            if turn_id == self._active_turn_id:
                self._active_turn_id = None

    def _fail_pending(self) -> None:
        self._commands.close()
        self._native_request_ids.clear()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ControlTransportError("codex control stream closed"))
        self._pending.clear()
