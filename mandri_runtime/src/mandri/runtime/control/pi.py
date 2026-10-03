import asyncio
import base64
import contextlib
import dataclasses
import inspect
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from mandri.core.ids import ApprovalDecision, ApprovalKind, HarnessSessionId, ModeApplication
from mandri.core.pi import GATEWAY_MODEL_ID, GATEWAY_PROVIDER
from mandri.core.ports.control import ControlRequest, ControlSink, PromptOutcome, PromptState
from mandri.core.types.prompt import UserPrompt, prompt_text
from mandri.runtime.control.errors import (
    ControlTransportError,
    HarnessNotInitializedError,
    ModeRejectedError,
    PromptDeliveryFailedError,
    PromptDeliveryUnknownError,
    ThreadOwnershipError,
)
from mandri.runtime.control.pi_commands import PiCommands
from mandri.runtime.control.pi_rpc import response_data
from mandri.runtime.pump import LineEventKind, LinePump

DIALOG_METHODS = frozenset({"select", "confirm", "input", "editor"})
RPC_RESPONSE_TIMEOUT_SECONDS = 60.0
ALLOW_DECISIONS = frozenset(
    {
        ApprovalDecision.ALLOW,
        ApprovalDecision.ONCE,
        ApprovalDecision.ALWAYS,
        ApprovalDecision.ACCEPT,
        ApprovalDecision.ACCEPT_FOR_SESSION,
    }
)


@dataclasses.dataclass(frozen=True)
class PiControlRequest(ControlRequest):
    kind: ApprovalKind = ApprovalKind.USER_INPUT
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)


class PiControlAdapter:
    def __init__(
        self,
        stdout_pump: LinePump,
        stdin: ControlSink,
        expected_session_id: HarnessSessionId | None = None,
        thinking_level: str | None = None,
        gateway_mode: bool = False,
        on_identity: Callable[[HarnessSessionId], None] | None = None,
        on_conversation_reset: Callable[[HarnessSessionId], Awaitable[None] | None] | None = None,
        on_session_path: Callable[[HarnessSessionId, str], None] | None = None,
        on_model_selection: Callable[[str, str | None], Awaitable[None] | None] | None = None,
    ) -> None:
        self._stdout_pump = stdout_pump
        self._stdin = stdin
        self._expected_session_id = expected_session_id
        self._thinking_level = thinking_level
        self._gateway_mode = gateway_mode
        self._on_identity = on_identity
        self._on_conversation_reset = on_conversation_reset
        self._on_session_path = on_session_path
        self._on_model_selection = on_model_selection
        self._model_selection: tuple[str, str | None] | None = None
        self._model_selection_initialized = False
        self._session_id: HarnessSessionId | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._dialogs: dict[str, tuple[dict[str, Any], float | None]] = {}
        self._requests: asyncio.Queue[PiControlRequest | None] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None
        self._refresh: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._identity_lock = asyncio.Lock()
        self._session_update_lock = asyncio.Lock()
        self._request_id = 0
        self._mode_ack: tuple[str, asyncio.Future[None]] | None = None
        self._mode_lock = asyncio.Lock()
        self._closed = False
        self._commands = PiCommands(self._call, gateway_mode)

    async def capture_identity(self) -> HarnessSessionId:
        async with self._identity_lock:
            return await self._initialize_identity()

    async def _initialize_identity(self) -> HarnessSessionId:
        if self._session_id is not None:
            return self._session_id
        state = await self._call("get_state", {})
        identifier = state.get("sessionId")
        if not isinstance(identifier, str) or not identifier:
            raise ControlTransportError("Pi did not return a session identity")
        if self._expected_session_id is not None and identifier != self._expected_session_id:
            raise ThreadOwnershipError("Pi resumed a different native session")
        self._validate_gateway(state)
        if self._thinking_level is not None:
            result = await self._call("set_thinking_level", {"level": self._thinking_level})
            state["thinkingLevel"] = result.get("level", self._thinking_level)
        await self._update_identity(state)
        return HarnessSessionId(identifier)

    async def next_native_request(self) -> PiControlRequest:
        self._ensure_reader()
        request = await self._requests.get()
        if request is None:
            raise ControlTransportError("Pi closed its control stream")
        return request

    async def send_prompt(self, content: str | UserPrompt) -> PromptOutcome:
        self._require_session()
        state = await self._call("get_state", {})
        await self._update_identity(state)
        self._validate_gateway(state)
        params: dict[str, Any] = {
            "message": prompt_text(content),
            "streamingBehavior": "steer",
        }
        if isinstance(content, UserPrompt):
            images = [
                {
                    "type": "image",
                    "data": base64.b64encode(attachment.data).decode("ascii"),
                    "mimeType": attachment.media_type,
                }
                for attachment in content.attachments
                if attachment.media_type.startswith("image/")
            ]
            if images:
                params["images"] = images
        try:
            result = await self._call("prompt", params)
            if result.get("disposition") in {None, "handled"}:
                await self._update_identity(await self._call("get_state", {}))
        except (ControlTransportError, TimeoutError) as error:
            raise PromptDeliveryUnknownError(str(error)) from error
        return PromptOutcome(
            state=PromptState.STEERED if state.get("isStreaming") else PromptState.QUEUED
        )

    async def list_commands(self) -> list[dict[str, Any]]:
        return await self._commands.list_commands()

    async def execute_command(self, identifier: str, arguments: str) -> dict[str, Any]:
        self._require_session()
        state = await self._call("get_state", {})
        await self._update_identity(state)
        self._validate_gateway(state)
        result = await self._commands.execute_command(identifier, arguments)
        await self._update_identity(await self._call("get_state", {}))
        return result

    async def answer_approval(
        self, native_request_ref: str, decision: ApprovalDecision | None
    ) -> bool:
        dialog = self._dialogs.get(native_request_ref)
        if dialog is None:
            return False
        if decision is None or decision is ApprovalDecision.CANCEL:
            result: dict[str, Any] = {"cancelled": True}
        elif dialog[0].get("method") == "confirm":
            result = {"confirmed": decision in ALLOW_DECISIONS}
        else:
            result = {"cancelled": True}
        return await self.answer_native_request(native_request_ref, result)

    async def answer_native_request(self, reference: str, result: dict[str, Any]) -> bool:
        pending = self._dialogs.get(reference)
        if pending is None:
            return False
        _, deadline = pending
        if deadline is not None and asyncio.get_running_loop().time() >= deadline:
            self._dialogs.pop(reference, None)
            return False
        await self._send({**result, "type": "extension_ui_response", "id": reference})
        self._dialogs.pop(reference, None)
        return True

    async def set_mode(self, mode: str) -> ModeApplication:
        self._require_session()
        if mode not in {"default", "acceptEdits", "plan", "bypassPermissions"}:
            raise ModeRejectedError("Unsupported Pi permission mode")
        async with self._mode_lock:
            commands = await self._call("get_commands", {})
            if not any(
                isinstance(row, dict)
                and row.get("name") == "mandri-permissions"
                and row.get("source") == "extension"
                for row in commands.get("commands", [])
            ):
                return ModeApplication.REQUIRES_RESTART
            token = uuid.uuid4().hex
            acknowledgement: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._mode_ack = (f"{token}:{mode}", acknowledgement)
            try:
                async with asyncio.timeout(15):
                    await self._call("prompt", {"message": f"/mandri-permissions {mode} {token}"})
                    await acknowledgement
            except TimeoutError as error:
                raise ControlTransportError(
                    "Pi did not acknowledge the permission change"
                ) from error
            finally:
                self._mode_ack = None
        return ModeApplication.MID_SESSION_APPLIED

    async def interrupt(self) -> bool:
        self._require_session()
        await self._call("clear_queue", {})
        await self._call("abort", {})
        return True

    async def aclose(self) -> None:
        self._stdout_pump.close()
        self._closed = True
        if self._refresh is not None:
            self._refresh.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh
            self._refresh = None
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        self._fail_pending()

    def _require_session(self) -> None:
        if self._session_id is None:
            raise HarnessNotInitializedError("Pi control used before identity capture")

    def _validate_gateway(self, state: dict[str, Any]) -> None:
        if not self._gateway_mode:
            return
        model = state.get("model")
        if not isinstance(model, dict) or (model.get("provider"), model.get("id")) != (
            GATEWAY_PROVIDER,
            GATEWAY_MODEL_ID,
        ):
            raise ControlTransportError("Pi selected a model outside the Mandri gateway")

    async def _update_identity(self, state: dict[str, Any]) -> None:
        async with self._session_update_lock:
            identifier = state.get("sessionId")
            if not isinstance(identifier, str) or not identifier:
                return
            native_id = HarnessSessionId(identifier)
            if identifier != self._session_id:
                if self._session_id is not None and self._gateway_mode:
                    state["model"] = await self._call(
                        "set_model", {"provider": GATEWAY_PROVIDER, "modelId": GATEWAY_MODEL_ID}
                    )
                    if self._thinking_level is not None:
                        await self._call("set_thinking_level", {"level": self._thinking_level})
                if self._session_id is not None and self._on_conversation_reset is not None:
                    result = self._on_conversation_reset(native_id)
                    if inspect.isawaitable(result):
                        await result
                elif self._on_identity is not None:
                    self._on_identity(native_id)
                self._session_id = native_id
            path = state.get("sessionFile")
            if isinstance(path, str) and path and self._on_session_path is not None:
                self._on_session_path(native_id, path)
            await self._update_model_selection(state)

    async def _update_model_selection(self, state: dict[str, Any]) -> None:
        if self._gateway_mode:
            return
        model = state.get("model")
        selection = None
        if isinstance(model, dict):
            provider, identifier = model.get("provider"), model.get("id")
            if (
                isinstance(provider, str)
                and provider
                and isinstance(identifier, str)
                and identifier
            ):
                thinking = state.get("thinkingLevel")
                selection = (
                    f"{provider}/{identifier}",
                    thinking if isinstance(thinking, str) else None,
                )
        if not self._model_selection_initialized:
            self._model_selection = selection
            self._model_selection_initialized = True
            return
        if selection is None or selection == self._model_selection:
            return
        if self._on_model_selection is not None:
            result = self._on_model_selection(*selection)
            if inspect.isawaitable(result):
                await result
        self._model_selection = selection

    async def _call(self, command: str, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_reader()
        self._request_id += 1
        identifier = str(self._request_id)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[identifier] = future
        timeout = None if command in {"prompt", "compact", "bash"} else RPC_RESPONSE_TIMEOUT_SECONDS
        try:
            async with asyncio.timeout(timeout):
                await self._send({**params, "type": command, "id": identifier})
                record = await future
            if (
                command == "prompt"
                and record.get("command") == command
                and record.get("success") is False
            ):
                raise PromptDeliveryFailedError(str(record.get("error") or "Pi rejected prompt"))
            return response_data(record, command)
        finally:
            self._pending.pop(identifier, None)

    async def _send(self, record: dict[str, Any]) -> None:
        async with self._write_lock:
            try:
                self._stdin.write((json.dumps(record) + "\n").encode("utf-8"))
                await self._stdin.drain()
            except (ConnectionError, OSError) as error:
                raise ControlTransportError(f"Pi control write failed: {error}") from error

    def _ensure_reader(self) -> None:
        if self._closed:
            raise ControlTransportError("Pi control stream is closed")
        if self._reader is None:
            self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            async for event in self._stdout_pump.lines():
                if event.kind is not LineEventKind.LINE:
                    continue
                try:
                    record = json.loads(event.text)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    self._dispatch(record)
        finally:
            self._closed = True
            self._fail_pending()

    def _dispatch(self, record: dict[str, Any]) -> None:
        if (
            self._mode_ack is not None
            and record.get("type") == "extension_ui_request"
            and record.get("method") == "setStatus"
            and record.get("statusKey") == "_mandri_permissions"
            and record.get("statusText") == self._mode_ack[0]
            and not self._mode_ack[1].done()
        ):
            self._mode_ack[1].set_result(None)
        self._commands.observe(record)
        if record.get("type") in {
            "agent_settled",
            "session_info_changed",
            "thinking_level_changed",
        } and (self._refresh is None or self._refresh.done()):
            self._refresh = asyncio.create_task(self._refresh_identity())
        identifier = record.get("id")
        if not isinstance(identifier, str):
            return
        if record.get("type") == "response":
            future = self._pending.get(identifier)
            if future is not None and not future.done():
                future.set_result(record)
        elif (
            record.get("type") == "extension_ui_request" and record.get("method") in DIALOG_METHODS
        ):
            timeout = record.get("timeout")
            deadline = (
                asyncio.get_running_loop().time() + max(timeout, 0) / 1000
                if isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
                else None
            )
            self._dialogs[identifier] = (record, deadline)
            self._requests.put_nowait(
                PiControlRequest(
                    native_request_ref=identifier,
                    tool_name=record["method"],
                    tool_input=record,
                    tool_use_id=identifier,
                    kind=ApprovalKind.PERMISSION_SCOPE
                    if record["method"] == "confirm"
                    else ApprovalKind.USER_INPUT,
                    detail=record,
                )
            )

    async def _refresh_identity(self) -> None:
        if self._session_id is None:
            return
        with contextlib.suppress(ControlTransportError, TimeoutError):
            await self._update_identity(await self._call("get_state", {}))

    def _fail_pending(self) -> None:
        if self._mode_ack is not None and not self._mode_ack[1].done():
            self._mode_ack[1].set_exception(ControlTransportError("Pi control stream closed"))
        self._commands.close()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ControlTransportError("Pi closed its control stream"))
        self._pending.clear()
        self._dialogs.clear()
        self._requests.put_nowait(None)
