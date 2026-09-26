"""OpenCode HTTP+SSE harness control adapter."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
from collections.abc import AsyncIterator
from typing import Any, final
from urllib.parse import quote

import httpx
from mandri.core.ids import (
    ApprovalDecision,
    ApprovalKind,
    ApprovalStatus,
    HarnessSessionId,
    ModeApplication,
)
from mandri.core.opencode import gateway_model
from mandri.core.ports.control import ControlRequest, HarnessControl
from mandri.core.types.approvals import ApprovalRequest
from mandri.core.types.prompt import UserPrompt
from mandri.runtime.control import PromptOutcome, PromptState
from mandri.runtime.control.agents.opencode import OpencodeAgentControl
from mandri.runtime.control.errors import (
    ControlTransportError,
    ModeRejectedError,
    PromptDeliveryFailedError,
)
from mandri.runtime.control.modes import OPENCODE_PERMISSION_RULES
from mandri.runtime.control.opencode_commands import OpencodeCommands
from mandri.runtime.control.opencode_questions import question_answers, recovered_inputs
from mandri.runtime.control.prompt import native_parts

_REQUEST_TIMEOUT = httpx.Timeout(None)
_STREAM_TIMEOUT = httpx.Timeout(10.0, read=None)
_RECONNECT_DELAY_SECONDS = 0.5
_PERMISSION_EVENT_TYPES = ("permission.asked", "permission.v2.asked")
_PERMISSION_REPLY: dict[ApprovalDecision, str] = {
    ApprovalDecision.ALLOW: "once",
    ApprovalDecision.ONCE: "once",
    ApprovalDecision.ACCEPT: "once",
    ApprovalDecision.ALWAYS: "always",
    ApprovalDecision.ACCEPT_FOR_SESSION: "always",
    ApprovalDecision.DENY: "reject",
    ApprovalDecision.CANCEL: "reject",
}
_EXPIRED_REPLY = "reject"


def _reply_for(decision: ApprovalDecision | None) -> str:
    if decision is None:
        return _EXPIRED_REPLY
    return _PERMISSION_REPLY[decision]


def _is_ok(status_code: int) -> bool:
    return 200 <= status_code < 300


@dataclasses.dataclass(frozen=True)
class OpencodeControlRequest(ControlRequest):
    kind: ApprovalKind = ApprovalKind.PERMISSION_SCOPE
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)


def _parse_sse_event(line: str) -> dict[str, Any] | None:
    if not line.startswith("data: "):
        return None
    try:
        event = json.loads(line[len("data: ") :])
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _normalize_permission_event(
    event: dict[str, Any], session_id: HarnessSessionId
) -> OpencodeControlRequest | None:
    if event.get("type") not in (*_PERMISSION_EVENT_TYPES, "question.asked"):
        return None
    properties = event.get("properties")
    if not isinstance(properties, dict):
        return None
    owner = properties.get("sessionID")
    if owner != session_id:
        return None
    reference = properties.get("id")
    if not isinstance(reference, str) or not reference:
        return None
    return OpencodeControlRequest(
        native_request_ref=reference,
        tool_name=str(properties.get("tool", "permission")),
        tool_input=properties,
        tool_use_id=reference,
        kind=(
            ApprovalKind.USER_INPUT
            if event.get("type") == "question.asked"
            else ApprovalKind.PERMISSION_SCOPE
        ),
        detail=properties,
    )


class OpencodeControlAdapter(HarnessControl):
    def __init__(
        self,
        base_url: str,
        session_id: str,
        events: AsyncIterator[dict[str, Any]] | None = None,
        auth: tuple[str, str] | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=_REQUEST_TIMEOUT, auth=auth, trust_env=False
        )
        self._session_id = HarnessSessionId(session_id)
        self._events = events
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._pump_task: asyncio.Task[None] | None = None
        self._recovered = False
        self._seen: set[str] = set()
        self._closing = False
        self.agents = OpencodeAgentControl(self._request, str(self._session_id))

    async def capture_identity(self) -> HarnessSessionId:
        response = await self._request("GET", f"/session/{self._session_id}")
        if not _is_ok(response.status_code):
            raise ControlTransportError(
                f"opencode session {self._session_id} is unavailable: status {response.status_code}"
            )
        return self._session_id

    async def next_native_request(self) -> OpencodeControlRequest:
        while True:
            raw = await self.next_event()
            if raw is None:
                raise ControlTransportError("opencode event stream closed")
            request = _normalize_permission_event(raw, self._session_id)
            if request is None or request.native_request_ref in self._seen:
                continue
            self._seen.add(request.native_request_ref)
            return request

    async def next_event(self) -> dict[str, Any] | None:
        """Return the next SSE event verbatim, or None once the stream closes."""
        self._ensure_event_pump()
        if not self._recovered:
            self._recovered = True
            await self._recover_pending_permissions()
        return await self._queue.get()

    async def answer_approval(
        self, native_request_ref: str, decision: ApprovalDecision | None
    ) -> bool:
        reply = _reply_for(decision)
        response = await self._request(
            "POST", f"/permission/{native_request_ref}/reply", {"reply": reply}
        )
        if response.status_code == 404:
            return False
        if not _is_ok(response.status_code):
            raise ControlTransportError(
                f"opencode permission reply failed: status {response.status_code}"
            )
        return True

    async def set_mode(self, mode: str) -> ModeApplication:
        rules = OPENCODE_PERMISSION_RULES.get(mode)
        if rules is None:
            raise ModeRejectedError(f"opencode does not support mode: {mode}")
        response = await self._request(
            "PATCH",
            f"/session/{self._session_id}",
            {
                "permission": [
                    {"permission": name, "pattern": "*", "action": action}
                    for name, action in rules.items()
                ]
            },
        )
        if not _is_ok(response.status_code):
            raise ModeRejectedError(f"opencode rejected mode {mode}: status {response.status_code}")
        return ModeApplication.MID_SESSION_APPLIED

    async def send_prompt(self, text: str | UserPrompt) -> PromptOutcome:
        body = {"parts": native_parts(text, "opencode"), "model": gateway_model()}
        response = await self._request("POST", f"/session/{self._session_id}/prompt_async", body)
        if not _is_ok(response.status_code):
            raise PromptDeliveryFailedError(
                f"opencode prompt delivery failed: status {response.status_code}"
            )
        return PromptOutcome(state=PromptState.STEERED)

    async def list_commands(self) -> list[dict[str, Any]]:
        async with asyncio.timeout(15):
            return await OpencodeCommands(self._request, str(self._session_id)).list_commands()

    async def execute_command(self, command_id: str, arguments: str) -> dict[str, Any]:
        return await OpencodeCommands(self._request, str(self._session_id)).execute_command(
            command_id, arguments
        )

    async def answer_question(
        self, native_request_ref: str, answers: list[list[str]] | None
    ) -> bool:
        action = "reject" if answers is None else "reply"
        response = await self._request(
            "POST",
            f"/question/{quote(native_request_ref, safe='')}/{action}",
            None if answers is None else {"answers": answers},
        )
        if response.status_code == 404:
            return False
        if not _is_ok(response.status_code):
            raise ControlTransportError(f"OpenCode question reply failed ({response.status_code})")
        return True

    async def interrupt(self) -> bool:
        response = await self._request("POST", f"/session/{self._session_id}/abort")
        return _is_ok(response.status_code)

    async def aclose(self) -> None:
        self._closing = True
        task = self._pump_task
        self._pump_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
                await task
        self._queue.put_nowait(None)
        await self._client.aclose()

    async def _request(
        self, method: str, path: str, json_body: dict[str, Any] | None = None
    ) -> httpx.Response:
        try:
            return await self._client.request(method, path, json=json_body)
        except httpx.HTTPError as error:
            raise ControlTransportError(f"opencode control request failed: {error}") from error

    def _ensure_event_pump(self) -> None:
        if self._pump_task is not None:
            return
        if self._events is not None:
            self._pump_task = asyncio.create_task(self._consume_injected_events())
        else:
            self._pump_task = asyncio.create_task(self._consume_sse_events())

    async def _consume_injected_events(self) -> None:
        if self._events is None:
            return
        async for event in self._events:
            if isinstance(event, dict):
                self._queue.put_nowait(event)

    async def _consume_sse_events(self) -> None:
        while not self._closing:
            try:
                async with self._client.stream(
                    "GET", "/event", timeout=_STREAM_TIMEOUT
                ) as response:
                    if not _is_ok(response.status_code):
                        await asyncio.sleep(_RECONNECT_DELAY_SECONDS)
                        continue
                    await self._recover_pending_permissions()
                    async for line in response.aiter_lines():
                        event = _parse_sse_event(line)
                        if event is not None:
                            self._queue.put_nowait(event)
            except (httpx.HTTPError, ControlTransportError):
                if self._closing:
                    return
                await asyncio.sleep(_RECONNECT_DELAY_SECONDS)

    async def _recover_pending_permissions(self) -> None:
        for path, event_type in (
            ("/permission", "permission.asked"),
            ("/question", "question.asked"),
        ):
            response = await self._request("GET", path)
            if path == "/question" and response.status_code == 404:
                continue
            if not _is_ok(response.status_code):
                raise ControlTransportError(
                    f"OpenCode pending input recovery failed ({response.status_code})"
                )
            try:
                records = response.json()
            except ValueError:
                raise ControlTransportError("OpenCode returned invalid pending inputs") from None
            for event in recovered_inputs(records, event_type, str(self._session_id)):
                self._queue.put_nowait(event)


@final
class OpencodeApprovalDelivery:
    """Delivers approval decisions to an opencode harness as permission replies."""

    def __init__(self, control: OpencodeControlAdapter) -> None:
        self._control = control

    async def deliver(self, request: ApprovalRequest) -> bool:
        if request.kind is ApprovalKind.USER_INPUT:
            allowed = request.status is ApprovalStatus.ANSWERED and request.decision not in (
                None,
                ApprovalDecision.DENY,
                ApprovalDecision.CANCEL,
            )
            answers = question_answers(request) if allowed else None
            return await self._control.answer_question(request.native_request_ref, answers)
        return await self._control.answer_approval(request.native_request_ref, request.decision)
