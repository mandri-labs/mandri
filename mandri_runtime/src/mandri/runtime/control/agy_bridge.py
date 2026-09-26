import asyncio
import hmac
import json
import secrets
from collections.abc import Awaitable, Callable
from typing import Any

from mandri.core.ids import ApprovalDecision, ApprovalStatus
from mandri.core.types.approvals import ApprovalRequest
from mandri.runtime.control.agy_policy import AgyPolicy, tool_key
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.question_answers import valid_question_answers

_ALLOW = frozenset(
    {
        ApprovalDecision.ALLOW,
        ApprovalDecision.ONCE,
        ApprovalDecision.ALWAYS,
        ApprovalDecision.ACCEPT,
        ApprovalDecision.ACCEPT_FOR_SESSION,
    }
)


class AgyBridge:
    def __init__(
        self,
        policy: AgyPolicy,
        publish: Callable[[dict[str, Any]], Awaitable[None]],
        timeout: int = 120,
    ) -> None:
        self.token = secrets.token_urlsafe(32)
        self.policy = policy
        self._publish = publish
        self._timeout = timeout
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._calls: dict[str, tuple[str, dict[str, Any]]] = {}
        self._replies: dict[str, dict[str, Any]] = {}
        self._injections: list[dict[str, str]] = []
        self._closed = False

    def authenticates(self, token: str) -> bool:
        return not self._closed and hmac.compare_digest(token, self.token)

    async def handle(self, event: str, data: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise ControlTransportError("Antigravity hook session is closed")
        if event not in {"PreInvocation", "PostInvocation", "PreToolUse", "PostToolUse", "Stop"}:
            raise ControlTransportError("Unknown Antigravity hook")
        await self._publish({"event": "hook", "hook": event, "data": data})
        if event == "PreInvocation":
            injections, self._injections = self._injections, []
            return {"injectSteps": injections}
        if event != "PreToolUse":
            return {}
        call = data.get("toolCall")
        if not isinstance(call, dict) or not isinstance(call.get("args"), dict):
            return {"decision": "deny", "reason": "Invalid tool request"}
        name, args = call.get("name"), call["args"]
        if not isinstance(name, str):
            return {"decision": "deny", "reason": "Missing tool name"}
        conversation, step = data.get("conversationId"), data.get("stepIdx")
        if (
            not isinstance(conversation, str)
            or not conversation
            or type(step) is not int
            or step < 0
        ):
            return {"decision": "deny", "reason": "Invalid tool identity"}
        ref = f"{conversation}:{step}"
        previous = self._calls.get(ref)
        if previous is not None and tool_key(*previous) != tool_key(name, args):
            return {"decision": "deny", "reason": "Tool identity collision"}
        if ref in self._replies:
            return self._replies[ref]
        decision = self.policy.decision(name, args)
        if decision != "ask":
            return {"decision": decision}
        pending = self._pending.get(ref)
        if pending is None:
            pending = asyncio.get_running_loop().create_future()
            self._pending[ref] = pending
            self._calls[ref] = (name, args)
            await self._publish(
                {
                    "event": "approval_request",
                    "request_id": ref,
                    "tool_name": name,
                    "input": args,
                    "data": data,
                }
            )
        try:
            reply = await asyncio.wait_for(asyncio.shield(pending), self._timeout + 5)
        except TimeoutError:
            reply = {"decision": "deny", "reason": "Mandri approval expired"}
        self._replies[ref] = reply
        self._pending.pop(ref, None)
        await self._publish(
            {
                "event": "approval_response",
                "request_id": ref,
                "response": reply,
                "data": {"conversationId": conversation, "stepIdx": step},
            }
        )
        return reply

    async def deliver(self, request: ApprovalRequest) -> bool:
        ref = request.native_request_ref
        pending = self._pending.get(ref)
        if pending is None or pending.done():
            return False
        name, args = self._calls[ref]
        allowed = request.status is ApprovalStatus.ANSWERED and request.decision in _ALLOW
        if name == "ask_question":
            if allowed and not valid_question_answers(request.answers):
                return False
            if allowed and request.answers:
                self._injections.append(
                    {"userMessage": json.dumps(request.answers, ensure_ascii=False)}
                )
            reply = {
                "decision": "deny",
                "reason": "Question handled by Mandri" if allowed else "Question cancelled",
            }
        else:
            reply = {"decision": "allow" if allowed else "deny"}
            if allowed and request.decision in {
                ApprovalDecision.ALWAYS,
                ApprovalDecision.ACCEPT_FOR_SESSION,
            }:
                self.policy.grants.add(tool_key(name, args))
        pending.set_result(reply)
        return True

    async def aclose(self) -> None:
        self._closed = True
        for pending in self._pending.values():
            if not pending.done():
                pending.set_result({"decision": "deny", "reason": "Session closed"})
