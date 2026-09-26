"""Approval lifecycle service: registration, decisions, cancellation, expiry."""

import asyncio
import dataclasses
import json
import typing
import uuid
from typing import Any, final

from mandri.core.ids import (
    ApprovalDecision,
    ApprovalId,
    ApprovalKind,
    ApprovalStatus,
    EpochMs,
    HarnessKind,
    RawEvent,
    SessionId,
)
from mandri.core.types.approvals import (
    ApprovalAlreadyAnsweredError,
    ApprovalExpiredError,
    ApprovalNotPendingError,
    ApprovalRequest,
)
from mandri.runtime.approvals.registry import ApprovalRegistry
from mandri.runtime.question_answers import valid_question_answers

ExpiryHandler = typing.Callable[[ApprovalRequest], None]


class ApprovalClock(typing.Protocol):
    def __call__(self) -> EpochMs: ...


@final
class ApprovalService:
    def __init__(
        self,
        registry: ApprovalRegistry,
        clock: ApprovalClock,
        on_expiry: ExpiryHandler,
    ) -> None:
        self._registry = registry
        self._clock = clock
        self._on_expiry = on_expiry
        self._lock = asyncio.Lock()

    async def register(
        self,
        session_id: SessionId,
        harness: HarnessKind,
        native_request: RawEvent,
        native_request_ref: str,
        kind: ApprovalKind,
        timeout_seconds: int,
    ) -> ApprovalRequest:
        created_at = self._clock()
        timeout_ms = timeout_seconds * 1_000
        if harness is HarnessKind.PI:
            try:
                raw = json.loads(native_request)
            except (ValueError, TypeError):
                raw = {}
            native_timeout = raw.get("timeout") if isinstance(raw, dict) else None
            if isinstance(native_timeout, (int, float)) and not isinstance(native_timeout, bool):
                timeout_ms = min(timeout_ms, max(0, int(native_timeout)))
        request = ApprovalRequest(
            id=ApprovalId(str(uuid.uuid4())),
            session_id=session_id,
            harness=harness,
            native_request=native_request,
            native_request_ref=native_request_ref,
            kind=kind,
            created_at=created_at,
            deadline=EpochMs(created_at + timeout_ms),
            status=ApprovalStatus.PENDING,
            decision=None,
        )
        async with self._lock:
            self._registry.add(request)
        return request

    async def answer(
        self,
        approval_id: ApprovalId,
        decision: ApprovalDecision,
        updated_input: RawEvent | None = None,
        answers: list[dict[str, Any]] | None = None,
        deliver: typing.Callable[[ApprovalRequest], typing.Awaitable[None]] | None = None,
    ) -> ApprovalRequest:
        async with self._lock:
            request = self._require(approval_id)
            if request.status is ApprovalStatus.PENDING and self._clock() >= request.deadline:
                raise ApprovalExpiredError("The native interaction has expired")
            if request.status is ApprovalStatus.CANCELLED:
                raise ApprovalAlreadyAnsweredError(
                    f"approval {approval_id} already resolved as cancelled"
                )
            if (
                request.harness
                in {
                    HarnessKind.AGY,
                    HarnessKind.CLAUDE,
                    HarnessKind.OPENCODE,
                    HarnessKind.CODEX,
                    HarnessKind.PI,
                }
                and request.kind is ApprovalKind.USER_INPUT
                and decision
                in {
                    ApprovalDecision.ALLOW,
                    ApprovalDecision.ONCE,
                    ApprovalDecision.ALWAYS,
                    ApprovalDecision.ACCEPT,
                    ApprovalDecision.ACCEPT_FOR_SESSION,
                }
                and not valid_question_answers(answers)
            ):
                raise ApprovalNotPendingError("The native question requires non-empty answers")
            resolved = request.transition(ApprovalStatus.ANSWERED, decision)
            resolved = dataclasses.replace(resolved, answers=answers)
            if deliver is not None:
                await deliver(resolved)
            object.__setattr__(request, "answers", answers)
            return self._resolve(request, resolved)

    async def cancel(
        self,
        approval_id: ApprovalId,
        deliver: typing.Callable[[ApprovalRequest], typing.Awaitable[None]] | None = None,
    ) -> ApprovalRequest:
        async with self._lock:
            request = self._require(approval_id)
            resolved = request.transition(ApprovalStatus.CANCELLED)
            if deliver is not None:
                await deliver(resolved)
            return self._resolve(request, resolved)

    async def resolve_native(self, approval_id: ApprovalId) -> ApprovalRequest | None:
        async with self._lock:
            request = self._registry.get(approval_id)
            if request is None or request.status is not ApprovalStatus.PENDING:
                return None
            return self._resolve(request, request.transition(ApprovalStatus.CANCELLED))

    async def cancel_all_for_session(self, session_id: SessionId) -> list[ApprovalRequest]:
        async with self._lock:
            cancelled: list[ApprovalRequest] = []
            for request in self._registry.pending_for_session(session_id):
                resolved = request.transition(ApprovalStatus.CANCELLED)
                cancelled.append(self._resolve(request, resolved))
            return cancelled

    async def expire_due(self) -> list[ApprovalRequest]:
        async with self._lock:
            now = self._clock()
            expired: list[ApprovalRequest] = []
            for request in self._registry.pending():
                if now < request.deadline:
                    continue
                resolved = request.transition(ApprovalStatus.EXPIRED)
                expired.append(self._resolve(request, resolved))
            for request in expired:
                self._on_expiry(request)
            return expired

    def pending_for_session(self, session_id: SessionId) -> list[ApprovalRequest]:
        return self._registry.pending_for_session(session_id)

    def _require(self, approval_id: ApprovalId) -> ApprovalRequest:
        request = self._registry.get(approval_id)
        if request is None:
            raise ApprovalNotPendingError(f"approval {approval_id} is not pending")
        return request

    @staticmethod
    def _resolve(request: ApprovalRequest, resolved: ApprovalRequest) -> ApprovalRequest:
        object.__setattr__(request, "status", resolved.status)
        object.__setattr__(request, "decision", resolved.decision)
        return request
