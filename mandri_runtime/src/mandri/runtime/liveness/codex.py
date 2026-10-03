"""Codex evidence adapter translating session-feed frames into liveness evidence."""

import asyncio
import contextlib
import logging
import typing
from collections.abc import Awaitable, Callable
from typing import Any

from mandri.core.hub import Hub, SubscriberHandle, Topic
from mandri.core.ids import HarnessKind, SessionId
from mandri.runtime.codex_request_ids import request_reference
from mandri.runtime.control.codex_liveness import CodexLivenessSnapshot
from mandri.runtime.liveness.degradation import is_stream_degradation
from mandri.runtime.liveness.evidence import LivenessEvidenceKind
from mandri.runtime.liveness.observation import WorkEventContext
from mandri.runtime.liveness.port import LivenessPort

logger = logging.getLogger(__name__)

SOURCE = "codex"

TURN_STARTED_NOTIFICATION = "turn/started"
TURN_COMPLETED_NOTIFICATION = "turn/completed"
THREAD_STATUS_CHANGED_NOTIFICATION = "thread/status/changed"
THREAD_QUEUE_CHANGED_NOTIFICATION = "thread/queue/changed"
ITEM_STARTED_NOTIFICATION = "item/started"
ITEM_COMPLETED_NOTIFICATION = "item/completed"
ELICITATION_REQUEST_METHOD = "mcpServer/elicitation/request"

APPROVAL_REQUEST_SUFFIX = "requestApproval"
APPROVAL_RESPONSE_SUFFIX = "/response"

_ACTIVE_STATUS = "active"
_BACKGROUND_ITEM_MARKERS = ("subagent", "agentthread", "subagentactivity")
_DECISION_ID_KEYS = ("id", "requestId", "request_id")


@typing.final
class CodexLivenessAdapter:
    """Translate evidence for the managed thread without absorbing child turns."""

    def __init__(
        self,
        hub: Hub,
        topic: Topic,
        port: LivenessPort,
        session_id: SessionId,
        native_identity: Callable[[], str | None] | None = None,
        *,
        since: int = 0,
        read_queue: Callable[[], Awaitable[bool]] | None = None,
        read_state: Callable[[], Awaitable[CodexLivenessSnapshot]] | None = None,
    ) -> None:
        self._hub = hub
        self._topic = topic
        self._port = port
        self._session_id = session_id
        self._work_context = WorkEventContext(HarnessKind.CODEX)
        self._native_identity = native_identity
        self._task: asyncio.Task[None] | None = None
        self._handle: SubscriberHandle | None = None
        self._last_seq = since
        self._turn_open = False
        self._status_turn_open = False
        self._queue_open = False
        self._background_items: set[str] = set()
        self._children: set[str] = set()
        self._pending_approvals: set[str] = set()
        self._snapshot_approval_pending = False
        self._read_queue = read_queue
        self._read_state = read_state
        self._queue_task: asyncio.Task[None] | None = None
        self._state_task: asyncio.Task[None] | None = None
        self._queue_revision = 0
        self._revision = 0
        self._reconciliation_required = False

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.get_running_loop().create_task(
            self._consume(), name=f"codex-liveness:{self._session_id}"
        )

    async def stop(self) -> None:
        task = self._task
        self._task = None
        for pending in (self._queue_task, self._state_task):
            if pending is not None:
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _consume(self) -> None:
        try:
            while True:
                handle = self._hub.subscribe(self._topic, since=self._last_seq, internal=True)
                self._handle = handle
                ended = await self._run_pass(handle)
                self._hub.unsubscribe(handle)
                self._handle = None
                if ended:
                    self._observe_uncertain()
        finally:
            leftover = self._handle
            self._handle = None
            if leftover is not None:
                self._hub.unsubscribe(leftover)

    async def _run_pass(self, handle: SubscriberHandle) -> bool:
        for replayed in handle.replay:
            if self._absorb(replayed):
                return True
        while True:
            frame = await handle.queue.get()
            if frame is None:
                return True
            if self._absorb(frame):
                return True

    def _absorb(self, frame: dict[str, Any]) -> bool:
        if frame.get("type") == "gap":
            if frame.get("reason") == "slow_consumer":
                return True
            self._observe_uncertain()
            self._skip_lost(frame)
            return False
        self._last_seq = _frame_seq(frame)
        self._translate(frame.get("payload"))
        return False

    def _skip_lost(self, frame: dict[str, Any]) -> None:
        end = _frame_seq(frame)
        if end - 1 > self._last_seq:
            self._last_seq = end - 1

    def _translate(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        raw = payload.get("raw")
        if is_stream_degradation(payload):
            self._observe_uncertain()
            return
        if payload.get("type") == "approval.resolved" and isinstance(raw, dict):
            self._revision += 1
            reference = raw.get("native_request_ref")
            if isinstance(reference, str):
                kind = self._close_approval(reference)
                if kind is not None:
                    self._emit(kind)
            return
        if "type" in payload:
            return
        if payload.get("source") != SOURCE:
            return
        if not isinstance(raw, dict):
            return
        identity = self._native_identity() if self._native_identity else None
        params = raw.get("params")
        thread_id = params.get("threadId") if isinstance(params, dict) else None
        if identity is not None and isinstance(thread_id, str) and thread_id != identity:
            return
        if isinstance(raw.get("method"), str):
            self._revision += 1
        self._work_context.observe(raw, payload.get("ts"))
        try:
            kind = self._evidence_kind(raw)
        except Exception:
            logger.warning(
                "session %s dropped poison codex frame: %s",
                self._session_id,
                raw,
                exc_info=True,
            )
            self._observe_uncertain()
            return
        if kind is not None:
            if kind is LivenessEvidenceKind.STATE_UNCERTAIN:
                self._observe_uncertain()
            else:
                self._emit(kind)
        if isinstance(raw.get("method"), str) and raw["method"]:
            self._request_reconciliation()

    def _evidence_kind(self, message: dict[str, Any]) -> LivenessEvidenceKind | None:
        method = message.get("method")
        if not isinstance(method, str) or not method:
            return None
        if "id" in message:
            return self._opened_from_request(method, message)
        return self._kind_from_notification(method, _params_of(message))

    def _kind_from_notification(
        self, method: str, params: dict[str, Any]
    ) -> LivenessEvidenceKind | None:
        closed = self._closed_from_decision(method, params)
        if closed is not None:
            return closed
        if method == TURN_STARTED_NOTIFICATION:
            self._turn_open = True
            return LivenessEvidenceKind.TURN_STARTED
        if method == TURN_COMPLETED_NOTIFICATION:
            self._turn_open = False
            self._status_turn_open = False
            return LivenessEvidenceKind.TURN_ENDED
        if method == THREAD_STATUS_CHANGED_NOTIFICATION:
            return self._status_kind(params)
        if method == THREAD_QUEUE_CHANGED_NOTIFICATION:
            return self._queue_kind(params)
        return self._background_kind(method, params)

    def _status_kind(self, params: dict[str, Any]) -> LivenessEvidenceKind | None:
        status = params.get("status")
        status_type = status.get("type") if isinstance(status, dict) else status
        if isinstance(status, dict) and isinstance(status.get("activeFlags"), list):
            self._snapshot_approval_pending = bool(
                {"waitingOnApproval", "waitingOnUserInput"}.intersection(status["activeFlags"])
            )
            self._sync_approvals()
        if status_type == _ACTIVE_STATUS:
            if self._turn_open or self._status_turn_open:
                return None
            self._status_turn_open = True
            return LivenessEvidenceKind.TURN_STARTED
        if status_type != "idle":
            return LivenessEvidenceKind.STATE_UNCERTAIN
        self._snapshot_approval_pending = False
        self._sync_approvals()
        was_open = self._turn_open or self._status_turn_open
        self._turn_open = False
        self._status_turn_open = False
        if was_open:
            return LivenessEvidenceKind.TURN_ENDED
        return None

    def _queue_kind(self, params: dict[str, Any]) -> LivenessEvidenceKind | None:
        self._queue_open = True
        self._queue_revision += 1
        if self._read_queue is not None:
            if self._queue_task is None or self._queue_task.done():
                self._queue_task = asyncio.create_task(self._refresh_queue())
        else:
            self._observe_uncertain()
        return LivenessEvidenceKind.BACKGROUND_STARTED

    async def _refresh_queue(self) -> None:
        if self._read_queue is None:
            return
        try:
            while True:
                revision = self._queue_revision
                queued = await self._read_queue()
                if revision != self._queue_revision:
                    continue
                self._queue_open = queued
                self._emit(self._background_state())
                return
        except Exception:
            logger.warning("session %s queue state unavailable", self._session_id, exc_info=True)
            self._observe_uncertain()

    def _background_kind(self, method: str, params: dict[str, Any]) -> LivenessEvidenceKind | None:
        item = params.get("item")
        if not isinstance(item, dict):
            return None
        if item.get("type") == "subAgentActivity":
            child = item.get("agentThreadId")
            if not isinstance(child, str):
                return LivenessEvidenceKind.STATE_UNCERTAIN
            if item.get("kind") in {"completed", "interrupted"}:
                self._children.discard(child)
            elif item.get("kind") in {"started", "interacted"}:
                self._children.add(child)
            return self._background_state()
        if item.get("type") == "collabAgentToolCall":
            states = item.get("agentsStates")
            if isinstance(states, dict):
                for child, state in states.items():
                    status = state.get("status") if isinstance(state, dict) else None
                    if status in {"pendingInit", "running"}:
                        self._children.add(child)
                    elif status in {"completed", "interrupted", "errored", "shutdown", "notFound"}:
                        self._children.discard(child)
            return self._background_state()
        haystack = " ".join([method, _item_text(item, "type"), _item_text(item, "kind")]).lower()
        if not any(marker in haystack for marker in _BACKGROUND_ITEM_MARKERS):
            return None
        identifier = _item_text(item, "id")
        if not identifier:
            return LivenessEvidenceKind.STATE_UNCERTAIN
        if method == ITEM_COMPLETED_NOTIFICATION or _item_text(item, "kind") == "completed":
            self._background_items.discard(identifier)
            return self._background_state()
        if method == ITEM_STARTED_NOTIFICATION or _item_text(item, "kind") == "started":
            self._background_items.add(identifier)
            return LivenessEvidenceKind.BACKGROUND_STARTED
        return None

    def _background_state(self) -> LivenessEvidenceKind:
        if self._background_items or self._children or self._queue_open:
            return LivenessEvidenceKind.BACKGROUND_STARTED
        return LivenessEvidenceKind.BACKGROUND_ENDED

    def _opened_from_request(
        self, method: str, message: dict[str, Any]
    ) -> LivenessEvidenceKind | None:
        if not (
            method.endswith(APPROVAL_REQUEST_SUFFIX)
            or method in {ELICITATION_REQUEST_METHOD, "item/tool/requestUserInput"}
        ):
            return None
        request_id = message.get("id")
        reference = request_reference(request_id)
        if reference is None:
            return LivenessEvidenceKind.STATE_UNCERTAIN
        self._pending_approvals.add(reference)
        return LivenessEvidenceKind.APPROVAL_OPENED

    def _closed_from_decision(
        self, method: str, params: dict[str, Any]
    ) -> LivenessEvidenceKind | None:
        if method != "serverRequest/resolved" and not method.endswith(APPROVAL_RESPONSE_SUFFIX):
            return None
        for key in _DECISION_ID_KEYS:
            value = params.get(key)
            reference = request_reference(value)
            if reference is not None:
                closed = self._close_approval(reference)
                if closed is not None:
                    return closed
        return None

    def _close_approval(self, request_id: str) -> LivenessEvidenceKind | None:
        if request_id not in self._pending_approvals:
            return None
        self._pending_approvals.discard(request_id)
        if self._pending_approvals:
            return None
        self._snapshot_approval_pending = False
        return LivenessEvidenceKind.APPROVAL_CLOSED

    def _sync_approvals(self) -> None:
        self._emit(
            LivenessEvidenceKind.APPROVAL_OPENED
            if self._pending_approvals or self._snapshot_approval_pending
            else LivenessEvidenceKind.APPROVAL_CLOSED
        )

    def _observe_uncertain(self) -> None:
        self._revision += 1
        self._reconciliation_required = True
        self._emit(LivenessEvidenceKind.STATE_UNCERTAIN)
        self._request_reconciliation()

    def _request_reconciliation(self) -> None:
        if (
            self._reconciliation_required
            and self._read_state is not None
            and (self._state_task is None or self._state_task.done())
        ):
            self._state_task = asyncio.create_task(self._refresh_state())

    async def _refresh_state(self) -> None:
        if self._read_state is None:
            return
        try:
            while True:
                revision = self._revision
                snapshot = await self._read_state()
                if revision != self._revision:
                    continue
                self._work_context.observe({"method": "state/snapshot", "status": snapshot.status})
                kind = self._status_kind({"status": snapshot.status})
                if kind is not None:
                    self._emit(kind)
                self._queue_open = snapshot.queued
                self._queue_revision += 1
                self._children = set(snapshot.children)
                self._background_items.clear()
                self._snapshot_approval_pending = snapshot.approval_pending
                if not snapshot.approval_pending:
                    self._pending_approvals.clear()
                self._sync_approvals()
                self._emit(self._background_state())
                self._reconciliation_required = False
                self._emit(LivenessEvidenceKind.STATE_SYNCED)
                return
        except Exception:
            logger.warning("session %s liveness state unavailable", self._session_id, exc_info=True)

    def _emit(self, kind: LivenessEvidenceKind) -> None:
        try:
            self._port.observe(self._work_context.evidence(self._session_id, kind))
        except Exception:
            logger.debug(
                "codex liveness evidence %s dropped for session %s",
                kind.value,
                self._session_id,
                exc_info=True,
            )


def _params_of(message: dict[str, Any]) -> dict[str, Any]:
    params = message.get("params")
    return params if isinstance(params, dict) else {}


def _item_text(item: Any, key: str) -> str:
    if not isinstance(item, dict):
        return ""
    value = item.get(key)
    return value if isinstance(value, str) else ""


def _frame_seq(frame: dict[str, Any]) -> int:
    seq = frame.get("seq")
    return seq if isinstance(seq, int) else 0
