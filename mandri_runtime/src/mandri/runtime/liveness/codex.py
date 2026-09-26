"""Codex evidence adapter translating session-feed frames into liveness evidence."""

import asyncio
import contextlib
import logging
import typing
from collections.abc import Callable
from typing import Any

from mandri.core.hub import Hub, SubscriberHandle, Topic
from mandri.core.ids import SessionId
from mandri.runtime.liveness.evidence import LivenessEvidence, LivenessEvidenceKind
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
    ) -> None:
        self._hub = hub
        self._topic = topic
        self._port = port
        self._session_id = session_id
        self._native_identity = native_identity
        self._task: asyncio.Task[None] | None = None
        self._handle: SubscriberHandle | None = None
        self._last_seq = 0
        self._turn_open = False
        self._status_turn_open = False
        self._queue_open = False
        self._background_items = 0
        self._pending_approvals: set[str] = set()

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.get_running_loop().create_task(
            self._consume(), name=f"codex-liveness:{self._session_id}"
        )

    async def stop(self) -> None:
        task = self._task
        self._task = None
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
        if not isinstance(payload, dict) or "type" in payload:
            return
        if payload.get("source") != SOURCE:
            return
        raw = payload.get("raw")
        if not isinstance(raw, dict):
            return
        identity = self._native_identity() if self._native_identity else None
        params = raw.get("params")
        thread_id = params.get("threadId") if isinstance(params, dict) else None
        if identity is not None and isinstance(thread_id, str) and thread_id != identity:
            return
        try:
            kind = self._evidence_kind(raw)
        except Exception:
            logger.warning(
                "session %s dropped poison codex frame: %s",
                self._session_id,
                raw,
                exc_info=True,
            )
            return
        if kind is not None:
            self._emit(kind)

    def _evidence_kind(self, message: dict[str, Any]) -> LivenessEvidenceKind | None:
        method = message.get("method")
        if not isinstance(method, str) or not method:
            return self._closed_from_response(message)
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
            return LivenessEvidenceKind.TURN_ENDED
        if method == THREAD_STATUS_CHANGED_NOTIFICATION:
            return self._status_kind(params)
        if method == THREAD_QUEUE_CHANGED_NOTIFICATION:
            return self._queue_kind(params)
        return self._background_kind(method, params)

    def _status_kind(self, params: dict[str, Any]) -> LivenessEvidenceKind | None:
        if params.get("status") == _ACTIVE_STATUS:
            if self._turn_open or self._status_turn_open:
                return None
            self._status_turn_open = True
            return LivenessEvidenceKind.TURN_STARTED
        was_open = self._turn_open or self._status_turn_open
        self._turn_open = False
        self._status_turn_open = False
        if was_open:
            return LivenessEvidenceKind.TURN_ENDED
        return None

    def _queue_kind(self, params: dict[str, Any]) -> LivenessEvidenceKind | None:
        queue = params.get("queue")
        if not isinstance(queue, list):
            return None
        if queue:
            self._queue_open = True
            return LivenessEvidenceKind.BACKGROUND_STARTED
        self._queue_open = False
        if self._background_items == 0:
            return LivenessEvidenceKind.BACKGROUND_ENDED
        return None

    def _background_kind(self, method: str, params: dict[str, Any]) -> LivenessEvidenceKind | None:
        item = params.get("item")
        haystack = " ".join([method, _item_text(item, "type"), _item_text(item, "kind")]).lower()
        if not any(marker in haystack for marker in _BACKGROUND_ITEM_MARKERS):
            return None
        if method == ITEM_COMPLETED_NOTIFICATION or _item_text(item, "kind") == "completed":
            return self._background_ended()
        if method == ITEM_STARTED_NOTIFICATION or _item_text(item, "kind") == "started":
            self._background_items += 1
            return LivenessEvidenceKind.BACKGROUND_STARTED
        return None

    def _background_ended(self) -> LivenessEvidenceKind | None:
        if self._background_items > 0:
            self._background_items -= 1
        if self._background_items == 0 and not self._queue_open:
            return LivenessEvidenceKind.BACKGROUND_ENDED
        return None

    def _opened_from_request(
        self, method: str, message: dict[str, Any]
    ) -> LivenessEvidenceKind | None:
        if not (method.endswith(APPROVAL_REQUEST_SUFFIX) or method == ELICITATION_REQUEST_METHOD):
            return None
        request_id = message.get("id")
        if isinstance(request_id, (str, int)):
            self._pending_approvals.add(str(request_id))
        return LivenessEvidenceKind.APPROVAL_OPENED

    def _closed_from_response(self, message: dict[str, Any]) -> LivenessEvidenceKind | None:
        if "id" not in message:
            return None
        request_id = message.get("id")
        if not isinstance(request_id, (str, int)):
            return None
        return self._close_approval(str(request_id))

    def _closed_from_decision(
        self, method: str, params: dict[str, Any]
    ) -> LivenessEvidenceKind | None:
        if not method.endswith(APPROVAL_RESPONSE_SUFFIX):
            return None
        for key in _DECISION_ID_KEYS:
            value = params.get(key)
            if isinstance(value, (str, int)):
                closed = self._close_approval(str(value))
                if closed is not None:
                    return closed
        return None

    def _close_approval(self, request_id: str) -> LivenessEvidenceKind | None:
        if request_id not in self._pending_approvals:
            return None
        self._pending_approvals.discard(request_id)
        return LivenessEvidenceKind.APPROVAL_CLOSED

    def _observe_uncertain(self) -> None:
        self._emit(LivenessEvidenceKind.STATE_UNCERTAIN)

    def _emit(self, kind: LivenessEvidenceKind) -> None:
        try:
            self._port.observe(LivenessEvidence(session_id=self._session_id, kind=kind))
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
