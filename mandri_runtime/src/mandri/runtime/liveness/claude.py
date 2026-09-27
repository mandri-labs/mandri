"""Claude adapter translating stream-json frames into neutral liveness evidence."""

import asyncio
import contextlib
import logging
from typing import Any, final

from mandri.core.hub import Hub, SubscriberHandle, Topic
from mandri.core.ids import SessionId
from mandri.runtime.liveness.evidence import LivenessEvidence, LivenessEvidenceKind
from mandri.runtime.liveness.port import LivenessPort

logger = logging.getLogger(__name__)

CLAUDE_SOURCE = "claude"

_TURN_FRAME_TYPES = frozenset({"assistant", "user"})
_TERMINAL_TASK_NOTIFICATION_STATUSES = frozenset({"completed", "failed", "stopped"})
_TERMINAL_TASK_UPDATED_STATUSES = frozenset({"completed", "failed", "killed"})
_SLOW_CONSUMER_REASON = "slow_consumer"


@final
class ClaudeLivenessAdapter:
    """Feeds the liveness port from an internal subscription to one session topic."""

    def __init__(
        self, hub: Hub, topic: Topic, port: LivenessPort, session_id: SessionId, *, since: int = 0
    ) -> None:
        self._hub = hub
        self._topic = topic
        self._port = port
        self._session_id = session_id
        self._handle: SubscriberHandle | None = None
        self._task: asyncio.Task[None] | None = None
        self._last_seq = since
        self._turn_active = False
        self._stopped = False

    def start(self) -> None:
        self._stopped = False
        self._handle = self._hub.subscribe(self._topic, since=self._last_seq, internal=True)
        self._task = asyncio.get_running_loop().create_task(
            self._consume(), name=f"liveness-claude:{self._session_id}"
        )

    async def stop(self) -> None:
        self._stopped = True
        handle = self._handle
        task = self._task
        self._handle = None
        self._task = None
        if handle is not None:
            self._hub.unsubscribe(handle)
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _consume(self) -> None:
        while not self._stopped:
            handle = self._current_handle()
            if handle is None:
                return
            try:
                ended = await self._drain(handle)
            finally:
                self._hub.unsubscribe(handle)
                if self._handle is handle:
                    self._handle = None
            if not ended or self._stopped:
                return

    def _current_handle(self) -> SubscriberHandle | None:
        handle = self._handle
        if handle is not None:
            return handle
        if self._stopped:
            return None
        handle = self._hub.subscribe(self._topic, since=self._last_seq, internal=True)
        self._handle = handle
        return handle

    async def _drain(self, handle: SubscriberHandle) -> bool:
        for replayed in handle.replay:
            if self._absorb(replayed):
                return True
        while True:
            frame: dict[str, Any] | None = await handle.queue.get()
            if frame is None:
                self._observe_uncertain()
                return True
            if self._absorb(frame):
                return True

    def _absorb(self, frame: dict[str, Any]) -> bool:
        try:
            return self._handle_frame(frame)
        except Exception:
            logger.exception(
                "liveness adapter dropped a poison frame for session %s", self._session_id
            )
            return False

    def _handle_frame(self, frame: dict[str, Any]) -> bool:
        if frame.get("type") == "gap":
            self._observe_uncertain()
            if frame.get("reason") == _SLOW_CONSUMER_REASON:
                return True
            self._skip_lost(frame)
            return False
        seq = frame.get("seq")
        if isinstance(seq, int):
            self._last_seq = seq
        payload = frame.get("payload")
        if not isinstance(payload, dict) or "type" in payload:
            return False
        if payload.get("source") != CLAUDE_SOURCE:
            return False
        raw = payload.get("raw")
        if not isinstance(raw, dict):
            return False
        self._translate(raw)
        return False

    def _skip_lost(self, frame: dict[str, Any]) -> None:
        end = frame.get("seq")
        if isinstance(end, int) and end - 1 > self._last_seq:
            self._last_seq = end - 1

    def _translate(self, raw: dict[str, Any]) -> None:
        if raw.get("parent_tool_use_id") is not None and raw.get("type") in (
            "assistant",
            "user",
            "result",
            "stream_event",
        ):
            return
        frame_type = raw.get("type")
        if frame_type in _TURN_FRAME_TYPES:
            self._observe_turn_started()
            return
        if frame_type == "result":
            self._observe_result(raw)
            return
        if frame_type == "system":
            self._translate_system(raw)
            return
        if frame_type == "control_request":
            if raw.get("subtype") == "can_use_tool":
                self._emit(LivenessEvidenceKind.APPROVAL_OPENED)
            return
        if frame_type == "control_response":
            self._emit(LivenessEvidenceKind.APPROVAL_CLOSED)

    def _translate_system(self, raw: dict[str, Any]) -> None:
        subtype = raw.get("subtype")
        if subtype == "task_started":
            self._emit(LivenessEvidenceKind.BACKGROUND_STARTED)
            return
        if subtype == "task_progress":
            return
        if subtype == "task_notification":
            if raw.get("status") in _TERMINAL_TASK_NOTIFICATION_STATUSES:
                self._emit(LivenessEvidenceKind.BACKGROUND_ENDED)
            return
        if subtype == "task_updated" and raw.get("status") in _TERMINAL_TASK_UPDATED_STATUSES:
            self._emit(LivenessEvidenceKind.BACKGROUND_ENDED)

    def _observe_turn_started(self) -> None:
        if self._turn_active:
            return
        self._turn_active = True
        self._emit(LivenessEvidenceKind.TURN_STARTED)

    def _observe_result(self, raw: dict[str, Any]) -> None:
        queued = raw.get("queued_turn_count")
        if isinstance(queued, int) and queued > 0:
            return
        self._turn_active = False
        self._emit(LivenessEvidenceKind.TURN_ENDED)

    def _observe_uncertain(self) -> None:
        self._emit(LivenessEvidenceKind.STATE_UNCERTAIN)

    def _emit(self, kind: LivenessEvidenceKind) -> None:
        try:
            self._port.observe(LivenessEvidence(self._session_id, kind))
        except Exception:
            logger.exception("liveness evidence rejected for session %s", self._session_id)
