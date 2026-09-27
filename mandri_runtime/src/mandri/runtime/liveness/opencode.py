"""OpenCode evidence adapter translating hub SSE frames into neutral liveness evidence."""

import asyncio
import contextlib
import logging
from typing import Any, final

from mandri.core.hub import Hub, SubscriberHandle, Topic
from mandri.core.ids import SessionId
from mandri.runtime.liveness.errors import UnknownSessionError
from mandri.runtime.liveness.evidence import LivenessEvidence, LivenessEvidenceKind
from mandri.runtime.liveness.port import LivenessPort

logger = logging.getLogger(__name__)

_RECONNECT_DELAY_SECONDS = 0.5

_SOURCE = "opencode"
_STATUS_BUSY_TYPES = frozenset({"busy", "retry"})
_PERMISSION_OPEN_TYPES = frozenset({"permission.asked", "permission.v2.asked", "question.asked"})
_PERMISSION_REPLY_TYPES = frozenset(
    {"permission.replied", "permission.updated", "question.replied", "question.rejected"}
)
_RESOLUTION_KEYS = ("response", "reply", "decision", "status")


@final
class OpencodeLivenessAdapter:
    """Feeds session events to the liveness port with subscription continuation."""

    def __init__(
        self,
        hub: Hub,
        topic: Topic,
        port: LivenessPort,
        session_id: SessionId,
        *,
        since: int = 0,
    ) -> None:
        self._hub = hub
        self._topic = topic
        self._port = port
        self._session_id = session_id
        self._handle: SubscriberHandle | None = None
        self._task: asyncio.Task[None] | None = None
        self._last_seq = since
        self._closed = False
        self._root: str | None = None
        self._root_turn_active = False
        self._children: set[str] = set()

    def start(self) -> None:
        self._handle = self._hub.subscribe(self._topic, since=self._last_seq, internal=True)
        self._task = asyncio.get_running_loop().create_task(
            self._consume(), name=f"opencode-liveness:{self._session_id}"
        )

    async def stop(self) -> None:
        self._closed = True
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
        while not self._closed:
            handle = self._handle
            if handle is None:
                handle = self._hub.subscribe(self._topic, since=self._last_seq, internal=True)
                self._handle = handle
            for frame in handle.replay:
                self._absorb(frame)
            while True:
                delivered = await handle.queue.get()
                if delivered is None or _is_slow_consumer_gap(delivered):
                    break
                self._absorb(delivered)
            if self._closed:
                return
            self._hub.unsubscribe(handle)
            self._handle = None
            self._mark_uncertain()
            await asyncio.sleep(_RECONNECT_DELAY_SECONDS)

    def _absorb(self, frame: dict[str, Any]) -> None:
        if _is_gap(frame):
            self._skip_lost(frame)
            return
        self._last_seq = _seq_of(frame)
        try:
            self._observe_frame(frame)
        except Exception as error:
            logger.warning(
                "opencode liveness dropped poison frame for session %s: %s",
                self._session_id,
                error,
            )

    def _observe_frame(self, frame: dict[str, Any]) -> None:
        payload = frame.get("payload")
        if not isinstance(payload, dict) or "type" in payload:
            return
        if payload.get("source") != _SOURCE:
            return
        raw = payload.get("raw")
        if not isinstance(raw, dict):
            return
        for kind in self._translate(raw):
            self._emit(kind)

    def _translate(self, raw: dict[str, Any]) -> list[LivenessEvidenceKind]:
        event_type = raw.get("type")
        if not isinstance(event_type, str):
            return []
        properties = _properties_of(raw)
        owner = _owner_of(properties)
        if event_type == "session.status":
            status = _status_type_of(properties)
            if status in _STATUS_BUSY_TYPES:
                return self._turn_activity(owner)
            if status == "idle":
                return self._turn_end(owner)
            return []
        if event_type in ("session.idle", "session.error"):
            return self._turn_end(owner)
        if event_type == "session.deleted":
            return self._turn_end(owner)
        if event_type == "message.part.updated":
            return self._turn_activity(owner)
        if event_type in _PERMISSION_OPEN_TYPES:
            return [LivenessEvidenceKind.APPROVAL_OPENED]
        if event_type in _PERMISSION_REPLY_TYPES and (
            event_type in {"permission.replied", "question.replied", "question.rejected"}
            or _has_resolution(properties)
        ):
            return [LivenessEvidenceKind.APPROVAL_CLOSED]
        return []

    def _turn_activity(self, owner: str | None) -> list[LivenessEvidenceKind]:
        if owner is None or self._root is None or owner == self._root:
            if owner is not None and self._root is None:
                self._root = owner
            if self._root_turn_active:
                return []
            self._root_turn_active = True
            return [LivenessEvidenceKind.TURN_STARTED]
        if owner in self._children:
            return []
        self._children.add(owner)
        return [LivenessEvidenceKind.BACKGROUND_STARTED]

    def _turn_end(self, owner: str | None) -> list[LivenessEvidenceKind]:
        if owner is not None and self._root is not None and owner != self._root:
            if owner not in self._children:
                return []
            self._children.discard(owner)
            return [LivenessEvidenceKind.BACKGROUND_ENDED]
        self._root_turn_active = False
        return [LivenessEvidenceKind.TURN_ENDED]

    def _emit(self, kind: LivenessEvidenceKind) -> None:
        try:
            self._port.observe(LivenessEvidence(session_id=self._session_id, kind=kind))
        except UnknownSessionError as error:
            logger.debug(
                "opencode liveness evidence discarded for session %s: %s", self._session_id, error
            )

    def _mark_uncertain(self) -> None:
        self._emit(LivenessEvidenceKind.STATE_UNCERTAIN)

    def _skip_lost(self, frame: dict[str, Any]) -> None:
        end = _seq_of(frame)
        if end - 1 > self._last_seq:
            self._last_seq = end - 1
        self._mark_uncertain()


def _is_gap(frame: dict[str, Any]) -> bool:
    return frame.get("type") == "gap"


def _is_slow_consumer_gap(frame: dict[str, Any]) -> bool:
    return _is_gap(frame) and frame.get("reason") == "slow_consumer"


def _seq_of(frame: dict[str, Any]) -> int:
    seq = frame.get("seq")
    return seq if isinstance(seq, int) else 0


def _properties_of(raw: dict[str, Any]) -> dict[str, Any]:
    properties = raw.get("properties")
    return properties if isinstance(properties, dict) else {}


def _owner_of(properties: dict[str, Any]) -> str | None:
    owner = properties.get("sessionID")
    return owner if isinstance(owner, str) else None


def _status_type_of(properties: dict[str, Any]) -> str | None:
    status = properties.get("status")
    value = status.get("type") if isinstance(status, dict) else status
    return value if isinstance(value, str) else None


def _has_resolution(properties: dict[str, Any]) -> bool:
    return any(properties.get(key) is not None for key in _RESOLUTION_KEYS)
