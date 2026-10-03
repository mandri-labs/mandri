import asyncio
import contextlib
from typing import Any

from mandri.core.hub import Hub, SubscriberHandle, Topic
from mandri.core.ids import HarnessKind, SessionId
from mandri.runtime.liveness.degradation import is_stream_degradation
from mandri.runtime.liveness.evidence import LivenessEvidenceKind
from mandri.runtime.liveness.observation import WorkEventContext
from mandri.runtime.liveness.port import LivenessPort


class PiLivenessAdapter:
    def __init__(
        self, hub: Hub, topic: Topic, port: LivenessPort, session_id: SessionId, *, since: int = 0
    ) -> None:
        self._hub, self._topic, self._port, self._session = hub, topic, port, session_id
        self._work_context = WorkEventContext(HarnessKind.PI)
        self._handle: SubscriberHandle | None = None
        self._task: asyncio.Task[None] | None = None
        self._seq = since
        self._closed = False
        self._running = False
        self._dialogs: set[str] = set()
        self._approval_refs: dict[str, str] = {}

    def start(self) -> None:
        self._handle = self._hub.subscribe(self._topic, since=self._seq, internal=True)
        self._task = asyncio.create_task(self._consume(), name=f"liveness-pi:{self._session}")

    async def stop(self) -> None:
        self._closed = True
        if self._handle is not None:
            self._hub.unsubscribe(self._handle)
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _consume(self) -> None:
        while not self._closed:
            handle = self._handle
            if handle is None:
                return
            for frame in handle.replay:
                self._absorb(frame)
            while not self._closed:
                delivered = await handle.queue.get()
                if delivered is None:
                    self._emit(LivenessEvidenceKind.STATE_UNCERTAIN)
                    break
                self._absorb(delivered)
            self._hub.unsubscribe(handle)
            if not self._closed:
                self._handle = self._hub.subscribe(self._topic, since=self._seq, internal=True)

    def _absorb(self, frame: dict[str, Any]) -> None:
        if frame.get("type") == "gap":
            self._emit(LivenessEvidenceKind.STATE_UNCERTAIN)
            sequence = frame.get("seq")
            if isinstance(sequence, int):
                self._seq = max(self._seq, sequence - 1)
            return
        sequence = frame.get("seq")
        if isinstance(sequence, int):
            if sequence <= self._seq:
                return
            self._seq = sequence
        payload = frame.get("payload")
        if not isinstance(payload, dict):
            return
        raw = payload.get("raw")
        if not isinstance(raw, dict):
            return
        if is_stream_degradation(payload):
            self._emit(LivenessEvidenceKind.STATE_UNCERTAIN)
            return
        if payload.get("type") == "approval.resolved":
            reference = self._approval_refs.pop(str(raw.get("approval_id")), None)
            if reference is not None:
                self._dialogs.discard(reference)
                self._sync_dialogs()
            return
        if payload.get("source") != "pi":
            return
        if payload.get("type") == "approval.pending":
            identifier, approval_id = raw.get("id"), payload.get("approval_id")
            if isinstance(identifier, str) and isinstance(approval_id, str):
                self._approval_refs[approval_id] = identifier
            return
        if "type" not in payload:
            self._observe(raw, payload.get("ts"))

    def _observe(self, raw: dict[str, Any], timestamp: object = None) -> None:
        self._work_context.observe(raw, timestamp)
        event = raw.get("type")
        if event in {"agent_start", "message_update", "tool_execution_start"}:
            self._running = True
            self._emit(LivenessEvidenceKind.TURN_STARTED)
        elif event == "agent_settled":
            self._running = False
            self._emit(LivenessEvidenceKind.TURN_ENDED)
        elif event in {"auto_compaction_start", "compaction_start", "auto_retry_start"}:
            self._emit(LivenessEvidenceKind.BACKGROUND_STARTED)
        elif event in {"auto_compaction_end", "compaction_end", "auto_retry_end"}:
            self._emit(LivenessEvidenceKind.BACKGROUND_ENDED)
        elif event == "extension_ui_request" and raw.get("method") in {
            "select",
            "confirm",
            "input",
            "editor",
        }:
            identifier = raw.get("id")
            if isinstance(identifier, str):
                self._dialogs.add(identifier)
                self._sync_dialogs()
        elif event == "response" and raw.get("success") is True:
            data = raw.get("data")
            if raw.get("command") == "get_state" and isinstance(data, dict):
                self._running = bool(data.get("isStreaming") or data.get("pendingMessageCount"))
                self._emit(
                    LivenessEvidenceKind.TURN_STARTED
                    if self._running
                    else LivenessEvidenceKind.TURN_ENDED
                )
            elif (
                raw.get("command") == "prompt"
                and isinstance(data, dict)
                and data.get("disposition") == "handled"
                and not self._running
            ):
                self._emit(LivenessEvidenceKind.TURN_ENDED)

    def _sync_dialogs(self) -> None:
        self._emit(
            LivenessEvidenceKind.APPROVAL_OPENED
            if self._dialogs
            else LivenessEvidenceKind.APPROVAL_CLOSED
        )

    def _emit(self, kind: LivenessEvidenceKind) -> None:
        self._port.observe(self._work_context.evidence(self._session, kind))
