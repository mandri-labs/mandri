import asyncio
import contextlib
from typing import Any

from mandri.core.hub import Hub, SubscriberHandle, Topic
from mandri.core.ids import SessionId
from mandri.runtime.liveness.evidence import LivenessEvidence, LivenessEvidenceKind
from mandri.runtime.liveness.port import LivenessPort


class AgyLivenessAdapter:
    def __init__(self, hub: Hub, topic: Topic, port: LivenessPort, session_id: SessionId) -> None:
        self._hub, self._topic, self._port, self._session = hub, topic, port, session_id
        self._handle: SubscriberHandle | None = None
        self._task: asyncio.Task[None] | None = None
        self._seq = 0
        self._closed = False
        self._native_id: str | None = None
        self._children: set[str] = set()
        self._completed_children: set[str] = set()
        self._background = False
        self._root_idle = False
        self._approvals: set[str] = set()
        self._uncertain = False

    def start(self) -> None:
        self._handle = self._hub.subscribe(self._topic, since=0, internal=True)
        self._task = asyncio.create_task(self._consume())

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
                    self._uncertain = True
                    self._emit(LivenessEvidenceKind.STATE_UNCERTAIN)
                    break
                self._absorb(delivered)
            self._hub.unsubscribe(handle)
            if not self._closed:
                self._handle = self._hub.subscribe(self._topic, since=self._seq, internal=True)

    def _emit(self, kind: LivenessEvidenceKind) -> None:
        self._port.observe(LivenessEvidence(self._session, kind))

    def _absorb(self, frame: dict[str, Any]) -> None:
        if frame.get("type") == "gap":
            self._uncertain = True
            self._emit(LivenessEvidenceKind.STATE_UNCERTAIN)
            return
        seq = frame.get("seq")
        if isinstance(seq, int):
            if seq <= self._seq:
                return
            self._seq = seq
        payload = frame.get("payload")
        if not isinstance(payload, dict) or payload.get("source") != "agy" or "type" in payload:
            return
        raw = payload.get("raw")
        if not isinstance(raw, dict):
            return
        self._observe(raw)
        if self._uncertain:
            self._emit(LivenessEvidenceKind.STATE_UNCERTAIN)

    def _sync_background(self) -> None:
        self._emit(
            LivenessEvidenceKind.BACKGROUND_STARTED
            if self._background or self._children
            else LivenessEvidenceKind.BACKGROUND_ENDED
        )

    def _observe(self, raw: dict[str, Any]) -> None:
        event = raw.get("event")
        if event == "init":
            native_id = raw.get("conversation_id")
            if isinstance(native_id, str) and native_id:
                self._native_id = native_id
            return
        if event == "step_update":
            step = raw.get("step_update")
            if isinstance(step, dict):
                self._step(step)
        elif event == "result":
            result = raw.get("result")
            native_id = raw.get("conversation_id")
            if isinstance(result, dict):
                native_id = result.get("conversation_id", native_id)
            if native_id is None or native_id == self._native_id:
                self._emit(LivenessEvidenceKind.TURN_ENDED)
                rejected_command = (
                    isinstance(result, dict)
                    and result.get("status") == "ERROR"
                    and result.get("num_turns") == 0
                    and isinstance(result.get("error"), str)
                    and "unavailable with --input-format stream-json" in result["error"]
                )
                if not self._root_idle and not self._background and not rejected_command:
                    self._uncertain = True
        elif event in {"approval_request", "approval_response"}:
            ref = raw.get("request_id")
            if not isinstance(ref, str):
                self._uncertain = True
                return
            if event == "approval_request":
                self._approvals.add(ref)
            else:
                self._approvals.discard(ref)
            self._emit(
                LivenessEvidenceKind.APPROVAL_OPENED
                if self._approvals
                else LivenessEvidenceKind.APPROVAL_CLOSED
            )
        elif event == "hook":
            data = raw.get("data")
            if isinstance(data, dict):
                if raw.get("hook") == "Stop":
                    self._stop(data)
                elif raw.get("hook") == "PreInvocation":
                    native_id = data.get("conversationId")
                    if native_id == self._native_id:
                        self._root_idle = False
                        self._emit(LivenessEvidenceKind.TURN_STARTED)
                    elif isinstance(native_id, str):
                        self._completed_children.discard(native_id)
                        self._children.add(native_id)
                        self._sync_background()

    def _step(self, step: dict[str, Any]) -> None:
        native_id = step.get("conversation_id")
        if isinstance(native_id, str) and native_id != self._native_id:
            if step.get("state") == "ACTIVE":
                self._completed_children.discard(native_id)
                self._children.add(native_id)
                self._sync_background()
            return
        if step.get("state") == "ACTIVE" or step.get("step_type") == "user_input":
            self._root_idle = False
            self._emit(LivenessEvidenceKind.TURN_STARTED)
        info = step.get("subagent_info")
        children = info.get("subagents") if isinstance(info, dict) else None
        if isinstance(children, list):
            for child in children:
                child_id = child.get("conversation_id") if isinstance(child, dict) else None
                if (
                    isinstance(child_id, str)
                    and child_id != self._native_id
                    and child_id not in self._completed_children
                ):
                    self._children.add(child_id)
            self._sync_background()

    def _stop(self, data: dict[str, Any]) -> None:
        native_id, idle = data.get("conversationId"), data.get("fullyIdle")
        if self._native_id is None or not isinstance(native_id, str) or not isinstance(idle, bool):
            self._uncertain = True
            return
        if native_id != self._native_id:
            if idle:
                self._children.discard(native_id)
                self._completed_children.add(native_id)
            else:
                self._completed_children.discard(native_id)
                self._children.add(native_id)
        else:
            self._root_idle = idle
            self._background = not idle
            if idle:
                self._emit(LivenessEvidenceKind.TURN_ENDED)
        self._sync_background()
        if self._root_idle and not self._children:
            self._uncertain = False
