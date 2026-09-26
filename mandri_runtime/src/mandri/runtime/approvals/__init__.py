"""Per-session watcher recognizing approval-bearing harness events on the live feed."""

import asyncio
import contextlib
import json
from typing import Any, final

from mandri.core.clock import system_now_ms
from mandri.core.hub import Hub, SubscriberHandle
from mandri.core.ids import HarnessKind, RawEvent, SessionId
from mandri.runtime.approvals.errors import DuplicateApprovalRequestError
from mandri.runtime.approvals.recognition import ApprovalDescriptor, detect, with_approval_metadata
from mandri.runtime.approvals.service import ApprovalService
from mandri.runtime.session_feed import session_topic

PENDING_FRAME_TYPE = "approval.pending"


@final
class ApprovalWatcher:
    """Subscribes to one session topic, registers recognized requests, publishes pending."""

    def __init__(
        self,
        hub: Hub,
        service: ApprovalService,
        session_id: SessionId,
        harness: HarnessKind,
        timeout_seconds: int,
    ) -> None:
        self._hub = hub
        self._service = service
        self._session_id = session_id
        self._harness = harness
        self._timeout_seconds = timeout_seconds
        self._handle: SubscriberHandle | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._handle = self._hub.subscribe(session_topic(str(self._session_id)))
        self._task = asyncio.get_running_loop().create_task(
            self._consume(), name=f"approval-watcher:{self._session_id}"
        )

    async def stop(self) -> None:
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
        handle = self._handle
        if handle is None:
            return
        while True:
            frame = await handle.queue.get()
            if frame is None:
                return
            await self._process(frame.get("payload"))

    async def _process(self, payload: Any) -> None:
        if not isinstance(payload, dict) or "type" in payload:
            return
        if payload.get("source") != self._harness.value:
            return
        raw = payload.get("raw")
        if not isinstance(raw, dict):
            return
        line = json.dumps(raw)
        descriptor = detect(self._harness, line)
        if descriptor is not None:
            await self._register(raw, line, descriptor)

    async def _register(
        self, raw: dict[str, Any], line: str, descriptor: ApprovalDescriptor
    ) -> None:
        try:
            request = await self._service.register(
                session_id=self._session_id,
                harness=self._harness,
                native_request=RawEvent(line),
                native_request_ref=descriptor.native_request_ref,
                kind=descriptor.kind,
                timeout_seconds=self._timeout_seconds,
            )
        except DuplicateApprovalRequestError:
            return
        envelope = {"source": self._harness.value, "raw": raw, "ts": system_now_ms()}
        payload = {**with_approval_metadata(envelope, request), "type": PENDING_FRAME_TYPE}
        self._hub.publish(session_topic(str(self._session_id)), payload)
