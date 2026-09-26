"""Per-session watcher recognizing approval-bearing harness events on the live feed."""

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, final

from mandri.core.clock import system_now_ms
from mandri.core.hub import Hub, SubscriberHandle
from mandri.core.ids import HarnessKind, RawEvent, SessionId
from mandri.runtime.approvals.errors import DuplicateApprovalRequestError
from mandri.runtime.approvals.native_resolution import matches_resolution
from mandri.runtime.approvals.recognition import ApprovalDescriptor, detect, with_approval_metadata
from mandri.runtime.approvals.service import ApprovalService
from mandri.runtime.control.errors import ControlError
from mandri.runtime.session_feed import session_topic
from mandri.runtime.translators.base import EventPublisher

PENDING_FRAME_TYPE = "approval.pending"
_logger = logging.getLogger(__name__)


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
        auto_answer: Callable[[str], Awaitable[None]] | None = None,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._hub = hub
        self._service = service
        self._session_id = session_id
        self._harness = harness
        self._timeout_seconds = timeout_seconds
        self._auto_answer = auto_answer
        self._publisher = publisher
        self._handle: SubscriberHandle | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._handle = self._hub.subscribe(session_topic(str(self._session_id)), internal=True)
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
        if await self._resolve_native(raw):
            return
        line = json.dumps(raw)
        descriptor = detect(self._harness, line)
        if descriptor is not None:
            await self._register(raw, line, descriptor)

    async def _resolve_native(self, raw: dict[str, Any]) -> bool:
        matched = False
        for pending in self._service.pending_for_session(self._session_id):
            if not matches_resolution(pending, raw):
                continue
            matched = True
            request = await self._service.resolve_native(pending.id)
            if request is None:
                continue
            payload = {
                "type": "approval.resolved",
                "source": "mandri",
                "raw": {"approval_id": str(request.id), "outcome": request.status.value},
                "ts": system_now_ms(),
            }
            topic = session_topic(str(self._session_id))
            if self._publisher is not None:
                await self._publisher(topic, payload)
            else:
                self._hub.publish(topic, payload)
        return matched

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
        if self._auto_answer is not None:
            try:
                await self._auto_answer(str(request.id))
            except ControlError:
                _logger.exception("Automatic approval delivery failed for %s", request.id)
            return
        envelope = {"source": self._harness.value, "raw": raw, "ts": system_now_ms()}
        payload = {**with_approval_metadata(envelope, request), "type": PENDING_FRAME_TYPE}
        topic = session_topic(str(self._session_id))
        if self._publisher is not None:
            await self._publisher(topic, payload)
        else:
            self._hub.publish(topic, payload)
