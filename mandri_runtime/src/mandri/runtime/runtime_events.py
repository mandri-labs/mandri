import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

from mandri.core.clock import system_now_ms
from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind, SessionId, SessionState, SessionStopCause
from mandri.core.types.approvals import ApprovalRequest
from mandri.runtime.adapters import EventSource
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.control.errors import ControlError
from mandri.runtime.launch_preparation import harness_kind
from mandri.runtime.liveness import LivenessPort
from mandri.runtime.liveness.factory import liveness_adapter
from mandri.runtime.registry import SessionRegistry
from mandri.runtime.session_feed import session_topic
from mandri.runtime.session_state import RuntimeStates
from mandri.runtime.translators.base import EventPublisher

_RESOLVED_FRAME_TYPE = "approval.resolved"
_SESSIONS_ALL_TOPIC = Topic("sessions.all")
_FEED_DRAIN_TIMEOUT_SECONDS = 5.0


class RuntimeEvents:
    def __init__(
        self,
        states: RuntimeStates,
        registry: SessionRegistry,
        hub: Hub | None,
        liveness: LivenessPort | None,
    ) -> None:
        self._session_state = states.session
        self._registry = registry
        self._hub = hub
        self._liveness = liveness
        self.publisher: EventPublisher | None = None
        self.approval_topic: Callable[[ApprovalRequest], Topic | None] | None = None

    async def publish_event(self, topic: Topic, payload: dict[str, Any]) -> None:
        if self.publisher is not None:
            await self.publisher(topic, payload)
        elif self._hub is not None:
            self._hub.publish(topic, payload)

    def publish_resolved(self, request: ApprovalRequest) -> None:
        if self._hub is None:
            return
        self._hub.publish(session_topic(str(request.session_id)), _resolved_payload(request))
        child_topic = self.approval_topic(request) if self.approval_topic else None
        if child_topic is not None:
            self._hub.publish(child_topic, _resolved_payload(request))

    async def drain_feed(self, session_id: str) -> None:
        feed = self._session_state(session_id).feed
        if feed is not None and feed.tasks:
            # The process can exit while its final event is still being published.
            # Bound the wait in case a descendant keeps an inherited pipe open.
            await asyncio.wait(feed.tasks, timeout=_FEED_DRAIN_TIMEOUT_SECONDS)

    async def detach_feed(self, session_id: str) -> None:
        feed = self._session_state(session_id).feed
        self._session_state(session_id).feed = None
        if feed is not None:
            await feed.stop()

    def attach_liveness(self, session_id: str, harness: str) -> None:
        port = self._liveness
        hub = self._hub
        kind = harness_kind(harness)
        if port is None or hub is None or kind is None:
            return
        native_session_id = SessionId(session_id)
        since = self._session_state(session_id).feed_start_seq
        topic = session_topic(session_id)
        control = self._session_state(session_id).control
        adapter = liveness_adapter(
            kind,
            hub,
            topic,
            port,
            native_session_id,
            since=since,
            native_identity=lambda: self._session_state(session_id).native_id,
            read_queue=control.read_queue if isinstance(control, CodexControlAdapter) else None,
            read_state=control.read_liveness if isinstance(control, CodexControlAdapter) else None,
        )
        port.register(native_session_id)
        adapter.start()
        self._session_state(session_id).liveness_adapter = adapter

    async def stop_liveness_adapter(self, session_id: str) -> None:
        adapter = self._session_state(session_id).liveness_adapter
        self._session_state(session_id).liveness_adapter = None
        if adapter is not None:
            await adapter.stop()

    async def forget_liveness(self, session_id: str) -> None:
        try:
            await self.stop_liveness_adapter(session_id)
        finally:
            if self._liveness is not None:
                self._liveness.forget(SessionId(session_id))

    def start_event_pump(self, session_id: str, events: EventSource | None) -> None:
        if self._hub is None or events is None:
            return
        task = asyncio.create_task(
            self._pump_events(session_id, events),
            name=f"opencode-events:{session_id}",
        )
        self._session_state(session_id).event_pump = task

    async def _pump_events(self, session_id: str, events: EventSource) -> None:
        hub = self._hub
        if hub is None:
            return
        try:
            while True:
                event = await events.next_event()
                if event is None:
                    return
                await self.publish_event(session_topic(session_id), _opencode_event_payload(event))
        except ControlError:
            return

    def publish(self, payload: dict[str, Any]) -> None:
        if self._hub is not None:
            self._hub.publish(_SESSIONS_ALL_TOPIC, self._with_policy(payload))

    def _with_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self._session_state(str(payload["session_id"]))
        return {
            **payload,
            "execution_backend": state.policy.execution_backend.value,
            "privacy_mode": state.policy.privacy_mode.value,
            "policy_revision": state.policy_revision,
        }

    def publish_both(self, session_id: str, payload: dict[str, Any]) -> None:
        hub = self._hub
        if hub is None:
            return
        payload = self._with_policy(payload)
        hub.publish(_SESSIONS_ALL_TOPIC, payload)
        hub.publish(session_topic(session_id), _session_envelope(payload))

    def publish_stopped(self, session_id: str, cause: SessionStopCause) -> None:
        kind = harness_kind(self._registry.harness_of(session_id) or "")
        if kind is None:
            return
        if not self._registry.claim_stopped_signal(session_id):
            return
        self.publish_both(session_id, _stopped_payload(session_id, kind, cause))

    def publish_control_lost(self, session_id: str, kind: HarnessKind) -> None:
        self.publish_both(session_id, _control_lost_payload(session_id, kind))

    def publish_started(self, session_id: str, kind: HarnessKind) -> None:
        self.publish(_started_payload(session_id, kind))

    async def stop_pump(self, session_id: str) -> None:
        state = self._session_state(session_id)
        pump, state.event_pump = state.event_pump, None
        if pump is not None:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump


def _opencode_event_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {"source": HarnessKind.OPENCODE.value, "raw": event, "ts": system_now_ms()}


def _resolved_payload(request: ApprovalRequest) -> dict[str, Any]:
    decision = request.decision.value if request.decision is not None else None
    raw: dict[str, Any] = {
        "approval_id": str(request.id),
        "outcome": request.status.value,
        "native_request_ref": request.native_request_ref,
    }
    if decision is not None:
        raw["decision"] = decision
    return {
        "type": _RESOLVED_FRAME_TYPE,
        "source": "mandri",
        "raw": raw,
        "ts": system_now_ms(),
        "approval_id": str(request.id),
        "outcome": request.status.value,
        "decision": decision,
    }


def _started_payload(session_id: str, harness: HarnessKind) -> dict[str, Any]:
    return {
        "type": "session_started",
        "session_id": session_id,
        "harness": harness.value,
        "state": SessionState.LIVE.value,
    }


def _session_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    raw = {key: value for key, value in payload.items() if key != "type"}
    return {"type": payload["type"], "source": "mandri", "raw": raw, "ts": system_now_ms()}


def _stopped_payload(
    session_id: str, harness: HarnessKind, cause: SessionStopCause
) -> dict[str, Any]:
    return {
        "type": "session_stopped",
        "session_id": session_id,
        "harness": harness.value,
        "state": SessionState.STOPPED.value,
        "cause": cause.value,
    }


def _control_lost_payload(session_id: str, harness: HarnessKind) -> dict[str, Any]:
    return {
        "type": "control_lost",
        "session_id": session_id,
        "harness": harness.value,
    }
