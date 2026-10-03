import asyncio
import contextlib
import logging
from typing import Any

from mandri.core.errors import MandriError
from mandri.core.hub import Hub, Topic
from mandri.core.ids import SessionId
from mandri.core.types.agents import Agent
from mandri.core.types.availability import SessionOwner
from mandri.core.types.conversation_status import WorkObservation, WorkOutcome, WorkState
from mandri.core.work_outcomes import work_event_key
from mandri.runtime.agents import AgentService
from mandri.runtime.liveness.evidence import LivenessEvidence, LivenessEvidenceKind
from mandri.runtime.liveness.factory import liveness_adapter
from mandri.runtime.liveness.tracker import WorkingStateTracker
from mandri.runtime.liveness.types import BusyReason, WorkingState
from mandri.runtime.service import RuntimeService
from mandri.runtime.session_state import LivenessAdapter
from mandri.sessions.conversation_status import ConversationStatuses
from mandri.sessions.service import SessionsService

logger = logging.getLogger(__name__)


def observe_liveness(
    statuses: ConversationStatuses, evidence: LivenessEvidence, state: WorkingState
) -> None:
    identifier = str(evidence.session_id)
    target = identifier if identifier.startswith("agent:") else f"session:{identifier}"
    work_state: WorkState = (
        "unknown"
        if state.uncertain
        else (
            "waiting"
            if BusyReason.APPROVAL_PENDING in state.reasons
            else ("working" if state.busy else "idle")
        )
    )
    if evidence.kind is LivenessEvidenceKind.TURN_ENDED and evidence.outcome is None:
        work_state = "unknown"
    statuses.enqueue(
        target,
        WorkObservation(
            state=work_state,
            progress=evidence.event_key is not None
            and evidence.kind
            in {
                LivenessEvidenceKind.TURN_STARTED,
                LivenessEvidenceKind.BACKGROUND_STARTED,
            },
            source="live",
            content_key=evidence.content_key,
            outcome=evidence.outcome,
            key=evidence.event_key,
        ),
    )


class ConversationStatusObserver:
    def __init__(
        self,
        statuses: ConversationStatuses,
        sessions: SessionsService,
        runtime: RuntimeService,
        agents: AgentService,
        hub: Hub,
    ) -> None:
        self._statuses, self._sessions = statuses, sessions
        self._runtime, self._agents, self._hub = runtime, agents, hub
        self._adapters: dict[str, LivenessAdapter] = {}
        self._tracker = WorkingStateTracker(
            lambda evidence, state: observe_liveness(statuses, evidence, state)
        )
        self._slots = asyncio.Semaphore(4)
        self._lifecycle = hub.subscribe(Topic("sessions.all"), internal=True)
        self._managed: set[str] = set()
        self._linked_managed: set[str] = set()

    async def run(self) -> None:
        lifecycle = asyncio.create_task(self._watch_lifecycle())
        try:
            while True:
                try:
                    rows = await self._sessions.list_sessions()
                    agents = await self._agents.history_store.list()
                    for agent in agents:
                        if agent.session_id:
                            await self._statuses.link(
                                f"agent:{agent.id}", f"session:{agent.session_id}"
                            )
                        await self._attach_agent(agent)
                    self._linked_managed = {
                        agent.session_id
                        for agent in agents
                        if agent.session_id and agent.id in self._adapters
                    }
                    await asyncio.gather(
                        *(self._scan_session(str(row.id)) for row in rows if row.native_id),
                        *(self._scan_agent(agent) for agent in agents if not agent.session_id),
                    )
                except Exception:
                    logger.exception("Conversation observation pass failed")
                await asyncio.sleep(1)
        finally:
            lifecycle.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await lifecycle
            self._hub.unsubscribe(self._lifecycle)
            for adapter in self._adapters.values():
                await adapter.stop()

    async def _attach_agent(self, agent: Agent) -> None:
        managed = self._runtime.registry.status(agent.parent_session_id) == "live"
        if not managed:
            adapter = self._adapters.pop(agent.id, None)
            if adapter is not None:
                await adapter.stop()
                self._tracker.forget(SessionId(f"agent:{agent.id}"))
            return
        if agent.id in self._adapters:
            return
        identifier = SessionId(f"agent:{agent.id}")
        self._tracker.register(identifier)
        adapter = liveness_adapter(
            agent.harness,
            self._hub,
            Topic(f"agent.{agent.id}"),
            self._tracker,
            identifier,
            native_identity=lambda: agent.native_id,
        )
        self._adapters[agent.id] = adapter
        adapter.start()

    async def _scan_session(self, identifier: str) -> None:
        async with self._slots:
            target = f"session:{identifier}"
            managed = (
                self._runtime.registry.status(identifier) == "live"
                or identifier in self._linked_managed
            )
            checkpoint = await self._statuses.repository.checkpoint(target, "native")
            try:
                await self._statuses.flush()
                current = self._statuses.get(target)
                if managed and current.cycle_active and checkpoint is not None:
                    await self._statuses.observe(
                        target, [], "native", {**checkpoint, "managed": True}
                    )
                    self._managed.add(identifier)
                    return
                delta = await self._sessions.work_delta(
                    SessionId(identifier), None if managed else checkpoint
                )
                observations = list(delta.observations) if not managed else []
                if not managed and (delta.baseline or self._statuses.get(target).cycle_active):
                    busy, _ = await self._sessions.external_status(SessionId(identifier))
                    owner = await self._sessions.native_ownership(SessionId(identifier))
                    if owner.owner is SessionOwner.EXTERNAL and busy is True:
                        observations.append(
                            WorkObservation(
                                state="working",
                                progress=True,
                                key=f"native-active:{delta.checkpoint}",
                            )
                        )
                    elif self._statuses.get(target).cycle_active and not any(
                        item.state in {"idle", "working", "waiting"} for item in observations
                    ):
                        if owner.owner is SessionOwner.UNOWNED:
                            state: WorkState = (
                                "idle" if self._statuses.get(target).pending_outcome else "unknown"
                            )
                        elif delta.checkpoint.get("context", {}).get("tasks"):
                            state = "working"
                        else:
                            state = "unknown"
                        observations.append(WorkObservation(state=state))
                if managed:
                    self._managed.add(identifier)
                elif (
                    identifier in self._managed
                    or bool(checkpoint and checkpoint.get("managed"))
                    or (current.observation_source == "live" and not current.cycle_active)
                ):
                    if not self._statuses.get(target).cycle_active:
                        observations = [WorkObservation(source="native")]
                    self._managed.discard(identifier)
                if self._runtime.registry.status(identifier) == "live" and not managed:
                    return
                await self._statuses.observe(
                    target, observations, "native", {**delta.checkpoint, "managed": managed}
                )
            except (MandriError, OSError, ValueError, TypeError, RuntimeError) as error:
                if self._statuses.get(target).cycle_active:
                    await self._statuses.observe(target, [WorkObservation(state="unknown")])
                logger.debug("Conversation observation unavailable: %s", error)

    async def _scan_agent(self, agent: Agent) -> None:
        async with self._slots:
            target = f"agent:{agent.id}"
            checkpoint = await self._statuses.repository.checkpoint(target, "native")
            try:
                managed = agent.id in self._adapters
                delta = await self._agents.history_store.work_delta(
                    agent.id, None if managed else checkpoint
                )
                await self._statuses.observe(
                    target, [] if managed else delta.observations, "native", delta.checkpoint
                )
            except (MandriError, OSError, ValueError, TypeError, RuntimeError) as error:
                logger.debug("Agent observation unavailable: %s", error)

    async def drain_lifecycle(self) -> None:
        await self._lifecycle.queue.join()
        await self._statuses.flush()

    async def _watch_lifecycle(self) -> None:
        while True:
            frame = await self._lifecycle.queue.get()
            self._lifecycle.queue.task_done()
            if frame is None:
                return
            payload = frame.get("payload")
            if not isinstance(payload, dict):
                continue
            raw: Any = payload.get("raw", payload)
            if not isinstance(raw, dict) or not isinstance(raw.get("session_id"), str):
                continue
            target = f"session:{raw['session_id']}"
            if raw.get("type") == "session_stopped":
                outcome: WorkOutcome = "failed" if raw.get("cause") == "crash" else "interrupted"
                self._statuses.enqueue(
                    target,
                    WorkObservation(
                        state="idle",
                        outcome=outcome,
                        key=work_event_key(raw, payload.get("ts")),
                    ),
                )
            elif raw.get("type") == "control_lost":
                self._statuses.enqueue(target, WorkObservation(state="unknown"))
