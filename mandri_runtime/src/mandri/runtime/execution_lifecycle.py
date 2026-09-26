import asyncio
import json
from typing import Any

from mandri.core.clock import system_now_ms
from mandri.core.hub import Hub, Topic
from mandri.core.ports.executions import ExecutionRepositoryPort
from mandri.core.types.execution import ExecutionPhase
from mandri.core.types.execution_generation import ExecutionGeneration
from mandri.runtime.session_feed import session_topic


class ExecutionLifecycle:
    def __init__(self, repository: ExecutionRepositoryPort | None, hub: Hub | None) -> None:
        self.repository = repository
        self._hub = hub
        self._records: dict[str, ExecutionGeneration] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def begin(self, session_id: str, owner: str, context: dict[str, Any]) -> None:
        if self.repository is None:
            return
        record = await self.repository.create(session_id, owner, json.dumps(context))
        self._records[session_id] = record
        self._publish(record)

    async def phase(
        self,
        session_id: str,
        phase: ExecutionPhase,
        *,
        context: dict[str, Any] | None = None,
        container_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        if self.repository is None:
            return
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            record = self._records.get(session_id)
            if record is None or (
                record.phase is phase and context is None and container_id is None
            ):
                return
            updated = await self.repository.update(
                record,
                phase,
                context=json.dumps({**json.loads(record.context), **context})
                if context is not None
                else None,
                container_id=container_id,
                reason=reason,
            )
            self._records[session_id] = updated
            self._publish(updated)

    async def latest(self, session_id: str) -> ExecutionGeneration | None:
        if self.repository is None:
            return None
        return await self.repository.latest(session_id)

    async def reconcile(self, owner: str, active_container_ids: frozenset[str]) -> None:
        if self.repository is None:
            return
        for record in await self.repository.active(owner):
            if record.container_id and record.container_id in active_container_ids:
                continue
            updated = await self.repository.update(
                record, ExecutionPhase.FAILED, reason="execution_interrupted"
            )
            self._records[record.session_id] = updated
            self._publish(updated)

    async def reconcile_stopped_sessions(self, owner: str, excluded: frozenset[str]) -> list[str]:
        if self.repository is None:
            return []
        stopped = await self.repository.reconcile_stopped_sessions(owner, excluded)
        if stopped and self._hub is not None:
            self._hub.publish(
                Topic("sessions.all"),
                {"source": "mandri", "ts": system_now_ms(), "raw": {"type": "sessions_changed"}},
            )
        return stopped

    def _publish(self, record: ExecutionGeneration) -> None:
        if self._hub is not None:
            payload = {
                "type": "execution.updated",
                "session_id": record.session_id,
                "generation": record.generation,
                "revision": record.revision,
                "phase": record.phase.value,
                "reason": record.reason,
                "operation_id": json.loads(record.context).get("operation_id"),
            }
            self._hub.publish(Topic("executions.all"), payload)
            self._hub.publish(Topic(f"execution.{record.session_id}"), payload)
            self._hub.publish(
                session_topic(record.session_id),
                {
                    "source": "mandri",
                    "type": "execution.updated",
                    "raw": payload,
                    "ts": system_now_ms(),
                },
            )
