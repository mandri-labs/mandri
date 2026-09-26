from typing import Any

from mandri.core.clock import system_now_ms
from mandri.core.ports.database import DatabasePort
from mandri.core.types.execution import ExecutionPhase, ProtectionError
from mandri.core.types.execution_generation import ExecutionGeneration

_NEXT = {
    ExecutionPhase.CHECKING: {
        ExecutionPhase.PREPARING_IMAGE,
        ExecutionPhase.STARTING,
        ExecutionPhase.BLOCKED,
        ExecutionPhase.FAILED,
        ExecutionPhase.STOPPING,
    },
    ExecutionPhase.PREPARING_IMAGE: {
        ExecutionPhase.PREPARING_STATE,
        ExecutionPhase.BLOCKED,
        ExecutionPhase.FAILED,
        ExecutionPhase.STOPPING,
    },
    ExecutionPhase.PREPARING_STATE: {
        ExecutionPhase.STARTING,
        ExecutionPhase.BLOCKED,
        ExecutionPhase.FAILED,
        ExecutionPhase.STOPPING,
    },
    ExecutionPhase.STARTING: {
        ExecutionPhase.READY,
        ExecutionPhase.BLOCKED,
        ExecutionPhase.FAILED,
        ExecutionPhase.STOPPING,
    },
    ExecutionPhase.READY: {ExecutionPhase.FAILED, ExecutionPhase.STOPPING},
    ExecutionPhase.STOPPING: {ExecutionPhase.STOPPED, ExecutionPhase.FAILED},
    ExecutionPhase.BLOCKED: {ExecutionPhase.STOPPING},
    ExecutionPhase.FAILED: {ExecutionPhase.STOPPING},
    ExecutionPhase.STOPPED: set(),
}


def execution_record(row: dict[str, Any]) -> ExecutionGeneration:
    return ExecutionGeneration(
        session_id=str(row["session_id"]),
        generation=int(row["generation"]),
        owner=str(row["owner"]),
        phase=ExecutionPhase(row["phase"]),
        revision=int(row["revision"]),
        context=str(row["context"]),
        container_id=row["container_id"],
        reason=row["reason"],
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
    )


class ExecutionRepository:
    def __init__(self, database: DatabasePort) -> None:
        self._database = database

    async def create(self, session_id: str, owner: str, context: str) -> ExecutionGeneration:
        now = system_now_ms()
        row = await self._database.fetch_one(
            "INSERT INTO execution_generation"
            " (session_id,generation,owner,phase,revision,context,created_at,updated_at)"
            " SELECT ?,COALESCE(MAX(generation),0)+1,?,'checking',1,?,?,?"
            " FROM execution_generation WHERE session_id=? RETURNING *",
            (session_id, owner, context, now, now, session_id),
        )
        if row is None:
            raise ProtectionError("execution_state_unavailable", "Execution allocation failed")
        return execution_record(row)

    async def update(
        self,
        record: ExecutionGeneration,
        phase: ExecutionPhase,
        *,
        context: str | None = None,
        container_id: str | None = None,
        reason: str | None = None,
    ) -> ExecutionGeneration:
        if phase is not record.phase and phase not in _NEXT[record.phase]:
            raise ProtectionError(
                "execution_state_conflict", "Execution phase transition is invalid"
            )
        row = await self._database.fetch_one(
            "UPDATE execution_generation SET phase=?,revision=revision+1,context=?,"
            "container_id=?,reason=?,updated_at=?"
            " WHERE session_id=? AND generation=? AND revision=? RETURNING *",
            (
                phase.value,
                context if context is not None else record.context,
                container_id if container_id is not None else record.container_id,
                reason,
                system_now_ms(),
                record.session_id,
                record.generation,
                record.revision,
            ),
        )
        if row is None:
            raise ProtectionError(
                "execution_state_conflict", "Execution state changed concurrently"
            )
        return execution_record(row)

    async def latest(self, session_id: str) -> ExecutionGeneration | None:
        row = await self._database.fetch_one(
            "SELECT * FROM execution_generation WHERE session_id=?"
            " ORDER BY generation DESC LIMIT 1",
            (session_id,),
        )
        return execution_record(row) if row else None

    async def active(self, owner: str) -> list[ExecutionGeneration]:
        rows = await self._database.fetch_all(
            "SELECT * FROM execution_generation WHERE owner=?"
            " AND phase NOT IN ('stopped','blocked','failed')",
            (owner,),
        )
        return [execution_record(row) for row in rows]

    async def reconcile_stopped_sessions(self, owner: str, excluded: frozenset[str]) -> list[str]:
        exclusion = (
            " AND id NOT IN (" + ",".join("?" for _ in excluded) + ")" if excluded else ""
        )
        rows = await self._database.fetch_all(
            "UPDATE session SET state='stopped',updated_at=?"
            " WHERE state='live' AND execution_backend='docker' AND deleted=0"
            + exclusion
            + " AND EXISTS (SELECT 1 FROM execution_generation AS execution"
            " WHERE execution.session_id=session.id AND execution.owner=?"
            " AND execution.phase IN ('stopped','failed','blocked')"
            " AND execution.generation=(SELECT MAX(latest.generation)"
            " FROM execution_generation AS latest WHERE latest.session_id=session.id))"
            " RETURNING id",
            (system_now_ms(), *sorted(excluded), owner),
        )
        return [str(row["id"]) for row in rows]
