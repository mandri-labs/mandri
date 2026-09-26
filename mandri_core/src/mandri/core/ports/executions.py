from typing import Protocol

from mandri.core.types.execution import ExecutionPhase
from mandri.core.types.execution_generation import ExecutionGeneration


class ExecutionRepositoryPort(Protocol):
    async def create(self, session_id: str, owner: str, context: str) -> ExecutionGeneration: ...

    async def update(
        self,
        record: ExecutionGeneration,
        phase: ExecutionPhase,
        *,
        context: str | None = None,
        container_id: str | None = None,
        reason: str | None = None,
    ) -> ExecutionGeneration: ...

    async def latest(self, session_id: str) -> ExecutionGeneration | None: ...

    async def active(self, owner: str) -> list[ExecutionGeneration]: ...

    async def reconcile_stopped_sessions(
        self, owner: str, excluded: frozenset[str]
    ) -> list[str]: ...
