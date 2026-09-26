from dataclasses import dataclass

from mandri.core.types.execution import ExecutionPhase


@dataclass(frozen=True)
class ExecutionGeneration:
    session_id: str
    generation: int
    owner: str
    phase: ExecutionPhase
    revision: int
    context: str
    created_at: int
    updated_at: int
    container_id: str | None = None
    reason: str | None = None
