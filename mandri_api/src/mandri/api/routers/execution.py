
from fastapi import APIRouter, Response
from mandri.api.deps import Runtime, Sessions
from mandri.api.errors import ApiError
from mandri.core.ids import SessionId
from mandri.core.protocol.types import SessionId as ApiSessionId
from mandri.core.types.execution import (
    ExecutionBackend,
    ExecutionPhase,
    PrivacyMode,
    ProtectionError,
)
from mandri.runtime.registry import LIVE
from pydantic import BaseModel

router = APIRouter(tags=["runtime"])


@router.post(
    "/runtime/operations/{operation_id}/cancel",
    operation_id="cancel_runtime_start",
    status_code=204,
)
async def cancel_runtime_start(operation_id: str, runtime: Runtime) -> Response:
    try:
        await runtime.cancel_start(operation_id)
    except ProtectionError as error:
        raise ApiError(code=error.code, message=str(error), status=409) from None
    return Response(status_code=204)


class ExecutionStatusOut(BaseModel):
    session_id: str
    execution_backend: ExecutionBackend
    privacy_mode: PrivacyMode
    policy_revision: int
    generation: int
    revision: int
    phase: ExecutionPhase
    reason: str | None = None
    container_id: str | None = None
    effective_binding: bool


@router.get("/runtime/sessions/{session_id}/execution", operation_id="execution_status")
async def execution_status(
    session_id: ApiSessionId, runtime: Runtime, sessions: Sessions
) -> ExecutionStatusOut:
    session = await sessions.get_session(SessionId(session_id))
    generation = await sessions.execution_generation(SessionId(session_id))
    process = runtime.registry.process(str(session_id))
    live = (
        runtime.registry.status(str(session_id)) == LIVE
        and process is not None
        and process.returncode is None
    )
    return ExecutionStatusOut(
        session_id=str(session_id),
        execution_backend=session.execution_backend,
        privacy_mode=session.privacy_mode,
        policy_revision=session.policy_revision,
        generation=generation.generation if generation else 0,
        revision=generation.revision if generation else 0,
        phase=generation.phase
        if generation
        else ExecutionPhase.READY
        if live
        else ExecutionPhase.STOPPED,
        reason=generation.reason if generation else None,
        container_id=generation.container_id if generation else None,
        effective_binding=live and (generation is None or generation.phase is ExecutionPhase.READY),
    )
