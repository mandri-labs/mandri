from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Response
from mandri.api.deps import Runtime, Sessions
from mandri.api.errors import NOT_FOUND_CONFLICT, ApiError
from mandri.api.routers.sessions import SessionOut, session_out
from mandri.core.ids import SessionId
from mandri.core.protocol.types import SessionId as ApiSessionId
from mandri.core.types.execution import ProtectionError
from mandri.core.types.worktree_integration import IntegrationPreview, IntegrationStrategy
from mandri.sessions.errors import SessionRunningError
from pydantic import BaseModel, Field

router = APIRouter(prefix="/sessions/{session_id}/worktree", tags=["sessions"])


class ReviewIn(BaseModel):
    target: str = Field(min_length=1)
    strategy: IntegrationStrategy = "squash"
    token: str = Field(min_length=64, max_length=64)


class IntegrateIn(ReviewIn):
    message: str = Field(min_length=1, max_length=10000)


@asynccontextmanager
async def worktree_errors() -> AsyncIterator[None]:
    try:
        yield
    except ProtectionError as error:
        raise ApiError(code=error.code, message=str(error), status=409) from None
    except SessionRunningError as error:
        raise ApiError(code="session_running", message=str(error), status=409) from None


@router.get(
    "/integration", operation_id="preview_worktree_integration", responses=NOT_FOUND_CONFLICT
)
async def preview_worktree_integration(
    session_id: ApiSessionId,
    runtime: Runtime,
    target: str | None = None,
    strategy: IntegrationStrategy = "squash",
) -> IntegrationPreview:
    async with worktree_errors():
        return await runtime.preview_worktree(str(session_id), target, strategy)


@router.post("/integration", operation_id="integrate_worktree", responses=NOT_FOUND_CONFLICT)
async def integrate_worktree(
    session_id: ApiSessionId,
    body: IntegrateIn,
    runtime: Runtime,
    service: Sessions,
) -> SessionOut:
    async with worktree_errors():
        await runtime.integrate_worktree(
            str(session_id),
            body.target,
            body.strategy,
            body.token,
            body.message,
        )
    return session_out(await service.get_session(SessionId(session_id)), service)


@router.post("/resolve", operation_id="resolve_worktree_conflicts", responses=NOT_FOUND_CONFLICT)
async def resolve_worktree_conflicts(
    session_id: ApiSessionId,
    body: ReviewIn,
    runtime: Runtime,
) -> Response:
    async with worktree_errors():
        await runtime.resolve_worktree(str(session_id), body.target, body.strategy, body.token)
    return Response(status_code=204)


@router.post("/finish", operation_id="finish_worktree", responses=NOT_FOUND_CONFLICT)
async def finish_worktree(
    session_id: ApiSessionId, runtime: Runtime, service: Sessions
) -> SessionOut:
    async with worktree_errors():
        await runtime.finish_worktree(str(session_id))
    return session_out(await service.get_session(SessionId(session_id)), service)
