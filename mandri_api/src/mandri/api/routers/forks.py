from typing import Any

from fastapi import APIRouter
from mandri.api.deps import Runtime
from mandri.api.errors import ApiError
from mandri.api.routers.runtime import RuntimeSessionOut
from mandri.core.protocol.types import SessionId as ApiSessionId
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError
from mandri.runtime.control.errors import ControlError
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.sessions.errors import SessionRunningError
from pydantic import BaseModel, ConfigDict

router = APIRouter(tags=["sessions"])


class SessionForkIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    execution_backend: ExecutionBackend
    privacy_mode: PrivacyMode
    mode: str | None = None
    operation_id: str | None = None
    worktree: bool = False
    worktree_id: str | None = None


@router.post("/sessions/{session_id}/fork", operation_id="fork_runtime_session", status_code=201)
async def fork_runtime_session(
    session_id: ApiSessionId, body: SessionForkIn, runtime: Runtime
) -> RuntimeSessionOut:
    options: dict[str, Any] = {}
    if body.worktree or body.worktree_id is not None:
        options.update(worktree=True, worktree_id=body.worktree_id)
    try:
        session = await runtime.fork_session(
            str(session_id),
            body.execution_backend,
            body.privacy_mode,
            mode=body.mode,
            operation_id=body.operation_id,
            **options,
        )
    except ProtectionError as error:
        raise ApiError(code=error.code, message=str(error), status=409) from None
    except DockerExecutionError as error:
        raise ApiError(code=error.reason, message=str(error), status=409) from None
    except SessionRunningError as error:
        raise ApiError(code="session_running", message=str(error), status=409) from None
    except ControlError:
        raise ApiError(
            code="session_transition_failed", message="Native history fork failed", status=409
        ) from None
    return RuntimeSessionOut(
        id=session.id,
        harness=session.harness,
        gateway_route_id=session.route_id,
        state="live",
        project_path=session.project_path,
        mode=session.mode,
        execution_backend=session.execution_backend,
        privacy_mode=session.privacy_mode,
        policy_revision=session.policy_revision,
        worktree=getattr(session, "worktree", None),
    )
