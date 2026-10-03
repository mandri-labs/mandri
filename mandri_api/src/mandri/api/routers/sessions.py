"""REST routes for unified session management."""

from typing import Any

from fastapi import APIRouter, Response
from mandri.api.deps import Gateway, Runtime, Sessions
from mandri.api.errors import (
    ERROR_RESPONSES,
    NOT_FOUND,
    NOT_FOUND_CONFLICT,
    ApiError,
    ErrorEnvelope,
)
from mandri.core.ids import HarnessKind, PageToken, SessionId, SessionState, SessionTitle
from mandri.core.protocol.types import SessionId as ApiSessionId
from mandri.core.types.availability import SessionAvailability
from mandri.core.types.conversation_status import ConversationStatus
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.sessions import InteractionMode, Session
from mandri.core.types.worktrees import Worktree
from mandri.runtime.control.errors import ControlError, ThreadOwnershipError
from mandri.sessions.errors import SessionRunningError
from mandri.sessions.service import HISTORY_DEFAULT_LIMIT, SessionsService
from mandri.sessions.transcripts import PageTokenInvalidError, TranscriptError
from pydantic import BaseModel, Field, ValidationError

router = APIRouter(prefix="/sessions", tags=["sessions"])

HISTORY_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: ERROR_RESPONSES[400],
    404: ERROR_RESPONSES[404],
    503: {"model": ErrorEnvelope, "description": "Harness transcript store unavailable"},
}


class InteractionModeOut(BaseModel):
    mode: str
    applied: str


class SessionOut(BaseModel):
    id: str
    harness: str
    native_id: str | None
    title: str
    project_path: str
    created_at: int
    updated_at: int
    state: str
    model: str | None
    model_source: ModelSource | None = None
    reasoning_effort: str | None = None
    interaction_mode: InteractionModeOut | None = None
    activity: str | None = None
    last_activity_at: int | None = None
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE
    policy_revision: int = 1
    worktree: Worktree | None = None
    status: ConversationStatus | None = None


class PrivacyEntryOut(BaseModel):
    kind: str
    redacted: str


class SessionPrivacyOut(BaseModel):
    revision: int
    entries: list[PrivacyEntryOut]


class RenameIn(BaseModel):
    title: str


class SessionPrivacyIn(BaseModel):
    privacy_mode: PrivacyMode


class SessionModelIn(BaseModel):
    model: str = Field(min_length=1)
    model_source: ModelSource | None = None


@router.patch(
    "/{session_id}/privacy", operation_id="set_session_privacy", responses=NOT_FOUND_CONFLICT,
    response_model_exclude_unset=True,
)
async def set_session_privacy(
    session_id: ApiSessionId, body: SessionPrivacyIn, runtime: Runtime, sessions: Sessions
) -> SessionOut:
    try:
        session = await runtime.set_session_privacy(str(session_id), body.privacy_mode)
    except ProtectionError as error:
        raise ApiError(code=error.code, message=str(error), status=409) from None
    except SessionRunningError as error:
        raise ApiError(code="session_running", message=str(error), status=409) from None
    return session_out(session, sessions)


class HistoryPageOut(BaseModel):
    entries: list[str]
    next_cursor: str | None
    has_more: bool


def session_out(session: Session, service: SessionsService) -> SessionOut:
    values: dict[str, Any] = {
        "id": str(session.id),
        "harness": session.harness.value,
        "native_id": None if session.native_id is None else str(session.native_id),
        "title": session.effective_title,
        "project_path": str(session.project_path),
        "created_at": int(session.created_at),
        "updated_at": int(session.updated_at),
        "state": session.state.value,
        "model": None if session.model is None else str(session.model),
        "model_source": session.model_source,
        "reasoning_effort": session.reasoning_effort,
        "execution_backend": session.execution_backend,
        "privacy_mode": session.privacy_mode,
        "policy_revision": session.policy_revision,
        "worktree": session.worktree,
    }
    if session.interaction_mode is not None:
        values["interaction_mode"] = _mode_out(session.interaction_mode)
    activity = service.activity_of(session.id)
    if activity is not None:
        values["activity"] = activity.state.value
        values["last_activity_at"] = int(activity.last_activity_at)
    if service.statuses is not None:
        values["status"] = service.statuses.get(f"session:{session.id}")
    return SessionOut(**values)


def _mode_out(mode: InteractionMode) -> InteractionModeOut:
    return InteractionModeOut(mode=mode.mode, applied=mode.applied)


def _parse_harness(raw: str | None) -> HarnessKind | None:
    if raw is None:
        return None
    try:
        return HarnessKind(raw)
    except ValueError:
        raise ApiError(
            code="validation_error",
            message=f"Unknown harness {raw!r}",
            status=400,
            detail={"harness": raw},
        ) from None


def _parse_state(raw: str | None) -> SessionState | None:
    if raw is None:
        return None
    try:
        return SessionState(raw)
    except ValueError:
        raise ApiError(
            code="validation_error",
            message=f"Unknown session state {raw!r}",
            status=400,
            detail={"state": raw},
        ) from None


def _history_error(exc: TranscriptError, session_id: str) -> ApiError:
    if isinstance(exc, PageTokenInvalidError):
        return ApiError(
            code="history_cursor_invalid",
            message=str(exc),
            status=400,
            detail={"session_id": session_id},
        )
    return ApiError(
        code="harness_store_unavailable",
        message=str(exc),
        status=503,
        detail={"session_id": session_id},
    )


@router.get(
    "",
    operation_id="list_sessions",
    response_model=list[SessionOut],
    response_model_exclude_unset=True,
)
async def list_sessions(
    service: Sessions,
    harness: str | None = None,
    state: str | None = None,
    project_path: str | None = None,
) -> list[SessionOut]:
    sessions = await service.list_sessions(
        harness=_parse_harness(harness),
        state=_parse_state(state),
        project_path=project_path,
    )
    return [session_out(session, service) for session in sessions]


@router.get(
    "/{session_id}",
    operation_id="get_session",
    responses=NOT_FOUND,
    response_model_exclude_unset=True,
)
async def get_session(session_id: ApiSessionId, service: Sessions) -> SessionOut:
    return session_out(await service.get_session(SessionId(session_id)), service)


@router.get(
    "/{session_id}/privacy",
    operation_id="get_session_privacy",
    responses=HISTORY_RESPONSES,
)
async def get_session_privacy(
    session_id: ApiSessionId, service: Sessions, gateway: Gateway, response: Response
) -> SessionPrivacyOut:
    response.headers["Cache-Control"] = "no-store"
    session = await service.get_session(SessionId(session_id))
    if session.privacy_mode is not PrivacyMode.SURROGATE:
        return SessionPrivacyOut(revision=0, entries=[])
    if gateway.privacy is None or not session.privacy_scope_id:
        raise ApiError(
            code="privacy_state_unavailable", message="Privacy registry is unavailable", status=503
        )
    try:
        revision, entries = await gateway.privacy.scopes.inventory(session.privacy_scope_id)
    except ProtectionError as error:
        raise ApiError(code=error.code, message=str(error), status=503) from None
    return SessionPrivacyOut(
        revision=revision, entries=[PrivacyEntryOut(**item) for item in entries]
    )


@router.get(
    "/{session_id}/history",
    operation_id="get_session_history",
    responses=HISTORY_RESPONSES,
)
async def get_session_history(
    session_id: ApiSessionId,
    service: Sessions,
    cursor: str | None = None,
    limit: int = HISTORY_DEFAULT_LIMIT,
) -> HistoryPageOut:
    token = PageToken(cursor) if cursor is not None else None
    try:
        page = await service.history(SessionId(session_id), token, limit)
    except TranscriptError as error:
        raise _history_error(error, session_id) from None
    return HistoryPageOut(
        entries=[str(entry) for entry in page.entries],
        next_cursor=None if page.next_token is None else str(page.next_token),
        has_more=page.has_more,
    )


@router.patch(
    "/{session_id}",
    operation_id="rename_session",
    responses=NOT_FOUND,
    response_model_exclude_unset=True,
)
async def rename_session(
    session_id: ApiSessionId,
    body: RenameIn,
    service: Sessions,
) -> SessionOut:
    return session_out(
        await service.rename_session(SessionId(session_id), SessionTitle(body.title)), service
    )


@router.delete(
    "/{session_id}",
    operation_id="delete_session",
    status_code=204,
    responses=NOT_FOUND,
)
async def delete_session(
    session_id: ApiSessionId,
    service: Sessions,
    purge: bool = False,
    discard_worktree: bool = False,
) -> Response:
    try:
        options = {"discard_worktree": True} if discard_worktree else {}
        await service.delete_session(SessionId(session_id), purge=purge, **options)
    except ProtectionError as error:
        raise ApiError(code=error.code, message=str(error), status=409) from None
    except SessionRunningError as error:
        raise ApiError(
            code="session_running",
            message=str(error),
            status=409,
            detail={"session_id": str(session_id)},
        ) from None
    return Response(status_code=204)


@router.patch(
    "/{session_id}/model",
    operation_id="set_session_model",
    responses=NOT_FOUND_CONFLICT,
    response_model_exclude_unset=True,
)
async def set_session_model(
    session_id: ApiSessionId,
    body: dict[str, Any],
    service: Sessions,
) -> SessionOut:
    if "harness" in body:
        raise ApiError(
            code="session_conflict",
            message="Harness binding is immutable",
            status=409,
            detail={"session_id": str(session_id)},
        )
    if "model" not in body or set(body) - {"model", "model_source"}:
        raise ApiError(
            code="validation_error",
            message="Body must contain model and optionally model_source",
            status=400,
            detail={"received": sorted(body)},
        )
    try:
        parsed = SessionModelIn.model_validate(body)
    except ValidationError as error:
        raise ApiError(
            code="validation_error",
            message="Invalid model",
            status=400,
            detail={"errors": error.errors(include_url=False)},
        ) from None
    if parsed.model_source is None:
        session = await service.set_session_model(SessionId(session_id), parsed.model)
    else:
        session = await service.set_session_model(
            SessionId(session_id), parsed.model, model_source=parsed.model_source
        )
    return session_out(session, service)


class ReleaseSessionIn(BaseModel):
    confirmed: bool = False


@router.get("/{session_id}/availability", operation_id="session_availability", responses=NOT_FOUND)
async def session_availability(session_id: ApiSessionId, runtime: Runtime) -> SessionAvailability:
    return await runtime.session_availability(str(session_id))


@router.post("/{session_id}/release", operation_id="release_session", responses=NOT_FOUND_CONFLICT)
async def release_session(
    session_id: ApiSessionId, body: ReleaseSessionIn, runtime: Runtime
) -> SessionAvailability:
    try:
        return await runtime.release_session(str(session_id), confirmed=body.confirmed)
    except SessionRunningError as error:
        raise ApiError("session_running", str(error), 409) from None


@router.post(
    "/{session_id}/restore-native-model",
    operation_id="restore_native_model",
    responses=NOT_FOUND_CONFLICT,
)
async def restore_native_model(session_id: ApiSessionId, runtime: Runtime) -> SessionAvailability:
    try:
        return await runtime.restore_native_model(str(session_id))
    except (SessionRunningError, ThreadOwnershipError) as error:
        raise ApiError("session_running", str(error), 409) from None
    except (ControlError, TimeoutError):
        raise ApiError("native_restore_failed", "Native model restoration failed", 502) from None


class RenameWorktreeIn(BaseModel):
    id: str = Field(min_length=1, max_length=100)


@router.patch(
    "/{session_id}/worktree", operation_id="rename_session_worktree", responses=NOT_FOUND_CONFLICT
)
async def rename_session_worktree(
    session_id: ApiSessionId,
    body: RenameWorktreeIn,
    runtime: Runtime,
    service: Sessions,
) -> SessionOut:
    try:
        await runtime.rename_worktree(str(session_id), body.id)
    except ProtectionError as error:
        raise ApiError(code=error.code, message=str(error), status=409) from None
    except SessionRunningError as error:
        raise ApiError(code="session_running", message=str(error), status=409) from None
    return session_out(await service.get_session(SessionId(session_id)), service)
