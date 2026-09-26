"""REST routes for daemon-started runtime sessions and harness availability."""

from typing import Any

from fastapi import APIRouter, Request, Response
from mandri.api.deps import Gateway, Runtime, Sessions
from mandri.api.errors import NOT_FOUND, NOT_FOUND_CONFLICT, ApiError
from mandri.api.routers._effort import normalize_effort, validate_effort
from mandri.api.routers.sessions import SessionOut, session_out
from mandri.config.errors import ConfigError
from mandri.core.ids import HarnessKind, SessionId
from mandri.core.protocol.types import SessionId as ApiSessionId
from mandri.core.types.execution import (
    ExecutionBackend,
    PrivacyMode,
    ProtectionError,
    SessionPolicy,
)
from mandri.core.types.model_selection import ModelSource, model_capabilities
from mandri.core.types.worktrees import Worktree
from mandri.providers.errors import ProviderInvalidError, ProviderNotFoundError
from mandri.runtime.control.errors import ControlTransportError, ThreadOwnershipError
from mandri.runtime.errors import (
    HarnessNotInstalledError,
    OpencodeSessionMissingError,
    ProcessSpawnError,
    SessionNotResumableError,
    SessionNotRunningError,
)
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.runtime.native_catalog import NativeModel
from mandri.sessions.errors import SessionRunningError
from mandri.sessions.service import SessionsService
from mandri.sessions.sync import HarnessState
from mandri.sessions.transcripts.errors import TranscriptError
from pydantic import BaseModel, ConfigDict

router = APIRouter(tags=["runtime"])


class RuntimeStartIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    harness: str
    model: str
    cwd: str
    mode: str | None = None
    effort: str | None = None
    model_source: ModelSource | None = None
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE
    operation_id: str | None = None
    worktree: bool = False
    worktree_id: str | None = None


class RuntimeResumeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str | None = None


class SessionEffortIn(BaseModel):
    effort: str | None = None


class RuntimeSessionOut(BaseModel):
    id: str
    harness: str
    gateway_route_id: str | None
    state: str
    project_path: str | None = None
    mode: str | None = None
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE
    policy_revision: int = 1
    worktree: Worktree | None = None


class RuntimeOut(BaseModel):
    harness: str
    installed: bool
    version: str | None = None
    degraded: bool
    capabilities: dict[str, Any] | None = None


class AgyHookIn(BaseModel):
    event: str
    data: dict[str, Any]


@router.post("/runtime/agy/{session_id}/hook", include_in_schema=False)
async def agy_hook(
    session_id: str, body: AgyHookIn, request: Request, runtime: Runtime
) -> dict[str, Any]:
    authorization = request.headers.get("authorization", "")
    if not authorization.startswith("Bearer "):
        raise ApiError(code="unauthorized", message="Missing hook token", status=401)
    try:
        return await runtime.agy_hook(session_id, authorization[7:], body.event, body.data)
    except PermissionError:
        raise ApiError(code="unauthorized", message="Invalid hook token", status=401) from None


def _parse_harness(raw: str) -> HarnessKind:
    try:
        return HarnessKind(raw)
    except ValueError:
        raise ApiError(
            code="validation_error",
            message=f"Unknown harness {raw!r}",
            status=400,
            detail={"harness": raw},
        ) from None


@router.post("/runtime/sessions", operation_id="start_runtime_session", status_code=201)
async def start_runtime_session(
    body: RuntimeStartIn, wiring: Gateway, runtime: Runtime
) -> RuntimeSessionOut:
    _parse_harness(body.harness)
    if body.model_source is not ModelSource.NATIVE:
        validate_effort(wiring, body.model, body.effort)
    options: dict[str, Any] = {}
    defaults = getattr(runtime, "creation_policy", SessionPolicy())
    execution_backend = (
        body.execution_backend
        if "execution_backend" in body.model_fields_set
        else defaults.execution_backend
    )
    privacy_mode = (
        body.privacy_mode if "privacy_mode" in body.model_fields_set else defaults.privacy_mode
    )
    try:
        SessionPolicy(execution_backend, privacy_mode).validate(
            body.model_source or ModelSource.GATEWAY
        )
    except ProtectionError as error:
        raise ApiError(code=error.code, message=str(error), status=400) from None
    if (
        execution_backend is not ExecutionBackend.HOST
        or "execution_backend" in body.model_fields_set
    ):
        options["execution_backend"] = execution_backend
    if privacy_mode is not PrivacyMode.NONE or "privacy_mode" in body.model_fields_set:
        options["privacy_mode"] = privacy_mode
    if body.model_source is ModelSource.NATIVE:
        options["model_source"] = body.model_source
    if body.worktree or body.worktree_id is not None:
        options["worktree"] = True
        options["worktree_id"] = body.worktree_id
    if body.operation_id is not None:
        options["operation_id"] = body.operation_id
    try:
        session = await runtime.start_session(
            body.harness,
            body.model,
            body.cwd,
            mode=body.mode,
            effort=normalize_effort(body.effort),
            **options,
        )
    except ProtectionError as error:
        raise ApiError(code=error.code, message=str(error), status=400) from None
    except ProviderNotFoundError as error:
        raise ApiError(code="provider_not_found", message=str(error), status=404) from None
    except DockerExecutionError as error:
        raise ApiError(code=error.reason, message=str(error), status=409) from None
    except ProviderInvalidError as error:
        raise ApiError(code="validation_error", message=str(error), status=400) from None
    except HarnessNotInstalledError as error:
        raise ApiError(code="harness_not_installed", message=str(error), status=400) from None
    except ConfigError as error:
        raise ApiError(
            code="validation_error",
            message=str(error),
            status=400,
            detail={"mode": body.mode},
        ) from None
    return RuntimeSessionOut(
        id=session.id,
        harness=session.harness,
        gateway_route_id=session.route_id,
        state="live",
        project_path=session.project_path,
        mode=session.mode,
        execution_backend=getattr(session, "execution_backend", ExecutionBackend.HOST),
        privacy_mode=getattr(session, "privacy_mode", PrivacyMode.NONE),
        policy_revision=getattr(session, "policy_revision", 1),
        worktree=getattr(session, "worktree", None),
    )


@router.patch(
    "/runtime/sessions/{session_id}/effort",
    operation_id="set_session_effort",
    responses=NOT_FOUND,
)
async def set_session_effort(
    session_id: ApiSessionId,
    body: SessionEffortIn,
    wiring: Gateway,
    runtime: Runtime,
    sessions: Sessions,
) -> SessionOut:
    record = await sessions.get_session(SessionId(session_id))
    if record.model is not None and record.model_source is ModelSource.GATEWAY:
        validate_effort(wiring, record.model, body.effort)
    try:
        session = await runtime.set_session_effort(str(session_id), normalize_effort(body.effort))
    except SessionNotResumableError as error:
        raise ApiError(code="session_not_resumable", message=str(error), status=400) from None
    return session_out(session, sessions)


@router.post("/sessions/{session_id}/stop", operation_id="stop_session", status_code=204)
async def stop_session(session_id: ApiSessionId, runtime: Runtime, sessions: Sessions) -> Response:
    await sessions.get_session(SessionId(session_id))
    try:
        await runtime.stop_session(str(session_id), force=True, restore_native=False)
    except SessionNotRunningError as error:
        raise ApiError(code="session_not_running", message=str(error), status=409) from None
    return Response(status_code=204)


@router.post(
    "/sessions/{session_id}/resume",
    operation_id="resume_session",
    responses=NOT_FOUND_CONFLICT,
)
async def resume_session(
    session_id: ApiSessionId,
    runtime: Runtime,
    sessions: Sessions,
    body: RuntimeResumeIn | None = None,
) -> RuntimeSessionOut:
    await sessions.get_session(SessionId(session_id))
    try:
        session = await runtime.resume_session(str(session_id), mode=body.mode if body else None)
    except ProtectionError as error:
        raise ApiError(code=error.code, message=str(error), status=409) from None
    except DockerExecutionError as error:
        raise ApiError(code=error.reason, message=str(error), status=409) from None
    except ConfigError as error:
        raise ApiError(code="validation_error", message=str(error), status=400) from None
    except SessionRunningError as error:
        raise ApiError(code="session_running", message=str(error), status=409) from None
    except SessionNotResumableError as error:
        raise ApiError(code="session_not_resumable", message=str(error), status=400) from None
    except OpencodeSessionMissingError as error:
        raise ApiError(code="session_not_resumable", message=str(error), status=400) from None
    except ThreadOwnershipError as error:
        raise ApiError(code="session_owned_externally", message=str(error), status=409) from None
    except ControlTransportError as error:
        raise ApiError(code="control_delivery_failed", message=str(error), status=503) from None
    except TranscriptError as error:
        raise ApiError(code="harness_store_unavailable", message=str(error), status=503) from None
    except HarnessNotInstalledError as error:
        raise ApiError(code="harness_not_installed", message=str(error), status=400) from None
    return RuntimeSessionOut(
        id=session.id,
        harness=session.harness,
        gateway_route_id=session.route_id,
        state="live",
        project_path=session.project_path,
        mode=session.mode,
        execution_backend=getattr(session, "execution_backend", ExecutionBackend.HOST),
        privacy_mode=getattr(session, "privacy_mode", PrivacyMode.NONE),
        policy_revision=getattr(session, "policy_revision", 1),
        worktree=getattr(session, "worktree", None),
    )


def _degraded(sessions: SessionsService, kind: HarnessKind) -> bool:
    state: HarnessState = sessions.harness_state(kind)
    return state.degraded


def _capabilities(kind: HarnessKind) -> dict[str, Any]:
    capabilities: dict[str, Any] = {"model_sources": list(model_capabilities(kind).sources)}
    if kind is HarnessKind.AGY:
        capabilities.update(
            steering="stop_resume",
            input_types=["text"],
            permission_modes=["default", "acceptEdits", "plan", "bypassPermissions"],
        )
    elif kind is HarnessKind.PI:
        capabilities.update(
            steering="native",
            input_types=["text", "image"],
            permission_modes=["default", "acceptEdits", "plan", "bypassPermissions"],
        )
    return capabilities


@router.get("/runtimes", operation_id="list_runtimes")
async def list_runtimes(runtime: Runtime, sessions: Sessions) -> list[RuntimeOut]:
    installed = set(runtime.host_harnesses())
    runtimes = [
        RuntimeOut(
            harness=kind.value,
            installed=kind.value in installed,
            degraded=_degraded(sessions, kind),
            capabilities=_capabilities(kind),
        )
        for kind in HarnessKind
    ]
    known = {kind.value for kind in HarnessKind}
    runtimes.extend(
        RuntimeOut(
            harness=harness,
            installed=True,
            degraded=False,
            capabilities={"model_sources": list(model_capabilities(harness).sources)},
        )
        for harness in sorted(installed - known)
    )
    return runtimes


@router.get("/runtimes/{harness}/models", operation_id="list_native_models")
async def list_native_models(
    harness: str, runtime: Runtime, cwd: str | None = None
) -> list[NativeModel]:
    _parse_harness(harness)
    try:
        return await runtime.native_models(harness, cwd)
    except (ProviderInvalidError, HarnessNotInstalledError) as error:
        raise ApiError(code="validation_error", message=str(error), status=400) from None
    except (ControlTransportError, ProcessSpawnError, TimeoutError, OSError) as error:
        raise ApiError(code="native_catalog_unavailable", message=str(error), status=503) from None
