"""Action handlers binding websocket request frames to the approvals and control domains."""

import contextlib
import typing
from collections.abc import Awaitable, Callable

from mandri.api.agent_actions import register_agent_actions
from mandri.api.command_actions import register_command_actions
from mandri.api.conversation_actions import register_conversation_actions
from mandri.api.routers.sessions import session_out
from mandri.core.ids import ModeApplication, PageToken, SessionId
from mandri.core.protocol.errors import ProtocolError, ProtocolErrorCode
from mandri.core.protocol.registry import (
    ActionRegistry,
    ApprovalAnswerParams,
    ApprovalCancelParams,
    SessionHistoryParams,
    SessionInterruptParams,
    SessionModeParams,
    SessionPromptParams,
)
from mandri.core.types.approvals import (
    ApprovalAlreadyAnsweredError,
    ApprovalError,
    ApprovalExpiredError,
    ApprovalNotPendingError,
    DuplicateApprovalAnswerError,
)
from mandri.core.types.sessions import SessionError
from mandri.core.work_content import contains_completed_content
from mandri.runtime.agents import AgentService
from mandri.runtime.attachments import AttachmentError, AttachmentStorageError
from mandri.runtime.control import PromptState
from mandri.runtime.control.errors import (
    ControlError,
    ControlTransportError,
    HarnessNotInitializedError,
    ModeRejectedError,
    PromptDeliveryFailedError,
    PromptDeliveryUnknownError,
    SteerNoActiveTurnError,
    SteerUnsupportedError,
)
from mandri.runtime.errors import RuntimeDomainError, SessionNotRunningError
from mandri.runtime.service import RuntimeService
from mandri.sessions.errors import SessionConflictError
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts.errors import PageTokenInvalidError
from pydantic import BaseModel


def build_action_registry(
    runtime: RuntimeService,
    sessions: SessionsService | None = None,
    agents: AgentService | None = None,
) -> ActionRegistry:
    registry = ActionRegistry()
    register_command_actions(registry, runtime.commands)
    registry.register("approval.answer", _make_answer_handler(runtime))
    registry.register("approval.cancel", _make_cancel_handler(runtime))
    registry.register("session.mode", _make_mode_handler(runtime))
    registry.register("session.prompt", _make_prompt_handler(runtime))
    registry.register("session.interrupt", _make_interrupt_handler(runtime))
    if sessions is not None:
        register_conversation_actions(registry, sessions, agents)
        registry.register("session.history", _make_history_handler(sessions, runtime))
        registry.register("session.list", _make_list_handler(sessions))
    if agents is not None:
        register_agent_actions(registry, agents, sessions)
    return registry


def _make_list_handler(
    sessions: SessionsService,
) -> Callable[[BaseModel], Awaitable[dict[str, typing.Any]]]:
    async def handle(params: BaseModel) -> dict[str, typing.Any]:
        if sessions.statuses is not None:
            await sessions.statuses.flush()
        rows = await sessions.list_sessions()
        return {
            "sessions": [session_out(row, sessions).model_dump(exclude_unset=True) for row in rows]
        }

    return handle


def _make_history_handler(
    sessions: SessionsService,
    runtime: RuntimeService | None = None,
) -> Callable[[BaseModel], Awaitable[dict[str, typing.Any]]]:
    async def handle(params: BaseModel) -> dict[str, typing.Any]:
        if not isinstance(params, SessionHistoryParams):
            raise ProtocolError(ProtocolErrorCode.INVALID_PARAMS, "invalid history params")
        completion_revision = None
        status = None
        if params.cursor is None and sessions.statuses is not None:
            await sessions.statuses.flush()
            status = sessions.statuses.get(f"session:{params.session_id}")
            if status.work_state == "idle":
                completion_revision = status.completion_revision
        try:
            page = await sessions.history(
                SessionId(str(params.session_id)),
                PageToken(params.cursor) if params.cursor is not None else None,
                params.limit,
                recent=True,
            )
        except PageTokenInvalidError as error:
            raise ProtocolError(
                ProtocolErrorCode.INVALID_PARAMS, "History cursor expired"
            ) from error
        except (SessionError, OSError, ValueError, TypeError) as error:
            raise ProtocolError(
                ProtocolErrorCode.HARNESS_STORE_UNAVAILABLE, "Stored history is unavailable"
            ) from error
        busy, model = None, None
        if runtime is None or runtime.registry.status(str(params.session_id)) != "live":
            with contextlib.suppress(SessionError, OSError, ValueError, TypeError):
                busy, model = await sessions.external_status(SessionId(str(params.session_id)))
        turn_active = busy
        if runtime is not None and runtime.registry.status(str(params.session_id)) == "live":
            busy = False
            turn_active = runtime.is_busy(str(params.session_id))
        if completion_revision is not None and status is not None:
            session = await sessions.get_session(SessionId(str(params.session_id)))
            if not contains_completed_content(
                session.harness, list(page.entries), status.completion_content_key
            ):
                completion_revision = None
        return {
            "completion_revision": completion_revision,
            "completion_target": f"session:{params.session_id}",
            "entries": page.entries,
            "next_cursor": page.next_token,
            "has_more": page.has_more,
            "turn_active": turn_active,
            "external_busy": busy,
            "external_model": model,
        }

    return handle


def _make_answer_handler(
    runtime: RuntimeService,
) -> Callable[[BaseModel], Awaitable[dict[str, typing.Any]]]:
    async def handle(params: BaseModel) -> dict[str, typing.Any]:
        if not isinstance(params, ApprovalAnswerParams):
            raise ProtocolError(
                ProtocolErrorCode.INVALID_PARAMS, "invalid params for action: approval.answer"
            )
        try:
            request = await runtime.answer_approval(
                str(params.approval_id),
                params.decision,
                params.updated_input,
                **({"answers": params.answers} if params.answers is not None else {}),
                **(
                    {"permission_mode": params.permission_mode}
                    if params.permission_mode is not None
                    else {}
                ),
            )
        except ControlError as exc:
            raise ProtocolError(ProtocolErrorCode.CONTROL_DELIVERY_FAILED, str(exc)) from exc
        except ApprovalError as exc:
            raise _protocol_error(exc) from exc
        return {"approval_id": str(request.id), "status": request.status.value}

    return handle


def _make_cancel_handler(
    runtime: RuntimeService,
) -> Callable[[BaseModel], Awaitable[dict[str, typing.Any]]]:
    async def handle(params: BaseModel) -> dict[str, typing.Any]:
        if not isinstance(params, ApprovalCancelParams):
            raise ProtocolError(
                ProtocolErrorCode.INVALID_PARAMS, "invalid params for action: approval.cancel"
            )
        try:
            request = await runtime.cancel_approval(str(params.approval_id))
        except ControlError as exc:
            raise ProtocolError(ProtocolErrorCode.CONTROL_DELIVERY_FAILED, str(exc)) from exc
        except ApprovalError as exc:
            raise _protocol_error(exc) from exc
        return {"approval_id": str(request.id), "status": request.status.value}

    return handle


def _make_mode_handler(
    runtime: RuntimeService,
) -> Callable[[BaseModel], Awaitable[dict[str, typing.Any]]]:
    async def handle(params: BaseModel) -> dict[str, typing.Any]:
        if not isinstance(params, SessionModeParams):
            raise ProtocolError(
                ProtocolErrorCode.INVALID_PARAMS, "invalid params for action: session.mode"
            )
        try:
            application = await runtime.set_session_mode(str(params.session_id), params.mode)
        except ModeRejectedError as exc:
            raise ProtocolError(ProtocolErrorCode.MODE_REJECTED, str(exc)) from exc
        except SessionNotRunningError as exc:
            raise ProtocolError(ProtocolErrorCode.SESSION_NOT_RUNNING, str(exc)) from exc
        except SessionConflictError as exc:
            raise ProtocolError(ProtocolErrorCode.SESSION_CONFLICT, str(exc)) from exc
        except (SessionError, RuntimeDomainError, OSError) as exc:
            raise ProtocolError(
                ProtocolErrorCode.CONTROL_DELIVERY_FAILED, "Unable to restart the harness"
            ) from exc
        except ControlError as exc:
            raise ProtocolError(ProtocolErrorCode.CONTROL_DELIVERY_FAILED, str(exc)) from exc
        if application is ModeApplication.REQUIRES_RESTART:
            raise ProtocolError(
                ProtocolErrorCode.MODE_REQUIRES_RESTART,
                f"mode {params.mode!r} can only apply at the next session start",
            )
        return {
            "session_id": str(params.session_id),
            "mode": params.mode,
            "outcome": application.value,
        }

    return handle


def _make_prompt_handler(
    runtime: RuntimeService,
) -> Callable[[BaseModel], Awaitable[dict[str, typing.Any]]]:
    async def handle(params: BaseModel) -> dict[str, typing.Any]:
        if not isinstance(params, SessionPromptParams):
            raise ProtocolError(
                ProtocolErrorCode.INVALID_PARAMS, "invalid params for action: session.prompt"
            )
        try:
            outcome = (
                await runtime.send_session_prompt(
                    str(params.session_id), params.content, params.attachments
                )
                if params.attachments
                else await runtime.send_session_prompt(str(params.session_id), params.content)
            )
        except AttachmentStorageError as exc:
            raise ProtocolError(ProtocolErrorCode.ATTACHMENT_STORAGE_UNAVAILABLE, str(exc)) from exc
        except AttachmentError as exc:
            raise ProtocolError(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc
        except SteerNoActiveTurnError as exc:
            raise ProtocolError(ProtocolErrorCode.STEER_NO_ACTIVE_TURN, str(exc)) from exc
        except (PromptDeliveryUnknownError, TimeoutError) as exc:
            raise ProtocolError(ProtocolErrorCode.DELIVERY_UNKNOWN, str(exc)) from exc
        except PromptDeliveryFailedError as exc:
            raise ProtocolError(ProtocolErrorCode.PROMPT_DELIVERY_FAILED, str(exc)) from exc
        except (ControlTransportError, HarnessNotInitializedError) as exc:
            raise ProtocolError(ProtocolErrorCode.PROMPT_DELIVERY_FAILED, str(exc)) from exc
        except SteerUnsupportedError as exc:
            raise ProtocolError(ProtocolErrorCode.STEER_UNSUPPORTED, str(exc)) from exc
        except SessionNotRunningError as exc:
            raise ProtocolError(ProtocolErrorCode.SESSION_NOT_RUNNING, str(exc)) from exc
        except SessionConflictError as exc:
            raise ProtocolError(ProtocolErrorCode.SESSION_CONFLICT, str(exc)) from exc
        except SessionError as exc:
            raise ProtocolError(
                ProtocolErrorCode.PROMPT_DELIVERY_FAILED,
                "Unable to prepare the session for delivery",
            ) from exc
        except ControlError as exc:
            raise ProtocolError(ProtocolErrorCode.CONTROL_DELIVERY_FAILED, str(exc)) from exc
        if outcome.state is PromptState.ERROR:
            raise ProtocolError(
                ProtocolErrorCode.PROMPT_DELIVERY_FAILED,
                outcome.code or "prompt delivery failed",
            )
        return {
            "session_id": str(params.session_id),
            "state": outcome.state.value,
            "code": outcome.code,
        }

    return handle


def _make_interrupt_handler(
    runtime: RuntimeService,
) -> Callable[[BaseModel], Awaitable[dict[str, typing.Any]]]:
    async def handle(params: BaseModel) -> dict[str, typing.Any]:
        if not isinstance(params, SessionInterruptParams):
            raise ProtocolError(
                ProtocolErrorCode.INVALID_PARAMS, "invalid params for action: session.interrupt"
            )
        try:
            delivered = await runtime.interrupt_session(str(params.session_id))
        except SteerUnsupportedError as exc:
            raise ProtocolError(ProtocolErrorCode.STEER_UNSUPPORTED, str(exc)) from exc
        except ControlError as exc:
            raise ProtocolError(ProtocolErrorCode.CONTROL_DELIVERY_FAILED, str(exc)) from exc
        return {"session_id": str(params.session_id), "interrupted": delivered}

    return handle


def _protocol_error(error: ApprovalError) -> ProtocolError:
    return ProtocolError(_code(error), str(error))


def _code(error: ApprovalError) -> ProtocolErrorCode:
    match error:
        case ApprovalExpiredError():
            return ProtocolErrorCode.APPROVAL_EXPIRED
        case ApprovalAlreadyAnsweredError() | DuplicateApprovalAnswerError():
            return ProtocolErrorCode.APPROVAL_ALREADY_ANSWERED
        case ApprovalNotPendingError():
            return ProtocolErrorCode.APPROVAL_NOT_PENDING
        case _:
            return ProtocolErrorCode.INVALID_PARAMS
