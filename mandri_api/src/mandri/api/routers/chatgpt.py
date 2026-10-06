"""REST routes for the ChatGPT sign-in flow behind a ChatGPT provider."""

from typing import Any

from fastapi import APIRouter
from mandri.api.deps import ChatGpt, ChatGptWiring, Providers
from mandri.api.errors import ERROR_RESPONSES, ApiError
from mandri.providers.chatgpt_login import ChatGptLoginSession
from mandri.providers.errors import ProviderExistsError, ProviderNotFoundError
from pydantic import BaseModel

router = APIRouter(prefix="/providers/chatgpt", tags=["providers"])

INVALID: dict[int | str, dict[str, Any]] = {400: ERROR_RESPONSES[400]}
NOT_FOUND_ONLY: dict[int | str, dict[str, Any]] = {404: ERROR_RESPONSES[404]}
LOGIN_RESPONSES: dict[int | str, dict[str, Any]] = {
    **INVALID,
    404: ERROR_RESPONSES[404],
    409: ERROR_RESPONSES[409],
    503: {"model": ERROR_RESPONSES[400]["model"], "description": "Sign-in is unavailable"},
}


class LoginStartIn(BaseModel):
    name: str | None = None
    api_base: str | None = None


class LoginCallbackIn(BaseModel):
    redirect_url: str


class LoginOut(BaseModel):
    login_id: str
    provider_name: str
    status: str
    authorize_url: str | None = None
    error: str | None = None
    error_code: str | None = None


def _to_out(session: ChatGptLoginSession) -> LoginOut:
    return LoginOut(
        login_id=session.id,
        provider_name=session.provider_name,
        status=session.status.value,
        authorize_url=session.authorize_url or None,
        error=session.error,
        error_code=session.error_code,
    )


def _error(code: str, message: str, status: int, **detail: Any) -> ApiError:
    return ApiError(code=code, message=message, status=status, detail=detail or None)


def _missing(login_id: str) -> ApiError:
    return _error(
        "chatgpt_login_not_found", f"unknown ChatGPT login {login_id}", 404, login_id=login_id
    )


def _login_service(wiring: ChatGptWiring) -> Any:
    if wiring.login is None:
        raise _error("chatgpt_unavailable", "ChatGPT sign-in is not available", 503)
    return wiring.login


def _conflicts(providers: Providers, name: str) -> bool:
    """An existing ChatGPT provider may be reconnected; anything else collides."""
    try:
        provider = providers.get(name)
    except ProviderNotFoundError:
        return False
    return provider.kind.value != "chatgpt"


@router.post(
    "/login",
    operation_id="start_chatgpt_login",
    status_code=201,
    responses=LOGIN_RESPONSES,
)
async def start_chatgpt_login(
    body: LoginStartIn, wiring: ChatGpt, providers: Providers
) -> LoginOut:
    name = (body.name or "").strip()
    if name and _conflicts(providers, name):
        raise _error("provider_exists", f"provider {name!r} already exists", 409, name=name)
    service = _login_service(wiring)
    session = service.pending_for(name)
    if session is None:
        try:
            session = await service.start(body.name, body.api_base)
        except ProviderExistsError as error:
            raise _error("provider_exists", str(error), 409) from None
    return _to_out(session)


@router.get("/login/{login_id}", operation_id="get_chatgpt_login", responses=NOT_FOUND_ONLY)
async def get_chatgpt_login(login_id: str, wiring: ChatGpt) -> LoginOut:
    session = _login_service(wiring).get(login_id)
    if session is None:
        raise _missing(login_id)
    return _to_out(session)


@router.post(
    "/login/{login_id}/callback",
    operation_id="complete_chatgpt_login",
    responses=LOGIN_RESPONSES,
)
async def complete_chatgpt_login(login_id: str, body: LoginCallbackIn, wiring: ChatGpt) -> LoginOut:
    service = _login_service(wiring)
    if service.get(login_id) is None:
        raise _missing(login_id)
    session = await service.submit_redirect(login_id, body.redirect_url)
    if session is None:
        raise _missing(login_id)
    return _to_out(session)


@router.delete("/login/{login_id}", operation_id="cancel_chatgpt_login", responses=NOT_FOUND_ONLY)
async def cancel_chatgpt_login(login_id: str, wiring: ChatGpt) -> LoginOut:
    service = _login_service(wiring)
    if service.get(login_id) is None:
        raise _missing(login_id)
    session = await service.cancel(login_id)
    if session is None:
        raise _missing(login_id)
    return _to_out(session)
