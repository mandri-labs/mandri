"""REST routes for provider credential management."""

from typing import Any

import httpx
from fastapi import APIRouter, Response
from mandri.api.deps import ChatGpt, ChatGptWiring, Gateway, GatewayWiring, Http, Providers
from mandri.api.errors import ERROR_RESPONSES, NOT_FOUND, NOT_FOUND_CONFLICT, ApiError
from mandri.core.ids import ProviderKind
from mandri.gateway.catalog_enrichment import enrich_entries
from mandri.gateway.model_capabilities import metadata_capabilities
from mandri.gateway.reasoning_catalog import parse_chatgpt_entry, parse_lm_studio_entry
from mandri.providers.catalog import model_entries
from mandri.providers.chatgpt_login import ChatGptLoginSession
from mandri.providers.errors import (
    ProviderExistsError,
    ProviderInUseError,
    ProviderInvalidError,
    ProviderNotFoundError,
    ProviderVerificationError,
)
from mandri.providers.refs import requires_api_base
from mandri.providers.service import Provider, ProviderState
from mandri.providers.verify import models_endpoint, models_headers, models_params
from pydantic import BaseModel

router = APIRouter(prefix="/providers", tags=["providers"])

INVALID: dict[int | str, dict[str, Any]] = {400: ERROR_RESPONSES[400]}
CONFLICT: dict[int | str, dict[str, Any]] = {**INVALID, 409: ERROR_RESPONSES[409]}

_MODELS_TIMEOUT_SECONDS = 10.0
_MAX_ERROR_BODY_CHARS = 500


class ProviderIn(BaseModel):
    name: str
    kind: str
    api_base: str | None = None
    api_key: str = ""
    verify: bool = True


class ProviderUpdateIn(BaseModel):
    api_base: str | None = None
    api_key: str | None = None


class ProviderOut(BaseModel):
    name: str
    kind: str
    api_base: str | None
    state: str
    authorize_url: str | None = None
    login_id: str | None = None


class ModelOut(BaseModel):
    id: str
    reasoning_efforts: list[str]
    default_effort: str | None
    display_name: str | None = None
    image_input: bool | None = None
    tool_call: bool | None = None
    reasoning_supported: bool | None = None
    input_modalities: list[str] | None = None


def _to_out(provider: Provider, session: ChatGptLoginSession | None = None) -> ProviderOut:
    return ProviderOut(
        name=provider.name,
        kind=provider.kind.value,
        api_base=None if provider.api_base is None else str(provider.api_base),
        state=provider.state.value,
        authorize_url=None if session is None else session.authorize_url or None,
        login_id=None if session is None else session.id,
    )


def _not_found(name: str) -> ApiError:
    return ApiError(
        code="provider_not_found",
        message=f"unknown provider {name}",
        status=404,
        detail={"name": name},
    )


def _parse_kind(raw: str) -> ProviderKind:
    try:
        return ProviderKind(raw)
    except ValueError:
        raise ApiError(
            code="provider_invalid",
            message=f"Unknown provider kind {raw!r}",
            status=400,
            detail={"kind": raw},
        ) from None


def _verification_failed(error: ProviderVerificationError) -> ApiError:
    return ApiError(
        code="provider_verification_failed",
        message=str(error),
        status=400,
        detail={"kind": error.kind.value, "reason": error.reason},
    )


def _redact(secret: str, text: str) -> str:
    if secret:
        return text.replace(secret, "[redacted]")
    return text


def _login_session(wiring: ChatGptWiring, provider: Provider) -> ChatGptLoginSession | None:
    if provider.kind is not ProviderKind.CHATGPT:
        return None
    if provider.state is not ProviderState.PENDING_AUTH or wiring.login is None:
        return None
    return wiring.login.pending_for(provider.name)


async def _create_chatgpt(body: ProviderIn, service: Providers, wiring: ChatGpt) -> ProviderOut:
    if body.api_key.strip():
        raise ApiError(
            code="provider_invalid",
            message="ChatGPT providers authenticate through OAuth sign-in, not an api_key",
            status=400,
        )
    if wiring.login is None:
        raise ApiError(
            code="chatgpt_unavailable", message="ChatGPT sign-in is not available", status=503
        )
    name = body.name.strip()
    if not name:
        raise ApiError(
            code="provider_invalid", message="provider name must be a non-empty string", status=400
        )
    try:
        await service.add(name, ProviderKind.CHATGPT, body.api_base, "", verify=False)
    except ProviderExistsError as error:
        raise ApiError(code="provider_exists", message=str(error), status=409) from None
    except ProviderInvalidError as error:
        raise ApiError(code="provider_invalid", message=str(error), status=400) from None
    session = wiring.login.pending_for(name)
    if session is None:
        session = await wiring.login.start(name, body.api_base)
    return _to_out(service.get(name), session)


def _require_signed_in(provider: Provider) -> None:
    if provider.state is not ProviderState.PENDING_AUTH:
        return
    raise ApiError(
        code="chatgpt_auth_pending",
        message=f"provider {provider.name!r} is waiting for ChatGPT sign-in to complete",
        status=409,
        detail={"name": provider.name},
    )


def _require_api_base(provider: Provider) -> None:
    if requires_api_base(provider.kind) and provider.api_base is None:
        raise ApiError(
            code="provider_invalid",
            message=f"api_base is required for provider {provider.kind.value}",
            status=400,
        )


@router.get("", operation_id="list_providers")
async def list_providers(service: Providers, wiring: ChatGpt) -> list[ProviderOut]:
    return [_to_out(provider, _login_session(wiring, provider)) for provider in service.list()]


@router.post("", operation_id="create_provider", status_code=201, responses=CONFLICT)
async def create_provider(body: ProviderIn, service: Providers, wiring: ChatGpt) -> ProviderOut:
    kind = _parse_kind(body.kind)
    if kind is ProviderKind.CHATGPT:
        return await _create_chatgpt(body, service, wiring)
    try:
        provider = await service.add(
            body.name, kind, body.api_base, body.api_key, verify=body.verify
        )
    except ProviderExistsError as error:
        raise ApiError(code="provider_exists", message=str(error), status=409) from None
    except ProviderInvalidError as error:
        raise ApiError(code="provider_invalid", message=str(error), status=400) from None
    except ProviderVerificationError as error:
        raise _verification_failed(error) from None
    return _to_out(provider)


@router.patch("/{name}", operation_id="update_provider", responses=CONFLICT)
async def update_provider(name: str, body: ProviderUpdateIn, service: Providers) -> ProviderOut:
    updates: dict[str, Any] = {}
    if "api_base" in body.model_fields_set:
        updates["api_base"] = body.api_base
    if "api_key" in body.model_fields_set:
        updates["api_key"] = body.api_key
    if not updates:
        raise ApiError(code="provider_invalid", message="no fields to update", status=400)
    try:
        provider = await service.update(name, **updates)
    except ProviderNotFoundError:
        raise _not_found(name) from None
    except ProviderInvalidError as error:
        raise ApiError(code="provider_invalid", message=str(error), status=400) from None
    except ProviderVerificationError as error:
        raise _verification_failed(error) from None
    return _to_out(provider)


@router.delete(
    "/{name}",
    operation_id="delete_provider",
    status_code=204,
    responses=NOT_FOUND_CONFLICT,
)
async def delete_provider(name: str, service: Providers, wiring: ChatGpt) -> Response:
    try:
        await service.remove(name)
    except ProviderNotFoundError:
        raise _not_found(name) from None
    except ProviderInUseError as error:
        raise ApiError(
            code="provider_in_use",
            message=str(error),
            status=409,
            detail={"route_ids": error.route_ids},
        ) from None
    if wiring.login is not None:
        await wiring.login.cancel_provider(name)
    return Response(status_code=204)


@router.post("/{name}/verify", operation_id="verify_provider", responses=NOT_FOUND)
async def verify_provider(name: str, service: Providers, wiring: ChatGpt) -> ProviderOut:
    try:
        provider = await service.verify(name)
    except ProviderNotFoundError:
        raise _not_found(name) from None
    except ProviderInvalidError:
        pending = _pending_session(wiring, name)
        if pending is None:
            raise ApiError(
                code="chatgpt_auth_pending", message="Reconnect this ChatGPT account", status=409
            ) from None
        return _to_out(service.get(name), pending)
    except ProviderVerificationError as error:
        raise _verification_failed(error) from None
    return _to_out(provider, _login_session(wiring, provider))


def _pending_session(wiring: ChatGptWiring, name: str) -> ChatGptLoginSession | None:
    return None if wiring.login is None else wiring.login.pending_for(name)


def _model_out(wiring: GatewayWiring, provider: Provider, entry: dict[str, Any]) -> ModelOut:
    model_id = entry["id"]
    provider_name = provider.name
    catalog = wiring.reasoning_catalog
    info = None if catalog is None else catalog.lookup(provider_name, f"{provider_name}/{model_id}")
    if provider.kind is ProviderKind.LM_STUDIO:
        info = parse_lm_studio_entry(entry) or info
    if provider.kind is ProviderKind.CHATGPT:
        info = parse_chatgpt_entry(entry) or info
    capabilities = metadata_capabilities(entry)
    reasoning = capabilities["reasoning_supported"]
    if reasoning is None and info is not None:
        reasoning = any(effort != "off" for effort in info.efforts)
    return ModelOut(
        id=model_id,
        reasoning_efforts=list(info.efforts) if info else [],
        default_effort=info.default_effort if info else None,
        display_name=entry.get("display_name", entry.get("name")),
        image_input=capabilities["image_input"],
        tool_call=capabilities["tool_call"],
        reasoning_supported=reasoning,
        input_modalities=capabilities["input_modalities"],
    )


@router.get("/{name}/models", operation_id="list_provider_models", responses=NOT_FOUND)
async def list_provider_models(
    name: str, wiring: Gateway, service: Providers, client: Http
) -> list[ModelOut]:
    try:
        await service.ensure_fresh(name)
        provider = service.get(name)
    except ProviderNotFoundError:
        raise _not_found(name) from None
    except ProviderInvalidError as error:
        raise ApiError(code="chatgpt_auth_pending", message=str(error), status=409) from None
    _require_api_base(provider)
    _require_signed_in(provider)
    url = models_endpoint(provider.kind, provider.api_base)
    headers = models_headers(provider.kind, provider.api_key)
    try:
        response = await client.get(
            url,
            headers=headers,
            params=models_params(provider.kind),
            timeout=_MODELS_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError:
        raise ApiError(
            code="provider_models_failed",
            message="provider unreachable",
            status=502,
        ) from None
    if not 200 <= response.status_code < 300:
        body = _redact(provider.api_key, response.text)[:_MAX_ERROR_BODY_CHARS]
        raise ApiError(
            code="provider_models_failed",
            message=f"provider models request failed (status {response.status_code}): {body}",
            status=502,
        )
    try:
        entries = model_entries(provider.kind, response.json())
    except ValueError:
        raise ApiError(
            code="provider_models_failed",
            message="provider returned an invalid model catalog",
            status=502,
        ) from None
    entries = await enrich_entries(provider.kind, entries, client)
    return [_model_out(wiring, provider, entry) for entry in entries]
