"""REST routes for provider credential management."""

from typing import Any

import httpx
from fastapi import APIRouter, Response
from mandri.api.deps import Gateway, GatewayWiring, Http, Providers
from mandri.api.errors import ERROR_RESPONSES, NOT_FOUND, NOT_FOUND_CONFLICT, ApiError
from mandri.core.ids import ProviderKind
from mandri.gateway.catalog_enrichment import enrich_entries
from mandri.gateway.model_capabilities import metadata_capabilities
from mandri.gateway.reasoning_catalog import parse_lm_studio_entry
from mandri.providers.catalog import model_entries
from mandri.providers.errors import (
    ProviderExistsError,
    ProviderInUseError,
    ProviderInvalidError,
    ProviderNotFoundError,
    ProviderVerificationError,
)
from mandri.providers.refs import requires_api_base
from mandri.providers.service import Provider
from mandri.providers.verify import models_endpoint, models_headers
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
    api_key: str
    verify: bool = True


class ProviderUpdateIn(BaseModel):
    api_base: str | None = None
    api_key: str | None = None


class ProviderOut(BaseModel):
    name: str
    kind: str
    api_base: str | None
    state: str


class ModelOut(BaseModel):
    id: str
    reasoning_efforts: list[str]
    default_effort: str | None
    display_name: str | None = None
    image_input: bool | None = None
    tool_call: bool | None = None
    reasoning_supported: bool | None = None
    input_modalities: list[str] | None = None


def _to_out(provider: Provider) -> ProviderOut:
    return ProviderOut(
        name=provider.name,
        kind=provider.kind.value,
        api_base=None if provider.api_base is None else str(provider.api_base),
        state=provider.state.value,
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


def _require_api_base(provider: Provider) -> None:
    if requires_api_base(provider.kind) and provider.api_base is None:
        raise ApiError(
            code="provider_invalid",
            message=f"api_base is required for provider {provider.kind.value}",
            status=400,
        )


@router.get("", operation_id="list_providers")
async def list_providers(service: Providers) -> list[ProviderOut]:
    return [_to_out(provider) for provider in service.list()]


@router.post("", operation_id="create_provider", status_code=201, responses=CONFLICT)
async def create_provider(body: ProviderIn, service: Providers) -> ProviderOut:
    kind = _parse_kind(body.kind)
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
async def delete_provider(name: str, service: Providers) -> Response:
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
    return Response(status_code=204)


@router.post("/{name}/verify", operation_id="verify_provider", responses=NOT_FOUND)
async def verify_provider(name: str, service: Providers) -> ProviderOut:
    try:
        provider = await service.verify(name)
    except ProviderNotFoundError:
        raise _not_found(name) from None
    except ProviderVerificationError as error:
        raise _verification_failed(error) from None
    return _to_out(provider)


def _model_out(wiring: GatewayWiring, provider: Provider, entry: dict[str, Any]) -> ModelOut:
    model_id = entry["id"]
    provider_name = provider.name
    catalog = wiring.reasoning_catalog
    info = None if catalog is None else catalog.lookup(provider_name, f"{provider_name}/{model_id}")
    if provider.kind is ProviderKind.LM_STUDIO:
        info = parse_lm_studio_entry(entry) or info
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
        provider = service.get(name)
    except ProviderNotFoundError:
        raise _not_found(name) from None
    _require_api_base(provider)
    url = models_endpoint(provider.kind, provider.api_base)
    headers = models_headers(provider.kind, provider.api_key)
    try:
        response = await client.get(url, headers=headers, timeout=_MODELS_TIMEOUT_SECONDS)
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
