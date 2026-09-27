"""REST routes for gateway route management and harness-facing LLM endpoints."""

import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from mandri.api.deps import Gateway, GatewayWiring, Providers
from mandri.api.errors import NOT_FOUND, ApiError
from mandri.api.routers._effort import lookup_efforts, normalize_effort
from mandri.api.routers._gateway_privacy import (
    model_body,
    privacy_error,
    restore_response,
    upstream_response,
)
from mandri.core.ids import RouteId, WireFormat
from mandri.core.protocol.types import RouteId as ApiRouteId
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.gateway.auth import child_token_from_headers
from mandri.gateway.dropped_params import dropped_params
from mandri.gateway.errors.upstream import RouteNotFoundError, UpstreamError
from mandri.gateway.listings import anthropic_listing, codex_listing, openai_listing
from mandri.gateway.litellm_adapter import (
    upstream_error,
)
from mandri.gateway.model_metadata import fetch as fetch_model_metadata
from mandri.gateway.privacy import prepare_request
from mandri.gateway.privacy_protocol import GatewayProtocol
from mandri.gateway.privacy_transport import install_transport_observers
from mandri.gateway.ratelimit_headers import from_success
from mandri.gateway.reasoning_catalog import ReasoningInfo
from mandri.gateway.response_stream import normalize_stream, sse_frame
from mandri.gateway.route_registry import ResolvedRoute, Route
from mandri.gateway.stream_coalescing import StreamCoalescer
from mandri.gateway.types.model import Model
from mandri.gateway.usage import UsageCollector, UsageStream, collect_call
from mandri.gateway.usage_attribution import UsageAttribution
from mandri.gateway.usage_context import request_modality
from mandri.providers.errors import ProviderInvalidError, ProviderNotFoundError
from mandri.providers.service import Provider, parse_model_arg, provider_model_ref, split_model_ref
from pydantic import BaseModel
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gateway", tags=["gateway"])

_SSE_MEDIA_TYPE = "text/event-stream"
_LOCAL_TOKENIZER = "local_tokenizer"
_UNAUTHORIZED_MESSAGE = "missing or invalid gateway child token"
_EMPTY_STREAM_MESSAGE = "upstream produced no chunks"

_OPENAI_ERROR_TYPES: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "invalid_request_error",
    429: "rate_limit_error",
}
_ANTHROPIC_ERROR_TYPES: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    529: "overloaded_error",
}
_GEMINI_ERROR_STATUSES: dict[int, str] = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    429: "RESOURCE_EXHAUSTED",
    503: "UNAVAILABLE",
}


class RouteCreateIn(BaseModel):
    model: str
    formats: list[str]
    effort: str | None = None


class RouteModelIn(BaseModel):
    model: str


class ReasoningOut(BaseModel):
    efforts: list[str]
    default_effort: str | None


class RouteOut(BaseModel):
    id: str
    provider: str
    model_ref: str
    formats: list[str]
    created_at: int


class RouteCreatedOut(RouteOut):
    child_token: str


class ProvidersInfoOut(BaseModel):
    name: str
    kind: str
    api_base: str | None
    state: str


class GatewayInfoOut(BaseModel):
    providers: list[ProvidersInfoOut]
    routes: list[RouteOut]


class CountTokensOut(BaseModel):
    input_tokens: int
    tokenizer_type: str
    estimated: bool


def _to_out(route: Route) -> RouteOut:
    return RouteOut(
        id=str(route.id),
        provider=route.provider_name,
        model_ref=str(route.model_ref),
        formats=[fmt.value for fmt in route.formats],
        created_at=int(route.created_at),
    )


def _provider_info(provider: Provider) -> ProvidersInfoOut:
    return ProvidersInfoOut(
        name=provider.name,
        kind=provider.kind.value,
        api_base=None if provider.api_base is None else str(provider.api_base),
        state=provider.state.value,
    )


@router.get("/info", operation_id="gateway_info")
async def gateway_info(wiring: Gateway, providers: Providers) -> GatewayInfoOut:
    return GatewayInfoOut(
        providers=[_provider_info(provider) for provider in providers.list()],
        routes=[_to_out(route) for route in await wiring.registry.list_routes()],
    )


def _parse_formats(raw: list[str]) -> tuple[WireFormat, ...]:
    try:
        return tuple(WireFormat(item) for item in raw)
    except ValueError:
        raise ApiError(code="route_invalid", message="unknown wire format", status=400) from None


def _error_type(status: int, table: dict[int, str]) -> str:
    mapped = table.get(status)
    if mapped is not None:
        return mapped
    return "api_error" if status < 500 else "server_error"


def _openai_error(status: int, message: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": _error_type(status, _OPENAI_ERROR_TYPES)}}


def _anthropic_error(status: int, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {"type": _error_type(status, _ANTHROPIC_ERROR_TYPES), "message": message},
    }


def _gemini_error(status: int, message: str) -> dict[str, Any]:
    return {
        "error": {
            "code": status,
            "message": message,
            "status": _GEMINI_ERROR_STATUSES.get(status, "INTERNAL"),
        }
    }


def _as_json(result: Any) -> Any:
    dump = getattr(result, "model_dump_json", None)
    if dump is not None:
        return json.loads(dump())
    return result


def _with_dropped_params(content: Any, dropped: list[str]) -> Any:
    if not dropped or not isinstance(content, dict):
        return content
    return {**content, "mandri": {"dropped_params": dropped}}


def _report_dropped_params(route_id: str, dropped: list[str]) -> None:
    if dropped:
        logger.warning("gateway route %s dropped params: %s", route_id, dropped)


def _sse_bytes(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _authorize(
    wiring: GatewayWiring,
    route_id: str,
    request: Request,
    error_builder: Callable[[int, str], dict[str, Any]],
) -> Response | None:
    token = child_token_from_headers(request.headers)
    if wiring.auth.verify(route_id, token):
        return None
    return JSONResponse(status_code=401, content=error_builder(401, _UNAUTHORIZED_MESSAGE))


async def _peek_stream(
    stream: AsyncIterator[Any],
    model: Model,
    error_builder: Callable[[int, str], dict[str, Any]],
) -> tuple[dict[str, str], Any | None, Response | None]:
    try:
        first = await stream.__anext__()
    except StopAsyncIteration:
        return {}, None, None
    except UpstreamError as error:
        return (
            {},
            None,
            JSONResponse(
                status_code=error.status,
                content=error_builder(error.status, error.message),
                headers=error.headers,
            ),
        )
    except Exception as error:
        upstream = upstream_error(error, model)
        return (
            {},
            None,
            JSONResponse(
                status_code=upstream.status,
                content=error_builder(upstream.status, upstream.message),
                headers=upstream.headers,
            ),
        )
    return from_success(first), first, None


async def _openai_stream(
    first: Any, rest: AsyncIterator[Any], model: Model
) -> AsyncIterator[bytes]:
    coalescer = StreamCoalescer()
    try:
        for released in coalescer.add(first):
            yield _sse_bytes(json.loads(released.model_dump_json()))
        async for chunk in rest:
            for released in coalescer.add(chunk):
                yield _sse_bytes(json.loads(released.model_dump_json()))
        for tail in coalescer.flush():
            yield _sse_bytes(json.loads(tail.model_dump_json()))
    except Exception as error:
        upstream = _upstream_of(error, model)
        for released in coalescer.flush():
            yield _sse_bytes(json.loads(released.model_dump_json()))
        yield _sse_bytes(_openai_error(upstream.status, upstream.message))
        yield b"data: [DONE]\n\n"
        return
    yield b"data: [DONE]\n\n"


def _upstream_of(error: Exception, model: Model) -> UpstreamError:
    if isinstance(error, UpstreamError):
        return error
    return upstream_error(error, model)


async def _anthropic_stream(
    first: Any, rest: AsyncIterator[Any], model: Model
) -> AsyncIterator[bytes]:
    try:
        yield first if isinstance(first, bytes) else str(first).encode()
        async for frame in rest:
            yield frame if isinstance(frame, bytes) else str(frame).encode()
    except Exception as error:
        upstream = _upstream_of(error, model)
        yield _anthropic_error_frame(upstream.status, upstream.message)


def _anthropic_error_frame(status: int, message: str) -> bytes:
    payload = _anthropic_error(status, message)
    return f"event: error\ndata: {json.dumps(payload)}\n\n".encode()


def _route_invalid(error: ProviderInvalidError) -> ApiError:
    return ApiError(code="route_invalid", message=str(error), status=400)


@router.post("/routes", operation_id="create_route", status_code=201, responses=NOT_FOUND)
async def create_route(body: RouteCreateIn, wiring: Gateway) -> RouteCreatedOut:
    try:
        provider_name, model_id = parse_model_arg(body.model)
    except ProviderInvalidError as error:
        raise _route_invalid(error) from None
    try:
        route = await wiring.registry.create(
            provider_name,
            model_id,
            _parse_formats(body.formats),
            reasoning_effort=normalize_effort(body.effort),
        )
    except ProviderNotFoundError as error:
        raise ApiError(code="provider_not_found", message=str(error), status=404) from None
    except ProviderInvalidError as error:
        raise _route_invalid(error) from None
    return RouteCreatedOut(
        **_to_out(route).model_dump(), child_token=wiring.issue_child_token(str(route.id))
    )


@router.get("/routes", operation_id="list_routes")
async def list_routes(wiring: Gateway) -> list[RouteOut]:
    return [_to_out(route) for route in await wiring.registry.list_routes()]


@router.patch("/routes/{route_id}/model", operation_id="set_route_model", responses=NOT_FOUND)
async def set_route_model(route_id: ApiRouteId, body: RouteModelIn, wiring: Gateway) -> RouteOut:
    try:
        provider_name, model_id = parse_model_arg(body.model)
    except ProviderInvalidError as error:
        raise _route_invalid(error) from None
    try:
        route = await wiring.registry.swap(RouteId(route_id), provider_name, model_id)
    except RouteNotFoundError as error:
        raise ApiError(code="route_not_found", message=str(error), status=404) from None
    except ProviderNotFoundError as error:
        raise ApiError(code="provider_not_found", message=str(error), status=404) from None
    except ProviderInvalidError as error:
        raise _route_invalid(error) from None
    return _to_out(route)


@router.delete(
    "/routes/{route_id}",
    operation_id="delete_route",
    status_code=204,
    responses=NOT_FOUND,
)
async def delete_route(route_id: ApiRouteId, wiring: Gateway) -> Response:
    try:
        await wiring.registry.delete(RouteId(route_id))
    except RouteNotFoundError as error:
        raise ApiError(code="route_not_found", message=str(error), status=404) from None
    return Response(status_code=204)


def _reasoning_payload(info: ReasoningInfo | None) -> dict[str, Any] | None:
    if info is None:
        return None
    return ReasoningOut(efforts=info.efforts, default_effort=info.default_effort).model_dump()


async def _serve_model_metadata(
    wiring: GatewayWiring, providers: Providers, model_arg: str
) -> Response:
    try:
        provider_name, model_id = parse_model_arg(model_arg)
    except ProviderInvalidError as error:
        raise _route_invalid(error) from None
    try:
        provider = providers.get(provider_name)
    except ProviderNotFoundError as error:
        raise ApiError(code="provider_not_found", message=str(error), status=404) from None
    metadata = await fetch_model_metadata(
        provider.kind,
        provider_model_ref(provider.kind, model_id),
        provider.api_base,
        str(provider.api_key),
    )
    if metadata is None:
        raise ApiError(
            code="model_metadata_unavailable",
            message=f"no metadata for {model_arg}",
            status=404,
        )
    payload = metadata.to_payload()
    payload["reasoning"] = _reasoning_payload(lookup_efforts(wiring, model_arg))
    return JSONResponse(content=payload)


@router.get("/model-metadata", operation_id="gateway_model_metadata", responses=NOT_FOUND)
async def gateway_model_metadata(
    wiring: Gateway, providers: Providers, model: Annotated[str, Query(min_length=3)]
) -> Response:
    return await _serve_model_metadata(wiring, providers, model)


def _dropped(resolved: ResolvedRoute, body: dict[str, Any]) -> list[str]:
    return dropped_params(str(resolved.model.model_ref), body)


async def _collect_usage(
    wiring: GatewayWiring,
    request: Request,
    resolved: ResolvedRoute,
    protocol: GatewayProtocol,
    call: Callable[[], Awaitable[Any]],
    body: dict[str, Any] | None = None,
) -> Any:
    sink = getattr(wiring, "usage_sink", None)
    if sink is None:
        sink = getattr(request.app.state, "gateway_usage_sink", None)
    if sink is None:
        return await call()
    install_transport_observers()
    attribution = UsageAttribution()
    lookup = getattr(getattr(wiring, "registry", None), "usage_attribution", None)
    if lookup is not None:
        try:
            attribution = await lookup(resolved.route_id)
        except Exception:
            logger.warning("Gateway usage attribution unavailable for route %s", resolved.route_id)
    body = body or {}
    tier = body.get("service_tier")
    collector = UsageCollector(
        resolved,
        protocol.value,
        sink,
        attribution,
        modality=request_modality(protocol.value, body),
        service_tier=(
            tier
            if isinstance(tier, str) and tier in {"auto", "default", "flex", "priority", "scale"}
            else None
        ),
    )
    return await collect_call(collector, call)


class _UsageStreamingResponse(StreamingResponse):
    def __init__(self, content: AsyncIterator[Any], source: AsyncIterator[Any], **kwargs: Any):
        super().__init__(content, **kwargs)
        self.usage_source = source

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if isinstance(self.usage_source, UsageStream):
                await self.usage_source.aclose()


async def _serve_openai(
    wiring: GatewayWiring, route_id: str, body: dict[str, Any], request: Request
) -> Response:
    unauthorized = _authorize(wiring, route_id, request, _openai_error)
    if unauthorized is not None:
        return unauthorized
    try:
        resolved = await wiring.registry.resolve(RouteId(route_id))
        prepared = await prepare_request(wiring.privacy, resolved, GatewayProtocol.CHAT, body)
        result = await _collect_usage(
            wiring,
            request,
            resolved,
            GatewayProtocol.CHAT,
            lambda: wiring.openai.chat_completions(
                resolved, prepared.body, **({"guard": prepared.guard} if prepared.guard else {})
            ),
            prepared.body,
        )
    except ProtectionError as error:
        return privacy_error(error, _openai_error)
    except (RouteNotFoundError, ProviderNotFoundError) as error:
        return JSONResponse(status_code=404, content=_openai_error(404, str(error)))
    except UpstreamError as error:
        return upstream_response(error, prepared, GatewayProtocol.CHAT, _openai_error)
    dropped = _dropped(resolved, body)
    if isinstance(result, AsyncIterator):
        headers, first, empty = await _peek_stream(result, resolved.model, _openai_error)
        if empty is not None:
            return restore_response(empty, prepared, GatewayProtocol.CHAT)
        if first is None:
            return JSONResponse(status_code=502, content=_openai_error(502, _EMPTY_STREAM_MESSAGE))
        _report_dropped_params(route_id, dropped)
        return restore_response(
            _UsageStreamingResponse(
                _openai_stream(first, result, resolved.model),
                result,
                media_type=_SSE_MEDIA_TYPE,
                headers=headers,
            ),
            prepared,
            GatewayProtocol.CHAT,
        )
    return restore_response(
        JSONResponse(
            content=_with_dropped_params(_as_json(result), dropped), headers=from_success(result)
        ),
        prepared,
        GatewayProtocol.CHAT,
    )


async def _serve_responses(
    wiring: GatewayWiring, route_id: str, body: dict[str, Any], request: Request
) -> Response:
    unauthorized = _authorize(wiring, route_id, request, _openai_error)
    if unauthorized is not None:
        return unauthorized
    try:
        resolved = await wiring.registry.resolve(RouteId(route_id))
        prepared = await prepare_request(wiring.privacy, resolved, GatewayProtocol.RESPONSES, body)
        result = await _collect_usage(
            wiring,
            request,
            resolved,
            GatewayProtocol.RESPONSES,
            lambda: wiring.responses.responses(
                resolved, prepared.body, **({"guard": prepared.guard} if prepared.guard else {})
            ),
            prepared.body,
        )
    except ProtectionError as error:
        return privacy_error(
            error,
            _openai_error,
            responses_stream=bool(body.get("stream"))
            and resolved.privacy_mode is PrivacyMode.SURROGATE,
        )
    except (RouteNotFoundError, ProviderNotFoundError) as error:
        return JSONResponse(status_code=404, content=_openai_error(404, str(error)))
    except UpstreamError as error:
        return upstream_response(error, prepared, GatewayProtocol.RESPONSES, _openai_error)
    if isinstance(result, AsyncIterator):
        return restore_response(
            _UsageStreamingResponse(
                _responses_stream_all(result, resolved.model), result, media_type=_SSE_MEDIA_TYPE
            ),
            prepared,
            GatewayProtocol.RESPONSES,
        )
    return restore_response(
        JSONResponse(content=_as_json(result)), prepared, GatewayProtocol.RESPONSES
    )


async def _responses_stream_all(stream: AsyncIterator[Any], model: Model) -> AsyncIterator[bytes]:
    response: dict[str, Any] = {}
    sequence = 0
    try:
        async for event in normalize_stream(stream):
            sequence = event.get("sequence_number", sequence) + 1
            if isinstance(event.get("response"), dict):
                response = event["response"]
            yield sse_frame(event)
    except Exception as error:
        upstream = _upstream_of(error, model)
        yield sse_frame(
            {
                "type": "response.failed",
                "sequence_number": sequence,
                "response": {
                    **response,
                    "id": response.get("id") or f"resp_{uuid.uuid4().hex}",
                    "object": "response",
                    "status": "failed",
                    "output": response.get("output", []),
                    "error": {
                        "code": _error_type(upstream.status, _OPENAI_ERROR_TYPES),
                        "message": upstream.message,
                    },
                },
            }
        )


async def _serve_anthropic(
    wiring: GatewayWiring, route_id: str, body: dict[str, Any], request: Request
) -> Response:
    unauthorized = _authorize(wiring, route_id, request, _anthropic_error)
    if unauthorized is not None:
        return unauthorized
    try:
        resolved = await wiring.registry.resolve(RouteId(route_id))
        prepared = await prepare_request(wiring.privacy, resolved, GatewayProtocol.ANTHROPIC, body)
        result = await _collect_usage(
            wiring,
            request,
            resolved,
            GatewayProtocol.ANTHROPIC,
            lambda: wiring.anthropic.messages(
                resolved, prepared.body, **({"guard": prepared.guard} if prepared.guard else {})
            ),
            prepared.body,
        )
    except ProtectionError as error:
        return privacy_error(error, _anthropic_error)
    except (RouteNotFoundError, ProviderNotFoundError) as error:
        return JSONResponse(status_code=404, content=_anthropic_error(404, str(error)))
    except UpstreamError as error:
        return upstream_response(error, prepared, GatewayProtocol.ANTHROPIC, _anthropic_error)
    dropped = _dropped(resolved, body)
    if isinstance(result, AsyncIterator):
        headers, first, empty = await _peek_stream(result, resolved.model, _anthropic_error)
        if empty is not None:
            return restore_response(empty, prepared, GatewayProtocol.ANTHROPIC)
        if first is None:
            return JSONResponse(
                status_code=502, content=_anthropic_error(502, _EMPTY_STREAM_MESSAGE)
            )
        _report_dropped_params(route_id, dropped)
        return restore_response(
            _UsageStreamingResponse(
                _anthropic_stream(first, result, resolved.model),
                result,
                media_type=_SSE_MEDIA_TYPE,
                headers=headers,
            ),
            prepared,
            GatewayProtocol.ANTHROPIC,
        )
    return restore_response(
        JSONResponse(
            content=_with_dropped_params(_as_json(result), dropped), headers=from_success(result)
        ),
        prepared,
        GatewayProtocol.ANTHROPIC,
    )


async def _serve_gemini(
    wiring: GatewayWiring, route_id: str, body: dict[str, Any], stream: bool, request: Request
) -> Response:
    unauthorized = _authorize(wiring, route_id, request, _gemini_error)
    if unauthorized is not None:
        return unauthorized
    if wiring.gemini is None:
        raise ApiError(
            code="service_unavailable", message="Gemini handler is not available", status=503
        )
    try:
        resolved = await wiring.registry.resolve(RouteId(route_id))
        prepared = await prepare_request(wiring.privacy, resolved, GatewayProtocol.GEMINI, body)
        gemini = wiring.gemini
        result = await _collect_usage(
            wiring,
            request,
            resolved,
            GatewayProtocol.GEMINI,
            lambda: gemini.generate_content(
                resolved,
                prepared.body,
                stream and prepared.guard is None,
                **({"guard": prepared.guard} if prepared.guard else {}),
            ),
            prepared.body,
        )
    except ProtectionError as error:
        return privacy_error(error, _gemini_error)
    except (RouteNotFoundError, ProviderNotFoundError) as error:
        return JSONResponse(status_code=404, content=_gemini_error(404, str(error)))
    except UpstreamError as error:
        return upstream_response(error, prepared, GatewayProtocol.GEMINI, _gemini_error)
    if isinstance(result, AsyncIterator):
        return restore_response(
            _UsageStreamingResponse(result, result, media_type=_SSE_MEDIA_TYPE),
            prepared,
            GatewayProtocol.GEMINI,
        )
    if prepared.guard is not None and stream:
        prepared = replace(prepared, client_stream=True)
    return restore_response(
        JSONResponse(content=_as_json(result)), prepared, GatewayProtocol.GEMINI
    )


@router.post("/llm/{route_id}/v1/chat/completions", operation_id="openai_chat")
async def openai_chat(route_id: ApiRouteId, request: Request, wiring: Gateway) -> Response:
    return await _serve_openai(
        wiring,
        route_id,
        await model_body(request),
        request,
    )


@router.post("/llm/{route_id}/responses", operation_id="openai_responses")
async def openai_responses(route_id: ApiRouteId, request: Request, wiring: Gateway) -> Response:
    return await _serve_responses(
        wiring,
        route_id,
        await model_body(request),
        request,
    )


@router.post("/llm/{route_id}/v1/responses", operation_id="openai_responses_v1")
async def openai_responses_v1(route_id: ApiRouteId, request: Request, wiring: Gateway) -> Response:
    return await _serve_responses(
        wiring,
        route_id,
        await model_body(request),
        request,
    )


@router.post("/llm/{route_id}/v1/messages", operation_id="anthropic_messages")
async def anthropic_messages(route_id: ApiRouteId, request: Request, wiring: Gateway) -> Response:
    return await _serve_anthropic(
        wiring,
        route_id,
        await model_body(request),
        request,
    )


@router.post("/llm/{route_id}/v1/messages/count_tokens", operation_id="anthropic_count_tokens")
async def anthropic_count_tokens(
    route_id: ApiRouteId, request: Request, wiring: Gateway
) -> Response:
    unauthorized = _authorize(wiring, route_id, request, _anthropic_error)
    if unauthorized is not None:
        return unauthorized
    body = await model_body(request)
    try:
        resolved = await wiring.registry.resolve(RouteId(route_id))
        prepared = await prepare_request(
            wiring.privacy, resolved, GatewayProtocol.COUNT_TOKENS, body
        )
        result = await wiring.anthropic.count_tokens(
            resolved, prepared.body, **({"guard": prepared.guard} if prepared.guard else {})
        )
    except ProtectionError as error:
        return privacy_error(error, _anthropic_error)
    except (RouteNotFoundError, ProviderNotFoundError) as error:
        return JSONResponse(status_code=404, content=_anthropic_error(404, str(error)))
    except UpstreamError as error:
        return upstream_response(error, prepared, GatewayProtocol.COUNT_TOKENS, _anthropic_error)
    tokenizer_type = str(result.tokenizer_type)
    return JSONResponse(
        content=CountTokensOut(
            input_tokens=int(result.total_tokens),
            tokenizer_type=tokenizer_type,
            estimated=tokenizer_type == _LOCAL_TOKENIZER,
        ).model_dump()
    )


def _governed_reasoning(
    wiring: GatewayWiring, provider_name: str, model_ref: str
) -> ReasoningInfo | None:
    if wiring.reasoning_catalog is None:
        return None
    return wiring.reasoning_catalog.lookup(provider_name, model_ref)


async def _serve_models_listing(
    wiring: GatewayWiring,
    route_id: str,
    request: Request,
    error_builder: Callable[[int, str], dict[str, Any]],
    builder: Callable[[str], dict[str, Any]],
    builder_override: Callable[[str, ReasoningInfo | None], dict[str, Any]] | None = None,
) -> Response:
    unauthorized = _authorize(wiring, route_id, request, error_builder)
    if unauthorized is not None:
        return unauthorized
    try:
        resolved = await wiring.registry.resolve(RouteId(route_id))
    except (RouteNotFoundError, ProviderNotFoundError) as error:
        return JSONResponse(status_code=404, content=error_builder(404, str(error)))
    bare = split_model_ref(resolved.provider.kind, str(resolved.model.model_ref))
    model_id = f"{resolved.provider.name}/{bare}"
    if builder_override is not None and request.query_params.get("client_version"):
        reasoning = _governed_reasoning(
            wiring, resolved.provider.name, str(resolved.model.model_ref)
        )
        return JSONResponse(content=builder_override(model_id, reasoning))
    return JSONResponse(content=builder(model_id))


@router.get("/llm/{route_id}/v1/models", operation_id="anthropic_list_models")
async def anthropic_list_models(
    route_id: ApiRouteId, request: Request, wiring: Gateway
) -> Response:
    return await _serve_models_listing(
        wiring, route_id, request, _anthropic_error, anthropic_listing, codex_listing
    )


@router.get("/llm/{route_id}/models", operation_id="openai_list_models")
async def openai_list_models(route_id: ApiRouteId, request: Request, wiring: Gateway) -> Response:
    return await _serve_models_listing(wiring, route_id, request, _openai_error, openai_listing)


@router.post(
    "/llm/{route_id}/v1beta/models/{model_name}:generateContent", operation_id="gemini_generate"
)
async def gemini_generate(
    route_id: ApiRouteId, model_name: str, request: Request, wiring: Gateway
) -> Response:
    return await _serve_gemini(
        wiring,
        route_id,
        await model_body(request),
        stream=False,
        request=request,
    )


@router.post(
    "/llm/{route_id}/v1beta/models/{model_name}:streamGenerateContent",
    operation_id="gemini_stream",
)
async def gemini_stream(
    route_id: ApiRouteId, model_name: str, request: Request, wiring: Gateway
) -> Response:
    return await _serve_gemini(
        wiring,
        route_id,
        await model_body(request),
        stream=True,
        request=request,
    )


@router.post("/llm/{route_id}/{operation:path}", include_in_schema=False)
async def unsupported_model_operation(
    route_id: ApiRouteId, operation: str, request: Request, wiring: Gateway
) -> Response:
    unauthorized = _authorize(wiring, route_id, request, _openai_error)
    if unauthorized is not None:
        return unauthorized
    try:
        await wiring.registry.resolve(RouteId(route_id))
    except (RouteNotFoundError, ProviderNotFoundError):
        raise ApiError("not_found", "Not Found", 404) from None
    raise ApiError("not_found", "Not Found", 404)
