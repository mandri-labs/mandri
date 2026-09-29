import json
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from mandri.core.types.execution import ProtectionError
from mandri.gateway.complete_response import complete_sse
from mandri.gateway.errors.upstream import UpstreamError
from mandri.gateway.privacy import PreparedRequest
from mandri.gateway.privacy_protocol import GatewayProtocol
from mandri.gateway.privacy_response import restore_json


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


async def model_body(request: Request) -> dict[str, Any]:
    buffer = bytearray()
    async for part in request.stream():
        buffer.extend(part)
    try:
        body = json.loads(buffer, object_pairs_hook=_object)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ProtectionError("gateway_request_invalid", "Invalid model request JSON") from None
    if not isinstance(body, dict):
        raise ProtectionError("gateway_request_invalid", "The model request must be an object")
    return body


def restore_response(
    response: Response, prepared: PreparedRequest, protocol: GatewayProtocol
) -> Response:
    if prepared.guard is None:
        if (
            prepared.client_stream
            and not isinstance(response, StreamingResponse)
            and response.status_code < 400
        ):
            return Response(
                content=complete_sse(json.loads(bytes(response.body)), protocol),
                status_code=response.status_code,
                media_type="text/event-stream",
            )
        return response
    engine = prepared.guard.engine
    headers = {name: value for name, value in response.headers.items() if name != "content-length"}
    try:
        if isinstance(response, StreamingResponse):
            raise ProtectionError(
                "privacy_response_invalid", "Protected provider responses must be complete"
            )
        payload = restore_json(json.loads(bytes(response.body)), engine)
        if prepared.client_stream and response.status_code < 400:
            return Response(
                content=complete_sse(payload, protocol),
                status_code=response.status_code,
                headers={name: value for name, value in headers.items() if name != "content-type"},
                media_type="text/event-stream",
            )
        return JSONResponse(content=payload, status_code=response.status_code, headers=headers)
    except ProtectionError as error:
        if prepared.client_stream and protocol is GatewayProtocol.RESPONSES:
            return _terminal_privacy_error(error)
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "invalid_request_error",
                    "code": error.code,
                    "message": str(error),
                }
            },
        )


def _terminal_privacy_error(error: ProtectionError) -> Response:
    return Response(
        content=complete_sse(
            {
                "id": "resp_" + uuid.uuid4().hex,
                "object": "response",
                "status": "failed",
                "output": [],
                "error": {"code": "invalid_prompt", "message": str(error)},
            },
            GatewayProtocol.RESPONSES,
        ),
        media_type="text/event-stream",
    )


def privacy_error(
    error: ProtectionError,
    factory: Callable[[int, str], dict[str, Any]],
    *,
    responses_stream: bool = False,
) -> Response:
    if responses_stream:
        return _terminal_privacy_error(error)
    payload = factory(422, str(error))
    payload["error"]["code"] = error.code
    return JSONResponse(status_code=422, content=payload)


def upstream_response(
    error: UpstreamError,
    prepared: PreparedRequest,
    protocol: GatewayProtocol,
    factory: Callable[[int, str], dict[str, Any]],
) -> Response:
    return restore_response(
        JSONResponse(
            status_code=error.status,
            content=factory(error.status, error.message),
            headers=error.headers,
        ),
        prepared,
        protocol,
    )
