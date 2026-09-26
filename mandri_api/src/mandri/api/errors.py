"""API error models, the ApiError exception, and shared response declarations."""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mandri.core.types.execution import ProtectionError
from mandri.sessions.errors import SessionConflictError, SessionNotFoundError
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class ApiError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status: int,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorEnvelope, "description": "Malformed request"},
    404: {"model": ErrorEnvelope, "description": "Resource not found"},
    409: {"model": ErrorEnvelope, "description": "Conflicting state"},
    422: {"model": ErrorEnvelope, "description": "Request validation failed"},
    502: {"model": ErrorEnvelope, "description": "Upstream harness failure"},
    504: {"model": ErrorEnvelope, "description": "Upstream harness timeout"},
    500: {"model": ErrorEnvelope, "description": "Filesystem or internal read failure"},
}

NOT_FOUND: dict[int | str, dict[str, Any]] = {404: ERROR_RESPONSES[404]}
NOT_FOUND_CONFLICT: dict[int | str, dict[str, Any]] = {
    **NOT_FOUND,
    409: ERROR_RESPONSES[409],
}


async def _handle_api_error(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ApiError):
        raise exc
    return JSONResponse(
        status_code=exc.status,
        content=ErrorEnvelope(
            error=ErrorBody(code=exc.code, message=exc.message, detail=exc.detail or {})
        ).model_dump(),
    )


async def _handle_protection_error(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ProtectionError):
        raise exc
    return JSONResponse(
        status_code=422,
        content=ErrorEnvelope(error=ErrorBody(code=exc.code, message=str(exc))).model_dump(),
    )


async def _handle_validation_error(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        raise exc
    return JSONResponse(
        status_code=422,
        content=ErrorEnvelope(
            error=ErrorBody(
                code="validation_error",
                message="Request validation failed",
                detail={"errors": exc.errors()},
            )
        ).model_dump(),
    )


async def _handle_http_exception(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):
        raise exc
    code = "not_found" if exc.status_code == 404 else f"http_{exc.status_code}"
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorEnvelope(
            error=ErrorBody(code=code, message=str(exc.detail), detail={})
        ).model_dump(),
    )


async def _handle_session_not_found(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, SessionNotFoundError):
        raise exc
    return JSONResponse(
        status_code=404,
        content=ErrorEnvelope(
            error=ErrorBody(
                code="session_not_found",
                message=str(exc),
                detail={
                    key: value
                    for key, value in request.path_params.items()
                    if isinstance(value, str)
                },
            )
        ).model_dump(),
    )


async def _handle_session_conflict(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, SessionConflictError):
        raise exc
    return JSONResponse(
        status_code=409,
        content=ErrorEnvelope(
            error=ErrorBody(code="session_conflict", message=str(exc), detail={})
        ).model_dump(),
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _handle_api_error)
    app.add_exception_handler(ProtectionError, _handle_protection_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(SessionNotFoundError, _handle_session_not_found)
    app.add_exception_handler(SessionConflictError, _handle_session_conflict)
