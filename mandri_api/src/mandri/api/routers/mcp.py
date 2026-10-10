import typing
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse
from mandri.api.deps import app_state
from mandri.api.errors import ApiError
from mandri.core.ids import HarnessKind
from mandri.core.types.mcp import McpBinding, McpServerConfig, McpServerView, McpSnapshot
from mandri.mcp.service import McpService
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/mcp", tags=["mcp"])


def service(request: Request) -> McpService:
    state = app_state(request.app)
    if state is None or state.mcp is None:
        raise ApiError("service_unavailable", "MCP service is unavailable", 503)
    return state.mcp


Mcp = typing.Annotated[McpService, Depends(service)]


class McpUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int = Field(ge=1)
    changes: dict[str, Any]


class McpBindRequest(BaseModel):
    harness: HarnessKind


@router.get("/servers", operation_id="list_mcp_servers")
async def list_servers(mcp: Mcp) -> McpSnapshot:
    return mcp.snapshot()


@router.post("/servers", operation_id="create_mcp_server", status_code=201)
async def create_server(body: McpServerConfig, mcp: Mcp) -> McpServerView:
    try:
        return await mcp.add(body)
    except ValueError as error:
        raise ApiError("mcp_invalid", str(error), 409) from None


@router.patch("/servers/{identity}", operation_id="update_mcp_server")
async def update_server(identity: str, body: McpUpdate, mcp: Mcp) -> McpServerView:
    try:
        return await mcp.update(identity, body.revision, body.changes)
    except KeyError:
        raise ApiError("mcp_not_found", "Server not found", 404) from None
    except ValueError:
        raise ApiError("mcp_invalid", "Invalid changes or stale server revision", 409) from None


@router.delete("/servers/{identity}", operation_id="delete_mcp_server", status_code=204)
async def delete_server(identity: str, mcp: Mcp) -> Response:
    try:
        await mcp.remove(identity)
    except KeyError:
        raise ApiError("mcp_not_found", "Server not found", 404) from None
    except ValueError as error:
        raise ApiError("mcp_invalid", str(error), 409) from None
    return Response(status_code=204)


@router.post("/servers/{identity}/reconnect", operation_id="reconnect_mcp_server")
async def reconnect_server(identity: str, mcp: Mcp) -> McpServerView:
    try:
        return await mcp.reconnect(identity)
    except KeyError:
        raise ApiError("mcp_not_found", "Server not found", 404) from None


@router.post("/bindings", operation_id="create_mcp_binding", status_code=201)
async def create_binding(body: McpBindRequest, mcp: Mcp) -> McpBinding:
    return mcp.bind(f"cli-{uuid.uuid4().hex}", body.harness, lease=True)


@router.put("/bindings/{identity}", operation_id="renew_mcp_binding", status_code=204)
async def renew_binding(identity: str, mcp: Mcp) -> Response:
    try:
        mcp.bindings.renew(identity)
    except KeyError:
        raise ApiError("mcp_binding_expired", "Binding expired", 404) from None
    return Response(status_code=204)


@router.delete("/bindings/{identity}", operation_id="release_mcp_binding", status_code=204)
async def release_binding(identity: str, mcp: Mcp) -> Response:
    mcp.bindings.revoke(identity)
    return Response(status_code=204)


class McpOAuthCallback:
    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        request = Request(scope)
        identity = scope["path"].rsplit("/", 1)[-1]
        try:
            service(request).oauth.complete(
                identity,
                request.query_params["code"],
                request.query_params["state"],
                request.query_params.get("iss"),
            )
        except (ValueError, KeyError):
            await HTMLResponse("Authorization failed", status_code=400)(scope, receive, send)
            return
        await HTMLResponse("Authorization received")(scope, receive, send)


@router.post("/servers/{identity}/login", operation_id="login_mcp_server")
async def login_server(identity: str, mcp: Mcp) -> McpServerView:
    try:
        return await mcp.login(identity)
    except KeyError:
        raise ApiError("mcp_not_found", "Server not found", 404) from None
    except ValueError as error:
        raise ApiError("mcp_invalid", str(error), 409) from None


@router.post("/servers/{identity}/logout", operation_id="logout_mcp_server")
async def logout_server(identity: str, mcp: Mcp) -> McpServerView:
    try:
        return await mcp.logout(identity)
    except KeyError:
        raise ApiError("mcp_not_found", "Server not found", 404) from None


class McpData:
    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        request = Request(scope)
        await service(request).dispatch(scope, receive, send)
