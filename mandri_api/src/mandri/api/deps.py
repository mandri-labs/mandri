"""Typed application state and FastAPI dependency accessors."""

import dataclasses
import typing
from collections.abc import Awaitable, Callable

import httpx
from fastapi import Depends, FastAPI, Request
from mandri.api.agent_observer import AgentObserver
from mandri.api.errors import ApiError
from mandri.api.transcript_observer import TranscriptObserver
from mandri.core.hub import Hub
from mandri.core.ports.database import DatabasePort
from mandri.core.protocol.registry import ActionRegistry
from mandri.database.usage import UsageRepository
from mandri.gateway.auth import ChildTokenAuth
from mandri.gateway.litellm_adapter import (
    AnthropicHandler,
    GeminiHandler,
    OpenAIHandler,
    ResponsesHandler,
)
from mandri.gateway.privacy import GatewayPrivacy
from mandri.gateway.reasoning_catalog import ReasoningCatalog
from mandri.gateway.route_registry import RouteRegistry
from mandri.gateway.usage import GatewayUsageSink
from mandri.providers.service import ProvidersRegistry
from mandri.runtime.agents import AgentService
from mandri.runtime.lifetime import SessionLifetimePort
from mandri.runtime.service import RuntimeService
from mandri.sessions.service import SessionsService


@dataclasses.dataclass
class GatewayWiring:
    registry: RouteRegistry
    openai: OpenAIHandler
    anthropic: AnthropicHandler
    responses: ResponsesHandler = dataclasses.field(default_factory=ResponsesHandler)
    gemini: GeminiHandler | None = None
    auth: ChildTokenAuth = dataclasses.field(default_factory=ChildTokenAuth)
    reasoning_catalog: ReasoningCatalog | None = None
    privacy: GatewayPrivacy | None = None
    usage_sink: GatewayUsageSink | None = None

    def issue_child_token(self, route_id: str) -> str:
        return self.auth.issue(route_id)


@dataclasses.dataclass
class LifespanState:
    usage: UsageRepository | None = None
    usage_refresh: Callable[[], Awaitable[bool]] | None = None
    db: DatabasePort | None = None
    http: httpx.AsyncClient | None = None
    hub: Hub | None = None
    sessions: SessionsService | None = None
    providers: ProvidersRegistry | None = None
    gateway: GatewayWiring | None = None
    runtime: RuntimeService | None = None
    lifetime: SessionLifetimePort | None = None
    actions: ActionRegistry | None = None
    agents: AgentService | None = None
    agent_observer: AgentObserver | None = None
    heartbeat_idle_seconds: float = 15.0
    heartbeat_max_misses: int = 2
    heartbeat_configured: bool = False
    transcript_observer: TranscriptObserver | None = None
    viewer_counts: dict[str, int] = dataclasses.field(default_factory=dict)


def app_state(app: FastAPI) -> LifespanState | None:
    state = getattr(app.state, "lifespan", None)
    return state if isinstance(state, LifespanState) else None


def _state(request: Request) -> LifespanState:
    state = app_state(request.app)
    if state is None:
        raise ApiError(
            code="service_unavailable",
            message="Application state is not available",
            status=503,
        )
    return state


def _require[T](value: T | None, message: str) -> T:
    if value is None:
        raise ApiError(code="service_unavailable", message=message, status=503)
    return value


def sessions_service(request: Request) -> SessionsService:
    return _require(_state(request).sessions, "Sessions service is not available")


def runtime_service(request: Request) -> RuntimeService:
    return _require(_state(request).runtime, "Runtime service is not available")


def providers_registry(request: Request) -> ProvidersRegistry:
    return _require(_state(request).providers, "Providers service is not available")


def http_client(request: Request) -> httpx.AsyncClient:
    return _require(_state(request).http, "HTTP client is not available")


def database(request: Request) -> DatabasePort:
    return _require(_state(request).db, "Database is not available")


def gateway_wiring(request: Request) -> GatewayWiring:
    return _require(_state(request).gateway, "Gateway is not available")


def usage_repository(request: Request) -> UsageRepository:
    return _require(_state(request).usage, "Usage statistics are not available")


Sessions = typing.Annotated[SessionsService, Depends(sessions_service)]
Runtime = typing.Annotated[RuntimeService, Depends(runtime_service)]
Providers = typing.Annotated[ProvidersRegistry, Depends(providers_registry)]
Http = typing.Annotated[httpx.AsyncClient, Depends(http_client)]
Database = typing.Annotated[DatabasePort, Depends(database)]
Gateway = typing.Annotated[GatewayWiring, Depends(gateway_wiring)]
Usage = typing.Annotated[UsageRepository, Depends(usage_repository)]
