"""Gateway route registry persisted in the gateway_route table."""

import dataclasses
import json
import uuid

from mandri.core.clock import system_now_ms
from mandri.core.ids import EpochMs, ModelRef, ProviderKind, RouteId, WireFormat
from mandri.core.ports.database import DatabasePort
from mandri.core.ports.gateway_events import GatewayEventSink
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError
from mandri.core.types.gateway import GatewayEventKind
from mandri.gateway.errors.upstream import RouteNotFoundError
from mandri.gateway.types.model import Model
from mandri.gateway.usage_attribution import UsageAttribution, route_attribution
from mandri.providers.errors import ProviderInvalidError
from mandri.providers.service import Provider, ProvidersRegistry, provider_model_ref

_ROUTE_COLUMNS = (
    "id, provider_name, model_ref, formats, created_at, reasoning_effort,"
    " execution_backend, privacy_mode, privacy_scope_id"
)


@dataclasses.dataclass(frozen=True)
class Route:
    id: RouteId
    provider_name: str
    model_ref: ModelRef
    formats: tuple[WireFormat, ...]
    created_at: EpochMs
    reasoning_effort: str | None = None
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE
    privacy_scope_id: str | None = None


@dataclasses.dataclass(frozen=True)
class ResolvedRoute:
    route_id: RouteId
    provider: Provider
    model: Model
    reasoning_effort: str | None = None
    conversation_id: str | None = None
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE
    privacy_scope_id: str | None = None


def _formats(raw: str) -> tuple[WireFormat, ...]:
    return tuple(WireFormat(item) for item in json.loads(raw))


def _row_to_route(row: dict[str, object]) -> Route:
    effort = row.get("reasoning_effort")
    return Route(
        id=RouteId(str(row["id"])),
        provider_name=str(row["provider_name"]),
        model_ref=ModelRef(str(row["model_ref"])),
        formats=_formats(str(row["formats"])),
        created_at=EpochMs(int(str(row["created_at"]))),
        reasoning_effort=str(effort) if isinstance(effort, str) else None,
        execution_backend=ExecutionBackend(str(row.get("execution_backend", "host"))),
        privacy_mode=PrivacyMode(str(row.get("privacy_mode", "none"))),
        privacy_scope_id=str(row["privacy_scope_id"]) if row.get("privacy_scope_id") else None,
    )


def _ref(provider: Provider, model_id: str) -> ModelRef:
    if not model_id.strip():
        raise ProviderInvalidError("model_id must be a non-empty string")
    return provider_model_ref(provider.kind, model_id)


class RouteRegistry:
    def __init__(
        self,
        db: DatabasePort,
        providers: ProvidersRegistry,
        events: GatewayEventSink | None = None,
    ) -> None:
        self._db = db
        self._providers = providers
        self._events = events

    @property
    def providers(self) -> ProvidersRegistry:
        return self._providers

    async def usage_attribution(self, route_id: RouteId) -> UsageAttribution:
        return await route_attribution(self._db, str(route_id))

    async def create(
        self,
        provider_name: str,
        model_id: str,
        formats: tuple[WireFormat, ...],
        reasoning_effort: str | None = None,
        execution_backend: ExecutionBackend = ExecutionBackend.HOST,
        privacy_mode: PrivacyMode = PrivacyMode.NONE,
        privacy_scope_id: str | None = None,
    ) -> Route:
        if privacy_mode is PrivacyMode.SURROGATE and not privacy_scope_id:
            raise ProtectionError("privacy_state_unavailable", "A protected route requires a scope")
        provider = self._providers.get(provider_name)
        effort = reasoning_effort.strip() if isinstance(reasoning_effort, str) else None
        route = Route(
            id=RouteId(str(uuid.uuid4())),
            provider_name=provider.name,
            model_ref=_ref(provider, model_id),
            formats=tuple(formats),
            created_at=system_now_ms(),
            reasoning_effort=effort or None,
            execution_backend=execution_backend,
            privacy_mode=privacy_mode,
            privacy_scope_id=privacy_scope_id,
        )
        await self._db.execute(
            "INSERT INTO gateway_route"
            " (id, provider_name, model_ref, formats, created_at, reasoning_effort,"
            " execution_backend, privacy_mode, privacy_scope_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(route.id),
                route.provider_name,
                str(route.model_ref),
                json.dumps([fmt.value for fmt in route.formats]),
                int(route.created_at),
                route.reasoning_effort,
                route.execution_backend.value,
                route.privacy_mode.value,
                route.privacy_scope_id,
            ),
        )
        self._emit(GatewayEventKind.ROUTE_CREATED, route.id, route.provider_name)
        return route

    async def list_routes(self) -> list[Route]:
        rows = await self._db.fetch_all(
            f"SELECT {_ROUTE_COLUMNS} FROM gateway_route ORDER BY rowid"
        )
        return [_row_to_route(row) for row in rows]

    async def route_ids_for_provider(self, provider_name: str) -> list[RouteId]:
        rows = await self._db.fetch_all(
            "SELECT id FROM gateway_route WHERE provider_name = ?", (provider_name,)
        )
        return [RouteId(str(row["id"])) for row in rows]

    async def get(self, route_id: RouteId) -> Route:
        row = await self._db.fetch_one(
            f"SELECT {_ROUTE_COLUMNS} FROM gateway_route WHERE id = ?", (str(route_id),)
        )
        if row is None:
            raise RouteNotFoundError(f"unknown route {route_id}")
        return _row_to_route(row)

    async def delete(self, route_id: RouteId) -> None:
        route = await self.get(route_id)
        await self._db.execute("DELETE FROM gateway_route WHERE id = ?", (str(route_id),))
        self._emit(GatewayEventKind.ROUTE_DELETED, route.id, route.provider_name)

    async def swap(self, route_id: RouteId, provider_name: str, model_id: str) -> Route:
        provider = self._providers.get(provider_name)
        await self.get(route_id)
        await self._db.execute(
            "UPDATE gateway_route SET provider_name = ?, model_ref = ? WHERE id = ?",
            (provider.name, str(_ref(provider, model_id)), str(route_id)),
        )
        route = await self.get(route_id)
        self._emit(GatewayEventKind.ROUTE_UPDATED, route.id, route.provider_name)
        return route

    async def set_reasoning_effort(self, route_id: RouteId, effort: str | None) -> None:
        route = await self.get(route_id)
        normalized = effort.strip() if isinstance(effort, str) else None
        await self._db.execute(
            "UPDATE gateway_route SET reasoning_effort = ? WHERE id = ?",
            (normalized or None, str(route_id)),
        )
        self._emit(GatewayEventKind.ROUTE_UPDATED, route.id, route.provider_name)

    async def resolve(self, route_id: RouteId) -> ResolvedRoute:
        route = await self.get(route_id)
        conversation_id = await self._bound_conversation(route)
        provider = self._providers.get(route.provider_name)
        model = Model(
            provider=provider.kind,
            model_ref=route.model_ref,
            api_base=provider.api_base,
            api_key=provider.api_key,
        )
        return ResolvedRoute(
            route_id=route.id,
            provider=provider,
            model=model,
            reasoning_effort=route.reasoning_effort,
            conversation_id=(
                conversation_id
                if provider.kind in (ProviderKind.OPENCODE, ProviderKind.OPENCODE_GO)
                else None
            ),
            execution_backend=route.execution_backend,
            privacy_mode=route.privacy_mode,
            privacy_scope_id=route.privacy_scope_id,
        )

    async def _bound_conversation(self, route: Route) -> str | None:
        rows = await self._db.fetch_all(
            "SELECT id, execution_backend, privacy_mode, privacy_scope_id, deleted"
            " FROM session WHERE gateway_route_id = ? ORDER BY created_at, id",
            (str(route.id),),
        )
        active = [row for row in rows if not row["deleted"]]
        if not active and (rows or route.privacy_mode is PrivacyMode.SURROGATE):
            raise ProtectionError("privacy_route_unbound", "The route has no active session")
        if route.privacy_mode is PrivacyMode.SURROGATE and not route.privacy_scope_id:
            raise ProtectionError("privacy_state_unavailable", "The protected route has no scope")
        for row in active:
            if (
                row["execution_backend"] != route.execution_backend.value
                or row["privacy_mode"] != route.privacy_mode.value
                or row["privacy_scope_id"] != route.privacy_scope_id
            ):
                raise ProtectionError(
                    "privacy_route_mismatch", "The route does not match its session policy"
                )
        return str(active[0]["id"]) if active else None

    def _emit(self, event: GatewayEventKind, route_id: RouteId, provider_name: str | None) -> None:
        if self._events is not None:
            self._events.publish_event(event, route_id, provider_name)
