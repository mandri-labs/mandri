"""Tests for route registry persistence and gateway event emission."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
import pytest_asyncio
from mandri.core.ids import ProviderKind, RouteId, WireFormat
from mandri.core.ports.database import DatabasePort, SqlParams
from mandri.core.ports.gateway_events import GatewayEventSink
from mandri.core.ports.provider_verifier import ProviderVerifierPort, VerificationResult
from mandri.core.ports.routes import RouteLookupPort
from mandri.core.types.config import DaemonConfig, ProviderConfig
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError
from mandri.core.types.gateway import GatewayEventKind
from mandri.gateway.errors.upstream import RouteNotFoundError
from mandri.gateway.route_registry import RouteRegistry
from mandri.providers.errors import ProviderInvalidError, ProviderNotFoundError
from mandri.providers.service import ProvidersRegistry

SCHEMA = (
    "CREATE TABLE gateway_route_history (id INTEGER PRIMARY KEY, route_id TEXT NOT NULL,"
    " effective_from INTEGER NOT NULL, provider_name TEXT NOT NULL,"
    " provider_kind TEXT NOT NULL, model_ref TEXT NOT NULL)",
    """
    CREATE TABLE gateway_route (
      id TEXT PRIMARY KEY,
      provider_name TEXT NOT NULL,
      model_ref TEXT NOT NULL,
      formats TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      reasoning_effort TEXT,
      execution_backend TEXT NOT NULL DEFAULT 'host',
      privacy_mode TEXT NOT NULL DEFAULT 'none',
      privacy_scope_id TEXT
    )
    """,
    """
    CREATE TABLE session (
      id TEXT PRIMARY KEY,
      gateway_route_id TEXT,
      parent_session_id TEXT,
      project_path TEXT NOT NULL DEFAULT '/project',
      harness TEXT NOT NULL DEFAULT 'codex',
      execution_backend TEXT NOT NULL DEFAULT 'host',
      privacy_mode TEXT NOT NULL DEFAULT 'none',
      privacy_scope_id TEXT,
      deleted INTEGER NOT NULL DEFAULT 0,
      created_at INTEGER NOT NULL DEFAULT 0
    )
    """,
)


class FakeDatabase(DatabasePort):
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    @classmethod
    async def create(cls) -> "FakeDatabase":
        connection = await aiosqlite.connect(":memory:", isolation_level=None)
        connection.row_factory = aiosqlite.Row
        for statement in SCHEMA:
            await connection.execute(statement)
        return cls(connection)

    async def connect(self, db_path: Path | str) -> None:
        connection = await aiosqlite.connect(db_path, isolation_level=None)
        connection.row_factory = aiosqlite.Row
        self._connection = connection

    async def close(self) -> None:
        await self._connection.close()

    async def migrate(self) -> None:
        return None

    async def execute(self, sql: str, params: SqlParams = ()) -> None:
        await self._connection.execute(sql, params)

    async def fetch_all(self, sql: str, params: SqlParams = ()) -> list[dict[str, Any]]:
        cursor = await self._connection.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def fetch_one(self, sql: str, params: SqlParams = ()) -> dict[str, Any] | None:
        cursor = await self._connection.execute(sql, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    return {row.keys()[index]: row[index] for index in range(len(row))}


class FakeConfig:
    def __init__(self, config: DaemonConfig | None = None) -> None:
        self.config = config if config is not None else DaemonConfig()

    @property
    def config_path(self) -> Path:
        return Path("config.toml")

    def load(self) -> DaemonConfig:
        return self.config

    def save(self, config: DaemonConfig) -> None:
        self.config = config


class FakeRoutes:
    async def route_ids_for_provider(self, provider_name: str) -> list[RouteId]:
        return []


class FakeVerifier(ProviderVerifierPort):
    def verify(self, kind: ProviderKind, api_base: Any, api_key: str) -> VerificationResult:
        return VerificationResult(ok=True, provider=kind, reason=None)

    async def verify_async(
        self, kind: ProviderKind, api_base: Any, api_key: str
    ) -> VerificationResult:
        return self.verify(kind, api_base, api_key)


class FakeEventSink(GatewayEventSink):
    def __init__(self) -> None:
        self.events: list[tuple[GatewayEventKind, RouteId | None, str | None]] = []

    def publish_event(
        self,
        event: GatewayEventKind,
        route_id: RouteId | None,
        provider_name: str | None,
    ) -> None:
        self.events.append((event, route_id, provider_name))


def default_config() -> DaemonConfig:
    return DaemonConfig(providers=[ProviderConfig(name="prov", kind="openrouter", api_key="k")])


@pytest_asyncio.fixture
async def make_registry() -> AsyncIterator[Any]:
    databases: list[FakeDatabase] = []

    async def create(sink: FakeEventSink, config: DaemonConfig | None = None) -> RouteRegistry:
        db = await FakeDatabase.create()
        databases.append(db)
        providers = ProvidersRegistry(
            FakeConfig(config if config is not None else default_config()),
            FakeRoutes(),
            FakeVerifier(),
        )
        return RouteRegistry(db, providers, sink)

    try:
        yield create
    finally:
        for database in databases:
            await database.close()


def two_provider_config() -> DaemonConfig:
    return DaemonConfig(
        providers=[
            ProviderConfig(name="prov", kind="openrouter", api_key="k1"),
            ProviderConfig(name="other", kind="ollama", api_base="http://localhost:11434"),
        ]
    )


@pytest.mark.parametrize("field", ["execution_backend", "privacy_mode", "privacy_scope_id"])
async def test_session_route_policy_mismatch_cannot_resolve(make_registry, field):
    registry = await make_registry(FakeEventSink())
    route = await registry.create(
        "prov",
        "model",
        (WireFormat.OPENAI,),
        execution_backend=ExecutionBackend.DOCKER,
        privacy_mode=PrivacyMode.SURROGATE,
        privacy_scope_id="scope-one",
    )
    await registry._db.execute(
        "INSERT INTO session (id, gateway_route_id, execution_backend, privacy_mode,"
        " privacy_scope_id)"
        " VALUES ('session-one', ?, 'docker', 'surrogate', 'scope-one')",
        (str(route.id),),
    )
    assert (await registry.resolve(route.id)).privacy_scope_id == "scope-one"
    replacement = {"execution_backend": "host", "privacy_mode": "none", "privacy_scope_id": "other"}
    await registry._db.execute(
        f"UPDATE session SET {field} = ? WHERE id = 'session-one'", (replacement[field],)
    )
    with pytest.raises(ProtectionError, match="policy"):
        await registry.resolve(route.id)


async def test_unbound_or_tombstoned_protected_routes_cannot_send(make_registry):
    registry = await make_registry(FakeEventSink())
    route = await registry.create(
        "prov",
        "model",
        (WireFormat.OPENAI,),
        privacy_mode=PrivacyMode.SURROGATE,
        privacy_scope_id="scope-one",
    )
    with pytest.raises(ProtectionError, match="active session"):
        await registry.resolve(route.id)
    await registry._db.execute(
        "INSERT INTO session (id, gateway_route_id, privacy_mode, privacy_scope_id, deleted)"
        " VALUES ('session-one', ?, 'surrogate', 'scope-one', 1)",
        (str(route.id),),
    )
    with pytest.raises(ProtectionError, match="active session"):
        await registry.resolve(route.id)


async def test_create_persists_route_and_emits_event(make_registry) -> None:
    sink = FakeEventSink()
    registry = await make_registry(sink)
    route = await registry.create("prov", "claude-x", (WireFormat.ANTHROPIC,))
    assert route.provider_name == "prov"
    assert str(route.model_ref) == "openrouter/claude-x"
    routes = await registry.list_routes()
    assert [item.id for item in routes] == [route.id]
    assert sink.events == [(GatewayEventKind.ROUTE_CREATED, route.id, "prov")]


async def test_delete_emits_event_and_route_becomes_unknown(make_registry) -> None:
    sink = FakeEventSink()
    registry = await make_registry(sink)
    route = await registry.create("prov", "claude-x", (WireFormat.OPENAI,))
    await registry.delete(route.id)
    with pytest.raises(RouteNotFoundError):
        await registry.get(route.id)
    assert sink.events[-1] == (GatewayEventKind.ROUTE_DELETED, route.id, "prov")


async def test_swap_updates_model_and_emits_event(make_registry) -> None:
    sink = FakeEventSink()
    registry = await make_registry(sink)
    route = await registry.create("prov", "claude-x", (WireFormat.OPENAI,))
    swapped = await registry.swap(route.id, "prov", "claude-y")
    assert str(swapped.model_ref) == "openrouter/claude-y"
    assert sink.events[-1] == (GatewayEventKind.ROUTE_UPDATED, route.id, "prov")


async def test_route_history_preserves_provider_kind_across_swap_and_delete(make_registry):
    registry = await make_registry(
        FakeEventSink(),
        DaemonConfig(
            providers=[
                ProviderConfig(name="go-account", kind="opencode_go", api_key="fixture-go"),
                ProviderConfig(name="zen-account", kind="opencode", api_key="fixture-zen"),
            ]
        ),
    )
    route = await registry.create("go-account", "glm-5.3-flash", (WireFormat.OPENAI,))
    await registry.swap(route.id, "zen-account", "glm-5.3-flash")
    await registry.delete(route.id)
    history = await registry._db.fetch_all(
        "SELECT provider_name,provider_kind,model_ref FROM gateway_route_history"
        " WHERE route_id=? ORDER BY effective_from,id",
        (str(route.id),),
    )
    assert history == [
        {
            "provider_name": "go-account",
            "provider_kind": "opencode_go",
            "model_ref": "custom_openai/glm-5.3-flash",
        },
        {
            "provider_name": "zen-account",
            "provider_kind": "opencode",
            "model_ref": "custom_openai/glm-5.3-flash",
        },
    ]


async def test_route_ids_for_provider_filters_by_provider(make_registry) -> None:
    sink = FakeEventSink()
    registry = await make_registry(sink, two_provider_config())
    first = await registry.create("prov", "m1", (WireFormat.OPENAI,))
    second = await registry.create("other", "m2", (WireFormat.OPENAI,))
    lookup: RouteLookupPort = registry
    assert await lookup.route_ids_for_provider("prov") == [first.id]
    assert await lookup.route_ids_for_provider("other") == [second.id]
    assert await lookup.route_ids_for_provider("unknown") == []


async def test_create_rejects_empty_model_id(make_registry) -> None:
    registry = await make_registry(FakeEventSink())
    with pytest.raises(ProviderInvalidError):
        await registry.create("prov", "   ", (WireFormat.OPENAI,))


async def test_create_unknown_provider_raises(make_registry) -> None:
    registry = await make_registry(FakeEventSink())
    with pytest.raises(ProviderNotFoundError):
        await registry.create("ghost", "m1", (WireFormat.OPENAI,))


async def test_get_unknown_route_raises(make_registry) -> None:
    registry = await make_registry(FakeEventSink())
    with pytest.raises(RouteNotFoundError):
        await registry.get(RouteId("missing"))


async def test_create_persists_reasoning_effort(make_registry) -> None:
    sink = FakeEventSink()
    registry = await make_registry(sink)
    route = await registry.create("prov", "claude-x", (WireFormat.OPENAI,), reasoning_effort="high")
    assert route.reasoning_effort == "high"
    stored = await registry.get(route.id)
    assert stored.reasoning_effort == "high"
    resolved = await registry.resolve(route.id)
    assert resolved.reasoning_effort == "high"


async def test_set_reasoning_effort_updates_and_emits_event(make_registry) -> None:
    sink = FakeEventSink()
    registry = await make_registry(sink)
    route = await registry.create("prov", "claude-x", (WireFormat.OPENAI,))
    await registry.set_reasoning_effort(route.id, "medium")
    stored = await registry.get(route.id)
    assert stored.reasoning_effort == "medium"
    assert sink.events[-1] == (GatewayEventKind.ROUTE_UPDATED, route.id, "prov")
    await registry.set_reasoning_effort(route.id, None)
    assert (await registry.get(route.id)).reasoning_effort is None


async def test_set_reasoning_effort_unknown_route_raises(make_registry) -> None:
    registry = await make_registry(FakeEventSink())
    with pytest.raises(RouteNotFoundError):
        await registry.set_reasoning_effort(RouteId("missing"), "high")


async def test_usage_shared_route_attributes_root_not_arbitrary_child(make_registry):
    registry = await make_registry(FakeEventSink())
    route = await registry.create("prov", "m", (WireFormat.OPENAI,))
    await registry._db.execute("INSERT INTO session(id) VALUES ('root')")
    for child in ("child-a", "child-b"):
        await registry._db.execute(
            "INSERT INTO session(id, parent_session_id, gateway_route_id) VALUES (?, 'root', ?)",
            (child, str(route.id)),
        )
    attribution = await registry.usage_attribution(route.id)
    assert attribution.root_session_id == "root"
    assert attribution.project_path == "/project"
    assert attribution.harness == "codex"
    await registry._db.execute(
        "INSERT INTO session(id, parent_session_id, project_path, harness)"
        " VALUES ('inherited-child', 'root', '/other', 'claude')"
    )
    attribution = await registry.usage_attribution(route.id)
    assert attribution.root_session_id == "root"
    assert attribution.project_path is None and attribution.harness is None


@pytest.mark.parametrize("parent", [None, "missing", "child"])
async def test_usage_ambiguous_missing_or_cyclic_lineage_is_unassigned(make_registry, parent):
    registry = await make_registry(FakeEventSink())
    route = await registry.create("prov", "m", (WireFormat.OPENAI,))
    await registry._db.execute(
        "INSERT INTO session(id, gateway_route_id) VALUES ('root', ?)", (str(route.id),)
    )
    await registry._db.execute(
        "INSERT INTO session(id, parent_session_id, gateway_route_id) VALUES ('child', ?, ?)",
        (parent, str(route.id)),
    )
    attribution = await registry.usage_attribution(route.id)
    assert attribution.root_session_id is None and attribution.project_path is None
