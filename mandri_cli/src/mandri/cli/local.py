"""Per-command composition roots for CLI-local management commands."""

from pathlib import Path

from mandri.config.toml_adapter import DEFAULT_BASE_DIR, TomlConfigAdapter
from mandri.config.types import default_opencode_db_path
from mandri.core.ids import RouteId
from mandri.core.ports.routes import RouteLookupPort
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.providers.service import ProvidersRegistry
from mandri.sessions.bootstrap import build_sessions_backends
from mandri.sessions.docker_titles import docker_title
from mandri.sessions.service import SessionsService
from mandri.sessions.sync import SyncEngine


class DatabaseRouteLookup(RouteLookupPort):
    """Provider route lookup backed by the gateway route table."""

    def __init__(self, db: AiosqliteDatabase) -> None:
        self._db = db

    async def route_ids_for_provider(self, provider_name: str) -> list[RouteId]:
        rows = await self._db.fetch_all(
            "SELECT id FROM gateway_route WHERE provider_name = ?", (provider_name,)
        )
        return [RouteId(str(row["id"])) for row in rows]


def resolve_base_dir(base_dir: Path | None) -> Path:
    return base_dir if base_dir is not None else DEFAULT_BASE_DIR


def build_config(base_dir: Path | None) -> TomlConfigAdapter:
    return TomlConfigAdapter(resolve_base_dir(base_dir))


async def build_database(base_dir: Path | None) -> AiosqliteDatabase:
    resolved = resolve_base_dir(base_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    db = AiosqliteDatabase()
    await db.connect(resolved / "mandri.db")
    await db.migrate()
    return db


async def build_providers_registry(
    base_dir: Path | None,
) -> tuple[ProvidersRegistry, AiosqliteDatabase]:
    db = await build_database(base_dir)
    return ProvidersRegistry(build_config(base_dir), DatabaseRouteLookup(db)), db


async def build_sessions_service(
    base_dir: Path | None,
) -> tuple[SessionsService, AiosqliteDatabase]:
    db = await build_database(base_dir)
    config = build_config(base_dir).load()
    engine = SyncEngine(
        db,
        {
            kind: backend
            for kind, backend in build_sessions_backends(
                config.sessions, Path(default_opencode_db_path())
            ).items()
            if backend is not None
        },
        ttl_seconds=config.sync.ttl_seconds,
        docker_title_reader=docker_title,
    )
    return SessionsService(db, engine), db
