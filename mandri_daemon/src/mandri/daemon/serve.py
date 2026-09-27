"""Uvicorn server assembly and runtime wiring for the Mandri daemon."""

import asyncio
import contextlib
import dataclasses
import logging
import random
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from codex_cli_bin import bundled_codex_path
from fastapi import FastAPI
from mandri.api.actions import build_action_registry
from mandri.api.app import create_app
from mandri.api.deps import GatewayWiring
from mandri.api.local_security import LocalSecurity
from mandri.config.toml_adapter import TomlConfigAdapter
from mandri.config.types import default_opencode_db_path
from mandri.core.hub import Hub
from mandri.core.ids import HarnessKind, RouteId, SessionId, SessionStopCause, SessionTitle
from mandri.core.ports.control import ApprovalDelivery, HarnessControl
from mandri.core.ports.routes import RouteLookupPort
from mandri.core.ports.sessions import (
    CheckSessionExistsPort,
    DeleteSessionPort,
    FetchSessionsPort,
    RenameSessionPort,
)
from mandri.core.protocol.commands import CommandCatalogParams
from mandri.core.protocol.registry import ActionRegistry
from mandri.core.types.config import DEFAULT_CORS_ORIGINS, CorsOrigin, DaemonConfig, SessionsConfig
from mandri.core.types.execution import PrivacyMode, SessionPolicy
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.sessions import Session, SessionStateError
from mandri.core.usage_pricing import bundled_prices
from mandri.core.version import __version__
from mandri.daemon.agent_cache import backfill_agents, discover_agents
from mandri.daemon.command_catalogs import create_command_catalog_cache
from mandri.daemon.daemonctl import remove_pid_file, write_pid_file
from mandri.daemon.desktop import install_controls
from mandri.daemon.docker import docker_config
from mandri.daemon.interrupts import DaemonServer
from mandri.daemon.privacy import build_privacy
from mandri.daemon.usage import UsageCoordinator
from mandri.daemon.usage_accounts import NativeQuotaSync
from mandri.daemon.usage_history import UsageHistorySync
from mandri.daemon.usage_prices import PriceCatalogSync
from mandri.database.executions import ExecutionRepository
from mandri.database.native_sessions import NativePiSessionIdentities
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.gateway.adapters.hub_event_sink import HubGatewayEventSink
from mandri.gateway.litellm_adapter import AnthropicHandler, GeminiHandler, OpenAIHandler
from mandri.gateway.reasoning_catalog import build_reasoning_catalog
from mandri.gateway.route_registry import RouteRegistry
from mandri.providers.service import ProvidersRegistry
from mandri.runtime.adapters import AdapterContext, HarnessAdapters
from mandri.runtime.agent_events import AgentEventRouter
from mandri.runtime.agents import AgentService
from mandri.runtime.control.agy import AgyControlAdapter
from mandri.runtime.control.claude import ClaudeApprovalMessenger, ClaudeControlAdapter
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.control.codex_delivery import CodexApprovalDelivery
from mandri.runtime.control.opencode import OpencodeApprovalDelivery, OpencodeControlAdapter
from mandri.runtime.control.pi import PiControlAdapter
from mandri.runtime.control.pi_delivery import PiApprovalDelivery
from mandri.runtime.control_hub import HubEventLines
from mandri.runtime.executable import resolve_executable
from mandri.runtime.liveness import WorkingStateTracker
from mandri.runtime.pump import LinePump
from mandri.runtime.service import RuntimeService
from mandri.runtime.stdin_sink import ProcessStdinSink
from mandri.runtime.usage import NativeUsageCollector
from mandri.sessions.adapters.agy_sessions import AgySessionsAdapter
from mandri.sessions.adapters.claude_fetch_sessions import ClaudeSdkFetchSessionsAdapter
from mandri.sessions.adapters.claude_mutations import (
    ClaudeSdkCheckSessionExistsAdapter,
    ClaudeSdkDeleteSessionAdapter,
    ClaudeSdkRenameSessionAdapter,
)
from mandri.sessions.adapters.codex_fetch_sessions import CodexFetchSessions
from mandri.sessions.adapters.codex_mutations import (
    CodexMutationsAdapter,
    connect_codex_app_server,
)
from mandri.sessions.adapters.opencode_fetch_sessions import (
    OpencodeSqliteFetchSessionsAdapter,
)
from mandri.sessions.adapters.opencode_mutations import (
    OpencodeRestCheckSessionExistsAdapter,
    OpencodeRestDeleteSessionAdapter,
    OpencodeRestRenameSessionAdapter,
)
from mandri.sessions.adapters.pi_sessions import PiSessionsAdapter
from mandri.sessions.agents.bootstrap import build_agent_history
from mandri.sessions.agy_profiles import default_agy_root
from mandri.sessions.docker_titles import docker_title
from mandri.sessions.errors import (
    SessionDeleteError,
    SessionRenameError,
)
from mandri.sessions.pi_store import PiSessionStore
from mandri.sessions.service import SessionsService
from mandri.sessions.sync import SessionsBackend, SyncEngine
from mandri.sessions.transcripts import (
    ClaudeTranscriptReader,
    CodexTranscriptReader,
    OpencodeTranscriptReader,
    TranscriptResolver,
)
from mandri.sessions.transcripts.agy_transcripts import AgyTranscriptReader
from mandri.sessions.transcripts.pi_transcripts import PiTranscriptReader

DEFAULT_CODEX_HOME = Path.home() / ".codex"
CODEX_STATE_DB_NAME = "state_5.sqlite"
CODEX_SESSIONS_DIR_NAME = "sessions"
PID_BIND_POLL_SECONDS = 0.05

logger = logging.getLogger(__name__)

HARNESS_LAUNCH_ARGS: Mapping[str, tuple[str, ...]] = {
    HarnessKind.AGY.value: (
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--print-timeout",
        "24h",
    ),
    HarnessKind.PI.value: ("--mode", "rpc"),
    HarnessKind.CODEX.value: ("app-server",),
    HarnessKind.CLAUDE.value: (
        "--output-format",
        "stream-json",
        "--input-format",
        "stream-json",
        "--include-partial-messages",
        "--permission-prompt-tool",
        "stdio",
        "--verbose",
        "-p",
    ),
    HarnessKind.OPENCODE.value: ("serve", "--port", "{listen_port}"),
}


def build_harness_commands(
    overrides: Mapping[str, list[str]] | None = None,
    *,
    include_docker: bool = False,
) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for harness, launch_args in HARNESS_LAUNCH_ARGS.items():
        if overrides and harness in overrides:
            continue
        path = bundled_codex_path() if harness == "codex" else resolve_executable(harness)
        if path is not None:
            commands[harness] = [str(path), *launch_args]
        elif include_docker:
            commands[harness] = [harness, *launch_args]
    if overrides:
        commands.update({harness: list(argv) for harness, argv in overrides.items()})
    return commands


def build_harness_adapters(context: AdapterContext) -> HarnessAdapters | None:
    if context.kind is HarnessKind.OPENCODE:
        return _opencode_adapters(context)
    return _stdio_adapters(context)


def _opencode_adapters(context: AdapterContext) -> HarnessAdapters | None:
    if context.listen_port is None or context.native_session_id is None:
        return None
    control = OpencodeControlAdapter(
        base_url=_opencode_base_url(context.listen_port),
        session_id=context.native_session_id,
        auth=context.control_auth,
    )
    return HarnessAdapters(
        control=control, delivery=OpencodeApprovalDelivery(control), events=control
    )


def _stdio_adapters(context: AdapterContext) -> HarnessAdapters | None:
    if context.process is None or context.hub is None or context.topic is None:
        return None
    lines = HubEventLines(
        context.hub, context.topic, context.kind.value, since=context.feed_start_seq
    )
    stdout_pump = LinePump(lines.chunks, limit=None, on_close=lines.close)
    stdin = ProcessStdinSink(context.process)
    control: HarnessControl
    delivery: ApprovalDelivery
    if context.kind is HarnessKind.AGY:
        if context.agy_bridge is None:
            lines.close()
            return None
        control = AgyControlAdapter(
            stdout_pump,
            stdin,
            context.agy_bridge,
            context.process.interrupt,
            context.resume_thread_id,
            context.on_identity,
            commands=context.agy_commands,
        )
        return HarnessAdapters(control=control, delivery=context.agy_bridge)
    if context.kind is HarnessKind.CLAUDE:
        stderr_pump = LinePump(_empty_lines)
        control = ClaudeControlAdapter(
            stdout_pump=stdout_pump,
            stderr_pump=stderr_pump,
            stdin=stdin,
            on_identity=context.on_identity,
            on_conversation_reset=context.on_conversation_reset,
        )
        delivery = ClaudeApprovalMessenger(stdin=stdin)
    elif context.kind is HarnessKind.PI:
        control = PiControlAdapter(
            stdout_pump,
            stdin,
            expected_session_id=context.resume_thread_id,
            thinking_level=context.reasoning_effort,
            gateway_mode=context.model_source is ModelSource.GATEWAY,
            on_session_path=context.on_session_path,
            on_model_selection=context.on_model_selection,
            on_identity=context.on_identity,
            on_conversation_reset=context.on_conversation_reset,
        )
        delivery = PiApprovalDelivery(control)
    elif context.kind is HarnessKind.CODEX:
        params: dict[str, Any] = (
            dict(context.launch_mode.codex_params) if context.launch_mode is not None else {}
        )
        if context.model_source is ModelSource.NATIVE:
            params["modelProvider"] = "openai"
            if context.reasoning_effort is not None:
                params["config"] = {"model_reasoning_effort": context.reasoning_effort}
            if context.model and context.model != "default":
                params["model"] = context.model
        elif context.model:
            params.update(model=context.model, modelProvider="mandri")
        if context.fork_thread_id is not None and context.workspace_root is not None:
            params.update(
                cwd=context.workspace_root, runtimeWorkspaceRoots=[context.workspace_root]
            )
        control = CodexControlAdapter(
            stdout_pump,
            stdin,
            thread_start_params=params,
            resume_thread_id=context.resume_thread_id,
            fork_thread_id=context.fork_thread_id,
            fork_path=context.fork_path,
        )
        delivery = CodexApprovalDelivery(control)
    else:
        lines.close()
        return None
    return HarnessAdapters(control=control, delivery=delivery)


async def _empty_lines() -> AsyncIterator[bytes]:
    return
    yield


def _opencode_base_url(listen_port: int) -> str:
    return f"http://127.0.0.1:{listen_port}"


class _DeferredRoutes:
    """Resolves the route registry after both wiring directions are constructed."""

    def __init__(self, handle: list[RouteLookupPort]) -> None:
        self._handle = handle

    async def route_ids_for_provider(self, provider_name: str) -> list[RouteId]:
        return await self._handle[0].route_ids_for_provider(provider_name)


class RuntimeResources:
    def __init__(self) -> None:
        self.db: AiosqliteDatabase | None = None
        self.http: httpx.AsyncClient | None = None
        self.hub: Hub | None = None
        self.sessions: SessionsService | None = None
        self.runtime: RuntimeService | None = None
        self.providers: ProvidersRegistry | None = None
        self.gateway: GatewayWiring | None = None
        self.actions: ActionRegistry | None = None
        self.agents: AgentService | None = None
        self.agent_tasks: list[asyncio.Task[None]] = []
        self.usage: UsageCoordinator | None = None


class HarnessSessionsBackend:
    def __init__(
        self,
        harness: HarnessKind,
        fetch: FetchSessionsPort,
        rename: RenameSessionPort | None = None,
        delete: DeleteSessionPort | None = None,
        exists: CheckSessionExistsPort | None = None,
    ) -> None:
        self._harness = harness
        self._fetch = fetch
        self._rename = rename
        self._delete = delete
        self._exists = exists

    def fetch(self) -> list[Session]:
        return self._fetch.fetch()

    def rename(self, session_id: SessionId, title: SessionTitle) -> None:
        if self._rename is None:
            raise SessionRenameError(f"mutations disabled for {self._harness.value}")
        self._rename.rename(session_id, title)

    def delete(self, session_id: SessionId) -> None:
        if self._delete is None:
            raise SessionDeleteError(f"mutations disabled for {self._harness.value}")
        self._delete.delete(session_id)

    def exists(self, session_id: SessionId) -> bool:
        if self._exists is None:
            raise SessionStateError(f"mutations disabled for {self._harness.value}")
        return self._exists.exists(session_id)


def build_sessions_backends(config: DaemonConfig) -> Mapping[HarnessKind, SessionsBackend | None]:
    sessions = config.sessions
    agy = AgySessionsAdapter(
        Path(sessions.agy_home) if sessions.agy_home else default_agy_root(),
        Path(sessions.agy_profiles_dir) if sessions.agy_profiles_dir else None,
    )
    return {
        HarnessKind.AGY: HarnessSessionsBackend(HarnessKind.AGY, agy, delete=agy, exists=agy),
        HarnessKind.PI: _build_pi_backend(),
        HarnessKind.OPENCODE: _build_opencode_backend(sessions),
        HarnessKind.CODEX: _build_codex_backend(sessions),
        HarnessKind.CLAUDE: _build_claude_backend(sessions),
    }


def load_config(
    base_dir: Path,
    host: str | None = None,
    port: int | None = None,
    *,
    persist_overrides: bool = True,
) -> DaemonConfig:
    adapter = TomlConfigAdapter(base_dir)
    config = adapter.load()
    if not adapter.config_path.is_file():
        adapter.save(config)
    if host is not None or port is not None:
        server = replace(
            config.server,
            host=host if host is not None else config.server.host,
            port=port if port is not None else config.server.port,
        )
        config = replace(config, server=server)
    if persist_overrides and (host is not None or port is not None):
        adapter.save(config)
    return config


def build_server(
    host: str,
    port: int,
    cors_origins: Sequence[CorsOrigin] | None = None,
    desktop_token: str | None = None,
) -> tuple[uvicorn.Server, RuntimeResources]:
    resources = RuntimeResources()
    app = build_app(resources, cors_origins)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = DaemonServer(config)
    if host in {"127.0.0.1", "localhost"}:
        app.add_middleware(
            LocalSecurity,
            hosts=["127.0.0.1", "localhost"],
            origins=list(cors_origins if cors_origins is not None else DEFAULT_CORS_ORIGINS),
            token=desktop_token,
        )
    if desktop_token:
        install_controls(app, server)
    return server, resources


def build_app(
    resources: RuntimeResources,
    cors_origins: Sequence[CorsOrigin] | None = None,
) -> FastAPI:
    app = create_app(cors_origins)
    original = app.router.lifespan_context

    @asynccontextmanager
    async def wired(incoming: FastAPI) -> AsyncIterator[None]:
        async with original(incoming):
            state = incoming.state.lifespan
            state.db = resources.db
            state.http = resources.http
            state.hub = resources.hub
            state.sessions = resources.sessions
            state.runtime = resources.runtime
            state.lifetime = resources.runtime
            state.providers = resources.providers
            state.gateway = resources.gateway
            state.actions = resources.actions
            state.agents = resources.agents
            if resources.usage is not None:
                state.usage = resources.usage.repository
                state.usage_refresh = resources.usage.refresh
            yield

    app.router.lifespan_context = wired
    return app


async def wire_runtime(
    resources: RuntimeResources,
    base_dir: Path,
    config: DaemonConfig,
    hub: Hub,
    db: AiosqliteDatabase,
    http: httpx.AsyncClient,
) -> SyncEngine:
    if config.sessions.agy_profiles_dir is None:
        config = replace(
            config,
            sessions=replace(config.sessions, agy_profiles_dir=str(base_dir / "agy-profiles")),
        )
    engine = SyncEngine(
        db,
        {kind: b for kind, b in build_sessions_backends(config).items() if b is not None},
        ttl_seconds=config.sync.ttl_seconds,
        docker_title_reader=docker_title,
        hub=hub,
        sync_config=config.sync,
    )
    with contextlib.suppress(Exception):
        await engine.sync()
    resources.db = db
    resources.http = http
    resources.hub = hub
    resources.usage = UsageCoordinator(UsageRepository(db), hub)
    for price in bundled_prices():
        await resources.usage.repository.add_price(price)
    usage_history = UsageHistorySync(
        db,
        resources.usage.repository,
        Path(config.sessions.codex_home) if config.sessions.codex_home else DEFAULT_CODEX_HOME,
        claude_home=(
            Path(config.sessions.claude_config_dir)
            if config.sessions.claude_config_dir else Path.home() / ".claude"
        ),
        opencode_db=Path(config.sessions.opencode_db_path or default_opencode_db_path()),
    )
    resources.usage.reconcile = usage_history.reconcile
    resources.usage.refresh_prices = PriceCatalogSync(resources.usage.repository).refresh
    executions = ExecutionRepository(db)
    privacy = build_privacy(
        config.privacy,
        base_dir,
        db,
        lambda: (provider.api_key for provider in TomlConfigAdapter(base_dir).load().providers),
    )
    resources.sessions = SessionsService(
        db,
        engine,
        transcripts=_build_transcripts(config.sessions),
        executions=executions,
        pi_identities=NativePiSessionIdentities(db),
        privacy_scopes=privacy.scopes,
        worktrees_dir=base_dir / "worktrees",
    )
    await resources.sessions.worktrees.recover()
    events = HubGatewayEventSink(hub)
    wired_routes: list[RouteLookupPort] = []
    resources.providers = ProvidersRegistry(
        TomlConfigAdapter(base_dir), _DeferredRoutes(wired_routes)
    )
    registry = RouteRegistry(db, resources.providers, events)
    wired_routes.append(registry)
    resources.gateway = GatewayWiring(
        registry,
        OpenAIHandler(),
        AnthropicHandler(),
        gemini=GeminiHandler(),
        privacy=privacy,
        usage_sink=resources.usage.gateway,
    )
    with contextlib.suppress(Exception):
        resources.gateway.reasoning_catalog = await build_reasoning_catalog(resources.providers)
    resources.runtime = RuntimeService(
        attachments_dir=base_dir / "attachments",
        harness_commands=build_harness_commands(
            config.sessions.launch_args, include_docker=config.docker.image is not None
        ),
        sessions=resources.sessions,
        routes=registry,
        hub=hub,
        gateway_port=config.server.port,
        token_issuer=resources.gateway.issue_child_token,
        adapters=build_harness_adapters,
        approval_timeout_seconds=config.approvals.timeout_seconds,
        mode_defaults=config.sessions.mode,
        idle_release_seconds=config.sessions.idle_release_seconds,
        liveness=WorkingStateTracker(),
        agy_home=Path(config.sessions.agy_home) if config.sessions.agy_home else None,
        agy_profiles_dir=Path(config.sessions.agy_profiles_dir)
        if config.sessions.agy_profiles_dir
        else base_dir / "agy-profiles",
        docker_config=docker_config(config.docker, base_dir),
        executions=executions,
        creation_policy=SessionPolicy(
            config.defaults.execution_backend, config.defaults.privacy_mode
        ),
        privacy_scopes=privacy.scopes,
    )
    resources.runtime.commands.catalogs = create_command_catalog_cache(
        build_harness_commands(config.sessions.launch_args), config.sessions,
        resources.runtime._spawn_harness, default_cwd=str(Path.cwd()),
        profiles_dir=Path(config.sessions.agy_profiles_dir or base_dir / "agy-profiles"),
        docker_backend=resources.runtime._docker,
        docker_commands=build_harness_commands(config.sessions.launch_args, include_docker=True),
    )
    projects: dict[tuple[str, str, str], CommandCatalogParams] = {}
    for session in (await resources.sessions.list_sessions())[:32]:
        if not session.project_path or session.privacy_mode is not PrivacyMode.NONE:
            continue
        key = (session.harness.value, str(session.project_path), session.execution_backend.value)
        projects.setdefault(key, CommandCatalogParams(
            harness=session.harness.value, cwd=str(session.project_path),
            execution_backend=session.execution_backend.value,
        ))
        if len(projects) >= 8:
            break
    native_usage = NativeUsageCollector(
        resources.usage.record, account_sink=resources.usage.account
    )
    resources.runtime.set_usage_observer(native_usage)
    resources.usage.refresh_accounts = NativeQuotaSync(
        build_harness_commands(config.sessions.launch_args),
        config.sessions,
        resources.usage.account,
    ).refresh
    try:
        await resources.runtime.reconcile_docker()
    except Exception as error:
        logger.warning("Docker startup reconciliation failed: %s", error)
    resources.runtime.commands.catalogs.start(list(projects.values()))
    resources.agents = AgentService(
        build_agent_history(
            db,
            resources.sessions,
            config.sessions,
            _build_transcripts(config.sessions),
            Path(default_opencode_db_path()),
        ),
        resources.runtime,
        resources.sessions,
    )
    resources.actions = build_action_registry(
        resources.runtime, resources.sessions, resources.agents
    )
    agent_events = AgentEventRouter(hub, resources.agents.history_store, resources.sessions)
    resources.runtime.set_event_router(agent_events.publish, agent_events.approval_topic)
    return engine


@dataclasses.dataclass(frozen=True)
class BackgroundTasks:
    sync: asyncio.Task[None]
    reconcile: asyncio.Task[None]
    approvals: asyncio.Task[None]
    pid_watch: asyncio.Task[None]


def start_background_tasks(
    engine: SyncEngine,
    runtime: RuntimeService,
    config: DaemonConfig,
    server: uvicorn.Server,
    base_dir: Path,
) -> BackgroundTasks:
    return BackgroundTasks(
        sync=asyncio.create_task(_sync_loop(engine, config.sync.mtime_poll_seconds)),
        reconcile=asyncio.create_task(
            _reconcile_loop(runtime, config.sessions.reconcile_interval_seconds)
        ),
        approvals=asyncio.create_task(_approvals_loop(runtime)),
        pid_watch=asyncio.create_task(_write_pid_file_once_serving(server, base_dir)),
    )


async def stop_background_tasks(
    tasks: BackgroundTasks,
    resources: RuntimeResources,
    hub: Hub,
    http: httpx.AsyncClient,
    db: AiosqliteDatabase,
    base_dir: Path,
) -> None:
    tasks.pid_watch.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tasks.pid_watch
    remove_pid_file(base_dir)
    for task in (tasks.sync, tasks.reconcile, tasks.approvals):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    for task in resources.agent_tasks:
        task.cancel()
    for task in resources.agent_tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task
    if resources.runtime is not None:
        await resources.runtime.commands.catalogs.aclose()
        for session_id in resources.runtime.registry.live_ids():
            with contextlib.suppress(Exception):
                await resources.runtime.stop_session(
                    session_id, grace=2.0, cause=SessionStopCause.DAEMON_STOP
                )
    if resources.usage is not None:
        resources.usage.close()
    await hub.close_all()
    await http.aclose()
    await db.close()


async def serve(server: uvicorn.Server, resources: RuntimeResources, base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(base_dir)
    config = dataclasses.replace(
        config,
        server=dataclasses.replace(config.server, host=server.config.host, port=server.config.port),
    )
    db = AiosqliteDatabase()
    await db.connect(base_dir / "mandri.db")
    await db.migrate()
    hub = Hub()
    http = httpx.AsyncClient()
    engine = await wire_runtime(resources, base_dir, config, hub, db, http)
    assert resources.runtime is not None
    assert resources.agents is not None
    resources.agent_tasks = [
        asyncio.create_task(discover_agents(resources.agents.history_store)),
        asyncio.create_task(backfill_agents(resources.agents.history_store)),
    ]
    if resources.usage is not None:
        resources.agent_tasks.append(asyncio.create_task(resources.usage.run()))
    tasks = start_background_tasks(engine, resources.runtime, config, server, base_dir)
    try:
        await server.serve()
    finally:
        await stop_background_tasks(tasks, resources, hub, http, db, base_dir)


async def _write_pid_file_once_serving(server: uvicorn.Server, base_dir: Path) -> None:
    while not server.started and not server.should_exit:
        await asyncio.sleep(PID_BIND_POLL_SECONDS)
    if server.started:
        write_pid_file(base_dir)
        (base_dir / "desktop-version").write_text(__version__)


async def _sync_loop(engine: SyncEngine, interval_seconds: int) -> None:
    interval = float(max(interval_seconds, 1))
    while True:
        await asyncio.sleep(_jittered(interval))
        try:
            await engine.sync()
        except Exception as error:
            logger.warning("session sync pass failed: %s", error)


def _jittered(seconds: float) -> float:
    return seconds * random.uniform(0.85, 1.15)


async def _reconcile_loop(runtime: RuntimeService, interval_seconds: int) -> None:
    interval = float(max(interval_seconds, 1))
    while True:
        try:
            await runtime.reconcile_and_persist()
        except Exception as error:
            logger.warning("session reconcile pass failed: %s", error)
        await asyncio.sleep(interval)


APPROVAL_EXPIRY_INTERVAL_S = 1.0


async def _approvals_loop(runtime: RuntimeService) -> None:
    while True:
        try:
            await runtime.expire_approvals()
        except Exception as error:
            logger.warning("approval expiry pass failed: %s", error)
        await asyncio.sleep(APPROVAL_EXPIRY_INTERVAL_S)


def _build_transcripts(sessions: SessionsConfig) -> TranscriptResolver:
    config_dir = (
        Path(sessions.claude_config_dir) if sessions.claude_config_dir else Path.home() / ".claude"
    )
    codex_home = Path(sessions.codex_home) if sessions.codex_home else DEFAULT_CODEX_HOME
    opencode_db = (
        Path(sessions.opencode_db_path)
        if sessions.opencode_db_path
        else Path(default_opencode_db_path())
    )
    return TranscriptResolver(
        {
            HarnessKind.AGY: AgyTranscriptReader(
                Path(sessions.agy_home) if sessions.agy_home else default_agy_root(),
                Path(sessions.agy_profiles_dir) if sessions.agy_profiles_dir else None,
            ),
            HarnessKind.CLAUDE: ClaudeTranscriptReader(config_dir / "projects"),
            HarnessKind.CODEX: CodexTranscriptReader(codex_home / CODEX_SESSIONS_DIR_NAME),
            HarnessKind.PI: PiTranscriptReader(),
            HarnessKind.OPENCODE: OpencodeTranscriptReader(opencode_db),
        }
    )


def _build_opencode_backend(sessions: SessionsConfig) -> HarnessSessionsBackend | None:
    db_path = (
        Path(sessions.opencode_db_path)
        if sessions.opencode_db_path
        else Path(default_opencode_db_path())
    )
    if not db_path.is_file():
        return None
    fetch: FetchSessionsPort = OpencodeSqliteFetchSessionsAdapter(db_path)
    rename: RenameSessionPort | None = None
    delete: DeleteSessionPort | None = None
    exists: CheckSessionExistsPort | None = None
    if sessions.opencode_base_url is not None:
        client = httpx.AsyncClient(base_url=sessions.opencode_base_url)
        rename = OpencodeRestRenameSessionAdapter(client)
        delete = OpencodeRestDeleteSessionAdapter(client)
        exists = OpencodeRestCheckSessionExistsAdapter(client)
    return HarnessSessionsBackend(HarnessKind.OPENCODE, fetch, rename, delete, exists)


def _build_codex_backend(sessions: SessionsConfig) -> HarnessSessionsBackend | None:
    codex_home = Path(sessions.codex_home) if sessions.codex_home else DEFAULT_CODEX_HOME
    if not codex_home.is_dir():
        return None
    fetch = CodexFetchSessions(
        codex_home / CODEX_STATE_DB_NAME,
        codex_home / CODEX_SESSIONS_DIR_NAME,
    )
    command = build_harness_commands(sessions.launch_args)["codex"]
    mutations = CodexMutationsAdapter(partial(connect_codex_app_server, command=command))
    return HarnessSessionsBackend(HarnessKind.CODEX, fetch, mutations, mutations, mutations)


def _build_claude_backend(sessions: SessionsConfig) -> HarnessSessionsBackend:
    config_dir = sessions.claude_config_dir
    return HarnessSessionsBackend(
        HarnessKind.CLAUDE,
        ClaudeSdkFetchSessionsAdapter(config_dir),
        ClaudeSdkRenameSessionAdapter(config_dir),
        ClaudeSdkDeleteSessionAdapter(config_dir),
        ClaudeSdkCheckSessionExistsAdapter(config_dir),
    )


def _build_pi_backend() -> HarnessSessionsBackend:
    adapter = PiSessionsAdapter(store=PiSessionStore())
    return HarnessSessionsBackend(HarnessKind.PI, adapter, adapter, adapter, adapter)
