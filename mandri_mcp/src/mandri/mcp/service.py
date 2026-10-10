import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind
from mandri.core.types.mcp import (
    McpBinding,
    McpEndpoint,
    McpServerConfig,
    McpServerView,
    McpSnapshot,
)
from mandri.mcp.bindings import Bindings
from mandri.mcp.oauth import OAuthManager, OAuthStorage
from mandri.mcp.proxy import Proxy
from mandri.mcp.registry import Definition, Registry
from mandri.mcp.supervisor import Supervisor
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send


class McpService:
    def __init__(self, base_dir: Path, port: int, hub: Hub | None = None) -> None:
        self.registry = Registry(base_dir)
        self.port = port
        self.hub = hub
        self.revision = 0
        self.supervisors: dict[str, Supervisor] = {}
        self.proxies: dict[str, Proxy] = {}
        self.bindings = Bindings()
        self.oauth = OAuthManager(self.registry.secrets, port)
        self._edits = asyncio.Lock()
        self._base_dir = base_dir

    def start(self) -> None:
        self.registry.load()
        for row in self.registry.definitions.values():
            self._start(row)

    def _changed(self, identity: str, *, deleted: bool = False) -> None:
        self.revision += 1
        supervisor = self.supervisors.get(identity)
        proxy = self.proxies.get(identity)
        if supervisor is not None and supervisor.client is not None and proxy is not None:
            proxy.server.instructions = supervisor.client.instructions
            info = supervisor.client.server_info
            if info is not None:
                proxy.server.version = info.version
        if self.hub is not None:
            self.hub.publish(
                Topic("mcp.events"),
                {"revision": self.revision, "server_id": identity, "deleted": deleted},
            )

    def _start(self, row: Definition) -> None:
        supervisor = Supervisor(row, self._base_dir, self._changed, self.oauth)
        self.supervisors[row.id] = supervisor
        if row.config.enabled:
            proxy = Proxy(supervisor)
            self.proxies[row.id] = proxy
            supervisor.task = asyncio.create_task(self._run(supervisor, proxy))
        self._changed(row.id)

    async def _run(self, supervisor: Supervisor, proxy: Proxy) -> None:
        try:
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(proxy.app.router.lifespan_context(proxy.app))
                await supervisor.run()
        except asyncio.CancelledError:
            raise
        except Exception:
            supervisor.set_state("error", "Local MCP endpoint failed")

    def snapshot(self) -> McpSnapshot:
        servers = [supervisor.view() for supervisor in self.supervisors.values()]
        for identity, error in self.registry.errors.items():
            servers.append(
                McpServerView(
                    id=identity,
                    revision=0,
                    name=identity,
                    enabled=True,
                    transport="stdio",
                    state="error",
                    error=error,
                )
            )
        return McpSnapshot(revision=self.revision, servers=servers)

    async def add(self, config: McpServerConfig) -> McpServerView:
        async with self._edits:
            row = self.registry.add(config)
            self._start(row)
            return self.supervisors[row.id].view()

    async def update(self, identity: str, revision: int, values: dict[str, Any]) -> McpServerView:
        async with self._edits:
            current = self.registry.get(identity)
            values = dict(values)
            for source, target in (("env_updates", "env"), ("header_updates", "headers")):
                entries = values.pop(source, None)
                if entries is None:
                    continue
                if not isinstance(entries, dict):
                    raise ValueError("Invalid secret updates")
                existing = dict(values.get(target, getattr(current.config, target)))
                for name, value in entries.items():
                    if not isinstance(name, str) or (
                        value is not None and not isinstance(value, str)
                    ):
                        raise ValueError("Invalid secret update")
                    if value is None:
                        existing.pop(name, None)
                    else:
                        existing[name] = value
                values[target] = existing
            row = self.registry.update(identity, revision, values)
            await self._stop(row.id)
            self._start(row)
            return self.supervisors[row.id].view()

    async def remove(self, identity: str) -> None:
        async with self._edits:
            if identity in self.registry.invalid:
                self.registry.discard_invalid(identity)
                self._changed(identity, deleted=True)
                return
            row = self.registry.remove(identity)
            await self._stop(row.id)
            self._changed(row.id, deleted=True)

    async def reconnect(self, identity: str) -> McpServerView:
        async with self._edits:
            row = self.registry.get(identity)
            await self._stop(row.id)
            self._start(row)
            return self.supervisors[row.id].view()

    async def login(self, identity: str) -> McpServerView:
        async with self._edits:
            return await self._login(identity)

    async def _login(self, identity: str) -> McpServerView:
        row = self.registry.get(identity)
        if row.config.auth != "oauth" or not row.config.enabled:
            raise ValueError("Enable OAuth for this server before signing in")
        await self._stop(row.id)
        self.oauth.begin(row.id)
        self._start(row)
        return self.supervisors[row.id].view()

    async def logout(self, identity: str) -> McpServerView:
        async with self._edits:
            return await self._logout(identity)

    async def _logout(self, identity: str) -> McpServerView:
        row = self.registry.get(identity)
        await self._stop(row.id)
        self.oauth.cancel(row.id)
        OAuthStorage(self.registry.secrets, row.id).clear()
        self._start(row)
        return self.supervisors[row.id].view()

    async def _stop(self, identity: str) -> None:
        self.bindings.withdraw(identity)
        self.oauth.cancel(identity)
        supervisor = self.supervisors.pop(identity, None)
        self.proxies.pop(identity, None)
        if supervisor is not None:
            await supervisor.stop()

    def bind(
        self, owner: str, harness: HarnessKind, *, protected: bool = False, lease: bool = False
    ) -> McpBinding:
        endpoints = [
            McpEndpoint(
                id=row.definition.id,
                name=row.definition.config.name,
                url=f"http://127.0.0.1:{self.port}/v1/mcp/connect/{{binding}}/{row.definition.id}",
            )
            for row in self.supervisors.values()
            if harness is not HarnessKind.AGY
            and row.state == "ready"
            and (not protected or row.definition.config.allow_protected)
        ]
        return self.bindings.issue(owner, endpoints, lease=lease)

    def release(self, owner: str) -> None:
        self.bindings.release(owner)

    async def aclose(self) -> None:
        self.bindings.clear()
        self.oauth.close()
        await asyncio.gather(*(self._stop(key) for key in list(self.supervisors)))

    async def dispatch(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope["path"].removeprefix(scope.get("root_path", "")).strip("/")
        parts = path.split("/")
        headers = dict(scope.get("headers", []))
        token = headers.get(b"authorization", b"").decode("latin-1").removeprefix("Bearer ")
        if len(parts) != 2 or not self.bindings.authenticate(parts[0], parts[1], token):
            await JSONResponse({"error": "Access denied"}, 401)(scope, receive, send)
            return
        proxy = self.proxies.get(parts[1])
        if proxy is None:
            await JSONResponse({"error": "Server unavailable"}, 503)(scope, receive, send)
            return
        child = dict(scope)
        child["root_path"] = scope["path"]
        child["path"] = scope["path"] + "/"
        child["raw_path"] = child["path"].encode()
        await proxy.app(child, receive, send)
