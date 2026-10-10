import asyncio
import contextlib
import random
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path

from mandri.core.types.mcp import McpServerView, McpState
from mandri.mcp.catalog import Catalog
from mandri.mcp.oauth import AuthenticationRequired, OAuthManager
from mandri.mcp.registry import Definition
from mandri.mcp.transport import connect

from mcp import Client, MCPError, types
from mcp.client.session import IncomingMessage


class Supervisor:
    def __init__(
        self,
        definition: Definition,
        base_dir: Path,
        changed: Callable[[str], None],
        oauth: OAuthManager | None = None,
    ) -> None:
        self.oauth = oauth
        self.definition = definition
        self.base_dir = base_dir
        self.changed = changed
        self.catalog: Catalog | None = None
        self.client: Client | None = None
        self.state: McpState = "starting" if definition.config.enabled else "disabled"
        self.error: str | None = None
        self.lock = asyncio.Lock()
        self.refresh = asyncio.Event()
        self.failed = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    def set_state(self, state: McpState, error: str | None = None) -> None:
        self.state, self.error = state, error
        if state != "ready":
            self.catalog = None
        self.changed(self.definition.id)

    def view(self) -> McpServerView:
        config = self.definition.config
        catalog = self.catalog
        return McpServerView(
            id=self.definition.id,
            revision=self.definition.revision,
            **config.model_dump(exclude={"env", "headers"}),
            env_keys=sorted(config.env),
            header_keys=sorted(config.headers),
            state=self.state,
            error=self.error,
            tools=len(catalog.tools) if catalog else 0,
            resources=len(catalog.resources) + len(catalog.templates) if catalog else 0,
            prompts=len(catalog.prompts) if catalog else 0,
            authorize_url=self.oauth.url(self.definition.id) if self.oauth else None,
        )

    async def run(self) -> None:
        delay = 1.0
        connecting: McpState = "starting"
        while True:
            identity = self.definition.id
            config = self.definition.config
            auth = None
            if config.auth == "oauth" and self.oauth is not None:
                if not self.oauth.has_tokens(identity) and not self.oauth.interactive(identity):
                    self.set_state("auth_required", "Sign in required")
                    await self.oauth.wait_for_login(identity)
                auth = self.oauth.provider(identity, config.url or "", self.authorization_ready)
            self.set_state(connecting)
            connecting = "reconnecting"
            try:
                async with AsyncExitStack() as stack:
                    timeout = config.startup_timeout
                    if self.oauth is not None and self.oauth.interactive(identity):
                        timeout += 300
                    async with asyncio.timeout(timeout):
                        client = await stack.enter_async_context(
                            connect(config, self.base_dir, auth, self.notification)
                        )
                        catalog = await Catalog.fetch(client)
                    self.client = client
                    self.catalog = catalog
                    self.set_state("ready")
                    if self.oauth is not None:
                        self.oauth.cancel(identity)
                    delay = 1.0
                    while True:
                        with contextlib.suppress(TimeoutError):
                            await asyncio.wait_for(self.refresh.wait(), timeout=30)
                        self.refresh.clear()
                        if self.failed.is_set():
                            self.failed.clear()
                            raise ConnectionError("Connection closed")
                        if self.lock.locked():
                            continue
                        async with self.lock:
                            async with asyncio.timeout(2):
                                await client.session.send_ping()
                            async with asyncio.timeout(self.definition.config.startup_timeout):
                                catalog = await Catalog.fetch(client)
                        if catalog != self.catalog:
                            self.catalog = catalog
                            self.set_state("ready")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if contains_auth_required(error) and self.oauth is not None:
                    self.oauth.cancel(identity)
                    self.set_state("auth_required", "Sign in required")
                    await self.oauth.wait_for_login(identity)
                    continue
                if self.oauth is not None:
                    self.oauth.cancel(identity)
                self.set_state("error", error_label(error))
            finally:
                self.client = None
                if self.state == "ready":
                    self.set_state("error", "Connection closed")
            await asyncio.sleep(delay * random.uniform(0.85, 1.15))
            delay = min(delay * 2, 60)

    async def call[T](self, action: Callable[[Client], Awaitable[T]]) -> T:
        client = self.client
        if self.state != "ready" or client is None:
            raise MCPError(-32000, "Server unavailable")
        try:
            async with asyncio.timeout(self.definition.config.call_timeout):
                async with self.lock:
                    if self.state != "ready" or self.client is not client:
                        raise MCPError(-32000, "Server unavailable")
                    return await action(client)
        except MCPError:
            raise
        except TimeoutError:
            raise MCPError(-32000, "Server call timed out") from None
        except Exception:
            self.set_state("error", "Connection failed")
            self.failed.set()
            self.refresh.set()
            raise MCPError(-32000, "Server connection failed") from None

    def authorization_ready(self, identity: str) -> None:
        self.set_state("auth_required", "Sign in required")

    async def notification(self, message: IncomingMessage) -> None:
        if isinstance(
            message,
            (
                types.ToolListChangedNotification,
                types.ResourceListChangedNotification,
                types.PromptListChangedNotification,
            ),
        ):
            self.refresh.set()
        elif isinstance(message, Exception):
            self.failed.set()
            self.refresh.set()

    async def stop(self) -> None:
        self.catalog = None
        self.client = None
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task


def error_label(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return error_label(error.exceptions[0])
    if isinstance(error, FileNotFoundError):
        return "Command or directory not found"
    if isinstance(error, PermissionError):
        return "Access denied"
    if isinstance(error, TimeoutError):
        return "Server did not respond"
    if isinstance(error, ValueError):
        return "Invalid server configuration or catalog"
    return "Server connection failed"


def contains_auth_required(error: BaseException) -> bool:
    if isinstance(error, BaseExceptionGroup):
        return any(contains_auth_required(child) for child in error.exceptions)
    return isinstance(error, AuthenticationRequired)
